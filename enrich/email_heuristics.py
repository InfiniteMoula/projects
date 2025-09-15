
# FILE: enrich/email_heuristics.py
import re
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa

from utils.parquet import ParquetBatchWriter, iter_batches

EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")


def _s(x: pd.Series) -> pd.Series:
    return x.astype("string")


def _slug(x: pd.Series) -> pd.Series:
    """Simplify strings to lowercase ASCII slugs.

    >>> _slug(pd.Series(["?l?", None])).tolist()
    ['ele', '']
    """

    s = _s(x).fillna("").str.lower()
    s = s.str.normalize("NFKD").str.encode("ascii", "ignore").str.decode("ascii")
    return s.str.replace(r"[^a-z]", "", regex=True)


def _first_letter(x: pd.Series) -> pd.Series:
    return _slug(x).str.slice(0, 1)


def _pick_first_valid(cands: list[pd.Series], mx_ok: pd.Series) -> pd.Series:
    """Return the first valid email from *cands* aligned with *mx_ok*.

    >>> mx = pd.Series([True, False, True], dtype="boolean")
    >>> c1 = pd.Series(["a@example.com", "bad", ""], dtype="string")
    >>> c2 = pd.Series(["", "b@example.com", "c@example.com"], dtype="string")
    >>> _pick_first_valid([c1, c2], mx).tolist()
    ['a@example.com', '', 'c@example.com']
    """

    out = pd.Series("", index=cands[0].index, dtype="string")
    for series in cands:
        candidate = _s(series)
        ok = candidate.str.match(EMAIL_RE) & mx_ok.fillna(False)
        out = out.mask(out.str.len() == 0, candidate.where(ok, ""))
    return out


def run(cfg, ctx):
    t0 = time.time()
    outdir = Path(ctx.get("outdir_path") or ctx.get("outdir"))
    srcs = [outdir / "enriched_dns.parquet", outdir / "enriched_domain.parquet"]
    src = next((p for p in srcs if p.exists()), None)
    if not src:
        return {"status": "WARN", "error": "no input for email heuristics"}

    outp = outdir / "enriched_email.parquet"
    total = 0
    logger = ctx.get("logger")

    try:
        with ParquetBatchWriter(outp) as writer:
            for pdf in iter_batches(src):
                if pdf.empty:
                    continue

                for col in ["nom", "prenom", "domain_root", "email", "mx_ok"]:
                    if col not in pdf.columns:
                        pdf[col] = pd.NA

                pdf["nom"] = _slug(pdf["nom"])
                pdf["prenom"] = _slug(pdf["prenom"])
                pdf["p"] = _first_letter(pdf["prenom"])
                pdf["domain_root"] = _s(pdf["domain_root"]).fillna("").str.strip()
                pdf["email"] = _s(pdf["email"])
                pdf["mx_ok"] = pdf["mx_ok"].astype("boolean")

                existing_ok = pdf["email"].fillna("").str.match(EMAIL_RE) & pdf["mx_ok"].fillna(False)
                best = pdf["email"].where(existing_ok, "")

                need_mask = best.str.len().eq(0) & pdf["domain_root"].str.len().gt(0)

                replacements = pd.Series("", index=best.index, dtype="string")
                if need_mask.any():
                    idx = need_mask[need_mask].index
                    domain = pdf["domain_root"].reindex(idx)
                    mx_subset = pdf["mx_ok"].reindex(idx)
                    candidates = [
                        _s(pdf["prenom"].reindex(idx) + "." + pdf["nom"].reindex(idx) + "@" + domain),
                        _s(pdf["prenom"].reindex(idx) + pdf["nom"].reindex(idx) + "@" + domain),
                        _s(pdf["p"].reindex(idx) + pdf["nom"].reindex(idx) + "@" + domain),
                        _s(pd.Series("contact@", index=idx, dtype="string") + domain),
                        _s(pd.Series("info@", index=idx, dtype="string") + domain),
                    ]
                    replacements.loc[idx] = _pick_first_valid(candidates, mx_subset)

                best = best.mask(need_mask, replacements)
                pdf["best_email"] = _s(best)

                table = pa.Table.from_pandas(pdf, preserve_index=False)
                writer.write_table(table)
                total += len(pdf)

        return {"status": "OK", "file": str(outp), "rows": total, "duration_s": round(time.time() - t0, 3)}
    except Exception as exc:
        if logger:
            logger.exception("email heuristics failed: %s", exc)
        return {"status": "FAIL", "error": str(exc), "duration_s": round(time.time() - t0, 3)}
