
# FILE: normalize/standardize.py
import re
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa

from utils.parquet import ArrowCsvWriter, ParquetBatchWriter, iter_batches

# --- config -------------------------------------------------------------
TEL_RE = re.compile(r"\D+")
# on inclut les variantes CamelCase + snake_case courantes
DEFAULT_USECOLS = [
    "siren","siret","nic",
    "denominationUniteLegale","denominationusuelleetablissement","denominationUsuelleEtablissement",
    "enseigne1etablissement","enseigne1Etablissement",
    "libellecommuneetablissement","libelleCommuneEtablissement",
    "codepostaletablissement","codePostalEtablissement",
    "adresseetablissement","adresseEtablissement",
    "numeroVoieEtablissement","typeVoieEtablissement","libelleVoieEtablissement",
    "complementAdresseEtablissement",
    "activitePrincipaleEtablissement","activiteprincipaleetablissement",
    "activitePrincipaleUniteLegale","activiteprincipaleunitelegale",
    "dateCreationEtablissement","datecreationetablissement",
    "nomunitelegale","nomUniteLegale","prenomsunitelegale","prenomsUniteLegale",
    "telephone","email","siteweb",
    "etatAdministratifEtablissement","etatadministratifetablissement",
]

ARROW_OUT_SCHEMA = pa.schema([
    ("siren", pa.string()),
    ("siret", pa.string()),
    ("raison_sociale", pa.string()),
    ("enseigne", pa.string()),
    ("commune", pa.string()),
    ("cp", pa.string()),
    ("adresse", pa.string()),
    ("naf", pa.string()),
    ("date_creation", pa.string()),
    ("telephone_norm", pa.string()),
    ("email", pa.string()),
    ("siteweb", pa.string()),
    ("nom", pa.string()),
    ("prenom", pa.string()),
])

# --- helpers ------------------------------------------------------------
def _to_str(s: pd.Series | None) -> pd.Series:
    if s is None:
        return pd.Series(pd.NA, dtype="string")
    if str(s.dtype) != "string":
        return s.astype("string", copy=False)
    return s


def _fr_tel_norm(s: pd.Series | None) -> pd.Series:
    if s is None:
        return pd.Series(pd.NA, dtype="string")
    x = s.fillna("").astype("string", copy=False).str.replace(TEL_RE, "", regex=True)

    def _fmt(v: str) -> str:
        if not v:
            return ""
        if len(v) == 10 and v.startswith("0"):
            return "+33" + v[1:]
        if len(v) == 9:
            return "+33" + v
        return v

    x = x.map(_fmt).replace({"": pd.NA})
    return x.astype("string")


def _pick_first(df: pd.DataFrame, names: list[str]) -> pd.Series | None:
    for name in names:
        if name in df.columns:
            return df[name]
    return None


# --- main ---------------------------------------------------------------
def run(cfg: dict, ctx: dict) -> dict:
    t0 = time.time()
    input_path = ctx.get("input_path") or ctx.get("input")
    input_path = Path(input_path) if input_path else None
    if input_path is None or not input_path.exists():
        return {"status": "FAIL", "error": f"Input parquet not found: {input_path}"}

    job = ctx.get("job", {}) or {}
    filters = (job.get("filters") or {})
    naf_include_raw = [x.replace(".", "").replace(" ", "").lower() for x in (filters.get("naf_include") or [])]
    naf_prefixes = tuple(filter(None, naf_include_raw))
    active_only = bool(filters.get("active_only", False))

    usecols = job.get("standardize_usecols", DEFAULT_USECOLS)
    batch_rows = int(job.get("standardize_batch_rows", 200_000))

    outdir = Path(ctx.get("outdir_path") or ctx.get("outdir"))
    out_parquet = outdir / "normalized.parquet"
    out_csv = outdir / "normalized.csv"

    for target in (out_parquet, out_csv):
        if target.exists():
            try:
                target.unlink()
            except OSError:
                pass

    total = 0
    logger = ctx.get("logger")

    try:
        with ParquetBatchWriter(out_parquet, schema=ARROW_OUT_SCHEMA) as pq_writer, ArrowCsvWriter(out_csv) as csv_writer:
            for pdf in iter_batches(input_path, columns=usecols, batch_size=batch_rows):
                if pdf.empty:
                    continue

                object_cols = [col for col in pdf.columns if str(pdf[col].dtype) == "object"]
                if object_cols:
                    pdf[object_cols] = pdf[object_cols].astype("string", copy=False)

                naf_col = next(
                    (cand for cand in [
                        "activitePrincipaleEtablissement","activiteprincipaleetablissement",
                        "activitePrincipaleUniteLegale","activiteprincipaleunitelegale"
                    ] if cand in pdf.columns),
                    None,
                )

                if naf_prefixes and naf_col:
                    naf_norm = _to_str(pdf[naf_col]).str.replace(r"[\s\.]", "", regex=True).str.lower()
                    mask = naf_norm.fillna("").str.startswith(naf_prefixes)
                    pdf = pdf[mask.fillna(False)]
                    if pdf.empty:
                        continue

                if active_only:
                    state_col = next((c for c in [
                        "etatAdministratifEtablissement","etatadministratifetablissement"
                    ] if c in pdf.columns), None)
                    if state_col:
                        pdf = pdf[_to_str(pdf[state_col]).eq("A")]
                        if pdf.empty:
                            continue

                res = pd.DataFrame({
                    "siren": _to_str(_pick_first(pdf, ["siren"])),
                    "siret": _to_str(_pick_first(pdf, ["siret"])),
                    "raison_sociale": _to_str(_pick_first(pdf, ["denominationUniteLegale","denominationunitelegale"])).fillna("")
                        .where(lambda s: s.str.len() > 0,
                               _to_str(_pick_first(pdf, ["denominationUsuelleEtablissement","denominationusuelleetablissement"]))),
                    "enseigne": _to_str(_pick_first(pdf, ["enseigne1Etablissement","enseigne1etablissement"])),
                    "commune": _to_str(_pick_first(pdf, ["libelleCommuneEtablissement","libellecommuneetablissement"])),
                    "cp": _to_str(_pick_first(pdf, ["codePostalEtablissement","codepostaletablissement"])),
                    "adresse": _to_str(_pick_first(pdf, ["adresseEtablissement","adresseetablissement"])),
                    "naf": _to_str(_pick_first(pdf, [
                        "activitePrincipaleEtablissement","activiteprincipaleetablissement",
                        "activitePrincipaleUniteLegale","activiteprincipaleunitelegale"
                    ])).str.replace(r"\s", "", regex=True),
                    "date_creation": _to_str(_pick_first(pdf, ["dateCreationEtablissement","datecreationetablissement"])),
                    "telephone_norm": _fr_tel_norm(_pick_first(pdf, ["telephone"])),
                    "email": _to_str(_pick_first(pdf, ["email"])),
                    "siteweb": _to_str(_pick_first(pdf, ["siteweb"])),
                    "nom": _to_str(_pick_first(pdf, ["nomUniteLegale","nomunitelegale"])),
                    "prenom": _to_str(_pick_first(pdf, ["prenomsUniteLegale","prenomsunitelegale"])),
                })

                res["cp"] = res["cp"].str.extract(r"(\d{5})", expand=False).astype("string")
                res["naf"] = res["naf"].astype("string")

                total += len(res)
                if res.empty:
                    continue

                table = pa.Table.from_pandas(res, preserve_index=False).cast(ARROW_OUT_SCHEMA)
                pq_writer.write_table(table)
                csv_writer.write_table(table)

        return {
            "status": "OK",
            "files": [str(out_parquet), str(out_csv)],
            "rows": total,
            "duration_s": round(time.time() - t0, 3),
        }

    except Exception as exc:
        if logger:
            logger.exception("standardize failed: %s", exc)
        return {"status": "FAIL", "error": str(exc), "duration_s": round(time.time() - t0, 3)}
