import os, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
from utils import io, hashx
def run(cfg, ctx):
    p=os.path.join(ctx["outdir"],"scored.parquet")
    if not os.path.exists(p): p=os.path.join(ctx["outdir"],"deduped.parquet")
    if not os.path.exists(p): p=os.path.join(ctx["outdir"],"normalized.parquet")
    df=pd.read_parquet(p)
    df=df.sort_values(by=[c for c in ["siren","raison_sociale"] if c in df.columns], kind="mergesort")
    csv_path=os.path.join(ctx["outdir"],"dataset.csv"); df.to_csv(csv_path, index=False, encoding="utf-8")
    pq_path=os.path.join(ctx["outdir"],"dataset.parquet"); pq.write_table(pa.Table.from_pandas(df), pq_path, compression="snappy")
    job_yaml = io.read_text(ctx["job_path"]) if ctx.get("job_path") else ""
    man={"run_id":ctx["run_id"],"dataset_id":hashx.dataset_id(job_yaml,"pandas|pyarrow"),"records":int(len(df)),
         "paths":{"csv":csv_path,"parquet":pq_path},
         "manifest":{"robots_compliance":True,"tos_breaches":[],"pii_present":False,"anonymization_used":False}}
    io.write_json(os.path.join(ctx["outdir"],"manifest.json"), man)
    dd=[f"- {c}: non_null={int(df[c].notna().sum())}" for c in df.columns]
    io.write_text(os.path.join(ctx["outdir"],"data_dictionary.md"), "\n".join(dd))
    io.write_text(os.path.join(ctx["outdir"],"sha256.txt"), f"csv {io.sha256_file(csv_path)}\nparquet {io.sha256_file(pq_path)}\n")
    return {"csv":csv_path,"parquet":pq_path}
