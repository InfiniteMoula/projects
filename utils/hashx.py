import hashlib, json
def record_id(record: dict, keys):
    s=json.dumps({k:(record.get(k,"") or "") for k in keys}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()
def dataset_id(job_yaml:str, tools_versions:str):
    return hashlib.sha1((job_yaml+"|"+tools_versions).encode("utf-8")).hexdigest()
