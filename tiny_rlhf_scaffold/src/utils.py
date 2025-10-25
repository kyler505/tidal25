import yaml, json, os
from pathlib import Path

def load_config(path='config/settings.yaml'):
    with open(path) as f:
        return yaml.safe_load(f)

def write_jsonl(path, records):
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    with open(path,'a') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
