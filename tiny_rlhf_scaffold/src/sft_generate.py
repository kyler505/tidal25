import argparse, json, time
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from src.utils import load_config, write_jsonl

def load_model(model_id, quantization=4):
    try:
        # Attempt 4-bit load via bitsandbytes if available (fast path)
        model = AutoModelForCausalLM.from_pretrained(model_id,
                                                     load_in_4bit=True,
                                                     device_map='auto')
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        return model, tokenizer
    except Exception as e:
        # Fallback to CPU small model
        print("Quantized load failed or not available, falling back to standard load:", e)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        return model, tokenizer

def get_prompts(cfg):
    # prefer API if provided in config, else local file
    api = cfg.get('api',{}).get('prompts')
    if api:
        from src.api_connector import APIConnector
        conn = APIConnector(api)
        return conn.get_prompts()
    path = cfg['data_paths']['prompts']
    with open(path) as f:
        return [json.loads(line) for line in f]

def generate_all(model_id=None, out_path=None):
    cfg = load_config()
    model_id = model_id or cfg['model_id']
    model, tokenizer = load_model(model_id, cfg.get('quantization',4))
    prompts = get_prompts(cfg)
    out_path = out_path or cfg['data_paths']['outputs']
    Path(Path(out_path).parent).mkdir(parents=True, exist_ok=True)
    results = []
    for p in prompts:
        inputs = tokenizer(p['text'], return_tensors='pt')
        gen = model.generate(**inputs, do_sample=True, top_k=50, num_return_sequences=cfg['params']['num_candidates'],
                             max_new_tokens=cfg['params']['max_new_tokens'])
        texts = [tokenizer.decode(g, skip_special_tokens=True) for g in gen]
        results.append({'id': p.get('id'), 'prompt': p['text'], 'candidates': texts, 'ts': time.time()})
    write_jsonl(out_path, results)
    print(f"Wrote baseline outputs to {out_path}")

if __name__ == '__main__':
    generate_all()
