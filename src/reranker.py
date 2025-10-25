import json, time, os
from src.utils import load_config, write_jsonl
from src.reward_model import load_reward, score_text
from src.api_connector import APIConnector

def rerank_all(baseline_path=None, out_path=None):
    cfg = load_config()
    baseline_path = baseline_path or cfg['data_paths']['outputs']
    out_path = out_path or "data/outputs/reranked.jsonl"
    with open(baseline_path) as f:
        entries = [json.loads(l) for l in f]
    emb, clf = load_reward()
    api_url = cfg.get('api',{}).get('reward')
    conn = APIConnector(api_url) if api_url else None
    results = []
    for e in entries:
        scored = []
        for c in e['candidates']:
            if conn:
                try:
                    res = conn.post_reward(c)
                    score = float(res.get('reward',0))
                except Exception:
                    score = score_text(c, emb, clf)
            else:
                score = score_text(c, emb, clf)
            scored.append((c, score))
        best = sorted(scored, key=lambda x: x[1], reverse=True)[0]
        results.append({'id': e.get('id'), 'prompt': e['prompt'], 'best': best[0], 'best_score': best[1], 'all': scored, 'ts': time.time()})
    write_jsonl(out_path, results)
    print("Wrote reranked outputs to", out_path)

if __name__ == '__main__':
    rerank_all()
