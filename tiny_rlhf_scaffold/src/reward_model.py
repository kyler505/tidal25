import argparse, json, os
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
import joblib
from src.utils import load_config

MODEL_PATH = "models/reward/reward_clf.joblib"

def load_pairs(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f]

def train_reward(comparisons_path=None):
    cfg = load_config()
    path = comparisons_path or cfg['data_paths']['comparisons']
    pairs = load_pairs(path)
    if len(pairs) == 0:
        print("No comparison data found at", path)
        return None, None
    emb = SentenceTransformer('all-MiniLM-L6-v2')
    X, y = [], []
    for r in pairs:
        a = r['a']
        b = r['b']
        pref = r.get('preferred','a')
        e_a = emb.encode(a)
        e_b = emb.encode(b)
        X.append(e_a - e_b)
        y.append(1 if pref=='a' else 0)
    clf = LogisticRegression(max_iter=1000).fit(X, y)
    Path(Path(MODEL_PATH).parent).mkdir(parents=True, exist_ok=True)
    joblib.dump({'emb': 'all-MiniLM-L6-v2', 'clf': clf}, MODEL_PATH)
    print("Saved reward model to", MODEL_PATH)
    return emb, clf

def load_reward():
    if not os.path.exists(MODEL_PATH):
        return SentenceTransformer('all-MiniLM-L6-v2'), None
    data = joblib.load(MODEL_PATH)
    emb = SentenceTransformer(data['emb'])
    clf = data['clf']
    return emb, clf

def score_text(text, emb=None, clf=None):
    if emb is None or clf is None:
        emb, clf = load_reward()
    v = emb.encode(text)
    # If no clf, fall back to pseudo-score (length-normalized)
    if clf is None:
        return min(1.0, max(0.0, len(text.split())/200.0))
    prob = clf.predict_proba([v])[0][1]
    return float(prob)

if __name__ == '__main__':
    train_reward()
