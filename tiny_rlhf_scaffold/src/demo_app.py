import streamlit as st
import json, time
from pathlib import Path
from src.utils import load_config, write_jsonl
from src.reward_model import load_reward, score_text
from src.reranker import rerank_all

cfg = load_config()

st.title("Tiny-RLHF Demo")
st.markdown("Baseline generations vs Reranked (preference-improved) outputs")

baseline_path = cfg['data_paths']['outputs']
reranked_path = "data/outputs/reranked.jsonl"

if not Path(baseline_path).exists():
    st.warning(f'Baseline outputs not found at {baseline_path}. Run sft_generate first.')
else:
    with open(baseline_path) as f:
        baseline = [json.loads(l) for l in f]
    for entry in baseline:
        st.subheader(f"Prompt: {entry['prompt']}")
        cols = st.columns(3)
        for i,c in enumerate(entry['candidates']):
            cols[i%3].text_area(f"Candidate {i+1}", c, height=120)
    if st.button("Rerank (score & pick best)"):
        rerank_all(baseline_path, reranked_path)
        st.success("Reranked! Scroll down to see results.")

if Path(reranked_path).exists():
    st.header("Reranked Results")
    with open(reranked_path) as f:
        reranked = [json.loads(l) for l in f]
    for r in reranked:
        st.write("---")
        st.write(f"**Prompt:** {r['prompt']}")
        st.write(f"**Best (score={r['best_score']:.3f}):**\n{r['best']}")
        if st.button(f"Prefer this best for prompt {r['id']}"):
            # append a comparison to comparisons.jsonl as (a=best, b=baseline[0], preferred='a')
            rec = {'a': r['best'], 'b': r['all'][0][0], 'preferred': 'a', 'ts': time.time()}
            write_jsonl(cfg['data_paths']['comparisons'], [rec])
            st.experimental_rerun()
