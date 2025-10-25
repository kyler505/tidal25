#!/bin/bash
set -e
python -m src.sft_generate
python -m src.reward_model
python -m src.reranker
streamlit run src/demo_app.py
