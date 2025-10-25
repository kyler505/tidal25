# Tiny-RLHF (Hackathon scaffold)

Lightweight, modular RLHF-style pipeline designed to run on CPU / small GPU with quantized models.
Drop-in API endpoints supported.

Structure:
- config/settings.yaml : runtime configuration
- data/ : prompts, comparisons, outputs
- models/ : base (quantized models), reward
- src/ : python modules (sft_generate, reward_model, reranker, api_connector, utils, demo_app)
- scripts/ : run orchestration

See `requirements.txt` for needed packages.

