# causal_rl_recommender — One Page Quick Summary

What this repo is
- A backend for a causal-aware RL recommender built around the MIND news dataset.

Key features
- Deterministic preprocessing for MIND
- TF-IDF + Sentence-Transformer embeddings
- Causal estimation (DoWhy/EconML wrappers)
- Gym-compatible recommendation environment and PPO training
- FastAPI server with fallback baseline recommenders

Quick commands

Clone + venv + install
```bash
git clone <your-repo-url>
cd causal_rl_recommender
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Run minimal pipeline
```bash
./.venv/bin/python -m src.preprocessing.auto_clean
./.venv/bin/python -m src.preprocessing.embedder
./.venv/bin/python -m src.preprocessing.populate_database
uvicorn api.server:app --reload
```

Notes
- Training (`train_ppo`) is heavy — use the `Makefile` for convenience targets.
- Use `REBUILD.md` for full Codespaces rebuild instructions.
