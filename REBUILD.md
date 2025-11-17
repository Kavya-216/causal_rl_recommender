REBUILD.md
===========

This file contains exact commands to rebuild the project in a fresh Codespaces (or other Linux) environment.

Prerequisites
- Host with ~10GB free disk recommended for full runs (embeddings + optional RL).
- Git and Docker (optional) for container-based runs.

Quick rebuild (recommended minimal)

```bash
# 1. Clone and enter project
git clone <your-repo-url> causal_rl_recommender
cd causal_rl_recommender

# 2. Create Python 3.10 venv and activate
python3.10 -m venv .venv
source .venv/bin/activate

# 3. Upgrade pip and install minimal requirements
pip install --upgrade pip
pip install -r requirements.txt

# 4. Place MIND dataset files under data/raw/MIND/
#    - news.tsv
#    - behaviors.tsv

# 5. Run preprocessing and embedding (these are serial and may take time)
./.venv/bin/python -m src.preprocessing.auto_clean
./.venv/bin/python -m src.preprocessing.embedder

# 6. Populate the SQLite DB (this can be memory heavy for large datasets)
./.venv/bin/python -m src.preprocessing.populate_database

# 7. Quick checks (optional)
./.venv/bin/python -m src.utils.evaluate

# 8. Start API for local development
uvicorn api.server:app --reload --host 0.0.0.0 --port 8000
```

Notes for Codespaces / Devcontainer
- Consider adding a `.devcontainer/devcontainer.json` that sets the container image and installs system packages needed by PyTorch if you intend to train with GPU support (Codespaces may not provide GPUs).
- For CI (lightweight tests) configure CI to run only the unit tests and skip heavy steps like `embedder` and `train_ppo`.

Optional: full reproducible run (heavy)

```bash
# If you want to run everything (embeddings + RL training), make sure you have sufficient disk
# and preferably a machine with a GPU and CUDA. Then run:
./run_all.sh
```

If you want a `devcontainer` template or a GitHub Actions workflow that runs only lightweight checks, tell me and I will add it.
