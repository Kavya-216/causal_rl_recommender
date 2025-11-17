#!/usr/bin/env bash
set -euo pipefail

echo "Running full pipeline for Causal-Informed RL Recommender"

PY=python3


echo "1/8 - Auto-clean / Preprocessing MIND files"
$PY -m src.preprocessing.auto_clean

echo "2/8 - Building embeddings"
$PY -m src.preprocessing.embedder

echo "3/8 - Populating database"
$PY -m src.preprocessing.populate_database

echo "4/8 - Estimating causal effects"
$PY -m src.causal.counterfactual_estimator

echo "5/8 - Training PPO model (this may take a while)"
$PY -m src.rl.train_ppo --config configs/ppo_config.yaml

echo "6/8 - Evaluating models"
$PY -m src.utils.evaluate

echo "7/8 - Starting FastAPI server"
nohup uvicorn api.server:app --host 0.0.0.0 --port 8000 &

echo "8/8 - Pipeline finished"

echo "Starting FastAPI server (uvicorn) in background"
nohup uvicorn api.server:app --host 0.0.0.0 --port 8000 &

echo "Pipeline complete. Server running on port 8000"
