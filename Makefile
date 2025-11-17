SHELL := /bin/bash
PY := ./.venv/bin/python

.PHONY: help venv install preprocess embed populate evaluate train serve clean docker-build docker-run

help:
	@echo "Makefile targets:"
	@echo "  venv         - create and activate virtualenv (local)"
	@echo "  install      - install requirements into .venv"
	@echo "  preprocess   - run preprocessing (auto_clean)"
	@echo "  embed        - run embedder (TF-IDF + SBERT)"
	@echo "  populate     - populate sqlite DB"
	@echo "  evaluate     - run quick evaluation"
	@echo "  train        - train PPO (config-driven, heavy)"
	@echo "  serve        - start uvicorn API server"
	@echo "  clean        - remove generated artifacts (models/data/db)"
	@echo "  docker-build - build Docker image"
	@echo "  docker-run   - run Docker image (port 8000)"

venv:
	python3.10 -m venv .venv
	@echo "Activate with: source .venv/bin/activate"

install: venv
	. .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

preprocess:
	$(PY) -m src.preprocessing.auto_clean

embed:
	$(PY) -m src.preprocessing.embedder

populate:
	$(PY) -m src.preprocessing.populate_database

evaluate:
	$(PY) -m src.utils.evaluate

train:
	$(PY) -m src.rl.train_ppo --config configs/ppo_config.yaml

serve:
	uvicorn api.server:app --reload --host 0.0.0.0 --port 8000

clean:
	@echo "Removing generated artifacts (database, models, processed data)"
	rm -rf database/recommender.db models/*.npy models/news_embeddings.npy models/tfidf_matrix.npy models/news_id2idx.json models/rl || true
	rm -rf data/processed/* || true

docker-build:
	docker build -t causal-rl-recommender:latest .

docker-run:
	docker run --rm -p 8000:8000 --name causal-rl-recommender causal-rl-recommender:latest
