from pathlib import Path
import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from src.rl.reco_env import RecoEnv
import sqlite3
import json
import argparse


BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "models" / "rl"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_ROOT = BASE_DIR / "models"
DB_PATH = BASE_DIR / "database" / "recommender.db"
CONFIGS_DIR = BASE_DIR / "configs"
CONFIGS_DIR.mkdir(parents=True, exist_ok=True)


def load_news_ids():
    if not DB_PATH.exists():
        raise FileNotFoundError("Database not found. Run populate_database first.")
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id FROM news")
    rows = cur.fetchall()
    conn.close()
    return [r[0] for r in rows]


def _build_policy_kwargs(network_sizes):
    # network_sizes: list of ints e.g. [64,64]
    return dict(net_arch=[dict(pi=network_sizes, vf=network_sizes)])


def train(config_path: Path):
    # load config
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    total_timesteps = int(cfg.get("total_timesteps", 20000))
    learning_rate = float(cfg.get("learning_rate", 2.5e-4))
    gamma = float(cfg.get("gamma", 0.99))
    clip_range = float(cfg.get("clip_range", 0.2))
    net_sizes = cfg.get("network_sizes", [64, 64])

    emb_path = MODELS_ROOT / "news_embeddings.npy"
    id2idx_path = MODELS_ROOT / "news_id2idx.json"
    if not emb_path.exists() or not id2idx_path.exists():
        raise FileNotFoundError("Embeddings or mapping not found. Run embedder first.")

    embeddings = np.load(emb_path)
    news_ids = load_news_ids()

    env = RecoEnv(embeddings, news_ids)

    checkpoint_callback = CheckpointCallback(save_freq=5000, save_path=str(MODELS_DIR), name_prefix="ppo_checkpoint")

    policy_kwargs = _build_policy_kwargs(net_sizes)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=learning_rate,
        gamma=gamma,
        clip_range=clip_range,
        policy_kwargs=policy_kwargs,
    )
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    model_path = MODELS_DIR / "ppo_model.zip"
    model.save(str(model_path))
    print(f"Model trained and saved to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(CONFIGS_DIR / "ppo_config.yaml"), help="Path to PPO config yaml")
    args = parser.parse_args()
    train(Path(args.config))
