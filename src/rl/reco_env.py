from pathlib import Path
import numpy as np
import json
import random
import gymnasium as gym
from gymnasium import spaces
from sqlalchemy import create_engine, text
import sqlite3
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "models"
DB_PATH = BASE_DIR / "database" / "recommender.db"
RESULTS_DIR = BASE_DIR / "results"


class RecoEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, embeddings: np.ndarray, news_ids: list, db_path: Path = DB_PATH,
                 causal_effect_path: Path = RESULTS_DIR / "causal_effects.json"):
        super().__init__()
        self.embeddings = embeddings
        self.news_ids = news_ids
        self.num_news = len(news_ids)
        self.db_path = db_path

        # load causal effect
        if (causal_effect_path).exists():
            with open(causal_effect_path, "r", encoding="utf-8") as f:
                self.causal = json.load(f).get("ate", 0.0)
        else:
            self.causal = 0.0

        emb_dim = int(self.embeddings.shape[1]) if self.embeddings is not None else 128
        # observation: user vector + last_clicked vector
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(emb_dim * 2,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.num_news)

        # internal state
        self.current_user = None
        self.user_history = []
        self.last_clicked_emb = np.zeros((emb_dim,), dtype=np.float32)

        # load interactions into memory for quick lookup
        if db_path.exists():
            conn = sqlite3.connect(db_path)
            try:
                self.interactions = pd.read_sql_query(
                    "SELECT user_id, news_id, clicked FROM interactions", conn
                )
            except Exception:
                # DB exists but table missing or corrupt — fall back to empty interactions
                self.interactions = pd.DataFrame(columns=["user_id", "news_id", "clicked"])
            finally:
                conn.close()
        else:
            self.interactions = pd.DataFrame(columns=["user_id", "news_id", "clicked"])

    def _get_user_embedding(self, user_id: str):
        # average embeddings of user history
        hist = self.interactions[(self.interactions["user_id"] == user_id) & (self.interactions["clicked"] == 1)]["news_id"].tolist()
        idxs = [self.news_ids.index(nid) for nid in hist if nid in self.news_ids]
        if not idxs:
            return np.zeros(self.embeddings.shape[1], dtype=np.float32)
        return np.nanmean(self.embeddings[np.array(idxs)], axis=0).astype(np.float32)

    def reset(self, seed=None, options=None):
        # pick a random user; if no users exist in the interactions table,
        # create a placeholder test user so the environment can still be used in tests.
        users = self.interactions["user_id"].unique().tolist()
        if not users:
            self.current_user = "test_user"
        else:
            self.current_user = random.choice(users)
        self.user_history = self.interactions[self.interactions["user_id"] == self.current_user]
        user_emb = self._get_user_embedding(self.current_user)
        self.last_clicked_emb = np.zeros_like(user_emb)
        obs = np.concatenate([user_emb, self.last_clicked_emb])
        return obs, {}

    def step(self, action: int):
        assert self.action_space.contains(action)
        news_idx = int(action)
        news_id = self.news_ids[news_idx]

        # check if clicked historically
        rec = self.interactions[(self.interactions["user_id"] == self.current_user) & (self.interactions["news_id"] == news_id)]
        clicked = 0
        if not rec.empty:
            clicked = int(rec.iloc[0]["clicked"])

        # reward composed of causal effect + click reward
        click_reward = 1.0 if clicked == 1 else 0.0
        causal_reward = float(self.causal) if clicked == 1 else 0.0
        reward = click_reward + causal_reward

        # update last clicked embedding
        if clicked == 1:
            self.last_clicked_emb = self.embeddings[news_idx]

        user_emb = self._get_user_embedding(self.current_user)
        obs = np.concatenate([user_emb, self.last_clicked_emb])

        done = True  # episodic: one-step per recommendation for training simplicity
        info = {"news_id": news_id, "clicked": int(clicked)}
        return obs, float(reward), done, False, info

    def render(self, mode="human"):
        print(f"User: {self.current_user}")

    def close(self):
        pass


if __name__ == "__main__":
    # quick sanity check
    import numpy as np
    embeddings = np.random.randn(10, 128).astype(np.float32)
    news_ids = [f"N{i}" for i in range(10)]
    env = RecoEnv(embeddings, news_ids)
    obs, _ = env.reset()
    a = env.action_space.sample()
    print(env.step(a))
