from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import sqlite3
import traceback

# Import local recommenders lazily to avoid heavy imports at module import time
from src.utils.baseline_recommenders import (
    popularity_recommend,
    tfidf_cosine_recommend,
    random_recommend,
)

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
DATABASE_PATH = BASE_DIR.parent / "database" / "recommender.db"


class Interaction(BaseModel):
    user_id: str
    article_id: str
    clicked: bool = False


@app.on_event("startup")
async def load_resources():
    # Attach resources to app.state for reuse
    app.state._model = None
    app.state.news_embeddings = None
    app.state.news_id2idx = None

    try:
        emb_path = MODELS_DIR / "news_embeddings.npy"
        map_path = MODELS_DIR / "news_id2idx.json"
        if emb_path.exists() and map_path.exists():
            # lazy-load numpy here to avoid import-time cost
            import numpy as np

            app.state.news_embeddings = np.load(emb_path)
            with open(map_path, "r") as f:
                app.state.news_id2idx = json.load(f)
    except Exception:
        traceback.print_exc()


@app.get("/recommend/{user_id}")
async def recommend(user_id: str, k: int = 5):
    # Try RL model first if available
    try:
        model_path = MODELS_DIR / "rl" / "ppo_model.zip"
        if getattr(app.state, "_model", None) is None and model_path.exists():
            try:
                # lazy import heavy RL library only when model exists
                from stable_baselines3 import PPO  # type: ignore[import]
                app.state._model = PPO.load(str(model_path))
            except Exception:
                # If RL libs aren't installed or loading fails, fall back
                app.state._model = None

        if getattr(app.state, "_model", None) is not None and getattr(app.state, "news_embeddings", None) is not None:
            # For now return top-k indices as placeholder for real policy
            n = min(k, app.state.news_embeddings.shape[0])
            return {"user_id": user_id, "recommendations": list(range(n))}
    except Exception:
        traceback.print_exc()

    # Fallback to TF-IDF or popularity. Catch and degrade to empty list if assets missing.
    try:
        recs = tfidf_cosine_recommend(user_id, top_n=k)
        return {"user_id": user_id, "recommendations": recs}
    except Exception:
        try:
            recs = popularity_recommend(top_n=k)
            return {"user_id": user_id, "recommendations": recs}
        except Exception:
            try:
                recs = random_recommend(top_n=k)
                return {"user_id": user_id, "recommendations": recs}
            except Exception:
                # As a last resort, return empty list instead of raising
                return {"user_id": user_id, "recommendations": []}


@app.get("/article/{article_id}")
async def get_article(article_id: str):
    # Read basic article info from DB if present
    if DATABASE_PATH.exists():
        try:
            conn = sqlite3.connect(str(DATABASE_PATH))
            cur = conn.cursor()
            cur.execute("SELECT news_id, title, abstract FROM news WHERE news_id=?", (article_id,))
            row = cur.fetchone()
            conn.close()
            if row:
                return {"article": {"news_id": row[0], "title": row[1], "abstract": row[2]}}
        except Exception:
            traceback.print_exc()
    raise HTTPException(status_code=404, detail="Article not found")


@app.get("/user/{user_id}/history")
async def user_history(user_id: str):
    if DATABASE_PATH.exists():
        try:
            conn = sqlite3.connect(str(DATABASE_PATH))
            cur = conn.cursor()
            cur.execute(
                "SELECT article_id, clicked FROM interactions WHERE user_id=? ORDER BY timestamp DESC LIMIT 20",
                (user_id,),
            )
            rows = cur.fetchall()
            conn.close()
            return {"user_id": user_id, "history": [{"article_id": r[0], "clicked": bool(r[1])} for r in rows]}
        except Exception:
            traceback.print_exc()
    return {"user_id": user_id, "history": []}


@app.post("/interact")
async def post_interaction(inter: Interaction):
    if DATABASE_PATH.exists():
        try:
            conn = sqlite3.connect(str(DATABASE_PATH))
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO interactions (user_id, article_id, clicked, timestamp) VALUES (?, ?, ?, CURRENT_TIMESTAMP)",
                (inter.user_id, inter.article_id, int(inter.clicked)),
            )
            conn.commit()
            conn.close()
            return {"status": "ok"}
        except Exception:
            traceback.print_exc()
    return {"status": "ok", "warning": "DB not available, not persisted"}



if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
