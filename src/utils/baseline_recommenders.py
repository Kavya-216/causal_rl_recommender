from pathlib import Path
import numpy as np
import sqlite3
import json
from typing import List
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "models"
DB_PATH = BASE_DIR / "database" / "recommender.db"


def popularity_recommend(top_n: int = 10) -> List[str]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT news_id, SUM(CASE WHEN clicked THEN 1 ELSE 0 END) as clicks FROM interactions GROUP BY news_id ORDER BY clicks DESC LIMIT ?", (top_n,))
    rows = cur.fetchall()
    conn.close()
    return [r[0] for r in rows]


def tfidf_cosine_recommend(user_id: str, top_n: int = 10) -> List[str]:
    tfidf_path = MODELS_DIR / "tfidf_matrix.npy"
    mapping_path = MODELS_DIR / "news_id2idx.json"
    if not tfidf_path.exists() or not mapping_path.exists():
        raise FileNotFoundError("TF-IDF matrix or news mapping missing. Run embedder first.")

    tfidf = np.load(tfidf_path)
    with open(mapping_path, "r", encoding="utf-8") as f:
        id2idx = json.load(f)

    conn = sqlite3.connect(DB_PATH)
    import pandas as pd
    inter = pd.read_sql_query("SELECT user_id, news_id, clicked FROM interactions", conn)
    conn.close()

    clicked = inter[(inter["user_id"] == user_id) & (inter["clicked"] == 1)]["news_id"].tolist()
    idxs = [id2idx[n] for n in clicked if n in id2idx]
    if not idxs:
        # fallback to popularity
        return popularity_recommend(top_n)

    profile = tfidf[idxs].sum(axis=0).reshape(1, -1)
    sims = cosine_similarity(profile, tfidf)[0]

    # rank
    ranked_idx = np.argsort(-sims)
    inv_map = {int(v): k for k, v in id2idx.items()}
    recs = []
    for idx in ranked_idx:
        nid = inv_map.get(int(idx))
        if nid is None:
            continue
        if nid in clicked:
            continue
        recs.append(nid)
        if len(recs) >= top_n:
            break
    return recs


def random_recommend(top_n: int = 10) -> List[str]:
    with open(MODELS_DIR / "news_id2idx.json", "r", encoding="utf-8") as f:
        id2idx = json.load(f)
    import random
    keys = list(id2idx.keys())
    return random.sample(keys, min(top_n, len(keys)))


if __name__ == "__main__":
    print("Popularity top10:", popularity_recommend(10))
