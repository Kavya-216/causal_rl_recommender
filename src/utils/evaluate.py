from typing import List, Dict, Callable
from pathlib import Path
import json

# Lightweight evaluation utilities. Avoid heavy imports at module import time.
BASE_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = BASE_DIR / "models"
DB_PATH = BASE_DIR / "database" / "recommender.db"


def compute_ctr(recommendations: Dict[str, List[str]], interactions: Dict[str, List[str]]) -> float:
    clicks = 0
    impressions = 0
    for u, recs in recommendations.items():
        impressions += len(recs)
        clicked = set(interactions.get(u, []))
        clicks += sum(1 for r in recs if r in clicked)
    return float(clicks) / impressions if impressions > 0 else 0.0


def save_metrics(metrics: Dict, fname: str = "metrics.json") -> None:
    with open(RESULTS_DIR / fname, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def evaluate(recommender_fn: Callable[[str, int], List[str]], users: List[str], interactions: Dict[str, List[str]], k: int = 5) -> Dict:
    recs: Dict[str, List[str]] = {}
    for u in users:
        try:
            recs[u] = recommender_fn(u, k=k)
        except Exception:
            # some recommenders may ignore user id
            try:
                recs[u] = recommender_fn(k=k)
            except Exception:
                recs[u] = []

    ctr = compute_ctr(recs, interactions)
    metrics = {"ctr": ctr, "num_users": len(users)}
    save_metrics(metrics)
    return metrics
from typing import List, Dict, Callable
from pathlib import Path
import json

# Lightweight evaluation utilities. Avoid heavy imports at module import time.
BASE_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = BASE_DIR / "models"
DB_PATH = BASE_DIR / "database" / "recommender.db"


def compute_ctr(recommendations: Dict[str, List[str]], interactions: Dict[str, List[str]]) -> float:
    clicks = 0
    impressions = 0
    for u, recs in recommendations.items():
        impressions += len(recs)
        clicked = set(interactions.get(u, []))
        clicks += sum(1 for r in recs if r in clicked)
    return float(clicks) / impressions if impressions > 0 else 0.0


def save_metrics(metrics: Dict, fname: str = "metrics.json") -> None:
    with open(RESULTS_DIR / fname, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def evaluate(recommender_fn: Callable[[str, int], List[str]], users: List[str], interactions: Dict[str, List[str]], k: int = 5) -> Dict:
    recs: Dict[str, List[str]] = {}
    for u in users:
        try:
            recs[u] = recommender_fn(u, k=k)
        except Exception:
            # some recommenders may ignore user id
            try:
                recs[u] = recommender_fn(k=k)
            except Exception:
                recs[u] = []

    ctr = compute_ctr(recs, interactions)
    metrics = {"ctr": ctr, "num_users": len(users)}
    save_metrics(metrics)
    return metrics


def evaluate_all(top_n: int = 10) -> Dict:
    """Evaluate built-in baseline recommenders. This function keeps imports local to avoid heavy deps.

    Requires a populated `interactions` table in `database/recommender.db` if you want real evaluation.
    """
    if not DB_PATH.exists():
        raise FileNotFoundError("Database not found. Run populate_database first.")

    import sqlite3
    import pandas as pd
    import numpy as np

    conn = sqlite3.connect(DB_PATH)
    inter = pd.read_sql_query("SELECT user_id, news_id, clicked FROM interactions", conn)
    users = inter["user_id"].unique().tolist()
    conn.close()

    # Local imports of baseline recommenders
    from src.utils.baseline_recommenders import popularity_recommend, tfidf_cosine_recommend, random_recommend

    results: Dict[str, Dict] = {}
    results["popularity"] = evaluate(lambda u, k=top_n: popularity_recommend(k=top_n), users, inter.groupby("user_id").apply(lambda df: df[df["clicked"] == 1]["news_id"].tolist()).to_dict(), top_n)
    results["tfidf"] = evaluate(lambda u, k=top_n: tfidf_cosine_recommend(u, k=top_n), users, inter.groupby("user_id").apply(lambda df: df[df["clicked"] == 1]["news_id"].tolist()).to_dict(), top_n)
    results["random"] = evaluate(lambda u, k=top_n: random_recommend(k=top_n), users, inter.groupby("user_id").apply(lambda df: df[df["clicked"] == 1]["news_id"].tolist()).to_dict(), top_n)

    # RL evaluation is optional and lazy
    model_path = MODELS_DIR / "rl" / "ppo_model.zip"
    emb_path = MODELS_DIR / "news_embeddings.npy"
    id2idx_path = MODELS_DIR / "news_id2idx.json"
    if model_path.exists() and emb_path.exists() and id2idx_path.exists():
        try:
            from stable_baselines3 import PPO  # type: ignore[import]
            from src.rl.reco_env import RecoEnv

            embs = np.load(emb_path)
            with open(id2idx_path, "r", encoding="utf-8") as f:
                id2idx = json.load(f)
            inv_map = {int(v): k for k, v in id2idx.items()}
            model = PPO.load(str(model_path))
            env = RecoEnv(embs, [inv_map[i] for i in range(embs.shape[0])])

            total_recs = 0
            total_clicks = 0
            inter_map = pd.read_sql_query("SELECT user_id, news_id, clicked FROM interactions", sqlite3.connect(DB_PATH)).groupby("user_id").apply(lambda df: df[df["clicked"] == 1]["news_id"].tolist()).to_dict()

            for uid in users:
                obs, _ = env.reset()
                action, _ = model.predict(obs)
                rec = env.news_ids[int(action)]
                total_recs += 1
                if rec in inter_map.get(uid, []):
                    total_clicks += 1

            ctr = float(total_clicks) / total_recs if total_recs > 0 else 0.0
            results["rl_ppo"] = {"ctr": ctr}
        except Exception:
            results["rl_ppo"] = {"ctr": None}
    else:
        results["rl_ppo"] = {"ctr": None}

    with open(RESULTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("Evaluation complete. Results saved to results/metrics.json")
    from typing import List, Dict, Callable
    from pathlib import Path
    import json

    # Lightweight evaluation utilities. Avoid heavy imports at module import time.
    BASE_DIR = Path(__file__).resolve().parents[2]
    RESULTS_DIR = BASE_DIR / "results"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR = BASE_DIR / "models"
    DB_PATH = BASE_DIR / "database" / "recommender.db"


    def compute_ctr(recommendations: Dict[str, List[str]], interactions: Dict[str, List[str]]) -> float:
        clicks = 0
        impressions = 0
        for u, recs in recommendations.items():
            impressions += len(recs)
            clicked = set(interactions.get(u, []))
            clicks += sum(1 for r in recs if r in clicked)
        return float(clicks) / impressions if impressions > 0 else 0.0


    def save_metrics(metrics: Dict, fname: str = "metrics.json") -> None:
        with open(RESULTS_DIR / fname, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)


    def evaluate(recommender_fn: Callable[[str, int], List[str]], users: List[str], interactions: Dict[str, List[str]], k: int = 5) -> Dict:
        recs: Dict[str, List[str]] = {}
        for u in users:
            try:
                recs[u] = recommender_fn(u, k=k)
            except Exception:
                # some recommenders may ignore user id
                try:
                    recs[u] = recommender_fn(k=k)
                except Exception:
                    recs[u] = []

        ctr = compute_ctr(recs, interactions)
        metrics = {"ctr": ctr, "num_users": len(users)}
        save_metrics(metrics)
        return metrics


    def evaluate_all(top_n: int = 10) -> Dict:
        """Evaluate built-in baseline recommenders. This function keeps imports local to avoid heavy deps.

        Requires a populated `interactions` table in `database/recommender.db` if you want real evaluation.
        """
        if not DB_PATH.exists():
            raise FileNotFoundError("Database not found. Run populate_database first.")

        import sqlite3
        import pandas as pd
        import numpy as np

        conn = sqlite3.connect(DB_PATH)
        inter = pd.read_sql_query("SELECT user_id, news_id, clicked FROM interactions", conn)
        users = inter["user_id"].unique().tolist()
        conn.close()

        # Local imports of baseline recommenders
        from src.utils.baseline_recommenders import popularity_recommend, tfidf_cosine_recommend, random_recommend

        inter_map = inter.groupby("user_id").apply(lambda df: df[df["clicked"] == 1]["news_id"].tolist()).to_dict()

        results: Dict[str, Dict] = {}
        results["popularity"] = evaluate(lambda u, k=top_n: popularity_recommend(k=top_n), users, inter_map, top_n)
        results["tfidf"] = evaluate(lambda u, k=top_n: tfidf_cosine_recommend(u, k=top_n), users, inter_map, top_n)
        results["random"] = evaluate(lambda u, k=top_n: random_recommend(k=top_n), users, inter_map, top_n)

        # RL evaluation is optional and lazy
        model_path = MODELS_DIR / "rl" / "ppo_model.zip"
        emb_path = MODELS_DIR / "news_embeddings.npy"
        id2idx_path = MODELS_DIR / "news_id2idx.json"
        if model_path.exists() and emb_path.exists() and id2idx_path.exists():
            try:
                from stable_baselines3 import PPO  # type: ignore[import]
                from src.rl.reco_env import RecoEnv

                embs = np.load(emb_path)
                with open(id2idx_path, "r", encoding="utf-8") as f:
                    id2idx = json.load(f)
                inv_map = {int(v): k for k, v in id2idx.items()}
                model = PPO.load(str(model_path))
                env = RecoEnv(embs, [inv_map[i] for i in range(embs.shape[0])])

                total_recs = 0
                total_clicks = 0

                for uid in users:
                    obs, _ = env.reset()
                    action, _ = model.predict(obs)
                    rec = env.news_ids[int(action)]
                    total_recs += 1
                    if rec in inter_map.get(uid, []):
                        total_clicks += 1

                ctr = float(total_clicks) / total_recs if total_recs > 0 else 0.0
                results["rl_ppo"] = {"ctr": ctr}
            except Exception:
                results["rl_ppo"] = {"ctr": None}
        else:
            results["rl_ppo"] = {"ctr": None}

        with open(RESULTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        print("Evaluation complete. Results saved to results/metrics.json")
        return results


    if __name__ == "__main__":
        evaluate_all()

