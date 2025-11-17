from pathlib import Path
import json
import numpy as np
import pandas as pd
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    LargeBinary,
    Boolean,
    Text,
)
from sqlalchemy.orm import declarative_base, sessionmaker
from io import BytesIO


BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
DB_DIR = BASE_DIR / "database"
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DB_DIR / "recommender.db"

Base = declarative_base()


def _serialize_embedding(arr: np.ndarray) -> bytes:
    bio = BytesIO()
    np.save(bio, arr.astype(np.float32), allow_pickle=False)
    return bio.getvalue()


def _deserialize_embedding(blob: bytes) -> np.ndarray:
    bio = BytesIO(blob)
    bio.seek(0)
    return np.load(bio)


class News(Base):
    __tablename__ = "news"
    id = Column(String, primary_key=True)
    category = Column(String)
    title = Column(Text)
    abstract = Column(Text)
    embedding = Column(LargeBinary)


class User(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True)
    history = Column(Text)  # JSON list of news ids


class Interaction(Base):
    __tablename__ = "interactions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String)
    news_id = Column(String)
    clicked = Column(Boolean)


def create_db(engine):
    Base.metadata.create_all(engine)


def populate():
    if not (PROCESSED_DIR / "news_processed.csv").exists():
        raise FileNotFoundError("Processed news not found. Run preprocessing first.")
    if not (PROCESSED_DIR / "interactions_processed.csv").exists():
        raise FileNotFoundError("Processed interactions not found. Run preprocessing first.")
    if not (MODELS_DIR / "news_embeddings.npy").exists():
        raise FileNotFoundError("News embeddings not found. Run embedder first.")
    if not (MODELS_DIR / "news_id2idx.json").exists():
        raise FileNotFoundError("news_id2idx.json not found. Run embedder first.")

    news_df = pd.read_csv(PROCESSED_DIR / "news_processed.csv", dtype=str).fillna("")
    inter_df = pd.read_csv(PROCESSED_DIR / "interactions_processed.csv", dtype=str).fillna("")
    inter_df["clicked"] = inter_df["clicked"].astype(int)

    embeddings = np.load(MODELS_DIR / "news_embeddings.npy")
    with open(MODELS_DIR / "news_id2idx.json", "r", encoding="utf-8") as f:
        id2idx = json.load(f)

    engine = create_engine(f"sqlite:///{DB_PATH}")
    create_db(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    # News
    for _, r in news_df.iterrows():
        nid = r["news_id"]
        idx = id2idx.get(nid)
        if idx is None or idx >= embeddings.shape[0]:
            emb_blob = None
        else:
            emb_blob = _serialize_embedding(embeddings[int(idx)])

        news = News(
            id=nid,
            category=r.get("category", ""),
            title=r.get("title", ""),
            abstract=r.get("abstract", ""),
            embedding=emb_blob,
        )
        session.merge(news)

    session.commit()

    # Users and interactions
    users = {}
    for _, r in inter_df.iterrows():
        user_id = r["user_id"]
        news_id = r["news_id"]
        clicked = int(r["clicked"]) == 1
        inter = Interaction(user_id=user_id, news_id=news_id, clicked=clicked)
        session.add(inter)
        users.setdefault(user_id, []).append(news_id)

    # create user entries
    for uid, history in users.items():
        user = User(id=uid, history=json.dumps(history))
        session.merge(user)

    session.commit()
    session.close()
    print(f"Database created and populated at {DB_PATH}")


if __name__ == "__main__":
    populate()
