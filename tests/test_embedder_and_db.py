import tempfile
from pathlib import Path
import numpy as np
import json

from src.preprocessing import embedder as embed_mod
from src.preprocessing import populate_database as pop_db


def test_embedder_generates_shapes(monkeypatch, tmp_path):
    # create fake processed news csv
    news_csv = tmp_path / "news_processed.csv"
    news_csv.write_text("news_id,category,title,abstract,title_norm,abstract_norm\nN1,cat,Title,Abs,t1,a1\nN2,cat,Title2,Abs2,t2,a2\n")

    # monkeypatch paths
    embed_mod.BASE_DIR = tmp_path
    embed_mod.PROCESSED_DIR = tmp_path
    embed_mod.MODELS_DIR = tmp_path

    # monkeypatch sentence transformer to return deterministic embeddings
    class DummyModel:
        def encode(self, texts, show_progress_bar=True, convert_to_numpy=True):
            return np.arange(len(texts) * 8).reshape(len(texts), 8).astype(np.float32)

    monkeypatch.setattr(embed_mod, "SentenceTransformer", lambda name: DummyModel())

    embed_mod.main()

    emb = np.load(tmp_path / "news_embeddings.npy")
    assert emb.shape[0] == 2
    with open(tmp_path / "news_id2idx.json", "r", encoding="utf-8") as f:
        d = json.load(f)
    assert "N1" in d


def test_populate_database_creates_db(monkeypatch, tmp_path):
    # create processed files and embeddings
    news_csv = tmp_path / "news_processed.csv"
    news_csv.write_text("news_id,category,title,abstract,title_norm,abstract_norm\nN1,cat,Title,Abs,t1,a1\n")
    inter_csv = tmp_path / "interactions_processed.csv"
    inter_csv.write_text("impression_id,user_id,time,history,news_id,clicked\nI1,u1,now,,N1,1\n")
    emb = np.random.randn(1, 8).astype(np.float32)
    np.save(tmp_path / "news_embeddings.npy", emb)
    (tmp_path / "news_id2idx.json").write_text('{"N1": 0}')

    monkeypatch.setattr(pop_db, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(pop_db, "MODELS_DIR", tmp_path)
    # set DB path to tmp
    monkeypatch.setattr(pop_db, "DB_DIR", tmp_path)
    monkeypatch.setattr(pop_db, "DB_PATH", tmp_path / "recommender.db")

    pop_db.populate()
    assert (tmp_path / "recommender.db").exists()
