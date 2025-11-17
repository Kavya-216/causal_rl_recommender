from pathlib import Path
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def build_tfidf(news_texts):
    tf = TfidfVectorizer(max_features=2048)
    mat = tf.fit_transform(news_texts)
    return tf, mat


def build_sentence_transformer(news_texts, model_name: str = "all-MiniLM-L6-v2"):
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers is not installed in this environment")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(list(news_texts), show_progress_bar=True, convert_to_numpy=True)
    return model, embeddings


def save_embeddings(embeddings: np.ndarray, out_path: Path):
    np.save(out_path, embeddings)


def main():
    news_csv = PROCESSED_DIR / "news_processed.csv"
    if not news_csv.exists():
        raise FileNotFoundError("Processed news not found. Run preprocessing first.")

    news = pd.read_csv(news_csv)
    news["text"] = (news["title_norm"].fillna("") + " " + news["abstract_norm"].fillna(""))
    texts = news["text"].tolist()

    # TF-IDF
    tf, tf_mat = build_tfidf(texts)
    # Save TF-IDF matrix as npy and feature names
    np.save(MODELS_DIR / "tfidf_matrix.npy", tf_mat.toarray())
    with open(MODELS_DIR / "tfidf_vocab.json", "w", encoding="utf-8") as f:
        json.dump(tf.get_feature_names_out().tolist(), f)

    # Sentence Transformer
    st_model, st_embeddings = build_sentence_transformer(texts)
    save_embeddings(st_embeddings.astype(np.float32), MODELS_DIR / "news_embeddings.npy")

    # save mapping news_id -> index
    id2idx = {nid: int(idx) for idx, nid in enumerate(news["news_id"].tolist())}
    with open(MODELS_DIR / "news_id2idx.json", "w", encoding="utf-8") as f:
        json.dump(id2idx, f)

    print("Embeddings built and saved to models/")


if __name__ == "__main__":
    main()
