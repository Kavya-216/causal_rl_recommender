from pathlib import Path
import pandas as pd
import re
import json
import hashlib
import logging
from datetime import datetime


BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data" / "raw" / "MIND"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("mind_preprocess")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)


STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "is",
    "in",
    "on",
    "for",
    "of",
    "to",
}


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[\r\n]+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [t for t in text.split() if t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)


def load_news(news_path: Path) -> pd.DataFrame:
    cols = ["news_id", "category", "subcategory", "title", "abstract", "url"]
    df = pd.read_csv(news_path, sep="\t", header=None, names=cols, quoting=3, dtype=str)
    df = df.fillna("")
    df["title_norm"] = df["title"].map(normalize_text)
    df["abstract_norm"] = df["abstract"].map(normalize_text)
    # validate
    if df["news_id"].isnull().any():
        logger.warning("Some news_id values are missing")
    return df[["news_id", "category", "title", "abstract", "title_norm", "abstract_norm"]]


def parse_impressions(impr: str):
    # impressions format: "Nxxx-1 Nyyy-0 ..." where -1 means clicked
    items = []
    for token in str(impr).split():
        if "-" in token:
            try:
                nid, clicked = token.split("-")
                clicked = int(clicked)
                items.append((nid, 1 if clicked == 1 else 0))
            except Exception:
                continue
    return items


def load_behaviors(beh_path: Path) -> pd.DataFrame:
    # Expected columns: impression_id, user_id, time, history, impressions
    cols = ["impression_id", "user_id", "time", "history", "impressions"]
    df = pd.read_csv(beh_path, sep="\t", header=None, names=cols, quoting=3, dtype=str)
    df = df.fillna("")

    rows = []
    for _, r in df.iterrows():
        impressions = parse_impressions(r["impressions"]) if r["impressions"] else []
        for news_id, clicked in impressions:
            rows.append({
                "impression_id": r["impression_id"],
                "user_id": r["user_id"],
                "time": r["time"],
                "history": r["history"],
                "news_id": news_id,
                "clicked": clicked,
            })

    out = pd.DataFrame(rows)
    # minimal cleaning
    out = out.dropna(subset=["news_id", "user_id"]) 
    if out.empty:
        logger.warning("No impressions parsed from behaviors file")
    out["clicked"] = out["clicked"].astype(int)
    return out


def merge_and_save(news_df: pd.DataFrame, interactions_df: pd.DataFrame):
    merged = interactions_df.merge(news_df, on="news_id", how="left")
    merged = merged.fillna("")
    news_out = PROCESSED_DIR / "news_processed.csv"
    interactions_out = PROCESSED_DIR / "interactions_processed.csv"
    news_df.to_csv(news_out, index=False)
    merged.to_csv(interactions_out, index=False)
    # Save a small metadata file
    meta = {
        "news_count": int(news_df.shape[0]),
        "interactions_count": int(merged.shape[0]),
        "processed_at": datetime.utcnow().isoformat() + "Z",
    }
    with open(PROCESSED_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def needs_reprocessing(news_path: Path, beh_path: Path) -> bool:
    news_out = PROCESSED_DIR / "news_processed.csv"
    inter_out = PROCESSED_DIR / "interactions_processed.csv"
    sig_path = PROCESSED_DIR / "raw_signature.json"
    if not news_out.exists() or not inter_out.exists() or not sig_path.exists():
        return True
    # compute raw hashes
    try:
        h_news = _file_hash(news_path)
        h_beh = _file_hash(beh_path)
    except Exception:
        return True
    with open(sig_path, "r", encoding="utf-8") as f:
        sig = json.load(f)
    return sig.get("news_hash") != h_news or sig.get("beh_hash") != h_beh


def update_signature(news_path: Path, beh_path: Path):
    sig_path = PROCESSED_DIR / "raw_signature.json"
    h_news = _file_hash(news_path)
    h_beh = _file_hash(beh_path)
    with open(sig_path, "w", encoding="utf-8") as f:
        json.dump({"news_hash": h_news, "beh_hash": h_beh}, f)


def main(force: bool = False):
    news_path = RAW_DIR / "news.tsv"
    beh_path = RAW_DIR / "behaviors.tsv"
    if not news_path.exists() or not beh_path.exists():
        raise FileNotFoundError("news.tsv or behaviors.tsv not found in data/raw/MIND/")

    if not force and not needs_reprocessing(news_path, beh_path):
        logger.info("Processed files are up-to-date. Skipping reprocessing.")
        return

    logger.info("Starting preprocessing")
    news_df = load_news(news_path)
    interactions_df = load_behaviors(beh_path)
    merge_and_save(news_df, interactions_df)
    update_signature(news_path, beh_path)
    logger.info("Preprocessing complete. Files saved to data/processed/")


if __name__ == "__main__":
    main()
