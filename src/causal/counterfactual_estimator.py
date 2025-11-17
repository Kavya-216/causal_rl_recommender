from pathlib import Path
import pandas as pd
import json
from dowhy import CausalModel


BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def prepare_data():
    news = pd.read_csv(PROCESSED_DIR / "news_processed.csv", dtype=str).fillna("")
    inter = pd.read_csv(PROCESSED_DIR / "interactions_processed.csv", dtype=str).fillna("")
    inter["clicked"] = inter["clicked"].astype(int)

    # user_pref: most frequent category in user's clicked history
    merged = inter.merge(news[["news_id", "category"]], on="news_id", how="left")

    user_prefs = (
        merged[merged["clicked"] == 1].groupby("user_id")
        ["category"]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else "")
        .to_dict()
    )

    merged["user_pref"] = merged["user_id"].map(lambda x: user_prefs.get(x, ""))
    # Define treatment: whether recommended category matches user_pref
    merged["treatment"] = (merged["category"] == merged["user_pref"]).astype(int)
    return merged


def estimate_effect():
    df = prepare_data()
    # Keep only rows with non-empty category and user_pref
    df = df[["treatment", "clicked", "user_pref", "category"]].copy()
    df = df.fillna("")

    # For DoWhy we need numeric covariates; encode user_pref and category as dummies
    df_enc = pd.get_dummies(df, columns=["user_pref", "category"], drop_first=True)

    model = CausalModel(
        data=df_enc,
        treatment="treatment",
        outcome="clicked",
        common_causes=[c for c in df_enc.columns if c not in ["treatment", "clicked"]],
    )

    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.linear_regression",
    )

    ate = float(estimate.value)
    out = {"ate": ate, "num_samples": int(df.shape[0])}
    with open(RESULTS_DIR / "causal_effects.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("Causal estimation complete. Results saved to results/causal_effects.json")
    return out


if __name__ == "__main__":
    estimate_effect()
