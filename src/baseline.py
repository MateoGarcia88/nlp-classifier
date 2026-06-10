"""Optional supervised baseline: TF-IDF + logistic regression.

Purpose: a reference point. Zero-shot uses *zero* labeled training data; this
baseline trains on a small labeled set in seconds. It frames "how good is
zero-shot really?" The baseline trains on AG News *train* split (held entirely
separate from our evaluation subset, which came from the test split) and is
evaluated on the SAME frozen subset the zero-shot model saw.
"""
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score

from config import SEED, SUBSET_CSV, RESULTS_DIR, CLASS_NAMES, DATASET_NAME

N_TRAIN = 8000  # small labeled training set; trains in seconds


def main() -> None:
    eval_df = pd.read_csv(SUBSET_CSV)

    print(f"[baseline] loading AG News train split ({N_TRAIN} examples)...")
    train = load_dataset(DATASET_NAME, split="train").shuffle(seed=SEED) \
        .select(range(N_TRAIN))
    X_train, y_train = train["text"], train["label"]

    clf = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                        stop_words="english"),
        LogisticRegression(max_iter=1000, C=10.0),
    )
    print("[baseline] fitting TF-IDF + logistic regression...")
    clf.fit(X_train, y_train)

    y_true = eval_df["label"]
    y_pred = clf.predict(eval_df["text"])

    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro")
    per = f1_score(y_true, y_pred, average=None, labels=range(len(CLASS_NAMES)))

    out = pd.DataFrame([{
        "model": f"TF-IDF + LogReg (trained on {N_TRAIN})",
        "accuracy": acc,
        "macro_f1": mf1,
        **{f"f1_{c}": v for c, v in zip(CLASS_NAMES, per)},
    }])
    path = RESULTS_DIR / "baseline_metrics.csv"
    out.to_csv(path, index=False)
    print(f"[baseline] accuracy={acc:.3f}  macro_f1={mf1:.3f}  -> {path}")


if __name__ == "__main__":
    main()
