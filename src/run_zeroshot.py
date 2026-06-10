"""Step 3 of the trail: run zero-shot classification over the frozen subset.

Reframes each candidate label as an entailment hypothesis via the HF
`zero-shot-classification` pipeline (model: facebook/bart-large-mnli) and records,
for every example, the predicted class and the full score vector over labels.

Reusable: pass a LABEL_SETS key. The same function powers both the primary run
and the wording-sensitivity sweep, so all runs are strictly comparable.

Usage:
    python run_zeroshot.py                 # runs the PRIMARY_LABEL_SET
    python run_zeroshot.py canonical       # runs a specific label set
    python run_zeroshot.py --all           # runs every label set in LABEL_SETS
"""
import sys
import time

import pandas as pd
import torch
from transformers import pipeline

from config import (
    MODEL_NAME, LABEL_SETS, HYPOTHESIS_TEMPLATE, PRIMARY_LABEL_SET,
    SUBSET_CSV, RESULTS_DIR, CLASS_NAMES,
)

_PIPE = None  # lazily built, reused across label sets in one process


def get_pipe():
    global _PIPE
    if _PIPE is None:
        device = 0 if torch.cuda.is_available() else -1
        print(f"[zeroshot] loading {MODEL_NAME} (device={'cuda' if device==0 else 'cpu'})...")
        _PIPE = pipeline("zero-shot-classification", model=MODEL_NAME, device=device)
    return _PIPE


def run_label_set(key: str, df: pd.DataFrame) -> pd.DataFrame:
    """Classify every row of df under label set `key`. Returns a results frame
    with the predicted *class id* (mapped back via position) and per-class scores."""
    if key not in LABEL_SETS:
        raise KeyError(f"unknown label set '{key}'. options: {list(LABEL_SETS)}")
    labels = LABEL_SETS[key]
    # position of each phrasing == its canonical class id
    phrasing_to_id = {lab: i for i, lab in enumerate(labels)}

    pipe = get_pipe()
    texts = df["text"].tolist()

    print(f"[zeroshot] label set '{key}': {labels}")
    t0 = time.time()
    out = pipe(
        texts,
        candidate_labels=labels,
        hypothesis_template=HYPOTHESIS_TEMPLATE,
        multi_label=False,
        batch_size=8,
    )
    dt = time.time() - t0
    print(f"[zeroshot] '{key}' done: {len(texts)} examples in {dt:.1f}s "
          f"({dt/len(texts):.3f}s/example)")

    rows = []
    for true_id, res in zip(df["label"].tolist(), out):
        # pipeline returns labels sorted by score desc; rebuild a fixed-order vector
        score_by_phrasing = dict(zip(res["labels"], res["scores"]))
        pred_phrasing = res["labels"][0]
        pred_id = phrasing_to_id[pred_phrasing]
        row = {
            "true_id": true_id,
            "true_name": CLASS_NAMES[true_id],
            "pred_id": pred_id,
            "pred_name": CLASS_NAMES[pred_id],
            "top_phrasing": pred_phrasing,
            "top_score": res["scores"][0],
        }
        # per-class scores in canonical order, with stable column names
        for i, lab in enumerate(labels):
            row[f"score_{CLASS_NAMES[i]}"] = score_by_phrasing[lab]
        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_df.insert(0, "text", df["text"].values)
    return out_df


def save(key: str, out_df: pd.DataFrame) -> None:
    path = RESULTS_DIR / f"preds_{key}.csv"
    out_df.to_csv(path, index=False)
    acc = (out_df["true_id"] == out_df["pred_id"]).mean()
    print(f"[zeroshot] saved -> {path}  (accuracy={acc:.3f})")


def main(argv) -> None:
    df = pd.read_csv(SUBSET_CSV)
    if "--all" in argv:
        keys = list(LABEL_SETS)
    elif len(argv) > 1:
        keys = [argv[1]]
    else:
        keys = [PRIMARY_LABEL_SET]

    for key in keys:
        save(key, run_label_set(key, df))


if __name__ == "__main__":
    main(sys.argv)
