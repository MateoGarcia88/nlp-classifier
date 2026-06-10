"""Step 2 of the trail: load AG News and freeze a fixed, seeded 500-example
subset to CSV so every downstream run uses identical data.

AG News test split has 7,600 examples balanced across 4 classes. We take a
stratified-ish random sample via a fixed seed shuffle and keep the first
N_SUBSET. Saved columns: text, label (id 0-3), label_name.
"""
import pandas as pd
from datasets import load_dataset

from config import SEED, N_SUBSET, SUBSET_CSV, CLASS_NAMES, DATASET_NAME


def main() -> None:
    if SUBSET_CSV.exists():
        df = pd.read_csv(SUBSET_CSV)
        print(f"[prepare_data] subset already exists: {SUBSET_CSV} "
              f"({len(df)} rows). Delete it to regenerate.")
        print(df["label_name"].value_counts())
        return

    print("[prepare_data] downloading AG News (test split)...")
    ds = load_dataset(DATASET_NAME, split="test")
    # Shuffle deterministically, then slice.
    ds = ds.shuffle(seed=SEED).select(range(N_SUBSET))

    df = pd.DataFrame({"text": ds["text"], "label": ds["label"]})
    df["label_name"] = df["label"].map(dict(enumerate(CLASS_NAMES)))
    df.to_csv(SUBSET_CSV, index=False)

    print(f"[prepare_data] saved {len(df)} rows -> {SUBSET_CSV}")
    print("[prepare_data] class balance in subset:")
    print(df["label_name"].value_counts())


if __name__ == "__main__":
    main()
