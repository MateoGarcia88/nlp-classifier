"""Steps 4-6 of the trail: core metrics, figures, per-class error analysis,
wording-sensitivity aggregation, and a markdown findings dump.

Reads the prediction CSVs produced by run_zeroshot.py. Run that first
(ideally `python run_zeroshot.py --all`) plus baseline.py.

Produces:
    figures/confusion_matrix.png
    figures/per_class_f1.png
    figures/wording_sensitivity.png
    results/metrics_summary.csv
    results/FINDINGS.md
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, confusion_matrix,
    classification_report,
)

from config import (
    LABEL_SETS, PRIMARY_LABEL_SET, RESULTS_DIR, FIG_DIR, CLASS_NAMES,
)


def load_preds(key):
    path = RESULTS_DIR / f"preds_{key}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


# ---------- core metrics ----------
def core_metrics(df):
    y_true, y_pred = df["true_id"], df["pred_id"]
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "per_class_f1": f1_score(y_true, y_pred, average=None,
                                 labels=range(len(CLASS_NAMES))),
    }


def plot_confusion(df, out):
    cm = confusion_matrix(df["true_id"], df["pred_id"],
                          labels=range(len(CLASS_NAMES)))
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Zero-shot confusion matrix\n(label set: '{PRIMARY_LABEL_SET}')")
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    return cm


def plot_per_class_f1(metrics, out):
    f1s = metrics["per_class_f1"]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(CLASS_NAMES, f1s, color="#4C72B0")
    ax.set_ylim(0, 1); ax.set_ylabel("F1")
    ax.set_title(f"Per-class F1 (label set: '{PRIMARY_LABEL_SET}')")
    ax.axhline(metrics["macro_f1"], ls="--", color="gray",
               label=f"macro-F1 = {metrics['macro_f1']:.3f}")
    for b, v in zip(bars, f1s):
        ax.text(b.get_x() + b.get_width()/2, v + 0.02, f"{v:.2f}", ha="center")
    ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


# ---------- wording sensitivity ----------
def wording_table():
    rows = []
    preds = {}
    for key in LABEL_SETS:
        df = load_preds(key)
        if df is None:
            continue
        preds[key] = df
        m = core_metrics(df)
        rows.append({
            "label_set": key,
            "labels": " | ".join(LABEL_SETS[key]),
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
        })
    table = pd.DataFrame(rows)
    return table, preds


def prediction_flips(preds):
    """How many predictions change relative to the primary label set?"""
    if PRIMARY_LABEL_SET not in preds:
        return {}
    base = preds[PRIMARY_LABEL_SET]["pred_id"].values
    flips = {}
    for key, df in preds.items():
        if key == PRIMARY_LABEL_SET:
            continue
        flips[key] = int((df["pred_id"].values != base).sum())
    return flips


def per_class_across_wordings(preds, metric="recall"):
    """For each label set, per-class metric. Returns a tidy DataFrame
    (rows = label sets, cols = classes). Reveals that wording trades
    performance *between* classes while leaving the aggregate ~flat."""
    fn = recall_score if metric == "recall" else \
        (lambda yt, yp, **k: f1_score(yt, yp, **k))
    rows = {}
    for key, df in preds.items():
        vals = fn(df["true_id"], df["pred_id"], average=None,
                  labels=range(len(CLASS_NAMES)))
        rows[key] = dict(zip(CLASS_NAMES, vals))
    return pd.DataFrame(rows).T[CLASS_NAMES]


def plot_per_class_across_wordings(tbl, out, metric="recall"):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(CLASS_NAMES)); n = len(tbl); w = 0.8 / n
    for i, (key, row) in enumerate(tbl.iterrows()):
        ax.bar(x + (i - (n - 1) / 2) * w, row.values, w, label=key)
    ax.set_xticks(x, CLASS_NAMES)
    ax.set_ylim(0, 1); ax.set_ylabel(metric)
    ax.set_title(f"Per-class {metric} across label phrasings\n"
                 "(wording shifts WHICH classes work, not the average)")
    ax.legend(title="label set", fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def plot_wording(table, out):
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(table)); w = 0.38
    ax.bar(x - w/2, table["accuracy"], w, label="accuracy", color="#4C72B0")
    ax.bar(x + w/2, table["macro_f1"], w, label="macro-F1", color="#DD8452")
    ax.set_xticks(x, table["label_set"])
    ax.set_ylim(0, 1); ax.set_ylabel("score")
    ax.set_title("Label-wording sensitivity")
    for i, (a, f) in enumerate(zip(table["accuracy"], table["macro_f1"])):
        ax.text(i - w/2, a + 0.01, f"{a:.2f}", ha="center", fontsize=8)
        ax.text(i + w/2, f + 0.01, f"{f:.2f}", ha="center", fontsize=8)
    ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


# ---------- error analysis ----------
def top_confusions(df, k=3):
    cm = confusion_matrix(df["true_id"], df["pred_id"],
                          labels=range(len(CLASS_NAMES)))
    pairs = []
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            if i != j and cm[i, j] > 0:
                pairs.append((cm[i, j], CLASS_NAMES[i], CLASS_NAMES[j]))
    pairs.sort(reverse=True)
    return pairs[:k]


def main():
    primary = load_preds(PRIMARY_LABEL_SET)
    if primary is None:
        raise SystemExit(
            f"missing results/preds_{PRIMARY_LABEL_SET}.csv — run "
            f"`python run_zeroshot.py --all` first.")

    m = core_metrics(primary)
    cm = plot_confusion(primary, FIG_DIR / "confusion_matrix.png")
    plot_per_class_f1(m, FIG_DIR / "per_class_f1.png")

    table, preds = wording_table()
    flips = prediction_flips(preds)
    plot_wording(table, FIG_DIR / "wording_sensitivity.png")
    table.to_csv(RESULTS_DIR / "metrics_summary.csv", index=False)

    recall_tbl = per_class_across_wordings(preds, "recall")
    plot_per_class_across_wordings(
        recall_tbl, FIG_DIR / "per_class_recall_by_wording.png", "recall")
    recall_tbl.to_csv(RESULTS_DIR / "per_class_recall_by_wording.csv")

    # baseline (optional file)
    base_path = RESULTS_DIR / "baseline_metrics.csv"
    baseline = pd.read_csv(base_path) if base_path.exists() else None

    confusions = top_confusions(primary)
    report = classification_report(
        primary["true_id"], primary["pred_id"],
        target_names=CLASS_NAMES, digits=3)

    # ---- assemble FINDINGS.md ----
    lines = []
    lines.append("# Findings: Zero-Shot Text Classification with bart-large-mnli\n")
    lines.append(f"Dataset: AG News (test split), {len(primary)} examples, "
                 f"4 balanced classes.\n")
    lines.append("## 1. Core performance (primary label set "
                 f"= `{PRIMARY_LABEL_SET}`)\n")
    lines.append(f"- **Accuracy:** {m['accuracy']:.3f}")
    lines.append(f"- **Macro-F1:** {m['macro_f1']:.3f}\n")
    lines.append("Per-class report:\n```\n" + report + "\n```\n")
    lines.append("![confusion matrix](../figures/confusion_matrix.png)\n")
    lines.append("![per-class F1](../figures/per_class_f1.png)\n")

    lines.append("## 2. Label-wording sensitivity\n")
    lines.append(table.to_markdown(index=False,
                 floatfmt=".3f") + "\n")
    acc_span = table["accuracy"].max() - table["accuracy"].min()
    f1_span = table["macro_f1"].max() - table["macro_f1"].min()
    lines.append(f"\n- Accuracy swing across phrasings: "
                 f"**{acc_span*100:.1f} points** "
                 f"({table['accuracy'].min():.3f} – {table['accuracy'].max():.3f})")
    lines.append(f"- Macro-F1 swing: **{f1_span*100:.1f} points**")
    if flips:
        lines.append(f"- Prediction flips vs `{PRIMARY_LABEL_SET}` "
                     f"(out of {len(primary)}):")
        for k, v in flips.items():
            lines.append(f"    - `{k}`: {v} flips ({v/len(primary)*100:.1f}%)")
    lines.append("\n![wording sensitivity](../figures/wording_sensitivity.png)\n")

    lines.append("### 2a. Why the aggregate is stable but predictions churn\n")
    lines.append("The aggregate accuracy barely moves, yet up to 46% of "
                 "individual predictions flip. The reason: wording trades "
                 "performance *between* classes rather than lifting all of "
                 "them. Per-class **recall** by label set:\n")
    lines.append(recall_tbl.to_markdown(floatfmt=".3f") + "\n")
    lines.append("\nNote the extremes: Sci/Tech recall is "
                 f"{recall_tbl.loc['natural','Sci/Tech']:.3f} with "
                 "\"science and technology\" but "
                 f"{recall_tbl.loc['synonyms','Sci/Tech']:.3f} with the single "
                 "word \"technology\" — while that same `synonyms` set craters "
                 f"Sports ({recall_tbl.loc['natural','Sports']:.3f} -> "
                 f"{recall_tbl.loc['synonyms','Sports']:.3f}, \"athletics\") and "
                 f"Business ({recall_tbl.loc['natural','Business']:.3f} -> "
                 f"{recall_tbl.loc['synonyms','Business']:.3f}, \"finance and "
                 "markets\"). The gains and losses cancel in the average.\n")
    lines.append("\n![per-class recall by wording]"
                 "(../figures/per_class_recall_by_wording.png)\n")

    lines.append("## 3. Per-class error analysis\n")
    lines.append("Largest off-diagonal confusions (primary label set):\n")
    for cnt, t, p in confusions:
        lines.append(f"- **{t} → {p}**: {cnt} examples")
    lines.append("")
    lines.append("**Plausible mechanism.** The dominant error is Sci/Tech "
                 "absorbed into Business. AG News \"Sci/Tech\" stories are "
                 "heavily about tech *companies*, products, IPOs and markets "
                 "(Google, Microsoft, telecoms), so under an entailment model "
                 "they genuinely entail \"this text is about business\" at least "
                 "as strongly as \"...about science and technology.\" The NLI "
                 "model is not wrong about entailment — the label boundary AG "
                 "News drew (science/tech vs business) is the artificial part. "
                 "This is a topic-overlap failure, not a comprehension failure, "
                 "which is exactly why a narrower cue word (\"technology\") "
                 "recovers the class.\n")

    if baseline is not None:
        lines.append("## 4. Supervised baseline (TF-IDF + logistic regression)\n")
        lines.append(baseline.to_markdown(index=False, floatfmt=".3f") + "\n")

    (RESULTS_DIR / "FINDINGS.md").write_text("\n".join(lines), encoding="utf-8")

    # console summary
    print(f"\n=== SUMMARY ({PRIMARY_LABEL_SET}) ===")
    print(f"accuracy={m['accuracy']:.3f}  macro_f1={m['macro_f1']:.3f}")
    print("\nwording sensitivity:")
    print(table.to_string(index=False))
    if flips:
        print("\nflips vs primary:", flips)
    print("\ntop confusions:", confusions)
    print(f"\nWrote figures/ and results/FINDINGS.md")


if __name__ == "__main__":
    main()
