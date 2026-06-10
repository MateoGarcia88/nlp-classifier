# Zero-Shot Text Classification with a Pretrained NLI Model

## What this project is
Use a model trained on **Natural Language Inference (NLI)** — `facebook/bart-large-mnli` —
to classify text into categories **it was never explicitly trained to predict**. No
fine-tuning. The trick: reframe each candidate label as an entailment hypothesis
("This text is about {label}") and let the NLI model score how strongly the input
premise *entails* each hypothesis. Highest-entailment label wins.

The deliverable is **analysis**, not a model. The model is a fixed, off-the-shelf tool;
the intellectual content is in measuring *how* and *why* it succeeds or fails.

## Why it's interesting
- It's a clean demonstration of **transfer via task reformulation**: one pretrained
  capability (entailment) repurposed for another (topic classification).
- It exposes a fragility that "real" classifiers hide: the predictions depend on the
  *wording of the labels*, not just the input. Quantifying that sensitivity is the
  star of the analysis.

## Approach / stack
- HF `transformers` `zero-shot-classification` pipeline + `facebook/bart-large-mnli` (~1.6 GB).
- `datasets` for AG News (4 classes: World, Sports, Business, Sci/Tech).
- `scikit-learn` for metrics (accuracy, macro-F1, confusion matrix) + optional TF-IDF
  logistic-regression baseline.
- `pandas` / `matplotlib` for tables and figures.
- CPU-only. Runtime is the main constraint (see Plan).

## Environment status
- Python **3.14.3** on Windows. Verified `torch` 2.12.0 ships a `cp314` wheel, so the
  stack installs natively — no Python downgrade or conda needed.

## Plan (mirrors the assignment trail)
1. **Env setup** — install transformers, datasets, torch, scikit-learn, pandas, matplotlib.
2. **Data** — load AG News, take a fixed-seed subset (proposed: **500** examples; see
   Open question). Save the subset so all runs use identical data.
3. **Zero-shot run** — classify the subset, save predictions + per-label scores to CSV.
4. **Core metrics** — accuracy, macro-F1, confusion matrix.
5. **Analysis** (the point of the project):
   - **Label-wording sensitivity**: rerun with 3–4 alternative label phrasings, measure
     the accuracy/F1 swing and how many predictions *flip*.
   - **Per-class error analysis**: which classes get confused, with plausible reasons
     (e.g. Business vs Sci/Tech overlap on tech-company news).
   - **Baseline (optional)**: TF-IDF + logistic regression, to frame zero-shot numbers
     against a trivially-trained supervised model.
6. **Write-up** — figures: confusion matrix, per-class F1 bars, wording-sensitivity
   comparison; short narrative of findings.

## Runtime budget (estimate, CPU)
`bart-large-mnli` does ~1 forward pass per (example × candidate-label). AG News texts are
short. Rough estimate ~0.2–0.4 s/pass → **500 examples × 4 labels ≈ 7–13 min per run**.
The wording-sensitivity experiment reruns this 3–4×, so total compute is the budget driver.
500 keeps the whole project well under an hour; 1000 roughly doubles it.

## Open questions for the user
1. **Subset size**: I propose 500 (fast, statistically fine for these comparisons). OK,
   or do you want 1000 for tighter metrics at ~2× runtime?
2. **TF-IDF baseline**: include it (adds a nice comparison, ~5 min) or skip to stay lean?

## Results (completed)
Run on 500 AG News examples, CPU, ~0.65 s/example (~5 min per label set).

**Core (label set `natural`):** accuracy **0.672**, macro-F1 **0.646**.
**Supervised baseline (TF-IDF + LogReg, 8k train):** accuracy **0.854**, macro-F1 **0.855**.

**Headline finding — wording trades performance *between* classes.** Across the 4
phrasings, aggregate accuracy spans only 0.664–0.712 (**4.8 pts**), but up to **46%
of individual predictions flip**. Per-class recall shows why: e.g. Sci/Tech recall
goes **0.21 → 0.84** just by using "technology" instead of "science and technology",
while the same `synonyms` set craters Sports (0.83→0.50, "athletics") and Business
(0.78→0.51, "finance and markets"). Gains and losses cancel in the mean — a stable
accuracy number hides large instability underneath.

**Dominant error:** Sci/Tech → Business (63/125). Plausibly a topic-overlap artifact
(AG News tech stories are largely about tech *companies*/markets), i.e. the NLI model's
entailment judgement is reasonable; the dataset's science-vs-business boundary is the
artificial part. A narrower cue word recovers the class, supporting that reading.

Figures: `figures/confusion_matrix.png`, `per_class_f1.png`, `wording_sensitivity.png`,
`per_class_recall_by_wording.png`. Full write-up: `results/FINDINGS.md`.

## Decisions log
- **Subset size = 500** (fixed seed). Keeps whole project under ~1 hr. ✓
- **TF-IDF + logistic-regression baseline = included** as a supervised reference point. ✓
- **Dataset id**: used `fancyzhx/ag_news` (namespaced); the legacy `ag_news` id breaks
  the new huggingface_hub URI parser.
- **Extra dependency**: `tabulate` (for pandas `to_markdown` in the write-up).
- **Hypothesis template held fixed** (`"This text is about {}."`) so the wording
  experiment isolates *label* wording as the only variable.
