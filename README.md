# Zero-Shot Text Classification with a Pretrained NLI Model

> Classify news topics with **zero training data** by repurposing an NLI model —
> and measure how fragile that trick really is.

This project takes `facebook/bart-large-mnli` — a model trained only on **Natural
Language Inference (entailment)** — and uses it to classify AG News topics it was
**never trained on**, with **no fine-tuning**. The deliverable is the *analysis*:
how well the borrowed capability works, where it breaks, and how sensitive it is to
the exact wording of the labels.

📄 Full write-up: [results/FINDINGS.md](results/FINDINGS.md) · framing & decisions: [PROJECT.md](PROJECT.md)

## How it works

There is no classifier head and no training. To classify a piece of text, each
candidate label is reframed as an entailment **hypothesis** and the NLI model scores
how strongly the text entails it. Highest score wins:

```
text  ─► "Apple unveiled a new chip today."   (premise)
labels ─► world / sports / business / technology
              │
              ▼  for each label, test entailment of:
          "This text is about {label}."        (hypothesis)
              │
              ▼  bart-large-mnli entailment scores
          business 0.61 · technology 0.20 · world 0.15 · sports 0.04
              │
              ▼
        PREDICTION: business
```

The candidate labels are an **input you choose** — which is exactly why their wording
matters, and why that becomes the subject of the analysis.

## Headline results (500 AG News examples, CPU)

| Model | Accuracy | Macro-F1 | Training labels used |
|-------|:--------:|:--------:|:--------------------:|
| Zero-shot NLI (label set `natural`) | 0.672 | 0.646 | **0** |
| Zero-shot NLI (best phrasing, `terse`) | 0.712 | 0.689 | **0** |
| TF-IDF + Logistic Regression (baseline) | 0.854 | 0.855 | 8,000 |

**The interesting finding — label wording trades accuracy *between* classes.**
Across four label phrasings the *aggregate* accuracy moves under 5 points, yet up to
**46% of individual predictions flip**. A stable headline number hides large
instability underneath:

- Sci/Tech recall jumps **0.21 → 0.84** just by using `"technology"` instead of
  `"science and technology"` …
- … while that same change craters **Sports** (`"athletics"`, 0.83 → 0.50) and
  **Business** (`"finance and markets"`, 0.78 → 0.51).

The dominant error is **Sci/Tech → Business** — a topic-overlap artifact (AG News
tech stories are mostly about tech *companies* and markets), not a comprehension
failure.

| Confusion matrix | Per-class recall by wording |
|---|---|
| ![confusion matrix](figures/confusion_matrix.png) | ![per-class recall by wording](figures/per_class_recall_by_wording.png) |
| Per-class F1 | Wording sensitivity |
| ![per-class F1](figures/per_class_f1.png) | ![wording sensitivity](figures/wording_sensitivity.png) |

## Setup

```bash
pip install -r requirements.txt
```

Python 3.14 works (torch 2.12 ships a `cp314` wheel). CPU-only; the model is ~1.6 GB
and downloads on first use.

## Reproduce (run from `src/`)

```bash
python prepare_data.py        # freeze a fixed 500-example AG News subset -> data/
python run_zeroshot.py --all  # zero-shot over all 4 label phrasings  (~20 min CPU)
python baseline.py            # TF-IDF + logistic-regression reference  (~10 s)
python analyze.py             # metrics, figures, results/FINDINGS.md
```

Every step writes its artifacts to disk, so they are independently rerunnable.

## Interactive demo (Gradio UI)

```bash
python src/app.py        # then open http://127.0.0.1:7860
```

Type any text, load or edit the candidate labels and the hypothesis template, and
see the predicted class plus a confidence score per label. Editing the labels live is
a hands-on demo of the wording-sensitivity finding.

## Run with Docker

CPU-only image; the Gradio demo is the default command.

```bash
docker compose up --build      # then open http://localhost:7860
```

The ~1.6 GB model downloads on first run and is cached in a named volume, so later
starts are fast. To run the pipeline scripts inside the container instead of the UI:

```bash
docker compose run --rm app python prepare_data.py
docker compose run --rm app python run_zeroshot.py --all
docker compose run --rm app python baseline.py
docker compose run --rm app python analyze.py
```

Generated `data/`, `results/`, and `figures/` are mounted back to the host, so those
artifacts appear in your working copy. (Plain Docker without Compose also works:
`docker build -t nlp-zeroshot . && docker run -p 7860:7860 nlp-zeroshot`.)

## Layout

| Path | What |
|------|------|
| `src/config.py` | constants, paths, the 4 label-set phrasings |
| `src/prepare_data.py` | download AG News, freeze seeded subset |
| `src/run_zeroshot.py` | NLI zero-shot run (reused for the wording sweep) |
| `src/baseline.py` | supervised TF-IDF + LogReg baseline |
| `src/analyze.py` | metrics, confusion matrix, wording sensitivity, figures, FINDINGS |
| `src/app.py` | Gradio web UI for interactive prediction |
| `data/` | frozen evaluation subset |
| `results/` | per-run predictions + scores, metric tables, `FINDINGS.md` |
| `figures/` | confusion matrix, per-class F1, wording-sensitivity, per-class trade-off |

## Caveats

- **n = 500** (~125 per class), so per-class numbers carry real sampling noise. The
  robust finding is the large per-class *trade-off* (4× recall swing), not the modest
  ~5-point aggregate move.
- The hypothesis template is held fixed (`"This text is about {}."`) so the experiment
  isolates *label* wording as the only variable.

---

*Course NLP project. Model and dataset are used as-is; no fine-tuning.*
