"""Shared configuration: constants, paths, and the label-set definitions
that drive both the main run and the wording-sensitivity experiment.

AG News canonical class order (datasets label ids):
    0 = World, 1 = Sports, 2 = Business, 3 = Sci/Tech
Every label set below lists its 4 phrasings in *that same order*, so a
predicted phrasing maps back to a true class id by position.
"""
from pathlib import Path

# --- Reproducibility / scale ---
SEED = 42
N_SUBSET = 500
MODEL_NAME = "facebook/bart-large-mnli"
# Namespaced repo id (the legacy unnamespaced "ag_news" breaks the new
# huggingface_hub URI parser, which requires "namespace/name").
DATASET_NAME = "fancyzhx/ag_news"

# --- Paths ---
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"
for _d in (DATA_DIR, RESULTS_DIR, FIG_DIR):
    _d.mkdir(exist_ok=True)

SUBSET_CSV = DATA_DIR / "agnews_subset.csv"

# Canonical human-readable class names (for plots / confusion matrix axes).
CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]

# --- Label sets for the wording-sensitivity experiment ---
# key -> list of 4 candidate-label strings (in canonical class order).
# The hypothesis template is held fixed ("This text is about {}.") so that the
# ONLY thing varying across these runs is the label wording itself.
LABEL_SETS = {
    # The dataset's own terse category names.
    "canonical": ["World", "Sports", "Business", "Sci/Tech"],
    # Natural, slightly more descriptive phrasings.
    "natural": [
        "world news",
        "sports",
        "business",
        "science and technology",
    ],
    # Deliberately reworded toward near-synonyms to probe sensitivity.
    "synonyms": [
        "international affairs",
        "athletics",
        "finance and markets",
        "technology",
    ],
    # Single-word, maximally terse.
    "terse": ["politics", "sports", "economy", "science"],
}

HYPOTHESIS_TEMPLATE = "This text is about {}."

# The label set treated as the "main" result for the core-metrics section.
PRIMARY_LABEL_SET = "natural"
