"""Baseline supervisado opcional: TF-IDF + regresion logistica.

Proposito: un punto de referencia. El zero-shot usa *cero* datos de entrenamiento
etiquetados; este baseline se entrena con un pequeno conjunto etiquetado en segundos.
Sirve para enmarcar "que tan bueno es el zero-shot en realidad". El baseline se entrena
con la particion *train* de AG News (totalmente separada de nuestro subconjunto de
evaluacion, que vino de la particion test) y se evalua sobre EL MISMO subconjunto
congelado que vio el modelo zero-shot.
"""
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score

from config import SEED, SUBSET_CSV, RESULTS_DIR, CLASS_NAMES, DATASET_NAME

N_TRAIN = 8000  # conjunto de entrenamiento pequeno; entrena en segundos


def main() -> None:
    eval_df = pd.read_csv(SUBSET_CSV)  # mismos 500 ejemplos que vio el zero-shot

    print(f"[baseline] loading AG News train split ({N_TRAIN} examples)...")
    # IMPORTANTE: se entrena con la particion TRAIN, distinta del test de evaluacion
    # => no hay fuga de datos (el modelo nunca ve los ejemplos con los que se mide).
    train = load_dataset(DATASET_NAME, split="train").shuffle(seed=SEED) \
        .select(range(N_TRAIN))
    X_train, y_train = train["text"], train["label"]

    # Pipeline clasico: TF-IDF (texto -> vectores de frecuencias ponderadas) y encima
    # una regresion logistica. Bigramas y eliminacion de stopwords en ingles.
    clf = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                        stop_words="english"),
        LogisticRegression(max_iter=1000, C=10.0),
    )
    print("[baseline] fitting TF-IDF + logistic regression...")
    clf.fit(X_train, y_train)

    # Se evalua sobre el subconjunto congelado, para comparar de forma justa.
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
