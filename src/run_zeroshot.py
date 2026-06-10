"""Paso 3 del recorrido: ejecutar la clasificacion zero-shot sobre el subconjunto.

Replantea cada etiqueta candidata como una hipotesis de entailment usando el pipeline
`zero-shot-classification` de HF (modelo: facebook/bart-large-mnli) y registra, para
cada ejemplo, la clase predicha y el vector completo de puntuaciones sobre las etiquetas.

Reutilizable: se le pasa una clave de LABEL_SETS. La misma funcion alimenta tanto la
corrida principal como el barrido de sensibilidad, asi todas las corridas son comparables.

Uso:
    python run_zeroshot.py                 # corre el PRIMARY_LABEL_SET
    python run_zeroshot.py canonical       # corre un conjunto de etiquetas concreto
    python run_zeroshot.py --all           # corre todos los conjuntos de LABEL_SETS
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

_PIPE = None  # se construye de forma perezosa y se reutiliza entre conjuntos


def get_pipe():
    # Construye el pipeline UNA sola vez (patron singleton perezoso): cargar el
    # modelo de 1.6 GB es caro, asi que se cachea para no repetirlo en cada conjunto.
    global _PIPE
    if _PIPE is None:
        device = 0 if torch.cuda.is_available() else -1   # 0 = GPU, -1 = CPU
        print(f"[zeroshot] loading {MODEL_NAME} (device={'cuda' if device==0 else 'cpu'})...")
        _PIPE = pipeline("zero-shot-classification", model=MODEL_NAME, device=device)
    return _PIPE


def run_label_set(key: str, df: pd.DataFrame) -> pd.DataFrame:
    """Clasifica cada fila de df bajo el conjunto de etiquetas `key`. Devuelve un
    DataFrame con el *id de clase* predicho (mapeado por posicion) y las puntuaciones
    por clase. Esta es la funcion central del proyecto."""
    if key not in LABEL_SETS:
        raise KeyError(f"unknown label set '{key}'. options: {list(LABEL_SETS)}")
    labels = LABEL_SETS[key]
    # La posicion de cada redaccion == su id de clase canonico (asi se mapea de vuelta).
    phrasing_to_id = {lab: i for i, lab in enumerate(labels)}

    pipe = get_pipe()
    texts = df["text"].tolist()

    print(f"[zeroshot] label set '{key}': {labels}")
    t0 = time.time()
    # AQUI ocurre el zero-shot: cada texto se contrasta contra cada etiqueta-hipotesis.
    # multi_label=False => las puntuaciones compiten entre si (softmax, suman 1).
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
        # El pipeline devuelve las etiquetas ORDENADAS por puntuacion descendente;
        # reconstruimos un vector en orden fijo para que las columnas sean estables.
        score_by_phrasing = dict(zip(res["labels"], res["scores"]))
        pred_phrasing = res["labels"][0]              # la etiqueta ganadora
        pred_id = phrasing_to_id[pred_phrasing]       # de vuelta a id de clase
        row = {
            "true_id": true_id,
            "true_name": CLASS_NAMES[true_id],
            "pred_id": pred_id,
            "pred_name": CLASS_NAMES[pred_id],
            "top_phrasing": pred_phrasing,
            "top_score": res["scores"][0],
        }
        # puntuaciones por clase en orden canonico, con nombres de columna estables
        for i, lab in enumerate(labels):
            row[f"score_{CLASS_NAMES[i]}"] = score_by_phrasing[lab]
        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_df.insert(0, "text", df["text"].values)
    return out_df


def save(key: str, out_df: pd.DataFrame) -> None:
    # Guarda el CSV de predicciones y calcula la accuracy al vuelo como control rapido.
    path = RESULTS_DIR / f"preds_{key}.csv"
    out_df.to_csv(path, index=False)
    acc = (out_df["true_id"] == out_df["pred_id"]).mean()
    print(f"[zeroshot] saved -> {path}  (accuracy={acc:.3f})")


def main(argv) -> None:
    df = pd.read_csv(SUBSET_CSV)
    # Selecciona que conjuntos correr segun los argumentos de linea de comandos.
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
