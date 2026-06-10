"""Paso 2 del recorrido: cargar AG News y "congelar" un subconjunto fijo de 500
ejemplos a CSV, para que toda corrida posterior use datos identicos.

La particion test de AG News tiene 7.600 ejemplos balanceados en 4 clases. Tomamos
una muestra aleatoria (via shuffle con semilla fija) y nos quedamos con los primeros
N_SUBSET. Columnas guardadas: text, label (id 0-3), label_name.
"""
import pandas as pd
from datasets import load_dataset

from config import SEED, N_SUBSET, SUBSET_CSV, CLASS_NAMES, DATASET_NAME


def main() -> None:
    # Si el subconjunto ya existe, no se vuelve a descargar (evita trabajo repetido).
    if SUBSET_CSV.exists():
        df = pd.read_csv(SUBSET_CSV)
        print(f"[prepare_data] subset already exists: {SUBSET_CSV} "
              f"({len(df)} rows). Delete it to regenerate.")
        print(df["label_name"].value_counts())
        return

    print("[prepare_data] downloading AG News (test split)...")
    ds = load_dataset(DATASET_NAME, split="test")
    # CLAVE: barajar de forma DETERMINISTA (semilla) y luego cortar los primeros 500.
    # Como la semilla es fija, siempre se obtienen exactamente los mismos ejemplos.
    ds = ds.shuffle(seed=SEED).select(range(N_SUBSET))

    df = pd.DataFrame({"text": ds["text"], "label": ds["label"]})
    df["label_name"] = df["label"].map(dict(enumerate(CLASS_NAMES)))  # id -> nombre
    df.to_csv(SUBSET_CSV, index=False)

    print(f"[prepare_data] saved {len(df)} rows -> {SUBSET_CSV}")
    print("[prepare_data] class balance in subset:")
    print(df["label_name"].value_counts())  # revisa que las 4 clases queden balanceadas


if __name__ == "__main__":
    main()
