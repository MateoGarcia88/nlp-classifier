"""Configuracion compartida: constantes, rutas y la definicion de los conjuntos
de etiquetas que alimentan tanto la corrida principal como el experimento de
sensibilidad a la redaccion.

Orden canonico de clases de AG News (ids de la libreria datasets):
    0 = World, 1 = Sports, 2 = Business, 3 = Sci/Tech
Cada conjunto de etiquetas de abajo lista sus 4 redacciones en *ese mismo orden*,
de modo que una redaccion predicha vuelve a su id de clase verdadero por posicion.
"""
from pathlib import Path

# --- Reproducibilidad / escala ---
SEED = 42                                  # semilla fija = mismos 500 ejemplos siempre
N_SUBSET = 500                             # tamano del subconjunto de evaluacion
MODEL_NAME = "facebook/bart-large-mnli"    # modelo NLI preentrenado (sin fine-tuning)
# Id del repo con namespace (el antiguo "ag_news" sin namespace rompe el nuevo
# parser de URIs de huggingface_hub, que exige el formato "namespace/name").
DATASET_NAME = "fancyzhx/ag_news"

# --- Rutas ---
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"
for _d in (DATA_DIR, RESULTS_DIR, FIG_DIR):
    _d.mkdir(exist_ok=True)                # crea las carpetas si aun no existen

SUBSET_CSV = DATA_DIR / "agnews_subset.csv"

# IMPORTANTE: nombres de clase legibles en ORDEN canonico (la posicion = el id).
# Este orden es el "contrato" que conecta cada redaccion con su clase verdadera.
CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]

# Conjuntos de etiquetas para el experimento de sensibilidad a la redaccion ---
# clave -> lista de 4 etiquetas candidatas (en orden canonico de clase).
# La plantilla de hipotesis se mantiene FIJA ("This text is about {}.") para que lo
# UNICO que varia entre estas corridas sea la propia redaccion de la etiqueta.
LABEL_SETS = {
    # Los nombres de categoria escuetos del propio dataset.
    "canonical": ["World", "Sports", "Business", "Sci/Tech"],
    # Redacciones naturales, algo mas descriptivas.
    "natural": [
        "world news",
        "sports",
        "business",
        "science and technology",
    ],
    # Reescritas a proposito hacia casi-sinonimos para sondear la sensibilidad.
    "synonyms": [
        "international affairs",
        "athletics",
        "finance and markets",
        "technology",
    ],
    # Una sola palabra, lo mas escueto posible.
    "terse": ["politics", "sports", "economy", "science"],
}

# CLAVE del enfoque NLI: convierte cada etiqueta en una frase de entailment.
# Se mantiene fija para aislar la redaccion de la etiqueta como unica variable.
HYPOTHESIS_TEMPLATE = "This text is about {}."

# Conjunto tratado como resultado "principal" en la seccion de metricas base.
PRIMARY_LABEL_SET = "natural"
