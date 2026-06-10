"""Front-end en Gradio para el clasificador zero-shot basado en NLI.

Escribe cualquier texto, elige (o redacta) las etiquetas candidatas y observa la
clase predicha con una puntuacion de confianza para cada etiqueta. Como las etiquetas
son libres, la interfaz funciona como demo en vivo del hallazgo central del proyecto:
cambia la redaccion de la etiqueta y mira como se mueve la prediccion.

Ejecutar:
    python app.py
luego abrir en el navegador la URL http://127.0.0.1:7860 que se imprime.
"""
import gradio as gr

from run_zeroshot import get_pipe
from config import LABEL_SETS, HYPOTHESIS_TEMPLATE, PRIMARY_LABEL_SET

# Redacciones del experimento como cadenas separadas por comas, para que el usuario
# pueda editarlas en el sitio.
PRESETS = {k: ", ".join(v) for k, v in LABEL_SETS.items()}

EXAMPLES = [
    "Apple reported record quarterly iPhone sales, sending its stock to an all-time high.",
    "Researchers discovered a new exoplanet orbiting a distant star.",
    "The home team scored in the final minute to win the championship.",
    "The central bank raised interest rates to curb inflation.",
    "Diplomats met in Geneva to negotiate a ceasefire agreement.",
]


def load_preset(key):
    # Al elegir un preset, devuelve sus etiquetas como texto editable.
    return PRESETS[key]


def predict(text, labels_str, template):
    # Callback central de la UI: valida la entrada, clasifica y formatea la salida.
    text = (text or "").strip()
    if not text:
        return {}, "Enter some text to classify."
    labels = [s.strip() for s in labels_str.split(",") if s.strip()]
    if len(labels) < 2:
        return {}, "Provide at least 2 comma-separated candidate labels."
    template = template.strip() or HYPOTHESIS_TEMPLATE
    if "{}" not in template:                       # la plantilla debe tener el hueco {}
        return {}, "Template must contain '{}' where the label goes."

    # Reutiliza el MISMO pipeline que run_zeroshot.py (no recarga el modelo).
    res = get_pipe()(
        text, candidate_labels=labels,
        hypothesis_template=template, multi_label=False,
    )
    scores = dict(zip(res["labels"], res["scores"]))  # gr.Label lo dibuja como barras
    top = res["labels"][0]                            # etiqueta ganadora
    note = (f"**Prediction: {top}**  (confidence {res['scores'][0]:.3f})\n\n"
            f"Hypothesis tested: _{template.format(top)}_")
    return scores, note


# --- Definicion de la interfaz (layout + componentes) ---
with gr.Blocks(title="Zero-Shot NLI Classifier") as demo:
    gr.Markdown(
        "# Zero-Shot Text Classifier (bart-large-mnli)\n"
        "No fine-tuning: each label is reframed as an entailment hypothesis "
        "(`template.format(label)`) and the model scores how strongly your text "
        "*entails* it. Edit the labels or template and watch the prediction shift "
        "— that instability is the project's main finding.")

    with gr.Row():
        with gr.Column(scale=3):
            text = gr.Textbox(label="Text to classify", lines=4,
                              placeholder="Paste a news headline or paragraph...")
            with gr.Row():
                preset = gr.Dropdown(
                    choices=list(PRESETS), value=PRIMARY_LABEL_SET,
                    label="Load a preset label set", scale=1)
                template = gr.Textbox(
                    value=HYPOTHESIS_TEMPLATE, label="Hypothesis template",
                    scale=2)
            labels = gr.Textbox(
                value=PRESETS[PRIMARY_LABEL_SET],
                label="Candidate labels (comma-separated, fully editable)")
            btn = gr.Button("Classify", variant="primary")
        with gr.Column(scale=2):
            out_scores = gr.Label(label="Entailment scores", num_top_classes=10)
            out_note = gr.Markdown()

    gr.Examples(EXAMPLES, inputs=text)

    # Conexion de eventos -> funciones (esto es lo que hace interactiva la app).
    preset.change(load_preset, inputs=preset, outputs=labels)
    btn.click(predict, inputs=[text, labels, template],
              outputs=[out_scores, out_note])
    text.submit(predict, inputs=[text, labels, template],
                outputs=[out_scores, out_note])


if __name__ == "__main__":
    print("Loading model (first call downloads/caches ~1.6 GB)...")
    get_pipe()  # precarga el modelo para que el primer clic del usuario sea rapido
    demo.launch()
