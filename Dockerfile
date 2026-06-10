# Zero-shot NLI classifier — CPU-only image.
# Default command launches the Gradio demo on port 7860. The same image can also
# run the pipeline scripts (see README "Docker" section).
#
# Python 3.12 (slim) is used here for broad wheel support; the project's pinned
# versions all ship cp312 wheels. (Locally the project runs on 3.14.)
FROM python:3.12-slim

# Faster, quieter Python; let Gradio bind to all interfaces so the port is
# reachable from outside the container (Gradio reads these env vars natively).
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1

WORKDIR /app

# 1) Install the CPU-only build of torch FIRST, from the PyTorch CPU index.
#    This avoids pip pulling the default CUDA build (~2 GB of GPU libs we never
#    use on CPU). The matching version is also pinned in requirements.txt.
RUN pip install --no-cache-dir torch==2.12.0 \
        --index-url https://download.pytorch.org/whl/cpu

# 2) Install the rest. torch is already satisfied, so it is not re-downloaded.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3) App code. Generated data/results/figures are excluded via .dockerignore and
#    are recreated inside the container if you run the pipeline scripts.
COPY src/ ./src/

WORKDIR /app/src
EXPOSE 7860

# The ~1.6 GB model is downloaded on first run and cached under
# /root/.cache/huggingface — mount a volume there (see docker-compose.yml) so it
# only downloads once. To bake it into the image instead (offline/instant start),
# add after step 2:
#   RUN python -c "from transformers import pipeline; \
#       pipeline('zero-shot-classification', model='facebook/bart-large-mnli')"
CMD ["python", "app.py"]
