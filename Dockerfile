# ---------------------------------------------------------------------------
# Technical-Document RAG Chatbot — single image for the WHOLE application.
# The same image runs BOTH services (selected by the compose `command`):
#   • chatBOT_engine.py  -> the chat UI        (port 8501)
#   • embeddingGUI.py    -> the Embedding Mgr  (port 8502)
# Runtime data (vector store, extracted assets, uploads, config, model) is
# provided via mounted volumes, so it is shared between the two containers and
# survives restarts — nothing stateful is baked into the image.
# ---------------------------------------------------------------------------
FROM python:3.12-slim

# System libraries needed by the Python deps:
#   build-essential            -> compile llama-cpp-python (local Qwen engine)
#   libgl1 / libglib2.0-0      -> PyMuPDF / Pillow image handling
#   libgomp1                   -> OpenMP runtime (fastembed ONNX, llama.cpp)
#   curl                       -> container HEALTHCHECK
# Installed in one layer, then apt caches removed to keep the image small.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer cached unless requirements.txt changes).
COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the application source. Runtime data dirs are NOT copied — they are
# mounted as volumes by docker-compose (see .dockerignore).
COPY *.py ./
COPY tests ./tests
COPY widget ./widget
COPY demo-site ./demo-site

# Pre-create the mount points so first run is clean even before volumes exist.
RUN mkdir -p pdf pptx table chroma_Name models

# Streamlit server defaults for container use: listen on all interfaces, no
# usage stats, no CORS/XSRF friction behind a proxy.
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    PYTHONUNBUFFERED=1

# Chatbot is the default service; compose overrides `command` for the GUI.
EXPOSE 8501 8502
CMD ["streamlit", "run", "chatBOT_engine.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
