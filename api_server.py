"""HTTP API for the embeddable chat widget.

A thin FastAPI layer over the SAME retrieval + answer logic the Streamlit app
uses (chatBOT_engine.generate_answer) — no Streamlit involved. This is what the
JavaScript widget (widget/) and the demo site (demo-site/) talk to.

Endpoints:
  GET  /api/health          -> {"status": "ok", "provider": "..."}
  POST /api/chat            -> {question, history} => {answer, sources, ...}
  GET  /widget/widget.js    -> the embeddable widget script (static)
  GET  /widget/widget.css   -> widget styles (static)

Run:
  .venv/bin/uvicorn api_server:app --host 0.0.0.0 --port 8000

Notes:
  - CORS is OPEN (allow_origins=["*"]) so the widget works when embedded on a
    different origin. This is fine for a DEMO. For real use, restrict origins
    and add auth (wishlist #2) — an open endpoint spends your Claude budget.
  - Non-streaming: returns the full answer in one JSON response (simplest first
    version). Streaming can be added later via a /api/chat/stream SSE endpoint.
"""
import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi import HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from logging_config import setup_logging
from config import get_llm_provider
import chatBOT_engine as engine

setup_logging()
logger = logging.getLogger("api_server")

app = FastAPI(title="Technical-Document RAG — Widget API", version="1.0")

# Open CORS for the demo so the widget works cross-origin. Lock this down for
# production (specific origins) + add auth.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_WIDGET_DIR = os.path.join(_HERE, "widget")


class ChatTurn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    history: list[ChatTurn] = []


class ChatResponse(BaseModel):
    answer: str
    sources: list[str] = []
    low_confidence: bool = False
    best_score: float | None = None
    refusal: bool = False
    followups: list[str] = []
    provider: str = "none"
    latency_ms: int = 0


def _effective_provider():
    """Resolve the configured provider with the same auto-fallback the chatbot
    uses (missing key/model -> extractive), so the API never hard-fails."""
    requested = get_llm_provider()
    try:
        return engine.resolve_provider(requested)
    except Exception:
        return "none"


@app.get("/api/health")
def health():
    return {"status": "ok", "provider": _effective_provider()}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Answer one question against the indexed documents (non-streaming)."""
    question = (req.question or "").strip()
    if not question:
        return ChatResponse(answer="Please enter a question.", provider="none")

    provider = _effective_provider()
    history = [{"role": t.role, "content": t.content} for t in req.history]

    try:
        result = engine.generate_answer(question, provider, history=history)
    except Exception as ex:
        logger.exception("chat failed")
        return ChatResponse(
            answer=f"Sorry, something went wrong answering that ({type(ex).__name__}).",
            provider=provider,
        )

    # Return basenames for sources (the widget shows document names, not paths).
    sources = [os.path.basename(s) for s in result.get("sources", [])]

    return ChatResponse(
        answer=result["answer"],
        sources=sources,
        low_confidence=result["low_confidence"],
        best_score=result["best_score"],
        refusal=result["refusal"],
        followups=result["followups"],
        provider=result["provider"],
        latency_ms=result["latency_ms"],
    )


# Serve the widget's static assets (so a site can <script src=".../widget/widget.js">).
if os.path.isdir(_WIDGET_DIR):
    app.mount("/widget", StaticFiles(directory=_WIDGET_DIR), name="widget")


# Folders that may hold source documents to serve back.
_SOURCE_DIRS = ("pdf", "pptx")
_ALLOWED_SOURCE_EXT = {".pdf", ".pptx", ".ppt"}


def _resolve_source(filename):
    """Resolve a source document name to an on-disk path, searching the known
    source folders. Hardened against path traversal: only the basename is used,
    and only whitelisted extensions are served."""
    base = os.path.basename(filename or "")      # strip any directory component
    if not base:
        return None
    ext = os.path.splitext(base)[1].lower()
    if ext not in _ALLOWED_SOURCE_EXT:
        return None
    for d in _SOURCE_DIRS:
        candidate = os.path.join(_HERE, d, base)
        # Ensure the resolved path stays inside the intended folder.
        root = os.path.realpath(os.path.join(_HERE, d))
        real = os.path.realpath(candidate)
        if real.startswith(root + os.sep) and os.path.isfile(real):
            return real
    return None


@app.get("/api/source/{filename}")
def get_source(filename: str):
    """Serve a cited source document by name (so the widget can link to it)."""
    path = _resolve_source(filename)
    if not path:
        raise HTTPException(status_code=404, detail="source not found")
    return FileResponse(path, filename=os.path.basename(path))


@app.get("/")
def root():
    return {"service": "techdoc-rag-widget-api",
            "endpoints": ["/api/health", "/api/chat", "/widget/widget.js"]}
