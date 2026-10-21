"""Generate article diagrams (architecture + RAG comparison graphics) as PNGs.

Run: .venv/bin/python docs/make_diagrams.py
Outputs into docs/images/.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

OUT = os.path.join(os.path.dirname(__file__), "images")
os.makedirs(OUT, exist_ok=True)

# Palette
GREEN = "#2e7d32"
GREEN_L = "#e8f5e9"
BLUE = "#1565c0"
BLUE_L = "#e3f2fd"
AMBER = "#ef6c00"
AMBER_L = "#fff3e0"
GREY = "#546e7a"
GREY_L = "#eceff1"
RED = "#c62828"
RED_L = "#ffebee"


def box(ax, x, y, w, h, text, face, edge, fontsize=10, bold=False, tcolor="#1a1a1a"):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                       linewidth=1.6, edgecolor=edge, facecolor=face, zorder=2)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=tcolor, zorder=3,
            weight="bold" if bold else "normal", wrap=True)


def arrow(ax, x1, y1, x2, y2, color=GREY, style="-|>", lw=1.8):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=16,
                        linewidth=lw, color=color, zorder=1,
                        connectionstyle="arc3,rad=0")
    ax.add_patch(a)


# ---------------------------------------------------------------------------
# 1) ARCHITECTURE DIAGRAM
# ---------------------------------------------------------------------------
def architecture():
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.text(6.5, 7.7, "Multi-Modal RAG Chatbot — Architecture",
            ha="center", fontsize=16, weight="bold", color=GREEN)

    # ---- Part 1: Ingestion ----
    ax.text(2.1, 7.0, "PART 1 · INGESTION  (embeddingEngine.py)",
            ha="center", fontsize=10, weight="bold", color=AMBER)
    box(ax, 0.4, 6.1, 3.4, 0.6, "PDF  +  PowerPoint (PPTX)\nslides map to page numbers",
        AMBER_L, AMBER, 9)
    box(ax, 0.4, 5.3, 3.4, 0.55, "ingest_file()  dispatcher", "#ffffff", AMBER, 9, bold=True)

    box(ax, 0.3, 4.3, 1.6, 0.7, "Text\n(pymupdf4llm /\npython-pptx)", BLUE_L, BLUE, 7.5)
    box(ax, 2.0, 4.3, 1.6, 0.7, "Tables\n(pymupdf /\npptx shapes)", GREEN_L, GREEN, 7.5)
    box(ax, 0.3, 3.4, 1.6, 0.7, "Images\n(Spire.PDF /\npptx pics)", "#f3e5f5", "#7b1fa2", 7.5)
    box(ax, 2.0, 3.4, 1.6, 0.7, "SHA-256\ndoc_hash\n(upsert)", GREY_L, GREY, 7.5)

    arrow(ax, 2.1, 6.1, 2.1, 5.87)
    arrow(ax, 2.1, 5.3, 2.1, 5.05)
    arrow(ax, 1.4, 5.3, 1.1, 5.02)
    arrow(ax, 2.8, 5.3, 2.8, 5.02)

    # ---- Center: stores ----
    ax.text(6.5, 7.0, "STORAGE", ha="center", fontsize=10, weight="bold", color=GREY)
    box(ax, 5.0, 4.9, 3.0, 1.0,
        "Chroma vector store\nMMR  k=8 / fetch_k=24\n+ table content embedded\nas its OWN chunk",
        GREEN_L, GREEN, 8.5, bold=False)
    box(ax, 5.0, 3.4, 3.0, 1.0,
        "Sidecar asset store (disk)\ntable/<doc>/<page>/*.pkl\npdf/<doc>/<page>/*.png\nkey = file_name + page",
        "#f3e5f5", "#7b1fa2", 8.5)

    arrow(ax, 3.8, 4.65, 5.0, 5.3, color=BLUE)      # text -> vector
    arrow(ax, 3.8, 4.5, 5.0, 3.9, color="#7b1fa2")  # tables/images -> disk

    # ---- Part 2 + 3: query side ----
    ax.text(10.8, 7.0, "PART 2 · CHATBOT   +   PART 3 · WIDGET",
            ha="center", fontsize=10, weight="bold", color=BLUE)
    box(ax, 9.2, 6.0, 3.4, 0.7, "User question\n(follow-up → rewrite_query)", BLUE_L, BLUE, 8.5)
    box(ax, 9.2, 5.0, 3.4, 0.7, "MMR retrieval\n+ low-confidence probe", "#ffffff", BLUE, 8.5, bold=True)
    box(ax, 9.2, 3.9, 3.4, 0.8,
        "Answer engine (pluggable)\nClaude · Qwen (local) · None\nfenced untrusted context",
        GREEN_L, GREEN, 8)
    box(ax, 9.2, 2.7, 3.4, 0.9,
        "Grounded streamed answer\n[doc p.N] citations +\ninline figures/tables (cited pages)",
        AMBER_L, AMBER, 8)

    arrow(ax, 10.9, 6.0, 10.9, 5.72)
    arrow(ax, 10.9, 5.0, 10.9, 4.72)
    arrow(ax, 10.9, 3.9, 10.9, 3.62)

    # store -> retrieval
    arrow(ax, 8.0, 5.4, 9.2, 5.4, color=GREEN)
    # disk assets -> answer render
    arrow(ax, 8.0, 3.7, 9.2, 3.1, color="#7b1fa2")

    # two front ends
    box(ax, 9.2, 1.5, 1.6, 0.7, "Streamlit\napp :8501", BLUE_L, BLUE, 8)
    box(ax, 11.0, 1.5, 1.6, 0.7, "JS widget\nvia API :8000", BLUE_L, BLUE, 8)
    arrow(ax, 10.4, 2.7, 10.0, 2.2)
    arrow(ax, 11.4, 2.7, 11.8, 2.2)
    ax.text(10.9, 1.15, "shared generate_answer() core", ha="center",
            fontsize=7.5, style="italic", color=GREY)

    # offline note
    ax.text(6.5, 0.5, "Runs fully self-hosted · can run 100% offline (local ONNX embeddings + Qwen/extractive)",
            ha="center", fontsize=9, style="italic", color=GREEN)

    plt.tight_layout()
    f = os.path.join(OUT, "architecture.png")
    fig.savefig(f, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", f)


# ---------------------------------------------------------------------------
# 2) PLAIN RAG vs THIS APPROACH
# ---------------------------------------------------------------------------
def plain_vs_multimodal():
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.5)
    ax.axis("off")
    ax.text(6, 6.15, "Plain RAG vs. this multi-modal approach",
            ha="center", fontsize=15, weight="bold", color=GREEN)

    # Left: plain RAG
    ax.text(3, 5.55, "Plain RAG", ha="center", fontsize=12, weight="bold", color=RED)
    box(ax, 0.4, 4.6, 5.2, 0.7, "Technical document\n(text + tables + images)", GREY_L, GREY, 9)
    box(ax, 0.4, 3.6, 5.2, 0.6, "Embed TEXT chunks only", BLUE_L, BLUE, 9)
    box(ax, 0.4, 2.55, 5.2, 0.7,
        "✗ tables flattened / lost\n✗ images discarded", RED_L, RED, 9)
    box(ax, 0.4, 1.5, 5.2, 0.7,
        "Query → text only\n(often unhelpful, low trust)", RED_L, RED, 9.5, bold=True)
    arrow(ax, 3, 4.6, 3, 4.22, color=GREY)
    arrow(ax, 3, 3.6, 3, 3.27, color=RED)
    arrow(ax, 3, 2.55, 3, 2.22, color=RED)

    # Right: this approach
    ax.text(9, 5.55, "This approach", ha="center", fontsize=12, weight="bold", color=GREEN)
    box(ax, 6.4, 4.6, 5.2, 0.7, "Technical document\n(text + tables + images)", GREY_L, GREY, 9)
    box(ax, 6.4, 3.6, 5.2, 0.6,
        "Embed text  +  embed each TABLE as its own chunk", GREEN_L, GREEN, 8.5)
    box(ax, 6.4, 2.55, 5.2, 0.7,
        "✓ tables saved as DataFrames\n✓ images saved to disk (keyed by page)", GREEN_L, GREEN, 9)
    box(ax, 6.4, 1.5, 5.2, 0.7,
        "Query → grounded answer + inline\ntable/figure + [doc p.N] citation", GREEN_L, GREEN, 9.5, bold=True)
    arrow(ax, 9, 4.6, 9, 4.22, color=GREY)
    arrow(ax, 9, 3.6, 9, 3.27, color=GREEN)
    arrow(ax, 9, 2.55, 9, 2.22, color=GREEN)

    # divider
    ax.add_line(Line2D([6, 6], [1.3, 5.3], color="#cccccc", linestyle="--", linewidth=1))
    ax.text(6, 0.9, "Same source document → very different answer quality",
            ha="center", fontsize=10, style="italic", color=GREY)

    plt.tight_layout()
    f = os.path.join(OUT, "plain-vs-multimodal.png")
    fig.savefig(f, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", f)


# ---------------------------------------------------------------------------
# 3) FEATURE COMPARISON vs OTHER RAG CATEGORIES
# ---------------------------------------------------------------------------
def comparison_matrix():
    cols = ["This\nproject", "Hosted\nSaaS\ndoc-chat", "DIY\nframework\n(LangChain)",
            "Enterprise\nRAG\nplatform", "LLM chat\n+ file\nupload"]
    rows = [
        "Tables & images survive ingestion",
        "Table content semantically searchable",
        "Per-claim page citations",
        "Grounded refusal (no hallucination)",
        "Runs 100% offline / self-hosted",
        "Pluggable local LLM (no API cost)",
        "Finished product (not a toolkit)",
        "Embeddable widget (1 line)",
        "Built-in eval harness (recall@k)",
        "Enterprise auth / SSO / RBAC",
    ]
    # 2 = full, 1 = partial, 0 = no
    data = [
        [2, 0, 1, 1, 0],   # tables/images survive
        [2, 0, 1, 1, 0],   # table searchable
        [2, 1, 1, 2, 1],   # citations
        [2, 1, 1, 2, 1],   # grounded refusal
        [2, 0, 2, 1, 0],   # offline/self-hosted
        [2, 0, 2, 1, 0],   # local LLM
        [2, 2, 0, 2, 2],   # finished product
        [2, 1, 1, 1, 0],   # widget
        [2, 0, 1, 1, 0],   # eval harness
        [0, 2, 1, 2, 2],   # enterprise auth (this project = roadmap)
    ]
    sym = {2: ("●", GREEN), 1: ("◐", AMBER), 0: ("○", "#bbbbbb")}

    fig, ax = plt.subplots(figsize=(12, 7.2))
    ncol = len(cols)
    nrow = len(rows)
    ax.set_xlim(0, ncol + 4)
    ax.set_ylim(-0.9, nrow + 1.6)
    ax.axis("off")
    ax.text((ncol + 4) / 2, nrow + 1.25,
            "Capability comparison across RAG categories",
            ha="center", fontsize=15, weight="bold", color=GREEN)

    # column headers
    for j, c in enumerate(cols):
        cx = 4 + j + 0.5
        face = GREEN_L if j == 0 else "#ffffff"
        box(ax, 4 + j + 0.08, nrow + 0.1, 0.84, 0.95, c, face,
            GREEN if j == 0 else GREY, 7.5, bold=(j == 0))

    # rows
    for i, r in enumerate(rows):
        y = nrow - 1 - i + 0.1
        ax.text(0.05, y + 0.4, r, ha="left", va="center", fontsize=9, color="#1a1a1a")
        for j in range(ncol):
            cx = 4 + j + 0.5
            s, color = sym[data[i][j]]
            ax.text(cx, y + 0.4, s, ha="center", va="center", fontsize=16, color=color)
        ax.add_line(Line2D([0, ncol + 4], [y, y], color="#eeeeee", linewidth=0.8))

    # legend
    ax.text(0.05, -0.55, "●  full      ◐  partial / configurable      ○  not (yet)",
            ha="left", fontsize=9, color=GREY)
    ax.text(ncol + 3.9, -0.55,
            "General category characteristics — verify specifics per vendor",
            ha="right", fontsize=7.5, style="italic", color="#999999")

    plt.tight_layout()
    f = os.path.join(OUT, "rag-comparison.png")
    fig.savefig(f, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", f)


if __name__ == "__main__":
    architecture()
    plain_vs_multimodal()
    comparison_matrix()
    print("done")
