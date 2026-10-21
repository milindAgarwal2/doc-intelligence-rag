"""Render the article's three Markdown tables as PNG images so they survive a
Medium paste/import (Medium drops HTML tables). Themed to match the diagrams.

Uses explicit text wrapping (textwrap) sized per column so no cell clips, and
sizes each row's height to the tallest wrapped cell.

Run: .venv/bin/python docs/make_tables.py  -> writes docs/images/table-*.png
"""
import os
import textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "images")
os.makedirs(OUT, exist_ok=True)

GREEN = "#2e7d32"
HEAD_BG = "#2e7d32"
HEAD_FG = "#ffffff"
ROW_A = "#ffffff"
ROW_B = "#f1f8e9"
EDGE = "#c8e6c9"


def wrap_cell(text, width):
    """Wrap a cell's text to `width` characters, preserving explicit newlines."""
    out = []
    for para in str(text).split("\n"):
        wrapped = textwrap.wrap(para, width=width) or [""]
        out.extend(wrapped)
    return "\n".join(out)


def render_table(title, headers, rows, col_chars, fname,
                 fontsize=10, header_fontsize=11, title_fontsize=14,
                 char_w=0.095, line_h=0.26, pad=0.18):
    """col_chars: wrap width (in characters) for each column.
    Column pixel widths are proportional to col_chars."""
    ncols = len(headers)
    total_chars = sum(col_chars)

    # wrap every cell; compute how many text lines each row needs
    def wrap_row(cells):
        return [wrap_cell(c, col_chars[i]) for i, c in enumerate(cells)]

    whead = wrap_row(headers)
    wrows = [wrap_row(r) for r in rows]

    def row_lines(wr):
        return max(cell.count("\n") + 1 for cell in wr)

    head_lines = row_lines(whead)
    body_lines = [row_lines(wr) for wr in wrows]

    # figure geometry
    fig_w = char_w * total_chars + 0.6
    title_h = 0.55 if title else 0.0
    total_text_lines = head_lines + sum(body_lines)
    fig_h = title_h + (total_text_lines * line_h) + (len(rows) + 1) * (2 * pad) + 0.3

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.set_xlim(0, total_chars)
    ax.set_ylim(0, fig_h)

    y = fig_h
    if title:
        ax.text(0, y - 0.35, title, ha="left", va="center",
                fontsize=title_fontsize, weight="bold", color=GREEN)
        y -= title_h

    # column x edges
    xedges = [0]
    for c in col_chars:
        xedges.append(xedges[-1] + c)

    def draw_row(wr, nlines, facecolors, fg, bold, fsize):
        nonlocal y
        rh = nlines * line_h + 2 * pad
        ytop = y
        ybot = y - rh
        for ci in range(ncols):
            x0, x1 = xedges[ci], xedges[ci + 1]
            ax.add_patch(plt.Rectangle((x0, ybot), x1 - x0, rh,
                                       facecolor=facecolors, edgecolor=EDGE,
                                       linewidth=1.0, zorder=1))
            ax.text(x0 + 0.3, ytop - pad, wr[ci], ha="left", va="top",
                    fontsize=fsize, color=fg,
                    weight="bold" if bold else "normal", zorder=2,
                    family="DejaVu Sans")
        y = ybot

    # header
    draw_row(whead, head_lines, HEAD_BG, HEAD_FG, True, header_fontsize)
    # body
    for i, wr in enumerate(wrows):
        draw_row(wr, body_lines[i], ROW_A if i % 2 == 0 else ROW_B,
                 "#1a1a1a", False, fontsize)

    f = os.path.join(OUT, fname)
    fig.savefig(f, dpi=150, bbox_inches="tight", facecolor="white", pad_inches=0.12)
    plt.close(fig)
    print("wrote", f)


# ---- Table 1: changelog (3 cols) ----
changelog_headers = ["Area", "2025 (original)", "2026 (now)"]
changelog_rows = [
    ["Embeddings", "sentence-transformers (all-mpnet-base-v2), PyTorch",
     "fastembed ONNX (BAAI/bge-small-en-v1.5) — local, offline, no PyTorch"],
    ["Answer generation", "facebook/bart-large-cnn summarizer per chunk (no synthesis)",
     "Pluggable, single-select: Claude (API) · Qwen 1.5B (local) · None (extractive). One synthesized, streamed answer"],
    ["Formats", "PDF only", "PDF and PowerPoint (PPTX/PPT) — slides map to pages"],
    ["Table searchability", "Table saved, but only prose was embedded",
     "Each table's content is embedded as its own chunk — a query matching cell values finds the page"],
    ["Citations", "Page links appended per chunk",
     "Inline [document p.N] per claim; assets scoped to cited pages only"],
    ["Groundedness", "Summaries, could drift",
     "Grounded refusal (\"I don't know\") + a low-confidence banner"],
    ["Conversation", "Flat history map",
     "Follow-up rewriting (\"what about at 85C?\") + suggested follow-ups"],
    ["Asset serving", "python -m http.server 8000, localhost:8000 links",
     "Images/PDFs/tables inlined as base64/HTML — no companion server"],
    ["Admin", "CLI only", "Web admin console — upload/delete, pick engine, tune everything"],
    ["Re-ingest", "reset_collection() every run",
     "Content-hash upsert — identical bytes skipped, changed bytes replaced, no duplicates"],
    ["Deployment", "Run two terminals by hand", "Docker Compose — 5 services, one command"],
    ["Logging", "print() for debugging", "Structured per-query logs + Postgres Q&A history"],
    ["Security", "—", "Prompt-injection defense (fenced untrusted context), path-traversal-safe serving"],
    ["Quality proof", "—", "Offline evaluation harness (recall@k + confidence calibration), CI-gating"],
]
render_table("What changed since 2025 (the changelog)",
             changelog_headers, changelog_rows,
             col_chars=[20, 40, 48], fname="table-changelog.png",
             fontsize=10, header_fontsize=11)

# ---- Table 2: sidecar store (3 cols) ----
sidecar_headers = ["Asset", "Extracted with (PDF / PPTX)", "Stored as"]
sidecar_rows = [
    ["Text", "pymupdf4llm / python-pptx text frames", "embedding in Chroma"],
    ["Tables", "pymupdf find_tables / python-pptx table shapes",
     "pickled DataFrame  table/<doc>/<page>/tableN.pkl"],
    ["Images", "Spire.PDF ImageHelper / python-pptx pictures",
     "PNG  pdf/<doc>/<page>/imageN.png"],
]
render_table("Page-as-anchor sidecar store",
             sidecar_headers, sidecar_rows,
             col_chars=[12, 44, 44], fname="table-sidecar.png",
             fontsize=11, header_fontsize=12)

# ---- Table 3: provider table (4 cols) ----
provider_headers = ["Provider", "What it does", "Requirements", "Notes"]
provider_rows = [
    ["Claude (Anthropic API)", "Streamed synthesis, reliable inline citations — best quality",
     "ANTHROPIC_API_KEY", "Fast"],
    ["Qwen 1.5B (local SLM)", "Same synthesis, fully offline",
     "llama-cpp-python + a ~1 GB GGUF", "Slow on CPU (~1-2 min/answer); may omit citation markers"],
    ["None (extractive)", "No generation — shows retrieved passages verbatim under [file p.N] headers",
     "nothing", "Instant, zero-cost, no key"],
]
render_table("The pluggable answer engine",
             provider_headers, provider_rows,
             col_chars=[22, 34, 22, 26], fname="table-providers.png",
             fontsize=10.5, header_fontsize=11)

print("done")
