"""Shared application configuration (tracked in git, non-secret).

Config lives in ``config.json`` next to this module so BOTH the upload GUI
(embeddingGUI.py) and the chatbot (chatBOT_engine.py) read a single source of
truth. Secrets (e.g. ANTHROPIC_API_KEY) stay in ``.env`` — this file is only for
non-secret settings such as which LLM provider to use, the low-confidence
threshold, and logging (level + optional file).

The upload GUI writes the config; the chatbot reads it once per session.
"""

import json
import os

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

# Allowed LLM providers for answering.
#   "claude"     -> Anthropic Claude (needs ANTHROPIC_API_KEY)
#   "qwen"       -> local Qwen2.5-1.5B GGUF via llama-cpp-python (needs the .gguf)
#   "none"       -> extractive: show retrieved passages verbatim (no generation)
VALID_PROVIDERS = ("claude", "qwen", "none")

# Confidence signal (wishlist #3). If the single best retrieval relevance score
# for a query falls below this floor, the chatbot shows a NON-BLOCKING "low
# confidence" banner above the answer (the answer still renders). The value is a
# cosine *relevance* score in [0, 1] (higher = more similar). It is model- and
# corpus-dependent; 0.63 is a data-backed default from the evaluation harness
# (tests/eval) — it sits in the clean gap between in-corpus scores (min ~0.65)
# and off-topic scores (max ~0.61) measured on the current store. Re-run the
# harness after changing the embedding model/corpus and retune. Set to 0.0 to
# effectively disable the banner.
LOW_CONFIDENCE_MIN = 0.0
LOW_CONFIDENCE_MAX = 1.0

# Logging (wishlist #6). Non-secret, tunable settings -> they live in config.json
# like every other knob. config.json is the SINGLE source of truth (no env vars),
# so there is no ambiguity about where logging is configured.
VALID_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

DEFAULTS = {
    "llm_provider": "none",
    "low_confidence_threshold": 0.63,
    "log_level": "INFO",
    "log_file": "",            # "" -> console/stderr only; a path -> also append there
    # Embeddable widget (#10): the greeting shown when a visitor opens the chat
    # panel. Read fresh each time a new session opens the widget. "" hides it.
    "widget_welcome_message": "Hi! Ask me a question about the indexed documents.",
    # Widget appearance (all read fresh when the widget loads).
    "widget_title": "📚 Document Assistant",
    "widget_input_placeholder": "Ask about the documents…",
    "widget_accent_color": "#2e7d32",
    # Streamlit chatbot UI (read once per session). Labels + visibility toggles
    # so deployments can rebrand and hide options a given client doesn't need.
    "chatbot_title": "📚 Technical Document Assistant",
    "chatbot_subtitle": (
        "Ask questions about your indexed PDFs. Answers are synthesized from "
        "the documents with inline source citations."
    ),
    "chatbot_input_placeholder": "Ask a question about your documents…",
    "chatbot_answer_options_label": "Answer options",
    "chatbot_include_figures_label": "Include figures from cited pages",
    "chatbot_include_tables_label": "Include tables from cited pages",
    "chatbot_conversation_label": "Conversation",
    "chatbot_clear_chat_label": "🗑️ Clear chat",
    # Visibility toggles: hide the figure/table options entirely for clients
    # whose documents have no images/tables. When hidden the feature is OFF.
    "chatbot_show_figures_option": True,
    "chatbot_show_tables_option": True,
}


def _clamp_threshold(value, fallback=0.63):
    """Coerce ``value`` to a float in [LOW_CONFIDENCE_MIN, LOW_CONFIDENCE_MAX].

    A missing/non-numeric value falls back to ``fallback`` so a hand-edited
    config.json can never crash the app or produce a nonsensical cutoff.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return fallback
    if v < LOW_CONFIDENCE_MIN:
        return LOW_CONFIDENCE_MIN
    if v > LOW_CONFIDENCE_MAX:
        return LOW_CONFIDENCE_MAX
    return v


def _normalize_log_level(value, fallback="INFO"):
    """Coerce ``value`` to a valid log-level name (uppercased). Bad values fall
    back so a hand-edited config.json can't break logging setup."""
    if not isinstance(value, str):
        return fallback
    v = value.strip().upper()
    return v if v in VALID_LOG_LEVELS else fallback


import re as _re

# A permissive but safe CSS color check: #hex (3/4/6/8), rgb()/rgba(), hsl()/hsla(),
# or a plain CSS named color (letters only). Prevents injecting arbitrary content
# via the accent color (it is used in a style attribute by the widget).
_COLOR_RE = _re.compile(
    r"^(#[0-9a-fA-F]{3,8}"
    r"|rgba?\([0-9.,%\s]+\)"
    r"|hsla?\([0-9.,%\s]+\)"
    r"|[a-zA-Z]+)$"
)


def _normalize_widget_appearance(cfg):
    """Validate the widget appearance keys in-place: non-empty strings for title
    and placeholder (else default), and a syntactically valid CSS color for the
    accent (else default). Keeps a hand-edited config from breaking the widget."""
    for key in ("widget_title", "widget_input_placeholder"):
        v = cfg.get(key)
        if not isinstance(v, str) or not v.strip():
            cfg[key] = DEFAULTS[key]
    color = cfg.get("widget_accent_color")
    if not isinstance(color, str) or not _COLOR_RE.match(color.strip()):
        cfg["widget_accent_color"] = DEFAULTS["widget_accent_color"]
    else:
        cfg["widget_accent_color"] = color.strip()


# Chatbot string keys: non-empty strings expected, else fall back to default.
_CHATBOT_STR_KEYS = (
    "chatbot_title", "chatbot_subtitle", "chatbot_input_placeholder",
    "chatbot_answer_options_label", "chatbot_include_figures_label",
    "chatbot_include_tables_label", "chatbot_conversation_label",
    "chatbot_clear_chat_label",
)
_CHATBOT_BOOL_KEYS = ("chatbot_show_figures_option", "chatbot_show_tables_option")


def _normalize_chatbot(cfg):
    """Validate chatbot UI keys in-place. Strings: non-empty else default (except
    subtitle, which MAY be empty to hide it). Booleans: coerced to real bools."""
    for key in _CHATBOT_STR_KEYS:
        v = cfg.get(key)
        if not isinstance(v, str):
            cfg[key] = DEFAULTS[key]
        elif not v.strip() and key != "chatbot_subtitle":
            # subtitle is allowed to be "" (hidden); others fall back if blank.
            cfg[key] = DEFAULTS[key]
    for key in _CHATBOT_BOOL_KEYS:
        cfg[key] = bool(cfg.get(key, DEFAULTS[key]))


def load_config():
    """Return the config dict, merged over defaults. Never raises: a missing or
    malformed file yields the defaults so the app always runs.
    """
    cfg = dict(DEFAULTS)
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            cfg.update(data)
    except (OSError, ValueError):
        pass
    # Normalize the provider to a known value.
    if cfg.get("llm_provider") not in VALID_PROVIDERS:
        cfg["llm_provider"] = DEFAULTS["llm_provider"]
    # Clamp the confidence threshold into [0, 1].
    cfg["low_confidence_threshold"] = _clamp_threshold(
        cfg.get("low_confidence_threshold"), DEFAULTS["low_confidence_threshold"]
    )
    # Normalize logging settings.
    cfg["log_level"] = _normalize_log_level(
        cfg.get("log_level"), DEFAULTS["log_level"]
    )
    if not isinstance(cfg.get("log_file"), str):
        cfg["log_file"] = DEFAULTS["log_file"]
    if not isinstance(cfg.get("widget_welcome_message"), str):
        cfg["widget_welcome_message"] = DEFAULTS["widget_welcome_message"]
    _normalize_widget_appearance(cfg)
    _normalize_chatbot(cfg)
    return cfg


def save_config(updates):
    """Merge ``updates`` into the current config and write config.json.

    Only known keys are persisted. Returns the written config dict.
    """
    cfg = load_config()
    for k, v in (updates or {}).items():
        cfg[k] = v
    if cfg.get("llm_provider") not in VALID_PROVIDERS:
        cfg["llm_provider"] = DEFAULTS["llm_provider"]
    cfg["low_confidence_threshold"] = _clamp_threshold(
        cfg.get("low_confidence_threshold"), DEFAULTS["low_confidence_threshold"]
    )
    cfg["log_level"] = _normalize_log_level(
        cfg.get("log_level"), DEFAULTS["log_level"]
    )
    if not isinstance(cfg.get("log_file"), str):
        cfg["log_file"] = DEFAULTS["log_file"]
    if not isinstance(cfg.get("widget_welcome_message"), str):
        cfg["widget_welcome_message"] = DEFAULTS["widget_welcome_message"]
    _normalize_widget_appearance(cfg)
    _normalize_chatbot(cfg)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")
    return cfg


def get_llm_provider():
    """Convenience: return the configured provider string."""
    return load_config()["llm_provider"]


def get_low_confidence_threshold():
    """Convenience: return the configured low-confidence threshold (float)."""
    return load_config()["low_confidence_threshold"]


def get_log_level():
    """Return the configured log level name (from config.json)."""
    return load_config()["log_level"]


def get_log_file():
    """Return the configured log-file path from config.json ("" = console only)."""
    return load_config()["log_file"]


def get_widget_welcome_message():
    """Return the widget greeting from config.json ("" = no greeting shown)."""
    return load_config()["widget_welcome_message"]


def get_widget_config():
    """Return all widget appearance/content settings as a dict (for the API)."""
    cfg = load_config()
    return {
        "welcome_message": cfg["widget_welcome_message"],
        "title": cfg["widget_title"],
        "input_placeholder": cfg["widget_input_placeholder"],
        "accent_color": cfg["widget_accent_color"],
    }


def get_chatbot_config():
    """Return the Streamlit chatbot UI settings (labels + visibility toggles)."""
    cfg = load_config()
    return {k: cfg[k] for k in (_CHATBOT_STR_KEYS + _CHATBOT_BOOL_KEYS)}
