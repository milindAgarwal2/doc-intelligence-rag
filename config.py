"""Shared application configuration (tracked in git, non-secret).

Config lives in ``config.json`` next to this module so BOTH the upload GUI
(embeddingGUI.py) and the chatbot (chatBOT_engine.py) read a single source of
truth. Secrets (e.g. ANTHROPIC_API_KEY) stay in ``.env`` — this file is only for
non-secret settings such as which LLM provider to use.

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

DEFAULTS = {
    "llm_provider": "none",
    "low_confidence_threshold": 0.63,
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
