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

DEFAULTS = {
    "llm_provider": "none",
}


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
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")
    return cfg


def get_llm_provider():
    """Convenience: return the configured provider string."""
    return load_config()["llm_provider"]
