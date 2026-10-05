"""stt-vocab: give the speech-to-text model the names it would otherwise mis-hear.

Registers one ``pre_transcription`` hook. Before Hermes sends a voice message to its STT
backend, the hook appends a short list of terms to the backend's prompt (Whisper's
``initial_prompt`` / the OpenAI-compatible ``prompt`` field), which biases the transcript toward
those spellings. The list is built from, most important first:

1. ``source_terms[<source>]``: terms for one caller surface (``gateway``, ``voice_mode``, ...);
2. ``terms``: terms for every transcription;
3. names found in this profile's memory files (``memories/USER.md`` then ``memories/MEMORY.md``),
   when ``from_memory`` is on. Memory-derived terms go only to a local backend (``local``
   faster-whisper or ``local_command``) unless ``allow_remote`` is set, because the prompt is
   uploaded with the audio.

Terms already in ``stt.prompt`` are skipped, and the result stays within ``max_chars`` (896 by
default, the length Hermes keeps for Whisper-family backends), so Hermes never has to truncate
it. Settings and memory files are read on every transcription. A setting the plugin cannot use
is logged once and the request goes out unchanged. No tools, network, subprocesses or writes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from .terms import build_prompt, extract_terms

logger = logging.getLogger(__name__)

HOOK = "pre_transcription"
LOCAL_PROVIDERS = frozenset({"local", "local_command"})
MEMORY_FILES = ("USER.md", "MEMORY.md")
ENTRY_DELIMITER = "\n§\n"
DEFAULT_MAX_CHARS = 896
MAX_CHARS_CEILING = 4000
MAX_MEMORY_BYTES = 256 * 1024

_warned: set = set()


class _Unusable(ValueError):
    pass


def _warn_once(key: str, message: str, *args: Any) -> None:
    if key not in _warned:
        _warned.add(key)
        logger.warning("stt-vocab: " + message, *args)


def _terms_list(value: Any, name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise _Unusable(f"{name} must be a list of strings, got {value!r}")
    return [v.strip() for v in value if v.strip()]


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise _Unusable(f"{name} must be true or false, got {value!r}")
    return value


def _settings(ctx) -> dict:
    max_chars = ctx.get_config("max_chars", DEFAULT_MAX_CHARS)
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or not 1 <= max_chars <= MAX_CHARS_CEILING:
        raise _Unusable(f"max_chars must be a whole number from 1 to {MAX_CHARS_CEILING}, got {max_chars!r}")
    source_terms = ctx.get_config("source_terms", {})
    if source_terms is None:
        source_terms = {}
    if not isinstance(source_terms, dict):
        raise _Unusable(f"source_terms must map a source name to a list of terms, got {source_terms!r}")
    return {
        "terms": _terms_list(ctx.get_config("terms", []), "terms"),
        "source_terms": {str(k): _terms_list(v, f"source_terms.{k}") for k, v in source_terms.items()},
        "from_memory": _bool(ctx.get_config("from_memory", True), "from_memory"),
        "allow_remote": _bool(ctx.get_config("allow_remote", False), "allow_remote"),
        "max_chars": max_chars,
    }


def _memory_entries() -> list[str]:
    # Resolved per call, so a gateway serving several profiles reads the active profile's memory.
    from hermes_constants import get_hermes_home
    entries: list[str] = []
    for name in MEMORY_FILES:
        path = Path(get_hermes_home()) / "memories" / name
        try:
            with open(path, "rb") as fh:
                raw = fh.read(MAX_MEMORY_BYTES + 1)
        except OSError:
            continue
        if len(raw) > MAX_MEMORY_BYTES:
            _warn_once(f"size:{path}", "%s is larger than %d bytes; reading only the start", path, MAX_MEMORY_BYTES)
            raw = raw[:MAX_MEMORY_BYTES]
        text = raw.decode("utf-8-sig", errors="replace")
        entries.extend(e for e in text.split(ENTRY_DELIMITER) if e.strip())
    return entries


def make_hook(ctx):
    def _on_transcription(provider: str = "", prompt: Optional[str] = None,
                          source: Optional[str] = None, **_kwargs: Any) -> Optional[dict]:
        try:
            settings = _settings(ctx)
        except _Unusable as exc:
            _warn_once(str(exc), "%s; leaving the transcription prompt unchanged", exc)
            return None
        groups = [settings["source_terms"].get(source or "", []), settings["terms"]]
        if settings["from_memory"] and (settings["allow_remote"] or provider in LOCAL_PROVIDERS):
            groups.append(extract_terms(_memory_entries()))
        new_prompt = build_prompt(prompt, groups, settings["max_chars"])
        return None if new_prompt is None else {"prompt": new_prompt}

    return _on_transcription


def register(ctx) -> None:
    ctx.register_hook(HOOK, make_hook(ctx))
