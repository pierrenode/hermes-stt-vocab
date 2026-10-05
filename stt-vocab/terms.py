"""Pick name-like terms out of free text and pack them into a speech-to-text prompt.

``extract_terms(entries)`` scans memory entries for the words a speech model is most likely to
mis-hear: names of people, products and projects, acronyms and identifiers. It keeps

- words with a capital letter after the first character (``gRPC``, ``PostgreSQL``, ``AWS``),
- words that mix letters and digits (``S3``, ``GPT-5``),
- capitalised words in the middle of a sentence (``Northwind``),
- runs of the above as one multi-word name (``Ayşe Demir``, ``Project Atlas``).

A lone capitalised word at the start of a sentence (``Prefers``, ``Uses``) is kept only when the
same word also appears capitalised mid-sentence somewhere. URLs, paths, e-mail addresses and
long strings that look like keys are never kept.

``build_prompt(base, groups, max_chars)`` appends terms to the existing prompt until
``max_chars`` is reached. ``groups`` runs from most to least important; the most important terms
are written last, because Whisper-family models condition on the end of the prompt.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, Optional, Sequence

_TOKEN = re.compile(r"[^\s,;:()\[\]{}\"“”«»!?<>|=]+")
_EDGE = ".'’`*_~-"
_SENTENCE_END = (".", "!", "?", "…")
_LEADING_STOPWORDS = frozenset({
    "a", "an", "the", "this", "that", "these", "those", "my", "our", "your", "his", "her",
    "their", "its", "i", "we", "he", "she", "they", "it", "user", "and", "or", "but", "also"})
# Capitalised only because they open a sentence in the notes Hermes keeps about a user.
_COMMON_STARTERS = frozenset({
    "uses", "used", "likes", "liked", "prefers", "preferred", "speaks", "hates", "works", "worked",
    "lives", "lived", "has", "had", "is", "was", "are", "were", "wants", "needs", "loves", "enjoys",
    "runs", "deploys", "keeps", "avoids", "asked", "said", "always", "never", "often", "usually",
    "sometimes", "currently", "recently", "prefer", "use", "dislikes", "knows", "studies",
    "plans", "owns", "reads", "writes", "builds", "maintains", "manages", "leads", "joined", "moved",
    "then", "later", "now", "today", "yesterday", "tomorrow", "next", "after", "before", "when",
    "while", "if", "since", "because", "so", "still", "just", "only", "both", "each", "every",
    "some", "many", "most", "all", "no", "not", "here", "there", "please", "thanks", "meet",
    "met", "ask", "call", "called", "talked", "told", "with", "for", "from", "at", "on", "in"})
_CALENDAR = frozenset({
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "january",
    "february", "march", "april", "may", "june", "july", "august", "september", "october",
    "november", "december"})
_POSSESSIVE = ("'s", "’s")
_INNER_DOT = re.compile(r"\w\.\w")
MAX_RUN = 4
MAX_TERM_CHARS = 40


def _looks_like_secret_or_locator(word: str) -> bool:
    if "://" in word or "@" in word or "/" in word or "\\" in word:
        return True
    digits = sum(c.isdigit() for c in word)
    return len(word) >= 16 and digits >= 3


def _shape(word: str) -> Optional[str]:
    """``strong`` for a word that reads as a name anywhere, ``cap`` for a plain capitalised word."""
    if len(word) < 2 or len(word) > MAX_TERM_CHARS or word.isdigit() or not any(c.isalpha() for c in word):
        return None
    if _looks_like_secret_or_locator(word):
        return None
    if any(c.isupper() for c in word[1:]) or (any(c.isdigit() for c in word) and any(c.isalpha() for c in word)):
        return "strong"
    if word[0].isupper():
        return "cap"
    return None


def _tokens(entry: str):
    """``(word, sentence_start, breaks_before)`` for each token, with edge punctuation stripped."""
    out = []
    sentence_start, prev_end, prev_raw = True, 0, ""
    for match in _TOKEN.finditer(entry):
        raw, gap = match.group(0), entry[prev_end:match.start()]
        prev_end = match.end()
        if "\n" in gap:
            sentence_start = True
        elif prev_raw.endswith(_SENTENCE_END) and not _INNER_DOT.search(prev_raw.rstrip(".")):
            sentence_start = True
        word = raw.strip(_EDGE)
        if word.endswith(_POSSESSIVE):
            word = word[:-2]
        breaks_before = bool(gap.strip()) or "\n" in gap or prev_raw.endswith(_SENTENCE_END + _POSSESSIVE)
        if word:
            out.append((word, sentence_start, breaks_before))
            sentence_start = False
        prev_raw = raw
    return out


def _runs(entry: str):
    """Yield ``(words, shapes, starts_sentence)`` for each run of name-like words in ``entry``."""
    run, shapes, initial = [], [], False
    for word, sentence_start, breaks_before in _tokens(entry):
        shape = _shape(word)
        if run and (breaks_before or not shape):
            yield run, shapes, initial
            run, shapes = [], []
        if shape:
            if not run:
                initial = sentence_start
            run.append(word)
            shapes.append(shape)
    if run:
        yield run, shapes, initial


def extract_terms(entries: Iterable[str]) -> list[str]:
    """Name-like terms from ``entries``, most frequent first, ties in order of first appearance."""
    entries = [e for e in entries if e]
    lowercase_words = {w.lower() for e in entries for w in re.findall(r"\b[^\W\d_][\w'’-]*", e) if w.islower()}

    def ordinary(word: str) -> bool:
        key = word.lower()
        return key in _LEADING_STOPWORDS or key in _COMMON_STARTERS or key in lowercase_words

    candidates: list[tuple[str, bool]] = []  # (term, weak)
    for entry in entries:
        for words, shapes, initial in _runs(entry):
            while words and (words[0].lower() in _LEADING_STOPWORDS
                             or (initial and shapes[0] == "cap" and ordinary(words[0]))):
                words, shapes, initial = words[1:], shapes[1:], False
            words_shapes = [(w, sh) for w, sh in zip(words, shapes) if w.lower() not in _CALENDAR]
            while words_shapes:
                chunk = words_shapes[:MAX_RUN]
                weak = initial and len(chunk) == 1 and chunk[0][1] == "cap"
                candidates.append((" ".join(w for w, _ in chunk), weak))
                words_shapes, initial = words_shapes[MAX_RUN:], False
    strong_keys = {term.lower() for term, weak in candidates if not weak}
    counts: Counter = Counter()
    first: dict[str, int] = {}
    spelling: dict[str, str] = {}
    for index, (term, weak) in enumerate(candidates):
        key = term.lower()
        if weak and key not in strong_keys:
            continue
        counts[key] += 1
        first.setdefault(key, index)
        spelling.setdefault(key, term)
    return [spelling[k] for k in sorted(counts, key=lambda k: (-counts[k], first[k]))]


def build_prompt(base: Optional[str], groups: Sequence[Sequence[str]], max_chars: int) -> Optional[str]:
    """``base`` with as many terms appended as fit in ``max_chars``, or None when none fit.

    ``groups`` is ordered from most to least important; terms already in ``base`` or in an
    earlier group are skipped. The chosen terms are written least important first.
    """
    base = (base or "").strip()
    head = base + ("" if not base else (" " if base.endswith(_SENTENCE_END) else ". "))
    budget = max_chars - len(head) - 1  # the closing "."
    seen = {base.lower()} if base else set()
    chosen: list[str] = []
    used = 0
    for group in groups:
        for term in group:
            term = term.strip()
            key = term.lower()
            if not term or key in seen or (base and key in base.lower()):
                continue
            cost = len(term) + (2 if chosen else 0)
            if used + cost > budget:
                continue
            seen.add(key)
            chosen.append(term)
            used += cost
    if not chosen:
        return None
    return head + ", ".join(reversed(chosen)) + "."
