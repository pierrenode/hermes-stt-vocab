"""Pick name-like terms out of free text and pack them into a speech-to-text prompt.

``extract_terms(entries)`` scans memory entries for the words a speech model is most likely to
mis-hear: names of people, products and projects, acronyms and identifiers. It keeps

- words with a capital letter after the first character (``gRPC``, ``PostgreSQL``, ``AWS``),
- words that mix letters and digits (``S3``, ``GPT-5``),
- capitalised words in the middle of a sentence (``Northwind``),
- runs of the above as one multi-word name (``Ayşe Demir``, ``Project Atlas``).

A lone capitalised word at the start of a sentence (``Prefers``, ``Uses``) is kept only when the
same word also appears capitalised mid-sentence somewhere. URLs, paths, e-mail addresses and
long strings that look like keys are never kept. A possessive or a suffix written after an
apostrophe (``Northwind's``, ``İstanbul'da``) is dropped and ends the name.

English function words are always known. ``languages`` adds the function words, note words and
month and weekday names of other languages (see languages.py), so a capitalised ``Kullanıcı``,
``Der`` or ``Luego`` is not glued onto the name after it.

``build_prompt(base, groups, max_chars)`` appends terms to the existing prompt until
``max_chars`` is reached. ``groups`` runs from most to least important; the most important terms
are written last, because Whisper-family models condition on the end of the prompt.
"""

from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache
from typing import Iterable, Optional, Sequence

from .languages import CALENDAR as _LANGUAGE_CALENDAR
from .languages import FUNCTION_WORDS as _FUNCTION_WORDS
from .languages import SUPPORTED as SUPPORTED_LANGUAGES

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
# Note-writing words the stop word lists leave out, per language: "user", "then", "now",
# "today", "yesterday", "tomorrow" and the verbs notes open with ("uses", "prefers", ...).
_NOTE_WORDS = {
    "de": ("nutzer", "nutzerin", "benutzer", "benutzerin", "danach", "später", "heute", "gestern",
           "morgen", "nutzt", "benutzt", "verwendet", "bevorzugt", "mag", "arbeitet", "wohnt",
           "lebt", "spricht"),
    "es": ("usuario", "usuaria", "luego", "después", "ahora", "hoy", "ayer", "mañana", "usa",
           "utiliza", "prefiere", "trabaja", "vive", "habla", "también"),
    "fr": ("utilisateur", "utilisatrice", "ensuite", "puis", "maintenant", "hier", "demain",
           "utilise", "préfère", "travaille", "habite", "vit", "parle", "aussi"),
    "it": ("utente", "poi", "dopo", "ora", "adesso", "oggi", "ieri", "domani", "usa", "utilizza",
           "preferisce", "lavora", "vive", "parla", "anche"),
    "nl": ("gebruiker", "gebruikster", "daarna", "later", "vandaag", "gisteren", "morgen",
           "gebruikt", "werkt", "woont", "spreekt", "ook"),
    "pt": ("usuário", "usuária", "utilizador", "utilizadora", "depois", "agora", "hoje", "ontem",
           "amanhã", "usa", "utiliza", "prefere", "trabalha", "mora", "vive", "fala", "também"),
    "tr": ("kullanıcı", "kullanıcının", "kullanıcıya", "kullanıcıyı", "kullanıcıdan", "sonra",
           "önce", "şimdi", "bugün", "dün", "yarın", "artık", "genelde", "genellikle", "bazen",
           "hâlâ"),
}
# A suffix after an apostrophe: English possessive "'s", Turkish case endings ("'da", "'nin").
_APOSTROPHE_SUFFIX = re.compile(r"^(.+?)['’]([^\W\d_]*)$")
_INNER_DOT = re.compile(r"\w\.\w")
MAX_RUN = 4
MAX_TERM_CHARS = 40


def _key(word: str) -> str:
    """Case-insensitive key that also folds Turkish dotted/dotless i (``İçin`` = ``için``)."""
    return word.lower().replace("\u0307", "").replace("ı", "i")


@lru_cache(maxsize=16)
def _language_words(languages: tuple[str, ...]) -> tuple[frozenset, frozenset]:
    """``(stop words, calendar words)`` as keys for the chosen languages, English included."""
    unknown = [code for code in languages if code not in SUPPORTED_LANGUAGES]
    if unknown:
        raise ValueError(f"unsupported languages {unknown!r}; supported: {', '.join(SUPPORTED_LANGUAGES)}")
    stop, calendar = set(), set(_CALENDAR)
    for code in languages:
        stop.update(_FUNCTION_WORDS[code], _NOTE_WORDS[code])
        calendar.update(_LANGUAGE_CALENDAR[code])
    return frozenset(_key(w) for w in stop), frozenset(_key(w) for w in calendar)


def _strip_suffix(word: str) -> tuple[str, bool]:
    """``word`` without a lowercase suffix after its last apostrophe, and whether one was cut."""
    match = _APOSTROPHE_SUFFIX.match(word)
    if match and match.group(2).islower():
        return match.group(1), True
    return word, False


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
    sentence_start, prev_end, prev_raw, prev_suffixed = True, 0, "", False
    for match in _TOKEN.finditer(entry):
        raw, gap = match.group(0), entry[prev_end:match.start()]
        prev_end = match.end()
        if "\n" in gap:
            sentence_start = True
        elif prev_raw.endswith(_SENTENCE_END) and not _INNER_DOT.search(prev_raw.rstrip(".")):
            sentence_start = True
        word, suffixed = _strip_suffix(raw.strip(_EDGE))
        breaks_before = bool(gap.strip()) or "\n" in gap or prev_raw.endswith(_SENTENCE_END) or prev_suffixed
        if word:
            out.append((word, sentence_start, breaks_before))
            sentence_start = False
        prev_raw, prev_suffixed = raw, suffixed
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


def extract_terms(entries: Iterable[str], languages: Sequence[str] = ()) -> list[str]:
    """Name-like terms from ``entries``, most frequent first, ties in order of first appearance.

    ``languages`` are codes from ``SUPPORTED_LANGUAGES`` whose notes may appear in ``entries``.
    """
    entries = [e for e in entries if e]
    stop_words, calendar = _language_words(tuple(sorted(set(languages))))
    lowercase_words = {_key(w) for e in entries for w in re.findall(r"\b[^\W\d_][\w'’-]*", e) if w.islower()}

    def leading(word: str, shape: str) -> bool:
        # A language's function word is dropped only in its plain capitalised form, so "USA"
        # stays a name when Spanish or Italian ("usa") is on.
        return word.lower() in _LEADING_STOPWORDS or (shape == "cap" and _key(word) in stop_words)

    def ordinary(word: str) -> bool:
        key = _key(word)
        return key in _LEADING_STOPWORDS or key in _COMMON_STARTERS or key in lowercase_words

    candidates: list[tuple[str, bool]] = []  # (term, weak)
    for entry in entries:
        for words, shapes, initial in _runs(entry):
            while words and (leading(words[0], shapes[0])
                             or (initial and shapes[0] == "cap" and ordinary(words[0]))):
                words, shapes, initial = words[1:], shapes[1:], False
            words_shapes = [(w, sh) for w, sh in zip(words, shapes) if _key(w) not in calendar]
            while words_shapes:
                chunk = words_shapes[:MAX_RUN]
                weak = initial and len(chunk) == 1 and chunk[0][1] == "cap"
                candidates.append((" ".join(w for w, _ in chunk), weak))
                words_shapes, initial = words_shapes[MAX_RUN:], False
    strong_keys = {_key(term) for term, weak in candidates if not weak}
    counts: Counter = Counter()
    first: dict[str, int] = {}
    spelling: dict[str, str] = {}
    for index, (term, weak) in enumerate(candidates):
        key = _key(term)
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
