"""Generate stt-vocab/languages.py: function words and calendar names for notes not in English.

    python tools/make_languages.py

Every source is fetched at a pinned commit and checked against its sha256 before use:

- function words for de, es, fr, it, nl, pt: the Snowball stop word lists
  (snowballstem/snowball-website, algorithms/<language>/stop.txt), BSD-3-Clause;
- function words for tr: Apache Lucene's Turkish stop word list
  (lucene/analysis/common/.../analysis/tr/stopwords.txt), Apache-2.0;
- month and weekday names: Unicode CLDR (cldr-json, cldr-dates-full/main/<locale>/
  ca-gregorian.json, wide format and stand-alone forms), Unicode License v3.

THIRD_PARTY_NOTICES.md carries the licences. The words a note-writer adds on top of these lists
(``user``, ``then``, ``today`` ...) are not generated; they live in terms.py next to the English
ones.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "stt-vocab" / "languages.py"

SNOWBALL = "https://raw.githubusercontent.com/snowballstem/snowball-website/a5c23fcf6cb8c34f450d5080d888e65bc4681c1b/algorithms/{}/stop.txt"
LUCENE_TR = "https://raw.githubusercontent.com/apache/lucene/d148795b182f4f300477c24bd620024627d653c6/lucene/analysis/common/src/resources/org/apache/lucene/analysis/tr/stopwords.txt"
CLDR = "https://raw.githubusercontent.com/unicode-org/cldr-json/91c267402229a59e3ef2774544f001bf959e8809/cldr-json/cldr-dates-full/main/{}/ca-gregorian.json"

SNOWBALL_LANGUAGES = {"de": "german", "es": "spanish", "fr": "french", "it": "italian",
                      "nl": "dutch", "pt": "portuguese"}
LANGUAGES = (*SNOWBALL_LANGUAGES, "tr")

SHA256 = {
    SNOWBALL.format("german"):
        "22a7785b53860256280fba74b93d8f09a0f3a8f52edaa3cdc3750e8c13a3874c",
    SNOWBALL.format("spanish"):
        "042669c6c97605f5e57278b67474c65d58d628a59a004e618a0ac06bc2d38222",
    SNOWBALL.format("french"):
        "e235f5e633bf831c601ce6f1dc87d8608c038209a7aab64e69a9e70f52f83d4c",
    SNOWBALL.format("italian"):
        "054a4926bcdf85fd016bd2a3647f0ef90c7498ddc3d69e165751fba86c4c026a",
    SNOWBALL.format("dutch"):
        "c3252e08b041b795da5ea8f1ad9103b34f64248760848226b5e5c2f5e89789e7",
    SNOWBALL.format("portuguese"):
        "be242d802beb0d1868589409dd998d33b45873ef36f2b9f582dd0cf9099e1ae7",
    LUCENE_TR:
        "84b15dfd727af4b84ee875487cdd80be58fa3b6f22604eb4b9d6dbae52dfff66",
    CLDR.format("de"): "afa871ef7f2b4b9a7d680342e18dcf6a37a910e81af8d7d9561722d790b6d20b",
    CLDR.format("es"): "159f621f147875bbd66daec061da512b20be7eb04cd18a245c8133a162555355",
    CLDR.format("fr"): "0a82c7914d023e375888e914d63be846cb9815c03db25308ae303c1ce3d8bace",
    CLDR.format("it"): "8737e1b12637de42fd952d58f81c2861f0030cf935b215231bd65c7e8b6ffc69",
    CLDR.format("nl"): "febb723af1f8425840d6f5dcff1d84fde70e9747753b7c9b03c1ec9599e833c9",
    CLDR.format("pt"): "a4326e3578a7b3e270e4b7ec74bf5122edb2cb1e6bade23180f7dfc9d09e2eca",
    CLDR.format("tr"): "b50aedb540f7b1f81ffd36e1eb531454cea428896be9d49d8b633d43d7657077",
}


def fetch(url: str) -> str:
    with urllib.request.urlopen(url, timeout=60) as response:
        raw = response.read()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SHA256[url]:
        raise SystemExit(f"sha256 mismatch for {url}: {digest}")
    return raw.decode("utf-8")


def snowball_words(text: str) -> set[str]:
    # "word   | comment"; a bar starts a comment, several words may share a line.
    return {w for line in text.splitlines() for w in line.split("|", 1)[0].split()}


def lucene_words(text: str) -> set[str]:
    return {line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")}


def calendar_words(text: str, code: str) -> set[str]:
    gregorian = json.loads(text)["main"][code]["dates"]["calendars"]["gregorian"]
    words = set()
    for kind in ("months", "days"):
        for context in ("format", "stand-alone"):
            words.update(gregorian[kind][context]["wide"].values())
    return {w.lower() for w in words}


def render(name: str, table: dict[str, set[str]]) -> str:
    lines = [f"{name} = {{"]
    for code in LANGUAGES:
        words = sorted(table[code])
        lines.append(f'    "{code}": frozenset({{')
        row = "       "
        for word in words:
            item = f' "{word}",'
            if len(row) + len(item) > 99:
                lines.append(row)
                row = "       "
            row += item
        lines.append(row)
        lines.append("    }),")
    lines.append("}")
    return "\n".join(lines)


def main() -> None:
    function_words = {code: snowball_words(fetch(SNOWBALL.format(lang)))
                      for code, lang in SNOWBALL_LANGUAGES.items()}
    function_words["tr"] = lucene_words(fetch(LUCENE_TR))
    calendar = {code: calendar_words(fetch(CLDR.format(code)), code) for code in LANGUAGES}
    header = (
        '"""Function words and calendar names per language. Generated by tools/make_languages.py;\n'
        "do not edit by hand.\n\n"
        "Sources (pinned commits, licences in THIRD_PARTY_NOTICES.md): Snowball stop word lists\n"
        "(BSD-3-Clause) for de, es, fr, it, nl, pt; Apache Lucene's Turkish stop words (Apache-2.0)\n"
        'for tr; Unicode CLDR month and weekday names (Unicode License v3).\n"""\n\n'
    )
    body = "\n\n".join([
        f"SUPPORTED = {LANGUAGES!r}",
        render("FUNCTION_WORDS", function_words),
        render("CALENDAR", calendar),
    ])
    OUT.write_text(header + body + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
