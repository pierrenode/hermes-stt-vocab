"""Term extraction and prompt packing on their own (no Hermes needed)."""

from __future__ import annotations

import pytest

from conftest import MEMORY_MD, USER_MD


@pytest.fixture
def terms(plugin):
    return plugin.terms


def _extract(terms, *entries):
    return terms.extract_terms(list(entries))


def test_names_products_and_identifiers_are_kept(terms):
    found = _extract(terms, *USER_MD.split("\n§\n"), *MEMORY_MD.split("\n§\n"))
    for name in ("Ayşe Demir", "Northwind", "Project Atlas", "Neovim", "Kubernetes", "PostgreSQL",
                 "Bartholomew", "Joaquín", "Siobhan O'Leary", "GitHub Actions", "AWS S3", "Node.js",
                 "gRPC", "Priya Raman"):
        assert name in found, name


def test_sentence_openers_and_ordinary_words_are_not_names(terms):
    found = [t.lower() for t in _extract(terms, *USER_MD.split("\n§\n"), "Likes Turkish coffee. Hates YAML.")]
    for word in ("prefers", "uses", "uses neovim", "likes", "likes turkish", "hates", "hates yaml",
                 "she", "user", "the"):
        assert word not in found, word
    assert "yaml" in found and "turkish" in found


def test_a_sentence_opener_counts_when_it_is_also_used_as_a_name(terms):
    # "Ayşe" alone opens a sentence; on its own it is not trusted ...
    assert "Ayşe" not in _extract(terms, "Ayşe meets Bartholomew on Fridays.")
    # ... but the same word capitalised mid-sentence elsewhere vouches for it.
    assert "Ayşe" in _extract(terms, "Ayşe meets Bartholomew.", "Lunch with Ayşe today.")


def test_a_word_seen_in_lowercase_is_ordinary_at_the_start_of_a_sentence(terms):
    found = _extract(terms, "Ships Kotlin code.", "The team ships weekly.")
    assert "Kotlin" in found and "Ships Kotlin" not in found


def test_possessives_end_a_name(terms):
    found = _extract(terms, "We met Northwind's CFO.")
    assert "Northwind" in found and "Northwind's CFO" not in found and "CFO" in found


def test_calendar_words_are_skipped(terms):
    assert _extract(terms, "Standup with Bartholomew every Monday in March.") == ["Bartholomew"]


@pytest.mark.parametrize("text", [
    "The repo lives at https://github.com/northwind/atlas today.",
    "Config is in ~/work/atlas/.env and C:\\Users\\Ayse\\notes.",
    "Mail Priya at priya@northwind.example for access.",
    "Token for staging was sk_live_9fA2kq81ZzP0xyab3 last week.",
])
def test_locators_and_key_like_strings_are_never_kept(terms, text):
    found = " ".join(_extract(terms, text))
    for bad in ("://", "/", "\\", "@", "sk_live", "9fA2"):
        assert bad not in found, (bad, found)


def test_sentence_opening_function_words_are_not_part_of_a_name(terms):
    assert _extract(terms, "Then Bartholomew left.", "Later Priya Raman joined.") == ["Bartholomew", "Priya Raman"]


def test_more_frequent_terms_come_first(terms):
    found = _extract(terms, "Talked to Priya Raman.", "Asked Bartholomew.", "Then Bartholomew again.")
    assert found == ["Bartholomew", "Priya Raman"]


def test_long_runs_are_split(terms):
    assert _extract(terms, "The plan is Alpha Bravo Charlie Delta Echo Foxtrot today.") == [
        "Alpha Bravo Charlie Delta", "Echo Foxtrot"]


def test_line_breaks_end_names_and_sentences(terms):
    assert _extract(terms, "- Likes Turkish coffee\n- Speaks English") == ["Turkish", "English"]


def test_prompt_puts_the_most_important_terms_last(terms):
    assert terms.build_prompt(None, [["Atlas"], ["Neovim", "Kubernetes"]], 200) == "Kubernetes, Neovim, Atlas."


def test_prompt_keeps_the_base_and_skips_terms_already_in_it(terms):
    out = terms.build_prompt("Hermes, Nous Research", [["Nous Research", "Teknium"], ["hermes", "Atlas"]], 200)
    assert out == "Hermes, Nous Research. Atlas, Teknium."
    assert terms.build_prompt("Done.", [["Atlas"]], 200) == "Done. Atlas."


def test_prompt_dedupes_across_groups_case_insensitively(terms):
    assert terms.build_prompt("", [["gRPC"], ["GRPC", "Atlas"]], 200) == "Atlas, gRPC."


def test_prompt_never_exceeds_max_chars(terms):
    many = [f"Name{i:03d}x" for i in range(200)]
    for limit in (40, 100, 896):
        out = terms.build_prompt("Base prompt", [many], limit)
        assert out is not None and len(out) <= limit, (limit, out)
    # Lower-priority terms are the ones dropped.
    out = terms.build_prompt(None, [["Important"], many], 40)
    assert out.endswith("Important.")


def test_prompt_is_none_when_nothing_fits_or_nothing_is_new(terms):
    assert terms.build_prompt("x" * 50, [["Atlas"]], 40) is None
    assert terms.build_prompt("Atlas", [["atlas"]], 200) is None
    assert terms.build_prompt(None, [[], ["  "]], 200) is None


def test_a_term_that_does_not_fit_does_not_stop_shorter_ones(terms):
    assert terms.build_prompt(None, [["A" * 50, "Atlas"]], 20) == "Atlas."


def test_lowercase_identifiers_with_digits_are_kept(terms):
    assert _extract(terms, "Automations run in n8n and k8s clusters.") == ["n8n", "k8s"]


def test_the_closing_period_counts_toward_max_chars(terms):
    assert terms.build_prompt(None, [["Atlas"]], 6) == "Atlas."
    assert terms.build_prompt(None, [["Atlas"]], 5) is None
