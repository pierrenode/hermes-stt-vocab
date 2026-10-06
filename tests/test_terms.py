"""Term extraction and prompt packing on their own (no Hermes needed)."""

from __future__ import annotations

import pytest

from conftest import MEMORY_MD, NOTES_BY_LANGUAGE, USER_MD


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


@pytest.mark.parametrize("text, name", [
    ("Ayrıca İstanbul'da yaşıyor.", "İstanbul"),
    ("Dün Northwind’in ofisine gitti.", "Northwind"),
    ("Sonra Bartholomew'a sordu.", "Bartholomew"),
])
def test_a_suffix_after_an_apostrophe_is_dropped(terms, text, name):
    assert terms.extract_terms([text], ["tr"]) == [name]


def test_a_suffix_ends_the_name_like_a_possessive(terms):
    assert terms.extract_terms(["Yarın Northwind'in CTO'su geliyor."], ["tr"]) == ["Northwind", "CTO"]


def test_an_apostrophe_inside_a_name_is_kept(terms):
    assert _extract(terms, "We met Siobhan O'Leary and visited L'Oréal.") == ["Siobhan O'Leary", "L'Oréal"]


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


NAMES = ("Northwind", "PostgreSQL", "Bartholomew", "Priya Raman")


@pytest.mark.parametrize("language", sorted(NOTES_BY_LANGUAGE))
def test_notes_in_another_language_give_clean_names_with_that_language_on(terms, language):
    notes = NOTES_BY_LANGUAGE[language]
    found = terms.extract_terms(notes, [language])
    for name in NAMES:
        assert name in found, (name, found)
    # Without the language, its sentence openers are glued onto the next name.
    assert any(" " in t and t.split()[0] not in ("Project", "Proyecto", "Projet", "Projekt", "Priya")
               for t in terms.extract_terms(notes))


@pytest.mark.parametrize("language, opener", [
    ("tr", "Kullanıcı"), ("tr", "Sonra"), ("tr", "Ayrıca"), ("tr", "İşte"),
    ("de", "Der"), ("de", "Nutzer"), ("de", "Bevorzugt"), ("de", "Dann"), ("de", "Seine"),
    ("es", "Usa"), ("es", "Prefiere"), ("es", "Luego"), ("es", "Su"),
    ("fr", "Utilise"), ("fr", "Préfère"), ("fr", "Ensuite"), ("fr", "Sa"),
])
def test_openers_never_become_part_of_a_term(terms, language, opener):
    found = terms.extract_terms(NOTES_BY_LANGUAGE[language], [language])
    assert not any(opener in t.split() for t in found), (opener, found)


@pytest.mark.parametrize("language, month", [("de", "März"), ("tr", "Kasım"), ("nl", "Maart"),
                                             ("it", "Marzo"), ("pt", "Março")])
def test_month_names_of_the_chosen_language_are_skipped(terms, language, month):
    text = f"Bartholomew joined Atlas in {month}."
    assert month in _extract(terms, text)
    assert month not in terms.extract_terms([text], [language])


def test_a_language_word_in_capitals_stays_a_name(terms):
    # "usa" is a Spanish/Italian/Portuguese function word; "USA" is still a name.
    assert terms.extract_terms(["She moved to the USA in May."], ["es", "it", "pt"]) == ["USA"]


def test_turkish_dotted_capital_i_matches_lowercase_words(terms):
    assert terms.extract_terms(["İşte Bartholomew geldi."], ["tr"]) == ["Bartholomew"]
    assert terms.extract_terms(["İşte Bartholomew geldi."]) == ["İşte Bartholomew"]


def test_turkish_letters_fold_when_matching_lowercase_uses(terms):
    # No language set: "Kullanıcı" is ordinary because the notes also use "kullanıcı".
    assert terms.extract_terms(["Kullanıcı Northwind'de çalışıyor.", "Bu kullanıcı kahve seviyor."]) == ["Northwind"]


def test_a_vouched_turkish_sentence_opener_counts_toward_frequency(terms):
    # "İstanbul" opens the second note; the mid-sentence use vouches for it, so it counts twice.
    assert terms.extract_terms(["We visited Northwind and İstanbul.", "İstanbul was hot."]) == ["İstanbul", "Northwind"]


def test_calendar_names_match_in_capitals(terms):
    assert terms.extract_terms(["We met Bartholomew in KASIM."], ["tr"]) == ["Bartholomew"]
    assert terms.extract_terms(["We met Bartholomew in KASIM."]) == ["Bartholomew", "KASIM"]


def test_english_notes_are_unchanged_by_languages(terms):
    entries = USER_MD.split("\n§\n") + MEMORY_MD.split("\n§\n")
    assert terms.extract_terms(entries, ["tr", "de"]) == terms.extract_terms(entries)


def test_an_unsupported_language_is_an_error(terms):
    with pytest.raises(ValueError, match="unsupported languages"):
        terms.extract_terms(["Bartholomew joined."], ["tr", "xx"])


def test_every_supported_language_has_words_and_calendar_names(terms):
    from importlib import import_module
    languages = import_module(terms.__name__.rsplit(".", 1)[0] + ".languages")
    assert set(terms.SUPPORTED_LANGUAGES) == set(languages.FUNCTION_WORDS) == set(languages.CALENDAR)
    assert set(terms._NOTE_WORDS) == set(terms.SUPPORTED_LANGUAGES)
    for code in terms.SUPPORTED_LANGUAGES:
        assert len(languages.FUNCTION_WORDS[code]) > 50 and len(languages.CALENDAR[code]) == 19


def test_lowercase_identifiers_with_digits_are_kept(terms):
    assert _extract(terms, "Automations run in n8n and k8s clusters.") == ["n8n", "k8s"]


def test_the_closing_period_counts_toward_max_chars(terms):
    assert terms.build_prompt(None, [["Atlas"]], 6) == "Atlas."
    assert terms.build_prompt(None, [["Atlas"]], 5) is None
