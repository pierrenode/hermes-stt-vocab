"""The hook callback with a fake ctx (no Hermes needed): sources, privacy gate, settings."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from conftest import MEMORY_MD, PLUGIN_DIR, USER_MD, FakeCtx

HOOK = "pre_transcription"


@pytest.fixture(autouse=True)
def _memory(plugin, monkeypatch):
    plugin._warned.clear()
    entries = USER_MD.split("\n§\n") + MEMORY_MD.split("\n§\n")
    monkeypatch.setattr(plugin, "_memory_entries", lambda: list(entries))


def _call(plugin, provider="local", prompt=None, source="gateway", **settings):
    hook = plugin.make_hook(FakeCtx(**settings))
    return hook(file_path="/tmp/a.ogg", provider=provider, model=None, language=None,
                prompt=prompt, source=source, telemetry_schema_version=1)


def test_register_adds_one_hook(plugin):
    ctx = FakeCtx()
    plugin.register(ctx)
    assert list(ctx.hooks) == [HOOK] and len(ctx.hooks[HOOK]) == 1


def test_local_backend_gets_memory_names(plugin):
    out = _call(plugin, provider="local")["prompt"]
    for name in ("Ayşe Demir", "Northwind", "Bartholomew", "Kubernetes"):
        assert name in out
    assert len(out) <= plugin.DEFAULT_MAX_CHARS


@pytest.mark.parametrize("provider", ["openai", "groq", "mistral", "deepinfra", "elevenlabs", "xai", "my_cmd"])
def test_cloud_and_command_backends_get_no_memory_names_by_default(plugin, provider):
    assert _call(plugin, provider=provider) is None
    out = _call(plugin, provider=provider, terms=["Atlas"])["prompt"]
    assert out == "Atlas." and "Northwind" not in out


def test_allow_remote_sends_memory_names_to_cloud_backends(plugin):
    assert "Northwind" in _call(plugin, provider="openai", allow_remote=True)["prompt"]


def test_local_command_counts_as_local(plugin):
    assert "Northwind" in _call(plugin, provider="local_command")["prompt"]


def test_from_memory_off_reads_no_memory(plugin, monkeypatch):
    monkeypatch.setattr(plugin, "_memory_entries", lambda: pytest.fail("memory was read"))
    assert _call(plugin, from_memory=False) is None
    assert _call(plugin, from_memory=False, terms=["Atlas"])["prompt"] == "Atlas."


def test_cloud_backend_does_not_read_memory_at_all(plugin, monkeypatch):
    monkeypatch.setattr(plugin, "_memory_entries", lambda: pytest.fail("memory was read"))
    assert _call(plugin, provider="openai", terms=["Atlas"])["prompt"] == "Atlas."


def test_source_terms_come_last_then_terms_then_memory(plugin):
    out = _call(plugin, provider="local", terms=["Hermes"], source="voice_mode",
                source_terms={"voice_mode": ["Teknium"], "gateway": ["Ignored"]})["prompt"]
    assert out.endswith(", Hermes, Teknium.")
    assert "Ignored" not in out


def test_the_configured_prompt_is_kept_and_not_repeated(plugin):
    out = _call(plugin, provider="openai", prompt="Northwind, Atlas", terms=["Atlas", "Kanban"])["prompt"]
    assert out == "Northwind, Atlas. Kanban."


def test_nothing_new_leaves_the_request_unchanged(plugin):
    assert _call(plugin, provider="openai", prompt="Atlas", terms=["atlas"]) is None


def test_max_chars_caps_the_prompt(plugin):
    out = _call(plugin, provider="local", max_chars=60)["prompt"]
    assert len(out) <= 60


@pytest.mark.parametrize("settings", [
    {"terms": "Atlas"}, {"terms": [1, 2]}, {"source_terms": ["Atlas"]}, {"source_terms": {"gateway": "Atlas"}},
    {"from_memory": "yes"}, {"allow_remote": 1}, {"max_chars": 0}, {"max_chars": 4001},
    {"max_chars": "896"}, {"max_chars": True},
])
def test_an_unusable_setting_leaves_the_prompt_unchanged_and_says_so_once(plugin, caplog, settings):
    with caplog.at_level(logging.WARNING):
        assert _call(plugin, **settings) is None
        assert _call(plugin, **settings) is None
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "leaving the transcription prompt unchanged" in warnings[0].getMessage()


def test_null_lists_are_treated_as_empty(plugin):
    assert _call(plugin, provider="openai", terms=None, source_terms=None) is None


def test_memory_files_are_read_from_the_active_home(plugin, tmp_path, monkeypatch):
    pytest.importorskip("hermes_constants")
    monkeypatch.undo()
    plugin._warned.clear()
    (tmp_path / "memories").mkdir()
    (tmp_path / "memories" / "USER.md").write_text("﻿" + USER_MD, encoding="utf-8", newline="\n")
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)
    entries = plugin._memory_entries()
    assert entries == USER_MD.split("\n§\n")


def test_only_the_start_of_a_huge_memory_file_is_read(plugin, tmp_path, monkeypatch, caplog):
    pytest.importorskip("hermes_constants")
    monkeypatch.undo()
    plugin._warned.clear()
    (tmp_path / "memories").mkdir()
    filler = "\n§\n".join(["Talked to Bartholomew."] + ["note"] * 60_000)
    (tmp_path / "memories" / "MEMORY.md").write_text(filler + "\n§\nMet Zebulon.", encoding="utf-8", newline="\n")
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)
    with caplog.at_level(logging.WARNING):
        entries = plugin._memory_entries()
    assert entries[0] == "Talked to Bartholomew." and "Met Zebulon." not in entries
    assert any("larger than" in r.getMessage() for r in caplog.records)


def test_manifest_defaults_match_the_code(plugin):
    text = (PLUGIN_DIR / "plugin.yaml").read_text(encoding="utf-8-sig")
    assert f"default: {plugin.DEFAULT_MAX_CHARS}\n" in text


def test_plugin_readme_is_the_repo_readme():
    root = (Path(PLUGIN_DIR).parent / "README.md").read_bytes()
    assert (PLUGIN_DIR / "README.md").read_bytes() == root
