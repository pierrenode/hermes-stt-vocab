"""The plugin inside a real Hermes: run with Hermes importable (CI checks out the release floor
and main). Each test installs the plugin under a fresh HERMES_HOME, enables and configures it with
the CLI, writes memory files where Hermes keeps them, and runs Hermes's own transcription in a
separate process. Cloud tests point the OpenAI STT backend at a local fake server and read the
``prompt`` field of the multipart request it receives."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from conftest import MEMORY_MD, PLUGIN_DIR, USER_MD

pytest.importorskip("tools.transcription_tools", reason="needs Hermes importable")

_TRANSCRIBE = r"""
import json, sys
from tools.transcription_tools import transcribe_audio
print(json.dumps(transcribe_audio(sys.argv[1], source=sys.argv[2])))
"""

# The dispatcher's own hook step, for a backend this test machine cannot run (local faster-whisper).
_HOOK_STEP = r"""
import json, sys
try:
    from tools.transcription_command import _apply_pre_transcription_hook
except ImportError:  # 0.21.0 keeps it in transcription_tools
    from tools.transcription_tools import _apply_pre_transcription_hook
model, language, prompt = _apply_pre_transcription_hook(
    file_path=sys.argv[1], provider=sys.argv[2], model=None, language=None, prompt=None, source="gateway")
print(json.dumps({"prompt": prompt}))
"""


class _FakeSTT:
    """OpenAI-compatible ``/v1/audio/transcriptions`` that records each request's ``prompt`` field."""

    def __init__(self) -> None:
        self.prompts: list = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                boundary = re.search(r'boundary="?([^";]+)', self.headers.get("Content-Type", "")).group(1)
                prompt = None
                for part in body.split(b"--" + boundary.encode()):
                    head, _, value = part.partition(b"\r\n\r\n")
                    if b'name="prompt"' in head:
                        prompt = value.rsplit(b"\r\n", 1)[0].decode("utf-8")
                outer.prompts.append(prompt)
                text = "hello from the fake server"
                wants_json = b'name="response_format"\r\n\r\njson' in body
                out = (json.dumps({"text": text}) if wants_json else text).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json" if wants_json else "text/plain")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

            def log_message(self, *_args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def close(self) -> None:
        self.server.shutdown()


@pytest.fixture
def fake_stt():
    srv = _FakeSTT()
    yield srv
    srv.close()


def _env(home) -> dict:
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("HERMES_", "OPENAI", "ANTHROPIC", "VOICE_", "GROQ", "MISTRAL"))}
    env["HERMES_HOME"] = str(home)
    return env


def _cli(home, *args: str) -> None:
    subprocess.run([sys.executable, "-m", "hermes_cli.main", *args], env=_env(home), check=True,
                   capture_output=True, text=True, timeout=180, stdin=subprocess.DEVNULL)


def _run(home, script: str, *args: str) -> dict:
    proc = subprocess.run([sys.executable, "-c", script, *args], env=_env(home), check=True,
                          capture_output=True, text=True, timeout=180, stdin=subprocess.DEVNULL)
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _home(tmp_path, *, enabled=True, base_url=None, stt_prompt=None, **settings):
    home = tmp_path / "home"
    (home / "memories").mkdir(parents=True)
    (home / "memories" / "USER.md").write_text(USER_MD, encoding="utf-8", newline="\n")
    (home / "memories" / "MEMORY.md").write_text(MEMORY_MD, encoding="utf-8", newline="\n")
    if enabled:
        shutil.copytree(PLUGIN_DIR, home / "plugins" / "stt-vocab")
        _cli(home, "plugins", "enable", "stt-vocab")
    if base_url:
        for key, value in (("stt.provider", "openai"), ("stt.openai.api_key", "sk-fake-stt"),
                           ("stt.openai.base_url", base_url), ("stt.cloud_trim_silence", "false")):
            _cli(home, "config", "set", key, value)
    if stt_prompt:
        _cli(home, "config", "set", "stt.prompt", stt_prompt)
    for key, value in settings.items():
        _cli(home, "config", "set", f"plugins.entries.stt-vocab.settings.{key}", value)
    return home


def _wav(tmp_path):
    path = tmp_path / "voice.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 8000)
    return str(path)


def test_cloud_request_carries_configured_terms_but_not_memory_names(tmp_path, fake_stt):
    home = _home(tmp_path, base_url=fake_stt.base_url, terms='["Teknium", "Kanban"]')
    result = _run(home, _TRANSCRIBE, _wav(tmp_path), "gateway")
    assert result["success"] is True and result["transcript"] == "hello from the fake server"
    assert fake_stt.prompts == ["Kanban, Teknium."]


def test_without_the_plugin_the_cloud_request_has_no_prompt(tmp_path, fake_stt):
    home = _home(tmp_path, enabled=False, base_url=fake_stt.base_url)
    assert _run(home, _TRANSCRIBE, _wav(tmp_path), "gateway")["success"] is True
    assert fake_stt.prompts == [None]


def test_allow_remote_sends_memory_names_to_the_cloud(tmp_path, fake_stt):
    home = _home(tmp_path, base_url=fake_stt.base_url, allow_remote="true")
    _run(home, _TRANSCRIBE, _wav(tmp_path), "gateway")
    (prompt,) = fake_stt.prompts
    for name in ("Ayşe Demir", "Northwind", "Bartholomew", "Joaquín", "PostgreSQL"):
        assert name in prompt, (name, prompt)
    assert "github.com" not in prompt and len(prompt) <= 896


def test_stt_prompt_is_kept_in_front(tmp_path, fake_stt):
    home = _home(tmp_path, base_url=fake_stt.base_url, stt_prompt="Hermes, Teknium", terms='["Teknium", "Kanban"]')
    _run(home, _TRANSCRIBE, _wav(tmp_path), "gateway")
    assert fake_stt.prompts == ["Hermes, Teknium. Kanban."]


def test_source_terms_apply_only_to_their_source(tmp_path, fake_stt):
    home = _home(tmp_path, base_url=fake_stt.base_url, source_terms='{"voice_mode": ["Atlas"]}')
    _run(home, _TRANSCRIBE, _wav(tmp_path), "voice_mode")
    _run(home, _TRANSCRIBE, _wav(tmp_path), "gateway")
    assert fake_stt.prompts == ["Atlas.", None]


def test_local_backend_gets_names_from_this_profiles_memory(tmp_path):
    home = _home(tmp_path)
    prompt = _run(home, _HOOK_STEP, _wav(tmp_path), "local")["prompt"]
    for name in ("Ayşe Demir", "Northwind", "Project Atlas", "Bartholomew", "Siobhan O'Leary", "gRPC"):
        assert name in prompt, (name, prompt)
    assert _run(home, _HOOK_STEP, _wav(tmp_path), "openai")["prompt"] is None
