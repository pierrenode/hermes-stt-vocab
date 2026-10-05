# stt-vocab

A Hermes plugin that tells the speech-to-text model which names to expect. Whisper-family
models mis-hear the words that matter most in a voice message: people, products, projects,
acronyms. They spell them right far more often when the request carries a short prompt that
contains them. Hermes can send one static prompt (`stt.prompt`); this plugin builds it for you,
per request, from:

- the terms you list in its settings, globally or for one surface (gateway voice messages,
  voice mode), and
- names it finds in this profile's memory files (`memories/USER.md` and `memories/MEMORY.md`),
  the notes Hermes already keeps about you. Off for cloud STT unless you allow it (see Privacy).

```text
stt.prompt (yours, kept as is)      Hermes, Nous Research.
+ stt-vocab                         PostgreSQL, Kubernetes, Project Atlas, Northwind, Ayşe Demir, Bartholomew.
```

It uses Hermes's `pre_transcription` hook, so it works with every built-in backend that accepts
a prompt: local faster-whisper, OpenAI, Groq, Mistral and DeepInfra. Backends that do not take a
prompt (xAI, ElevenLabs, command providers) transcribe as before.

## Install

```bash
hermes plugins install stt-vocab --enable
hermes gateway restart   # if you use voice messages through the gateway
```

## Settings

All optional, under `plugins.entries.stt-vocab.settings`. They are read on every transcription,
so a change applies without a restart.

| Setting | Default | Meaning |
|---|---|---|
| `terms` | `[]` | Terms for every transcription. |
| `source_terms` | `{}` | Extra terms for one caller surface: `gateway` (voice messages), `voice_mode`, ... These come first. |
| `from_memory` | `true` | Add names found in `memories/USER.md` and `memories/MEMORY.md`. |
| `allow_remote` | `false` | Also send memory-derived names to cloud STT providers. |
| `max_chars` | `896` | Longest prompt the plugin builds (1–4000). 896 is what Hermes keeps for Whisper-family backends, so it never has to cut the prompt. |

```yaml
plugins:
  entries:
    stt-vocab:
      settings:
        terms: ["Teknium", "Kanban"]
        source_terms:
          gateway: ["Ayşe", "Northwind"]
```

Order matters: Whisper conditions on the end of the prompt, so the plugin writes the least
important terms first and the most important last: memory names, then `terms`, then
`source_terms`. Your `stt.prompt` stays at the front untouched, and terms it already contains
are not repeated. When everything does not fit in `max_chars`, memory names are dropped first.

A setting the plugin cannot use (for example `terms: "Atlas"` instead of a list) is logged once
as a warning and the transcription goes out unchanged.

## How names are found in memory

Memory entries are free text, so the plugin keeps the words that look like names:

- words with a capital letter after the first one (`gRPC`, `PostgreSQL`, `AWS`),
- words that mix letters and digits (`S3`, `GPT-5`),
- capitalised words in the middle of a sentence (`Northwind`),
- runs of these as one name (`Ayşe Demir`, `Project Atlas`, `GitHub Actions`).

A capitalised word that only opens a sentence is not trusted on its own. Common sentence openers
(`Prefers`, `Uses`, `Then`, ...), words that also appear in lowercase in your notes, weekday and
month names, URLs, paths, e-mail addresses and long letter-and-digit strings that look like keys
are never sent. More frequent names win when space is short. The rules are tuned for notes
written in English; in languages that capitalise every noun (German) more ordinary words will
pass.

## Privacy

The prompt is uploaded to the STT provider together with the audio. Hermes's own documentation
asks that nothing session-derived goes into it for a hosted API, so by default:

- your `terms` and `source_terms` go to every provider, like `stt.prompt` does;
- memory-derived names go only to a backend running on this machine: `local` (faster-whisper)
  or `local_command`. For any other provider, including `type: command` providers whose
  destination the plugin cannot know, the memory files are not even read.

Set `allow_remote: true` to send memory-derived names to cloud providers too.

## Security and footprint

- `register()` makes a single call, `ctx.register_hook("pre_transcription", ...)`. No tools,
  middleware, commands or slash commands.
- No network access and no subprocesses. Hermes, not the plugin, sends the request.
- Reads only its own settings and, when memory names are on and the backend is local (or
  `allow_remote` is set), the active profile's `memories/USER.md` and `memories/MEMORY.md`, at
  most 256 KB each. Writes nothing.
- Returns at most a new `prompt`; never changes `model`, `language` or the audio file. Hermes runs
  the hook fail-open: if it raised, the request would go out unchanged.

## Compatibility

Needs Hermes 0.21.0 or later (`pre_transcription` and the `stt.prompt` threading shipped there).
Tested against `v2026.8.31` (0.21.0) and `main`.

## Development

```bash
python -m pytest tests/test_terms.py tests/test_plugin.py   # no Hermes needed
PYTHONPATH=/path/to/hermes-agent /path/to/hermes-agent/.venv/bin/python -m pytest tests
```

`tests/test_hermes.py` installs the plugin into a fresh `HERMES_HOME`, enables and configures it
with the Hermes CLI, writes memory files where Hermes keeps them, and runs Hermes's own
`transcribe_audio` against a local fake OpenAI-compatible server that records the `prompt` field
of each request.

## License

MIT
