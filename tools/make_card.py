"""Render docs/card.png, the 1200x600 catalog card.

The prompt line is real output: the plugin's own pre_transcription hook is called for a local
backend with the two memory notes shown above it. No Hermes needed:

    python tools/make_card.py --fonts /path/to/fonts

Fonts (OFL), from google/fonts at commit 9710da1eacb3be272583c3224dcb70f9da6eadbb:
    ofl/dmsans/DMSans[opsz,wght].ttf          -> DMSans.ttf
        sha256 8cd08d97e89c24d0aa92edd2f0f4c8ee6195eee9b7c9f154865a58b02f0c1c0d
    ofl/jetbrainsmono/JetBrainsMono[wght].ttf -> JetBrainsMono.ttf
        sha256 48715a42ec242c21e9f02692891e147d022299a52e48d5e413e1a942193ffeda

The plugin page hero crops the card to rows 110-490 at its widest; everything drawn here
must sit inside rows 132-468 (about 20px of air on each side), and the script asserts it.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
W, H = 1200, 600
BAND = (132, 468)
MARGIN_X = 96
FONT_SHA256 = {
    "DMSans.ttf": "8cd08d97e89c24d0aa92edd2f0f4c8ee6195eee9b7c9f154865a58b02f0c1c0d",
    "JetBrainsMono.ttf": "48715a42ec242c21e9f02692891e147d022299a52e48d5e413e1a942193ffeda",
}

BG = (15, 17, 21)
PANEL = (24, 27, 33)
PANEL_EDGE = (44, 48, 56)
TEXT = (236, 238, 241)
MUTED = (150, 156, 166)
ACCENT = (126, 214, 160)

NOTES = [
    "Teammates: Bartholomew (backend), Joaquín (design) and Siobhan O'Leary.",
    "Deploys with GitHub Actions to AWS S3; the API runs on gRPC.",
]


def load_plugin():
    plugin_dir = ROOT / "stt-vocab"
    spec = importlib.util.spec_from_file_location(
        "stt_vocab_card", plugin_dir / "__init__.py", submodule_search_locations=[str(plugin_dir)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _Ctx:
    def get_config(self, key, default=None):
        return default


def font(path: Path, size: int, weight: int, opsz: int | None = None) -> ImageFont.FreeTypeFont:
    f = ImageFont.truetype(str(path), size)
    f.set_variation_by_axes([opsz, weight] if opsz is not None else [weight])
    return f


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fonts", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "card.png")
    args = parser.parse_args()
    for name, digest in FONT_SHA256.items():
        actual = hashlib.sha256((args.fonts / name).read_bytes()).hexdigest()
        assert actual == digest, f"{name}: sha256 {actual} != pinned {digest}"

    plugin = load_plugin()
    plugin._memory_entries = lambda: list(NOTES)
    result = plugin.make_hook(_Ctx())(file_path="voice.ogg", provider="local", model=None, language=None,
                                     prompt=None, source="gateway")
    prompt = result["prompt"]
    names = [t.strip() for t in prompt.rstrip(".").split(",")]
    assert all(any(n in note for note in NOTES) for n in names), names

    dm = args.fonts / "DMSans.ttf"
    mono = args.fonts / "JetBrainsMono.ttf"
    title_f = font(dm, 58, 700, 36)
    sub_f = font(dm, 26, 400, 14)
    label_f = font(dm, 19, 500, 14)
    code_f = font(mono, 20, 400)

    ink = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ink)
    y = BAND[0] + 9
    d.text((MARGIN_X, y), "stt-vocab", font=title_f, fill=TEXT)
    y += 74
    d.text((MARGIN_X, y), "The names your notes know, handed to speech-to-text", font=sub_f, fill=MUTED)
    y += 50

    pad, line_h = 18, 29
    rows = [("label", "memories/USER.md"), *[("note", n) for n in NOTES],
            ("label", "transcription prompt for local Whisper"), ("prompt", prompt)]
    gap = 10  # extra air above every label but the first
    panel_bottom = y + pad * 2 + line_h * len(rows) + gap * (sum(k == "label" for k, _ in rows) - 1) - 4
    d.rounded_rectangle((MARGIN_X - 4, y, W - MARGIN_X + 4, panel_bottom), radius=12,
                        fill=PANEL, outline=PANEL_EDGE, width=2)
    width = W - 2 * MARGIN_X - 2 * pad
    ty = y + pad
    for i, (kind, text) in enumerate(rows):
        if kind == "label" and i:
            ty += gap
        f = label_f if kind == "label" else code_f
        x = MARGIN_X + pad + (0 if kind == "label" else 16)
        assert d.textlength(text, font=f) <= width - 16, f"does not fit: {text}"
        d.text((x, ty), text, font=f, fill=MUTED if kind == "label" else ACCENT if kind == "prompt" else TEXT)
        ty += line_h

    left, top, right, bottom = ink.getbbox()
    assert BAND[0] <= top and bottom <= BAND[1], f"ink rows {top}-{bottom} outside {BAND}"
    assert 0 < left and right < W, f"ink columns {left}-{right} outside the card"

    card = Image.new("RGBA", (W, H), BG + (255,))
    card.alpha_composite(ink)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    card.convert("RGB").save(args.out, optimize=True)
    print(f"{args.out} ink rows {top}-{bottom}, columns {left}-{right}")
    print(f"  {prompt}")


if __name__ == "__main__":
    main()
