"""Shared helpers: the plugin loaded the way Hermes loads it, and a fake ctx."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PLUGIN_DIR = REPO_ROOT / "stt-vocab"

# Notes in the shape Hermes keeps them: entries joined by "\n§\n". Every name is made up.
USER_MD = "\n§\n".join([
    "User's name is Ayşe Demir. She works at Northwind on Project Atlas.",
    "Prefers concise answers. Uses Neovim, Kubernetes and PostgreSQL daily.",
    "Teammates: Bartholomew (backend), Joaquín (design) and Siobhan O'Leary.",
])
MEMORY_MD = "\n§\n".join([
    "Deploys with GitHub Actions to AWS S3; the API runs on Node.js and gRPC.",
    "Ayşe meets Bartholomew every Monday. Northwind's CFO is Priya Raman.",
    "The repo lives at https://github.com/northwind/atlas and config in ~/work/atlas/.env",
])


def load_plugin():
    name = "stt_vocab_under_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, PLUGIN_DIR / "__init__.py", submodule_search_locations=[str(PLUGIN_DIR)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class FakeCtx:
    """The two PluginContext calls the plugin makes, backed by a plain settings dict."""

    def __init__(self, **settings) -> None:
        self.settings = settings
        self.hooks = {}

    def get_config(self, key, default=None):
        return self.settings.get(key, default)

    def register_hook(self, name, callback):
        self.hooks.setdefault(name, []).append(callback)


@pytest.fixture(scope="session")
def plugin():
    return load_plugin()
