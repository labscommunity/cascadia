"""Unit tests for tools/model_aliases.py."""

from __future__ import annotations

import os
import sys

import pytest

_TOOLS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

from model_aliases import ALIASES, resolve  # noqa: E402


@pytest.mark.parametrize(
    "alias,expected",
    [
        ("r1-distill-qwen-7b", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"),
        ("r1-distill-llama-8b", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"),
        ("llama-3.1-8b", "unsloth/Meta-Llama-3.1-8B-Instruct"),
        ("mistral-nemo-12b", "mistralai/Mistral-Nemo-Instruct-2407"),
        ("gemma-4-e2b", "unsloth/gemma-4-E2B-it"),
        ("phi-4", "microsoft/phi-4"),
        ("phi-4-mini", "microsoft/Phi-4-mini-instruct"),
        ("qwen3-1.7b", "Qwen/Qwen3-1.7B"),
    ],
)
def test_known_aliases_resolve(alias, expected):
    assert resolve(alias) == expected


def test_alias_case_insensitive():
    """Aliases should be case-insensitive for user convenience."""
    assert resolve("R1-Distill-Qwen-7B") == ALIASES["r1-distill-qwen-7b"]
    assert resolve("LLAMA-3.1-8B") == ALIASES["llama-3.1-8b"]


def test_unknown_alias_passes_through():
    """Unknown short names should pass through unchanged so HF gives
    the natural 'model not found' error instead of a cascadia-specific
    one."""
    assert resolve("not-a-real-alias-xyz") == "not-a-real-alias-xyz"


def test_canonical_hf_id_passes_through():
    """Strings containing '/' are HF repo ids — return as-is even if a
    similarly-named alias exists."""
    assert (
        resolve("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
        == "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    )


def test_local_paths_pass_through():
    assert resolve("/tmp/my-model") == "/tmp/my-model"
    assert resolve("./local-model") == "./local-model"
    assert resolve("C:\\models\\foo") == "C:\\models\\foo"


def test_alias_keys_are_all_lowercase():
    """Keys must be lowercase so `resolve(name.lower())` consistently
    matches user input."""
    for k in ALIASES:
        assert k == k.lower(), f"alias {k!r} has uppercase letters"


def test_no_alias_collides_with_canonical_hf_id():
    """An alias should never contain '/' or start with '.' — that'd
    bypass the alias map via the path-passthrough branch."""
    for k in ALIASES:
        assert "/" not in k, f"alias {k!r} contains '/'"
        assert "\\" not in k, f"alias {k!r} contains '\\\\'"
        assert not k.startswith("."), f"alias {k!r} starts with '.'"


def test_alias_count_in_expected_range():
    """Sanity check — the registry should grow over time. As of PR5
    landing, it has ~20 entries. Bumping the lower bound on each new
    family addition keeps the file from accidentally shrinking."""
    assert len(ALIASES) >= 15
