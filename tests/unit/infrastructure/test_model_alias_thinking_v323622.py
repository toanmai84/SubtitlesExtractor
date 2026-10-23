"""[v3.23.622] Bí danh model đã ngừng và mức suy nghĩ hợp lệ theo tài liệu Gemini 10/2026.

* ``gemini-3.5-flash`` → 3.6-flash, ``gemini-3.7-flash`` → 3.8-flash (deprecations.md).
* ``minimal`` là "Not supported (error)" trên 3.8 Flash VÀ Gemini 4; thăm dò thật: bí danh
  3.7-flash trả 400 "Thinking level MINIMAL is not supported" (3/3 lần). Trước đây chỉ
  3.8-flash được chặn → 3.7-flash/gemini-4 tắt suy nghĩ vẫn gửi ``minimal`` và tốn một lời
  gọi 400 mỗi phiên.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from subtitles_extractor.application.services.model_capability import (
    model_capability,
    result_reusable_for,
)
from subtitles_extractor.domain.value_objects.model_alias import (
    canonical_model_id,
    is_deprecated_alias,
)
from subtitles_extractor.infrastructure.translation.thinking_policy import (
    resolve_thinking,
    supports_minimal_level,
)


@pytest.mark.parametrize(("name", "target"), [
    ("gemini-3.5-flash", "gemini-3.6-flash"),
    ("Gemini-3.7-Flash", "gemini-3.8-flash"),
    ("gemini-3.8-flash", "gemini-3.8-flash"),
    ("gemini-3.5-flash-lite", "gemini-3.5-flash-lite"),  # KHÔNG phải bí danh
    ("", ""),
])
def test_canonical_model_id(name: str, target: str) -> None:
    assert canonical_model_id(name) == target


def test_only_two_aliases() -> None:
    assert is_deprecated_alias("gemini-3.5-flash") and is_deprecated_alias("gemini-3.7-flash")
    assert not is_deprecated_alias("gemini-3.5-flash-lite")


@pytest.mark.parametrize(("name", "expected"), [
    ("gemini-3.8-flash", False),
    ("gemini-3.7-flash", False),       # bí danh → 3.8
    ("gemini-4-argon", False),
    ("gemini-4-argon-preview", False),
    ("gemini-3.6-flash", True),
    ("gemini-3.5-flash", True),        # bí danh → 3.6
    ("gemini-3.5-flash-lite", True),
    ("gemini-3.1-flash-lite", True),
    ("gemini-3.8-flash-lite", True),
    ("gemini-3.1-pro-preview", False),
])
def test_supports_minimal_level(name: str, expected: bool) -> None:
    assert supports_minimal_level(name) is expected


@pytest.mark.parametrize("name", ["gemini-3.7-flash", "gemini-4-argon", "gemini-3.8-flash"])
def test_thinking_off_never_sends_minimal_where_rejected(name: str) -> None:
    settings = resolve_thinking(name, enable_thinking=False, thinking_budget=-1,
                                thinking_level="low", minimal_rejected=False)
    assert settings is not None and settings.level == "low"


def test_thinking_off_still_minimal_on_lite() -> None:
    settings = resolve_thinking("gemini-3.1-flash-lite", enable_thinking=False,
                                thinking_budget=-1, thinking_level="low", minimal_rejected=False)
    assert settings is not None and settings.level == "minimal"


def test_alias_results_are_reusable_both_ways() -> None:
    assert model_capability("gemini-3.7-flash") == model_capability("gemini-3.8-flash")
    assert result_reusable_for("gemini-3.7-flash", "gemini-3.8-flash")
    assert result_reusable_for("gemini-3.8-flash", "gemini-3.7-flash")
    assert result_reusable_for("gemini-3.5-flash", "gemini-3.6-flash")
    assert not result_reusable_for("gemini-3.6-flash", "gemini-3.8-flash")  # đổi LÊN: dịch lại


def test_pricing_has_alias_and_argon_rows() -> None:
    data = json.loads(
        (Path(__file__).resolve().parents[3] / "src/subtitles_extractor/data/gemini_pricing.json")
        .read_text(encoding="utf-8"))["models"]
    assert data["gemini-3.5-flash"]["standard"] == data["gemini-3.6-flash"]["standard"]
    assert data["gemini-4-argon"]["standard"] == {"input": 2.0, "output": 10.0}
