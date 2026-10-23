"""[v3.23.643] 3.1-flash-lite và 3.5-flash-lite ngang hạng: không vứt bản dịch đã hoàn tất khi đổi qua lại."""

from __future__ import annotations

from subtitles_extractor.application.services.model_capability import (
    model_capability,
    result_reusable_for,
)


def test_measured_equal_lite_models_reuse_each_other() -> None:
    assert result_reusable_for("gemini-3.1-flash-lite", "gemini-3.5-flash-lite")
    assert result_reusable_for("gemini-3.5-flash-lite", "gemini-3.1-flash-lite")


def test_lite_still_does_not_replace_full_flash() -> None:
    assert not result_reusable_for("gemini-3.1-flash-lite", "gemini-3.8-flash")
    assert result_reusable_for("gemini-3.8-flash", "gemini-3.5-flash-lite")


def test_unmeasured_generations_keep_the_version_order() -> None:
    assert not result_reusable_for("gemini-3.1-flash-lite", "gemini-4-flash-lite")
    assert model_capability("gemini-3.1-flash-lite") == model_capability("gemini-3.5-flash-lite")
