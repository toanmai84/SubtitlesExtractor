"""Test [v3.23.58] sửa bug TM: hiệu năng truy hồi + overview không phình lũy tiến."""

from __future__ import annotations

import time

import pytest

pytest.importorskip("PyQt6")

from PySide6.QtWidgets import QApplication

from subtitles_extractor.composition.bootstrap import bootstrap_for_gui
from subtitles_extractor.application.services.translation_memory import (
    TranslationMemoryEntry as E,
    derive_series_key,
)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _vm(c):
    from subtitles_extractor.presentation.view_models.translate_page_view_model import (
        TranslatePageViewModel,
    )
    return TranslatePageViewModel(c)


def test_retrieve_bounded_performance(app) -> None:
    c = bootstrap_for_gui()
    vm = _vm(c)
    key = derive_series_key("/data/Perf58Test/EP01.mp4")
    c.translation_memory_store.add_entries(
        key, [E(f"句子内容测试{i}", f"Câu {i}") for i in range(3000)]
    )
    src = [f"句子内容测试{i}" for i in range(5000)]
    start = time.time()
    block = vm.retrieve_memory_for_lines("/data/Perf58Test/EP02.mp4", src)
    elapsed = time.time() - start
    # Sau khi giới hạn, phải nhanh (< 2s kể cả TM/nguồn lớn).
    assert elapsed < 2.0
    assert isinstance(block, str)
    c.translation_memory_store.clear_series(key)


def test_overview_not_inflated_across_episodes(app) -> None:
    c = bootstrap_for_gui()
    vm = _vm(c)
    # Tập 1
    vm.accumulate_series_context(
        "/data/Inflate58/EP01.mp4", "林恒 => Lâm Hằng", "林恒", "Phim tu tiên"
    )
    ctx1 = vm.restore_series_context("/data/Inflate58/EP01.mp4")
    assert "BỘ NHỚ DỊCH" not in ctx1.overview
    len1 = len(ctx1.overview)
    # Tập 2, 3 — overview không được phình thêm khối TM.
    for ep in ("EP02", "EP03"):
        vm.accumulate_series_context(
            f"/data/Inflate58/{ep}.mp4", "", "", "Phim tu tiên"
        )
    ctx3 = vm.restore_series_context("/data/Inflate58/EP03.mp4")
    assert "BỘ NHỚ DỊCH" not in ctx3.overview
    assert len(ctx3.overview) == len1  # giữ nguyên độ dài
    c.translation_memory_store.clear_series("Inflate58")
