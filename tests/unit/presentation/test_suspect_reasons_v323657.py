"""[v3.23.657] Lọc "Nghi ngờ OCR" hiện LÝ DO ở cột cảnh báo + tooltip (trước chỉ lọc, không biết vì sao)."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt

from subtitles_extractor.presentation.widgets.subtitle_table_model import (
    COL_TEXT,
    COL_WARN,
    SubtitleFilterProxyModel,
    SubtitleTableModel,
)
from tests.unit.presentation.test_ocr_suspect_filter_v323654 import EVENTS, app  # noqa: F401

LABELS = {"few_frames": "ít khung", "progressive": "hiện dần", "mixed_script": "lẫn chữ Latin/số",
          "near_duplicate": "gần trùng câu kề"}


def _proxy(mode: int) -> SubtitleFilterProxyModel:
    model = SubtitleTableModel()
    model.set_events(EVENTS)
    proxy = SubtitleFilterProxyModel()
    proxy.setSourceModel(model)
    proxy.set_reason_labeler(lambda reason: LABELS.get(reason, reason))
    proxy.set_filter_params("", mode)
    proxy._keep_model = model  # giữ model sống cùng proxy trong test
    return proxy


def test_warn_column_shows_reasons_in_suspect_mode(app) -> None:  # noqa: F811
    proxy = _proxy(SubtitleFilterProxyModel.FILTER_SUSPECT)
    texts = {proxy.index(r, COL_TEXT).data(): proxy.index(r, COL_WARN).data() for r in range(proxy.rowCount())}
    assert "hiện dần" in texts["我"]
    assert "ít khung" in texts["200 2a1s wses"]
    assert "lẫn chữ Latin/số" in texts["嗯不对OVONET"]
    assert "gần trùng câu kề" in texts["你给我站住"]
    tip = proxy.index(0, COL_WARN).data(Qt.ItemDataRole.ToolTipRole)
    assert tip and "hiện dần" in tip


def test_other_modes_unchanged(app) -> None:  # noqa: F811
    proxy = _proxy(0)
    for row in range(proxy.rowCount()):
        value = proxy.index(row, COL_WARN).data() or ""
        assert "🔎" not in value
