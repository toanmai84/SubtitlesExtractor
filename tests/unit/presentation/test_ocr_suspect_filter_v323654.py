"""[v3.23.654] Bộ lọc "Nghi ngờ OCR" của trang Chỉnh sửa (tín hiệu hình thái thay cho độ tin cậy ~1,0 của PP-OCRv6)."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from subtitles_extractor.application.services.ocr_suspicion import (
    SUSPECT_FEW_FRAMES,
    SUSPECT_MIXED_SCRIPT,
    SUSPECT_NEAR_DUPLICATE,
    SUSPECT_PROGRESSIVE,
    ocr_suspicions,
)
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.value_objects.confidence import Confidence
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval
from subtitles_extractor.presentation.widgets.subtitle_table_model import (
    SubtitleFilterProxyModel,
    SubtitleTableModel,
)


def _ev(i, text, start, end, frames=20):
    return SubtitleEvent(index=i, text=text, interval=TimeInterval(start, end), confidence=Confidence(0.99),
                         frame_count=frames)


EVENTS = [
    _ev(1, "我", 34.68, 35.32), _ev(2, "我只是", 35.32, 36.9),         # hiện dần (tập 12)
    _ev(3, "200 2a1s wses", 40.0, 40.04, frames=1),                    # rác thật (tập 5): 1 khung
    _ev(4, "嗯不对OVONET", 41.0, 42.0),                                # watermark dính
    _ev(5, "你给我站住", 43.0, 44.0), _ev(6, "你给我站往", 44.1, 45.0),  # biến thể liền kề
    _ev(7, "好", 50.0, 51.0),
]


def test_signals() -> None:
    found = ocr_suspicions(EVENTS)
    assert SUSPECT_PROGRESSIVE in found[0]
    assert SUSPECT_FEW_FRAMES in found[2]
    assert SUSPECT_MIXED_SCRIPT in found[3]
    assert SUSPECT_NEAR_DUPLICATE in found[4] and SUSPECT_NEAR_DUPLICATE in found[5]
    assert 1 not in found and 6 not in found


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_proxy_filter_mode(app) -> None:
    model = SubtitleTableModel()
    model.set_events(EVENTS) if hasattr(model, "set_events") else setattr(model, "events", EVENTS)
    proxy = SubtitleFilterProxyModel()
    proxy.setSourceModel(model)
    proxy.set_filter_params("", SubtitleFilterProxyModel.FILTER_SUSPECT)
    assert proxy.rowCount() == 5
    proxy.set_filter_params("", 0)
    assert proxy.rowCount() == len(EVENTS)


def test_proxy_updates_after_edit_and_row_changes(app) -> None:
    """Rà soát độc lập: cache theo hàng bị dùng lại sau khi sửa/xoá dòng → lọc sai hàng."""
    model = SubtitleTableModel()
    model.set_events([_ev(1, "好", 1.0, 2.0), _ev(2, "走吧", 3.0, 4.0)])
    proxy = SubtitleFilterProxyModel()
    proxy.setSourceModel(model)
    proxy.set_filter_params("", SubtitleFilterProxyModel.FILTER_SUSPECT)
    assert proxy.rowCount() == 0
    model.set_events([_ev(1, "天气不错OVONET", 1.0, 2.0), _ev(2, "走吧", 3.0, 4.0)])
    assert proxy.rowCount() == 1
    model.set_events([_ev(2, "走吧", 3.0, 4.0)])
    assert proxy.rowCount() == 0
