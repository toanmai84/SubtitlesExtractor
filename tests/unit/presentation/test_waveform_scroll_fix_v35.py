"""Test cho fix bug v3.5: waveform scroll/zoom kẹt 60s.

Bug gốc: ``self._total_video_duration_sec or 60.0`` trong ``wheelEvent``
khi ``_total_video_duration_sec = 0`` (chưa load) hoặc nếu race condition
xảy ra, Python truthiness coi 0 là falsy → fallback về 60s ⇒ waveform
KHÔNG thể scroll/zoom quá 60s.

Fix v3.5: Thêm ``_effective_duration_sec()`` central:
    1. Ưu tiên ``_total_video_duration_sec`` nếu > 0.
    2. Suy ra từ audio samples đã load.
    3. Fallback 60s chỉ khi không có gì.

Test không cần Qt (chỉ test logic _effective_duration_sec) — dùng
unittest.mock thay vì pytest-qt.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Skip nếu Qt không có (chỉ chạy được trên môi trường có PyQt6).
PyQt6 = pytest.importorskip("PyQt6", reason="PyQt6 required for widget test")

# Đảm bảo có QApplication trước khi import widget.
from PySide6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

from subtitles_extractor.presentation.widgets.audio_waveform_widget import (  # noqa: E402
    AudioWaveformWidget,
)


@pytest.fixture
def widget():
    """Fixture tạo widget không gắn vào GUI."""
    w = AudioWaveformWidget()
    yield w
    w.close_widget()
    w.deleteLater()


class TestEffectiveDurationFix:
    """Verify bug fix v3.5 — waveform không còn kẹt 60s."""

    def test_initial_state_no_duration_no_samples_returns_fallback(
        self, widget: AudioWaveformWidget,
    ) -> None:
        """Khi widget mới khởi tạo, chưa có duration và samples → fallback 60s."""
        assert widget._effective_duration_sec() == 60.0

    def test_with_total_duration_returns_total_duration(
        self, widget: AudioWaveformWidget,
    ) -> None:
        """Khi đã set_duration(300) → trả về 300 (KHÔNG fallback 60)."""
        widget.set_duration(300.0)
        assert widget._effective_duration_sec() == 300.0

    def test_long_video_can_scroll_full_duration(
        self, widget: AudioWaveformWidget,
    ) -> None:
        """Bug repro: video 600s, scroll phải đến gần cuối, KHÔNG kẹt ở 60s."""
        widget.set_duration(600.0)
        widget._view_timeline_duration_sec = 30.0  # zoom view 30s

        # Cuộn vượt hết video — đích cuộn (endValue) phải clamp tại biên phải.
        widget._smooth_scroll_by(10_000.0)
        max_start = 600.0 - 30.0
        assert float(widget._scroll_anim.endValue()) == pytest.approx(max_start)

    def test_zoom_out_to_full_video_when_no_duration_set(
        self, widget: AudioWaveformWidget,
    ) -> None:
        """Khi có audio samples nhưng chưa có duration — vẫn scroll được tới hết."""
        # Giả lập 200s audio = 200 * 16000 = 3.2M samples.
        widget._recorded_samples_array = np.zeros(200 * 16000, dtype=np.float32)
        widget._audio_sample_rate = 16000

        # _total_video_duration_sec vẫn = 0, nhưng _effective_duration sẽ suy
        # từ samples: 200s.
        assert widget._effective_duration_sec() == pytest.approx(200.0, abs=0.1)

    def test_set_duration_zero_does_not_override_existing(
        self, widget: AudioWaveformWidget,
    ) -> None:
        """set_duration(0) hoặc âm KHÔNG được override duration đã đúng."""
        widget.set_duration(300.0)
        widget.set_duration(0.0)  # invalid
        widget.set_duration(-10.0)  # invalid

        assert widget._effective_duration_sec() == 300.0

    def test_set_duration_resets_view_if_out_of_bounds(
        self, widget: AudioWaveformWidget,
    ) -> None:
        """Khi đổi sang video ngắn hơn, view phải tự reset không kẹt ngoài biên."""
        widget.set_duration(600.0)
        widget._view_timeline_start_sec = 500.0
        widget._view_timeline_duration_sec = 30.0

        # Đổi sang video 60s.
        widget.set_duration(60.0)

        assert widget._view_timeline_start_sec <= 60.0
        assert widget._view_timeline_duration_sec <= 60.0


class TestSmoothScrollClamp:
    """[v3.20] Cập nhật theo API hiện tại: pan = _smooth_scroll_by (có animation).

    Logic clamp nằm ở endValue của animation; ta kiểm endValue thay vì đợi anim.
    """

    def test_scroll_right_clamped_at_end(self, widget: AudioWaveformWidget) -> None:
        widget.set_duration(120.0)
        widget._view_timeline_duration_sec = 30.0
        widget._view_timeline_start_sec = 80.0

        widget._smooth_scroll_by(300.0)  # vượt biên phải
        # max_start = 120 - 30 = 90
        assert float(widget._scroll_anim.endValue()) == pytest.approx(90.0)

    def test_scroll_left_clamped_at_zero(self, widget: AudioWaveformWidget) -> None:
        widget.set_duration(120.0)
        widget._view_timeline_duration_sec = 30.0
        widget._view_timeline_start_sec = 10.0

        widget._smooth_scroll_by(-150.0)  # ngược về âm
        assert float(widget._scroll_anim.endValue()) == pytest.approx(0.0)


class TestZoomAtCenter:
    """Test method _zoom_at_center (signature: factor, safe_bounds)."""

    def test_zoom_in_preserves_center(self, widget: AudioWaveformWidget) -> None:
        widget.set_duration(600.0)
        widget._view_timeline_start_sec = 100.0
        widget._view_timeline_duration_sec = 60.0
        # Center ban đầu = 100 + 30 = 130

        widget._zoom_at_center(0.5, 600.0)

        new_center = widget._view_timeline_start_sec + widget._view_timeline_duration_sec / 2.0
        assert new_center == pytest.approx(130.0, abs=0.01)
        assert widget._view_timeline_duration_sec == 30.0  # zoom in 2×

    def test_zoom_out_clamped_to_full_video(self, widget: AudioWaveformWidget) -> None:
        widget.set_duration(120.0)
        widget._view_timeline_start_sec = 0.0
        widget._view_timeline_duration_sec = 30.0

        widget._zoom_at_center(10.0, 120.0)

        # Không thể zoom out > toàn bộ video.
        assert widget._view_timeline_duration_sec == 120.0
