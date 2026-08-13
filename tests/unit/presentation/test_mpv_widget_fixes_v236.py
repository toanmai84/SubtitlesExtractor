"""Unit tests cho v2.36 — Fix ROI tỉ lệ + Video shift on page switch.

Bảo vệ các fix:
    * Bug 1: ROI rounding precision — dùng 2-point conversion thay round
      width/height độc lập. Trước đây off-by-1 pixel mỗi conversion.
    * Bug 2: Video player shift khi switch page — showEvent re-sync native
      window via QTimer.singleShot(0).

Tests dùng pure-Python math để verify rounding logic, không cần Qt runtime
cho phần lớn assertions.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MPV_WIDGET_PATH = (
    PROJECT_ROOT / "src" / "subtitles_extractor" / "presentation"
    / "widgets" / "mpv_video_widget.py"
)


@pytest.fixture(scope="module")
def widget_source() -> str:
    return MPV_WIDGET_PATH.read_text(encoding="utf-8")


class TestBug1RoiRoundingPrecision:
    """Bug 1: 3 conversion functions dùng 2-point thay round độc lập."""

    def test_video_to_widget_uses_two_point_conversion(self, widget_source: str) -> None:
        """``_video_to_widget_rect`` phải có pattern x1/x2/y1/y2."""
        idx = widget_source.index("def _video_to_widget_rect(self, roi: Roi)")
        next_idx = widget_source.index("def _video_box_to_widget_rect", idx)
        method_body = widget_source[idx:next_idx]
        # Phải có 4 variables x1/x2/y1/y2.
        assert "x1 = int(round" in method_body
        assert "y1 = int(round" in method_body
        assert "x2 = int(round" in method_body
        assert "y2 = int(round" in method_body
        # Width = x2 - x1, height = y2 - y1.
        assert "x2 - x1" in method_body
        assert "y2 - y1" in method_body

    def test_video_box_to_widget_uses_two_point(self, widget_source: str) -> None:
        idx = widget_source.index("def _video_box_to_widget_rect(")
        next_idx = widget_source.index("def _widget_to_video_roi(", idx)
        method_body = widget_source[idx:next_idx]
        assert "x1 = int(round" in method_body
        assert "x2 - x1" in method_body
        assert "y2 - y1" in method_body

    def test_widget_to_video_uses_two_point(self, widget_source: str) -> None:
        """``_widget_to_video_roi`` ngược về video coords — cũng 2-point."""
        idx = widget_source.index("def _widget_to_video_roi(")
        # Tìm end of method.
        next_class = widget_source.index("__all__ = ", idx)
        method_body = widget_source[idx:next_class]
        assert "x1 = int(round" in method_body
        assert "x2 = int(round" in method_body
        # Phải có `+ 1` correction cho Qt right()/bottom().
        assert "right() + 1" in method_body
        assert "bottom() + 1" in method_body


class TestBug1RoundingMath:
    """Bug 1: Verify math thực tế — 2-point conversion khử off-by-1 pixel."""

    @staticmethod
    def _new_2point_convert(roi_x: int, roi_y: int, roi_w: int, roi_h: int,
                             scale: float, offset_x: float, offset_y: float
                             ) -> tuple[int, int, int, int]:
        """Cách mới (đã fix): 2-point."""
        x1 = round(offset_x + roi_x * scale)
        y1 = round(offset_y + roi_y * scale)
        x2 = round(offset_x + (roi_x + roi_w) * scale)
        y2 = round(offset_y + (roi_y + roi_h) * scale)
        return x1, y1, x2 - x1, y2 - y1

    @staticmethod
    def _old_independent_round(roi_x: int, roi_y: int, roi_w: int, roi_h: int,
                                scale: float, offset_x: float, offset_y: float
                                ) -> tuple[int, int, int, int]:
        """Cách cũ (broken): round độc lập."""
        x = round(offset_x + roi_x * scale)
        y = round(offset_y + roi_y * scale)
        w = round(roi_w * scale)
        h = round(roi_h * scale)
        return x, y, w, h

    def test_2point_preserves_right_edge(self) -> None:
        """2-point đảm bảo right edge widget = round((origin+width) * scale)."""
        # Ví dụ thực tế: video 1920x1080, widget letterboxed 600x338.
        # scale = 600/1920 = 0.3125, offset_x = 0.
        scale = 0.31337  # Khó round.
        offset_x, offset_y = 12.5, 8.3
        roi_x, roi_y, roi_w, roi_h = 100, 50, 320, 30

        new_x, new_y, new_w, new_h = self._new_2point_convert(
            roi_x, roi_y, roi_w, roi_h, scale, offset_x, offset_y
        )
        old_x, old_y, old_w, old_h = self._old_independent_round(
            roi_x, roi_y, roi_w, roi_h, scale, offset_x, offset_y
        )

        # Right edge "đúng" tính từ 2-point math.
        correct_right = round(offset_x + (roi_x + roi_w) * scale)
        new_right = new_x + new_w
        old_right = old_x + old_w

        # 2-point: đúng.
        assert new_right == correct_right
        # Old: có thể off-by-1 hoặc đúng tuỳ giá trị.
        # Khẳng định: ở case này 2-point khác old by 1 hoặc bằng.
        assert abs(new_right - old_right) <= 1, (
            f"new_right={new_right} vs old_right={old_right}"
        )

    def test_2point_off_by_one_case(self) -> None:
        """Case cụ thể chứng minh old method off-by-1, new method đúng."""
        # Chọn params sao cho old method off-by-1.
        scale = 0.3
        offset_x, offset_y = 0.5, 0.5
        # roi_x = 1 → x_widget = round(0.5 + 0.3) = round(0.8) = 1
        # roi_w = 7 → w_widget = round(2.1) = 2 → right = 3
        # Correct: round(0.5 + 8 * 0.3) = round(2.9) = 3 → match!
        # Hmm, trùng. Tìm case khác:
        # roi_x = 2, roi_w = 7:
        # old: x = round(0.5 + 0.6) = round(1.1) = 1; w = round(2.1) = 2 → right = 3
        # new: x1 = 1; x2 = round(0.5 + 9*0.3) = round(3.2) = 3 → w = 2, right = 3
        # OK match.
        # roi_x = 1, roi_w = 6:
        # old: x = round(0.8) = 1, w = round(1.8) = 2 → right = 3
        # new: x1 = 1, x2 = round(0.5 + 2.1) = round(2.6) = 3 → w = 2, right = 3
        # OK match.
        # Try: scale = 0.7, offset = 0.4, roi_x=3, roi_w=8:
        # old: x = round(0.4 + 2.1) = round(2.5) = 2 (round-half-to-even); w = round(5.6) = 6 → right = 8
        # new: x1 = round(2.5) = 2; x2 = round(0.4 + 11*0.7) = round(8.1) = 8 → w = 6, right = 8
        # Match.
        # Try: scale = 0.33, offset = 0.5, roi_x=2, roi_w=7:
        # old: x = round(0.5 + 0.66) = round(1.16) = 1; w = round(2.31) = 2 → right = 3
        # new: x1 = round(1.16) = 1; x2 = round(0.5 + 9*0.33) = round(3.47) = 3 → w = 2, right = 3
        # Match.

        # Find a real off-by-one case via search.
        found_diff = False
        for scale in (0.31, 0.37, 0.41, 0.43, 0.59, 0.71, 0.83):
            for offset_x in (0.0, 0.25, 0.5, 0.7):
                for roi_x in range(1, 20):
                    for roi_w in range(1, 20):
                        old = self._old_independent_round(
                            roi_x, 0, roi_w, 1, scale, offset_x, 0
                        )
                        new = self._new_2point_convert(
                            roi_x, 0, roi_w, 1, scale, offset_x, 0
                        )
                        if old[2] != new[2]:  # width differs
                            found_diff = True
                            # New phải khớp với mathematical truth.
                            correct_x1 = round(offset_x + roi_x * scale)
                            correct_x2 = round(offset_x + (roi_x + roi_w) * scale)
                            correct_w = correct_x2 - correct_x1
                            assert new[2] == correct_w, (
                                f"scale={scale}, offset={offset_x}, "
                                f"roi=({roi_x},{roi_w}): new_w={new[2]}, correct={correct_w}"
                            )
        assert found_diff, "Không tìm thấy case off-by-1 — test không hiệu lực!"


class TestBug2ShowEventRecreatesGeometry:
    """Bug 2: showEvent re-sync native window via QTimer.singleShot."""

    def test_show_event_calls_layout_activate(self, widget_source: str) -> None:
        # Tìm theo prefix — không phụ thuộc type annotation cụ thể.
        idx = widget_source.index("def showEvent(self, event")
        next_idx = widget_source.index("def _post_show_native_sync(", idx)
        method_body = widget_source[idx:next_idx]
        # Phải có layout().activate() để force re-compute.
        assert "self.layout().activate()" in method_body

    def test_show_event_uses_qtimer_singleshot(self, widget_source: str) -> None:
        """Phải có QTimer.singleShot(0, ...) defer 1 event loop tick."""
        idx = widget_source.index("def showEvent(self, event")
        next_idx = widget_source.index("def _post_show_native_sync(", idx)
        method_body = widget_source[idx:next_idx]
        assert "QTimer.singleShot(0" in method_body

    def test_post_show_native_sync_method_exists(self, widget_source: str) -> None:
        """Method ``_post_show_native_sync`` tồn tại để defer sync."""
        assert "def _post_show_native_sync(self)" in widget_source

    def test_post_show_does_not_force_geometry(self, widget_source: str) -> None:
        """v2.36: KHÔNG force setGeometry trên container (conflict layout)."""
        idx = widget_source.index("def _post_show_native_sync(self)")
        # Tìm end (next method or class boundary).
        try:
            next_idx = widget_source.index("def hideEvent(", idx)
        except ValueError:
            next_idx = idx + 3000
        method_body = widget_source[idx:next_idx]
        # KHÔNG có direct setGeometry trên container (chỉ updateGeometry).
        assert "self._video_container.setGeometry(" not in method_body
        # CÓ updateGeometry (gentler).
        assert "updateGeometry()" in method_body

    def test_post_show_invalidates_layout(self, widget_source: str) -> None:
        """Phải gọi layout.invalidate() để Qt re-compute container."""
        idx = widget_source.index("def _post_show_native_sync(self)")
        try:
            next_idx = widget_source.index("def hideEvent(", idx)
        except ValueError:
            next_idx = idx + 3000
        method_body = widget_source[idx:next_idx]
        assert "invalidate()" in method_body

    def test_hide_event_keeps_container_visible(self, widget_source: str) -> None:
        """v2.36: hideEvent KHÔNG hide container để giữ native handle stable."""
        # Tìm theo prefix — không phụ thuộc type annotation cụ thể.
        idx = widget_source.index("def hideEvent(self, event")
        # Method body until next method.
        try:
            next_idx = widget_source.index("def send_mpv_command(", idx)
        except ValueError:
            next_idx = idx + 2000
        method_body = widget_source[idx:next_idx]
        # KHÔNG có ``_video_container.hide()`` (sẽ phá native handle).
        assert "self._video_container.hide(" not in method_body
        # CÓ pause MPV (tiết kiệm CPU).
        assert "self._player.pause()" in method_body


class TestAstParses:
    def test_module_parses(self, widget_source: str) -> None:
        ast.parse(widget_source)
