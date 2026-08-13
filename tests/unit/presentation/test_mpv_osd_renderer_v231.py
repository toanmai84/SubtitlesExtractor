"""Unit tests cho v2.31 — libmpv OSD overlay rendering.

LEGACY TEST — SKIPPED tại v3.3+ vì refactor đã xoá ``_build_ass_rectangle``
và rename ``_OSD_ID_OCR_BOXES`` → ``_OSD_ID_OCR_START``. Test chức năng OSD
được phủ bởi ``test_mpv_widget_fixes_v236.py`` ở phiên bản mới hơn.

Để mở lại test này, cần refactor cùng API mới (set_raw_ocr_chunks thay vì
_build_ass_rectangle).
"""

from __future__ import annotations

import pytest

# Skip toàn bộ module — API đã thay đổi ở v3.3.
pytest.skip(
    "Legacy v2.31 test — API _build_ass_rectangle đã xoá ở v3.3 refactor. "
    "Chức năng OSD được test bởi test_mpv_widget_fixes_v236.py",
    allow_module_level=True,
)


def _make_test_roi(x: int = 100, y: int = 200, w: int = 300, h: int = 50) -> Roi:
    return Roi(
        x=x, y=y, width=w, height=h,
        alignment=TextAlignment.CENTER,
        orientation=TextOrientation.HORIZONTAL,
    )


class TestBuildAssRectangle:
    """v2.31: ``_build_ass_rectangle`` tạo ASS event string đúng format."""

    def test_basic_rectangle_contains_all_tags(self) -> None:
        ass = _build_ass_rectangle(100, 200, 300, 50, _STYLE_COMMITTED)
        # Anchor top-left.
        assert "\\an7" in ass
        # Position tag với toạ độ chính xác.
        assert "\\pos(100,200)" in ass
        # Border width = 3 (committed style).
        assert "\\bord3" in ass
        # Fill + border color BGR.
        assert "\\1c&H78C800&" in ass
        assert "\\3c&H78C800&" in ass
        # Drawing mode bounds.
        assert "\\p1}" in ass
        assert "{\\p0}" in ass

    def test_drawing_path_describes_rectangle(self) -> None:
        """Path 'm 0 0 l W 0 l W H l 0 H' = 4 góc rectangle."""
        ass = _build_ass_rectangle(0, 0, 250, 80, _STYLE_COMMITTED)
        assert "m 0 0" in ass
        assert "l 250 0" in ass
        assert "l 250 80" in ass
        assert "l 0 80" in ass

    def test_secondary_style_uses_violet_bgr(self) -> None:
        ass = _build_ass_rectangle(0, 0, 100, 100, _STYLE_SECONDARY)
        # Style secondary: BGR DC82C8.
        assert "\\1c&HDC82C8&" in ass
        assert "\\bord2" in ass

    def test_ocr_style_uses_yellow_bgr_thin_border(self) -> None:
        ass = _build_ass_rectangle(0, 0, 100, 100, _STYLE_OCR)
        # Style OCR: BGR 32DCFF (vàng đậm) + border mỏng 1px.
        assert "\\1c&H32DCFF&" in ass
        assert "\\bord1" in ass

    def test_alpha_encoding_in_hex(self) -> None:
        """Alpha hex chỉ 2 chữ (00-FF)."""
        ass = _build_ass_rectangle(0, 0, 100, 100, _STYLE_COMMITTED)
        # Committed: fill_alpha C8, border_alpha 00.
        assert "\\1a&HC8&" in ass
        assert "\\3a&H00&" in ass


class TestStylesAreCorrect:
    """v2.31: 3 style constants có giá trị đúng."""

    def test_committed_style_is_green_opaque_border(self) -> None:
        # RGB(0,200,120) → BGR(78,C8,00). Border alpha 00 = opaque.
        assert _STYLE_COMMITTED.fill_color_bgr == "78C800"
        assert _STYLE_COMMITTED.border_alpha_hex == "00"
        assert _STYLE_COMMITTED.border_width_px == 3

    def test_secondary_style_is_violet_dotted(self) -> None:
        # RGB(200,130,220) → BGR(DC,82,C8).
        assert _STYLE_SECONDARY.fill_color_bgr == "DC82C8"
        assert _STYLE_SECONDARY.border_width_px == 2

    def test_ocr_style_is_yellow_thin(self) -> None:
        # RGB(255,220,50) → BGR(32,DC,FF).
        assert _STYLE_OCR.fill_color_bgr == "32DCFF"
        assert _STYLE_OCR.border_width_px == 1


class TestOsdIdsAreDistinct:
    """v2.31: 3 overlay layer có ID riêng, không trùng."""

    def test_ids_in_valid_mpv_range(self) -> None:
        # libmpv OSD overlay ID phải >= 1.
        assert _OSD_ID_OCR_BOXES >= 1
        assert _OSD_ID_SECONDARY_ROIS >= 1
        assert _OSD_ID_COMMITTED_ROI >= 1

    def test_ids_are_unique(self) -> None:
        ids = {_OSD_ID_OCR_BOXES, _OSD_ID_SECONDARY_ROIS, _OSD_ID_COMMITTED_ROI}
        assert len(ids) == 3, "3 overlay layer phải có ID riêng biệt"


class TestMpvOsdRendererSetCommittedRoi:
    """v2.31: set_committed_roi gọi đúng osd-overlay command."""

    def _make_player_mock(self) -> MagicMock:
        player_mock = MagicMock()
        player_mock.send_command = MagicMock(return_value=True)
        return player_mock

    def test_set_committed_roi_sends_osd_overlay_command(self) -> None:
        player_mock = self._make_player_mock()
        renderer = MpvOsdRenderer(player_mock)
        renderer.set_video_size(1920, 1080)

        roi = _make_test_roi(x=100, y=200, w=300, h=50)
        renderer.set_committed_roi(roi)

        player_mock.send_command.assert_called_once()
        call_args = player_mock.send_command.call_args.args
        # send_command(*args) → args là tuple positional.
        assert call_args[0] == "osd-overlay"
        assert call_args[1] == _OSD_ID_COMMITTED_ROI
        assert call_args[2] == "ass-events"
        # data string chứa ASS rectangle.
        assert "\\pos(100,200)" in call_args[3]
        # res_x, res_y = video size.
        assert call_args[4] == 1920
        assert call_args[5] == 1080

    def test_set_committed_roi_none_clears_overlay(self) -> None:
        player_mock = self._make_player_mock()
        renderer = MpvOsdRenderer(player_mock)
        renderer.set_video_size(1920, 1080)

        # Render rồi clear.
        renderer.set_committed_roi(_make_test_roi())
        player_mock.send_command.reset_mock()

        renderer.set_committed_roi(None)
        # Clear gọi command với format 'none'.
        player_mock.send_command.assert_called_once()
        call_args = player_mock.send_command.call_args.args
        assert call_args[0] == "osd-overlay"
        assert call_args[1] == _OSD_ID_COMMITTED_ROI
        assert call_args[2] == "none"

    def test_skip_render_when_video_size_not_set(self) -> None:
        """Nếu chưa set_video_size, OSD command không được gọi."""
        player_mock = self._make_player_mock()
        renderer = MpvOsdRenderer(player_mock)
        # KHÔNG gọi set_video_size.

        renderer.set_committed_roi(_make_test_roi())
        player_mock.send_command.assert_not_called()


class TestMpvOsdRendererSecondaryRois:
    """v2.31: set_secondary_rois render multi-rectangle qua 1 OSD layer."""

    def test_multiple_rois_joined_with_ass_newline(self) -> None:
        player_mock = MagicMock()
        player_mock.send_command = MagicMock(return_value=True)
        renderer = MpvOsdRenderer(player_mock)
        renderer.set_video_size(1920, 1080)

        rois = [
            _make_test_roi(x=100, y=200, w=300, h=50),
            _make_test_roi(x=500, y=400, w=400, h=60),
        ]
        renderer.set_secondary_rois(rois)

        player_mock.send_command.assert_called_once()
        call_args = player_mock.send_command.call_args.args
        assert call_args[1] == _OSD_ID_SECONDARY_ROIS
        # 2 rectangles join bằng \N (ASS newline).
        ass_data = call_args[3]
        assert "\\N" in ass_data
        assert "\\pos(100,200)" in ass_data
        assert "\\pos(500,400)" in ass_data

    def test_empty_secondary_list_clears_overlay(self) -> None:
        player_mock = MagicMock()
        player_mock.send_command = MagicMock(return_value=True)
        renderer = MpvOsdRenderer(player_mock)
        renderer.set_video_size(1920, 1080)

        # Set rồi clear.
        renderer.set_secondary_rois([_make_test_roi()])
        player_mock.send_command.reset_mock()

        renderer.set_secondary_rois([])
        # Clear command.
        player_mock.send_command.assert_called_once()
        assert player_mock.send_command.call_args.args[2] == "none"


class TestMpvOsdRendererOcrOverlay:
    """v2.31: set_ocr_overlay với visibility flag."""

    def test_visible_true_renders_boxes(self) -> None:
        player_mock = MagicMock()
        player_mock.send_command = MagicMock(return_value=True)
        renderer = MpvOsdRenderer(player_mock)
        renderer.set_video_size(1920, 1080)

        boxes = [(50, 100, 200, 30), (60, 200, 220, 30)]
        renderer.set_ocr_overlay(boxes, visible=True)

        player_mock.send_command.assert_called_once()
        call_args = player_mock.send_command.call_args.args
        assert call_args[1] == _OSD_ID_OCR_BOXES
        ass_data = call_args[3]
        assert "\\pos(50,100)" in ass_data
        assert "\\pos(60,200)" in ass_data

    def test_visible_false_clears_overlay_even_if_boxes_provided(self) -> None:
        player_mock = MagicMock()
        player_mock.send_command = MagicMock(return_value=True)
        renderer = MpvOsdRenderer(player_mock)
        renderer.set_video_size(1920, 1080)

        renderer.set_ocr_overlay([(0, 0, 100, 100)], visible=False)

        player_mock.send_command.assert_called_once()
        assert player_mock.send_command.call_args.args[2] == "none"


class TestMpvOsdRendererClearAll:
    """v2.31: clear_all xóa toàn bộ active overlay."""

    def test_clear_all_clears_each_active_layer(self) -> None:
        player_mock = MagicMock()
        player_mock.send_command = MagicMock(return_value=True)
        renderer = MpvOsdRenderer(player_mock)
        renderer.set_video_size(1920, 1080)

        # Render cả 3 layer.
        renderer.set_committed_roi(_make_test_roi())
        renderer.set_secondary_rois([_make_test_roi(x=10)])
        renderer.set_ocr_overlay([(0, 0, 50, 50)], visible=True)

        player_mock.send_command.reset_mock()
        renderer.clear_all()

        # Clear gọi 3 lần (1 cho mỗi layer active).
        assert player_mock.send_command.call_count == 3
        # Mọi call đều với format 'none'.
        for call in player_mock.send_command.call_args_list:
            assert call.args[2] == "none"

    def test_clear_all_idempotent_when_nothing_active(self) -> None:
        player_mock = MagicMock()
        player_mock.send_command = MagicMock(return_value=True)
        renderer = MpvOsdRenderer(player_mock)
        renderer.set_video_size(1920, 1080)

        renderer.clear_all()
        player_mock.send_command.assert_not_called()


class TestMpvOsdRendererLayerIsolation:
    """v2.31: Mỗi layer ID độc lập, update không ảnh hưởng layer khác."""

    def test_updating_committed_does_not_clear_secondary(self) -> None:
        player_mock = MagicMock()
        player_mock.send_command = MagicMock(return_value=True)
        renderer = MpvOsdRenderer(player_mock)
        renderer.set_video_size(1920, 1080)

        renderer.set_secondary_rois([_make_test_roi(x=10)])
        renderer.set_committed_roi(_make_test_roi(x=100))

        # Cả 2 layer phải còn active.
        assert _OSD_ID_SECONDARY_ROIS in renderer._active_overlay_ids
        assert _OSD_ID_COMMITTED_ROI in renderer._active_overlay_ids

    def test_clear_committed_keeps_secondary_active(self) -> None:
        player_mock = MagicMock()
        player_mock.send_command = MagicMock(return_value=True)
        renderer = MpvOsdRenderer(player_mock)
        renderer.set_video_size(1920, 1080)

        renderer.set_secondary_rois([_make_test_roi(x=10)])
        renderer.set_committed_roi(_make_test_roi(x=100))
        renderer.set_committed_roi(None)  # Clear committed.

        # Secondary vẫn còn.
        assert _OSD_ID_SECONDARY_ROIS in renderer._active_overlay_ids
        assert _OSD_ID_COMMITTED_ROI not in renderer._active_overlay_ids
