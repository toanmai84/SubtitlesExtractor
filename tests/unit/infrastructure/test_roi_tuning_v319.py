"""Test [v3.19]: kháng logo "chết", phụ đề nhiều dòng (hysteresis), độ nhạy heatmap.

Ba lỗi tái hiện bằng synthetic từ phản hồi thực tế của người dùng:
    1. Logo/watermark tĩnh suốt phim (frame-presence 100%) chiếm mất vị trí cụm
       phụ đề CHÍNH vì điểm chọn primary chỉ dựa frame_count thuần.
    2. Dòng THỨ HAI của phụ đề nhiều dòng (chỉ xuất hiện ~1/3 số câu) bị Band
       Refiner cắt vì mật độ < keep_ratio × peak.
    3. Vùng chữ thông tin xuất hiện ngắn (vài giây) bị ngưỡng heatmap lọc mất.

Khắc phục: (1) điểm tổng hợp ``frame_count × bề_rộng × vị_trí``; (2) hysteresis
2 ngưỡng ``band_extend_ratio`` (mặc định = keep → hành vi cũ, người dùng hạ khi
cần); (3) expose ``heatmap_threshold_multiplier`` qua Settings (giảm = nhạy hơn).
"""

from __future__ import annotations

from subtitles_extractor.infrastructure.video.bbox_analyzer import (
    BBoxAnalyzer,
    RawBBox,
)
from subtitles_extractor.infrastructure.video.roi_selection import (
    select_primary_subtitle_cluster,
)

_FRAME_W, _FRAME_H = 720, 1280


def _box(x0: float, y0: float, x1: float, y1: float, frame: int, conf: float = 0.96) -> RawBBox:
    return RawBBox(coord_x_min=x0, coord_y_min=y0, coord_x_max=x1, coord_y_max=y1,
                   confidence=conf, frame_idx=frame, timestamp_sec=frame * 0.04)


class TestDeadLogoResistance:
    def test_subtitle_beats_always_on_corner_logo(self) -> None:
        # Logo nhỏ góc trên hiện 100% phim; phụ đề rộng dưới đáy hiện 70%.
        boxes = [_box(560, 40, 700, 86, f, 0.92) for f in range(3000)]
        boxes += [_box(160, 1100, 560, 1160, f, 0.97) for f in range(3000) if f % 10 < 7]
        clusters = BBoxAnalyzer(frame_width=_FRAME_W, frame_height=_FRAME_H, padding=0).analyze(boxes)
        primary = select_primary_subtitle_cluster(clusters, _FRAME_H)
        assert primary is not None
        assert primary.coord_y_min > 1000, "Logo chết chiếm mất primary thay vì phụ đề"


class TestMultilineHysteresis:
    @staticmethod
    def _two_line_boxes() -> list[RawBBox]:
        boxes = [_box(160, 1080, 560, 1140, f, 0.97) for f in range(2000)]
        boxes += [_box(260, 1140, 460, 1200, f, 0.95) for f in range(2000) if (f // 60) % 3 == 0]
        return boxes

    def test_default_behaviour_unchanged(self) -> None:
        # Mặc định extend = keep → hành vi cũ (dải khít quanh dòng chính).
        primary = max(
            BBoxAnalyzer(frame_width=_FRAME_W, frame_height=_FRAME_H, padding=0).analyze(self._two_line_boxes()),
            key=lambda c: c.frame_count,
        )
        assert primary.coord_y_max <= 1150

    def test_lowered_extend_keeps_second_line(self) -> None:
        # Người dùng hạ extend (qua Settings) → dòng 2 (33% câu) được giữ trọn.
        primary = max(
            BBoxAnalyzer(
                frame_width=_FRAME_W, frame_height=_FRAME_H, padding=0, band_extend_ratio=0.25
            ).analyze(self._two_line_boxes()),
            key=lambda c: c.frame_count,
        )
        assert primary.coord_y_max >= 1195, "Dòng thứ hai vẫn bị cắt dù đã hạ extend"

    def test_extend_clamped_to_keep(self) -> None:
        analyzer = BBoxAnalyzer(
            frame_width=_FRAME_W, frame_height=_FRAME_H,
            band_keep_ratio=0.40, band_extend_ratio=0.90,
        )
        assert analyzer.band_extend_ratio <= analyzer.band_keep_ratio


class TestHeatmapSensitivity:
    def test_default_recovers_short_info_text_after_minconf_fix(self) -> None:
        # Chữ thông tin chỉ hiện 3s/120s, conf 0.90 — trước v3.19 bị "ngưỡng động"
        # (median quần thể 0.97 − 0.05 = 0.92) giết sạch ngay từ _filter_anomalies.
        # Sau fix (ngưỡng động không vượt Lệnh bài miễn tử) → bắt được Ở MẶC ĐỊNH.
        boxes = [_box(160, 1100, 560, 1160, f, 0.97) for f in range(3000) if f % 10 < 7]
        boxes += [_box(120, 320, 600, 450, f, 0.90) for f in range(500, 575)]
        clusters = BBoxAnalyzer(
            frame_width=_FRAME_W, frame_height=_FRAME_H, padding=0
        ).analyze(boxes)
        assert any(c.coord_y_max < 600 for c in clusters), (
            "Vùng chữ thông tin ngắn vẫn bị lọc mất ở cấu hình mặc định"
        )


class TestRoiSettingsExposed:
    def test_settings_fields_exist_with_safe_defaults(self) -> None:
        from subtitles_extractor.infrastructure.settings.application_settings import RoiSettings

        settings = RoiSettings()
        assert settings.auto_enable_band_refinement is True
        assert settings.auto_band_keep_ratio == 0.50
        assert settings.auto_band_extend_ratio == 0.50   # mặc định = keep (an toàn)
        assert settings.auto_bottom_padding_factor == 1.6
        assert settings.auto_sensitivity_multiplier == 1.0
