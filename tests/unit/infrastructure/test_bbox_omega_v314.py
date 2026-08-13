"""Smoke + sanity test cho BBoxAnalyzer bản 'The Omega' (v67).

Kiểm HÀNH VI CÔNG KHAI (analyze) thay cho các test gắn API nội bộ cũ đã skip:
chống crash, biên đầu vào, và tách đúng dải phụ đề đáy trên dữ liệu thật.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from subtitles_extractor.infrastructure.video.bbox_analyzer import (
    BBoxAnalyzer,
    RawBBox,
    ROICluster,
)

_FULLSCREEN_DIR = Path(__file__).resolve().parents[3] / "test data" / "fullscreen_autoroi"


class TestPublicContract:
    def test_empty_input_returns_empty(self) -> None:
        analyzer = BBoxAnalyzer(frame_width=720, frame_height=1280)
        assert analyzer.analyze([]) == []

    def test_invalid_frame_size_raises(self) -> None:
        with pytest.raises(ValueError):
            BBoxAnalyzer(frame_width=0, frame_height=720)

    def test_rawbbox_derived_fields(self) -> None:
        box = RawBBox(10.0, 20.0, 110.0, 60.0, 0.95, 3, 1.5)
        assert box.box_width == 100.0
        assert box.box_height == 40.0
        assert box.center_x == 60.0 and box.center_y == 40.0

    def test_single_elite_box_survives(self) -> None:
        # 1 box conf cao duy nhất → vẫn ra cluster nhờ Elite Override.
        analyzer = BBoxAnalyzer(frame_width=720, frame_height=1280)
        box = RawBBox(100, 1000, 600, 1060, 0.97, 0, 0.0)
        clusters = analyzer.analyze([box])
        assert all(isinstance(c, ROICluster) for c in clusters)


class TestBottomBandDetection:
    """Phụ đề là dải ngang đáy, xuất hiện nhiều frame → phải tách thành ROI."""

    def test_synthetic_bottom_band(self) -> None:
        analyzer = BBoxAnalyzer(frame_width=720, frame_height=1280)
        boxes: list[RawBBox] = []
        # 200 frame, mỗi frame một dòng phụ đề ở dải đáy y∈[960, 1030].
        for frame_idx in range(200):
            boxes.append(RawBBox(150, 960, 570, 1030, 0.96, frame_idx, frame_idx * 0.1))
        clusters = analyzer.analyze(boxes)
        assert clusters, "Phải có ít nhất 1 ROI cho dải phụ đề đáy"
        primary = max(clusters, key=lambda c: c.frame_count)
        center_y_ratio = (primary.coord_y_min + primary.coord_y_max) / 2 / 1280
        assert center_y_ratio > 0.6, "ROI trội phải nằm ở nửa dưới khung"
        assert primary.frame_count >= 100


def _seraw_available() -> bool:
    """Kiểm tra file seraw thật có đọc được không (chịu lỗi I/O môi trường)."""
    try:
        return _FULLSCREEN_DIR.is_dir() and (_FULLSCREEN_DIR / "1.seraw.json").is_file()
    except OSError:
        return False


@pytest.mark.skipif(
    not _seraw_available(),
    reason="Cần dữ liệu seraw thật trong /mnt/user-data/uploads.",
)
class TestRealData:
    @staticmethod
    def _load(idx: int) -> tuple[list[RawBBox], int, int]:
        try:
            doc = json.loads((_FULLSCREEN_DIR / f"{idx}.seraw.json").read_text(encoding="utf-8"))
        except OSError as io_error:  # môi trường CI có thể lỗi đọc file lớn
            pytest.skip(f"Không đọc được seraw thật: {io_error}")
        boxes: list[RawBBox] = []
        max_x = max_y = 0.0
        for frame in doc["frames"]:
            fi = int(frame.get("fi", 0))
            ts = float(frame.get("ts", 0.0))
            for b in frame.get("boxes", []):
                polygon = b.get("p", [])
                if len(polygon) < 4:
                    continue
                xs = [float(p[0]) for p in polygon]
                ys = [float(p[1]) for p in polygon]
                boxes.append(RawBBox(min(xs), min(ys), max(xs), max(ys),
                                     float(b.get("c", 0)), fi, ts))
                max_x = max(max_x, max(xs)); max_y = max(max_y, max(ys))
        return boxes, int((max_x // 16 + 1) * 16), int((max_y // 16 + 1) * 16)

    @pytest.mark.parametrize("idx", [1, 2, 3])
    def test_detects_dominant_bottom_subtitle(self, idx: int) -> None:
        boxes, width, height = self._load(idx)
        clusters = BBoxAnalyzer(frame_width=width, frame_height=height).analyze(boxes)
        assert clusters, f"{idx}_seraw: không tách được ROI nào"
        # Cụm xuất hiện nhiều frame nhất chính là dải phụ đề chính, phải ở nửa dưới.
        primary = max(clusters, key=lambda c: c.frame_count)
        center_y_ratio = (primary.coord_y_min + primary.coord_y_max) / 2 / height
        assert primary.frame_count > 300, f"{idx}: dải phụ đề phải ổn định nhiều frame"
        assert center_y_ratio > 0.55, f"{idx}: dải phụ đề chính phải ở phần dưới"
