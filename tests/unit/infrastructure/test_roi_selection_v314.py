"""Test logic chọn vùng ROI phụ đề chính (Phần 6 — preset tự nhận diện)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from subtitles_extractor.domain.value_objects.roi import TextAlignment, TextOrientation
from subtitles_extractor.infrastructure.video.bbox_analyzer import (
    BBoxAnalyzer,
    RawBBox,
    ROICluster,
)
from subtitles_extractor.infrastructure.video.roi_selection import (
    cluster_to_roi,
    detect_subtitle_band_from_boxes,
    select_primary_subtitle_cluster,
    select_primary_subtitle_roi,
)



def _cluster(cid: int, y0: int, y1: int, frames: int, boxes: int) -> ROICluster:
    return ROICluster(
        cluster_id=cid, orientation=TextOrientation.HORIZONTAL, alignment=TextAlignment.CENTER,
        coord_x_min=100, coord_y_min=y0, coord_x_max=600, coord_y_max=y1,
        bbox_count=boxes, frame_count=frames, mean_confidence=0.95,
    )


class TestSelectionLogic:
    def test_empty_returns_none(self) -> None:
        assert select_primary_subtitle_cluster([], 1280) is None
        assert select_primary_subtitle_roi([], 1280) is None

    def test_picks_highest_frame_presence(self) -> None:
        # Cụm nhiễu nhiều BOX nhưng ít FRAME vs phụ đề ít box hơn nhưng nhiều frame.
        noise = _cluster(0, 200, 260, frames=30, boxes=400)
        subtitle = _cluster(1, 1000, 1070, frames=900, boxes=300)
        chosen = select_primary_subtitle_cluster([noise, subtitle], 1280)
        assert chosen is subtitle  # frame-presence thắng số box

    def test_tiebreak_prefers_bottom(self) -> None:
        # Cùng frame_count + bbox_count → ưu tiên cụm nằm dưới (phụ đề hardsub).
        top = _cluster(0, 100, 170, frames=500, boxes=200)
        bottom = _cluster(1, 1000, 1070, frames=500, boxes=200)
        assert select_primary_subtitle_cluster([top, bottom], 1280) is bottom

    def test_cluster_to_roi_clamps(self) -> None:
        c = _cluster(0, -10, 100, frames=5, boxes=5)
        roi = cluster_to_roi(c)
        assert roi.x >= 0 and roi.y >= 0 and roi.width >= 1 and roi.height >= 1


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_TEST_DATA = _PROJECT_ROOT / "test data"
_FULLSCREEN_DIR = _TEST_DATA / "fullscreen_autoroi"


def _load_boxes(path: Path):
    doc = json.loads(path.read_text(encoding="utf-8"))
    boxes = []
    max_x = max_y = 0.0
    for frame in doc["frames"]:
        fi = int(frame.get("fi", 0)); ts = float(frame.get("ts", 0.0))
        for b in frame.get("boxes", []):
            polygon = b.get("p", [])
            if len(polygon) < 4:
                continue
            xs = [float(q[0]) for q in polygon]; ys = [float(q[1]) for q in polygon]
            boxes.append(RawBBox(min(xs), min(ys), max(xs), max(ys), float(b.get("c", 0)), fi, ts))
            max_x = max(max_x, max(xs)); max_y = max(max_y, max(ys))
    return boxes, int((max_x // 16 + 1) * 16), int((max_y // 16 + 1) * 16)


@pytest.mark.skipif(
    not _FULLSCREEN_DIR.is_dir(), reason="Cần thư mục 'test data/fullscreen_autoroi'."
)
class TestRealDataSelection:
    @pytest.mark.parametrize("name", ["01", "02", "03", "1", "2", "3", "4"])
    def test_primary_roi_is_lower_half(self, name: str) -> None:
        path = _FULLSCREEN_DIR / f"{name}.seraw.json"
        if not path.is_file():
            pytest.skip(f"Thiếu {path.name}")
        boxes, width, height = _load_boxes(path)
        from subtitles_extractor.infrastructure.video.roi_selection import (
            select_subtitle_roi_smart,
        )
        clusters = BBoxAnalyzer(frame_width=width, frame_height=height).analyze(boxes)
        roi = select_subtitle_roi_smart(clusters, boxes, width, height)
        assert roi is not None
        center_y_ratio = (roi.y + roi.height / 2) / height
        assert center_y_ratio > 0.45, f"{name}: ROI phụ đề chính phải ở phần dưới"


@pytest.mark.skipif(
    not (_TEST_DATA / "haomen_fullscreen.seraw.json").is_file(),
    reason="Cần haomen_fullscreen.seraw.json (ca mega-cluster).",
)
class TestMegaClusterDataset:
    """豪门Z外 fullscreen ~114k box: từng bị mega-cluster nuốt cả khung."""

    def test_smart_avoids_full_frame(self) -> None:
        from subtitles_extractor.infrastructure.video.roi_selection import (
            select_subtitle_roi_smart,
        )
        boxes, width, height = _load_boxes(_TEST_DATA / "haomen_fullscreen.seraw.json")
        clusters = BBoxAnalyzer(frame_width=width, frame_height=height).analyze(boxes)
        roi = select_subtitle_roi_smart(clusters, boxes, width, height)
        assert roi is not None
        assert roi.height < height * 0.55, "Không được là mega-cluster toàn khung"
        center_y_ratio = (roi.y + roi.height / 2) / height
        assert center_y_ratio > 0.6, "Dải phụ đề 豪门Z外 nằm gần đáy"


class TestFramePresenceBandDetection:
    """Phát hiện dải phụ đề bằng frame-presence — kháng mega-cluster (data lớn)."""

    @staticmethod
    def _boxes_bottom_band_plus_noise():
        boxes = []
        # Phụ đề: dải đáy y∈[900,960], xuất hiện 300 frame khác nhau.
        for f in range(300):
            boxes.append(RawBBox(150, 900, 570, 960, 0.95, f, f * 0.1))
        # Nhiễu cảnh: rải rác phía trên, mỗi vị trí chỉ vài frame.
        for f in range(8):
            boxes.append(RawBBox(100, 200, 300, 240, 0.95, f, f * 0.1))
            boxes.append(RawBBox(400, 450, 600, 490, 0.95, f, f * 0.1))
        return boxes

    def test_detects_bottom_band_over_noise(self) -> None:
        boxes = self._boxes_bottom_band_plus_noise()
        roi = detect_subtitle_band_from_boxes(boxes, 720, 1280)
        assert roi is not None
        center_y_ratio = (roi.y + roi.height / 2) / 1280
        assert center_y_ratio > 0.6, "Phải bắt đúng dải đáy, không phải nhiễu trên"

    def test_empty_returns_none(self) -> None:
        assert detect_subtitle_band_from_boxes([], 720, 1280) is None

    def test_smart_falls_back_on_mega_cluster(self) -> None:
        from subtitles_extractor.infrastructure.video.roi_selection import (
            select_subtitle_roi_smart,
        )
        # Mega-cluster nuốt cả khung (height ~ 100%) → smart phải bỏ, dùng band.
        mega = _cluster(0, 0, 1270, frames=2000, boxes=5000)
        boxes = self._boxes_bottom_band_plus_noise()
        roi = select_subtitle_roi_smart([mega], boxes, 720, 1280)
        assert roi is not None
        center_y_ratio = (roi.y + roi.height / 2) / 1280
        assert center_y_ratio > 0.6, "Mega-cluster → fallback band ở dải đáy"

    def test_smart_keeps_normal_cluster(self) -> None:
        from subtitles_extractor.infrastructure.video.roi_selection import (
            select_subtitle_roi_smart,
        )
        # Cụm chiều cao hợp lý (dải đáy hẹp) → giữ nguyên, không cần band.
        normal = _cluster(0, 1000, 1070, frames=500, boxes=200)
        roi = select_subtitle_roi_smart([normal], [], 720, 1280)
        assert roi is not None
        assert roi.y >= 1000 - 1
