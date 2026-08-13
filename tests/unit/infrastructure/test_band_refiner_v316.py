"""Test cho Temporal Density Band Refiner & padding biên dưới bất đối xứng (v3.16).

Hai mục tiêu của phiên v3.16:
    1. Diệt **mega-cluster**: dải phụ đề trên video nhiều frame nhiễu không còn
       phình ra gần/toàn khung — dải khít quanh dòng phụ đề thật.
    2. **Padding biên dưới** cho text NGANG nới hơn biên trên (chống cắt sát chân
       chữ / dấu phụ tiếng Việt).

Test chạy hoàn toàn headless trên toạ độ OCR thô (.seraw.json) — không cần GPU/Qt.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from subtitles_extractor.infrastructure.video.bbox_analyzer import (
    BBoxAnalyzer,
    RawBBox,
)
from subtitles_extractor.domain.value_objects.roi import TextOrientation

_FULLSCREEN_DIR = Path(__file__).resolve().parents[3] / "test data" / "fullscreen_autoroi"


def _load_raw_bboxes(
    seraw_path: Path, confidence_floor: float = 0.40
) -> tuple[list[RawBBox], int, int]:
    """Nạp box thô từ .seraw.json + suy kích thước khung từ toạ độ cực đại."""
    document = json.loads(seraw_path.read_text(encoding="utf-8"))
    boxes: list[RawBBox] = []
    max_x = max_y = 0.0
    for frame in document.get("frames", []):
        frame_idx = int(frame.get("fi", 0))
        timestamp = float(frame.get("ts", 0.0))
        for box in frame.get("boxes", []):
            confidence = float(box.get("c", 0.0))
            polygon = box.get("p", [])
            if confidence < confidence_floor or len(polygon) < 4:
                continue
            xs = [float(point[0]) for point in polygon]
            ys = [float(point[1]) for point in polygon]
            boxes.append(
                RawBBox(
                    coord_x_min=min(xs), coord_y_min=min(ys),
                    coord_x_max=max(xs), coord_y_max=max(ys),
                    confidence=confidence, frame_idx=frame_idx, timestamp_sec=timestamp,
                )
            )
            max_x, max_y = max(max_x, max(xs)), max(max_y, max(ys))
    frame_width = int(math.ceil(max_x / 8) * 8)
    frame_height = int(math.ceil(max_y / 8) * 8)
    return boxes, frame_width, frame_height


def _derive_gt_band(
    boxes: list[RawBBox], frame_width: int, frame_height: int
) -> tuple[float, float] | None:
    """GT độc lập: dải Y có mật độ thời gian (số frame) cao nhất."""
    rows: list[set[int]] = [set() for _ in range(frame_height)]
    used = False
    for box in boxes:
        if box.box_width < 0.12 * frame_width:
            continue
        used = True
        for row in range(max(0, int(box.coord_y_min)), min(frame_height - 1, int(box.coord_y_max)) + 1):
            rows[row].add(box.frame_idx)
    if not used:
        return None
    profile = np.fromiter((len(s) for s in rows), dtype=np.float64, count=frame_height)
    radius = max(1, int(0.01 * frame_height))
    smoothed = np.convolve(profile, np.ones(2 * radius + 1) / (2 * radius + 1), mode="same")
    peak = int(smoothed.argmax())
    threshold = 0.5 * float(smoothed[peak])
    top, bottom = peak, peak
    while top > 0 and smoothed[top - 1] >= threshold:
        top -= 1
    while bottom < frame_height - 1 and smoothed[bottom + 1] >= threshold:
        bottom += 1
    return top / frame_height, bottom / frame_height


def _primary_band(clusters, frame_height: int) -> tuple[float, float] | None:
    if not clusters:
        return None
    primary = max(clusters, key=lambda c: c.frame_count)
    return primary.coord_y_min / frame_height, primary.coord_y_max / frame_height


# Bộ dataset đại diện: gồm cả ca từng gây mega-cluster (file lớn nhiều nhiễu).
_MEGA_PRONE_DATASETS = ["01.seraw.json"]
_LARGE_DATASETS = [
    "Fantasy, starting to cultivate immortality at seventy years old (98 episodes).seraw.json",
    "The Hehuan Sect begins its cultivation journey by caring for its junior brothers and their partners (80 episodes).seraw.json",
]


@pytest.mark.skipif(not _FULLSCREEN_DIR.is_dir(), reason="Thiếu thư mục test data")
class TestBandRefinerAntiMegaCluster:
    """Refiner phải triệt tiêu mega-cluster và bám sát dải phụ đề thật."""

    @pytest.mark.parametrize("dataset_name", _MEGA_PRONE_DATASETS + _LARGE_DATASETS)
    def test_no_mega_cluster_after_refinement(self, dataset_name: str) -> None:
        seraw_path = _FULLSCREEN_DIR / dataset_name
        if not seraw_path.exists():
            pytest.skip(f"Thiếu dataset {dataset_name}")
        boxes, frame_width, frame_height = _load_raw_bboxes(seraw_path)
        analyzer = BBoxAnalyzer(frame_width=frame_width, frame_height=frame_height, padding=0)
        clusters = analyzer.analyze(boxes)
        band = _primary_band(clusters, frame_height)
        assert band is not None
        # Dải phụ đề một dòng không thể chiếm quá 30% chiều cao khung.
        assert band[1] - band[0] < 0.30, f"Mega-cluster còn sót: {band}"

    @pytest.mark.parametrize("dataset_name", _MEGA_PRONE_DATASETS)
    def test_refined_band_matches_ground_truth(self, dataset_name: str) -> None:
        seraw_path = _FULLSCREEN_DIR / dataset_name
        boxes, frame_width, frame_height = _load_raw_bboxes(seraw_path)
        gt_band = _derive_gt_band(boxes, frame_width, frame_height)
        assert gt_band is not None
        analyzer = BBoxAnalyzer(frame_width=frame_width, frame_height=frame_height, padding=0)
        band = _primary_band(analyzer.analyze(boxes), frame_height)
        assert band is not None
        # Tâm dải phát hiện gần tâm GT (sai số < 4% chiều cao khung).
        detected_center = (band[0] + band[1]) / 2
        gt_center = (gt_band[0] + gt_band[1]) / 2
        assert abs(detected_center - gt_center) < 0.04

    def test_refinement_can_be_disabled(self) -> None:
        seraw_path = _FULLSCREEN_DIR / "01.seraw.json"
        if not seraw_path.exists():
            pytest.skip("Thiếu dataset")
        boxes, frame_width, frame_height = _load_raw_bboxes(seraw_path)
        off = BBoxAnalyzer(
            frame_width=frame_width, frame_height=frame_height,
            enable_band_refinement=False, padding=0,
        ).analyze(boxes)
        on = BBoxAnalyzer(
            frame_width=frame_width, frame_height=frame_height,
            enable_band_refinement=True, padding=0,
        ).analyze(boxes)
        band_off = _primary_band(off, frame_height)
        band_on = _primary_band(on, frame_height)
        # Khi BẬT, dải hẹp hơn (hoặc bằng) khi TẮT — tinh chỉnh chỉ co lại.
        assert (band_on[1] - band_on[0]) <= (band_off[1] - band_off[0]) + 1e-6


@pytest.mark.skipif(not _FULLSCREEN_DIR.is_dir(), reason="Thiếu thư mục test data")
class TestAsymmetricBottomPadding:
    """Text NGANG: biên dưới phải được nới nhiều hơn biên trên."""

    def test_horizontal_cluster_bottom_padding_exceeds_top(self) -> None:
        seraw_path = _FULLSCREEN_DIR / "2.seraw.json"
        if not seraw_path.exists():
            pytest.skip("Thiếu dataset")
        boxes, frame_width, frame_height = _load_raw_bboxes(seraw_path)
        # Dải khít (tham chiếu) vs dải có padding (mặc định bottom_padding_factor=1.6).
        tight = BBoxAnalyzer(frame_width=frame_width, frame_height=frame_height, padding=0).analyze(boxes)
        padded = BBoxAnalyzer(
            frame_width=frame_width, frame_height=frame_height, bottom_padding_factor=1.6
        ).analyze(boxes)
        tight_primary = max(tight, key=lambda c: c.frame_count)
        padded_primary = max(padded, key=lambda c: c.frame_count)
        assert padded_primary.orientation == TextOrientation.HORIZONTAL
        top_pad = tight_primary.coord_y_min - padded_primary.coord_y_min
        bottom_pad = padded_primary.coord_y_max - tight_primary.coord_y_max
        assert top_pad > 0 and bottom_pad > 0
        # Biên dưới nới hơn biên trên (theo bottom_padding_factor > 1).
        assert bottom_pad > top_pad

    def test_user_padding_override_is_symmetric(self) -> None:
        seraw_path = _FULLSCREEN_DIR / "2.seraw.json"
        if not seraw_path.exists():
            pytest.skip("Thiếu dataset")
        boxes, frame_width, frame_height = _load_raw_bboxes(seraw_path)
        tight = BBoxAnalyzer(frame_width=frame_width, frame_height=frame_height, padding=0).analyze(boxes)
        forced = BBoxAnalyzer(frame_width=frame_width, frame_height=frame_height, padding=12).analyze(boxes)
        tight_primary = max(tight, key=lambda c: c.frame_count)
        forced_primary = max(forced, key=lambda c: c.frame_count)
        top_pad = tight_primary.coord_y_min - forced_primary.coord_y_min
        bottom_pad = forced_primary.coord_y_max - tight_primary.coord_y_max
        # _user_padding override → đối xứng (trừ trường hợp chạm biên khung).
        if tight_primary.coord_y_min - 12 >= 0 and tight_primary.coord_y_max + 12 <= frame_height:
            assert top_pad == bottom_pad == 12
