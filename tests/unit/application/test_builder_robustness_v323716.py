"""[v3.23.716] Câu rác từ chữ dọc trong cảnh, khung cùng điểm ảnh đọc rỗng sau hộp phình, OCR hụt rồi đọc lại yếu, khung ranh giới.

Lượt chạy thật 3.23.714 của chủ dự án (phim nối 86 tập, cấu hình dựng ghi trong ``meta.run.builder``): câu rác "肃疏C气"
(câu 1229), "是" kết thúc sớm 0,24 s, "切" kết thúc sớm 0,12 s; và ranh giới "历史至强雷劫" → "四灵天雷劫" lệch một khung trên
mọi bộ dữ liệu thô cũ.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from subtitles_extractor.application.dtos.extract_subtitles_dto import (
    SubtitleBuilderConfig,
)
from subtitles_extractor.application.services.subtitle_builder import SubtitleBuilder
from subtitles_extractor.application.services.subtitle_pipeline import grouping
from subtitles_extractor.application.services.subtitle_pipeline.voting import (
    clear_frame_build_cache,
)
from subtitles_extractor.domain.entities.ocr_frame_result import (
    OcrFrameResult,
    OcrTextBox,
)
from subtitles_extractor.domain.value_objects.confidence import Confidence
from subtitles_extractor.domain.value_objects.roi import Roi

_ROI = Roi(x=0, y=774, width=720, height=136)
_LINES = ("修炼一途的", "自炼气始吧", "经筑基之后", "寻求飞升啊", "检测到宿主", "即将面临了", "历史至强雷", "四灵天雷劫")
_LINE_BOX = (222, 65, 498, 133)
#: Cấu hình dựng của lượt chạy thật (``meta.run.builder``).
_RUN_CONFIG = {
    "sample_step_sec": 0.03, "merge_gap_sec": 0.1, "min_duration_sec": 0.09, "min_confidence": 0.4,
    "temporal_padding_sec": 0.0, "use_viterbi": True, "min_text_chars": 0, "similarity_threshold": 0.75,
    "y_clustering_tolerance_min_px": 15.0, "alignment_center_tolerance_ratio": 0.08,
}
Box = tuple[str, float, tuple[int, int, int, int]]


@pytest.fixture(autouse=True)
def _fresh_text_cache() -> Iterator[None]:
    """Bộ nhớ chữ theo khung khoá bằng ``id(frame)`` (xoá ở đầu mỗi lần dựng) — test gọi thẳng hàm nhóm phải tự xoá."""
    clear_frame_build_cache()
    yield
    clear_frame_build_cache()


def _repair(buckets: list[list[OcrFrameResult]]) -> list[list[OcrFrameResult]]:
    return grouping._return_boundary_frames_to_their_line(buckets, 0.3, 5.0)


def _frame(ts: float, boxes: list[Box], continues: bool = False) -> OcrFrameResult:
    return OcrFrameResult(
        frame_index=round(ts * 25), timestamp_sec=ts, continues_previous=continues,
        text_boxes=[
            OcrTextBox(text=text, confidence=Confidence(conf), polygon=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
            for text, conf, (x0, y0, x1, y1) in boxes
        ],
    )


class _Film:
    """Phim giả bước 0,04 s: 8 câu thường (để khâu dựng đo được cỡ chữ, tâm, dải phụ đề) rồi tới phần cần thử."""

    def __init__(self) -> None:
        self.frames: list[OcrFrameResult] = []
        self.clock = 0.0
        for line in _LINES:
            self.show([(line, 1.0, _LINE_BOX)], 30)
            self.show([], 8)

    def show(self, boxes: list[Box], count: int, same_pixels: bool = False) -> tuple[float, float]:
        """Trả ``(mốc khung đầu, mốc kết thúc = khung cuối + một bước)``. ``same_pixels``: cùng điểm ảnh khung trước."""
        first = round(self.clock + 0.04, 2)
        for index in range(count):
            self.clock = round(self.clock + 0.04, 2)
            self.frames.append(_frame(self.clock, boxes, continues=same_pixels or index > 0))
        return first, round(self.clock + 0.04, 2)

    def build(self) -> list:
        return SubtitleBuilder(SubtitleBuilderConfig(**_RUN_CONFIG)).build(self.frames, roi=_ROI)


def _near(value: float, expected: float) -> bool:
    return abs(value - expected) < 0.011


# ── Câu rác từ chữ dọc trong cảnh (câu 1229 "肃疏C气") ───────────────────────────────────────────────────────────────


def test_vertical_scene_text_spread_across_the_frame_makes_no_line() -> None:
    """2546,80–2547,36 s: 6–7 hộp chữ DỌC (cao hết vùng chữ) trải khắp bề ngang → "dòng" gộp có tâm đúng giữa nên qua
    bước kiểm căn lề; bộ lọc lệch dọc bỏ bớt hộp, còn hai hộp ở mép trái (tâm x ≈ 99) → mã cũ ra câu "肃疏C气"."""
    film = _Film()
    film.show([
        ("MAHO", 0.68, (43, 0, 86, 91)), ("青端际", 0.39, (62, 0, 117, 136)), ("川C气星", 0.65, (96, 0, 153, 136)),
        ("5", 0.84, (267, 0, 314, 81)), ("肖", 1.0, (307, 0, 420, 99)), ("气气", 0.98, (581, 0, 637, 136)),
        ("下背", 0.53, (612, 0, 676, 136)),
    ], 5)
    film.show([
        ("M向H图", 0.38, (34, 0, 78, 97)), ("肃疏", 0.53, (53, 0, 112, 136)), ("C气", 0.78, (92, 0, 145, 136)),
        ("肖", 1.0, (303, 0, 424, 106)), ("流气气", 0.94, (588, 0, 641, 136)), ("下", 0.83, (618, 0, 683, 136)),
    ], 8)
    film.show([
        ("M向片图", 0.54, (25, 0, 73, 107)), ("肃医", 0.43, (47, 0, 106, 136)), ("C气", 0.72, (87, 0, 140, 136)),
        ("肖", 1.0, (300, 0, 426, 115)), ("流气气", 0.94, (593, 0, 648, 136)), ("晾卜", 0.49, (623, 0, 689, 136)),
    ], 2)
    film.show([("2>", 0.53, (161, 5, 175, 19)), ("2>", 0.45, (207, 5, 223, 20)), ("7", 0.53, (556, 11, 575, 29))], 10)
    film.show([("哎呀", 1.0, (296, 64, 422, 133))], 30)
    film.show([], 8)
    assert [event.text for event in film.build()] == [*_LINES, "哎呀"]


# ── Khung cùng điểm ảnh đọc rỗng sau các khung hộp phình ("是") ──────────────────────────────────────────────────────


def test_empty_rereads_of_the_same_pixels_after_inflated_boxes_keep_the_line_to_its_end() -> None:
    """"是" 2654,64–2655,36 s: 12 khung hộp cao hết vùng chữ, rồi 6 khung CÙNG điểm ảnh đọc lại ra rỗng."""
    film = _Film()
    start, _ = film.show([("是", 1.0, (315, 0, 404, 136))], 12)
    _, end = film.show([], 6, same_pixels=True)
    film.show([], 2)
    film.show([("还不回去准备", 1.0, (183, 67, 535, 133))], 30)
    film.show([], 8)
    events = film.build()
    assert [event.text for event in events] == [*_LINES, "是", "还不回去准备"]
    assert _near(events[-2].start_sec, start) and _near(events[-2].end_sec, end)


# ── OCR hụt 1–2 khung rồi đọc lại yếu ("切") ─────────────────────────────────────────────────────────────────────────


def _cut_line(film: _Film, blank_frames: int) -> tuple[float, float, float]:
    start, strong_end = film.show([("切", 1.0, (322, 63, 396, 135))], 17)
    for _ in range(blank_frames):
        film.show([], 1)
    _, weak_end = film.show([("切", 0.57, (312, 0, 410, 136))], 1)
    film.show([], 1)
    film.show([], 10, same_pixels=True)
    return start, strong_end, weak_end


def test_weak_reread_after_a_two_frame_hole_extends_the_line() -> None:
    film = _Film()
    start, _, weak_end = _cut_line(film, blank_frames=2)
    events = film.build()
    assert [event.text for event in events] == [*_LINES, "切"]
    assert _near(events[-1].start_sec, start) and _near(events[-1].end_sec, weak_end)


def test_weak_reread_after_a_longer_blank_is_not_glued_on() -> None:
    """Từ 3 khung trống trở lên là ngưỡng "nói lại" của luật câu lặp — không kéo mốc cuối qua khoảng đó."""
    film = _Film()
    _, strong_end, _ = _cut_line(film, blank_frames=4)
    events = film.build()
    line = next(event for event in events if event.text == "切")
    assert _near(line.end_sec, strong_end)


# ── Khung sát ranh giới về đúng câu ──────────────────────────────────────────────────────────────────────────────────


def _bucket(text: str, first: float, count: int) -> list[OcrFrameResult]:
    return [_frame(round(first + 0.04 * index, 2), [(text, 1.0, _LINE_BOX)]) for index in range(count)]


def test_last_frame_of_previous_line_leaves_the_next_group() -> None:
    """Ảnh thật tập 1: khung 16,80 s vẫn là "历史至强雷劫"; Viterbi gán nó vào nhóm "四灵天雷劫" (chung "雷劫")."""
    previous = _bucket("历史至强雷劫", 15.56, 31)                       # 15,56 … 16,76
    following = [*_bucket("历史至强雷劫", 16.80, 1), *_bucket("四灵天雷劫", 16.84, 30)]
    left, right = _repair([previous, following])
    assert [round(left[-1].timestamp_sec, 2), round(right[0].timestamp_sec, 2)] == [16.80, 16.84]
    assert len(left) == 32 and len(right) == 30


def test_first_frame_of_next_line_leaves_the_previous_group() -> None:
    previous = [*_bucket("哎呀师叔", 10.00, 20), *_bucket("师叔", 10.80, 2)]
    following = _bucket("师叔", 10.88, 20)
    left, right = _repair([previous, following])
    assert len(left) == 20 and len(right) == 22 and round(right[0].timestamp_sec, 2) == 10.80


def test_boundary_repair_leaves_clean_and_degenerate_groups_alone() -> None:
    clean = [_bucket("历史至强雷劫", 1.0, 10), _bucket("四灵天雷劫", 1.4, 10)]
    assert [len(bucket) for bucket in _repair(clean)] == [10, 10]
    # Nhóm sau TOÀN khung đọc ra câu trước (bầu chữ ra chính câu đó) → hai nhóm cùng chữ, không dời gì.
    same = [_bucket("师叔", 1.0, 5), _bucket("师叔", 1.2, 3)]
    assert [len(bucket) for bucket in _repair(same)] == [5, 3]
    # Không bao giờ dời hết khung của một nhóm.
    tiny = [_bucket("历史至强雷劫", 1.0, 10), [*_bucket("历史至强雷劫", 1.4, 1), *_bucket("四灵天雷劫", 1.44, 1)]]
    repaired = _repair(tiny)
    assert all(repaired) and sum(len(bucket) for bucket in repaired) == 12
    assert _repair([]) == []
