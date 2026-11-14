"""[v3.23.717] Đón đầu lỗi OCR chưa gặp: các ca rút ra từ bộ thử sức gây nhiễu giả định lên OCR thô THẬT.

Bộ thử sức (lượt chạy 3.23.714, 208 752 khung, cấu hình dựng của chủ dự án) bỏ / đổi ngẫu nhiên lần đọc rồi dựng lại và
so đáp án. Mỗi test dưới đây là một ca nhỏ nhất tái hiện một điểm yếu bộ thử sức tìm ra (đỏ trên mã 3.23.716), kèm test
chặn chiều ngược lại (câu thật không được gộp / mất).
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from subtitles_extractor.application.dtos.extract_subtitles_dto import (
    SubtitleBuilderConfig,
)
from subtitles_extractor.application.services.subtitle_builder import SubtitleBuilder
from subtitles_extractor.application.services.subtitle_pipeline import (
    event_filters,
    spatial_filters,
)
from subtitles_extractor.application.services.subtitle_pipeline.voting import (
    clear_frame_build_cache,
)
from subtitles_extractor.domain.entities.ocr_frame_result import (
    OcrFrameResult,
    OcrTextBox,
)
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.value_objects.confidence import Confidence
from subtitles_extractor.domain.value_objects.roi import Roi
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval

_ROI = Roi(x=0, y=774, width=720, height=136)
_LINES = ("修炼一途的", "自炼气始吧", "经筑基之后", "寻求飞升啊", "检测到宿主", "即将面临了", "历史至强雷", "四灵天雷劫")
_CENTER = 360
#: Cấu hình dựng của lượt chạy thật (``meta.run.builder``).
_RUN_CONFIG = {
    "sample_step_sec": 0.03, "merge_gap_sec": 0.1, "min_duration_sec": 0.09, "min_confidence": 0.4,
    "temporal_padding_sec": 0.0, "use_viterbi": True, "min_text_chars": 0, "similarity_threshold": 0.75,
    "y_clustering_tolerance_min_px": 15.0, "alignment_center_tolerance_ratio": 0.08,
}
Box = tuple[str, float, tuple[int, int, int, int]]


@pytest.fixture(autouse=True)
def _fresh_text_cache() -> Iterator[None]:
    clear_frame_build_cache()
    yield
    clear_frame_build_cache()


def _line_box(text: str) -> tuple[int, int, int, int]:
    """Hộp căn giữa đúng cỡ phụ đề của phim mẫu (mỗi chữ ≈ 55 px, cao 65–133 px)."""
    half = round(len(text) * 27.6)
    return (_CENTER - half, 65, _CENTER + half, 133)


def _frame(ts: float, boxes: list[Box]) -> OcrFrameResult:
    return OcrFrameResult(
        frame_index=round(ts * 25), timestamp_sec=ts,
        text_boxes=[
            OcrTextBox(text=text, confidence=Confidence(conf), polygon=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
            for text, conf, (x0, y0, x1, y1) in boxes
        ],
    )


class _Film:
    """Phim giả bước 0,04 s: 8 câu thường rồi tới phần cần thử. Mọi khung là một lần đọc riêng (không cờ cùng điểm ảnh)."""

    def __init__(self) -> None:
        self.frames: list[OcrFrameResult] = []
        self.clock = 0.0
        for line in _LINES:
            self.say(line, 30)
            self.blank(8)

    def show(self, boxes: list[Box], count: int) -> tuple[float, float]:
        first = round(self.clock + 0.04, 2)
        for _ in range(count):
            self.clock = round(self.clock + 0.04, 2)
            self.frames.append(_frame(self.clock, boxes))
        return first, round(self.clock + 0.04, 2)

    def say(self, text: str, count: int, conf: float = 1.0) -> tuple[float, float]:
        return self.show([(text, conf, _line_box(text))], count)

    def blank(self, count: int) -> None:
        self.show([], count)

    def build(self) -> list[SubtitleEvent]:
        return SubtitleBuilder(SubtitleBuilderConfig(**_RUN_CONFIG)).build(self.frames, roi=_ROI)

    def tail(self) -> list[tuple[str, float, float]]:
        return [(event.text, round(event.start_sec, 2), round(event.end_sec, 2)) for event in self.build()[len(_LINES):]]


# ── OCR hụt giữa câu khi khoảng gộp đặt nhỏ ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("missed_frames", [1, 2])
def test_short_ocr_hole_inside_a_line_does_not_split_it(missed_frames: int) -> None:
    """Khoảng gộp 0,1 s, bước thật 0,04 s: hụt 2 khung liền (hở 0,12 s) từng tách câu làm hai câu trùng chữ."""
    film = _Film()
    start, _ = film.say("这一巴掌免不了", 20)
    film.blank(missed_frames)
    _, end = film.say("这一巴掌免不了", 14)
    film.blank(8)
    assert film.tail() == [("这一巴掌免不了", start, end)]


def test_three_blank_frames_between_the_same_text_is_still_a_repeat() -> None:
    """Từ 3 mẫu trống trở lên là nhân vật nói lại — không được gộp (ngưỡng của luật câu lặp, không đổi)."""
    film = _Film()
    film.say("师叔", 12)
    film.blank(3)
    film.say("师叔", 12)
    film.blank(8)
    assert [text for text, _start, _end in film.tail()] == ["师叔", "师叔"]


# ── Hai bản đọc lẫn chữ của cùng một câu, không có cờ cùng điểm ảnh ──────────────────────────────────────────────────


def test_alternating_confusable_reads_stay_one_line_with_the_confident_text() -> None:
    """"癞蛤蟆" 13 khung (0,99) → "痴蛤蟆" 20 khung (0,93) → "癞蛤蟆" 2 khung: chữ quay lại = cùng một hình.

    Mã cũ tách làm hai câu (mốc cuối sớm 0,88 s) và trong câu gộp thì chọn bản nhiều khung hơn ("痴").
    """
    film = _Film()
    start, _ = film.say("这不是癞蛤蟆吗", 13, conf=0.99)
    film.say("这不是痴蛤蟆吗", 20, conf=0.93)
    _, end = film.say("这不是癞蛤蟆吗", 2, conf=0.99)
    film.blank(8)
    assert film.tail() == [("这不是癞蛤蟆吗", start, end)]


def test_two_real_lines_differing_by_one_char_stay_two_lines() -> None:
    """"这一拳替我爹" → "这一拳替我娘": một chiều, cả hai tin cậy 1,0 — hai câu thật."""
    film = _Film()
    film.say("这一拳替我爹", 20)
    film.say("这一拳替我娘", 20)
    film.blank(8)
    assert [text for text, _start, _end in film.tail()] == ["这一拳替我爹", "这一拳替我娘"]


@pytest.mark.parametrize(("good", "bad"), [("知道了", "知她了"), ("叶然不至于扛不住", "叶然时至于扛不住")])
def test_low_confidence_one_char_misread_block_joins_its_line(good: str, bad: str) -> None:
    """Một lần đọc sai MỘT chữ (13 khung, tin cậy 0,87) rồi đọc đúng (20 khung, 1,0): mã cũ ra hai câu, mốc đầu trễ 0,5 s."""
    film = _Film()
    start, _ = film.say(bad, 13, conf=0.87)
    _, end = film.say(good, 20)
    film.blank(8)
    assert film.tail() == [(good, start, end)]


def _event(text: str, start: float, end: float, conf: float, frames: int) -> SubtitleEvent:
    return SubtitleEvent(
        index=0, text=text, interval=TimeInterval(start, end), confidence=Confidence(conf), frame_count=frames,
        bounding_box=(277, 65, 443, 133),
    )


def test_misread_merge_needs_adjacency_one_diff_and_a_clear_confidence_gap() -> None:
    merge = event_filters.merge_misread_substitution_events
    good, bad = _event("知道了", 10.52, 11.28, 1.0, 20), _event("知她了", 10.0, 10.48, 0.87, 13)
    merged, count = merge([bad, good], 0.04)
    assert count == 1 and [(e.text, e.start_sec, e.end_sec, e.frame_count) for e in merged] == [("知道了", 10.0, 11.28, 33)]
    # Cả hai tin cậy cao ("我去哪" → "你去哪" thật) → để nguyên.
    assert merge([_event("我去哪", 10.0, 10.48, 1.0, 13), _event("你去哪", 10.52, 11.28, 1.0, 20)], 0.04)[1] == 0
    # Cách nhau quá một mẫu, khác hai chữ, câu hai chữ, hoặc không biết bước lấy mẫu → để nguyên.
    assert merge([bad, _event("知道了", 10.68, 11.28, 1.0, 16)], 0.04)[1] == 0
    assert merge([_event("叫师她", 10.0, 10.48, 0.87, 13), _event("叫姐姐", 10.52, 11.28, 1.0, 20)], 0.04)[1] == 0
    assert merge([_event("师她", 10.0, 10.48, 0.87, 13), _event("师妹", 10.52, 11.28, 1.0, 20)], 0.04)[1] == 0
    assert merge([bad, good], 0.0)[1] == 0 and merge([], 0.04) == ([], 0)


# ── Chữ trong cảnh nằm cạnh / chồng lên câu ──────────────────────────────────────────────────────────────────────────


def _cleaned_texts(boxes: list[Box]) -> list[str]:
    cleaned = spatial_filters.clean_spatial_outliers(
        _frame(1.0, boxes), _ROI, SubtitleBuilderConfig(**_RUN_CONFIG), float(_CENTER),
    )
    return [box.text for box in cleaned.text_boxes]


def test_scene_char_overlapping_the_line_box_is_dropped() -> None:
    """Hộp "小" (0,94) nằm chồng trong bề ngang hộp câu — một dòng ngang không có hai hộp chồng nhau."""
    assert _cleaned_texts([("追回楚师姐了吧", 1.0, (155, 66, 562, 132)), ("小", 0.94, (381, 45, 450, 127))]) == ["追回楚师姐了吧"]


def test_confident_scene_char_beside_a_centered_line_is_dropped() -> None:
    """Hộp chính tự nó đúng tâm; hộp "三" tin cậy cao bên trái kéo tâm cả dòng lệch 40 px (vẫn trong dung sai câu dài)."""
    assert _cleaned_texts([("台上还剩一人", 1.0, (194, 65, 526, 133)), ("三", 0.92, (114, 60, 180, 130))]) == ["台上还剩一人"]


def test_line_split_into_two_boxes_is_kept_whole() -> None:
    """Phụ đề có dấu cách bị khoanh hai hộp: không hộp nào tự đúng tâm → giữ cả hai."""
    kept = _cleaned_texts([("哎呀", 1.0, (194, 65, 304, 133)), ("师叔你误会了", 1.0, (330, 65, 526, 133))])
    assert sorted(kept) == sorted(["哎呀", "师叔你误会了"])


def test_builder_line_with_noise_glued_on_one_read_stays_clean_and_whole() -> None:
    """Một lần đọc (10 khung) dính thêm hộp nhiễu: mã cũ ra "…小" thành câu riêng, câu thật trễ đầu 0,5 s."""
    film = _Film()
    start, _ = film.say("追回楚师姐了吧", 2)
    film.show([("追回楚师姐了吧", 1.0, _line_box("追回楚师姐了吧")), ("小", 0.94, (381, 45, 450, 127))], 10)
    _, end = film.say("追回楚师姐了吧", 20)
    film.blank(8)
    assert film.tail() == [("追回楚师姐了吧", start, end)]


# ── Mô hình đọc sai rất tự tin: tin cậy không được lật lựa chọn một mình ────────────────────────────────────────────


def test_confident_trailing_misread_does_not_replace_the_line() -> None:
    """Bộ 3.23.690, 4880,48 s: 28 khung "叶然肯定扛不住" (0,93–0,96) rồi 7 khung "叶然肯定拉不住" (0,99–1,0).

    Bản nháp của 3.23.717 chọn bản tin cậy cao hơn → câu đúng thành sai trên dữ liệu thật.
    """
    film = _Film()
    start, _ = film.say("叶然肯定扛不住", 28, conf=0.95)
    _, end = film.say("叶然肯定拉不住", 7, conf=0.99)
    film.blank(8)
    assert film.tail() == [("叶然肯定扛不住", start, end)]


@pytest.mark.parametrize(
    ("left", "right", "winner"),
    [
        ((15, 0.983), (20, 0.942), "L"),     # "癞蛤蟆" / "痴蛤蟆": đủ khung, tin cậy cao hơn rõ → thắng
        ((7, 0.99), (28, 0.95), "R"),        # "拉不住" / "扛不住": quá ít khung → thua dù tin cậy cao hơn
        ((20, 0.96), (20, 0.95), "L"),       # chênh 0,01 ("清清自白" / "清清白白"): bản nặng hơn
        ((10, 0.95), (20, 0.95), "R"),
    ],
)
def test_pick_reading_rule(left: tuple[int, float], right: tuple[int, float], winner: str) -> None:
    assert event_filters._pick_reading("L", left, "R", right) == winner
