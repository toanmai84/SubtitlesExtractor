"""[v3.23.718] Đón đầu lỗi OCR chưa gặp, đợt 2: mất khung, dấu câu thừa, logo cố định, đổi bố cục giữa phim, vùng chữ cắt
chân, chữ phồn lẻ tẻ, mốc VFR rung.

Mỗi test là ca nhỏ nhất tái hiện một điểm yếu mà ``tools/stress_subtitle_builder.py`` tìm ra trên OCR thô thật của lượt
3.23.714 (đỏ trên mã 3.23.717), kèm test chặn chiều ngược lại.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from subtitles_extractor.application.dtos.extract_subtitles_dto import (
    SubtitleBuilderConfig,
)
from subtitles_extractor.application.services.subtitle_builder import SubtitleBuilder
from subtitles_extractor.application.services.subtitle_pipeline import (
    box_geometry,
    orchestrator,
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

_ROI = Roi(x=0, y=774, width=720, height=136)
_LINES = ("修炼一途的", "自炼气始吧", "经筑基之后", "寻求飞升啊", "检测到宿主", "即将面临了", "历史至强雷", "四灵天雷劫")
_CENTER = 360
_RUN_CONFIG = {
    "sample_step_sec": 0.03, "merge_gap_sec": 0.1, "min_duration_sec": 0.09, "min_confidence": 0.4,
    "temporal_padding_sec": 0.0, "use_viterbi": True, "min_text_chars": 0, "similarity_threshold": 0.75,
    "y_clustering_tolerance_min_px": 15.0, "alignment_center_tolerance_ratio": 0.08,
}
Box = tuple[str, float, tuple[float, float, float, float]]


@pytest.fixture(autouse=True)
def _fresh_text_cache() -> Iterator[None]:
    clear_frame_build_cache()
    yield
    clear_frame_build_cache()


def _line_box(text: str, center: float = _CENTER, top: float = 65, bottom: float = 133) -> tuple[float, ...]:
    half = len(text) * 27.6
    return (center - half, top, center + half, bottom)


def _frame(ts: float, boxes: list[Box], continues: bool = False) -> OcrFrameResult:
    return OcrFrameResult(
        frame_index=round(ts * 25), timestamp_sec=ts, continues_previous=continues,
        text_boxes=[
            OcrTextBox(text=text, confidence=Confidence(conf), polygon=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
            for text, conf, (x0, y0, x1, y1) in boxes
        ],
    )


class _Film:
    """Phim giả bước 0,04 s; ``extra`` thêm vào mọi khung (logo); ``skip`` là các mốc khung bị mất khỏi dữ liệu."""

    def __init__(self, extra: list[Box] | None = None, intro: bool = True) -> None:
        self.frames: list[OcrFrameResult] = []
        self.clock = 0.0
        self.extra = extra or []
        self.skip: set[float] = set()
        if intro:
            for line in _LINES:
                self.say(line, 30)
                self.blank(8)

    def show(self, boxes: list[Box], count: int, same_pixels: bool = False) -> tuple[float, float]:
        first = round(self.clock + 0.04, 2)
        for index in range(count):
            self.clock = round(self.clock + 0.04, 2)
            if self.clock not in self.skip:
                self.frames.append(_frame(self.clock, boxes + self.extra, continues=same_pixels or index > 0))
        return first, round(self.clock + 0.04, 2)

    def say(self, text: str, count: int, conf: float = 1.0, **geometry: float) -> tuple[float, float]:
        return self.show([(text, conf, _line_box(text, **geometry))], count)

    def blank(self, count: int, same_pixels: bool = False) -> None:
        self.show([], count, same_pixels)

    def build(self) -> list[SubtitleEvent]:
        return SubtitleBuilder(SubtitleBuilderConfig(**_RUN_CONFIG)).build(self.frames, roi=_ROI)

    def tail(self, skip: int = len(_LINES)) -> list[tuple[str, float, float]]:
        return [(event.text, round(event.start_sec, 2), round(event.end_sec, 2)) for event in self.build()[skip:]]


# ── Mất khung (bộ giải mã bỏ khung / chạy tiếp từ checkpoint) ────────────────────────────────────────────────────────


def test_lost_change_frame_does_not_glue_blank_copies_onto_the_line() -> None:
    """"瑶瑶": khung đổi hình (rỗng) ngay sau câu bị mất; các khung rỗng mang cờ "cùng điểm ảnh" từng bị coi là bản chép
    của "瑶瑶" → câu kéo dài 0,36 s."""
    film = _Film()
    start, end = film.say("瑶瑶", 15)
    film.skip.add(round(film.clock + 0.04, 2))
    film.blank(1)
    film.blank(8, same_pixels=True)
    film.say("你出关了", 20)
    film.blank(8)
    assert film.tail()[0] == ("瑶瑶", start, end)


def test_chain_break_only_where_frames_are_missing() -> None:
    frames = [_frame(1.00, []), _frame(1.04, [], continues=True), _frame(1.20, [], continues=True)]
    fixed, broken = orchestrator.break_same_pixel_chains_at_gaps(frames, 0.04)
    assert broken == 1 and [frame.continues_previous for frame in fixed] == [False, True, False]
    assert orchestrator.break_same_pixel_chains_at_gaps(frames, 0.0) == (frames, 0)


def test_missing_frames_inside_a_line_are_not_counted_as_blank() -> None:
    """"一天都拦不住": ba khung giữa câu bị MẤT (không có trong dữ liệu) từng bị đếm thành ba khung trống → câu lặp."""
    film = _Film()
    start, _ = film.say("一天都拦不住", 17)
    for _ in range(3):
        film.skip.add(round(film.clock + 0.04, 2))
        film.blank(1)
    _, end = film.say("一天都拦不住", 25)
    film.blank(8)
    assert film.tail() == [("一天都拦不住", start, end)]


@pytest.mark.parametrize("jitter", [0.008, -0.008])
def test_real_repeat_survives_vfr_timestamp_jitter(jitter: float) -> None:
    """Câu lặp thật cách 4 khung trống: mốc rung ± 8 ms từng làm tròn khoảng hở ra 3 khung → gộp nhầm."""
    film = _Film()
    film.say("我走着走着", 12)
    film.blank(4)
    film.say("我走着走着", 12)
    film.blank(8)
    shaken = [
        OcrFrameResult(
            frame_index=frame.frame_index, text_boxes=frame.text_boxes, continues_previous=frame.continues_previous,
            timestamp_sec=round(frame.timestamp_sec + (jitter if index % 2 else -jitter), 4),
        )
        for index, frame in enumerate(film.frames)
    ]
    film.frames = shaken
    assert [text for text, _start, _end in film.tail()] == ["我走着走着", "我走着走着"]


# ── Dấu gạch đầu câu ─────────────────────────────────────────────────────────────────────────────────────────────────


def test_dash_added_to_one_read_of_a_centred_line_does_not_become_yi() -> None:
    """OCR thêm "—" vào MỘT lần đọc (5 khung) của câu đúng tâm: mã cũ ra "一够了" (58 câu ở thử sức)."""
    film = _Film()
    film.say("够了", 10)
    film.say("—够了", 5)
    film.say("够了", 10)
    film.blank(8)
    assert [text for text, _start, _end in film.tail()] == ["够了"]


def test_dropped_leading_yi_with_right_shifted_box_is_restored() -> None:
    """Bộ 3.23.690: "天呐" thiếu "一" lệch phải +20 px (chữ ≈ 58 px); bản đọc "-天呐" cũng lệch phải."""
    film = _Film()
    film.say("-天呐", 8, center=_CENTER + 20)
    film.say("天呐", 17, center=_CENTER + 20)
    film.blank(8)
    assert [text for text, _start, _end in film.tail()] == ["一天呐"]


# ── Logo cố định trong vùng chữ ──────────────────────────────────────────────────────────────────────────────────────


_LOGO: Box = ("@DuanJu88", 0.93, (8, 6, 150, 30))
_FULL_HEIGHT = (319, 0, 402, 136)


def test_static_logo_does_not_hide_single_char_lines() -> None:
    """"是" (hộp cao hết vùng chữ ở mọi khung) cần luật "khung một hộp"; logo ở mọi khung từng tắt luật đó → mất câu."""
    film = _Film(extra=[_LOGO])
    start, _end = film.show([("是", 1.0, _FULL_HEIGHT)], 18)
    film.blank(8)
    film.blank(250)                             # logo hiện liên tục > 10 s
    tail = film.tail()
    assert tail[0][0] == "是" and tail[0][1] == start
    assert all("@" not in text for text, _start, _end in film.tail(0))


def test_overlay_detection_needs_a_run_longer_than_any_line() -> None:
    frames = [_frame(round(0.04 * i, 2), [_LOGO]) for i in range(200)]               # 8 s: có thể là câu dài
    assert spatial_filters.strip_static_overlays(frames, 0.04) == (frames, [])
    lines = _LINES[:3]
    frames = [
        _frame(round(0.04 * i, 2), [_LOGO, (lines[i // 25 % 3], 1.0, _line_box(lines[i // 25 % 3]))])
        for i in range(300)                                                         # câu đổi mỗi giây, logo 12 s
    ]
    cleaned, texts = spatial_filters.strip_static_overlays(frames, 0.04)
    assert texts == ["@DuanJu88"] and all([box.text for box in frame.text_boxes] == [lines[i // 25 % 3]]
                                          for i, frame in enumerate(cleaned))
    assert spatial_filters.strip_static_overlays(frames, 0.0) == (frames, [])


# ── Đổi bố cục phụ đề giữa phim (phim nối nhiều tập) ─────────────────────────────────────────────────────────────────


def _two_layout_film(dx: float, dy: float) -> tuple[_Film, float, float]:
    film = _Film(intro=False)
    for minute in range(4):
        for index, line in enumerate(_LINES):
            film.say(line, 30)
            film.blank(8 + (minute + index) % 3)
        film.blank(1500)                         # mỗi "tập" ≈ 1 phút
    for minute in range(4):
        for index, line in enumerate(_LINES):
            film.say(line, 30, center=_CENTER + dx, top=65 + dy, bottom=133 + dy)
            film.blank(8 + (minute + index) % 3)
        film.blank(1500)
    start, end = film.show([("你", 1.0, (319 + dx, 0, 402 + dx, 136))], 18)
    film.blank(8)
    return film, start, end


@pytest.mark.parametrize(("dx", "dy"), [(40.0, 0.0), (0.0, -25.0)])
def test_layout_change_mid_film_is_built_per_segment(dx: float, dy: float) -> None:
    film, start, _ = _two_layout_film(dx, dy)
    cuts = orchestrator.find_layout_cuts(sorted(film.frames, key=lambda f: f.timestamp_sec), _ROI)
    assert len(cuts) == 1
    events = film.build()
    assert len(events) == 2 * 4 * len(_LINES) + 1
    assert events[-1].text == "你" and round(events[-1].start_sec, 2) == start


def test_single_layout_film_is_not_cut() -> None:
    film, _start, _end = _two_layout_film(0.0, 0.0)
    assert orchestrator.find_layout_cuts(sorted(film.frames, key=lambda f: f.timestamp_sec), _ROI) == []


# ── Vùng chữ cắt chân, chữ phồn lẻ tẻ ────────────────────────────────────────────────────────────────────────────────


def test_too_wide_check_uses_char_pitch_when_glyphs_are_clipped() -> None:
    """Vùng chữ cắt 25 px chân chữ: cao 45 px, rộng 55 px/chữ → hộp "哼" 93 px từng bị coi là quá rộng."""
    box = OcrTextBox(text="哼", confidence=Confidence(0.81), polygon=[(302, 62), (395, 62), (395, 111), (302, 111)])
    assert not box_geometry.is_plausible_subtitle_box(box, 45.0)
    assert box_geometry.is_plausible_subtitle_box(box, 45.0, char_pitch=55.0)


def test_traditional_variant_read_stays_one_line() -> None:
    """"出来了是吧" 12 khung rồi "出來了是吧" 11 khung (phồn lẻ tẻ): từng tách làm hai, câu đầu kết thúc sớm 0,44 s."""
    film = _Film()
    start, _ = film.say("出来了是吧", 12)
    _, end = film.say("出來了是吧", 11)
    film.blank(8)
    tail = film.tail()
    assert len(tail) == 1 and tail[0][1:] == (start, end)
