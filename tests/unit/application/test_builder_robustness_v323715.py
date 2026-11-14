"""[v3.23.715] Khâu dựng bền với hộp chữ phình toàn lượt, bước lấy mẫu thật, mảnh OCR hụt và khung nhiễu sát biên.

Lượt chạy thật 3.23.713 của chủ dự án (phim nối 86 tập, tiền xử lý không viền, khoảng gộp 0,1 s): mất câu "是" và "你"
(hộp cao hết vùng chữ ở mọi khung), "切" / "你" kết thúc sớm vì OCR hụt 1–2 khung, và nhiều mốc lệch một khung vì biên
câu bị nới bằng khung nhiễu ("N", "V", "11!").
"""

from __future__ import annotations

from subtitles_extractor.application.dtos.extract_subtitles_dto import SubtitleBuilderConfig
from subtitles_extractor.application.services.subtitle_builder import SubtitleBuilder
from subtitles_extractor.application.services.subtitle_pipeline.orchestrator import measure_sample_step
from subtitles_extractor.application.services.subtitle_pipeline.read_fusion import (
    measure_text_band,
    snap_inflated_boxes_to_band,
)
from subtitles_extractor.domain.entities.ocr_frame_result import OcrFrameResult, OcrTextBox
from subtitles_extractor.domain.value_objects.confidence import Confidence
from subtitles_extractor.domain.value_objects.roi import Roi

_ROI = Roi(x=0, y=774, width=720, height=136)
_LINES = ("修炼一途的", "自炼气始吧", "经筑基之后", "寻求飞升啊", "检测到宿主", "即将面临了", "历史至强雷", "四灵天雷劫")
_LINE_BOX = (222, 65, 498, 133)
_FULL_HEIGHT = (319, 0, 402, 136)        # hộp "是" thật: cao hết vùng chữ, đúng tâm, rộng 1,5 chữ


def _read(ts: float, text: str, box: tuple[int, int, int, int], conf: float = 1.0) -> OcrFrameResult:
    x0, y0, x1, y1 = box
    return OcrFrameResult(
        frame_index=round(ts * 25), timestamp_sec=ts,
        text_boxes=[OcrTextBox(text=text, confidence=Confidence(conf), polygon=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)])],
    )


class _Film:
    """Phim giả bước 0,04 s: 8 câu thường (để khâu dựng đo được cỡ chữ, tâm, dải phụ đề) rồi tới phần cần thử."""

    def __init__(self) -> None:
        self.frames: list[OcrFrameResult] = []
        self.clock = 0.0
        for line in _LINES:
            self.show(line, _LINE_BOX, 30)
            self.blank(8)

    def _tick(self) -> float:
        self.clock = round(self.clock + 0.04, 2)
        return self.clock

    def show(self, text: str, box: tuple[int, int, int, int], count: int, conf: float = 1.0) -> tuple[float, float]:
        """Trả ``(mốc khung đầu, mốc kết thúc = khung cuối + một bước)``."""
        first = round(self.clock + 0.04, 2)
        for _ in range(count):
            self.frames.append(_read(self._tick(), text, box, conf))
        return first, round(self.clock + 0.04, 2)

    def blank(self, count: int) -> None:
        for _ in range(count):
            ts = self._tick()
            self.frames.append(OcrFrameResult(frame_index=round(ts * 25), timestamp_sec=ts, text_boxes=[]))

    def build(self, **config: float) -> list:
        settings = {"sample_step_sec": 0.04, "merge_gap_sec": 0.2, "min_duration_sec": 0.15} | config
        return SubtitleBuilder(SubtitleBuilderConfig(**settings)).build(self.frames, roi=_ROI)


def _near(value: float, expected: float) -> bool:
    return abs(value - expected) < 0.011


def _ends_at(value: float, expected: float) -> bool:
    """Mốc cuối = khung cuối + một bước, cộng tối đa phần đệm cuối câu (0,05 s) khi sau câu là khoảng trống."""
    return expected - 0.011 <= value <= expected + 0.061


# ── Hộp phình toàn lượt → dải phụ đề của phim ─────────────────────────────────────────────────────────────────────────


def _geometry(film: _Film) -> tuple[float, float, float, tuple[float, float] | None]:
    return 360.0, 55.2, 68.0, measure_text_band(film.frames)


def test_band_is_measured_from_the_film() -> None:
    film = _Film()
    assert measure_text_band(film.frames) == (65.0, 133.0)
    assert measure_text_band(film.frames[:10]) is None


def test_full_height_single_char_box_snaps_to_the_band() -> None:
    film = _Film()
    frame = _read(900.0, "是", _FULL_HEIGHT)
    snapped, fixed = snap_inflated_boxes_to_band([frame], *_geometry(film))
    assert fixed == 1
    ys = sorted({point[1] for point in snapped[0].text_boxes[0].polygon})
    xs = sorted({point[0] for point in snapped[0].text_boxes[0].polygon})
    assert ys == [65, 133] and xs == [319, 402]
    assert snapped[0].text_boxes[0].text == "是" and float(snapped[0].text_boxes[0].confidence) == 1.0


def test_scene_noise_and_normal_boxes_are_left_alone() -> None:
    """Nhiễu đo thật: lệch tâm ≥ 40 px, hoặc rộng dư ≥ 1,2 chữ, hoặc không phải chữ Hán — không được đưa về dải."""
    film = _Film()
    frames = [
        _read(900.00, "N", (320, 0, 402, 136)),               # không phải chữ Hán
        _read(900.04, "小", (324, 0, 620, 135)),              # lệch tâm 112 px, rộng 5 chữ
        _read(900.08, "上", (595, 0, 720, 116)),              # mép phải
        _read(900.12, "是", (290, 0, 430, 136)),              # đúng tâm nhưng rộng 2,5 chữ
        _read(900.16, "是", _FULL_HEIGHT, 0.59),              # tin cậy thấp
        _read(900.20, "是", (322, 62, 397, 136)),             # hộp bình thường
        _read(900.24, "是", (319, 0, 402, 90)),               # không trùm dải phụ đề (vật ở nửa trên)
    ]
    snapped, fixed = snap_inflated_boxes_to_band(frames, *_geometry(film))
    assert fixed == 0 and snapped == frames
    assert snap_inflated_boxes_to_band(frames, None, 55.2, 68.0, (65.0, 133.0))[1] == 0      # không biết tâm → không làm gì


def test_builder_keeps_line_whose_every_box_is_inflated() -> None:
    """"是" 2654,64–2655,36 s: cả 18 khung hộp 0–136 px → mã cũ để bộ lọc lệch dọc xoá cả câu."""
    film = _Film()
    start, end = film.show("是", _FULL_HEIGHT, 18)
    film.blank(8)
    events = film.build(merge_gap_sec=0.1)
    assert [event.text for event in events] == [*_LINES, "是"]
    assert _near(events[-1].start_sec, start) and _ends_at(events[-1].end_sec, end)


# ── Bước lấy mẫu thật ────────────────────────────────────────────────────────────────────────────────────────────────


def test_sample_step_is_measured_from_frame_times() -> None:
    film = _Film()
    assert measure_sample_step(film.frames) == 0.04
    assert measure_sample_step(film.frames[:5]) is None


def test_line_with_inflated_boxes_and_a_two_frame_hole_stays_whole() -> None:
    """"你" 3269,32–3270,16 s của lượt chạy thật (bước cấu hình 0,03 s, khung cách nhau 0,04 s, khoảng gộp 0,1 s):
    2 khung hộp cao, 12 khung thường, OCR hụt 2 khung, 1 khung hộp cao hết vùng chữ, 2 khung thường.

    Mã cũ: 2 khung đầu bị lọc, hụt 2 khung bị đếm thành 3 khung trống ("câu lặp"), mảnh cuối bị xoá → còn 12 khung.
    """
    film = _Film()
    start, _ = film.show("你", (317, 26, 403, 136), 2)
    film.show("你", (319, 61, 397, 136), 12)
    film.blank(2)
    film.show("你", (318, 0, 402, 136), 1, conf=0.99)
    _, end = film.show("你", (319, 52, 400, 136), 2)
    film.blank(2)
    film.show("就算他现在金丹", (156, 66, 564, 132), 25)
    film.blank(8)
    events = film.build(sample_step_sec=0.03, merge_gap_sec=0.1, alignment_center_tolerance_ratio=0.05)
    assert [event.text for event in events] == [*_LINES, "你", "就算他现在金丹"]
    assert events[-2].frame_count == 17
    assert _near(events[-2].start_sec, start) and _ends_at(events[-2].end_sec, end)


# ── Mảnh OCR hụt được ghép, câu lặp thật giữ riêng ───────────────────────────────────────────────────────────────────


def test_single_frame_fragment_after_one_missed_frame_extends_the_line_to_its_true_end() -> None:
    """"切" 1729,36–1730,16 s: 17 khung, hụt 1 khung, 1 khung đọc yếu, đọc lại 1 khung (hộp cao hết vùng chữ).

    Mã cũ (khoảng gộp 0,1 s) xoá mảnh → câu kết thúc sớm 0,12 s; ghép theo mốc đã đệm thì lại trễ 0,15 s.
    """
    film = _Film()
    start, _ = film.show("切", (324, 66, 395, 134), 17)
    film.blank(1)
    film.show("切", (317, 0, 401, 136), 1, conf=0.59)          # khung đọc yếu, hộp cao hết vùng chữ
    _, end = film.show("切", (322, 0, 397, 136), 1)
    film.blank(8)
    events = film.build(merge_gap_sec=0.1)
    assert [event.text for event in events] == [*_LINES, "切"]
    assert events[-1].frame_count == 18            # mã cũ: 17 khung, khung đọc lại bị xoá
    assert _near(events[-1].start_sec, start) and _ends_at(events[-1].end_sec, end)


def test_real_repeat_with_scene_noise_in_the_gap_stays_two_lines() -> None:
    """"散了吧" nói hai lần cách 4 khung (lần đầu chỉ 7 khung = 0,28 s); trong khoảng hở OCR đọc ra vật trong cảnh ("1").

    Luật ghép mảnh không được coi 4 khung hở là OCR hụt — kể cả khi khung nhiễu làm "khung có chữ cuối cùng" nằm sát
    câu sau (bản nháp của luật từng gộp nhầm cặp này trên hai bộ dữ liệu vùng chữ rộng).
    """
    film = _Film()
    film.show("散了吧", (267, 64, 454, 136), 7)
    film.show("1", (5, 5, 226, 132), 4, conf=0.7)
    film.show("散了吧", (267, 64, 454, 136), 18)
    film.blank(8)
    texts = [event.text for event in film.build(merge_gap_sec=0.1)]
    assert texts[-2:] == ["散了吧", "散了吧"], texts


# ── Biên câu không nới bằng khung nhiễu ──────────────────────────────────────────────────────────────────────────────


def test_edges_are_not_extended_onto_unrelated_noise_frames() -> None:
    """"N" (hộp cao hết vùng chữ) 3 khung ngay trước câu và "V" ngay sau: ảnh thật cho thấy lúc đó chưa / hết chữ."""
    film = _Film()
    film.show("N", (8, 1, 214, 136), 3, conf=0.98)
    start, end = film.show("我只是毒性发作", (170, 67, 550, 132), 30)
    film.show("V", (356, 20, 495, 123), 6)
    film.blank(8)
    events = film.build()
    line = next(event for event in events if event.text == "我只是毒性发作")
    assert _near(line.start_sec, start) and _ends_at(line.end_sec, end)


def test_edge_is_extended_onto_a_misread_frame_of_the_same_line() -> None:
    """"哼" 1386,84 s: khung đầu đọc thành "敬哼" (0,47, hộp cao hết vùng chữ) — chữ đã hiện, câu phải bắt đầu từ đó."""
    film = _Film()
    start, _ = film.show("敬哼", (310, 0, 403, 136), 1, conf=0.47)
    _, end = film.show("哼", (320, 50, 397, 131), 12, conf=0.91)
    film.blank(8)
    events = film.build()
    assert events[-1].text == "哼"
    assert _near(events[-1].start_sec, start) and _ends_at(events[-1].end_sec, end)
