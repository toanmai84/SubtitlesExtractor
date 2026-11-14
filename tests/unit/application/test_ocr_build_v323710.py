"""[v3.23.710] Dựng phụ đề từ OCR: tâm phụ đề đo từ phim, hộp rìa yếu, chữ nhỏ không phải chữ Hán, câu một chữ.

Lượt chạy thật phim nối 86 tập bằng bản 3.23.708 (3663 câu đáp án): 3 câu sai, 7 câu sót, 2 câu gộp, 1 câu rác "000".
Phát lại trên đúng dữ liệu OCR thô đó: khớp 3644 → 3648, gộp 2 → 0, rác 1 → 0 (dữ liệu bản 3.23.690: giữ 3656).
"""

from __future__ import annotations

from types import SimpleNamespace

from subtitles_extractor.application.dtos.extract_subtitles_dto import SubtitleBuilderConfig
from subtitles_extractor.application.services.subtitle_pipeline.geometry_check import (
    OCR_NOTE_MISSING_CHAR,
    accept_reocr_text,
    drop_small_non_han_events,
)
from subtitles_extractor.application.services.subtitle_pipeline.spatial_filters import (
    CENTER_MIN_BOXES,
    clean_spatial_outliers,
    measure_subtitle_center,
)
from subtitles_extractor.domain.entities.ocr_frame_result import OcrFrameResult, OcrTextBox
from subtitles_extractor.domain.value_objects.confidence import Confidence
from subtitles_extractor.domain.value_objects.roi import Roi, TextAlignment

ROI = Roi(x=35, y=774, width=644, height=136)           # tâm vùng chữ 322 px
SUB_CENTER = 290.0                                      # tâm phụ đề trong dữ liệu thô thật


def _box(text: str, conf: float, left: float, right: float, top: float = 60, bottom: float = 120) -> OcrTextBox:
    return OcrTextBox(text=text, confidence=Confidence(conf),
                      polygon=[(int(left), int(top)), (int(right), int(top)), (int(right), int(bottom)), (int(left), int(bottom))])


def _centered(text: str, conf: float, center: float = SUB_CENTER) -> OcrTextBox:
    width = 49.8 * len(text) + 17.4
    return _box(text, conf, center - width / 2, center + width / 2)


def _frame(ts: float, *boxes: OcrTextBox) -> OcrFrameResult:
    return OcrFrameResult(frame_index=int(ts * 25), timestamp_sec=ts, text_boxes=list(boxes))


def _film(center: float = SUB_CENTER, count: int = CENTER_MIN_BOXES + 20) -> list[OcrFrameResult]:
    words = ["明白明白", "说这个的时候", "元婴大圆满", "不可能"]
    return [_frame(0.04 * i, _centered(words[i % 4], 0.99, center)) for i in range(count)]


class TestMeasuredCenter:
    def test_center_is_measured_from_the_film(self) -> None:
        assert abs(measure_subtitle_center(_film(), ROI) - SUB_CENTER) <= 1.0

    def test_no_verdict_without_enough_data_or_when_not_centred(self) -> None:
        assert measure_subtitle_center(_film(count=CENTER_MIN_BOXES - 1), ROI) is None
        assert measure_subtitle_center(_film(), None) is None
        left = Roi(x=0, y=0, width=644, height=136, alignment=TextAlignment.LEFT)
        assert measure_subtitle_center(_film(), left) is None
        # phụ đề căn trái (mép trái cố định) → tâm trải theo độ dài câu → không kết luận
        ragged = [_frame(0.04 * i, _box("字" * (2 + i % 9), 0.99, 100, 100 + 50 * (2 + i % 9))) for i in range(400)]
        assert measure_subtitle_center(ragged, ROI) is None

    def test_low_confidence_single_char_at_true_center_is_kept(self) -> None:
        """Câu thật "哼" (0,55; tâm 288,5) từng bị loại vì so với tâm vùng chữ 322 (dung sai 26 px)."""
        config = SubtitleBuilderConfig()
        frame = _frame(968.52, _box("哼", 0.55, 258, 319))
        assert clean_spatial_outliers(frame, ROI, config).is_empty                       # hành vi cũ: mất câu
        kept = clean_spatial_outliers(frame, ROI, config, measure_subtitle_center(_film(), ROI))
        assert [box.text for box in kept.text_boxes] == ["哼"]

    def test_off_center_noise_is_still_dropped(self) -> None:
        config = SubtitleBuilderConfig()
        frame = _frame(1.0, _box("土", 0.55, 420, 480))
        assert clean_spatial_outliers(frame, ROI, config, SUB_CENTER).is_empty


class TestWeakEdgeBox:
    def test_low_confidence_side_box_is_stripped(self) -> None:
        """Phim mẫu 3122 s: "元婴大圆满" (0,97; 158–421) + "小川" (0,49; 456–523) → từng ra "元婴大圆满小川"."""
        frame = _frame(3122.4, _box("元婴大圆满", 0.97, 158, 421, 60, 119), _box("小川", 0.49, 456, 523, 64, 110))
        kept = clean_spatial_outliers(frame, ROI, SubtitleBuilderConfig(), SUB_CENTER)
        assert [box.text for box in kept.text_boxes] == ["元婴大圆满"]

    def test_confident_side_box_beside_a_self_centred_line_goes_and_centred_pair_is_kept(self) -> None:
        config = SubtitleBuilderConfig()
        # [v3.23.717] Hộp chính TỰ NÓ đúng tâm thì hộp bên cạnh — dù tin cậy cao — không thể là một phần của câu (nó kéo tâm
        # cả dòng lệch đi): trước 3.23.717 hộp này được giữ, thử sức trên OCR thật cho thấy đó là chữ cảnh dính vào câu.
        confident = _frame(1.0, _box("元婴大圆满", 0.97, 158, 421), _box("小川", 0.95, 456, 523))
        kept = clean_spatial_outliers(confident, ROI, config, SUB_CENTER)
        assert [box.text for box in kept.text_boxes] == ["元婴大圆满"]
        # câu hai hộp cách nhau bằng dấu cách, cả dòng đúng tâm: hộp tin cậy thấp vẫn là một phần của câu
        pair = _frame(2.0, _box("师叔", 0.99, 150, 270), _box("你好", 0.50, 310, 430))
        assert len(clean_spatial_outliers(pair, ROI, config, SUB_CENTER).text_boxes) == 2

    def test_without_measured_center_nothing_is_stripped(self) -> None:
        frame = _frame(3122.4, _box("元婴大圆满", 0.97, 190, 453), _box("小川", 0.49, 488, 555))
        assert len(clean_spatial_outliers(frame, ROI, SubtitleBuilderConfig()).text_boxes) == 2


class TestSmallNonHan:
    @staticmethod
    def _events_and_frames(extra_text: str, extra_height: float) -> tuple[list, list]:
        events, frames, clock = [], [], 0.0
        lines = [("怎么回事", 61.0)] * 40 + [(extra_text, extra_height)]
        for text, height in lines:
            events.append(SimpleNamespace(text=text, start_sec=clock, end_sec=clock + 0.5, ocr_notes=()))
            for step in range(6):
                frames.append(SimpleNamespace(
                    timestamp_sec=clock + 0.08 * step,
                    text_boxes=[SimpleNamespace(text=text, polygon=[(200, 60), (380, 60), (380, 60 + height), (200, 60 + height)])],
                ))
            clock += 1.0
        return events, frames

    def test_small_digits_are_dropped(self) -> None:
        """Dòng 3308 của lượt chạy thật: "000", 6 khung, tin cậy 0,98, hộp cao 39 px (chữ phụ đề 61 px)."""
        events, frames = self._events_and_frames("000", 39.0)
        kept, dropped = drop_small_non_han_events(events, frames)
        assert dropped == 1 and all(event.text != "000" for event in kept)

    def test_full_size_latin_and_small_han_are_kept(self) -> None:
        events, frames = self._events_and_frames("OK", 58.0)
        assert drop_small_non_han_events(events, frames)[1] == 0
        events, frames = self._events_and_frames("嗯", 44.0)          # chữ Hán thấp nhất phim: 0,72 lần
        assert drop_small_non_han_events(events, frames)[1] == 0

    def test_too_few_reference_lines_keep_everything(self) -> None:
        events, frames = self._events_and_frames("000", 39.0)
        assert drop_small_non_han_events(events[-10:], frames)[1] == 0


class TestAcceptTwoYi:
    def test_two_missing_yi_are_accepted(self) -> None:
        assert accept_reocr_text(
            "百道雷劫起劈下来", OCR_NOTE_MISSING_CHAR, ["一百道雷劫一起劈下来"] * 3 + ["百道雷劫起劈下来"],
        ) == "一百道雷劫一起劈下来"

    def test_two_other_chars_are_still_rejected(self) -> None:
        assert accept_reocr_text("百道雷劫起劈下来", OCR_NOTE_MISSING_CHAR, ["三百道雷劫又起劈下来"] * 4) is None
        assert accept_reocr_text("雷劫", OCR_NOTE_MISSING_CHAR, ["一一一雷劫"] * 4) is None


class TestSingleCharOnSubtitleLine:
    @staticmethod
    def _events(single_box: tuple[int, int, int, int], frames: int, conf: float = 0.55) -> list:
        from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
        from subtitles_extractor.domain.value_objects.time_interval import TimeInterval

        events = [
            SubtitleEvent(index=i + 1, text="怎么回事", interval=TimeInterval(i * 2.0, i * 2.0 + 1.0),
                          confidence=Confidence(0.99), frame_count=25, bounding_box=(182, 60, 398, 120))
            for i in range(40)
        ]
        events.append(SubtitleEvent(index=41, text="哼", interval=TimeInterval(100.0, 100.5), confidence=Confidence(conf),
                                    frame_count=frames, bounding_box=single_box))
        return events

    def _kept(self, events: list) -> bool:
        from subtitles_extractor.application.services.subtitle_pipeline.event_filters import filter_short_text_events

        return any(event.text == "哼" for event in filter_short_text_events(events, SubtitleBuilderConfig()))

    def test_real_low_confidence_interjection_is_kept(self) -> None:
        assert self._kept(self._events((258, 60, 319, 120), frames=11))          # mã cũ: bị bỏ

    def test_off_center_wrong_size_or_brief_single_char_is_still_dropped(self) -> None:
        assert not self._kept(self._events((420, 60, 481, 120), frames=11))      # lệch tâm
        assert not self._kept(self._events((270, 70, 310, 108), frames=11))      # chữ nhỏ (0,63 lần)
        assert not self._kept(self._events((258, 60, 319, 120), frames=3))       # chỉ thoáng qua
        assert not self._kept(self._events((258, 60, 319, 120), frames=11, conf=0.40))


def test_bilingual_film_keeps_small_latin_lines() -> None:
    """Phim song ngữ: dòng tiếng Anh cỡ chữ nhỏ là phụ đề thật — luật "chữ nhỏ không phải chữ Hán" không được áp."""
    events, frames = TestSmallNonHan._events_and_frames("000", 39.0)
    clock = 100.0
    for _ in range(6):
        events.append(SimpleNamespace(text="Hello there", start_sec=clock, end_sec=clock + 0.5, ocr_notes=()))
        for step in range(6):
            frames.append(SimpleNamespace(
                timestamp_sec=clock + 0.08 * step,
                text_boxes=[SimpleNamespace(text="Hello there", polygon=[(200, 60), (380, 60), (380, 99), (200, 99)])],
            ))
        clock += 1.0
    assert drop_small_non_han_events(events, frames)[1] == 0


def test_vertical_subtitles_keep_the_roi_center() -> None:
    from subtitles_extractor.domain.value_objects.roi import TextOrientation

    vertical = Roi(x=0, y=0, width=644, height=136, orientation=TextOrientation.VERTICAL)
    assert measure_subtitle_center(_film(), vertical) is None


class TestSmallFilmGeometry:
    """Video ngắn (một tập ~30 câu): trước đây không số chữ nào đủ 5 câu → không gắn được cờ, không đọc lại."""

    @staticmethod
    def _measured(extra: list) -> list:
        from subtitles_extractor.application.services.subtitle_pipeline.geometry_check import EventGeometry

        # phân bố số chữ thật của tập 77: 2×5, 3×4, 4×9, 6×3, 7×3, 8×2, 9×2, 10×1
        counts = [2] * 5 + [3] * 4 + [4] * 9 + [6] * 3 + [7] * 3 + [8] * 2 + [9] * 2 + [10]
        items = [EventGeometry(i, n, 55.0 * n + 19.0 + (i % 3 - 1) * 4.0, 322.0 + (i % 2) * 3.0) for i, n in enumerate(counts)]
        return items + extra

    def test_missing_leading_char_is_flagged_in_a_short_video(self) -> None:
        from subtitles_extractor.application.services.subtitle_pipeline.geometry_check import EventGeometry, geometry_notes

        notes = geometry_notes(self._measured([EventGeometry(29, 7, 406.0, 351.0)]))      # "眼就察觉到了吗" thật
        assert notes == {29: OCR_NOTE_MISSING_CHAR}                                       # mã cũ: {}

    def test_clean_short_video_and_too_small_samples_give_no_flags(self) -> None:
        from subtitles_extractor.application.services.subtitle_pipeline.geometry_check import EventGeometry, geometry_notes

        assert geometry_notes(self._measured([])) == {}
        assert geometry_notes(self._measured([])[:14]) == {}
        assert geometry_notes([EventGeometry(i, 4, 239.0, 322.0) for i in range(20)]) == {}   # một cỡ số chữ
