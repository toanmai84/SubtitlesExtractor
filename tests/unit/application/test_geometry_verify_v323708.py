"""[v3.23.708] Vòng tự soát OCR: gắn cờ câu đọc thiếu / thừa chữ bằng hình học hộp chữ rồi OCR lại đúng các câu đó.

Số liệu thật (phim nối 86 tập, có tiền xử lý ảnh): bề rộng hộp ≈ 49,8 × số chữ + 17,4 px, tâm 293,5 px. Các câu sai
còn lại sau 3.23.707: "其实我直都在" (hộp rộng hơn +1,18 chữ), "是你师伯唯的弟子" (+1,09), "成功解除余一部分" (−0,98),
"个万年炼气的人" (lệch tâm +0,46 chữ), "任务" (−0,49). Trích xuất thật trên 4 tập (14, 15, 26, 02): cả 4 câu thiếu
"一" được đọc lại bằng ảnh gốc và sửa đúng.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest

from subtitles_extractor.application.services.ocr_suspicion import (
    SUSPECT_EXTRA_CHAR,
    SUSPECT_MISSING_CHAR,
    ocr_suspicions,
)
from subtitles_extractor.application.services.subtitle_pipeline.geometry_check import (
    OCR_NOTE_EXTRA_CHAR,
    OCR_NOTE_MISSING_CHAR,
    EventGeometry,
    accept_reocr_text,
    geometry_notes,
    mark_geometry_suspects,
    measure_events,
)

SLOPE, INTERCEPT, CENTER = 49.8, 17.4, 293.5
WORDS = ["师叔", "不可能", "怎么回事", "你说什么呢", "这是怎么回事", "我今天就告诉你", "你到底想要怎么样", "今天我们就到这里吧"]


def _box(text: str, width: float, center: float) -> SimpleNamespace:
    left, right = center - width / 2.0, center + width / 2.0
    return SimpleNamespace(text=text, polygon=[(left, 60), (right, 60), (right, 122), (left, 122)])


def _film(extra: list[tuple[str, float, float]]) -> tuple[list, list]:
    """Phim giả: 40 câu chuẩn + các câu ``(chữ, lệch bề rộng tính theo chữ, lệch tâm tính theo chữ)``."""
    events, frames, clock = [], [], 0.0
    lines = [(WORDS[i % len(WORDS)], 0.0, 0.0) for i in range(40)] + extra
    for text, width_chars, offset_chars in lines:
        width = SLOPE * (len(text) + width_chars) + INTERCEPT
        center = CENTER + offset_chars * SLOPE
        events.append(SimpleNamespace(text=text, start_sec=clock, end_sec=clock + 0.9, ocr_notes=()))
        for step in range(8):
            frames.append(SimpleNamespace(timestamp_sec=clock + 0.1 * step, text_boxes=[_box(text, width, center)]))
        clock += 1.0
    return events, frames


class TestGeometryNotes:
    def test_real_deviations_are_flagged(self) -> None:
        events, frames = _film([
            ("其实我直都在", 1.18, 0.0), ("是你师伯唯的弟子", 1.09, -0.05), ("成功解除余一部分", -0.98, -0.01),
            ("个万年炼气的人", -0.02, 0.46), ("任务", -0.06, -0.49),
        ])
        assert mark_geometry_suspects(events, frames) == 5
        flagged = {e.text: e.ocr_notes for e in events if e.ocr_notes}
        assert flagged == {
            "其实我直都在": (OCR_NOTE_MISSING_CHAR,), "是你师伯唯的弟子": (OCR_NOTE_MISSING_CHAR,),
            "成功解除余一部分": (OCR_NOTE_EXTRA_CHAR,), "个万年炼气的人": (OCR_NOTE_MISSING_CHAR,),
            "任务": (OCR_NOTE_MISSING_CHAR,),
        }

    def test_largest_deviations_of_correct_lines_are_not_flagged(self) -> None:
        # Câu ĐÚNG lệch nhiều nhất trong dữ liệu thật: +0,52 / −0,63 chữ, lệch tâm 0,34 chữ.
        events, frames = _film([("你为什么会成为", 0.52, 0.27), ("你们这些魔门的小可爱", -0.63, 0.0),
                                ("一百道雷劫一起劈下来", -0.49, 0.34)])
        assert mark_geometry_suspects(events, frames) == 0

    def test_too_few_lines_or_flat_widths_give_no_verdict(self) -> None:
        assert geometry_notes([EventGeometry(i, 4, 400.0, CENTER) for i in range(10)]) == {}
        flat = [EventGeometry(i, 2 + i % 6, 300.0, CENTER) for i in range(60)]
        assert geometry_notes(flat) == {}
        assert geometry_notes([]) == {}

    def test_uncentred_subtitles_skip_the_offset_check(self) -> None:
        # Phụ đề căn trái: tâm đổi theo độ dài câu → không được kết luận "lệch tâm".
        measured = [
            EventGeometry(i, n, SLOPE * n + INTERCEPT, 100.0 + (SLOPE * n + INTERCEPT) / 2.0)
            for i, n in enumerate([2, 3, 4, 5, 6, 7, 8, 9] * 6)
        ]
        assert geometry_notes(measured) == {}

    def test_measure_skips_multiline_latin_and_sparse_lines(self) -> None:
        frames = [SimpleNamespace(timestamp_sec=0.1 * i, text_boxes=[_box("师叔", 117.0, CENTER)]) for i in range(8)]
        events = [
            SimpleNamespace(text="师叔\n你好", start_sec=0.0, end_sec=0.7), SimpleNamespace(text="OK师叔", start_sec=0.0, end_sec=0.7),
            SimpleNamespace(text="师", start_sec=0.0, end_sec=0.7), SimpleNamespace(text="师叔", start_sec=0.0, end_sec=0.15),
            SimpleNamespace(text="师叔", start_sec=0.0, end_sec=0.7),
        ]
        assert [item.position for item in measure_events(events, frames)] == [4]

    def test_suspicion_reasons_reach_the_editor_filter(self) -> None:
        events = [SimpleNamespace(text="其实我直都在", ocr_notes=(OCR_NOTE_MISSING_CHAR,), frame_count=30, confidence=0.99,
                                  start_sec=0.0, end_sec=1.0),
                  SimpleNamespace(text="成功解除余一部分", ocr_notes=(OCR_NOTE_EXTRA_CHAR,), frame_count=30, confidence=0.99,
                                  start_sec=5.0, end_sec=6.0)]
        assert ocr_suspicions(events) == {0: [SUSPECT_MISSING_CHAR], 1: [SUSPECT_EXTRA_CHAR]}


class TestAcceptReocr:
    def test_real_fixes(self) -> None:
        assert accept_reocr_text("其实我直都在", OCR_NOTE_MISSING_CHAR, ["其实我一直都在"] * 4) == "其实我一直都在"
        assert accept_reocr_text("任务", OCR_NOTE_MISSING_CHAR, ["任务一", "任务一", "任务", ""]) == "任务一"
        assert accept_reocr_text("成功解除余一部分", OCR_NOTE_EXTRA_CHAR, ["成功解除一部分"] * 3) == "成功解除一部分"

    @pytest.mark.parametrize(("current", "note", "reread"), [
        ("其实我直都在", OCR_NOTE_MISSING_CHAR, ["其实我直都在"] * 4),              # đọc lại vẫn như cũ
        ("其实我直都在", OCR_NOTE_MISSING_CHAR, ["其实我一直都在"]),                  # chỉ một khung
        ("其实我直都在", OCR_NOTE_MISSING_CHAR, ["其实我一一直都在"] * 3),            # thêm hai chữ
        ("其实我直都在", OCR_NOTE_MISSING_CHAR, ["其买我一直都在"] * 3),              # đổi cả chữ khác
        ("其实我直都在", OCR_NOTE_MISSING_CHAR, ["隐藏我的实力"] * 3),                # câu khác hẳn
        ("其实我直都在", OCR_NOTE_EXTRA_CHAR, ["其实我一直都在"] * 3),                # sai chiều của cờ
        ("其实我直都在", OCR_NOTE_MISSING_CHAR, ["其实我一直都在", "其实我一直都在", "其实我直都在a", "其实我直都在a"]),  # hoà phiếu
        ("其实我直都在", OCR_NOTE_MISSING_CHAR, ["其实我-直都在"] * 3),               # chữ thêm không phải chữ Hán
        ("其实我直都在", OCR_NOTE_MISSING_CHAR, []),
        ("", OCR_NOTE_MISSING_CHAR, ["一"] * 3),
    ])
    def test_everything_else_is_rejected(self, current: str, note: str, reread: list[str]) -> None:
        assert accept_reocr_text(current, note, reread) is None


class _Reporter:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def is_cancelled(self) -> bool:
        return False

    def report(self, current: int, total: int, message: str) -> None:
        self.messages.append(message)


def _use_case(reread_text: str, calls: list):
    from subtitles_extractor.application.use_cases.extract_subtitles import ExtractSubtitlesUseCase
    from subtitles_extractor.domain.entities.ocr_frame_result import OcrFrameResult

    use_case = object.__new__(ExtractSubtitlesUseCase)

    def iter_frames(*, metadata, roi, config):  # noqa: ANN001, ANN202
        calls.append(config)
        for index in range(4):
            yield SimpleNamespace(frame_index=index, timestamp_sec=config.skip_intro_sec + 0.1 * index, image_rgb=None,
                                  is_duplicate=False, is_blank=False, is_error=False)

    use_case._frame_sampler_port = SimpleNamespace(iter_frames=iter_frames)
    use_case._execute_ocr_inference_on_batch = lambda frames, debug, request, metadata: [
        OcrFrameResult(frame_index=f.frame_index, timestamp_sec=f.timestamp_sec,
                       text_boxes=[SimpleNamespace(text=reread_text)]) for f in frames
    ]
    return use_case


def _request(**preprocess: bool):
    from subtitles_extractor.application.dtos.extract_subtitles_dto import ExtractSubtitlesRequest
    from subtitles_extractor.domain.ports.ocr_engine_port import OcrEngineConfig, PreprocessConfig
    from pathlib import Path

    base = PreprocessConfig(upscale_small_text=False, add_white_border=False, apply_sharpen=False,
                            apply_contrast_boost=False, apply_clahe=False)
    return ExtractSubtitlesRequest(
        video_path=Path("v.mp4"), output_path=Path("v.srt"),
        ocr=OcrEngineConfig(preprocess=dataclasses.replace(base, **preprocess)),
    )


class TestVerifyLoop:
    def _events(self) -> list:
        return [SimpleNamespace(text="你好", start_sec=1.0, end_sec=2.0, ocr_notes=()),
                SimpleNamespace(text="其实我直都在", start_sec=94.2, end_sec=95.4, ocr_notes=(OCR_NOTE_MISSING_CHAR,))]

    def test_suspect_line_is_reread_raw_and_fixed(self) -> None:
        calls: list = []
        events = self._events()
        fixed = _use_case("其实我一直都在", calls)._verify_suspect_lines(
            events, _request(apply_sharpen=True, upscale_small_text=True), SimpleNamespace(duration_sec=126.0, roi=None),
            _Reporter(),
        )
        assert fixed == 1 and events[1].text == "其实我一直都在"
        assert events[1].ocr_notes == ("restored_char",)          # vẫn đưa ra soát ở trang Chỉnh sửa
        assert events[0].text == "你好" and len(calls) == 1        # chỉ đọc lại câu bị gắn cờ
        config = calls[0]
        assert 94.2 < config.skip_intro_sec < 95.4 and config.skip_outro_sec == pytest.approx(126.0 - (95.4 - 0.12))
        assert not (config.vram_upscale_small_text or config.vram_sharpen or config.vram_add_border)

    def test_unconfirmed_reread_keeps_text_and_flag(self) -> None:
        events = self._events()
        fixed = _use_case("其实我直都在", [])._verify_suspect_lines(
            events, _request(apply_sharpen=True), SimpleNamespace(duration_sec=126.0, roi=None), _Reporter(),
        )
        assert fixed == 0 and events[1].text == "其实我直都在" and events[1].ocr_notes == (OCR_NOTE_MISSING_CHAR,)

    def test_already_raw_config_has_no_second_opinion(self) -> None:
        calls: list = []
        events = self._events()
        assert _use_case("其实我一直都在", calls)._verify_suspect_lines(
            events, _request(), SimpleNamespace(duration_sec=126.0, roi=None), _Reporter(),
        ) == 0
        assert calls == [] and events[1].text == "其实我直都在"
