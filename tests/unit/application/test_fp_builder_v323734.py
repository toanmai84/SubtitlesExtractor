"""[v3.23.734] Phim "反派：师尊师姐求你们了" (83 tập nối, 3 giờ 18 phút, model small, bản 3.23.729) — dữ liệu ở
``test data/fp_v3_23_729`` (seraw + bản người dùng sửa + 9 đoạn video vùng ROI quanh các điểm đã kiểm bằng ảnh).

Đối chiếu bản người dùng sửa và từng khung (bản sửa KHÔNG phải đáp án: bỏ sót "哦", "以后", "但"… có thật, gộp nhầm
"我的腿" | "我的命", cắt ngắn thời gian). Lỗi thật của ứng dụng đã khắc phục:

* mảnh ĐỌC THIẾU chữ ≤ 0,3 s tách thành câu riêng: "懂什么" | "懂什", "小翠" | "翠" | "小翠", "魔了" | "魔" | "魔了";
* cùng điểm ảnh mà bản đọc thiếu đoạn đuôi: "您可以看一下" | "您可以看";
* cổ áo chữ V đọc thành "V" (6 câu); cuộn sách thư pháp nền sáng thành câu rác dài;
* "哼" đọc xen kẽ "亨" → tách đôi rồi bị bỏ cả hai mảnh; "嚯" đọc thành "霍".
"""

from __future__ import annotations

import gzip
import json
from dataclasses import fields
from pathlib import Path

import pytest

from subtitles_extractor.application.services.subtitle_pipeline.box_geometry import (
    drop_scene_text_block_events,
    drop_tall_non_han_events,
)
from subtitles_extractor.application.services.subtitle_pipeline.event_refinement import (
    absorb_truncated_fragments,
)
from subtitles_extractor.application.services.subtitle_pipeline.read_fusion import (
    _frame_text,
    fix_lone_interjections,
    fuse_same_pixel_reads,
)
from subtitles_extractor.domain.entities.ocr_frame_result import (
    OcrFrameResult,
    OcrTextBox,
)
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.value_objects.confidence import Confidence
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval

DATA = Path(__file__).resolve().parents[3] / "test data" / "fp_v3_23_729"
STEP = 0.04


def _ev(text: str, start: float, end: float, frames: int = 10, conf: float = 1.0) -> SubtitleEvent:
    return SubtitleEvent(index=1, text=text, interval=TimeInterval(start, end), confidence=Confidence(conf),
                         frame_count=frames)


def _box(text: str, x0: float, x1: float, y0: float = 20, y1: float = 88, conf: float = 1.0) -> OcrTextBox:
    return OcrTextBox(text=text, confidence=Confidence(conf), polygon=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


# ── mảnh đọc thiếu chữ ──────────────────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(("events", "expected"), [
    ([_ev("懂什么", 5920.362, 5920.960), _ev("懂什", 5920.962, 5921.162)], [("懂什么", 5920.362, 5921.162)]),
    ([_ev("小翠", 6090.968, 6091.207), _ev("翠", 6091.208, 6091.367), _ev("小翠", 6091.368, 6091.567)],
     [("小翠", 6090.968, 6091.567)]),
    ([_ev("魔了", 11594.862, 11595.061), _ev("魔", 11595.062, 11595.261), _ev("魔了", 11595.262, 11595.581),
      _ev("魔", 11595.582, 11595.712)], [("魔了", 11594.862, 11595.712)]),
])
def test_truncated_fragments_are_absorbed(events: list, expected: list) -> None:
    out = absorb_truncated_fragments(events, STEP)
    assert [(e.text, round(e.start_sec, 3), round(e.end_sec, 3)) for e in out] == expected


@pytest.mark.parametrize("events", [
    [_ev("啊", 11056.069, 11056.309), _ev("这这全部啊", 11056.349, 11057.869)],   # hiện dần: phần ngắn đứng TRƯỚC
    [_ev("这", 4731.868, 4732.108), _ev("这不可能", 4732.148, 4732.828)],
    [_ev("冷静", 2318.945, 2319.745), _ev("冷静", 2319.945, 2320.745)],             # khoảng trống thật giữa hai lần
    [_ev("这买卖做得", 2357.05, 2358.72), _ev("做得", 2358.73, 2359.69)],           # câu sau dài ~1 s là câu thật
])
def test_real_consecutive_lines_are_kept(events: list) -> None:
    assert [e.text for e in absorb_truncated_fragments(events, STEP)] == [e.text for e in events]


def test_same_pixels_with_truncated_tail_take_the_full_reading() -> None:
    frames = [OcrFrameResult(i, 6125.2 + i * STEP, [_box("您可以看一下", 200, 520)], continues_previous=i > 0)
              for i in range(13)]
    frames += [OcrFrameResult(13 + i, 6125.73 + i * STEP, [_box("您可以看", 200, 420)], continues_previous=True)
               for i in range(18)]
    out, fixed, _ = fuse_same_pixel_reads(frames, center_x=359.0, char_pitch=55.0, text_height=65.0)
    assert fixed > 0 and {_frame_text(f) for f in out} == {"您可以看一下"}


# ── hình khối / khối chữ trong cảnh ────────────────────────────────────────────────────────────────────────────────
def _frames(boxes_per_frame: list[list[OcrTextBox]], start: float) -> list[OcrFrameResult]:
    return [OcrFrameResult(i, start + i * STEP, boxes) for i, boxes in enumerate(boxes_per_frame)]


def test_tall_latin_shape_dropped_in_chinese_film() -> None:
    height = 65.0
    film = [_ev(f"台词{i}", float(i), i + 0.8) for i in range(20)]
    v = _ev("V", 1830.159, 1830.519)
    ok = _ev("OK", 1900.0, 1900.8)
    frames = _frames([[_box("V", 278, 504, 0, 120, 0.98)]] * 9, 1830.159)          # cổ áo chữ V: cao 1,85 lần
    frames += _frames([[_box("OK", 320, 400, 22, 86)]] * 9, 1900.0)                # chữ Latin thật: cao bằng cỡ chữ
    kept, dropped = drop_tall_non_han_events([*film, v, ok], frames, height)
    assert dropped == 1 and "V" not in [e.text for e in kept] and "OK" in [e.text for e in kept]


def test_scene_text_block_dropped_but_one_inflated_subtitle_box_kept() -> None:
    height = 65.0
    scroll = [_box("得失言辭犀", -4, 226, -4, 123), _box("集尤精兵法", 112, 379, -4, 123),
              _box("重瑾瑶自幼", 261, 528, -4, 123), _box("修專司修", 409, 633, -4, 123)]
    frames = _frames([[_box("但", 326, 399, 13, 109)]] * 10, 980.0) + _frames([scroll] * 8, 6148.808)  # theo thời gian
    events = [_ev("但", 980.0, 980.4), _ev("得失言辞犀子集尤精兵法重瑾瑶自幼修专司修", 6148.808, 6149.128)]
    kept, dropped = drop_scene_text_block_events(events, frames, height)
    assert dropped == 1 and [e.text for e in kept] == ["但"]


# ── chữ cảm thán mất bộ 口 ──────────────────────────────────────────────────────────────────────────────────────────
def test_lone_interjections_without_mouth_radical_are_fixed() -> None:
    frames = [OcrFrameResult(0, 1.0, [_box("亨", 327, 392, conf=0.73)]),
              OcrFrameResult(1, 1.04, [_box("霍", 327, 392, conf=0.87)]),
              OcrFrameResult(2, 1.08, [_box("亨利来了", 250, 470)])]               # trong câu: giữ nguyên
    out, changed = fix_lone_interjections(frames)
    assert changed == 2 and [_frame_text(f) for f in out] == ["哼", "嚯", "亨利来了"]


# ── phát lại cả lượt thật ────────────────────────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def replayed() -> list[SubtitleEvent]:
    from subtitles_extractor.application.dtos.extract_subtitles_dto import (
        SubtitleBuilderConfig,
    )
    from subtitles_extractor.application.services.subtitle_pipeline.orchestrator import (
        SubtitleBuilder,
    )
    from subtitles_extractor.domain.value_objects.roi import Roi
    from subtitles_extractor.infrastructure.serializers.raw_ocr_serializer import (
        load_raw_ocr,
    )

    raw = DATA / "fp.seraw.json.gz"
    run = json.load(gzip.open(raw))["meta"]["run"]
    names = {f.name for f in fields(SubtitleBuilderConfig)}
    config = SubtitleBuilderConfig(**{k: v for k, v in run["builder"].items() if k in names})
    frames, meta = load_raw_ocr(raw)
    return SubtitleBuilder(config).build(frames, roi=Roi(*meta.roi_xywh))


def _at(events: list[SubtitleEvent], t: float) -> list[str]:
    return [e.text for e in events if e.start_sec - 0.01 <= t <= e.end_sec + 0.01]


def test_real_run(replayed: list[SubtitleEvent]) -> None:
    texts = [e.text for e in replayed]
    assert "V" not in texts and not any("言辞" in t for t in texts)
    assert _at(replayed, 5921.1) == ["懂什么"] and _at(replayed, 6091.3) == ["小翠"]
    assert _at(replayed, 6126.3) == ["您可以看一下"] and _at(replayed, 11595.5) == ["魔了"]
    assert _at(replayed, 7625.9) == ["哼"]                                      # trước: mất hẳn
    assert _at(replayed, 4979.0) == ["哼"]
    assert _at(replayed, 11056.2) == ["啊"] and _at(replayed, 11056.5) == ["这这全部啊"]   # hiện dần: vẫn tách
    assert _at(replayed, 2166.4) == ["我的腿"] and _at(replayed, 2166.9) == ["我的命"]       # hai câu thật
    assert len(replayed) == 6598  # [v3.23.736] chữ đơn A|B|A: "你|尔|你" (150:51) và "哼|停|哼" (113:22) thành một câu


def test_roi_clips_present() -> None:
    assert len(list((DATA / "clips").glob("*.roi.mp4"))) == 9


# ── lượt soát câu: thiếu/thừa chữ → PP-OCR ảnh gốc; hai bản đọc → PaddleOCR-VL ─────────────────────────────────────
def test_verify_uses_ppocr_for_missing_chars_and_vl_for_disputes() -> None:
    """Lượt thật phim "反派": VL đọc trong hộp bộ dò nên cũng thiếu chữ đầu ("个德行", "知死舌"), trong khi PP-OCR
    ảnh gốc sửa đúng 7/7 câu thiếu chữ; với câu hai bản đọc, cách cũ đổi SAI 3/5 ("你丢"→"尔丢", "怎么"→"怎公",
    "嚣张"→"器张") còn VL đọc đúng cả 3."""
    from types import SimpleNamespace

    from subtitles_extractor.application.dtos.extract_subtitles_dto import (
        ExtractSubtitlesRequest,
    )
    from subtitles_extractor.application.services.subtitle_pipeline.geometry_check import (
        OCR_NOTE_DISPUTED,
        OCR_NOTE_MISSING_CHAR,
    )
    from subtitles_extractor.application.use_cases.extract_subtitles import (
        ExtractSubtitlesUseCase,
    )
    from subtitles_extractor.domain.ports.ocr_engine_port import (
        OcrEngineConfig,
        PreprocessConfig,
    )
    from subtitles_extractor.domain.value_objects.ocr_model_names import (
        PADDLE_VL_RECOGNITION_MODEL,
    )

    seen: list = []
    answers = {False: "一个德行", True: "怎么"}
    use_case = object.__new__(ExtractSubtitlesUseCase)

    def iter_frames(*, metadata, roi, config):  # noqa: ANN001, ANN202
        for index in range(14):
            yield SimpleNamespace(frame_index=index, timestamp_sec=config.skip_intro_sec + 0.05 * index, image_rgb=None,
                                  is_duplicate=False, is_blank=False, is_error=False)

    def ocr(frames, debug, request, metadata):  # noqa: ANN001, ANN202
        vl = request.ocr.recognition_model_name == PADDLE_VL_RECOGNITION_MODEL
        seen.append((vl, len(frames)))
        return [OcrFrameResult(f.frame_index, f.timestamp_sec, [SimpleNamespace(text=answers[vl])]) for f in frames]

    use_case._frame_sampler_port = SimpleNamespace(iter_frames=iter_frames)
    use_case._execute_ocr_inference_on_batch = ocr
    sharpen = PreprocessConfig(upscale_small_text=True, apply_sharpen=True)
    request = ExtractSubtitlesRequest(video_path=Path("v.mp4"), output_path=Path("v.srt"),
                                      ocr=OcrEngineConfig(preprocess=sharpen), verify_with_vl=True)
    events = [SimpleNamespace(text="个德行", start_sec=1954.1, end_sec=1955.0, ocr_notes=(OCR_NOTE_MISSING_CHAR,),
                              confidence=0.99),
              SimpleNamespace(text="怎么", start_sec=706.5, end_sec=707.2, ocr_notes=(OCR_NOTE_DISPUTED,), confidence=0.97)]

    class _Reporter:
        def is_cancelled(self) -> bool:
            return False

        def report(self, current: int, total: int, message: str = "") -> None:
            pass

    use_case._verify_suspect_lines(events, request, SimpleNamespace(duration_sec=12000.0, roi=None), _Reporter())
    assert events[0].text == "一个德行" and events[1].text == "怎么"
    assert sorted(seen) == [(False, 7), (True, 3)]       # thiếu chữ: PP-OCR 5+2 khung; hai bản đọc: VL 3 khung


@pytest.mark.parametrize(("text", "first_start", "second_start"), [
    ("冷静", 2318.945, 2319.945), ("师姐", 2134.938, 2136.018), ("闭嘴", 11467.936, 11468.856),
])
def test_repeated_line_after_real_blank_stays_two_lines(
    replayed: list[SubtitleEvent], text: str, first_start: float, second_start: float,
) -> None:
    """[v3.23.735] Quyết định chủ dự án (17/11/2026): cùng câu lặp lại sau khoảng trống ngắn CÓ THẬT trên phim → giữ hai
    câu như phim (không gộp)."""
    starts = [round(e.start_sec, 3) for e in replayed if e.text == text]
    assert first_start in starts and second_start in starts
