"""[v3.23.733] PaddleOCR-VL-1.6: bộ đọc dòng chất lượng cao — chạy chính (phiên bản "PaddleOCR-VL-1.6") và soát câu.

Đo trên 148 dòng phụ đề thật (3 phim, 15 ca khó kiểm bằng mắt), CPU: PaddleOCR-VL-1.6 đúng 146/148 (CER 0,26 %) so với
PP-OCRv6 small 143/148 (0,66 %); sửa được "霍" → "嚯", "腌攒" → "腌臜", "注人" → "注入". Chạy thật bộ chuyển đổi ở chế độ
chạy chính trên khung cắt từ video ROI: 6/6 khung khó ra đúng (gồm "诶" do bước cứu chữ giữ nguyên).
Các test dưới dùng bộ đọc VL giả lập — không cần tải mô hình ~2 GB.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from subtitles_extractor.domain.entities.ocr_frame_result import (
    OcrFrameResult,
    OcrTextBox,
)
from subtitles_extractor.domain.ports.ocr_engine_port import OcrEngineConfig
from subtitles_extractor.domain.value_objects.confidence import Confidence
from subtitles_extractor.domain.value_objects.device_kind import DeviceKind
from subtitles_extractor.domain.value_objects.ocr_model_names import (
    PADDLE_VL_DETECTION_MODEL,
    PADDLE_VL_DRAFT_RECOGNITION_MODEL,
    PADDLE_VL_RECOGNITION_MODEL,
    PADDLE_VL_VERSION,
    is_vl_recognizer,
)
from subtitles_extractor.infrastructure.ocr.paddle_vl_reader import (
    PaddleVlLineReader,
    clean_vl_text,
    split_page_text,
    stack_lines,
)

LINE = [(340, 20), (520, 20), (520, 80), (340, 80)]          # toạ độ ảnh đưa vào OCR (đã phóng ×2, viền 8)
OTHER = [(100, 20), (220, 20), (220, 80), (100, 80)]


# ── hàm thuần của bộ đọc ────────────────────────────────────────────────────────────────────────────────────────────
def test_stacked_page_and_split() -> None:
    crops = [np.full((30, 120, 3), 200, np.uint8), np.full((60, 90, 3), 100, np.uint8)]
    page = stack_lines(crops, height=48, gap=24)
    assert page.shape == (2 * (48 + 24) + 24, 192 + 48, 3)
    assert split_page_text("嚯\n\n腌臜\n", 2) == ["嚯", "腌臜"]
    assert split_page_text("嚯", 2) is None


@pytest.mark.parametrize(("raw", "clean"), [
    ("注入\\(\\underline{\\text{灵力}}\\)", "注入灵力"),         # đo thật: 5/148 dòng xếp trang bị bọc như vậy
    ("你和他的那些腌\\(\\underline{\\text{臜事}}\\)", "你和他的那些腌臜事"),
    ("**你好**", "你好"), ("我们去KTV", "我们去KTV"), ("三 二一", "三 二一"),
])
def test_vl_markup_is_removed(raw: str, clean: str) -> None:
    assert clean_vl_text(raw) == clean


class _FakeVlModel:
    def __init__(self, page_text: str, single: str = "单") -> None:
        self.page_text, self.single, self.calls = page_text, single, []

    def predict(self, payload, max_new_tokens=0):  # noqa: ANN001, ANN201
        rows = payload["image"].shape[0]
        self.calls.append(rows)
        return [{"res": {"result": self.page_text if rows > 100 else self.single}}]


def test_reader_reads_a_stacked_page_and_falls_back_per_line() -> None:
    crops = [np.full((40, 100, 3), 255, np.uint8) for _ in range(3)]
    reader = PaddleVlLineReader(lines_per_page=8)
    reader._ensure = lambda device: _FakeVlModel("一\n二\n三")  # type: ignore[method-assign]
    assert reader.read(crops) == ["一", "二", "三"]
    fake = _FakeVlModel("一\n二")                                    # thiếu dòng → đọc lại từng dòng
    reader._ensure = lambda device: fake  # type: ignore[method-assign]
    assert reader.read(crops) == ["单", "单", "单"] and len(fake.calls) == 4


# ── tên mô hình / bộ phân giải ──────────────────────────────────────────────────────────────────────────────────────
def test_version_resolves_to_ppocr_detector_plus_vl() -> None:
    from subtitles_extractor.application.services.ocr_model_resolver import (
        default_models_for_version,
        resolve_ocr_model_names,
    )

    assert default_models_for_version(PADDLE_VL_VERSION) == (PADDLE_VL_DETECTION_MODEL, PADDLE_VL_RECOGNITION_MODEL)
    settings = SimpleNamespace(version=PADDLE_VL_VERSION, detection_model="PP-OCRv6_small_det",
                               recognition_model="PP-OCRv6_small_rec", detection_model_override="")
    assert resolve_ocr_model_names(settings) == (PADDLE_VL_DETECTION_MODEL, PADDLE_VL_RECOGNITION_MODEL)
    assert is_vl_recognizer(PADDLE_VL_RECOGNITION_MODEL) and not is_vl_recognizer("PP-OCRv6_small_rec")


# ── bộ chuyển đổi PaddleOCR: chạy chính bằng VL ─────────────────────────────────────────────────────────────────────
class _FakeReader:
    def __init__(self, texts: list[str] | None = None, fail: bool = False) -> None:
        self.texts, self.fail, self.calls = texts or [], fail, []

    def read(self, crops, device="cpu"):  # noqa: ANN001, ANN201
        self.calls.append(len(crops))
        if self.fail:
            raise RuntimeError("thiếu sentencepiece")
        return self.texts[: len(crops)]

    def release(self) -> None:
        pass


def _adapter(reader: _FakeReader):  # noqa: ANN202
    from subtitles_extractor.infrastructure.ocr.paddle_ocr_adapter import (
        PaddleOcrAdapter,
        _PreparedImage,
    )

    adapter = PaddleOcrAdapter(OcrEngineConfig(device=DeviceKind.CPU, recognition_model_name=PADDLE_VL_RECOGNITION_MODEL))
    adapter._vl_reader = reader  # type: ignore[assignment]
    prepared = _PreparedImage(image_bgr=np.zeros((136, 1456, 3), np.uint8), scale=2.0, border_px=8)
    return adapter, prepared, np.full((60, 720, 3), 60, np.uint8)


def _frame(*boxes: OcrTextBox) -> OcrFrameResult:
    return OcrFrameResult(frame_index=0, timestamp_sec=3959.3, text_boxes=list(boxes))


def _raw(*polys) -> dict:  # noqa: ANN002
    return {"res": {"dt_polys": [np.array(p) for p in polys]}}


def test_vl_replaces_draft_reading_and_keeps_rescued_char() -> None:
    reader = _FakeReader(["嚯"])
    adapter, prepared, original = _adapter(reader)
    draft = OcrTextBox("霍", Confidence(0.70), LINE)
    rescued = OcrTextBox("诶", Confidence(0.85), OTHER)
    (out,) = adapter._vl_reread([_frame(draft, rescued)], [_raw(LINE, OTHER)], [prepared], [original])
    assert reader.calls == [1]                                       # chỉ hộp "霍"; hộp "诶" của bước cứu chữ giữ
    assert [(b.text, round(float(b.confidence), 2)) for b in out.text_boxes] == [("嚯", 0.9), ("诶", 0.85)]


def test_agreement_raises_confidence_and_unrecognized_scene_boxes_stay_empty() -> None:
    adapter, prepared, original = _adapter(_FakeReader(["别"]))
    (out,) = adapter._vl_reread([_frame(OcrTextBox("别", Confidence(0.52), LINE))], [_raw(LINE)], [prepared], [original])
    assert [(b.text, round(float(b.confidence), 2)) for b in out.text_boxes] == [("别", 0.97)]
    # Hộp bộ dò thấy mà PP-OCR không đọc ra (vật trong cảnh): VL không được biến nó thành chữ ("品但", "福没事吧").
    reader = _FakeReader(["福"])
    adapter, prepared, original = _adapter(reader)
    (out,) = adapter._vl_reread([_frame()], [_raw(LINE)], [prepared], [original])
    assert out.text_boxes == [] and reader.calls == []


def test_confident_repeated_reads_reuse_the_vl_cache() -> None:
    reader = _FakeReader(["以人炼丹的龌龊之辈"])
    adapter, prepared, original = _adapter(reader)
    frame = _frame(OcrTextBox("以人炼丹的龌龊之辈", Confidence(0.96), LINE))
    for _ in range(3):
        (out,) = adapter._vl_reread([frame], [_raw(LINE)], [prepared], [original])
        assert [b.text for b in out.text_boxes] == ["以人炼丹的龌龊之辈"]
    assert reader.calls == [1]                                       # khung đọc lại cùng câu: không gọi VL lần nữa


def test_vl_failure_keeps_ppocr_reading_for_the_session() -> None:
    reader = _FakeReader(fail=True)
    adapter, prepared, original = _adapter(reader)
    frame = _frame(OcrTextBox("霍", Confidence(0.70), LINE))
    assert adapter._vl_reread([frame], [_raw(LINE)], [prepared], [original]) == [frame]
    assert adapter._vl_unavailable is True
    assert adapter._vl_reread([frame], [_raw(LINE)], [prepared], [original]) == [frame] and reader.calls == [1]


def test_switching_between_vl_and_its_draft_model_does_not_reload_the_pipeline() -> None:
    from subtitles_extractor.infrastructure.ocr.paddle_ocr_adapter import PaddleOcrAdapter

    adapter = PaddleOcrAdapter(OcrEngineConfig(recognition_model_name=PADDLE_VL_DRAFT_RECOGNITION_MODEL))
    assert not adapter._needs_model_reload(OcrEngineConfig(recognition_model_name=PADDLE_VL_RECOGNITION_MODEL))
    assert adapter._needs_model_reload(OcrEngineConfig(recognition_model_name="PP-OCRv6_medium_rec"))
    kwargs = PaddleOcrAdapter(OcrEngineConfig(recognition_model_name=PADDLE_VL_RECOGNITION_MODEL))._build_init_kwargs(
        use_gpu=False, use_trt=False)
    assert kwargs["text_recognition_model_name"] == PADDLE_VL_DRAFT_RECOGNITION_MODEL


# ── lượt soát câu nghi ngờ bằng VL ─────────────────────────────────────────────────────────────────────────────────
def _use_case(reread: str, seen: list):  # noqa: ANN202
    from subtitles_extractor.application.use_cases.extract_subtitles import (
        ExtractSubtitlesUseCase,
    )

    use_case = object.__new__(ExtractSubtitlesUseCase)

    def iter_frames(*, metadata, roi, config):  # noqa: ANN001, ANN202
        for index in range(6):
            yield SimpleNamespace(frame_index=index, timestamp_sec=config.skip_intro_sec + 0.05 * index, image_rgb=None,
                                  is_duplicate=False, is_blank=False, is_error=False)

    def ocr(frames, debug, request, metadata):  # noqa: ANN001, ANN202
        seen.append((request.ocr.recognition_model_name, len(frames)))
        return [OcrFrameResult(f.frame_index, f.timestamp_sec, [SimpleNamespace(text=reread)]) for f in frames]

    use_case._frame_sampler_port = SimpleNamespace(iter_frames=iter_frames)
    use_case._execute_ocr_inference_on_batch = ocr
    return use_case


def _request(verify_with_vl: bool):  # noqa: ANN202
    from pathlib import Path

    from subtitles_extractor.application.dtos.extract_subtitles_dto import (
        ExtractSubtitlesRequest,
    )
    from subtitles_extractor.domain.ports.ocr_engine_port import PreprocessConfig

    raw = PreprocessConfig(upscale_small_text=False, add_white_border=False, apply_sharpen=False,
                           apply_contrast_boost=False, apply_clahe=False)
    return ExtractSubtitlesRequest(video_path=Path("v.mp4"), output_path=Path("v.srt"),
                                   ocr=OcrEngineConfig(preprocess=raw), verify_with_vl=verify_with_vl)


class _Reporter:
    def is_cancelled(self) -> bool:
        return False

    def report(self, current: int, total: int, message: str = "") -> None:
        pass


def _events() -> list:
    return [SimpleNamespace(text="你好", start_sec=1.0, end_sec=2.0, ocr_notes=(), confidence=0.99),
            SimpleNamespace(text="霍", start_sec=10.0, end_sec=10.5, ocr_notes=(), confidence=0.87)]


def test_low_confidence_line_is_reread_by_vl_and_fixed() -> None:
    seen: list = []
    events = _events()
    fixed = _use_case("嚯", seen)._verify_suspect_lines(
        events, _request(True), SimpleNamespace(duration_sec=60.0, roi=None), _Reporter())
    assert fixed == 1 and events[1].text == "嚯" and events[0].text == "你好"
    assert seen == [(PADDLE_VL_RECOGNITION_MODEL, 3)]              # VL, 3 khung — kể cả khi ảnh vốn không tiền xử lý


def test_without_vl_low_confidence_lines_are_not_reread() -> None:
    seen: list = []
    events = _events()
    assert _use_case("嚯", seen)._verify_suspect_lines(
        events, _request(False), SimpleNamespace(duration_sec=60.0, roi=None), _Reporter()) == 0
    assert seen == [] and events[1].text == "霍"


@pytest.mark.parametrize(("current", "reread", "expected"), [
    ("霍", ["嚯", "嚯", "霍"], "嚯"),
    ("你和他的那些腌攒事", ["你和他的那些腌臜事"] * 3, "你和他的那些腌臜事"),
    ("别", ["别"] * 3, None),                                      # VL xác nhận → giữ
    ("别", ["嚯", "你"], None),                                    # không đủ 2 phiếu cùng một bản
    ("你好", ["今天天气很好"] * 3, None),                           # khác hẳn câu cũ → không thay
    ("你好", ["OK", "OK"], None),                                  # không phải chữ Hán
])
def test_accept_vl_reread(current: str, reread: list[str], expected: str | None) -> None:
    from subtitles_extractor.application.services.subtitle_pipeline.geometry_check import (
        accept_vl_reread,
    )

    assert accept_vl_reread(current, reread) == expected
