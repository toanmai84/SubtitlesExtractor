"""[v3.23.758] Chọn "Bộ đọc kiểm định: GLM-OCR" thì bước SOÁT CÂU cũng đọc lại bằng GLM-OCR (chủ dự án 19/11: "Sao tôi
chọn GLM-OCR soát câu mà còn thấy PaddleOCR-VL chạy" — "phải chạy theo tuỳ chọn trong cài đặt của người dùng").

Trước 3.23.758 tuỳ chọn chỉ áp cho gói kiểm định; bước soát câu luôn nạp PaddleOCR-VL. Nay: GLM-OCR thay VL trong bước
soát (đo 457 câu: 57/57 câu khó so với 50/57), một máy chủ GLM cho cả bước, lỗi → lùi về PaddleOCR-VL và ghi nhật ký.
"""

from __future__ import annotations

import contextlib
import dataclasses
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
    GLM_OCR_QA_MODEL,
    PADDLE_VL_RECOGNITION_MODEL,
    QA_READER_GLM_OCR,
    QA_READER_PADDLE_VL,
)

LINE = [(340, 20), (520, 20), (520, 80), (340, 80)]


class _Reader:
    def __init__(self, texts: list[str], fail: bool = False, name: str = "fake") -> None:
        self.texts, self.fail, self.calls, self.model_name = texts, fail, [], name

    def read(self, crops, device="cpu"):  # noqa: ANN001, ANN201
        self.calls.append([crop.shape for crop in crops])
        if self.fail:
            raise RuntimeError("llama-server sập")
        return self.texts[: len(crops)]

    def release(self) -> None:
        pass


def _adapter(vl: _Reader):  # noqa: ANN202
    from subtitles_extractor.infrastructure.ocr.paddle_ocr_adapter import (
        PaddleOcrAdapter,
        _PreparedImage,
    )

    adapter = PaddleOcrAdapter(OcrEngineConfig(device=DeviceKind.CPU, recognition_model_name=PADDLE_VL_RECOGNITION_MODEL))
    adapter._vl_reader = vl  # type: ignore[assignment]
    prepared = _PreparedImage(image_bgr=np.zeros((136, 1456, 3), np.uint8), scale=2.0, border_px=8)
    return adapter, prepared, np.full((60, 720, 3), 60, np.uint8)


def _reread(adapter, prepared, original, text: str = "抢刀"):  # noqa: ANN001, ANN202
    frame = OcrFrameResult(0, 1.0, [OcrTextBox(text, Confidence(0.96), LINE)])
    (out,) = adapter._vl_reread([frame], [{"res": {"dt_polys": [np.array(LINE)]}}], [prepared], [original])
    return [box.text for box in out.text_boxes]


# ── bộ chuyển đổi OCR: bộ đọc dòng thay thế ─────────────────────────────────────────────────────────────────────────
def test_override_reader_replaces_vl_and_cache_does_not_mix_models() -> None:
    vl = _Reader(["抢刀"])
    adapter, prepared, original = _adapter(vl)
    assert _reread(adapter, prepared, original) == ["抢刀"] and len(vl.calls) == 1     # VL đọc, vào bộ đệm
    glm = _Reader(["抡刀"])
    adapter.set_line_reader(glm)
    assert _reread(adapter, prepared, original) == ["抡刀"]                           # bộ đệm VL đã xoá: GLM đọc
    assert len(glm.calls) == 1 and len(vl.calls) == 1
    adapter.set_line_reader(None)
    assert _reread(adapter, prepared, original) == ["抢刀"] and len(vl.calls) == 2     # chữ GLM không lọt sang VL


def test_override_failure_keeps_drafts_without_loading_vl_then_vl_takes_over() -> None:
    vl = _Reader(["别冲动"])
    adapter, prepared, original = _adapter(vl)
    adapter.set_line_reader(_Reader([], fail=True, name=GLM_OCR_QA_MODEL))
    # [v3.23.759] Lô lỗi giữ bản nháp PP-OCR và KHÔNG nạp VL (llama-server còn giữ GLM trong VRAM; VL nạp lỗi vì tranh
    # VRAM sẽ tắt VL cả phiên — lượt rà độc lập 3.23.758).
    assert _reread(adapter, prepared, original, "剔冲动") == ["剔冲动"] and vl.calls == []
    assert adapter.line_reader_error is not None and "llama-server" in adapter.line_reader_error
    assert adapter._line_reader_override is None and adapter._vl_unavailable is False and adapter.vl_error is None
    assert _reread(adapter, prepared, original, "剔冲动") == ["别冲动"] and len(vl.calls) == 1   # lô sau: VL


def test_override_returning_wrong_count_is_a_failure() -> None:
    adapter, prepared, original = _adapter(_Reader(["x"]))
    adapter.set_line_reader(_Reader([]))                                              # 0 bản đọc cho 1 ảnh
    assert _reread(adapter, prepared, original, "剔冲动") == ["剔冲动"] and adapter.line_reader_error


def test_override_applies_only_to_the_thread_that_set_it() -> None:
    import threading

    vl = _Reader(["抢刀"])
    adapter, prepared, original = _adapter(vl)
    glm = _Reader(["抡刀"])
    adapter.set_line_reader(glm)
    other: list = []
    worker = threading.Thread(target=lambda: other.append(_reread(adapter, prepared, original)))  # vd Re-OCR song song
    worker.start()
    worker.join()
    assert other == [["抢刀"]] and len(vl.calls) == 1 and glm.calls == []
    assert _reread(adapter, prepared, original) == ["抡刀"] and len(glm.calls) == 1


def test_each_verify_batch_rereads_with_the_override() -> None:
    adapter, prepared, original = _adapter(_Reader(["抢刀"]))
    glm = _Reader(["抡刀"])
    adapter.set_line_reader(glm)
    for _ in range(3):                                            # 3 khung của cùng câu: 3 phiếu độc lập, không dùng đệm
        assert _reread(adapter, prepared, original) == ["抡刀"]
    assert len(glm.calls) == 3


def test_override_works_even_when_vl_is_unavailable() -> None:
    adapter, prepared, original = _adapter(_Reader([], fail=True))
    adapter._vl_unavailable = True
    adapter.set_line_reader(_Reader(["抡刀"]))
    assert _reread(adapter, prepared, original) == ["抡刀"]


def test_glm_line_reader_converts_bgr_to_rgb_uses_short_timeout_and_honours_cancel() -> None:
    from subtitles_extractor.infrastructure.quality.glm_ocr_runtime import (
        GLM_OCR_LABEL,
        LINE_MAX_CONSECUTIVE_READ_ERRORS,
        LINE_REQUEST_TIMEOUT_GPU_SEC,
        GlmLineReader,
    )

    seen: list = []

    def read_tolerant(image, timeout, max_failures):  # noqa: ANN001, ANN202
        seen.append((image, timeout, max_failures))
        return "抡刀"

    engine = SimpleNamespace(_read_tolerant=read_tolerant, read_errors=0)
    cancel = {"now": False}
    reader = GlmLineReader(engine, use_gpu=True, cancelled=lambda: cancel["now"])  # type: ignore[arg-type]
    crop = np.zeros((20, 30, 3), np.uint8)
    crop[..., 0] = 255                                                                # BGR: xanh lam
    assert reader.read([crop], device="gpu") == ["抡刀"] and reader.model_name == GLM_OCR_LABEL
    image, timeout, max_failures = seen[0]
    assert image[0, 0].tolist() == [0, 0, 255]                                         # RGB: xanh lam ở kênh cuối
    assert timeout == LINE_REQUEST_TIMEOUT_GPU_SEC <= 30 and max_failures == LINE_MAX_CONSECUTIVE_READ_ERRORS <= 3
    cancel["now"] = True
    assert reader.read([crop, crop]) == ["", ""] and len(seen) == 1                    # huỷ: không gọi server nữa


def test_encode_error_counts_as_one_bad_image() -> None:
    from subtitles_extractor.infrastructure.quality.glm_ocr_runtime import GlmOcrEngine

    engine = GlmOcrEngine(SimpleNamespace(base_url="http://127.0.0.1:1"))  # type: ignore[arg-type]
    engine.read = lambda image, timeout=0: (_ for _ in ()).throw(Exception("cv2.error: !_img.empty()"))  # type: ignore[method-assign]
    assert engine._read_tolerant(np.zeros((1, 1, 3), np.uint8)) == "" and engine.read_errors == 1


# ── use case: chọn bộ đọc theo Cài đặt ──────────────────────────────────────────────────────────────────────────────
class _Reporter:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def is_cancelled(self) -> bool:
        return False

    def report(self, current: int, total: int, message: str = "") -> None:
        self.messages.append(message)


def _request(qa_reader: str):  # noqa: ANN202
    from pathlib import Path

    from subtitles_extractor.application.dtos.extract_subtitles_dto import (
        ExtractSubtitlesRequest,
    )
    from subtitles_extractor.domain.ports.ocr_engine_port import PreprocessConfig

    raw = PreprocessConfig(upscale_small_text=False, add_white_border=False, apply_sharpen=False,
                           apply_contrast_boost=False, apply_clahe=False)
    request = ExtractSubtitlesRequest(video_path=Path("v.mp4"), output_path=Path("v.srt"),
                                      ocr=OcrEngineConfig(preprocess=raw, device=DeviceKind.GPU), verify_with_vl=True)
    return dataclasses.replace(request, qa_reader=qa_reader)


def _use_case(engine: object, factory, reread: str = "嚯"):  # noqa: ANN001, ANN202
    from subtitles_extractor.application.use_cases.extract_subtitles import (
        ExtractSubtitlesUseCase,
    )

    use_case = object.__new__(ExtractSubtitlesUseCase)
    use_case._ocr_engine_port = engine
    use_case._glm_line_reader_factory = factory
    seen: list = []

    def iter_frames(*, metadata, roi, config):  # noqa: ANN001, ANN202
        for index in range(6):
            yield SimpleNamespace(frame_index=index, timestamp_sec=config.skip_intro_sec + 0.05 * index, image_rgb=None,
                                  is_duplicate=False, is_blank=False, is_error=False)

    def ocr(frames, debug, request, metadata):  # noqa: ANN001, ANN202
        seen.append((request.ocr.recognition_model_name, getattr(engine, "override", None)))
        return [OcrFrameResult(f.frame_index, f.timestamp_sec, [SimpleNamespace(text=reread)]) for f in frames]

    use_case._frame_sampler_port = SimpleNamespace(iter_frames=iter_frames)
    use_case._execute_ocr_inference_on_batch = ocr
    return use_case, seen


class _Engine:
    def __init__(self) -> None:
        self.override = None
        self.history: list[object] = []
        self.line_reader_error = None
        self.vl_error = None

    def set_line_reader(self, reader) -> None:  # noqa: ANN001
        self.override = reader
        self.history.append(reader)

    def release_auxiliary_models(self) -> None:
        self.history.append("release")


def _events() -> list:
    return [SimpleNamespace(text="霍", start_sec=10.0, end_sec=10.5, ocr_notes=(), confidence=0.87)]


def _factory(calls: list, reader: object | None = None, fail: bool = False):  # noqa: ANN202
    @contextlib.contextmanager
    def factory(*, use_gpu, progress, cancelled):  # noqa: ANN001, ANN202
        calls.append(("open", use_gpu))
        if fail:
            raise OSError("không tải được")
        try:
            yield reader
        finally:
            calls.append("close")

    return factory


def test_glm_setting_verifies_with_glm_and_detaches_before_server_stops() -> None:
    engine, calls = _Engine(), []
    glm = _Reader(["嚯"], name=GLM_OCR_QA_MODEL)
    use_case, seen = _use_case(engine, _factory(calls, glm))
    log: list = []
    events = _events()
    fixed = use_case._verify_suspect_lines(events, _request(QA_READER_GLM_OCR),
                                           SimpleNamespace(duration_sec=60.0, roi=None), _Reporter(), log)
    assert fixed == 1 and events[0].text == "嚯"
    assert seen == [(PADDLE_VL_RECOGNITION_MODEL, glm)]                 # đường đọc dòng, bộ đọc là GLM
    assert calls == [("open", True), "close"]                           # GPU khi lượt chạy GPU; server tắt sau cùng
    assert engine.history == ["release", glm, None] and engine.override is None
    assert log[0]["reader"] == GLM_OCR_QA_MODEL


def test_paddle_vl_setting_never_starts_glm() -> None:
    engine, calls = _Engine(), []
    use_case, seen = _use_case(engine, _factory(calls, _Reader(["嚯"])))
    log: list = []
    use_case._verify_suspect_lines(_events(), _request(QA_READER_PADDLE_VL),
                                   SimpleNamespace(duration_sec=60.0, roi=None), _Reporter(), log)
    assert calls == [] and engine.history == [] and seen == [(PADDLE_VL_RECOGNITION_MODEL, None)]
    assert log[0]["reader"] == PADDLE_VL_RECOGNITION_MODEL


@pytest.mark.parametrize("engine_has_hook", [True, False])
def test_glm_start_failure_falls_back_to_paddle_vl(engine_has_hook: bool) -> None:
    engine = _Engine() if engine_has_hook else SimpleNamespace(vl_error=None)
    calls: list = []
    use_case, seen = _use_case(engine, _factory(calls, fail=True))
    log: list = []
    events = _events()
    fixed = use_case._verify_suspect_lines(events, _request(QA_READER_GLM_OCR),
                                           SimpleNamespace(duration_sec=60.0, roi=None), _Reporter(), log)
    assert fixed == 1 and seen == [(PADDLE_VL_RECOGNITION_MODEL, None)]
    assert calls == ([("open", True)] if engine_has_hook else [])
    assert [entry.get("reader") for entry in log if "line" in entry] == [PADDLE_VL_RECOGNITION_MODEL]
    if engine_has_hook:
        assert any("GLM-OCR không bật được" in entry.get("error", "") for entry in log)


def test_midway_glm_failure_is_logged() -> None:
    engine, calls = _Engine(), []
    engine.line_reader_error = "GlmOcrError: 5 ảnh liên tiếp"
    use_case, _seen = _use_case(engine, _factory(calls, _Reader(["嚯"], name=GLM_OCR_QA_MODEL)))
    log: list = []
    use_case._verify_suspect_lines(_events(), _request(QA_READER_GLM_OCR),
                                   SimpleNamespace(duration_sec=60.0, roi=None), _Reporter(), log)
    assert any("lỗi giữa chừng" in entry.get("error", "") for entry in log)


def test_container_wires_glm_line_reader_factory() -> None:
    import inspect

    from subtitles_extractor.composition import container

    source = inspect.getsource(container)
    assert "glm_line_reader_factory=self._glm_line_reader_factory" in source and "glm_ocr_line_reader(" in source


# ── [v3.23.759] lượt rà độc lập 3.23.758 ───────────────────────────────────────────────────────────────────────────
def test_glm_not_started_when_only_geometry_lines_need_rereading() -> None:
    from subtitles_extractor.application.services.subtitle_pipeline.geometry_check import (
        OCR_NOTE_MISSING_CHAR,
    )
    from subtitles_extractor.domain.ports.ocr_engine_port import PreprocessConfig

    engine, calls = _Engine(), []
    use_case, _seen = _use_case(engine, _factory(calls, _Reader(["x"])))
    events = [SimpleNamespace(text="个德行", start_sec=10.0, end_sec=10.5, ocr_notes=(OCR_NOTE_MISSING_CHAR,),
                              confidence=0.99)]
    request = _request(QA_READER_GLM_OCR)
    request = dataclasses.replace(request, ocr=dataclasses.replace(request.ocr, preprocess=PreprocessConfig()))
    use_case._verify_suspect_lines(events, request, SimpleNamespace(duration_sec=60.0, roi=None), _Reporter(), [])
    assert calls == [] and engine.history == []                       # câu thiếu chữ đọc bằng PP-OCR: không bật GLM


def test_midway_failure_closes_server_before_next_lines_and_logs_true_reader() -> None:
    engine, calls = _Engine(), []
    glm = _Reader(["嚯"], name=GLM_OCR_QA_MODEL)
    use_case, seen = _use_case(engine, _factory(calls, glm))
    inner = use_case._execute_ocr_inference_on_batch

    def failing_once(frames, debug, request, metadata):  # noqa: ANN001, ANN202
        result = inner(frames, debug, request, metadata)
        if engine.override is not None and engine.line_reader_error is None:
            engine.line_reader_error = "GlmOcrError: 3 ảnh liên tiếp"   # bộ chuyển đổi đã bỏ GLM trong lô này
            engine.override = None
        return result

    use_case._execute_ocr_inference_on_batch = failing_once
    events = [SimpleNamespace(text="霍", start_sec=10.0, end_sec=10.5, ocr_notes=(), confidence=0.87),
              SimpleNamespace(text="霍", start_sec=20.0, end_sec=20.5, ocr_notes=(), confidence=0.87)]
    log: list = []
    use_case._verify_suspect_lines(events, _request(QA_READER_GLM_OCR), SimpleNamespace(duration_sec=60.0, roi=None),
                                   _Reporter(), log)
    lines = [entry for entry in log if "line" in entry]
    assert lines[0]["reader"].startswith("PP-OCR") and lines[1]["reader"] == PADDLE_VL_RECOGNITION_MODEL
    assert calls == [("open", True), "close"] and engine.history[-1] is None      # server tắt ngay sau câu lỗi
    assert any("lỗi giữa chừng" in entry.get("error", "") for entry in log)


def test_sticky_vl_error_not_reported_when_glm_read_everything() -> None:
    engine, calls = _Engine(), []
    engine.vl_error = "RuntimeError: VRAM (lượt trước)"
    use_case, _seen = _use_case(engine, _factory(calls, _Reader(["嚯"], name=GLM_OCR_QA_MODEL)))
    log: list = []
    use_case._verify_suspect_lines(_events(), _request(QA_READER_GLM_OCR), SimpleNamespace(duration_sec=60.0, roi=None),
                                   _Reporter(), log)
    assert not any("PaddleOCR-VL lỗi" in entry.get("error", "") for entry in log)
