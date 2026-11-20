"""[v3.23.748] Gói kiểm định hỏng trên máy chủ dự án: "all input arrays must have the same shape" (log 19/11 00:16).

Ảnh cắt sát từng câu (3.23.744) mỗi câu một cỡ; bộ dò chữ chạy 64 ảnh mỗi lần ghép lô bằng ``np.stack``. Máy kiểm chứng
chạy bộ dò từng ảnh một nên không lộ. Bộ chuyển đổi đệm đen cạnh phải / dưới cho bằng ảnh lớn nhất của lô.
"""

from __future__ import annotations

import numpy as np

from subtitles_extractor.infrastructure.ocr.paddle_ocr_adapter import (
    pad_to_common_shape,
)


def test_mixed_sizes_padded_bottom_right_with_black() -> None:
    small = np.full((40, 100, 3), 200, np.uint8)
    large = np.full((60, 180, 3), 50, np.uint8)
    out = pad_to_common_shape([small, large])
    assert {image.shape for image in out} == {(60, 180, 3)}
    assert (out[0][:40, :100] == 200).all() and (out[0][40:] == 0).all() and (out[0][:, 100:] == 0).all()
    assert out[1] is large                                           # ảnh lớn nhất giữ nguyên


def test_same_size_batch_untouched() -> None:
    frames = [np.zeros((118, 720, 3), np.uint8) for _ in range(3)]
    assert pad_to_common_shape(frames) is frames


def test_adapter_pads_before_detection() -> None:
    import inspect

    from subtitles_extractor.infrastructure.ocr.paddle_ocr_adapter import (
        PaddleOcrAdapter,
    )

    source = inspect.getsource(PaddleOcrAdapter.infer_batch)
    assert "bgr_list = pad_to_common_shape([p.image_bgr for p in prepared_images])" in source


def test_paddle_auditor_keeps_frame_size_and_masks_outside_subtitle(tmp_path) -> None:  # noqa: ANN001
    """Gói kiểm định PaddleOCR-VL: ảnh CỠ KHUNG, ngoài câu phụ đề che đen — mọi ảnh trong lô cùng cỡ, bộ dò đúng tỉ lệ
    bước OCR chính (đệm theo ảnh lớn nhất lô: đo 97 câu 50/57 → 46/57)."""
    import shutil

    import pytest

    if shutil.which("ffmpeg") is None:
        pytest.skip("cần ffmpeg")
    from subtitles_extractor.domain.entities.ocr_frame_result import (
        OcrFrameResult,
        OcrTextBox,
    )
    from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
    from subtitles_extractor.domain.ports.ocr_engine_port import OcrEngineConfig
    from subtitles_extractor.domain.value_objects.confidence import Confidence
    from subtitles_extractor.domain.value_objects.time_interval import TimeInterval
    from subtitles_extractor.infrastructure.quality.quality_auditor import (
        VlQualityAuditor,
    )
    from tests.unit.infrastructure.test_quality_audit_v323737 import _make_clip

    clip, sidecar = _make_clip(tmp_path)                    # 2 s đen rồi 2 s trắng, ROI 320x60
    seen: list[np.ndarray] = []

    class _Engine:
        def infer_batch(self, images, frame_indices, timestamps, config=None):  # noqa: ANN001, ANN201
            seen.extend(images)
            return [OcrFrameResult(i, t, []) for i, t in zip(frame_indices, timestamps, strict=True)]

    def box(text: str, x0: int, x1: int) -> OcrTextBox:
        return OcrTextBox(text=text, confidence=Confidence(0.99), polygon=[(x0, 20), (x1, 20), (x1, 40), (x0, 40)])

    frames = [OcrFrameResult(k, k * 0.04, [box("白天", 100, 140) if k * 0.04 >= 2 else box("你好啊", 60, 200)])
              for k in range(100)]
    events = [SubtitleEvent(index=1, text="你好啊", interval=TimeInterval(0.2, 1.6)),
              SubtitleEvent(index=2, text="白天", interval=TimeInterval(2.4, 3.8))]
    VlQualityAuditor(_Engine(), mask_in_frame=True).audit(  # type: ignore[arg-type]
        events, clip, sidecar, tmp_path / "qa.json.gz", config=OcrEngineConfig(), frames=frames)
    assert {image.shape for image in seen} == {(60, 320, 3)}            # cùng cỡ khung dù hai câu khác cỡ
    white = seen[-1]
    assert white[25, 120].min() > 200 and white[25, 5].max() == 0 and white[25, 300].max() == 0


def test_qa_reread_runs_detector_one_image_at_a_time() -> None:
    """[v3.23.749] Đo 97 câu (lô dò 64 như máy chủ dự án): cắt sát + đệm theo lô 46/57, che khung 47/57, cắt sát + dò
    từng ảnh 50/57 → lượt kiểm định PaddleOCR-VL dò từng ảnh; bộ chứa giữ ảnh cắt sát."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[3] / "src/subtitles_extractor"
    use_case = (root / "application/use_cases/extract_subtitles.py").read_text(encoding="utf-8")
    block = use_case[use_case.index("def _export_qa_data"):use_case.index("def _audit_with_glm")]
    assert "det_batch_size=1," in block
    container = (root / "composition/container.py").read_text(encoding="utf-8")
    assert "quality_auditor=VlQualityAuditor(self.ocr_engine)," in container
