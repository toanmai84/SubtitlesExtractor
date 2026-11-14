"""[v3.23.711] Bộ lọc khung báo "khung OCR này cùng điểm ảnh chữ với lần đọc trước" và dữ liệu thô giữ được cờ đó."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from subtitles_extractor.domain.entities.ocr_frame_result import OcrFrameResult, OcrTextBox
from subtitles_extractor.domain.ports.frame_sampler_port import FrameSamplingConfig
from subtitles_extractor.domain.value_objects.confidence import Confidence
from subtitles_extractor.infrastructure.serializers.raw_ocr_serializer import RawOcrMeta, load_raw_ocr, save_raw_ocr
from subtitles_extractor.infrastructure.video.frame_dedupe import FrameDeduper, FrameVerdict

H, W, STEP = 120, 600, 0.04


def _frame(text_w: int = 0) -> np.ndarray:
    g = np.full((H, W), 40, np.uint8)
    if text_w:
        g[45:75, 300 - text_w // 2:300 + text_w // 2] = 255
    return g


def _run(frames: list[np.ndarray]) -> list[tuple[str, bool]]:
    dd = FrameDeduper(FrameSamplingConfig(white_text_gate=True, white_text_luma=200))
    out = []
    for i, g in enumerate(frames):
        verdict = dd.verdict(g, i * STEP)
        out.append((verdict.value, dd.continues_text))
        if verdict is FrameVerdict.OCR:
            dd.accept(g, i * STEP)
    return out


def test_rereads_of_static_text_are_marked_as_same_pixels() -> None:
    result = _run([_frame(200)] * 14)
    assert result[0] == ("ocr", False)                               # chữ vừa hiện: lần đọc đầu
    assert all(flag for _, flag in result[1:])                       # khung trùng và mọi lần đọc lại
    assert [verdict for verdict, _ in result].count("ocr") >= 3      # có khung xác nhận + định kỳ


def test_changed_text_and_blank_are_not_marked() -> None:
    result = _run([_frame(200)] * 5 + [_frame(360)] * 3 + [_frame(0)] * 2 + [_frame(200)] * 2)
    assert result[5] == ("ocr", False)                               # chữ đổi
    assert result[8] == ("blank", False) and result[9] == ("blank", False)
    assert result[10] == ("ocr", False)                              # cùng chữ nhưng sau khoảng trống


def test_text_change_on_a_scheduled_reread_frame_is_a_change() -> None:
    """Chữ đổi đúng vào khung tới lịch đọc lại: phải là "chữ đổi", không được coi là cùng câu."""
    result = _run([_frame(200)] * 2 + [_frame(360)] * 2)             # khung 2 trùng lịch xác nhận (0,08 s)
    assert result[2] == ("ocr", False)


def test_flag_survives_the_raw_ocr_file(tmp_path: Path) -> None:
    box = OcrTextBox(text="你", confidence=Confidence(1.0), polygon=[(1, 1), (9, 1), (9, 9), (1, 9)])
    frames = [
        OcrFrameResult(0, 0.0, [box]), OcrFrameResult(1, 0.04, [box], continues_previous=True),
        OcrFrameResult(2, 0.08, [], continues_previous=True), OcrFrameResult(3, 0.12, []),
    ]
    meta = RawOcrMeta(video_name="a.mp4", video_duration_sec=1.0, frame_count=4, sample_step_sec=0.04,
                      detection_model="d", recognition_model="r", score_threshold=0.3, saved_at="")
    path = tmp_path / "a.seraw.json.gz"
    save_raw_ocr(frames, path, meta)
    loaded, _ = load_raw_ocr(path)
    assert [frame.continues_previous for frame in loaded] == [False, True, True, False]
