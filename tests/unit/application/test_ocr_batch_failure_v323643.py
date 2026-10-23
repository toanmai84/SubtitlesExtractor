"""[v3.23.643] Một khung OCR lỗi không được làm mất cả lô; khung trùng không được rơi lặng lẽ.

Đọc mã: ``_execute_ocr_inference_on_batch`` trả KHUNG RỖNG cho cả lô khi có một ``OcrInferenceError``
(tới 128 khung ≈ 5 giây phụ đề mất, chỉ có một dòng log), và nhánh khung trùng bỏ khung khi chưa có
kết quả OCR trước đó (đầu lượt / ngay sau khung lỗi) làm chuỗi khung bị thủng.
"""

from __future__ import annotations

from pathlib import Path

from subtitles_extractor.domain.entities.ocr_frame_result import OcrFrameResult, OcrTextBox
from subtitles_extractor.domain.exceptions import OcrInferenceError
from subtitles_extractor.domain.value_objects.confidence import Confidence
from tests.unit.application.test_ocr_checkpoint_v323632 import (
    DURATION,
    STEP,
    _Engine,
    _frames,
    _run,
    _Sampler,
    _text_at,
)


class _OneBadFrameEngine(_Engine):
    """Cả lô có chứa khung ``bad_ts`` thì ném lỗi; từng khung riêng lẻ thì chỉ khung xấu lỗi."""

    def __init__(self, bad_ts: float) -> None:
        super().__init__()
        self.bad_ts = bad_ts
        self.calls: list[int] = []

    def infer_batch(self, images_rgb, frame_indices, timestamps_sec):
        self.calls.append(len(frame_indices))
        if any(abs(t - self.bad_ts) < 1e-6 for t in timestamps_sec):
            raise OcrInferenceError("khung hỏng")
        return super().infer_batch(images_rgb, frame_indices, timestamps_sec)


def test_one_bad_frame_loses_only_itself(tmp_path: Path) -> None:
    # Khung thứ 3 trong mỗi 3 khung là khung trùng và cắt lô → lô chứa 0,3 là {0,3 ; 0,4}.
    engine = _OneBadFrameEngine(bad_ts=0.3)
    _run(tmp_path, engine)
    texts = {round(ts, 3): boxes for _, ts, boxes in _frames(tmp_path)}
    assert texts[0.3] == []                                           # khung hỏng: rỗng
    assert texts[0.4] == [_text_at(0.4)]                              # cùng lô nhưng KHÔNG mất theo
    assert texts[0.1] == [_text_at(0.1)] and texts[0.6] == [_text_at(0.6)]
    assert any(call == 1 for call in engine.calls)                    # đã thử lại từng khung


def test_leading_duplicate_frame_is_not_dropped(tmp_path: Path) -> None:
    class _DupFirst(_Sampler):
        def iter_frames(self, *, metadata, roi, config):
            frames = list(super().iter_frames(metadata=metadata, roi=roi, config=config))
            first = frames[0]
            yield type(first)(frame_index=first.frame_index, timestamp_sec=first.timestamp_sec,
                              image_rgb=first.image_rgb, is_duplicate=True)
            yield from frames[1:]

    _run(tmp_path, _Engine(), _DupFirst())
    indices = [index for index, _ts, _boxes in _frames(tmp_path)]
    assert indices == list(range(int(DURATION / STEP)))               # không thủng khung nào


def test_broken_engine_does_not_retry_every_frame_of_every_batch(tmp_path: Path) -> None:
    class _AlwaysBroken(_Engine):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def infer_batch(self, images_rgb, frame_indices, timestamps_sec):
            self.calls += 1
            raise OcrInferenceError("engine hỏng")

    engine = _AlwaysBroken()
    _run(tmp_path, engine)
    batches = len([1 for _ in range(1)])  # chỉ để rõ ý: số lô nhỏ, mỗi lô tối đa 1 + 3 lần thử
    assert engine.calls <= (int(DURATION / STEP) // 2 + 1) * 4 + batches
    assert [index for index, _ts, _boxes in _frames(tmp_path)] == list(range(int(DURATION / STEP)))


def test_engine_returning_a_wrong_number_of_results_is_realigned(tmp_path: Path) -> None:
    """[v3.23.644] Lệch số kết quả (kể cả khi có tiền xử lý VRAM/viền) không làm sập cả lượt trích xuất."""
    import dataclasses

    from subtitles_extractor.domain.ports.frame_sampler_port import FrameSamplingConfig
    from tests.unit.application import test_ocr_checkpoint_v323632 as base

    class _ShortEngine(_Engine):
        def infer_batch(self, images_rgb, frame_indices, timestamps_sec):
            return super().infer_batch(images_rgb, frame_indices, timestamps_sec)[:-1]

    original = base.FrameSamplingConfig
    try:
        base.FrameSamplingConfig = lambda **kw: dataclasses.replace(
            FrameSamplingConfig(**kw), vram_add_border=True, vram_border_thickness_px=4
        )
        _run(tmp_path, _ShortEngine())
    finally:
        base.FrameSamplingConfig = original
    assert [index for index, _ts, _boxes in _frames(tmp_path)] == list(range(int(DURATION / STEP)))
