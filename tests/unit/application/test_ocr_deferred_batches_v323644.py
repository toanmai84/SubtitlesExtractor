"""[v3.23.644] Khung trùng / trống / lỗi không còn cắt lô OCR; kết quả vẫn đúng thứ tự khung."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np

from subtitles_extractor.domain.ports.frame_sampler_port import FrameSamplingConfig, SampledFrame
from tests.unit.application.test_ocr_checkpoint_v323632 import (
    DURATION,
    STEP,
    _Engine,
    _frames,
    _run,
    _text_at,
)

TOTAL = int(DURATION / STEP)


class _PatternSampler:
    """Mẫu cố định theo mốc thời gian: ``kind(i)`` ∈ {ocr, dup, blank, error}."""

    def __init__(self, kind) -> None:
        self.kind = kind

    def iter_frames(self, *, metadata, roi, config: FrameSamplingConfig) -> Iterator[SampledFrame]:
        for idx in range(TOTAL):
            ts = round(idx * STEP, 6)
            kind = self.kind(idx)
            yield SampledFrame(
                frame_index=idx, timestamp_sec=ts,
                image_rgb=np.zeros((20, 80, 3), np.uint8) if kind == "ocr" else np.empty(0, np.uint8),
                is_duplicate=kind == "dup", is_blank=kind == "blank", is_error=kind == "error",
            )


class _RecordingEngine(_Engine):
    def __init__(self) -> None:
        super().__init__()
        self.batches: list[int] = []

    def infer_batch(self, images_rgb, frame_indices, timestamps_sec):
        self.batches.append(len(frame_indices))
        return super().infer_batch(images_rgb, frame_indices, timestamps_sec)


def test_duplicates_no_longer_split_the_batches(tmp_path: Path) -> None:
    engine = _RecordingEngine()
    _run(tmp_path, engine, _PatternSampler(lambda i: "dup" if i % 3 == 2 else "ocr"))
    assert max(engine.batches) == 4                         # batch_size của test = 4: lô đầy
    assert sum(engine.batches) == len([i for i in range(TOTAL) if i % 3 != 2])
    assert engine.batches[:-1] == [4] * (len(engine.batches) - 1)
    # trước đây: lô dài trung bình 2 vì mỗi khung trùng cắt lô


def test_results_are_complete_ordered_and_duplicates_copy_the_previous_frame(tmp_path: Path) -> None:
    _run(tmp_path, _RecordingEngine(), _PatternSampler(lambda i: "dup" if i % 3 == 2 else "ocr"))
    frames = _frames(tmp_path)
    assert [f[0] for f in frames] == list(range(TOTAL))
    for index, ts, boxes in frames:
        source = index - 1 if index % 3 == 2 else index
        assert boxes == [_text_at(round(source * STEP, 6))]


def test_blank_frames_get_empty_results_without_running_ocr(tmp_path: Path) -> None:
    engine = _RecordingEngine()
    _run(tmp_path, engine, _PatternSampler(lambda i: "blank" if 20 <= i < 40 else "ocr"))
    frames = _frames(tmp_path)
    assert [f[0] for f in frames] == list(range(TOTAL))
    assert all(boxes == [] for index, _ts, boxes in frames if 20 <= index < 40)
    assert all(boxes != [] for index, _ts, boxes in frames if not 20 <= index < 40)
    assert sum(engine.batches) == TOTAL - 20


def test_duplicate_after_a_blank_does_not_resurrect_old_text(tmp_path: Path) -> None:
    def kind(i: int) -> str:
        return {10: "blank", 11: "dup"}.get(i, "ocr")

    _run(tmp_path, _RecordingEngine(), _PatternSampler(kind))
    by_index = {index: boxes for index, _ts, boxes in _frames(tmp_path)}
    assert by_index[10] == [] and by_index[11] == []


def test_error_frames_stay_in_sequence_as_empty(tmp_path: Path) -> None:
    def kind(i: int) -> str:
        return "error" if i == 7 else ("dup" if i == 8 else "ocr")

    _run(tmp_path, _RecordingEngine(), _PatternSampler(kind))
    by_index = {index: boxes for index, _ts, boxes in _frames(tmp_path)}
    assert by_index[7] == []
    # [v3.23.652] Khung trùng 8 trùng với khung ẢNH THẬT gần nhất (6 — bộ lọc trùng chỉ nhớ khung đã OCR), không với
    # khung lỗi 7: chép chữ của 6 (trước đây ra rỗng → câu bị cắt khi có khung lỗi giải mã xen giữa).
    assert by_index[8] == by_index[6]
    assert by_index[6] != [] and by_index[9] != []


def test_a_long_static_subtitle_does_not_grow_the_queue_without_bound(tmp_path: Path) -> None:
    engine = _RecordingEngine()
    _run(tmp_path, engine, _PatternSampler(lambda i: "ocr" if i == 0 else "dup"))
    assert [f[2] for f in _frames(tmp_path)] == [[_text_at(0.0)]] * TOTAL
    assert engine.batches == [1]
