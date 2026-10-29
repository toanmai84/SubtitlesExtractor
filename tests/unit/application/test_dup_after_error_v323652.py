"""[v3.23.652] Khung trùng sau một khung lỗi giải mã phải chép kết quả của khung ảnh thật gần nhất, không chép "rỗng"."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from subtitles_extractor.domain.ports.frame_sampler_port import SampledFrame
from tests.unit.application.test_ocr_checkpoint_v323632 import _Engine, _frames, _run, _text_at


class _ErrorThenDups:
    """Khung ảnh thật ở 0,0 s; 0,1 s lỗi giải mã; 0,2–0,9 s trùng với 0,0 s; sau đó khung thật mỗi 0,1 s."""

    def iter_frames(self, *, metadata, roi, config):
        idx = 0
        for step in range(100):
            ts = round(step * 0.1, 6)
            if step == 1:
                frame = SampledFrame(frame_index=idx, timestamp_sec=ts, image_rgb=np.empty(0), is_error=True)
            elif 2 <= step <= 9:
                frame = SampledFrame(frame_index=idx, timestamp_sec=ts, image_rgb=np.empty(0), is_duplicate=True)
            else:
                frame = SampledFrame(frame_index=idx, timestamp_sec=ts, image_rgb=np.zeros((20, 80, 3), np.uint8))
            yield frame
            idx += 1


def test_duplicates_after_decode_error_keep_text(tmp_path: Path) -> None:
    _run(tmp_path, _Engine(), _ErrorThenDups())
    texts = {round(ts, 3): boxes for _, ts, boxes in _frames(tmp_path)}
    assert texts[0.1] == []                       # khung lỗi: rỗng
    for step in range(2, 10):
        assert texts[round(step * 0.1, 3)] == [_text_at(0.0)], step
