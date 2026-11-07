"""[v3.23.645] Dự trù request của phân tích tính cả các lượt bổ sung người nói."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from subtitles_extractor.application.services.cue_completion import (
    SPEAKER_COMPLETION_PASSES,
)
from subtitles_extractor.presentation.workers.analyze_context_worker import (
    AnalyzeContextWorker,
)


def _worker(cues: bool, provider: object) -> AnalyzeContextWorker:
    return AnalyzeContextWorker(
        SimpleNamespace(), [], "vi", video_provider=provider,
        video_path="x.mp4", enable_visual_cues=cues,
    )


class _Provider:
    def estimate_chunk_count(self, path: Path) -> int:
        return 24


class _BrokenProvider:
    def estimate_chunk_count(self, path: Path) -> int:
        raise OSError("không đọc được")


@pytest.mark.parametrize("cues,extra", [(False, 0), (True, 2 + SPEAKER_COMPLETION_PASSES)])
def test_needed_requests_counts_speaker_completion(cues: bool, extra: int) -> None:
    # [v3.23.667] Có Visual Cues: thêm 2 lời gọi chữ của phân tích tách bước.
    assert _worker(cues, _Provider())._needed_requests() == 24 + extra
    assert _worker(cues, _BrokenProvider())._needed_requests() == 1 + extra
    assert _worker(cues, object())._needed_requests() == 1 + extra


def test_needed_requests_tinh_so_lo_quet_nguoi_noi_v323667() -> None:
    """Một đoạn video nhưng 600 dòng → bộ quét gửi 4 lô 150 dòng (mỗi lô một lời gọi)."""
    lines = [object()] * 600

    class _One:
        def estimate_chunk_count(self, path: Path) -> int:
            return 1

    worker = AnalyzeContextWorker(SimpleNamespace(), lines, "vi", video_provider=_One(),
                                  video_path="x.mp4", enable_visual_cues=True)
    assert worker._needed_requests() == 2 + 4 + SPEAKER_COMPLETION_PASSES
