"""[v3.23.613] Một dòng Edge TTS bị huỷ riêng lẻ không được làm hỏng cả lượt lồng tiếng.

``asyncio.gather(..., return_exceptions=True)`` trả ``CancelledError`` của tác vụ con như
một "kết quả". Từ Python 3.8 ``CancelledError`` là ``BaseException`` chứ không phải
``Exception``, nên bốn vòng lọc ``isinstance(_item, Exception)`` để lọt nó qua và đem đi
tách tuple → ``TypeError`` làm hỏng cả lượt (mất mọi dòng đã đọc xong).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

from subtitles_extractor.domain.ports.subtitle_tts_port import TTSRequest
from subtitles_extractor.infrastructure.tts.edge_tts_adapter import (
    EdgeTTSAdapter,
    completed_task_results,
)
from tests.unit.infrastructure.test_dubbing_engines_v323593 import _SR, EVENTS, _sine

CANCELLED_LINE = EVENTS[1].text


@pytest.mark.parametrize("double_pass", [False, True])
def test_cancelled_line_is_skipped(
    double_pass: bool, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def fake_generate(text: str, _voice: str, _rate: str) -> tuple[np.ndarray, int]:
        if text == CANCELLED_LINE:
            raise asyncio.CancelledError
        return _sine(), _SR

    monkeypatch.setattr(EdgeTTSAdapter, "_async_generate", staticmethod(fake_generate))
    monkeypatch.setattr(EdgeTTSAdapter, "is_available", lambda self: True)
    request = TTSRequest(
        events=EVENTS, speaker="vi-VN-HoaiMyNeural", normalize=False, retry_count=1,
        double_pass=double_pass,
    )
    results = EdgeTTSAdapter().generate(request, tmp_path / "out.wav")
    by_text = {result.text: result for result in results}
    assert set(by_text) == {event.text for event in EVENTS}
    assert all(
        not by_text[event.text].error_msg for event in EVENTS
        if event.text != CANCELLED_LINE
    )
    assert by_text[CANCELLED_LINE].error_msg


def test_completed_task_results_drops_every_failure() -> None:
    outcomes: list[object] = [
        (1, "a"), ValueError("lỗi"), asyncio.CancelledError(), (2, "b"),
    ]
    assert completed_task_results(outcomes, "Pass 1") == [(1, "a"), (2, "b")]


def test_gemini_tts_logs_failed_lines(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Gemini TTS: dòng lỗi trước đây bị ``gather`` nuốt im lặng (không một dòng log)."""
    import logging

    from subtitles_extractor.infrastructure.tts.gemini_tts_adapter import (
        GeminiTTSAdapter,
    )

    async def fake_single(**kwargs: object) -> np.ndarray:
        if kwargs["text"] == CANCELLED_LINE:
            raise RuntimeError("quota hết giữa chừng")
        return _sine()

    adapter = GeminiTTSAdapter(api_key="k")
    monkeypatch.setattr(adapter, "_async_native_single_with_retry", fake_single)
    request = TTSRequest(events=EVENTS, speaker="Kore", normalize=False, retry_count=1)
    with caplog.at_level(logging.ERROR):
        audio_map = asyncio.run(
            adapter._async_generate_concurrent(
                model="m", voice="Kore", valid_indices=[0, 1, 2], request=request,
                use_affective=False, cancel_cb=None, progress_cb=None, n_valid=3,
            )
        )
    assert sorted(audio_map) == [0, 2]
    assert any("quota hết giữa chừng" in r.getMessage() for r in caplog.records)
