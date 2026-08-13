"""[v3.23.218] Test "Ăn gian đầu" (lead-in): tận dụng khoảng
    # lặng câu trước để nén nhẹ hơn.

Phân tích 2 video thật (v216, base 1.3, trần 2.0): các câu phải nén KỊCH TRẦN
2.0x đều có **khung phụ đề rất ngắn** (0.20-0.72s), trong khi câu TRƯỚC thường đọc xong
sớm và để lại khoảng lặng không dùng đến. Edge vốn có tham số "Ăn gian đầu"
(lead_in) cho việc này; VieNeu/Gemini thì KHÔNG -> nén mạnh vô ích, giọng mất ~19%% độ
sắc phụ âm (đo ở v216).

Mô phỏng trên chính dữ liệu 2 video: cho phép đọc sớm tối đa 0.25s vào chỗ trống thật ->
số câu nén kịch trần **8 -> 2** (video 1) và **6 -> 2** (video 2); 29 và 17 câu được nén
nhẹ hơn. Không đè lên tiếng câu trước và KHÔNG dời câu sau -> không có hiệu ứng domino
(phương án "dời mốc" đã bị bác bỏ bằng mô phỏng ở v204).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from subtitles_extractor.domain.ports.subtitle_tts_port import TTSRequest
from subtitles_extractor.infrastructure.tts import vieneu_tts_adapter as vmod
from subtitles_extractor.infrastructure.tts.vieneu_tts_adapter import (
    VieNeuTtsAdapter,
    lead_in_seconds,
)

_SR = 24000


@pytest.fixture(autouse=True)
def _clear_cache():
    vmod._ENGINE_CACHE.clear()
    yield
    vmod._ENGINE_CACHE.clear()


# ── lead_in_seconds (hàm thuần) ──────────────────────────────────────────


def test_uses_free_gap_up_to_cap() -> None:
    assert lead_in_seconds(5.0, 4.5) == 0.25  # còn nhiều chỗ -> ăn tới trần


def test_limited_by_actual_free_gap() -> None:
    assert lead_in_seconds(5.0, 4.95) == pytest.approx(0.05)  # chỉ 50ms trống


def test_never_overlaps_previous_voice() -> None:
    # Câu trước TRÀN quá mốc câu này -> KHÔNG ăn gian (không đè lên tiếng).
    assert lead_in_seconds(5.0, 5.2) == 0.0


def test_first_event_can_lead() -> None:
    assert lead_in_seconds(5.0, 0.0) == 0.25


def test_first_event_near_zero_safe() -> None:
    # Câu đầu ở 0.1s -> chỉ ăn được 0.1s (không âm).
    assert lead_in_seconds(0.1, 0.0) == pytest.approx(0.1)


# ── [v3.23.219] CHỈ ăn gian khi thực sự CẦN ──────────────────────────────


def test_no_lead_when_not_needed() -> None:
    # Câu đã vừa khung ở tốc độ cơ bản -> GIỮ NGUYÊN mốc phụ đề (đồng bộ khẩu hình).
    # Bug v218: dời sớm 250ms cả 79/95 câu không cần -> nghe "không đồng bộ".
    assert lead_in_seconds(5.0, 4.0, 0.25, needed_lead_s=0.0) == 0.0


def test_lead_limited_to_actual_need() -> None:
    # Cần 0.1s -> chỉ ăn 0.1s, KHÔNG lấy hết trần 0.25s.
    assert lead_in_seconds(5.0, 4.0, 0.25, needed_lead_s=0.1) == pytest.approx(0.1)


def test_need_capped_by_max_lead() -> None:
    assert lead_in_seconds(5.0, 4.0, 0.25, needed_lead_s=0.5) == 0.25


def test_need_capped_by_free_gap() -> None:
    # Cần nhiều nhưng chỗ trống thật chỉ 80ms -> chỉ ăn 80ms (không đè tiếng câu trước).
    assert lead_in_seconds(5.0, 4.92, 0.25, needed_lead_s=0.3) == pytest.approx(0.08)


# ── End-to-end: khung rộng hơn -> nén nhẹ hơn ────────────────────────────


@dataclass
class _Event:
    start_sec: float
    end_sec: float
    text: str


class _FixedEngine:
    """Câu 1: giọng ngắn 0.5s. Câu 2: giọng 1.6s (cần nén trong khung 0.8s)."""

    def __init__(self) -> None:
        self.calls = 0

    def infer(self, text, voice=None):
        self.calls += 1
        dur = 0.5 if self.calls == 1 else 1.6
        n = int(_SR * dur)
        return (0.25 * np.sin(2 * np.pi * 200 * np.arange(n) / _SR)).astype(np.float32)

    def list_preset_voices(self):
        return [("Giọng A", "a")]

    def get_preset_voice(self, vid):
        return {"id": vid}


def _run(monkeypatch, tmp_path):
    adapter = VieNeuTtsAdapter()
    monkeypatch.setattr(adapter, "is_available", lambda: True)
    adapter._engine = _FixedEngine()
    adapter._force_cpu = False
    request = TTSRequest(
        # Câu 1 khung 0-2s (giọng chỉ 0.5s -> xong lúc ~0.5s, để trống tới 2.0s).
        # Câu 2 khung 2.0-2.8s (rất ngắn) + câu 3 sát ngay -> phải nén mạnh.
        events=[
            _Event(0.0, 2.0, "Câu ngắn"),
            _Event(2.0, 2.8, "Câu dài cần nén nhiều"),
            _Event(2.9, 4.0, "Câu ba"),
        ],
        speaker="a", normalize=False, base_speed=1.0, max_speed=2.5,
        media_duration_s=6.0,
    )
    return adapter.generate(request, tmp_path / "out")


def test_lead_in_reduces_compression(monkeypatch, tmp_path) -> None:
    results = _run(monkeypatch, tmp_path)
    second = results[1]
    # Khung gốc 0.8s + gap 0.0 -> nếu KHÔNG ăn gian: 1.6/0.8 = 2.0x (kịch trần).
    # Có ăn gian 0.25s (câu trước xong sớm) -> khung ~1.05s -> nén ~1.52x.
    assert second.speed_used < 1.95, "phải nén NHẸ hơn nhờ ăn gian đầu"
    assert second.speed_used > 1.0


def test_adjusted_start_reflects_early_read(monkeypatch, tmp_path) -> None:
    # Câu 2 THỰC SỰ cần khung rộng hơn (giọng 1.6s / khung 0.8s) -> được đọc sớm; SRT
    # đồng bộ TTS phải khớp tiếng.
    results = _run(monkeypatch, tmp_path)
    second = results[1]
    lead = TTSRequest(events=[]).lead_in_s  # "Ăn gian đầu" người dùng đặt
    assert second.adjusted_start_sec == pytest.approx(2.0 - lead, abs=0.01)
    assert second.adjusted_start_sec < second.start_sec


def test_first_event_keeps_original_timing(monkeypatch, tmp_path) -> None:
    # [v3.23.219] Câu 1 (giọng 0.5s, khung 2.0s) KHÔNG cần ăn gian -> giữ đúng mốc phụ
    # đề, tiếng không lệch trước khẩu hình.
    results = _run(monkeypatch, tmp_path)
    assert results[0].adjusted_start_sec == pytest.approx(results[0].start_sec, abs=0.01)


def test_no_lead_when_previous_voice_still_playing(monkeypatch, tmp_path) -> None:
    # Câu 3 bắt đầu 2.9s nhưng câu 2 (đã nén) còn đang đọc -> không ăn gian được nhiều.
    results = _run(monkeypatch, tmp_path)
    third = results[2]
    prev_end = results[1].adjusted_start_sec + results[1].audio_duration_s
    assert third.adjusted_start_sec >= prev_end - 0.01  # không đè lên tiếng câu 2
