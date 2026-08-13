"""[v3.23.205] Test audio TOÀN IM LẶNG kích hoạt retry (không mất thoại âm thầm).

Phát hiện từ đối chiếu FLAC thực (video 1.mp4): câu #58 "Ý chú là" — VieNeu sinh audio
2.2s nhưng TOÀN im lặng (RMS ~0.003), lọt qua lưới ``size > 0`` (chỉ bắt audio rỗng như
câu "Ưm") -> khung câu đó câm hoàn toàn mà kết quả vẫn báo OK. Fix:
``is_effectively_silent`` (RMS < 0.005) -> coi như thất bại -> retry; hết retry -> skip
có thông báo (người dùng BIẾT câu nào mất thay vì mất âm thầm).
"""

from __future__ import annotations

import numpy as np
import pytest

from subtitles_extractor.domain.ports.subtitle_tts_port import TTSRequest
from subtitles_extractor.infrastructure.tts import vieneu_tts_adapter as vmod
from subtitles_extractor.infrastructure.tts.vieneu_tts_adapter import (
    VieNeuTtsAdapter,
    is_effectively_silent,
)

_SR = 24000


@pytest.fixture(autouse=True)
def _clear_cache():
    vmod._ENGINE_CACHE.clear()
    yield
    vmod._ENGINE_CACHE.clear()


def _voice(duration_s: float = 0.5) -> np.ndarray:
    t = np.arange(int(_SR * duration_s)) / _SR
    return (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


# ── is_effectively_silent ────────────────────────────────────────────────


def test_long_silence_detected() -> None:
    # Ca #58: dài 2.2s, gần như im (nhiễu sàn ~0.003).
    noise = (0.003 * np.random.default_rng(7).standard_normal(int(_SR * 2.2))).astype(
        np.float32
    )
    assert is_effectively_silent(noise)


def test_normal_voice_not_flagged() -> None:
    assert not is_effectively_silent(_voice())


def test_quiet_but_audible_voice_not_flagged() -> None:
    # Giọng nhỏ nhất đo được thực tế ~0.04 RMS — không được bắt nhầm.
    quiet = (0.04 * np.sqrt(2) * np.sin(2 * np.pi * 220 * np.arange(_SR) / _SR)).astype(
        np.float32
    )
    assert not is_effectively_silent(quiet)


def test_empty_is_silent() -> None:
    assert is_effectively_silent(np.zeros(0, dtype=np.float32))


# ── retry khi im lặng ────────────────────────────────────────────────────


class _SilentThenVoiceEngine:
    """Lần 1 trả im lặng dài (ca #58), lần 2 trả tiếng — retry phải cứu được."""

    def __init__(self) -> None:
        self.calls = 0

    def infer(self, text, voice=None):
        self.calls += 1
        if self.calls == 1:
            return np.zeros(int(_SR * 2.2), dtype=np.float32)  # im lặng CÓ độ dài
        return _voice()


def test_silent_audio_triggers_retry(monkeypatch) -> None:
    adapter = VieNeuTtsAdapter()
    engine = _SilentThenVoiceEngine()
    request = TTSRequest(events=[], retry_count=3, retry_delay_s=0.0)
    audio = adapter._synthesize_with_retry(
        engine=engine, text="Ý chú là", request=request,
        voice_data={"id": "a"}, cancel_cb=None,
    )
    assert engine.calls == 2            # lần 1 im lặng -> retry, lần 2 OK
    assert audio is not None
    assert not is_effectively_silent(audio)


class _AlwaysSilentEngine:
    def infer(self, text, voice=None):
        return np.zeros(int(_SR * 2.0), dtype=np.float32)


def test_persistent_silence_returns_none(monkeypatch) -> None:
    # Hết retry vẫn im lặng -> None -> caller skip CÓ THÔNG BÁO (không mất âm thầm).
    adapter = VieNeuTtsAdapter()
    request = TTSRequest(events=[], retry_count=2, retry_delay_s=0.0)
    audio = adapter._synthesize_with_retry(
        engine=_AlwaysSilentEngine(), text="Ý chú là", request=request,
        voice_data={"id": "a"}, cancel_cb=None,
    )
    assert audio is None
