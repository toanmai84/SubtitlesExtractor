"""[v3.23.217] Test khoảng nghỉ hội thoại KHÔNG bị nén và không làm giọng nén oan.

Bug: pause (mặc định 300ms, chèn khi dòng có dấu '-') được PREPEND vào audio TRƯỚC khi
tính stretch -> (1) bị nén theo tỉ lệ chung — 300ms còn 171-231ms, **nhịp hội thoại bị
bóp 23-43%%**; (2) tính vào độ dài audio nên GIỌNG bị nén VÌ pause — sai bản chất, vì
pause là khoảng LẶNG có chủ đích, không phải tiếng nói.

Fix (VieNeu + Gemini 2 đường): tính tỉ lệ nén trên riêng GIỌNG, khung dành cho giọng đã
trừ pause (``available - pause_s``); pause chèn SAU khi nén -> giữ ĐÚNG độ dài người
dùng đặt.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

import numpy as np
import pytest

from subtitles_extractor.domain.ports.subtitle_tts_port import TTSRequest
from subtitles_extractor.infrastructure.tts import vieneu_tts_adapter as vmod
from subtitles_extractor.infrastructure.tts.vieneu_tts_adapter import VieNeuTtsAdapter

_SR = 24000


@pytest.fixture(autouse=True)
def _clear_cache():
    vmod._ENGINE_CACHE.clear()
    yield
    vmod._ENGINE_CACHE.clear()


@dataclass
class _Event:
    start_sec: float
    end_sec: float
    text: str


class _VoiceEngine:
    """Sinh giọng 2.0s (không im lặng biên)."""

    def infer(self, text, voice=None):
        n = int(_SR * 2.0)
        return (0.25 * np.sin(2 * np.pi * 200 * np.arange(n) / _SR)).astype(np.float32)

    def list_preset_voices(self):
        return [("Giọng A", "a")]

    def get_preset_voice(self, vid):
        return {"id": vid}


def _leading_silence_s(audio: np.ndarray) -> float:
    win = int(0.01 * _SR)
    for k in range(0, len(audio) - win, win):
        if float(np.sqrt((audio[k : k + win] ** 2).mean())) > 0.01:
            return k / _SR
    return len(audio) / _SR


def _run(monkeypatch, tmp_path, text: str, base: float = 1.3, pause_ms: int = 300):
    adapter = VieNeuTtsAdapter()
    monkeypatch.setattr(adapter, "is_available", lambda: True)
    adapter._engine = _VoiceEngine()
    adapter._force_cpu = False
    request = TTSRequest(
        events=[_Event(0.0, 2.0, text)], speaker="a", normalize=False,
        base_speed=base, max_speed=2.5, dialog_pause_ms=pause_ms,
    )
    results = adapter.generate(request, tmp_path / "out")
    written = sorted(tmp_path.glob("out.*"))
    import soundfile as sf

    audio, _ = sf.read(str(written[0]), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return results[0], audio


def test_dialog_pause_keeps_full_length(monkeypatch, tmp_path) -> None:
    # Dòng hội thoại (dấu '-') + pause 300ms + nén 1.3x -> pause vẫn ĐỦ 300ms
    # (trước fix: bị nén còn ~231ms).
    _, audio = _run(monkeypatch, tmp_path, "- Xin chào anh")
    assert _leading_silence_s(audio) == pytest.approx(0.30, abs=0.03)


def test_no_pause_for_non_dialog_line(monkeypatch, tmp_path) -> None:
    _, audio = _run(monkeypatch, tmp_path, "Câu thường không có gạch")
    assert _leading_silence_s(audio) < 0.05


def test_voice_not_compressed_because_of_pause(monkeypatch, tmp_path) -> None:
    # Giọng 2.0s, khung 2.0s, base 1.3 -> giọng nén đúng 1.3x; pause KHÔNG kéo tỉ lệ lên.
    result, audio = _run(monkeypatch, tmp_path, "- Xin chào anh")
    assert result.speed_used == pytest.approx(1.3, abs=0.02)
    # Tổng = pause 0.3 + giọng 2.0/1.3 = 0.3 + 1.54 = 1.84s
    assert result.audio_duration_s == pytest.approx(0.3 + 2.0 / 1.3, rel=0.05)


def test_pause_zero_behaves_as_before(monkeypatch, tmp_path) -> None:
    result, audio = _run(monkeypatch, tmp_path, "- Xin chào anh", pause_ms=0)
    assert _leading_silence_s(audio) < 0.05
    assert result.audio_duration_s == pytest.approx(2.0 / 1.3, rel=0.05)


def test_gemini_both_paths_synced() -> None:
    source = pathlib.Path(
        "src/subtitles_extractor/infrastructure/tts/gemini_tts_adapter.py"
    ).read_text(encoding="utf-8")
    # Khung dành cho giọng đã trừ pause, ở CẢ 2 đường.
    assert source.count("available_for_voice = max(0.05, effective_available - pause_s)") == 2
    assert source.count("voice_dur, available_for_voice") >= 2  # +1 dòng log debug
    # Pause chèn SAU nén, cả 2 đường (không còn prepend trước khi tính tỉ lệ).
    assert source.count("if pause_s > 0.0:") == 2
