"""[v3.23.207] Test file TTS xuất ra dài ĐÚNG thời lượng video (mux không lệch).

Người dùng báo: file âm thanh TTS không đồng bộ thời lượng với âm thanh gốc. Đo thực
video 第1集: video 221.99s nhưng FLAC 225.52s (+3.53s) = end câu cuối 220.52 + đệm 5s —
master không hề biết thời lượng video. Fix: ``TTSRequest.media_duration_s`` (UI đọc từ
metadata video khi nạp) -> cả 3 engine cấp phát master ĐÚNG bằng video; audio câu cuối
tràn quá biên bị cắt tại biên (video đã hết hình — mux kiểu gì cũng cắt) và đánh dấu
was_truncated trung thực; không biết thời lượng -> hành vi cũ (tương thích).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import soundfile as sf

from subtitles_extractor.domain.ports.subtitle_tts_port import TTSRequest
from subtitles_extractor.infrastructure.tts import vieneu_tts_adapter as vmod
from subtitles_extractor.infrastructure.tts.edge_tts_adapter import (
    _master_track_length_samples,
)
from subtitles_extractor.infrastructure.tts.vieneu_tts_adapter import (
    VieNeuTtsAdapter,
    master_length_samples,
)

_SR = 24000


@pytest.fixture(autouse=True)
def _clear_cache():
    vmod._ENGINE_CACHE.clear()
    yield
    vmod._ENGINE_CACHE.clear()


# ── master_length_samples (hàm thuần) ────────────────────────────────────


def test_media_duration_wins() -> None:
    # Ca thực tế 第1集: video 221.99s, câu cuối end 220.52 -> master = ĐÚNG video.
    assert master_length_samples(220.52, _SR, 221.99) == int(round(221.99 * _SR))


def test_fallback_when_unknown() -> None:
    # Không biết thời lượng -> hành vi cũ (last_end + đệm 5s).
    assert master_length_samples(220.52, _SR, None) == int((220.52 + 5.0) * _SR)


def test_zero_or_negative_media_guarded() -> None:
    assert master_length_samples(10.0, _SR, 0.0) == int(15.0 * _SR)
    assert master_length_samples(10.0, _SR, -3.0) == int(15.0 * _SR)


def test_edge_master_uses_media_duration() -> None:
    n = _master_track_length_samples(
        220.52, _SR, 0.1, 0.2, 0.25, 0.3, media_duration_s=221.99
    )
    assert n == int(round(221.99 * _SR))


def test_request_field_exists() -> None:
    assert TTSRequest(events=[]).media_duration_s is None
    assert TTSRequest(events=[], media_duration_s=100.0).media_duration_s == 100.0


# ── End-to-end VieNeu: file xuất đúng thời lượng ─────────────────────────


@dataclass
class _Event:
    start_sec: float
    end_sec: float
    text: str


class _FakeEngine:
    def infer(self, text, voice=None):
        return (0.2 * np.sin(2 * np.pi * 200 * np.arange(_SR) / _SR)).astype(np.float32)

    def list_preset_voices(self):
        return [("Giọng A", "a")]

    def get_preset_voice(self, vid):
        return {"id": vid}


def test_output_file_matches_media_duration(monkeypatch, tmp_path) -> None:
    adapter = VieNeuTtsAdapter()
    monkeypatch.setattr(adapter, "is_available", lambda: True)
    adapter._engine = _FakeEngine()
    adapter._force_cpu = False
    request = TTSRequest(
        events=[_Event(1.0, 2.0, "Câu một"), _Event(5.0, 6.5, "Câu cuối")],
        speaker="a", normalize=False, media_duration_s=20.0,
    )
    out = tmp_path / "out.wav"
    adapter.generate(request, out)
    written = list(tmp_path.glob("out.*"))
    assert written, "không thấy file xuất"
    info = sf.info(str(written[0]))
    # File dài ĐÚNG 20.0s (thời lượng video), không phải last_end 6.5 + 5 = 11.5s.
    assert info.duration == pytest.approx(20.0, abs=0.05)
