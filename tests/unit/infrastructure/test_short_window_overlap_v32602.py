"""[v3.23.202] Test câu cửa sổ ngắn KHÔNG bị bỏ khi "Cho phép chồng tiếng" bật.

Bug rà soát: ``gap_threshold_s = 0.3`` mặc định -> câu khung < 300ms bị BỎ HẲN (VieNeu
1 chỗ + Gemini 2 chỗ) BẤT KỂ "Cho phép chồng tiếng". Phim drama CJK nhịp nhanh đầy câu
thoại ngắn ("Đi!", "Sao?"...) — với chồng tiếng bật, audio tràn tự nhiên (fit_limit
None) nên hoàn toàn đọc được; bỏ chúng là MẤT nội dung thoại vô cớ. Fix: chỉ bỏ khi
text rỗng, hoặc khung quá ngắn VÀ không cho chồng tiếng.
"""

from __future__ import annotations

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


class _FakeEngine:
    def list_preset_voices(self):
        return [("Giọng A", "a")]

    def get_preset_voice(self, vid):
        return {"id": vid}

    def infer(self, text, voice=None):
        return (0.3 * np.sin(2 * np.pi * 200 * np.arange(int(_SR * 0.5)) / _SR)).astype(
            np.float32
        )


def _prime(adapter, monkeypatch):
    monkeypatch.setattr(adapter, "is_available", lambda: True)
    adapter._engine = _FakeEngine()
    adapter._force_cpu = False


def test_short_window_read_when_overlap_allowed(monkeypatch, tmp_path) -> None:
    # Khung 0.2s < ngưỡng 0.3s NHƯNG cho phép chồng tiếng -> VẪN ĐỌC (tràn tự nhiên).
    adapter = VieNeuTtsAdapter()
    _prime(adapter, monkeypatch)
    request = TTSRequest(
        events=[_Event(1.0, 1.2, "Đi!")], speaker="a", normalize=False,
        allow_audio_overlap=True, gap_threshold_s=0.3,
    )
    results = adapter.generate(request, tmp_path / "out")
    assert not results[0].was_skipped  # trước fix: bị bỏ -> mất thoại


def test_short_window_skipped_when_overlap_disallowed(monkeypatch, tmp_path) -> None:
    # Không cho chồng tiếng -> khung quá ngắn bị bỏ như cũ (cắt khít ra mẩu vô nghĩa).
    adapter = VieNeuTtsAdapter()
    _prime(adapter, monkeypatch)
    request = TTSRequest(
        events=[_Event(1.0, 1.2, "Đi!")], speaker="a", normalize=False,
        allow_audio_overlap=False, gap_threshold_s=0.3,
    )
    results = adapter.generate(request, tmp_path / "out")
    assert results[0].was_skipped


def test_empty_text_always_skipped(monkeypatch, tmp_path) -> None:
    # Text rỗng luôn bỏ, bất kể chồng tiếng (kèm 1 câu thật để không rơi early-return
    # "toàn bộ rỗng" của generate).
    adapter = VieNeuTtsAdapter()
    _prime(adapter, monkeypatch)
    request = TTSRequest(
        events=[_Event(1.0, 3.0, "   "), _Event(3.0, 5.0, "Câu thật")],
        speaker="a", normalize=False, allow_audio_overlap=True,
    )
    results = adapter.generate(request, tmp_path / "out")
    assert results[0].was_skipped        # câu rỗng bị bỏ
    assert not results[1].was_skipped    # câu thật vẫn đọc


def test_gemini_source_synced() -> None:
    # Gemini (2 đường) cùng quy tắc: window_too_short gồm điều kiện allow_audio_overlap.
    import pathlib

    source = pathlib.Path(
        "src/subtitles_extractor/infrastructure/tts/gemini_tts_adapter.py"
    ).read_text(encoding="utf-8")
    assert source.count("window_too_short") >= 4  # 2 định nghĩa + 2 sử dụng
    assert source.count('and not getattr(request, "allow_audio_overlap", True)') >= 2
