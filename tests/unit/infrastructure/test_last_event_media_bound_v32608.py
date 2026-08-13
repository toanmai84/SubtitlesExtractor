"""[v3.23.208] Test câu CUỐI nén nhẹ để VỪA biên video thay vì bị cắt tại biên.

Tương tác v207 (file = đúng thời lượng video) với logic nén: câu cuối dùng gap ẢO 2s
(không biết video kết thúc ở đâu) -> audio kết thúc SAU biên video -> bị cắt cứng tại
biên. Ca thật 第1集: câu cuối audio 4.49s, effective cũ 4.60s -> không nén -> kết thúc
222.41s > video 221.99 -> cắt 0.42s. Fix: dùng ``media_duration_s`` làm "câu sau ảo"
cho câu cuối -> effective 3.97s -> nén nhẹ 1.13x -> kết thúc 221.89s, TRỌN VẸN.
Đồng bộ cả VieNeu lẫn Gemini (2 đường).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from subtitles_extractor.domain.ports.subtitle_tts_port import TTSRequest
from subtitles_extractor.infrastructure.tts import vieneu_tts_adapter as vmod
from subtitles_extractor.infrastructure.tts.vieneu_tts_adapter import (
    VieNeuTtsAdapter,
    effective_available_seconds,
)

_SR = 24000


@pytest.fixture(autouse=True)
def _clear_cache():
    vmod._ENGINE_CACHE.clear()
    yield
    vmod._ENGINE_CACHE.clear()


def test_last_event_effective_bounded_by_media() -> None:
    # Ca thật 第1集: media làm next_start ảo -> effective 3.97s (thay vì 4.60s ảo).
    eff = effective_available_seconds(217.92, 220.52, 221.99)
    assert eff == pytest.approx(3.97, abs=0.02)


@dataclass
class _Event:
    start_sec: float
    end_sec: float
    text: str


class _LongTailEngine:
    """Sinh audio 4.49s cho câu cuối (tái hiện ca 第1集)."""

    def infer(self, text, voice=None):
        n = int(_SR * 4.49)
        return (0.2 * np.sin(2 * np.pi * 200 * np.arange(n) / _SR)).astype(np.float32)

    def list_preset_voices(self):
        return [("Giọng A", "a")]

    def get_preset_voice(self, vid):
        return {"id": vid}


def _run(monkeypatch, tmp_path, media: float | None):
    adapter = VieNeuTtsAdapter()
    monkeypatch.setattr(adapter, "is_available", lambda: True)
    adapter._engine = _LongTailEngine()
    adapter._force_cpu = False
    request = TTSRequest(
        events=[_Event(217.92, 220.52, "Câu cuối phim")],
        speaker="a", normalize=False, base_speed=1.0, max_speed=3.0,
        media_duration_s=media,
    )
    return adapter.generate(request, tmp_path / "out")


def test_last_event_fits_video_boundary(monkeypatch, tmp_path) -> None:
    # Biết thời lượng video -> nén nhẹ, KHÔNG bị cắt tại biên.
    results = _run(monkeypatch, tmp_path, media=221.99)
    r = results[0]
    assert not r.was_truncated
    assert r.speed_used > 1.05  # có nén nhẹ (~1.13x)
    # Audio kết thúc TRƯỚC biên video (tính từ mốc THẬT — v218 cho phép đọc sớm 0.25s).
    assert r.adjusted_start_sec + r.audio_duration_s <= 221.99 + 0.02


def test_last_event_old_behavior_without_media(monkeypatch, tmp_path) -> None:
    # Không biết thời lượng -> gap ảo 2s như cũ (không nén; master tự đệm 5s đủ chỗ).
    results = _run(monkeypatch, tmp_path, media=None)
    r = results[0]
    assert r.speed_used == pytest.approx(1.0, abs=0.02)
    assert not r.was_truncated


def test_gemini_sources_use_media_as_last_boundary() -> None:
    import pathlib

    source = pathlib.Path(
        "src/subtitles_extractor/infrastructure/tts/gemini_tts_adapter.py"
    ).read_text(encoding="utf-8")
    assert source.count('else getattr(request, "media_duration_s", None)') >= 2
