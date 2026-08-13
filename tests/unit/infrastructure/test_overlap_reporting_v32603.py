"""[v3.23.203] Test báo cáo CHỒNG THẬT (overlap_s) ở VieNeu + Gemini.

Bug rà soát (mâu thuẫn dữ liệu thật): config summary người dùng báo "Có chồng tiếng
(lấn): 0" trong khi phân tích CSV thấy câu #8 chồng 0.31s lên câu sau. Nguyên nhân: chỉ
Edge set ``overlap_s``; VieNeu/Gemini bỏ trống -> UI đếm luôn 0 -> người dùng MẤT thông
tin để tự điều chỉnh (nâng max_speed / biên tập phụ đề). Fix: cả hai set
``overlap_s = max(0, audio_cuối / sr - khung_hiệu_dụng)`` — đồng bộ ngữ nghĩa Edge
(phần vượt cửa sổ an toàn = chồng thật).
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


class _FixedDurEngine:
    def __init__(self, duration_s: float) -> None:
        self._duration_s = duration_s

    def list_preset_voices(self):
        return [("Giọng A", "a")]

    def get_preset_voice(self, vid):
        return {"id": vid}

    def infer(self, text, voice=None):
        n = int(_SR * self._duration_s)
        return (0.3 * np.sin(2 * np.pi * 200 * np.arange(n) / _SR)).astype(np.float32)


def _run(monkeypatch, tmp_path, audio_dur: float, frame_end: float, **req):
    adapter = VieNeuTtsAdapter()
    monkeypatch.setattr(adapter, "is_available", lambda: True)
    adapter._engine = _FixedDurEngine(audio_dur)
    adapter._force_cpu = False
    request = TTSRequest(
        events=[_Event(0.0, frame_end, "Câu đo")], speaker="a", normalize=False,
        base_speed=1.0, max_speed=1.0,  # không nén -> đo overlap thuần
        **req,
    )
    return adapter.generate(request, tmp_path / "out")


def test_overlap_reported_when_audio_spills(monkeypatch, tmp_path) -> None:
    # Audio 5s / khung 1s, câu cuối (nới trần gap 2s -> hiệu dụng 3s), không nén
    # -> chồng thật 2s phải được BÁO (trước fix: overlap_s = 0 mặc định).
    results = _run(monkeypatch, tmp_path, audio_dur=5.0, frame_end=1.0)
    assert results[0].overlap_s == pytest.approx(2.0, abs=0.1)


def test_no_overlap_when_fits(monkeypatch, tmp_path) -> None:
    # Audio 0.5s / khung 1s -> không chồng.
    results = _run(monkeypatch, tmp_path, audio_dur=0.5, frame_end=1.0)
    assert results[0].overlap_s == 0.0


def test_ui_counts_overlap_from_results() -> None:
    # Khoá điều kiện UI đếm: overlap_s > 0 (tts_page dòng ~1090).
    import pathlib

    source = pathlib.Path(
        "src/subtitles_extractor/presentation/pages/tts_page.py"
    ).read_text(encoding="utf-8")
    assert "r.overlap_s for r in ok if r.overlap_s > 0" in source


def test_gemini_source_sets_overlap_both_paths() -> None:
    import pathlib

    source = pathlib.Path(
        "src/subtitles_extractor/infrastructure/tts/gemini_tts_adapter.py"
    ).read_text(encoding="utf-8")
    assert source.count("overlap_s=max(0.0,") >= 2  # đường thường + batch
