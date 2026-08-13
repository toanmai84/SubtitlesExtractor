"""[v3.23.215] Test "Tốc độ cơ bản" THỰC SỰ áp dụng ở VieNeu/Gemini (hết cấu hình ma).

Bug: ``base_speed`` chỉ dùng để (1) tính trần và (2) gán NHÃN ``speed_used``. Audio chỉ
bị nén khi DÀI HƠN khung (``fit_ratio``) -> câu vừa khung (ĐA SỐ — median speed ghi nhận
trên dữ liệu thật = đúng base, chứng tỏ ratio=1) phát ở tốc độ GỐC 1.0x dù người dùng
đặt 1.3x (chậm hơn 30%% so với mong đợi), mà báo cáo vẫn ghi "1.30x" -> sai sự thật.

Fix: hàm thuần ``total_speed_ratio`` — audio luôn đọc TỐI THIỂU ở ``base_speed``, nén
thêm nếu chưa vừa khung, chặn bởi ``max_speed`` (trần người dùng) và trần chất lượng
2.0 (ngưỡng vật lý đo được ở v201: nén hơn nữa làm tan formant). ``speed_used`` = tỉ lệ
THẬT áp lên audio (không còn nhân đôi base).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from subtitles_extractor.domain.ports.subtitle_tts_port import TTSRequest
from subtitles_extractor.infrastructure.tts import vieneu_tts_adapter as vmod
from subtitles_extractor.infrastructure.tts.vieneu_tts_adapter import (
    VieNeuTtsAdapter,
    total_speed_ratio,
)

_SR = 24000


@pytest.fixture(autouse=True)
def _clear_cache():
    vmod._ENGINE_CACHE.clear()
    yield
    vmod._ENGINE_CACHE.clear()


# ── total_speed_ratio (hàm thuần) ────────────────────────────────────────


def test_base_speed_applied_when_audio_fits() -> None:
    # Ca ĐA SỐ: audio vừa khung -> vẫn phải đọc ở base (trước fix: 1.0x).
    assert total_speed_ratio(2.0, 2.0, base_speed=1.3, max_speed=2.5) == 1.3


def test_base_speed_applied_when_audio_shorter() -> None:
    # Audio ngắn hơn khung -> vẫn đọc ở base, KHÔNG kéo chậm lại.
    assert total_speed_ratio(1.0, 3.0, base_speed=1.3, max_speed=2.5) == 1.3


def test_fit_wins_when_longer_than_base() -> None:
    # Audio dài -> nén cho vừa khung (fit 2.5) nhưng chặn ở trần chất lượng 2.0.
    assert total_speed_ratio(5.0, 2.0, base_speed=1.3, max_speed=2.5) == 2.0


def test_max_speed_caps_base() -> None:
    # Trần người dùng thắng base (cấu hình mâu thuẫn).
    assert total_speed_ratio(2.0, 2.0, base_speed=2.0, max_speed=1.5) == 1.5


def test_never_slows_down() -> None:
    # base 1.0 + audio ngắn -> giữ nguyên tốc độ model (không bao giờ < 1.0).
    assert total_speed_ratio(1.0, 5.0, base_speed=1.0, max_speed=2.5) == 1.0


def test_zero_window_safe() -> None:
    assert total_speed_ratio(2.0, 0.0, base_speed=1.3, max_speed=2.5) == 1.3


# ── End-to-end VieNeu: audio THẬT ngắn lại theo base ─────────────────────


@dataclass
class _Event:
    start_sec: float
    end_sec: float
    text: str


class _FixedEngine:
    """Sinh audio 2.0s — vừa khít khung 2.0s (ca 'đa số')."""

    def infer(self, text, voice=None):
        n = int(_SR * 2.0)
        return (0.2 * np.sin(2 * np.pi * 200 * np.arange(n) / _SR)).astype(np.float32)

    def list_preset_voices(self):
        return [("Giọng A", "a")]

    def get_preset_voice(self, vid):
        return {"id": vid}


def _run(monkeypatch, tmp_path, base: float):
    adapter = VieNeuTtsAdapter()
    monkeypatch.setattr(adapter, "is_available", lambda: True)
    adapter._engine = _FixedEngine()
    adapter._force_cpu = False
    request = TTSRequest(
        events=[_Event(0.0, 2.0, "Câu vừa khung")], speaker="a", normalize=False,
        base_speed=base, max_speed=2.5,
    )
    return adapter.generate(request, tmp_path / "out")[0]


def test_audio_actually_compressed_by_base_speed(monkeypatch, tmp_path) -> None:
    result = _run(monkeypatch, tmp_path, base=1.3)
    # Audio THẬT phải ngắn lại ~1/1.3 = 0.77x (trước fix: giữ nguyên 2.0s).
    assert result.audio_duration_s == pytest.approx(2.0 / 1.3, rel=0.05)
    assert result.speed_used == pytest.approx(1.3, abs=0.01)


def test_base_one_keeps_original_length(monkeypatch, tmp_path) -> None:
    result = _run(monkeypatch, tmp_path, base=1.0)
    assert result.audio_duration_s == pytest.approx(2.0, rel=0.05)
    assert result.speed_used == pytest.approx(1.0, abs=0.01)


def test_gemini_uses_shared_helper() -> None:
    import pathlib

    source = pathlib.Path(
        "src/subtitles_extractor/infrastructure/tts/gemini_tts_adapter.py"
    ).read_text(encoding="utf-8")
    assert source.count("total_speed_ratio(") >= 2  # standard + batch
    assert "request.base_speed, request.max_speed" in source
