"""[v3.23.211] Test bỏ NGAY văn bản không có nội dung đọc được (không retry vô ích).

Rà chéo parity phát hiện: ``_has_speakable_content`` CHỈ có ở Edge. VieNeu/Gemini gửi
thẳng vào model những dòng chỉ gồm dấu câu/ký hiệu/ô vuông "□" rác từ OCR (đo thực trên
phụ đề gốc của người dùng: **34/896 dòng**) -> model sinh im lặng -> retry hết 10 lần
VÔ ÍCH (~30s inference + 45s delay; Gemini còn đốt 10 lần gọi API = tốn quota) -> cuối
cùng báo SAI nguyên nhân ("Thất bại sau 10 lần" thay vì "không có nội dung đọc được").

Fix: hàm thuần ``has_speakable_content`` dùng chung; cả 3 lưới skip (VieNeu 1 + Gemini
2) bỏ ngay với thông báo ĐÚNG. Lưu ý không bắt nhầm thán từ có chữ ("Ừm." vẫn đọc).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from subtitles_extractor.domain.ports.subtitle_tts_port import TTSRequest
from subtitles_extractor.infrastructure.tts import vieneu_tts_adapter as vmod
from subtitles_extractor.infrastructure.tts.vieneu_tts_adapter import (
    VieNeuTtsAdapter,
    has_speakable_content,
)

_SR = 24000


@pytest.fixture(autouse=True)
def _clear_cache():
    vmod._ENGINE_CACHE.clear()
    yield
    vmod._ENGINE_CACHE.clear()


# ── has_speakable_content (hàm thuần) ────────────────────────────────────


@pytest.mark.parametrize(
    "text",
    ["□□□", "□ □ ■", "...", "…", "—", "♪♪", "?!", "   ", ""],
)
def test_unspeakable_texts(text: str) -> None:
    assert not has_speakable_content(text)


@pytest.mark.parametrize(
    "text",
    ["Ừm.", "Ưm.", "Chú nói đi.", "60km/h", "A", "Ý chú là"],
)
def test_speakable_texts(text: str) -> None:
    # Thán từ ngắn có CHỮ vẫn phải đọc — không bắt nhầm (v205 retry lo phần im lặng).
    assert has_speakable_content(text)


# ── VieNeu: bỏ ngay, KHÔNG gọi model ─────────────────────────────────────


@dataclass
class _Event:
    start_sec: float
    end_sec: float
    text: str


class _CountingEngine:
    def __init__(self) -> None:
        self.calls = 0

    def infer(self, text, voice=None):
        self.calls += 1
        return (0.2 * np.sin(2 * np.pi * 220 * np.arange(_SR) / _SR)).astype(np.float32)

    def list_preset_voices(self):
        return [("Giọng A", "a")]

    def get_preset_voice(self, vid):
        return {"id": vid}


def test_unspeakable_line_skipped_without_model_call(monkeypatch, tmp_path) -> None:
    adapter = VieNeuTtsAdapter()
    monkeypatch.setattr(adapter, "is_available", lambda: True)
    engine = _CountingEngine()
    adapter._engine = engine
    adapter._force_cpu = False
    request = TTSRequest(
        events=[_Event(1.0, 3.0, "□□□"), _Event(4.0, 6.0, "Câu thật")],
        speaker="a", normalize=False, retry_count=10,
    )
    results = adapter.generate(request, tmp_path / "out")
    assert results[0].was_skipped
    assert "Không có nội dung đọc được" in (results[0].error_msg or "")
    assert not results[1].was_skipped
    # Model chỉ được gọi cho câu THẬT — dòng rác không tốn 10 lần inference.
    assert engine.calls == 1


def test_gemini_both_paths_check_speakable() -> None:
    import pathlib

    source = pathlib.Path(
        "src/subtitles_extractor/infrastructure/tts/gemini_tts_adapter.py"
    ).read_text(encoding="utf-8")
    assert source.count("has_speakable_content(") >= 2  # standard + native/batch
    assert source.count("Không có nội dung đọc được") >= 2
