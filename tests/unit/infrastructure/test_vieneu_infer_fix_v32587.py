"""[v3.23.187] Test sửa lỗi VieNeu ``infer`` — voice DICT + cache engine/voice.

Lỗi thực tế từ log máy người dùng: ``Must provide either 'voice' dict or both 'ref_codes'
and 'ref_text'``. Nguyên nhân: adapter v185 truyền ``voice=<string_id>`` và có nhánh
``infer(text=...)`` trơn — nhưng SDK bắt buộc ``voice`` là DICT (từ ``get_preset_voice``
hoặc ``encode_reference``). Bản sửa: giải quyết voice DICT một lần, cache lại; giọng mặc
định lấy preset đầu tiên. Test này khoá các hành vi đó.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from subtitles_extractor.domain.ports.subtitle_tts_port import (
    TTSGenerationError,
    TTSRequest,
)
from subtitles_extractor.infrastructure.tts import vieneu_tts_adapter as mod
from subtitles_extractor.infrastructure.tts.vieneu_tts_adapter import VieNeuTtsAdapter

_SR = 24000


@pytest.fixture(autouse=True)
def _clear_cache():
    mod._ENGINE_CACHE.clear()
    yield
    mod._ENGINE_CACHE.clear()


@dataclass
class _Event:
    start_sec: float
    end_sec: float
    text: str


class _CountingEngine:
    """Engine giả đếm số lần encode/get_preset để kiểm việc cache voice."""

    def __init__(self) -> None:
        self.encode_count = 0
        self.get_preset_count = 0
        self.infer_count = 0

    def list_preset_voices(self):
        return [("Giọng A", "voice_a"), ("Giọng B", "voice_b")]

    def get_preset_voice(self, voice_id: str) -> dict:
        self.get_preset_count += 1
        return {"id": voice_id}

    def encode_reference(self, ref_path: str) -> dict:
        self.encode_count += 1
        return {"ref": ref_path}

    def infer(self, text: str, voice=None):
        if voice is None:
            raise ValueError("Must provide either 'voice' dict or both 'ref_codes' and 'ref_text'.")
        self.infer_count += 1
        return (0.3 * np.sin(2 * np.pi * 200 * np.arange(int(_SR * 0.4)) / _SR)).astype(np.float32)


def _prime(adapter: VieNeuTtsAdapter, engine, monkeypatch) -> None:
    monkeypatch.setattr(adapter, "is_available", lambda: True)
    adapter._engine = engine


# ── Voice DICT được truyền đúng ──────────────────────────────────────────


def test_default_voice_uses_first_preset(monkeypatch, tmp_path) -> None:
    # Không chọn giọng, không ref -> phải lấy preset ĐẦU TIÊN làm mặc định (voice DICT).
    adapter = VieNeuTtsAdapter()
    engine = _CountingEngine()
    _prime(adapter, engine, monkeypatch)

    request = TTSRequest(
        events=[_Event(0.0, 2.0, "Xin chào")], speaker="", normalize=False,
    )
    results = adapter.generate(request, tmp_path / "out")
    assert not results[0].was_skipped
    assert engine.get_preset_count >= 1  # đã lấy preset mặc định
    assert engine.infer_count >= 1       # infer thành công (không còn lỗi voice)


def test_voice_data_resolved_once_for_all_events(monkeypatch, tmp_path) -> None:
    # 5 câu preset -> chỉ get_preset_voice MỘT lần (cache), không lặp mỗi câu.
    adapter = VieNeuTtsAdapter()
    engine = _CountingEngine()
    _prime(adapter, engine, monkeypatch)

    events = [_Event(i * 2.0, i * 2.0 + 1.8, f"Câu {i}") for i in range(5)]
    request = TTSRequest(events=events, speaker="voice_b", normalize=False)
    adapter.generate(request, tmp_path / "out")
    assert engine.get_preset_count == 1  # CHỈ một lần dù 5 câu
    assert engine.infer_count == 5


def test_cloning_encodes_reference_once(monkeypatch, tmp_path) -> None:
    adapter = VieNeuTtsAdapter()
    engine = _CountingEngine()
    _prime(adapter, engine, monkeypatch)

    ref = tmp_path / "voice.wav"
    ref.write_bytes(b"fake")
    events = [_Event(i * 2.0, i * 2.0 + 1.8, f"Câu {i}") for i in range(4)]
    request = TTSRequest(events=events, ref_audio_path=str(ref), normalize=False)
    adapter.generate(request, tmp_path / "out")
    assert engine.encode_count == 1  # encode ref CHỈ một lần cho 4 câu
    assert engine.infer_count == 4


def test_no_preset_no_ref_raises(monkeypatch, tmp_path) -> None:
    # Engine không có preset nào và không ref -> báo lỗi rõ ràng, không im lặng.
    class _NoPresetEngine(_CountingEngine):
        def list_preset_voices(self):
            return []

    adapter = VieNeuTtsAdapter()
    _prime(adapter, _NoPresetEngine(), monkeypatch)
    request = TTSRequest(events=[_Event(0.0, 2.0, "x")], speaker="", normalize=False)
    with pytest.raises(TTSGenerationError):
        adapter.generate(request, tmp_path / "out")


# ── Cache engine cấp module (chống nạp 2 lần) ────────────────────────────


def test_engine_cache_shared_between_instances(monkeypatch) -> None:
    # Hai adapter cùng (mode, emotion) phải DÙNG CHUNG engine đã nạp (không nạp lại).
    sentinel = _CountingEngine()
    monkeypatch.setattr(
        mod, "Vieneu", lambda mode, emotion: sentinel, raising=False
    )
    # Giả lập import thành công bằng cách bơm thẳng vào cache khoá chuẩn hoá.
    adapter1 = VieNeuTtsAdapter(mode="standard", emotion="natural")
    mod._ENGINE_CACHE[("standard", "natural")] = sentinel
    adapter2 = VieNeuTtsAdapter(mode="standard", emotion="natural")

    engine1 = adapter1._get_or_load_engine("auto")
    engine2 = adapter2._get_or_load_engine("auto")
    assert engine1 is engine2 is sentinel


def test_engine_cache_separate_per_mode(monkeypatch) -> None:
    std_engine = _CountingEngine()
    turbo_engine = _CountingEngine()
    mod._ENGINE_CACHE[("standard", "natural")] = std_engine
    mod._ENGINE_CACHE[("turbo", "natural")] = turbo_engine

    adapter_std = VieNeuTtsAdapter(mode="standard", emotion="natural")
    adapter_turbo = VieNeuTtsAdapter(mode="turbo", emotion="natural")
    assert adapter_std._get_or_load_engine("auto") is std_engine
    assert adapter_turbo._get_or_load_engine("auto") is turbo_engine
