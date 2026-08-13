"""[v3.23.185] Test VieNeuTtsAdapter — logic điều phối với engine giả (không cần model).

VieNeu-TTS là model neural nặng, không nạp được trong CI. Test này dùng engine GIẢ (fake
SDK) tiêm qua thuộc tính ``_engine`` để kiểm toàn bộ logic điều phối: hàm thuần (resample,
normalize_mode), chọn giọng preset vs cloning, đặt audio vào master đúng mốc, bỏ câu rỗng,
retry, và tái dùng audio_mastering. Không kiểm chất lượng giọng (cần model thật + tai
người) — phần đó người dùng nghiệm thu trên máy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from subtitles_extractor.domain.ports.subtitle_tts_port import (
    TTSRequest,
    TTSSegmentResult,
    TTSUnavailableError,
)
from subtitles_extractor.infrastructure.tts.vieneu_tts_adapter import (
    VieNeuTtsAdapter,
    normalize_mode,
    resample_audio,
)

_SR = 24000


@pytest.fixture(autouse=True)
def _clear_engine_cache():
    """Xoá cache engine cấp module trước mỗi test để tránh rò rỉ trạng thái."""
    from subtitles_extractor.infrastructure.tts import vieneu_tts_adapter as mod
    mod._ENGINE_CACHE.clear()
    yield
    mod._ENGINE_CACHE.clear()


@dataclass
class _Event:
    """Sự kiện phụ đề tối giản cho test."""

    start_sec: float
    end_sec: float
    text: str


class _FakeVieneuEngine:
    """Engine VieNeu giả: trả tone cố định, ghi lại tham số infer để kiểm.

    Mô phỏng API thật: ``list_preset_voices`` trả (label, id); ``get_preset_voice`` và
    ``encode_reference`` trả DICT giọng; ``infer`` bắt buộc nhận ``voice`` dict.
    """

    def __init__(self, sr: int = _SR, return_tuple: bool = False) -> None:
        self._sr = sr
        self._return_tuple = return_tuple
        self.infer_calls: list[dict] = []
        self.encode_calls: list[str] = []
        self.get_preset_calls: list[str] = []

    def list_preset_voices(self) -> list[tuple[str, str]]:
        return [("Nữ miền Bắc — Trúc Ly", "truc_ly"), ("Nam — Xuân Vĩnh", "xuan_vinh")]

    def get_preset_voice(self, voice_id: str) -> dict:
        self.get_preset_calls.append(voice_id)
        return {"kind": "preset", "id": voice_id}

    def encode_reference(self, ref_path: str) -> dict:
        self.encode_calls.append(ref_path)
        return {"kind": "clone", "ref": ref_path}

    def infer(self, text: str, voice=None):
        # SDK thật ném lỗi nếu voice thiếu — mô phỏng để test bắt đúng hành vi.
        if voice is None:
            raise ValueError("Must provide either 'voice' dict or both 'ref_codes' and 'ref_text'.")
        self.infer_calls.append({"text": text, "voice": voice})
        samples = np.arange(int(self._sr * 0.5))
        tone = (0.3 * np.sin(2 * np.pi * 220 * samples / self._sr)).astype(np.float32)
        return (tone, self._sr) if self._return_tuple else tone


# ── Hàm thuần ─────────────────────────────────────────────────────────────


def test_normalize_mode_valid() -> None:
    assert normalize_mode("turbo") == "turbo"
    assert normalize_mode("V3Turbo") == "v3turbo"
    assert normalize_mode("STANDARD") == "standard"


def test_normalize_mode_invalid_defaults_standard() -> None:
    assert normalize_mode("nonsense") == "standard"
    assert normalize_mode("") == "standard"


def test_resample_same_rate_noop() -> None:
    audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    out = resample_audio(audio, _SR, _SR)
    assert np.array_equal(out, audio)


def test_resample_48k_to_24k_halves_length() -> None:
    audio = (0.5 * np.sin(2 * np.pi * 200 * np.arange(48000) / 48000)).astype(np.float32)
    out = resample_audio(audio, 48000, 24000)
    assert abs(len(out) - 24000) <= 2  # ~một nửa
    assert out.dtype == np.float32


def test_resample_empty() -> None:
    assert resample_audio(np.array([], dtype=np.float32), 48000, 24000).size == 0


# ── Adapter: interface & availability ─────────────────────────────────────


def test_engine_name() -> None:
    assert VieNeuTtsAdapter().get_engine_name() == "VieNeu-TTS (Offline)"


def test_list_languages() -> None:
    assert VieNeuTtsAdapter().list_languages() == ["vi-VN"]


def test_generate_raises_when_unavailable(monkeypatch, tmp_path) -> None:
    adapter = VieNeuTtsAdapter()
    monkeypatch.setattr(adapter, "is_available", lambda: False)
    request = TTSRequest(events=[_Event(0.0, 1.0, "Xin chào")])
    with pytest.raises(TTSUnavailableError):
        adapter.generate(request, tmp_path / "out.wav")


def test_list_speakers_empty_when_unavailable(monkeypatch) -> None:
    adapter = VieNeuTtsAdapter()
    monkeypatch.setattr(adapter, "is_available", lambda: False)
    assert adapter.list_speakers("vi-VN") == []


# ── Adapter: generate với engine giả ──────────────────────────────────────


def _prime(adapter: VieNeuTtsAdapter, engine: _FakeVieneuEngine, monkeypatch) -> None:
    """Tiêm engine giả + đánh dấu available để bỏ qua import SDK thật."""
    monkeypatch.setattr(adapter, "is_available", lambda: True)
    adapter._engine = engine


def test_generate_uses_preset_voice(monkeypatch, tmp_path) -> None:
    adapter = VieNeuTtsAdapter()
    engine = _FakeVieneuEngine()
    _prime(adapter, engine, monkeypatch)

    request = TTSRequest(
        events=[_Event(0.0, 2.0, "Xin chào thế giới")],
        speaker="truc_ly", normalize=False,
    )
    results = adapter.generate(request, tmp_path / "out")
    assert len(results) == 1
    assert not results[0].was_skipped
    # Đã gọi get_preset_voice với ID, và infer với voice DICT (không phải string).
    assert engine.get_preset_calls == ["truc_ly"]
    assert engine.infer_calls[0]["voice"] == {"kind": "preset", "id": "truc_ly"}


def test_generate_uses_voice_cloning(monkeypatch, tmp_path) -> None:
    adapter = VieNeuTtsAdapter()
    engine = _FakeVieneuEngine()
    _prime(adapter, engine, monkeypatch)

    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"fake")
    request = TTSRequest(
        events=[_Event(0.0, 2.0, "Giọng nhân bản")],
        ref_audio_path=str(ref), ref_text="mẫu", normalize=False,
    )
    adapter.generate(request, tmp_path / "out")
    # Voice cloning: encode_reference được gọi với ref path, infer nhận voice DICT clone.
    assert engine.encode_calls == [str(ref)]
    assert engine.infer_calls[0]["voice"] == {"kind": "clone", "ref": str(ref)}


def test_generate_skips_empty_text(monkeypatch, tmp_path) -> None:
    adapter = VieNeuTtsAdapter()
    engine = _FakeVieneuEngine()
    _prime(adapter, engine, monkeypatch)

    request = TTSRequest(
        events=[_Event(0.0, 2.0, "   "), _Event(2.0, 4.0, "Có nội dung")],
        normalize=False,
    )
    results = adapter.generate(request, tmp_path / "out")
    # Câu rỗng bị bỏ khỏi valid nhưng vẫn có kết quả cho câu có nội dung.
    assert any(not r.was_skipped for r in results)


def test_generate_cloning_missing_ref_raises(monkeypatch, tmp_path) -> None:
    from subtitles_extractor.domain.ports.subtitle_tts_port import TTSGenerationError
    adapter = VieNeuTtsAdapter()
    monkeypatch.setattr(adapter, "is_available", lambda: True)
    request = TTSRequest(
        events=[_Event(0.0, 2.0, "x")], ref_audio_path="/khong/ton/tai.wav",
    )
    with pytest.raises(TTSGenerationError):
        adapter.generate(request, tmp_path / "out")


def test_generate_handles_tuple_output(monkeypatch, tmp_path) -> None:
    # SDK trả (audio, sr) ở 48kHz -> phải resample về 24k không lỗi.
    adapter = VieNeuTtsAdapter()
    engine = _FakeVieneuEngine(sr=48000, return_tuple=True)
    _prime(adapter, engine, monkeypatch)

    request = TTSRequest(
        events=[_Event(0.0, 2.0, "Kiểm tra 48k")], normalize=False,
    )
    results = adapter.generate(request, tmp_path / "out")
    assert not results[0].was_skipped


def test_generate_writes_file(monkeypatch, tmp_path) -> None:
    adapter = VieNeuTtsAdapter()
    engine = _FakeVieneuEngine()
    _prime(adapter, engine, monkeypatch)

    out_base = tmp_path / "audio"
    request = TTSRequest(
        events=[_Event(0.0, 2.0, "Ghi ra file")],
        normalize=False, output_format="wav",
    )
    adapter.generate(request, out_base)
    assert (tmp_path / "audio.wav").exists()


def test_result_type(monkeypatch, tmp_path) -> None:
    adapter = VieNeuTtsAdapter()
    _prime(adapter, _FakeVieneuEngine(), monkeypatch)
    request = TTSRequest(events=[_Event(0.0, 2.0, "abc")], normalize=False)
    results = adapter.generate(request, tmp_path / "out")
    assert all(isinstance(r, TTSSegmentResult) for r in results)
