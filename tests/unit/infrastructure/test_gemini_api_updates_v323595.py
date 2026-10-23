"""[v3.23.595] Theo Gemini API 9/2026: TTS 3.8 GA, bỏ tham số lấy mẫu, Gemini ≥ 3.

Nguồn: ghi chú phát hành Gemini API (21/7, 2/9, 18/9, 22/9/2026) và trang
"Text-to-speech generation". Các điểm kiểm:

* Model TTS 3.8 đọc ``text`` NGUYÊN VĂN → lời dẫn bọc câu của model cũ bị đọc thành tiếng.
* Model TTS 3.8 trả WAV có header 44 byte → giải mã như PCM thô ra tiếng "tách" biên độ
  ~0.57 ở đầu mỗi câu.
* Giọng ở ``voice_config.voice``, không còn ``language_code``.
* Model ≥ 3.6 (Flash-Lite ≥ 3.5) bỏ qua ``temperature``.
* ``gemini-4-*`` / ``*-latest`` phải dùng ``thinking_level`` như Gemini 3.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from subtitles_extractor.domain.ports.subtitle_tts_port import TTSRequest
from subtitles_extractor.infrastructure.gemini_model_family import (
    gemini_model_version,
    sampling_params_deprecated,
    uses_thinking_level,
)
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
)
from subtitles_extractor.infrastructure.tts import gemini_tts_adapter
from subtitles_extractor.infrastructure.tts.gemini_tts_adapter import GeminiTTSAdapter
from subtitles_extractor.infrastructure.tts.gemini_tts_schema import (
    GEMINI_TTS_SAMPLE_RATE,
    decode_tts_audio,
    uses_verbatim_tts_schema,
)
from subtitles_extractor.infrastructure.tts.text_prep import wrap_transcript_for_tts

_SR = GEMINI_TTS_SAMPLE_RATE
_LINE = "Lại có nhà chết sạch nữa à?"


def _sine(seconds: float, rate: int = _SR, amplitude: float = 0.2) -> np.ndarray:
    samples = np.arange(int(rate * seconds)) / rate
    return (amplitude * np.sin(2 * np.pi * 220 * samples)).astype(np.float32)


def _wav_bytes(audio: np.ndarray, rate: int = _SR) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, audio, rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def _pcm_bytes(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()


# ── Nhận diện thế hệ model ────────────────────────────────────────────────────


class TestModelFamily:
    @pytest.mark.parametrize(("name", "version"), [
        ("gemini-3.8-flash-lite-tts", (3, 8)),
        ("gemini-3.8-flash", (3, 8)),
        ("gemini-3.1-flash-tts-preview", (3, 1)),
        ("gemini-2.5-flash-preview-tts", (2, 5)),
        ("gemini-live-2.5-flash-native-audio", (2, 5)),
        ("gemini-3-pro", (3, 0)),
        ("gemini-4-flash", (4, 0)),
        ("models/gemini-3.6-flash", (3, 6)),
        ("gemini-flash-latest", None),
        ("", None),
    ])
    def test_version(self, name: str, version: tuple[int, int] | None) -> None:
        assert gemini_model_version(name) == version

    @pytest.mark.parametrize(("name", "deprecated"), [
        ("gemini-3.8-flash", True),
        ("gemini-3.7-flash", True),
        ("gemini-3.6-flash", True),
        ("gemini-3.5-flash-lite", True),
        ("gemini-3.8-flash-lite-tts", True),
        ("gemini-4-flash", True),
        ("gemini-3.5-flash", False),
        ("gemini-3.1-flash-lite", False),
        ("gemini-3.1-flash-tts-preview", False),
        ("gemini-2.5-flash", False),
        ("gemini-flash-latest", False),
    ])
    def test_sampling_params(self, name: str, deprecated: bool) -> None:
        assert sampling_params_deprecated(name) is deprecated

    @pytest.mark.parametrize(("name", "verbatim"), [
        ("gemini-3.8-flash-tts", True),
        ("gemini-3.8-flash-lite-tts", True),
        ("gemini-4-flash-tts", True),
        ("gemini-3.1-flash-tts-preview", False),
        ("gemini-2.5-flash-preview-tts", False),
        ("gemini-3.8-flash", False),  # không phải model TTS
    ])
    def test_verbatim_schema(self, name: str, verbatim: bool) -> None:
        assert uses_verbatim_tts_schema(name) is verbatim

    @pytest.mark.parametrize(("name", "level"), [
        ("gemini-3.8-flash", True),
        ("gemini-4-flash", True),
        ("gemini-flash-latest", True),
        ("gemini-pro-latest", True),
        ("gemini-2.5-flash", False),
        ("gemini-live-2.5-flash-native-audio", False),
        ("gemma-3-27b-it", False),
        ("", False),
    ])
    def test_thinking_level_family(self, name: str, level: bool) -> None:
        assert GeminiSubtitleTranslator._is_gemini_3_family(name) is level

    def test_pure_helper_matches_versions(self) -> None:
        assert uses_thinking_level("gemini-3.8-flash")
        assert not uses_thinking_level("gemini-flash-latest")  # bí danh xử lý ở adapter


class _FakeThinkingConfig:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _FakeConfig:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


def test_gemini_4_gets_thinking_level_not_budget() -> None:
    translator = GeminiSubtitleTranslator.__new__(GeminiSubtitleTranslator)
    translator._types_module = SimpleNamespace(
        ThinkingConfig=_FakeThinkingConfig,
        GenerateContentConfig=_FakeConfig,
        AutomaticFunctionCallingConfig=_FakeThinkingConfig,
    )
    config = translator._build_config(
        0.2, {}, "sys", enable_thinking=True, thinking_budget=2048,
        model_name="gemini-4-flash", thinking_level="medium",
    )
    assert config.kwargs["thinking_config"].kwargs == {"thinking_level": "medium"}
    assert "temperature" not in config.kwargs


# ── Giải mã âm thanh ──────────────────────────────────────────────────────────


class TestDecodeAudio:
    def test_wav_header_is_not_decoded_as_sound(self) -> None:
        original = _sine(0.5)
        raw = _wav_bytes(original)
        # Cách cũ: coi cả file là PCM thô → 22 mẫu đầu là header "RIFF....WAVEfmt ".
        old_way = np.frombuffer(raw[: len(raw) // 2 * 2], "<i2") / 32768.0
        assert float(np.abs(old_way[:22]).max()) > 0.5  # tiếng "tách" ~0.57
        decoded = decode_tts_audio(raw)
        assert decoded.dtype == np.float32
        assert len(decoded) == len(original)
        assert float(np.abs(decoded - original).max()) < 1.0 / 16384
        assert float(np.abs(decoded[:22]).max()) <= 0.2 + 1e-3

    def test_raw_pcm_unchanged(self) -> None:
        original = _sine(0.25)
        raw = _pcm_bytes(original) + b"\x01"  # lẻ 1 byte: bỏ byte thừa
        decoded = decode_tts_audio(raw)
        assert len(decoded) == len(original)
        assert float(np.abs(decoded - original).max()) < 1.0 / 16384

    def test_wav_other_rate_is_resampled(self) -> None:
        decoded = decode_tts_audio(_wav_bytes(_sine(0.5, rate=48_000), rate=48_000))
        assert abs(len(decoded) - int(0.5 * _SR)) <= 2

    def test_stereo_wav_is_mixed_to_mono(self) -> None:
        left = _sine(0.2)
        stereo = np.stack([left, left], axis=1)
        decoded = decode_tts_audio(_wav_bytes(stereo))
        assert decoded.ndim == 1
        assert float(np.abs(decoded - left).max()) < 1.0 / 16384

    def test_broken_wav_header_falls_back_to_pcm_body(self) -> None:
        body = _pcm_bytes(_sine(0.1))
        raw = b"RIFF" + b"\x00" * 4 + b"WAVE" + b"\xff" * 32 + body
        decoded = decode_tts_audio(raw)
        assert len(decoded) == len(body) // 2

    def test_empty(self) -> None:
        assert decode_tts_audio(b"").size == 0


# ── Yêu cầu gửi lên API ───────────────────────────────────────────────────────


@dataclass
class _Event:
    start_sec: float
    end_sec: float
    text: str


class _FakeModels:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.calls: list[dict[str, Any]] = []

    def generate_content(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        blob = SimpleNamespace(data=self.payload, mime_type="audio/wav")
        part = SimpleNamespace(inline_data=blob)
        return SimpleNamespace(
            candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]))]
        )


def _run_standard(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, model: str, payload: bytes
) -> tuple[dict[str, Any], np.ndarray]:
    adapter = GeminiTTSAdapter("k")
    models = _FakeModels(payload)
    monkeypatch.setattr(adapter, "is_available", lambda: True)
    monkeypatch.setattr(
        adapter, "_make_client", lambda **_kw: SimpleNamespace(models=models)
    )
    request = TTSRequest(
        events=[_Event(1.0, 3.0, _LINE)], language=model, speaker="Kore",
        normalize=False, retry_count=1, gemini_temperature=0.4,
    )
    output = tmp_path / "out.wav"
    adapter.generate(request, output)
    assert len(models.calls) == 1
    audio, _rate = sf.read(output, dtype="float32", always_2d=False)
    return models.calls[0], np.asarray(audio)


class TestStandardRequest:
    def test_default_model_is_38_flash_lite_tts(self) -> None:
        assert gemini_tts_adapter._STANDARD_TTS_MODELS[0] == "gemini-3.8-flash-lite-tts"
        assert "gemini-3.8-flash-tts" in gemini_tts_adapter.GEMINI_TTS_MODELS

    def test_38_reads_line_verbatim_with_voice_field(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        call, audio = _run_standard(
            monkeypatch, tmp_path, "gemini-3.8-flash-lite-tts", _wav_bytes(_sine(1.2))
        )
        assert call["contents"] == _LINE
        speech = call["config"].speech_config
        assert speech.voice_config.voice == "Kore"
        assert speech.language_code is None
        assert call["config"].temperature is None
        assert audio.size > 0
        assert float(np.abs(audio).max()) < 0.5  # không lọt header thành tiếng "tách"

    def test_old_model_keeps_previous_schema(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        call, audio = _run_standard(
            monkeypatch, tmp_path, "gemini-3.1-flash-tts-preview", _pcm_bytes(_sine(1.2))
        )
        assert call["contents"] == wrap_transcript_for_tts(_LINE)
        speech = call["config"].speech_config
        assert speech.language_code == "vi-VN"
        assert speech.voice_config.prebuilt_voice_config.voice_name == "Kore"
        assert call["config"].temperature == pytest.approx(0.4)
        assert audio.size > 0
