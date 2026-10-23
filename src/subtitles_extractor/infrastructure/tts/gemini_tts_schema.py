"""[v3.23.595] Khác biệt giữa các thế hệ model Gemini TTS — hàm thuần, kiểm thử được.

Theo tài liệu Gemini API (ghi chú phát hành 22/9/2026, trang "Text-to-speech generation"):

* ``gemini-3.8-flash-tts`` và ``gemini-3.8-flash-lite-tts`` đã GA; bản Flash-Lite thay cho
  ``gemini-3.1-flash-tts-preview``. Các model Gemini 2.5 bị giới hạn cho người đã dùng gần
  đây (18/9/2026).
* Từ 3.8, ``text`` được đọc NGUYÊN VĂN: chỉ dẫn diễn xuất phải nằm ở ``speech_metadata``
  chứ không viết lẫn vào câu. Lời dẫn "Đọc to đoạn thoại sau…" mà ứng dụng bọc quanh câu
  (v3.23.245, dành cho model cũ) sẽ bị đọc thành tiếng.
* Từ 3.8, gọi không streaming trả về file WAV đầy đủ (có header RIFF, 24 kHz, mono,
  16-bit); model cũ trả PCM thô. Giải mã WAV như PCM thô sẽ biến 44 byte header thành
  tiếng "tách" ở đầu mỗi câu.
* Giọng đặt bằng ``speech_config.voice_config.voice``; ngôn ngữ tự nhận diện, không còn
  trường ``language_code``.
* Từ Gemini 3.6 Flash và 3.5 Flash-Lite, ``temperature``/``top_p``/``top_k`` bị bỏ qua
  (ghi chú phát hành 21/7/2026) và các thế hệ sau có thể từ chối.
"""

from __future__ import annotations

import io
import logging

import numpy as np

from subtitles_extractor.infrastructure.gemini_model_family import (
    gemini_model_version,
    sampling_params_deprecated,
)

logger = logging.getLogger(__name__)

#: Tần số lấy mẫu pipeline Gemini TTS (cả WAV lẫn PCM thô của Gemini đều 24 kHz).
GEMINI_TTS_SAMPLE_RATE = 24_000
#: Thế hệ đầu tiên đọc transcript nguyên văn và trả WAV.
_VERBATIM_SCHEMA_FROM = (3, 8)


def uses_verbatim_tts_schema(model_name: str) -> bool:
    """Model TTS thế hệ 3.8+: transcript nguyên văn, trả WAV, giọng ở ``voice``.

    Args:
        model_name: Tên model TTS.

    Returns:
        ``True`` nếu là model TTS Gemini ≥ 3.8.
    """
    version = gemini_model_version(model_name)
    return (
        version is not None
        and version >= _VERBATIM_SCHEMA_FROM
        and "tts" in (model_name or "").lower()
    )


def decode_tts_audio(
    raw: bytes, target_rate: int = GEMINI_TTS_SAMPLE_RATE
) -> np.ndarray:
    """Giải mã âm thanh Gemini TTS (WAV có header, hoặc PCM 16-bit thô) thành float32.

    Args:
        raw: Bytes trả về ở ``inline_data.data``.
        target_rate: Tần số lấy mẫu của pipeline.

    Returns:
        Mảng float32 mono ở ``target_rate``; rỗng nếu không có dữ liệu.
    """
    if not raw:
        return np.zeros(0, dtype=np.float32)
    if raw[:4] == b"RIFF" and raw[8:12] == b"WAVE":
        return _decode_wav(raw, target_rate)
    usable = len(raw) - (len(raw) % 2)
    return np.frombuffer(raw[:usable], dtype="<i2").astype(np.float32) / 32768.0


def _decode_wav(raw: bytes, target_rate: int) -> np.ndarray:
    """Đọc WAV trong bộ nhớ, trộn mono, đổi tần số lấy mẫu nếu khác pipeline."""
    import soundfile as sf

    from subtitles_extractor.infrastructure.tts.audio_utils import resample_audio

    try:
        samples, sample_rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    except (RuntimeError, ValueError) as exc:
        # Header hỏng: lùi về đọc phần sau header chuẩn 44 byte như PCM thô, còn hơn mất
        # cả câu thoại.
        logger.warning("Không đọc được WAV từ Gemini TTS (%s) — đọc như PCM thô.", exc)
        body = raw[44:]
        usable = len(body) - (len(body) % 2)
        return np.frombuffer(body[:usable], dtype="<i2").astype(np.float32) / 32768.0
    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if int(sample_rate) != target_rate and audio.size:
        audio = resample_audio(audio, int(sample_rate), target_rate)
    return audio.astype(np.float32, copy=False)


__all__ = [
    "GEMINI_TTS_SAMPLE_RATE",
    "decode_tts_audio",
    "gemini_model_version",
    "sampling_params_deprecated",
    "uses_verbatim_tts_schema",
]
