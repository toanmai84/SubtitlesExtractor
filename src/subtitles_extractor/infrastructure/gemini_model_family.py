"""[v3.23.595] Nhận diện thế hệ model Gemini từ tên — dùng chung cho dịch và TTS.

Ghi chú phát hành Gemini API 21/7/2026: từ Gemini 3.6 Flash và 3.5 Flash-Lite,
``temperature``/``top_p``/``top_k`` bị bỏ qua, các thế hệ sau có thể từ chối. Gemini 3.x
trở đi dùng ``thinking_level`` thay ``thinking_budget``.
"""

from __future__ import annotations

import re

#: Thế hệ đầu tiên bỏ qua tham số lấy mẫu (Flash); Flash-Lite từ 3.5.
_SAMPLING_DEPRECATED_FROM = (3, 6)
_SAMPLING_DEPRECATED_LITE_FROM = (3, 5)
#: Thế hệ đầu tiên dùng ``thinking_level``.
_THINKING_LEVEL_FROM = (3, 0)
_VERSION_RE = re.compile(r"gemini-(?:[a-z]+-)*?(\d+)(?:\.(\d+))?(?![\d.])")


def gemini_model_version(model_name: str) -> tuple[int, int] | None:
    """Phiên bản ``(chính, phụ)`` của một model Gemini.

    Args:
        model_name: Vd ``"gemini-3.8-flash-lite-tts"``,
            ``"gemini-live-2.5-flash-native-audio"``.

    Returns:
        ``(3, 8)``, ``(2, 5)``…; ``None`` nếu tên không chứa phiên bản.
    """
    match = _VERSION_RE.search((model_name or "").lower())
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2) or 0)


def sampling_params_deprecated(model_name: str) -> bool:
    """Model có bỏ qua ``temperature``/``top_p``/``top_k`` không (không nên gửi).

    Args:
        model_name: Tên model.

    Returns:
        ``True`` với Gemini ≥ 3.6 (mọi biến thể, kể cả 4.x) và Flash-Lite ≥ 3.5.
    """
    version = gemini_model_version(model_name)
    if version is None:
        return False
    if "lite" in (model_name or "").lower() and version >= _SAMPLING_DEPRECATED_LITE_FROM:
        return True
    return version >= _SAMPLING_DEPRECATED_FROM


def uses_thinking_level(model_name: str) -> bool:
    """Model dùng ``thinking_level`` (Gemini 3.x trở đi, kể cả thế hệ sau).

    Args:
        model_name: Tên model.

    Returns:
        ``True`` nếu phiên bản ≥ 3.
    """
    version = gemini_model_version(model_name)
    return version is not None and version >= _THINKING_LEVEL_FROM


__all__ = [
    "gemini_model_version",
    "sampling_params_deprecated",
    "uses_thinking_level",
]
