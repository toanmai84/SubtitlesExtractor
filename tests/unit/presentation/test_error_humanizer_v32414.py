"""[v3.23.114] Test hàm thuần diễn giải lỗi Gemini -> hướng dẫn tiếng Việt."""

from __future__ import annotations

from subtitles_extractor.presentation.utils.error_humanizer import humanize_gemini_error


def test_api_key_error_gives_guidance() -> None:
    out = humanize_gemini_error("400 API key not valid. Please pass a valid API key.")
    assert "API Key" in out
    assert "aistudio.google.com/apikey" in out
    assert "Chi tiết kỹ thuật" in out  # vẫn giữ lỗi gốc


def test_quota_error() -> None:
    out = humanize_gemini_error("429 RESOURCE_EXHAUSTED: Quota exceeded")
    assert "hạn mức" in out.lower()


def test_unavailable_error_mentions_checkpoint() -> None:
    out = humanize_gemini_error("503 UNAVAILABLE: The model is overloaded.")
    assert "tạm thời" in out
    assert "checkpoint" in out


def test_network_error() -> None:
    out = humanize_gemini_error("Max retries exceeded: getaddrinfo failed")
    assert "kết nối mạng" in out.lower()


def test_safety_block() -> None:
    out = humanize_gemini_error("Response blocked due to SAFETY (block_reason: SAFETY)")
    assert "bộ lọc an toàn" in out


def test_unknown_error_passthrough() -> None:
    # Không khớp mẫu -> giữ nguyên, không bịa.
    raw = "Một lỗi lạ không xác định XYZ"
    assert humanize_gemini_error(raw) == raw


def test_empty_passthrough() -> None:
    assert humanize_gemini_error("") == ""


def test_key_pattern_takes_priority_over_generic() -> None:
    # Chuỗi chứa cả "403" (key) và "unavailable" -> ưu tiên hướng dẫn về Key.
    out = humanize_gemini_error("403 PERMISSION_DENIED: service unavailable for key")
    assert "API Key" in out
