"""[v3.23.478] Rút gọn lỗi Gemini khi log (429 JSON ~1000 ký tự → ~40 ký tự)."""
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator as G,
)


def test_429_input_token_quota():
    e = Exception("429 RESOURCE_EXHAUSTED. {'msg': 'generate_content_free_tier_"
                  "input_token_count limit 250000'}")
    r = G._concise_error(e)
    assert r == "429 RESOURCE_EXHAUSTED (quota token/phút free-tier)"


def test_504_stripped_to_head():
    assert G._concise_error(Exception("504 DEADLINE_EXCEEDED. {'error':{...}}")) == \
        "504 DEADLINE_EXCEEDED"


def test_503_stripped():
    assert G._concise_error(Exception("503 UNAVAILABLE. {json}")) == "503 UNAVAILABLE"


def test_short_error_kept():
    assert G._concise_error(Exception("Ngắn gọn")) == "Ngắn gọn"


def test_no_code_long_truncated():
    long = "x" * 200
    out = G._concise_error(Exception(long))
    assert len(out) <= 80 and out.endswith("…")


# [v3.23.480] Lỗi DNS/mất mạng (getaddrinfo) là TẠM THỜI → phải retryable (trước bị coi vĩnh viễn).
def test_getaddrinfo_is_retryable():
    from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
        GeminiSubtitleTranslator as G,
    )
    for txt in [
        "[Errno 11001] getaddrinfo failed",
        "[Errno -2] Name or service not known",
        "Temporary failure in name resolution",
        "Network is unreachable",
        "No route to host",
    ]:
        assert G._is_retryable(Exception(txt)) is True, txt


def test_permanent_errors_still_not_retryable():
    from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
        GeminiSubtitleTranslator as G,
    )
    assert G._is_retryable(Exception("400 INVALID_ARGUMENT: bad fps")) is False
    assert G._is_retryable(Exception("PERMISSION_DENIED")) is False


def test_network_error_friendly_message():
    from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
        GeminiSubtitleTranslator as G,
    )
    assert "MẤT MẠNG" in G._concise_error(Exception("[Errno 11001] getaddrinfo failed"))


# [v3.23.487] 403 loại "FILE" (file thuộc key cũ / hết hạn) là PHỤC HỒI ĐƯỢC — lần thử lại sẽ
# tải lại đoạn video. Trước bị xếp vĩnh viễn → bỏ luôn lượt thử lại (app_log: mất đoạn 21).
def test_403_file_error_is_retryable():
    from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
        GeminiSubtitleTranslator as G,
    )
    exc = Exception(
        "403 PERMISSION_DENIED. {'error': {'code': 403, 'message': 'You do not have "
        "permission to access the File qtq6iak0xbmr or it may not exist.'}}"
    )
    assert G._is_retryable(exc) is True


def test_403_invalid_key_still_permanent():
    from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
        GeminiSubtitleTranslator as G,
    )
    assert G._is_retryable(Exception("403 PERMISSION_DENIED: API key not valid")) is False
