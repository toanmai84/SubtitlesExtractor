"""[v3.23.130] Test: chuỗi NHIỀU API key không làm vỡ tải video/TTS.

Bug: ô 'nhiều key' nối các key bằng '\\n'; chuỗi gộp này lọt vào header HTTP
'x-goog-api-key' → httpx.LocalProtocolError "Illegal header value". Tải video & TTS
chỉ cần MỘT key → phải lấy key đầu tiên.
"""

from __future__ import annotations

from subtitles_extractor.infrastructure.translation.gemini_video_context import (
    GeminiVideoContextProvider,
)
from subtitles_extractor.infrastructure.tts.gemini_tts_adapter import GeminiTTSAdapter

_MULTI = "AIzaKEY_ONE\nAIzaKEY_TWO\nAIzaKEY_THREE"


def test_first_api_key_from_multiline() -> None:
    assert GeminiVideoContextProvider._first_api_key(_MULTI) == "AIzaKEY_ONE"


def test_first_api_key_from_comma() -> None:
    assert GeminiVideoContextProvider._first_api_key("K1, K2, K3") == "K1"


def test_first_api_key_single_and_empty() -> None:
    assert GeminiVideoContextProvider._first_api_key("OnlyKey") == "OnlyKey"
    assert GeminiVideoContextProvider._first_api_key("") == ""
    assert GeminiVideoContextProvider._first_api_key("  spaced  ") == "spaced"


def test_video_provider_stores_single_key() -> None:
    prov = GeminiVideoContextProvider(api_key=_MULTI)
    assert prov._api_key == "AIzaKEY_ONE"
    assert "\n" not in prov._api_key  # không còn xuống dòng → header hợp lệ


def test_tts_adapter_stores_single_key() -> None:
    tts = GeminiTTSAdapter(api_key=_MULTI)
    assert tts._api_key == "AIzaKEY_ONE"
    assert "\n" not in tts._api_key


def test_tts_adapter_handles_blank_lines() -> None:
    tts = GeminiTTSAdapter(api_key="\n\n  AIzaREAL  \n\n")
    assert tts._api_key == "AIzaREAL"
