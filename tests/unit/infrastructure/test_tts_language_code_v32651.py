"""[v3.23.251] Cố định ngôn ngữ đầu ra TTS (theo tài liệu Live API best-practices).

**Cải tiến chất lượng có căn cứ tài liệu.** Google Live API best-practices khuyến nghị:
*"Explicitly setting the language and voice code in your configuration is recommended
to maintain consistency; without this definition, Gemini might alter the conversation
language depending on the provided context."* Và thêm chỉ thị ngôn ngữ vào system
instruction.

Code cũ KHÔNG set ``language_code`` trong ``SpeechConfig`` -> model có thể đổi ngôn
ngữ theo ngữ cảnh (rủi ro thật khi bản dịch còn sót tên riêng CJK). App đọc tiếng Việt.

**Sửa — hai lớp bảo vệ:**
1. ``SpeechConfig(language_code="vi-VN")`` cho cả native audio + standard TTS (bọc an toàn
   cho SDK cũ không nhận field).
2. Thêm "Luôn đọc bằng tiếng Việt." vào system instruction — CHỈ khi dùng style MẶC ĐỊNH.
   Nếu người dùng tự đặt style (có thể cho ngôn ngữ khác), tôn trọng, không ép tiếng Việt.
"""

from __future__ import annotations

import pathlib

from subtitles_extractor.infrastructure.tts.gemini_tts_adapter import _TTS_LANGUAGE_CODE

_GEMINI_SRC = pathlib.Path(
    "src/subtitles_extractor/infrastructure/tts/gemini_tts_adapter.py"
).read_text(encoding="utf-8")


def test_hằng_số_ngôn_ngữ() -> None:
    assert _TTS_LANGUAGE_CODE == "vi-VN"


def test_native_audio_set_language_code() -> None:
    assert "language_code=_TTS_LANGUAGE_CODE" in _GEMINI_SRC


def test_cả_hai_đường_set_language_code() -> None:
    # native audio + standard TTS đều đặt language_code.
    assert _GEMINI_SRC.count("language_code=_TTS_LANGUAGE_CODE") == 2


def test_bọc_an_toàn_sdk_cũ() -> None:
    # Phải có fallback khi SDK cũ không nhận language_code.
    assert _GEMINI_SRC.count("except (TypeError, ValueError):") >= 2


def test_chỉ_thị_ngôn_ngữ_trong_system_instruction() -> None:
    assert "Luôn đọc bằng tiếng Việt." in _GEMINI_SRC


def test_chỉ_thị_ngôn_ngữ_chỉ_cho_style_mặc_định() -> None:
    # Chỉ thị "Luôn đọc bằng tiếng Việt" phải nằm trong nhánh style mặc định
    # (if not style), KHÔNG áp cho style người dùng tự đặt.
    idx = _GEMINI_SRC.find("if not style:")
    assert idx != -1
    # "Luôn đọc bằng tiếng Việt" phải xuất hiện SAU "if not style:" và gần đó.
    doan = _GEMINI_SRC[idx : idx + 800]
    assert "Luôn đọc bằng tiếng Việt." in doan
