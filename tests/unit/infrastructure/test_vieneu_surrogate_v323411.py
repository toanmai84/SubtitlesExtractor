"""[v3.23.411] Test làm sạch surrogate trước khi gửi payload qua stdin (UTF-8)."""

from __future__ import annotations

import json

from subtitles_extractor.infrastructure.tts.vieneu_tts_adapter import (
    _json_line_safe,
    _strip_surrogates,
)


def test_strip_removes_lone_surrogate() -> None:
    dirty = "Tức khắc đăng cơ\udc90 rồi\udc9d"
    clean = _strip_surrogates(dirty)
    clean.encode("utf-8")  # không được ném lỗi
    assert "\udc90" not in clean and "\udc9d" not in clean
    assert clean.startswith("Tức khắc đăng cơ")


def test_clean_string_unchanged() -> None:
    text = "Xin chào, đây là tiếng Việt có dấu!"
    assert _strip_surrogates(text) == text


def test_strip_recurses_dict_and_list() -> None:
    payload = {
        "text": "a\udc81b",
        "nested": {"key": "c\udc8dd"},
        "items": ["ok\udc90", "clean"],
        "num": 42,
    }
    cleaned = _strip_surrogates(payload)
    assert "\udc81" not in cleaned["text"]
    assert "\udc8d" not in cleaned["nested"]["key"]
    assert "\udc90" not in cleaned["items"][0]
    assert cleaned["items"][1] == "clean"
    assert cleaned["num"] == 42  # non-str giữ nguyên


def test_json_line_safe_encodes_utf8() -> None:
    payload = {"text": "Giá!\udc81", "voice_preset": "Minh Đức"}
    line = _json_line_safe(payload)
    assert line.endswith("\n")
    line.encode("utf-8")  # không ném lỗi surrogate
    parsed = json.loads(line)
    assert parsed["voice_preset"] == "Minh Đức"
    assert "\udc81" not in parsed["text"]


def test_json_line_safe_preserves_vietnamese() -> None:
    payload = {"text": "Tức khắc đăng cơ", "voice_preset": "Thái Sơn"}
    parsed = json.loads(_json_line_safe(payload))
    assert parsed["text"] == "Tức khắc đăng cơ"
    assert parsed["voice_preset"] == "Thái Sơn"


def test_preprocess_strips_surrogates_at_source() -> None:
    """[v3.23.413] Làm sạch surrogate NGAY tại preprocess_tts_text (mọi engine)."""
    from subtitles_extractor.infrastructure.tts.text_prep import preprocess_tts_text

    dirty = "Cách kinh thành Lâm An\udc90 chừng\udc9d trăm dặm"
    clean, _ = preprocess_tts_text(dirty, clean_tags=True)
    clean.encode("utf-8")  # không ném lỗi
    assert "\udc90" not in clean and "\udc9d" not in clean
    assert "Cách kinh thành Lâm An" in clean


def test_preprocess_keeps_clean_vietnamese() -> None:
    from subtitles_extractor.infrastructure.tts.text_prep import preprocess_tts_text

    clean, _ = preprocess_tts_text("Xin chào các bạn thân mến", clean_tags=True)
    assert clean == "Xin chào các bạn thân mến"
