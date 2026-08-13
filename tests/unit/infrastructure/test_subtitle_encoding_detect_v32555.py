"""[v3.23.155] Test dò MÃ HÓA phụ đề — đọc đúng mọi bảng mã phổ biến.

Trước đây importer đọc cứng utf-8-sig + errors=replace -> file GB18030/Big5/UTF-16/
Shift-JIS/CP1252 bị thay \ufffd hàng loạt (hỏng âm thầm). Nay dò 3 lớp: BOM -> UTF-8
strict -> chấm điểm ứng viên.
"""

# ruff: noqa: RUF001 — dấu câu fullwidth CJK là CHỦ ĐÍCH (dò mã hóa)
from __future__ import annotations

from subtitles_extractor.infrastructure.subtitle.encoding_detect import (
    decode_subtitle_bytes,
)

_VI = "Xin chào, đây là phụ đề tiếng Việt có dấu đầy đủ."
_ZH = "你好，这是一段中文字幕，用来测试编码识别是否正确。"


def test_utf8_plain() -> None:
    text, enc = decode_subtitle_bytes(_VI.encode("utf-8"))
    assert text == _VI and enc == "utf-8"


def test_utf8_bom() -> None:
    text, enc = decode_subtitle_bytes(b"\xef\xbb\xbf" + _VI.encode("utf-8"))
    assert text == _VI and enc == "utf-8-sig"


def test_utf16_le_bom() -> None:
    text, enc = decode_subtitle_bytes(_ZH.encode("utf-16"))  # utf-16 tự thêm BOM LE
    assert text == _ZH and enc.startswith("utf-16")


def test_gb18030_chinese() -> None:
    text, enc = decode_subtitle_bytes(_ZH.encode("gb18030"))
    assert text == _ZH
    assert enc == "gb18030"


def test_big5_traditional() -> None:
    trad = "妳好，這是繁體中文字幕，測試編碼偵測。"
    text, _enc = decode_subtitle_bytes(trad.encode("big5"))
    assert text == trad  # nội dung phải NGUYÊN VẸN (điểm hợp lý chọn đúng bộ)


def test_cp1252_western() -> None:
    western = 'He said: "café, naïve, résumé" — that\u2019s all.'
    text, _enc = decode_subtitle_bytes(western.encode("cp1252"))
    assert "café" in text and "naïve" in text  # không còn \ufffd


def test_no_replacement_chars_for_shift_jis() -> None:
    jp = "こんにちは、これは日本語の字幕です。エンコード検出のテスト。"
    text, _enc = decode_subtitle_bytes(jp.encode("shift_jis"))
    assert text == jp
