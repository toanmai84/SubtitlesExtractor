"""[v3.23.75] Test builder style tập trung ``caption_style`` và token cỡ chữ.

Mục tiêu cốt lõi: CHỨNG MINH việc gom style không đổi giao diện — ``caption_style`` phải
sinh ra ĐÚNG chuỗi mà lời gọi inline cũ tạo ra (``"font-size:11px;color:<màu>;"``), và
hằng số ``FONT_SIZE_CAPTION`` giữ nguyên 11.
"""

from __future__ import annotations

import pytest

from subtitles_extractor.presentation.theme import metrics as _m
from subtitles_extractor.presentation.theme import styles as _styles


def test_font_size_caption_value_unchanged() -> None:
    """Giữ nguyên 11px → không đổi giao diện so với bản hardcode trước đây."""
    assert _m.FONT_SIZE_CAPTION == 11


def test_caption_style_with_explicit_color() -> None:
    """Màu tường minh → chuỗi khớp chính xác định dạng cũ."""
    assert _styles.caption_style("#FF0000") == "font-size:11px;color:#FF0000;"


def test_caption_style_default_uses_muted_color(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mặc định dùng màu chữ mờ; định dạng trùng khít chuỗi inline cũ."""
    monkeypatch.setattr(_styles._c, "on_surface_muted", lambda: "#6B6B6B")
    assert _styles.caption_style() == "font-size:11px;color:#6B6B6B;"
    # Tương đương đúng chuỗi inline cũ từng dùng khắp các trang.
    expected = f"font-size:11px;color:{_styles._c.on_surface_muted()};"
    assert _styles.caption_style() == expected


def test_caption_style_tracks_font_size_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nếu đổi token cỡ chữ, chuỗi tự cập nhật (chỉnh một nơi)."""
    monkeypatch.setattr(_m, "FONT_SIZE_CAPTION", 14)
    assert _styles.caption_style("#000000") == "font-size:14px;color:#000000;"


def test_mono_label_style_default() -> None:
    """Mặc định dùng FONT_SIZE_SMALL (13px) → khớp chuỗi inline cũ."""
    assert _m.FONT_SIZE_SMALL == 13
    assert (
        _styles.mono_label_style()
        == "font-family: Consolas, monospace; font-size: 13px;"
    )


def test_mono_label_style_custom_size() -> None:
    """Cho phép truyền cỡ chữ tuỳ chọn."""
    assert (
        _styles.mono_label_style(11)
        == "font-family: Consolas, monospace; font-size: 11px;"
    )
