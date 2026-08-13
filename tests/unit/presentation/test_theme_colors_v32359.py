"""Test [v3.23.59] token màu thích ứng theme (sáng/tối)."""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_colors_change_with_theme(app) -> None:
    from subtitles_extractor.presentation.fluent_compat import Theme, setTheme
    from subtitles_extractor.presentation.theme import colors

    setTheme(Theme.LIGHT)
    light_surface = colors.surface()
    light_text = colors.on_surface()

    setTheme(Theme.DARK)
    dark_surface = colors.surface()
    dark_text = colors.on_surface()

    # Nền & chữ phải khác nhau giữa hai theme (thích ứng).
    assert light_surface != dark_surface
    assert light_text != dark_text


def test_all_tokens_return_hex(app) -> None:
    from subtitles_extractor.presentation.theme import colors
    fns = [
        colors.accent, colors.on_accent, colors.surface, colors.surface_variant,
        colors.on_surface, colors.on_surface_muted, colors.border, colors.mono_bg,
        colors.mono_fg, colors.success, colors.warning, colors.danger,
        colors.danger_bg, colors.info,
    ]
    for fn in fns:
        value = fn()
        assert isinstance(value, str)
        assert value.startswith("#") and len(value) == 7


def test_semantic_colors_distinct(app) -> None:
    from subtitles_extractor.presentation.theme import colors
    assert colors.success() != colors.danger()
    assert colors.warning() != colors.info()
