"""[v3.23.71 — Giai đoạn 6] Test tiện ích tương phản WCAG + audit bảng màu theme.

Hai phần:
1. Unit test cho :mod:`...theme.contrast` với các giá trị mốc đã biết (đen/trắng = 21:1…).
2. Audit: mọi cặp (màu chữ, màu nền) THỰC TẾ trong
   :data:`LIGHT_PALETTE`/:data:`DARK_PALETTE`
   đạt ngưỡng WCAG 2.1 mức AA — chống hồi quy nếu sau này ai đó chỉnh token quá nhạt.
"""

from __future__ import annotations

import pytest

from subtitles_extractor.presentation.theme.contrast import (
    AA_LARGE_RATIO,
    AA_NORMAL_RATIO,
    contrast_ratio,
    hex_to_rgb,
    meets_wcag_aa,
    relative_luminance,
)


class TestHexToRgb:
    def test_six_digit(self) -> None:
        assert hex_to_rgb("#ffffff") == (255, 255, 255)
        assert hex_to_rgb("#000000") == (0, 0, 0)
        assert hex_to_rgb("#58a6ff") == (88, 166, 255)

    def test_three_digit_shorthand(self) -> None:
        assert hex_to_rgb("#fff") == (255, 255, 255)
        assert hex_to_rgb("#abc") == (170, 187, 204)

    def test_without_hash(self) -> None:
        assert hex_to_rgb("1a7f37") == (26, 127, 55)

    @pytest.mark.parametrize("bad", ["#12", "#xyzxyz", "nothex", "#1234"])
    def test_invalid_raises(self, bad: str) -> None:
        with pytest.raises(ValueError):
            hex_to_rgb(bad)


class TestContrastRatio:
    def test_black_on_white_is_max(self) -> None:
        assert contrast_ratio("#000000", "#ffffff") == pytest.approx(21.0, abs=0.01)

    def test_identical_colors_is_one(self) -> None:
        assert contrast_ratio("#3fb950", "#3fb950") == pytest.approx(1.0, abs=0.001)

    def test_symmetric(self) -> None:
        a = contrast_ratio("#202020", "#e8e8e8")
        b = contrast_ratio("#e8e8e8", "#202020")
        assert a == pytest.approx(b)

    def test_luminance_bounds(self) -> None:
        assert relative_luminance("#000000") == pytest.approx(0.0, abs=1e-6)
        assert relative_luminance("#ffffff") == pytest.approx(1.0, abs=1e-6)


class TestMeetsWcagAa:
    def test_pass_normal(self) -> None:
        assert meets_wcag_aa("#1a1a1a", "#ffffff") is True

    def test_fail_normal_but_pass_large(self) -> None:
        # Cặp tương phản ~3.5: rớt chuẩn chữ thường nhưng đạt chữ lớn.
        fg, bg = "#888888", "#ffffff"
        assert meets_wcag_aa(fg, bg, large_text=False) is False
        assert meets_wcag_aa(fg, bg, large_text=True) is True


# ── Audit bảng màu thật ────────────────────────────────────────────────────────

# Các cặp (token_chữ, token_nền) là chữ THƯỜNG → phải đạt AA 4.5:1.
_NORMAL_TEXT_PAIRS = (
    ("on_surface", "surface"),
    ("on_surface_muted", "surface"),
    ("success", "surface"),
    ("warning", "surface"),
    ("danger", "surface"),
    ("info", "surface"),
    ("secondary", "surface"),
    ("on_surface", "surface_variant"),
    ("mono_fg", "mono_bg"),
    ("danger", "danger_bg"),
)

# Token gợi ý/placeholder in nghiêng (nhạt có chủ đích) → chỉ cần ngưỡng chữ lớn 3.0:1.
_LARGE_TEXT_PAIRS = (("muted_italic", "surface"),)


def _palettes() -> dict[str, dict[str, str]]:
    from subtitles_extractor.presentation.theme.colors import (
        DARK_PALETTE,
        LIGHT_PALETTE,
    )

    return {"dark": DARK_PALETTE, "light": LIGHT_PALETTE}


@pytest.mark.parametrize("theme_name", ["dark", "light"])
@pytest.mark.parametrize("fg,bg", _NORMAL_TEXT_PAIRS)
def test_normal_text_meets_aa(theme_name: str, fg: str, bg: str) -> None:
    palette = _palettes()[theme_name]
    ratio = contrast_ratio(palette[fg], palette[bg])
    assert ratio >= AA_NORMAL_RATIO, (
        f"[{theme_name}] {fg} trên {bg}: tương phản {ratio:.2f} < {AA_NORMAL_RATIO} "
        f"(WCAG AA chữ thường). Cần làm token đậm hơn."
    )


@pytest.mark.parametrize("theme_name", ["dark", "light"])
@pytest.mark.parametrize("fg,bg", _LARGE_TEXT_PAIRS)
def test_large_or_hint_text_meets_aa(theme_name: str, fg: str, bg: str) -> None:
    palette = _palettes()[theme_name]
    ratio = contrast_ratio(palette[fg], palette[bg])
    assert ratio >= AA_LARGE_RATIO, (
        f"[{theme_name}] {fg} trên {bg}: tương phản {ratio:.2f} < {AA_LARGE_RATIO} "
        f"(WCAG AA chữ lớn/gợi ý)."
    )
