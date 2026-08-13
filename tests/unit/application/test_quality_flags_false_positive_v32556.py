"""[v3.23.156] Test giảm DƯƠNG TÍNH GIẢ của quality flags (calibrate từ dữ liệu thật).

Dữ liệu The Hot Spot (968 dòng): 25/25 cờ identical_to_source đều là tên riêng/số
("Harry.", "357...", "Madox. Harry Madox.") — dịch giữ nguyên là ĐÚNG; 22/23 cờ
length_anomaly là bản dịch SÚC TÍCH chủ đích hoặc nguồn SONG NGỮ CJK+Latin bị đếm
đôi độ dài. Sau calibrate: identical 25 -> 0, length_anomaly 23 -> 0 trên phim này,
trong khi các ca lỗi THẬT vẫn bị bắt.
"""

from __future__ import annotations

from types import SimpleNamespace

from subtitles_extractor.application.services.translation_diagnostics import (
    _is_untranslatable_source,
    detect_quality_flags,
)


def _ev(index: int, text: str) -> SimpleNamespace:
    return SimpleNamespace(
        index=index, text=text, start_sec=float(index), end_sec=float(index) + 2.0
    )


def _flags(pairs: list[tuple[str, str]]) -> dict:
    src = [_ev(i + 1, s) for i, (s, _t) in enumerate(pairs)]
    dst = [_ev(i + 1, t) for i, (_s, t) in enumerate(pairs)]
    return detect_quality_flags(src, dst)


# ── identical_to_source: tên riêng/số không còn bị flag oan ─────────────


def test_proper_nouns_and_numbers_not_flagged() -> None:
    pairs = [
        ("Harry.", "[Harry Madox:] Harry."),
        ("Gloria...", "Gloria..."),
        ("357...", "357..."),
        ("Madox. Harry Madox.", "[Harry Madox:] Madox. Harry Madox."),
        ("Lon.", "Lon."),
    ]
    assert _flags(pairs)["identical_to_source_indices"] == []


def test_real_untranslated_sentence_still_flagged() -> None:
    pairs = [("Where are you going tonight my friend?",
              "Where are you going tonight my friend?")]
    assert _flags(pairs)["identical_to_source_indices"] == [1]


def test_is_untranslatable_source_boundaries() -> None:
    assert _is_untranslatable_source("Harry.") is True
    assert _is_untranslatable_source("...") is True
    assert _is_untranslatable_source("3957...") is True
    # 5 token TitleCase -> quá dài, coi là câu thật.
    assert _is_untranslatable_source("John Paul George Ringo Pete") is False
    # Có từ thường -> câu thật.
    assert _is_untranslatable_source("Harry is here.") is False


# ── length_anomaly: song ngữ + súc tích chủ đích không còn bị flag oan ──


def test_bilingual_cjk_latin_source_counts_longest_line_only() -> None:
    src_bilingual = "\u222e须得别处觅芳草\u222e\n\u222e Gotta find my baby \u222e"
    pairs = [(src_bilingual, "[Ca sĩ:] Phải tìm em yêu thôi")]
    assert _flags(pairs)["length_anomaly"] == []


def test_concise_translation_not_flagged() -> None:
    pairs = [(
        "Let me run inside and see what kind of deal I can work for you.",
        "Để tôi vào xem giá tốt cho ông.",
    )]
    assert _flags(pairs)["length_anomaly"] == []


def test_extreme_shrink_still_flagged() -> None:
    pairs = [(
        "Let me run inside and see what kind of deal I can work for you today, sir.",
        "Vào đi.",
    )]
    anomaly = _flags(pairs)["length_anomaly"]
    assert len(anomaly) == 1 and anomaly[0]["index"] == 1


def test_tiny_source_ratio_ignored() -> None:
    pairs = [("Yes, I am.", "Có.")]
    assert _flags(pairs)["length_anomaly"] == []
