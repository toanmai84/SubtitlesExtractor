"""[v3.23.426] Test cờ cjk_leftover — bắt chữ CJK chưa dịch trong bản dịch (đích non-CJK)."""

from __future__ import annotations

from types import SimpleNamespace as NS

from subtitles_extractor.application.services.translation_diagnostics import (
    detect_quality_flags,
)


def _flags(pairs: list[tuple[str, str]]):
    src = [NS(index=i, text=s) for i, (s, _) in enumerate(pairs)]
    dst = [NS(index=i, text=d) for i, (_, d) in enumerate(pairs)]
    return detect_quality_flags(src, dst)


def test_catches_leftover_cjk_char() -> None:
    """Chữ '夫' còn sót trong bản Việt phải bị gắn cờ."""
    flags = _flags([
        ("文人士大夫", "văn nhân sĩ đại夫"),  # sót '夫'
        ("你好", "Xin chào"),               # sạch
    ])
    leftover = flags["cjk_leftover"]
    assert len(leftover) == 1
    assert leftover[0]["index"] == 0
    assert leftover[0]["chars"] == "夫"
    assert flags["counts"]["cjk_leftover"] == 1


def test_clean_vietnamese_no_flag() -> None:
    flags = _flags([("你好世界", "Xin chào thế giới"), ("再见", "Tạm biệt")])
    assert flags["cjk_leftover"] == []


def test_target_cjk_not_flagged() -> None:
    """Dịch SANG tiếng Trung (đích CJK) → không gắn cờ dù bản dịch đầy CJK."""
    flags = _flags([("Hello world", "你好世界"), ("Goodbye", "再见朋友")])
    assert flags["cjk_leftover"] == []


def test_dedup_repeated_chars() -> None:
    flags = _flags([("测试", "thử 测 nghiệm 测 lại")])
    assert flags["cjk_leftover"][0]["chars"] == "测"  # bỏ trùng
