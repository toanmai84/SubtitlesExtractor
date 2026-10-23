"""[v3.23.428] Test đặt tên file chẩn đoán nhất quán (có timestamp, tránh ghi đè)."""

from __future__ import annotations

from datetime import datetime

from subtitles_extractor.application.services.diagnostic_naming import (
    build_diagnostic_filename,
    build_diagnostic_path,
)

_NOW = datetime(2026, 8, 25, 14, 30, 22)


def test_with_source_uses_basename_and_timestamp() -> None:
    name = build_diagnostic_filename(
        "F:/Phim/tap10.mkv", "translation_debug", "json", now=_NOW
    )
    assert name == "tap10.translation_debug.20260825_143022.json"


def test_no_source_uses_generic_base() -> None:
    name = build_diagnostic_filename(None, "tts_debug", "csv", now=_NOW)
    assert name == "subtitle.tts_debug.20260825_143022.csv"


def test_invalid_chars_sanitized() -> None:
    name = build_diagnostic_filename('a<b>:c"d.mkv', "ocr_raw", "json", now=_NOW)
    base = name.split(".")[0]
    assert not any(c in base for c in '<>:"/\\|?*')


def test_path_next_to_source() -> None:
    path = build_diagnostic_path(
        "F:/Phim/tap10.mkv", "translation_debug", "json", now=_NOW
    ).replace("\\", "/")
    assert path == "F:/Phim/tap10.translation_debug.20260825_143022.json"


def test_path_with_explicit_output_dir() -> None:
    path = build_diagnostic_path(
        "F:/Phim/tap10.mkv", "tts_debug", "csv", output_dir="D:/out", now=_NOW
    ).replace("\\", "/")
    assert path == "D:/out/tap10.tts_debug.20260825_143022.csv"


def test_different_timestamps_differ() -> None:
    a = build_diagnostic_filename("v.mkv", "x", "json", now=datetime(2026, 1, 1, 0, 0, 0))
    b = build_diagnostic_filename("v.mkv", "x", "json", now=datetime(2026, 1, 1, 0, 0, 1))
    assert a != b  # mỗi lần xuất một tên → không ghi đè


def test_long_base_truncated() -> None:
    name = build_diagnostic_filename("x" * 200 + ".mkv", "s", "json", now=_NOW)
    base = name.split(".")[0]
    assert len(base) <= 80


def test_ext_dot_stripped() -> None:
    name = build_diagnostic_filename("v.mkv", "s", ".json", now=_NOW)
    assert name.endswith(".json") and "..json" not in name
