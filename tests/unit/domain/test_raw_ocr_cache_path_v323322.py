"""Test quy ước đường dẫn cache OCR thô — v3.23.322.

OCR là khâu tốn thời gian nhất (hàng chục phút mỗi tập); dựng câu phụ đề chỉ mất vài
giây. Lưu cache OCR cho phép đổi tham số dựng câu rồi dựng lại tức thì.

Đường dẫn phải ổn định và KHÔNG đụng vào các tệp khác của quy trình
(``.original.srt`` / ``.translate.srt`` / ``.tts.srt``).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from subtitles_extractor.domain.value_objects.output_naming import (
    extracted_subtitle_path,
    raw_ocr_cache_path,
    tts_subtitle_path,
)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("D:/Phim/Tap01.mp4", "Tap01.seraw.json.gz"),
        ("/home/a/b.mkv", "b.seraw.json.gz"),
        ("phim.ts", "phim.seraw.json.gz"),
    ],
)
def test_cache_path_naming(source: str, expected: str) -> None:
    assert raw_ocr_cache_path(source).name == expected


def test_cache_path_is_gzipped() -> None:
    """Phải dùng ``.gz`` — dữ liệu OCR một tập rất lớn, serializer tự nén."""
    assert raw_ocr_cache_path("x.mp4").name.endswith(".gz")


def test_cache_sits_next_to_video() -> None:
    path = raw_ocr_cache_path("/videos/serie/Tap05.mkv")
    assert path.parent == Path("/videos/serie")


def test_cache_path_is_stable() -> None:
    """Gọi nhiều lần phải ra cùng đường dẫn (dùng để tìm lại cache)."""
    assert raw_ocr_cache_path("a/b.mp4") == raw_ocr_cache_path("a/b.mp4")


def test_does_not_collide_with_other_pipeline_files() -> None:
    """KHÔNG được trùng tệp phụ đề của các khâu khác — trùng là ghi đè mất dữ liệu."""
    video = "Tap01.mp4"
    cache = raw_ocr_cache_path(video)
    assert cache != extracted_subtitle_path(video)
    assert cache != tts_subtitle_path(video, "vi")
    assert cache.suffix != ".srt"


@pytest.mark.parametrize("source", ["Tap01.mp4", "Tap01.mkv", "Tap01.ts"])
def test_same_stem_different_container_shares_cache(source: str) -> None:
    """Cùng tên gốc -> cùng cache, bất kể đuôi video."""
    assert raw_ocr_cache_path(source).name == "Tap01.seraw.json.gz"
