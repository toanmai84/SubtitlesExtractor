"""[v3.23.638] Tệp cấu hình / điểm dừng / dữ liệu OCR thô bị HỎNG không được làm ứng dụng sập.

Fuzz tệp hỏng: byte không phải UTF-8 (``UnicodeDecodeError`` là ``ValueError``, KHÔNG phải
``JSONDecodeError``) làm ``JsonFileRepository`` ném lỗi ngay lúc khởi động; tệp ``.seraw.json.gz`` cụt
ném ``EOFError``; gốc JSON là danh sách ném ``AttributeError`` ở ``load_raw_ocr``.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from subtitles_extractor.infrastructure.serializers.raw_ocr_serializer import load_raw_ocr
from subtitles_extractor.infrastructure.settings.json_file_repository import JsonFileRepository

BAD = [b"", b"{", b"\xff\xfe\x00", b"\x80abc", b"[1,2]", b"null", b'{"a":', "tiếng Việt".encode("cp1252", "ignore")]


@pytest.mark.parametrize("raw", BAD)
def test_settings_file_corrupt_falls_back_to_empty(tmp_path: Path, raw: bytes) -> None:
    path = tmp_path / "config.json"
    path.write_bytes(raw)
    repo = JsonFileRepository(file_path=path)
    assert repo.get("anything") is None if hasattr(repo, "get") else True


@pytest.mark.parametrize("raw", BAD + [b'{"version": "x"}'])
def test_raw_ocr_loader_raises_only_value_error(tmp_path: Path, raw: bytes) -> None:
    path = tmp_path / "x.seraw.json"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        load_raw_ocr(path)


def test_truncated_gzip_raw_ocr_is_a_value_error(tmp_path: Path) -> None:
    good = gzip.compress(json.dumps({"version": "1"}).encode() * 500)
    path = tmp_path / "x.seraw.json.gz"
    path.write_bytes(good[: len(good) // 2])
    with pytest.raises(ValueError):
        load_raw_ocr(path)


def test_analysis_checkpoint_with_invalid_utf8_is_ignored(tmp_path: Path) -> None:
    from subtitles_extractor.infrastructure.translation.analysis_checkpoint import (
        AnalysisCheckpointStore,
    )

    store = AnalysisCheckpointStore(tmp_path)
    for path in tmp_path.glob("*"):
        path.unlink()
    (tmp_path / "bad.json").write_bytes(b"\xff\xfe\x00junk")
    # tên tệp do _path(key) quyết định: ghi đè đúng tệp đó
    target = store._path("k")
    target.write_bytes(b"\xff\xfe\x00junk")
    assert store.load("k") is None
