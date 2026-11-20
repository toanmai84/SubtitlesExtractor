"""[v3.23.750] Nhật ký ghi phiên bản lúc khởi động (log 19/11 07:28: không biết exe đang chạy bản nào)."""

from __future__ import annotations

from pathlib import Path


def test_bootstrap_logs_app_version() -> None:
    source = (Path(__file__).resolve().parents[2] / "src/subtitles_extractor/composition/bootstrap.py").read_text(
        encoding="utf-8")
    assert "from subtitles_extractor import __version__" in source
    assert 'logger.info("Subtitles Extractor %s — Bootstrap GUI hoàn tất' in source
