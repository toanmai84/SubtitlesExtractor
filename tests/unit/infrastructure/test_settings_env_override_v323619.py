"""[v3.23.619] Biến môi trường ``SE_*`` ghi đè cấu hình đã lưu (như README hướng dẫn).

Trước đây biến môi trường chỉ có tác dụng ở lần chạy ĐẦU (chưa có ``config.json``); từ lần
sau giá trị đã lưu thắng → ``export SE_HW_DEVICE=cpu`` / ``SE_OCR_SCORE_THRESHOLD=0.5`` bị
lờ đi hoàn toàn (gặp khi kiểm thử trên phim mẫu).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from subtitles_extractor.infrastructure.settings.json_file_repository import JsonFileRepository
from subtitles_extractor.infrastructure.settings.settings_service import SettingsService


def _service(path: Path) -> SettingsService:
    return SettingsService(JsonFileRepository(file_path=path))


def test_env_overrides_stored_value(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = tmp_path / "config.json"
    first = _service(config)
    first.update(ocr={"score_threshold": 0.75}, hardware={"device": "gpu"})
    first.flush()

    monkeypatch.setenv("SE_OCR_SCORE_THRESHOLD", "0.5")
    monkeypatch.setenv("SE_HW_DEVICE", "cpu")
    current = _service(config).current
    assert current.ocr.score_threshold == 0.5
    assert current.hardware.device.value == "cpu"
    # Nhóm/khoá không có biến môi trường giữ nguyên giá trị đã lưu.
    assert current.ocr.language == first.current.ocr.language


def test_invalid_env_value_is_ignored_not_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.json"
    first = _service(config)
    first.update(ocr={"score_threshold": 0.7})
    first.flush()
    monkeypatch.setenv("SE_OCR_SCORE_THRESHOLD", "không phải số")
    assert _service(config).current.ocr.score_threshold == 0.7
