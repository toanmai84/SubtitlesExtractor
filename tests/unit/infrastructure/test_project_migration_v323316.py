"""Test nâng cấp CSDL dự án khi thêm cột mới — v3.23.316.

Thêm trường vào :class:`ProjectRecord` mà quên nâng cấp bảng SQLite sẽ làm ứng dụng
lỗi ``no such column`` với MỌI người dùng đang có dự án. Bộ test này chống hồi quy đó.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from subtitles_extractor.domain.entities.project_record import (
    ProjectRecord,
    WorkflowStage,
)
from subtitles_extractor.infrastructure.database.sqlite_project_repository import (
    SqliteProjectRepository,
)

# Schema y hệt bản phát hành TRƯỚC khi thêm cột published_video_path.
_LEGACY_SCHEMA = """
CREATE TABLE projects (
    video_hash TEXT PRIMARY KEY, video_path TEXT, video_name TEXT,
    stage INTEGER DEFAULT 0, ocr_settings_json TEXT, ocr_raw_json TEXT,
    original_subtitle TEXT, subtitle_format TEXT DEFAULT 'srt',
    translated_subtitle TEXT, target_lang TEXT, translation_settings_json TEXT,
    tts_audio_path TEXT, tts_settings_json TEXT,
    created_at TEXT, updated_at TEXT
)
"""


def _make_legacy_db(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.execute(_LEGACY_SCHEMA)
    connection.execute(
        "INSERT INTO projects (video_hash, video_name, stage, original_subtitle) "
        "VALUES (?,?,?,?)",
        ("hash-cu", "Phim cu.mp4", int(WorkflowStage.TRANSLATED), "phu de cu"),
    )
    connection.commit()
    connection.close()


def test_migration_adds_missing_column(tmp_path: Path) -> None:
    database = tmp_path / "projects.db"
    _make_legacy_db(database)

    SqliteProjectRepository(database)  # mở là tự nâng cấp

    columns = {
        row[1]
        for row in sqlite3.connect(database).execute("PRAGMA table_info(projects)")
    }
    assert "published_video_path" in columns


def test_migration_preserves_existing_data(tmp_path: Path) -> None:
    """Dữ liệu người dùng KHÔNG được mất khi nâng cấp."""
    database = tmp_path / "projects.db"
    _make_legacy_db(database)

    record = SqliteProjectRepository(database).get("hash-cu")

    assert record is not None
    assert record.video_name == "Phim cu.mp4"
    assert record.stage is WorkflowStage.TRANSLATED
    assert record.original_subtitle == "phu de cu"
    assert record.published_video_path == ""  # cột mới mặc định rỗng


def test_can_write_and_read_new_stage(tmp_path: Path) -> None:
    database = tmp_path / "projects.db"
    _make_legacy_db(database)
    repository = SqliteProjectRepository(database)

    record = repository.get("hash-cu")
    assert record is not None
    record.stage = WorkflowStage.PUBLISHED
    record.published_video_path = "D:/out/phim_vi.mkv"
    repository.save(record)

    reloaded = repository.get("hash-cu")
    assert reloaded is not None
    assert reloaded.stage is WorkflowStage.PUBLISHED
    assert reloaded.published_video_path == "D:/out/phim_vi.mkv"


def test_migration_is_idempotent(tmp_path: Path) -> None:
    """Mở lại nhiều lần không được lỗi (hàm nâng cấp chạy mỗi lần khởi động)."""
    database = tmp_path / "projects.db"
    _make_legacy_db(database)

    for _ in range(3):
        repository = SqliteProjectRepository(database)
        assert repository.get("hash-cu") is not None


def test_fresh_database_has_new_column(tmp_path: Path) -> None:
    """CSDL tạo mới cũng phải có cột — không chỉ CSDL nâng cấp."""
    database = tmp_path / "fresh.db"
    repository = SqliteProjectRepository(database)
    record = ProjectRecord(video_hash="moi", video_name="Phim moi.mp4")
    record.published_video_path = "out.mkv"
    repository.save(record)

    reloaded = repository.get("moi")
    assert reloaded is not None
    assert reloaded.published_video_path == "out.mkv"
