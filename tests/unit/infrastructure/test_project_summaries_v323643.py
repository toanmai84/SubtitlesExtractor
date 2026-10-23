"""[v3.23.643] Danh sách dự án không nạp các cột lớn (phụ đề, OCR thô, cài đặt JSON)."""

from __future__ import annotations

import time
from pathlib import Path

from subtitles_extractor.domain.entities.project_record import ProjectRecord, WorkflowStage
from subtitles_extractor.infrastructure.database.sqlite_project_repository import (
    SqliteProjectRepository,
)


def _record(i: int, size: int = 0) -> ProjectRecord:
    return ProjectRecord(
        video_hash=f"h{i:04d}", video_path=f"/v/{i}.mkv", video_name=f"{i}.mkv",
        stage=WorkflowStage.TTS_DONE, target_lang="vi",
        original_subtitle="x" * size, translated_subtitle="y" * size,
        ocr_settings_json='{"a":1}', translation_settings_json="z" * size,
        tts_audio_path=f"/a/{i}.flac", published_video_path=f"/p/{i}.mp4",
        updated_at=f"2026-10-{(i % 28) + 1:02d}T10:00:00",
    )


def test_summaries_drop_heavy_columns_but_keep_what_the_list_shows(tmp_path: Path) -> None:
    repo = SqliteProjectRepository(tmp_path / "db.sqlite")
    repo.save(_record(1, size=1000))
    (summary,) = repo.list_summaries()
    full = repo.get("h0001")
    assert summary.original_subtitle == "" and summary.translated_subtitle == ""
    assert summary.translation_settings_json == "" and summary.ocr_raw_json == ""
    assert full is not None and len(full.original_subtitle) == 1000
    for field in ("video_hash", "video_path", "video_name", "stage", "target_lang",
                  "tts_audio_path", "published_video_path", "updated_at", "subtitle_format"):
        assert getattr(summary, field) == getattr(full, field), field


def test_summaries_are_ordered_like_list_all_and_much_cheaper(tmp_path: Path) -> None:
    repo = SqliteProjectRepository(tmp_path / "db.sqlite")
    for i in range(30):
        repo.save(_record(i, size=200_000))
    assert [r.video_hash for r in repo.list_summaries()] == [r.video_hash for r in repo.list_all()]
    start = time.perf_counter()
    repo.list_summaries()
    summaries_s = time.perf_counter() - start
    start = time.perf_counter()
    repo.list_all()
    all_s = time.perf_counter() - start
    assert summaries_s < all_s


def test_upgrading_from_a_database_without_the_newest_column(tmp_path: Path) -> None:
    import sqlite3

    path = tmp_path / "old.sqlite"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE projects (video_hash TEXT PRIMARY KEY, video_path TEXT, video_name TEXT, "
        "stage INTEGER, ocr_settings_json TEXT, ocr_raw_json TEXT, original_subtitle TEXT, "
        "subtitle_format TEXT, translated_subtitle TEXT, target_lang TEXT, "
        "translation_settings_json TEXT, tts_audio_path TEXT, tts_settings_json TEXT, "
        "created_at TEXT, updated_at TEXT)"
    )
    conn.execute("INSERT INTO projects (video_hash, video_name, stage) VALUES ('h1', 'a.mkv', 2)")
    conn.commit()
    conn.close()
    repo = SqliteProjectRepository(path)
    (summary,) = repo.list_summaries()
    assert summary.video_name == "a.mkv" and summary.published_video_path == ""
