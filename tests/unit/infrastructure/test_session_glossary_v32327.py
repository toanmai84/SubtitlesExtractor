"""Test [v3.23.27] lưu/khôi phục glossary trong session store + migration DB cũ."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from subtitles_extractor.infrastructure.database.sqlite_translation_session_store import (
    SqliteTranslationSessionStore,
)


class TestGlossaryPersistence:
    def test_save_and_get_glossary(self, tmp_path: Path) -> None:
        store = SqliteTranslationSessionStore(tmp_path / "s.db")
        store.save_analysis(
            "vid1", characters="林昆 (Lâm Côn)", overview="Phim",
            source_lang="zh", target_lang="vi", input_hash="h1",
            glossary="内功 => nội công",
        )
        session = store.get("vid1")
        assert session is not None
        assert session.analysis_glossary == "内功 => nội công"

    def test_glossary_defaults_empty(self, tmp_path: Path) -> None:
        store = SqliteTranslationSessionStore(tmp_path / "s.db")
        store.save_analysis(
            "vid2", characters="", overview="x", source_lang="zh",
            target_lang="vi", input_hash="h",
        )
        assert store.get("vid2").analysis_glossary == ""


class TestMigrationOldDb:
    def test_migrate_adds_column_keeps_data(self, tmp_path: Path) -> None:
        db = tmp_path / "old.db"
        # DB cũ KHÔNG có cột analysis_glossary.
        conn = sqlite3.connect(db)
        conn.execute(
            "CREATE TABLE translation_session (video_key TEXT PRIMARY KEY, "
            "source_hash TEXT DEFAULT '', analysis_characters TEXT DEFAULT '', "
            "analysis_overview TEXT DEFAULT '', analysis_source_lang TEXT DEFAULT '', "
            "analysis_target_lang TEXT DEFAULT '', analysis_input_hash TEXT DEFAULT '', "
            "stages_json TEXT DEFAULT '[]', cloud_files_json TEXT DEFAULT '[]', "
            "updated_at TEXT)"
        )
        conn.execute(
            "INSERT INTO translation_session (video_key, analysis_overview, updated_at) "
            "VALUES ('old', 'Phim cũ', '2026-01-01')"
        )
        conn.commit()
        conn.close()

        store = SqliteTranslationSessionStore(db)  # tự migrate
        session = store.get("old")
        assert session.analysis_overview == "Phim cũ"  # dữ liệu cũ còn nguyên
        assert session.analysis_glossary == ""          # cột mới mặc định rỗng

        # Ghi glossary vào row cũ.
        store.save_analysis(
            "old", characters="", overview="Phim cũ", source_lang="zh",
            target_lang="vi", input_hash="h", glossary="thuật ngữ",
        )
        assert store.get("old").analysis_glossary == "thuật ngữ"

    def test_migrate_idempotent(self, tmp_path: Path) -> None:
        db = tmp_path / "s.db"
        SqliteTranslationSessionStore(db)
        # Mở lại lần 2 — migration không được lỗi (cột đã có).
        store2 = SqliteTranslationSessionStore(db)
        store2.save_analysis(
            "v", characters="", overview="x", source_lang="zh",
            target_lang="vi", input_hash="h", glossary="g",
        )
        assert store2.get("v").analysis_glossary == "g"


class TestVisualCuesPersistence:
    def test_save_and_get_visual_cues(self, tmp_path) -> None:
        store = SqliteTranslationSessionStore(tmp_path / "s.db")
        store.save_analysis(
            "vid1", characters="", overview="x", source_lang="en",
            target_lang="vi", input_hash="h",
            visual_cues='[{"id":1,"spk":"Lâm Côn"}]',
        )
        session = store.get("vid1")
        assert session.analysis_visual_cues == '[{"id":1,"spk":"Lâm Côn"}]'

    def test_visual_cues_defaults_empty(self, tmp_path) -> None:
        store = SqliteTranslationSessionStore(tmp_path / "s.db")
        store.save_analysis(
            "vid2", characters="", overview="x", source_lang="en",
            target_lang="vi", input_hash="h",
        )
        assert store.get("vid2").analysis_visual_cues == ""

    def test_migrate_old_db_adds_visual_cues(self, tmp_path) -> None:
        import sqlite3
        db = tmp_path / "old.db"
        conn = sqlite3.connect(db)
        # DB cũ chỉ có tới analysis_glossary (v3.23.27), CHƯA có analysis_visual_cues.
        conn.execute(
            "CREATE TABLE translation_session (video_key TEXT PRIMARY KEY, "
            "source_hash TEXT DEFAULT '', analysis_characters TEXT DEFAULT '', "
            "analysis_overview TEXT DEFAULT '', analysis_glossary TEXT DEFAULT '', "
            "analysis_source_lang TEXT DEFAULT '', analysis_target_lang TEXT DEFAULT '', "
            "analysis_input_hash TEXT DEFAULT '', stages_json TEXT DEFAULT '[]', "
            "cloud_files_json TEXT DEFAULT '[]', updated_at TEXT)"
        )
        conn.execute(
            "INSERT INTO translation_session (video_key, analysis_glossary, updated_at) "
            "VALUES ('old', 'thuật ngữ cũ', '2026-01-01')"
        )
        conn.commit()
        conn.close()

        store = SqliteTranslationSessionStore(db)  # tự migrate
        session = store.get("old")
        assert session.analysis_glossary == "thuật ngữ cũ"  # dữ liệu cũ còn
        assert session.analysis_visual_cues == ""            # cột mới mặc định rỗng
        store.save_analysis(
            "old", characters="", overview="", source_lang="en",
            target_lang="vi", input_hash="h", visual_cues='[{"id":2}]',
        )
        assert store.get("old").analysis_visual_cues == '[{"id":2}]'
