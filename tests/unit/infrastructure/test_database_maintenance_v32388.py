"""[v3.23.88] Test dọn database về "như mới tạo" (reset_database)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from subtitles_extractor.infrastructure.database.maintenance import reset_database


def _make_db(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE projects (id INTEGER PRIMARY KEY, name TEXT)")
    conn.execute("CREATE TABLE translation_memory (id INTEGER PRIMARY KEY, txt TEXT)")
    conn.execute("INSERT INTO projects (name) VALUES ('a'), ('b')")
    conn.execute("INSERT INTO translation_memory (txt) VALUES ('x')")
    conn.commit()
    conn.close()


def test_reset_clears_all_tables(tmp_path: Path) -> None:
    db = tmp_path / "app_state.db"
    _make_db(db)
    cleared = reset_database(db)
    assert set(cleared) == {"projects", "translation_memory"}
    conn = sqlite3.connect(str(db))
    try:
        assert conn.execute("SELECT COUNT(*) FROM projects").fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM translation_memory"
        ).fetchone()[0] == 0
        # Schema VẪN còn (bảng tồn tại) -> "như mới tạo", không phải xoá tệp.
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        assert {"projects", "translation_memory"} <= {t[0] for t in tables}
    finally:
        conn.close()


def test_reset_missing_db_returns_empty(tmp_path: Path) -> None:
    assert reset_database(tmp_path / "khong_ton_tai.db") == []


def test_reset_ignores_sqlite_internal_tables(tmp_path: Path) -> None:
    db = tmp_path / "app_state.db"
    conn = sqlite3.connect(str(db))
    # AUTOINCREMENT tạo bảng nội bộ sqlite_sequence.
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY AUTOINCREMENT, v TEXT)")
    conn.execute("INSERT INTO t (v) VALUES ('a')")
    conn.commit()
    conn.close()
    cleared = reset_database(db)
    assert "t" in cleared
    assert not any(name.startswith("sqlite_") for name in cleared)
