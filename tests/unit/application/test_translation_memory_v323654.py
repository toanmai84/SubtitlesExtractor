"""[v3.23.654] Bộ nhớ dịch: không lấy câu của chính video đang dịch; truy hồi gộp nhanh hơn mà kết quả như cũ."""

from __future__ import annotations

import random
from pathlib import Path

from subtitles_extractor.application.services.translation_memory import (
    TranslationMemoryEntry,
    retrieve_for_lines,
    retrieve_relevant,
)
from subtitles_extractor.infrastructure.database.sqlite_translation_memory_store import SqliteTranslationMemoryStore
from subtitles_extractor.presentation.view_models.translate_page_view_model import _drop_own_legacy_entries


def _old_loop(queries, entries, per_line_k=2, max_references=40):
    collected, seen = [], set()
    for query in queries:
        for entry in retrieve_relevant(query, entries, top_k=per_line_k):
            if entry.source_text not in seen:
                seen.add(entry.source_text)
                collected.append(entry)
        if len(collected) >= max_references:
            break
    return collected[:max_references]


def test_bulk_retrieval_matches_old_loop() -> None:
    random.seed(7)
    alphabet = "我你他师兄妹叔走吧来了不是好的人天地心大小"
    entries = [TranslationMemoryEntry("".join(random.choice(alphabet) for _ in range(random.randint(2, 8))), str(i))
               for i in range(800)]
    queries = ["".join(random.choice(alphabet) for _ in range(random.randint(2, 9))) for _ in range(120)]
    for k, cap in ((1, 10), (2, 40), (3, 200)):
        assert [e.source_text for e in retrieve_for_lines(queries, entries, per_line_k=k, max_references=cap)] == [
            e.source_text for e in _old_loop(queries, entries, k, cap)
        ]


def test_store_excludes_current_video(tmp_path: Path) -> None:
    store = SqliteTranslationMemoryStore(tmp_path / "tm.db")
    store.add_entries("phim", [TranslationMemoryEntry("师兄", "Sư huynh")], video_key="/p/ep1.mp4")
    store.add_entries("phim", [TranslationMemoryEntry("走吧", "Đi thôi")], video_key="/p/ep2.mp4")
    assert {e.source_text for e in store.get_entries("phim", exclude_video="/p/ep1.mp4")} == {"走吧"}
    assert {e.source_text for e in store.get_entries("phim")} == {"师兄", "走吧"}


def test_legacy_rows_get_video_column(tmp_path: Path) -> None:
    import sqlite3

    db = tmp_path / "old.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE translation_memory (series_key TEXT NOT NULL, source_text TEXT NOT NULL, "
                     "target_text TEXT NOT NULL, updated_at TEXT, PRIMARY KEY (series_key, source_text))")
        conn.execute("INSERT INTO translation_memory VALUES ('phim', '老夫', 'Lão phu', '2026-01-01')")
    store = SqliteTranslationMemoryStore(db)
    assert [e.source_text for e in store.get_entries("phim", exclude_video="/p/ep1.mp4")] == ["老夫"]


def test_drop_own_legacy_entries_only_when_mostly_same_film() -> None:
    current = ["一", "二", "三", "四"]
    own = [TranslationMemoryEntry(t, "x") for t in ("一", "二", "其他")]
    assert [e.source_text for e in _drop_own_legacy_entries(own, current)] == ["其他"]
    other_film = [TranslationMemoryEntry(t, "x") for t in ("一", "甲", "乙", "丙")]
    assert _drop_own_legacy_entries(other_film, current) == other_film   # 1/4 trùng < 30%: giữ nguyên


def test_locked_database_falls_back_without_video_column(tmp_path: Path) -> None:
    """Rà soát độc lập: ALTER TABLE khi CSDL đang bị khoá từng làm hỏng khởi tạo (mỗi lần gọi treo 5 s + rò kết nối)."""
    import sqlite3

    db = tmp_path / "old.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE translation_memory (series_key TEXT NOT NULL, source_text TEXT NOT NULL, "
                     "target_text TEXT NOT NULL, updated_at TEXT, PRIMARY KEY (series_key, source_text))")
    blocker = sqlite3.connect(db, timeout=0)
    blocker.execute("BEGIN EXCLUSIVE")
    try:
        import subtitles_extractor.infrastructure.database.sqlite_translation_memory_store as mod

        original = mod.sqlite3.connect
        mod.sqlite3.connect = lambda path, **kw: original(path, timeout=0.2, **{k: v for k, v in kw.items() if k != "timeout"})
        try:
            try:
                store = SqliteTranslationMemoryStore(db)
            except sqlite3.OperationalError:
                return  # khoá cả lúc đọc lược đồ: lỗi được báo, kết nối đã đóng
        finally:
            mod.sqlite3.connect = original
        assert store._has_video_key is False
    finally:
        blocker.rollback()
        blocker.close()
    store.add_entries("phim", [TranslationMemoryEntry("走吧", "Đi thôi")], video_key="/v.mp4")
    assert [e.source_text for e in store.get_entries("phim", exclude_video="/v.mp4")] == ["走吧"]
