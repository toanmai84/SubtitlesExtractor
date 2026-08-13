"""Lưu trữ bền vững Translation Memory (TM) theo phim bộ, dùng SQLite (WAL).

[v3.23.55] Mỗi phim bộ (định danh bằng ``series_key``) có một tập các cặp câu đã dịch
tích luỹ qua các tập. Khi dịch tập mới, ứng dụng truy hồi các câu liên quan từ đây để
làm tham chiếu (RAG grounding), giúp tên riêng/thuật ngữ/lối xưng hô NHẤT QUÁN giữa các tập.

Thiết kế đồng bộ với các store SQLite khác trong dự án: kết nối chia sẻ + RLock để an toàn
khi truy cập từ worker dịch (QThread).
"""

from __future__ import annotations

import contextlib
import sqlite3
import threading
from pathlib import Path

from subtitles_extractor.application.services.translation_memory import (
    TranslationMemoryEntry,
)

__all__ = ["SqliteTranslationMemoryStore"]


class SqliteTranslationMemoryStore:
    """CRUD bộ nhớ dịch theo ``series_key``, lưu trong SQLite (WAL)."""

    def __init__(self, db_path: str | Path) -> None:
        """Mở/khởi tạo store.

        Args:
            db_path: Đường dẫn file SQLite (tạo nếu chưa có).
        """
        self._db_path = str(db_path)
        self._lock: threading.RLock = threading.RLock()
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with contextlib.suppress(sqlite3.Error):
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_db()

    def _init_db(self) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS translation_memory (
                    series_key   TEXT NOT NULL,
                    source_text  TEXT NOT NULL,
                    target_text  TEXT NOT NULL,
                    updated_at   TEXT,
                    PRIMARY KEY (series_key, source_text)
                )
                """
            )
            # [v3.23.56] Ngữ cảnh chung của phim bộ: bảng thuật ngữ + roster nhân vật +
            # tóm tắt — chia sẻ giữa các tập để không phải khai báo lại mỗi tập.
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS series_context (
                    series_key  TEXT PRIMARY KEY,
                    glossary    TEXT DEFAULT '',
                    characters  TEXT DEFAULT '',
                    overview    TEXT DEFAULT '',
                    updated_at  TEXT
                )
                """
            )

    @staticmethod
    def _normalize_key(series_key: str) -> str:
        return str(series_key or "").strip()

    def add_entries(
        self, series_key: str, entries: list[TranslationMemoryEntry]
    ) -> int:
        """Thêm/cập nhật các cặp câu đã dịch cho một phim bộ.

        Cặp trùng ``source_text`` sẽ được cập nhật bản dịch mới nhất (last-write-wins).
        Bỏ qua các mục có câu nguồn hoặc câu đích rỗng.

        Args:
            series_key: Khoá phim bộ.
            entries: Danh sách cặp câu nguồn → đích.

        Returns:
            Số mục thực sự được ghi.

        Raises:
            ValueError: Nếu ``series_key`` rỗng.
        """
        key = self._normalize_key(series_key)
        if not key:
            raise ValueError("series_key không được rỗng.")
        rows = [
            (key, e.source_text.strip(), e.target_text.strip())
            for e in entries
            if e.source_text.strip() and e.target_text.strip()
        ]
        if not rows:
            return 0
        with self._lock, self._conn:
            self._conn.executemany(
                """
                INSERT INTO translation_memory (series_key, source_text, target_text, updated_at)
                VALUES (?, ?, ?, datetime('now'))
                ON CONFLICT(series_key, source_text)
                DO UPDATE SET target_text = excluded.target_text,
                              updated_at = excluded.updated_at
                """,
                rows,
            )
        return len(rows)

    def get_entries(
        self, series_key: str, *, limit: int = 5000
    ) -> list[TranslationMemoryEntry]:
        """Lấy các cặp câu đã dịch của một phim bộ (mới nhất trước).

        Args:
            series_key: Khoá phim bộ.
            limit: Số mục tối đa trả về (chặn để tránh tải quá lớn).

        Returns:
            Danh sách mục TM (có thể rỗng).
        """
        key = self._normalize_key(series_key)
        if not key:
            return []
        with self._lock, self._conn:
            cursor = self._conn.execute(
                """
                SELECT source_text, target_text FROM translation_memory
                WHERE series_key = ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (key, limit),
            )
            return [
                TranslationMemoryEntry(
                    source_text=row["source_text"], target_text=row["target_text"]
                )
                for row in cursor.fetchall()
            ]

    def count_entries(self, series_key: str) -> int:
        """Đếm số cặp câu đã lưu cho một phim bộ."""
        key = self._normalize_key(series_key)
        if not key:
            return 0
        with self._lock, self._conn:
            cursor = self._conn.execute(
                "SELECT COUNT(*) AS n FROM translation_memory WHERE series_key = ?",
                (key,),
            )
            row = cursor.fetchone()
            return int(row["n"]) if row else 0

    def clear_series(self, series_key: str) -> None:
        """Xoá toàn bộ bộ nhớ dịch VÀ ngữ cảnh chung của một phim bộ."""
        key = self._normalize_key(series_key)
        if not key:
            return
        with self._lock, self._conn:
            self._conn.execute(
                "DELETE FROM translation_memory WHERE series_key = ?", (key,)
            )
            self._conn.execute(
                "DELETE FROM series_context WHERE series_key = ?", (key,)
            )

    def list_series(self) -> list[tuple[str, int]]:
        """Liệt kê các phim bộ đã có bộ nhớ dịch, kèm số cặp câu mỗi bộ.

        Returns:
            Danh sách ``(series_key, số_cặp_câu)``, sắp theo tên phim bộ.
        """
        with self._lock, self._conn:
            cursor = self._conn.execute(
                """
                SELECT series_key, COUNT(*) AS n FROM translation_memory
                GROUP BY series_key ORDER BY series_key
                """
            )
            return [(row["series_key"], int(row["n"])) for row in cursor.fetchall()]

    def close(self) -> None:
        """Đóng kết nối SQLite."""
        with self._lock:
            with contextlib.suppress(sqlite3.Error):
                self._conn.close()

    # ── Ngữ cảnh chung của phim bộ (glossary + roster + tóm tắt) ──────────
    def save_series_context(
        self, series_key: str, glossary: str, characters: str, overview: str
    ) -> None:
        """Lưu/cập nhật ngữ cảnh chung của một phim bộ.

        Args:
            series_key: Khoá phim bộ.
            glossary: Bảng thuật ngữ tích luỹ.
            characters: Roster nhân vật tích luỹ.
            overview: Tóm tắt cốt truyện.

        Raises:
            ValueError: Nếu ``series_key`` rỗng.
        """
        key = self._normalize_key(series_key)
        if not key:
            raise ValueError("series_key không được rỗng.")
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO series_context (series_key, glossary, characters, overview, updated_at)
                VALUES (?, ?, ?, ?, datetime('now'))
                ON CONFLICT(series_key) DO UPDATE SET
                    glossary = excluded.glossary,
                    characters = excluded.characters,
                    overview = excluded.overview,
                    updated_at = excluded.updated_at
                """,
                (key, glossary or "", characters or "", overview or ""),
            )

    def get_series_context(self, series_key: str):
        """Lấy ngữ cảnh chung của phim bộ. Trả ``SeriesContext`` hoặc None nếu chưa có."""
        from subtitles_extractor.application.services.translation_memory import (
            SeriesContext,
        )
        key = self._normalize_key(series_key)
        if not key:
            return None
        with self._lock, self._conn:
            cursor = self._conn.execute(
                "SELECT glossary, characters, overview FROM series_context WHERE series_key = ?",
                (key,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return SeriesContext(
                glossary=row["glossary"] or "",
                characters=row["characters"] or "",
                overview=row["overview"] or "",
            )
