"""Lưu trữ bền vững Translation Memory (TM) theo phim bộ, dùng SQLite (WAL).

[v3.23.55] Mỗi phim bộ (định danh bằng ``series_key``) có một tập các cặp câu đã dịch
tích luỹ qua các tập. Khi dịch tập mới, ứng dụng truy hồi các câu liên quan từ đây để
làm tham chiếu (RAG grounding), giúp tên riêng/thuật ngữ/lối xưng hô NHẤT QUÁN giữa các tập.

Thiết kế đồng bộ với các store SQLite khác trong dự án: kết nối chia sẻ + RLock để an toàn
khi truy cập từ worker dịch (QThread).
"""

from __future__ import annotations

import contextlib
import logging
import sqlite3

from subtitles_extractor.infrastructure.database.db_guard import prepare_database
import threading
from pathlib import Path

from subtitles_extractor.application.services.translation_memory import (
    TranslationMemoryEntry,
)

__all__ = ["SqliteTranslationMemoryStore"]


logger = logging.getLogger(__name__)


class SqliteTranslationMemoryStore:
    """CRUD bộ nhớ dịch theo ``series_key``, lưu trong SQLite (WAL)."""

    def __init__(self, db_path: str | Path) -> None:
        """Mở/khởi tạo store.

        Args:
            db_path: Đường dẫn file SQLite (tạo nếu chưa có).
        """
        self._db_path = str(db_path)
        self._lock: threading.RLock = threading.RLock()
        # [v3.23.649] Kiểm tra / sao lưu / tự phục hồi tệp trước khi kết nối.
        prepare_database(self._db_path, background=True)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with contextlib.suppress(sqlite3.Error):
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._has_video_key = False
        try:
            self._init_db()
        except sqlite3.Error:
            self._conn.close()  # [v3.23.654] không rò kết nối (Windows khoá tệp CSDL khi còn handle mở)
            raise

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
            # [v3.23.654] Ghi nhớ câu thuộc VIDEO nào để khi dịch lại chính video đó không lấy bản dịch cũ của nó làm
            # "tham chiếu" (bộ nhớ tự nuôi chính nó, kể cả lỗi). Hàng cũ: NULL.
            columns = {row[1] for row in self._conn.execute("PRAGMA table_info(translation_memory)")}
            self._has_video_key = "video_key" in columns
            if not self._has_video_key:
                try:
                    self._conn.execute("ALTER TABLE translation_memory ADD COLUMN video_key TEXT")
                    self._has_video_key = True
                except sqlite3.OperationalError as exc:
                    # CSDL đang bị khoá / chỉ đọc / phiên khác vừa thêm cột: dùng tiếp kiểu cũ, không treo/không vỡ.
                    if "duplicate column" in str(exc).lower():
                        self._has_video_key = True
                    else:
                        logger.warning("Bộ nhớ dịch: chưa thêm được cột video_key (%s) — tạm dùng như cũ.", exc)
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
        self, series_key: str, entries: list[TranslationMemoryEntry], video_key: str = "",
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
        video = str(video_key or "").strip() or None
        rows = [
            (key, e.source_text.strip(), e.target_text.strip(), video)
            for e in entries
            if e.source_text.strip() and e.target_text.strip()
        ]
        if not rows:
            return 0
        with self._lock, self._conn:
            if not self._has_video_key:
                self._conn.executemany(
                    """
                    INSERT INTO translation_memory (series_key, source_text, target_text, updated_at)
                    VALUES (?, ?, ?, datetime('now'))
                    ON CONFLICT(series_key, source_text)
                    DO UPDATE SET target_text = excluded.target_text, updated_at = excluded.updated_at
                    """,
                    [row[:3] for row in rows],
                )
                return len(rows)
            self._conn.executemany(
                """
                INSERT INTO translation_memory (series_key, source_text, target_text, video_key, updated_at)
                VALUES (?, ?, ?, ?, datetime('now'))
                ON CONFLICT(series_key, source_text)
                DO UPDATE SET target_text = excluded.target_text,
                              video_key = excluded.video_key,
                              updated_at = excluded.updated_at
                """,
                rows,
            )
        return len(rows)

    def get_entries(
        self, series_key: str, *, limit: int = 5000, exclude_video: str = "",
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
            if not self._has_video_key:
                # [v3.23.659] Chưa thêm được cột (CSDL bị khoá lúc mở): câu SQL có nhắc tới video_key lỗi ngay khi
                # biên dịch ("no such column") dù điều kiện "? = 0" — trước đây làm tắt bộ nhớ dịch cả phiên.
                cursor = self._conn.execute(
                    "SELECT source_text, target_text FROM translation_memory WHERE series_key = ? "
                    "ORDER BY updated_at DESC LIMIT ?",
                    (key, limit),
                )
            else:
                cursor = self._conn.execute(
                    """
                    SELECT source_text, target_text FROM translation_memory
                    WHERE series_key = ? AND (video_key IS NULL OR video_key != ?)
                    ORDER BY updated_at DESC
                    LIMIT ?
                    """,
                    (key, str(exclude_video or "").strip(), limit),
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
