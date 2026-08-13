"""Lưu trữ Ngữ cảnh Dịch (Bảng nhân vật / Tóm tắt bối cảnh) theo TỪNG dự án.

Trước đây ngữ cảnh được lưu vào QSettings TOÀN CỤC → mở phim B vẫn thấy tóm tắt của
phim A (context leak). Store này cô lập ngữ cảnh theo khoá dự án (đường dẫn video
hoặc hash) trong SQLite, để mỗi phim có ngữ cảnh riêng.
"""

from __future__ import annotations

import contextlib
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class TranslationContext:
    """Ngữ cảnh dịch của một dự án."""

    characters: str = ""
    overview: str = ""
    source_lang: str = ""
    target_lang: str = ""

    def is_empty(self) -> bool:
        return not (self.characters or self.overview or self.source_lang or self.target_lang)


class SqliteTranslationContextStore:
    """CRUD ngữ cảnh dịch theo khoá dự án, lưu trong SQLite (WAL)."""

    def __init__(self, db_path: str | Path) -> None:
        """Mở/khởi tạo store.

        Args:
            db_path: Đường dẫn file SQLite (tạo nếu chưa có).
        """
        self._db_path = str(db_path)
        # [v3.20 B1] check_same_thread=False + RLock — đồng bộ với 3 repo SQLite kia.
        # Cho phép truy cập an toàn từ worker dịch (QThread) nếu store được dùng
        # xuyên luồng; RLock tuần tự hoá mọi thao tác trên connection chia sẻ.
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
                CREATE TABLE IF NOT EXISTS translation_context (
                    project_key TEXT PRIMARY KEY,
                    characters  TEXT DEFAULT '',
                    overview    TEXT DEFAULT '',
                    source_lang TEXT DEFAULT '',
                    target_lang TEXT DEFAULT '',
                    updated_at  TEXT
                )
                """
            )

    @staticmethod
    def _normalize_key(project_key: str | Path) -> str:
        # str() phòng WindowsPath; sqlite3 không nhận kiểu Path trực tiếp.
        return str(project_key).strip()

    def save(self, project_key: str | Path, context: TranslationContext) -> None:
        """Lưu (chèn hoặc cập nhật) ngữ cảnh cho một dự án.

        Raises:
            ValueError: Nếu ``project_key`` rỗng.
        """
        key = self._normalize_key(project_key)
        if not key:
            raise ValueError("project_key không được rỗng.")
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO translation_context
                    (project_key, characters, overview, source_lang, target_lang, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(project_key) DO UPDATE SET
                    characters=excluded.characters,
                    overview=excluded.overview,
                    source_lang=excluded.source_lang,
                    target_lang=excluded.target_lang,
                    updated_at=excluded.updated_at
                """,
                (key, context.characters, context.overview,
                 context.source_lang, context.target_lang, now),
            )

    def get(self, project_key: str | Path) -> TranslationContext | None:
        """Đọc ngữ cảnh của một dự án, hoặc ``None`` nếu chưa có."""
        key = self._normalize_key(project_key)
        if not key:
            return None
        with self._lock:
            cursor = self._conn.execute(
                "SELECT characters, overview, source_lang, target_lang "
                "FROM translation_context WHERE project_key = ?",
                (key,),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return TranslationContext(
            characters=row["characters"] or "",
            overview=row["overview"] or "",
            source_lang=row["source_lang"] or "",
            target_lang=row["target_lang"] or "",
        )

    def delete(self, project_key: str | Path) -> None:
        """Xoá ngữ cảnh của một dự án (không lỗi nếu không tồn tại)."""
        key = self._normalize_key(project_key)
        if not key:
            return
        with self._lock, self._conn:
            self._conn.execute(
                "DELETE FROM translation_context WHERE project_key = ?", (key,)
            )

    def close(self) -> None:
        with self._lock, contextlib.suppress(sqlite3.Error):
            self._conn.close()
