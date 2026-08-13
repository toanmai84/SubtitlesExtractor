"""Adapter hiện thực :class:`ProjectRepositoryPort` bằng SQLite.

Lưu mỗi dự án Auto-Dubbing thành một hàng trong bảng ``projects``, khoá chính
là ``video_hash`` (định danh nội dung video). Dùng WAL để truy cập đồng thời
an toàn giữa các luồng GUI/worker.
"""

from __future__ import annotations

import contextlib
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from subtitles_extractor.domain.entities.project_record import (
    ProjectRecord,
    WorkflowStage,
)
from subtitles_extractor.domain.ports.project_repository_port import (
    ProjectRepositoryPort,
)

logger = logging.getLogger(__name__)

_COLUMNS = (
    "video_hash", "video_path", "video_name", "stage",
    "ocr_settings_json", "ocr_raw_json",
    "original_subtitle", "subtitle_format",
    "translated_subtitle", "target_lang", "translation_settings_json",
    "tts_audio_path", "tts_settings_json",
    "published_video_path",
    "created_at", "updated_at",
)

# [v3.23.316] Cột thêm sau khi CSDL đã phát hành -> phải NÂNG CẤP bảng cũ, nếu không
# người dùng đang có dự án sẽ lỗi "no such column". Dạng: (tên cột, kiểu SQL).
_ADDED_COLUMNS: tuple[tuple[str, str], ...] = (
    ("published_video_path", "TEXT"),
)


def _optional_column(row: sqlite3.Row, name: str) -> str:
    """Đọc một cột có thể CHƯA tồn tại ở CSDL cũ, trả chuỗi rỗng nếu thiếu."""
    try:
        return row[name] or ""
    except (IndexError, KeyError):
        return ""


class SqliteProjectRepository(ProjectRepositoryPort):
    """Kho dự án Auto-Dubbing trên SQLite, định danh theo hash video."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            self._db_path, timeout=10.0, check_same_thread=False
        )
        self._init_db()
        logger.debug("SqliteProjectRepository sẵn sàng tại %s", self._db_path)

    def _init_db(self) -> None:
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA synchronous=NORMAL;")
            with self._conn:
                self._conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS projects (
                        video_hash TEXT PRIMARY KEY,
                        video_path TEXT,
                        video_name TEXT,
                        stage INTEGER DEFAULT 0,
                        ocr_settings_json TEXT,
                        ocr_raw_json TEXT,
                        original_subtitle TEXT,
                        subtitle_format TEXT DEFAULT 'srt',
                        translated_subtitle TEXT,
                        target_lang TEXT,
                        translation_settings_json TEXT,
                        tts_audio_path TEXT,
                        tts_settings_json TEXT,
                        published_video_path TEXT,
                        created_at TEXT,
                        updated_at TEXT
                    )
                    """
                )
                self._migrate_add_missing_columns()

    def _migrate_add_missing_columns(self) -> None:
        """Thêm các cột mới vào CSDL đã tồn tại từ phiên bản trước.

        SQLite không có ``ADD COLUMN IF NOT EXISTS`` nên phải tự đọc ``PRAGMA
        table_info`` rồi mới thêm. Bỏ qua an toàn nếu cột đã có — hàm này chạy mỗi lần
        khởi động nên phải idempotent.
        """
        existing = {
            row[1] for row in self._conn.execute("PRAGMA table_info(projects)")
        }
        for column_name, column_type in _ADDED_COLUMNS:
            if column_name in existing:
                continue
            try:
                self._conn.execute(
                    f"ALTER TABLE projects ADD COLUMN {column_name} {column_type}"
                )
                logger.info("Đã nâng cấp CSDL dự án: thêm cột '%s'.", column_name)
            except sqlite3.OperationalError as exc:
                # Trường hợp hiếm: hai tiến trình cùng nâng cấp -> cột đã có.
                logger.debug("Bỏ qua thêm cột '%s': %s.", column_name, exc)

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> ProjectRecord:
        return ProjectRecord(
            video_hash=row["video_hash"],
            video_path=row["video_path"] or "",
            video_name=row["video_name"] or "",
            stage=WorkflowStage(row["stage"] or 0),
            ocr_settings_json=row["ocr_settings_json"] or "",
            ocr_raw_json=row["ocr_raw_json"] or "",
            original_subtitle=row["original_subtitle"] or "",
            subtitle_format=row["subtitle_format"] or "srt",
            translated_subtitle=row["translated_subtitle"] or "",
            target_lang=row["target_lang"] or "",
            translation_settings_json=row["translation_settings_json"] or "",
            tts_audio_path=row["tts_audio_path"] or "",
            tts_settings_json=row["tts_settings_json"] or "",
            published_video_path=_optional_column(row, "published_video_path"),
            created_at=row["created_at"] or "",
            updated_at=row["updated_at"] or "",
        )

    def get(self, video_hash: str) -> ProjectRecord | None:
        with self._lock:
            self._conn.row_factory = sqlite3.Row
            cur = self._conn.cursor()
            try:
                cur.execute(
                    "SELECT * FROM projects WHERE video_hash = ?", (video_hash,)
                )
                row = cur.fetchone()
                return self._row_to_record(row) if row else None
            finally:
                cur.close()

    def save(self, record: ProjectRecord) -> None:
        now = datetime.now(timezone.utc).isoformat()
        if not record.created_at:
            record.created_at = now
        record.updated_at = now
        # SQLite chỉ bind được kiểu cơ bản; ép mọi đường dẫn về str để tránh lỗi
        # "type 'WindowsPath' is not supported" (gặp khi record.video_path là Path).
        video_path = str(record.video_path) if record.video_path is not None else None
        tts_audio_path = (
            str(record.tts_audio_path) if record.tts_audio_path is not None else None
        )
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO projects (
                    video_hash, video_path, video_name, stage,
                    ocr_settings_json, ocr_raw_json,
                    original_subtitle, subtitle_format,
                    translated_subtitle, target_lang, translation_settings_json,
                    tts_audio_path, tts_settings_json, published_video_path,
                    created_at, updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(video_hash) DO UPDATE SET
                    video_path=excluded.video_path,
                    video_name=excluded.video_name,
                    stage=MAX(projects.stage, excluded.stage),
                    ocr_settings_json=excluded.ocr_settings_json,
                    ocr_raw_json=excluded.ocr_raw_json,
                    original_subtitle=excluded.original_subtitle,
                    subtitle_format=excluded.subtitle_format,
                    translated_subtitle=excluded.translated_subtitle,
                    target_lang=excluded.target_lang,
                    translation_settings_json=excluded.translation_settings_json,
                    tts_audio_path=excluded.tts_audio_path,
                    tts_settings_json=excluded.tts_settings_json,
                    published_video_path=excluded.published_video_path,
                    updated_at=excluded.updated_at
                """,
                (
                    record.video_hash, video_path, record.video_name,
                    int(record.stage),
                    record.ocr_settings_json, record.ocr_raw_json,
                    record.original_subtitle, record.subtitle_format,
                    record.translated_subtitle, record.target_lang,
                    record.translation_settings_json,
                    tts_audio_path, record.tts_settings_json,
                    record.published_video_path,
                    record.created_at, record.updated_at,
                ),
            )
        logger.debug(
            "Lưu dự án %s (%s) — khâu %s",
            record.video_hash, record.video_name, record.stage.name,
        )

    def list_all(self) -> list[ProjectRecord]:
        with self._lock:
            self._conn.row_factory = sqlite3.Row
            cur = self._conn.cursor()
            try:
                cur.execute("SELECT * FROM projects ORDER BY updated_at DESC")
                return [self._row_to_record(r) for r in cur.fetchall()]
            finally:
                cur.close()

    def delete(self, video_hash: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "DELETE FROM projects WHERE video_hash = ?", (video_hash,)
            )
        logger.debug("Đã xoá dự án %s", video_hash)

    def close(self) -> None:
        with self._lock:
            try:
                with contextlib.closing(self._conn.cursor()) as cur:
                    cur.execute("PRAGMA wal_checkpoint(TRUNCATE);")
                self._conn.close()
            except sqlite3.Error as exc:
                logger.warning("Lỗi khi đóng SQLite project repo: %s", exc)


__all__ = ["SqliteProjectRepository"]
