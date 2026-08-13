"""Adapter hiện thực SubtitleRepositoryPort bằng SQLite.

Lưu trữ danh sách SubtitleEvent dưới dạng JSON để bảo toàn 100% siêu dữ liệu.
TỐI ƯU HÓA: Sử dụng Persistent Connection, WAL mode và Contextlib an toàn.
"""

from __future__ import annotations

import contextlib
import json
import logging
import sqlite3
import threading
import uuid
from pathlib import Path
from typing import Any

from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.ports.subtitle_repository_port import (
    SubtitleRepositoryPort,
)
from subtitles_extractor.domain.value_objects.confidence import Confidence
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval

logger = logging.getLogger(__name__)


class SqliteSubtitleRepository(SubtitleRepositoryPort):
    def __init__(self, db_path: Path) -> None:
        self._db_path: Path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock: threading.RLock = threading.RLock()

        # Persistent Connection
        self._conn: sqlite3.Connection = sqlite3.connect(
            self._db_path,
            timeout=10.0,
            check_same_thread=False
        )
        self._init_db()

    def _init_db(self) -> None:
        """Khởi tạo cấu trúc Database với cấu hình tối ưu I/O."""
        with self._lock:
            with contextlib.closing(self._conn.cursor()) as cursor:
                cursor.execute("PRAGMA journal_mode=WAL;")
                cursor.execute("PRAGMA synchronous=NORMAL;")
                cursor.execute("PRAGMA cache_size=-64000;")

            with self._conn:
                self._conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS video_subtitles (
                        video_path TEXT PRIMARY KEY,
                        events_json TEXT,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )

    def has_events(self, video_path: str) -> bool:
        """Kiểm tra nhanh xem video này đã có dữ liệu phụ đề trong DB chưa."""
        video_path = str(video_path)  # [WindowsPath Fix] defensive
        with self._lock, contextlib.closing(self._conn.cursor()) as cursor:
            cursor.execute(
                "SELECT 1 FROM video_subtitles WHERE video_path = ?",
                (video_path,)
            )
            return cursor.fetchone() is not None

    def load_events(self, video_path: str) -> list[SubtitleEvent] | None:
        """Đọc danh sách phụ đề đã lưu."""
        video_path = str(video_path)  # [WindowsPath Fix] defensive
        with self._lock, contextlib.closing(self._conn.cursor()) as cursor:
            cursor.execute(
                "SELECT events_json FROM video_subtitles WHERE video_path = ?",
                (video_path,)
            )
            row = cursor.fetchone()
            if row is None or not row[0]:
                return None
            return self._deserialize_events(str(row[0]))

    def save_events(self, video_path: str, events: list[SubtitleEvent]) -> None:
        """Lưu toàn bộ danh sách phụ đề (Ghi đè bản cũ)."""
        video_path = str(video_path)  # [WindowsPath Fix] defensive
        with self._lock, self._conn:
            json_str: str = self._serialize_events(events)
            self._conn.execute(
                """
                    INSERT INTO video_subtitles (video_path, events_json, last_updated)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(video_path) DO UPDATE SET
                        events_json=excluded.events_json,
                        last_updated=CURRENT_TIMESTAMP
                    """,
                (video_path, json_str)
            )

    def _serialize_events(self, events: list[SubtitleEvent]) -> str:
        """Chuyển đổi danh sách SubtitleEvent sang JSON string tốc độ cao."""
        data: list[dict[str, Any]] = [
            {
                "index": e.index,
                "text": e.text,
                "start_sec": e.start_sec,
                "end_sec": e.end_sec,
                "confidence": float(e.confidence),
                "frame_count": e.frame_count,
                "position": e.position,
                "bounding_box": e.bounding_box,
                "uid": e.uid
            }
            for e in events
        ]
        return json.dumps(data, ensure_ascii=False, separators=(",", ":"))

    def _deserialize_events(self, json_str: str) -> list[SubtitleEvent]:
        """Phục hồi danh sách SubtitleEvent từ JSON string."""
        try:
            data: list[dict[str, Any]] = json.loads(json_str)
            events: list[SubtitleEvent] = []

            for d in data:
                uid_str: str = str(d.get("uid", "")).strip()
                if not uid_str:
                    uid_str = str(uuid.uuid4())

                position = tuple(d["position"]) if d.get("position") else None
                bounding_box = tuple(d["bounding_box"]) if d.get("bounding_box") else None

                events.append(SubtitleEvent(
                    index=int(d["index"]),
                    text=str(d["text"]),
                    interval=TimeInterval(float(d["start_sec"]), float(d["end_sec"])),
                    confidence=Confidence(float(d["confidence"])),
                    frame_count=int(d.get("frame_count", 0)),
                    position=position,  # type: ignore
                    bounding_box=bounding_box,  # type: ignore
                    uid=uid_str
                ))
            return events
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
            logger.warning("Lỗi phân tích JSON Subtitle từ SQLite: %s", exc)
            return []

    def close(self) -> None:
        """[v3.20.3 #2] Dọn WAL khi đóng App — dùng PASSIVE thay TRUNCATE.

        TRUNCATE đòi khoá độc quyền (writer lock) → nếu còn luồng ghi ngầm (auto-save
        background) sẽ ném ``database is locked``. PASSIVE cho SQLite tự dọn WAL ở
        mức tốt nhất có thể mà KHÔNG xung đột với luồng đang ghi.
        """
        with self._lock:
            try:
                with contextlib.closing(self._conn.cursor()) as cursor:
                    cursor.execute("PRAGMA wal_checkpoint(PASSIVE);")
                self._conn.close()
            except sqlite3.Error as exc:
                logger.warning("Lỗi khi đóng SQLite subtitle repo: %s.", exc)

    def __del__(self) -> None:
        """Last-resort cleanup — phải suppress toàn bộ exception vì __del__ có thể
        được gọi trong quá trình interpreter shutdown khi các object đã bị destroy."""
        with contextlib.suppress(Exception):
            # Không dùng self._lock trong __del__ — lock object có thể đã bị GC.
            # Chỉ cần PRAGMA + close là đủ.
            if hasattr(self, "_conn") and self._conn is not None:
                with contextlib.closing(self._conn.cursor()) as cur:
                    cur.execute("PRAGMA wal_checkpoint(TRUNCATE);")
                self._conn.close()


__all__ = ["SqliteSubtitleRepository"]
