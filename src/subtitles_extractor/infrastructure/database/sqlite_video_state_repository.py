"""Adapter hiện thực :class:`VideoStateRepositoryPort` bằng SQLite."""

from __future__ import annotations

import contextlib
import json
import logging
import sqlite3
import threading
from pathlib import Path

from subtitles_extractor.domain.entities.video_state import VideoState
from subtitles_extractor.domain.ports.video_state_repository_port import (
    VideoStateRepositoryPort,
)
from subtitles_extractor.domain.value_objects.roi import (
    Roi,
    TextAlignment,
    TextOrientation,
)

logger = logging.getLogger(__name__)


class SqliteVideoStateRepository(VideoStateRepositoryPort):
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

        self._conn = sqlite3.connect(
            self._db_path,
            timeout=10.0,
            check_same_thread=False
        )
        self._init_db()

    def _init_db(self) -> None:
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA synchronous=NORMAL;")
            with self._conn:
                self._conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS video_states (
                        video_path TEXT PRIMARY KEY,
                        roi_json TEXT,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )

    def get(self, video_path: str) -> VideoState | None:
        video_path = str(video_path)  # [WindowsPath Fix] defensive: SQLite chỉ nhận str
        with self._lock:
            cursor = self._conn.cursor()
            try:
                cursor.execute(
                    "SELECT roi_json FROM video_states WHERE video_path = ?",
                    (video_path,)
                )
                row = cursor.fetchone()
                if row is None:
                    return None

                roi = self._deserialize_roi(row[0])
                return VideoState(video_path=video_path, roi=roi)
            finally:
                cursor.close()

    def save(self, state: VideoState) -> None:
        with self._lock, self._conn:
            roi_json = self._serialize_roi(state.roi)
            video_path = str(state.video_path)  # [WindowsPath Fix] defensive
            self._conn.execute(
                """
                    INSERT INTO video_states (video_path, roi_json, last_updated)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(video_path) DO UPDATE SET
                        roi_json=excluded.roi_json,
                        last_updated=CURRENT_TIMESTAMP
                    """,
                (video_path, roi_json)
            )

    def _serialize_roi(self, roi: Roi | None) -> str | None:
        if roi is None:
            return None
        data = {
            "x": roi.x,
            "y": roi.y,
            "width": roi.width,
            "height": roi.height,
            "alignment": roi.alignment.name,
            "orientation": roi.orientation.name
        }
        return json.dumps(data)

    def _deserialize_roi(self, roi_json: str | None) -> Roi | None:
        if not roi_json:
            return None
        try:
            data = json.loads(roi_json)
            # [v3.6 bugfix DB-1]: TextAlignment["UNKNOWN"] raises KeyError vì
            # "UNKNOWN" không phải tên enum hợp lệ. Trước đây bị catch bởi
            # `except KeyError` → trả None, làm mất ROI. Dùng fallback "CENTER".
            raw_alignment = data.get("alignment", "CENTER")
            try:
                alignment = TextAlignment[raw_alignment]
            except KeyError:
                logger.warning(
                    "ROI JSON có alignment không hợp lệ %r — dùng fallback CENTER.",
                    raw_alignment,
                )
                alignment = TextAlignment["CENTER"]

            raw_orientation = data.get("orientation", "HORIZONTAL")
            try:
                orientation = TextOrientation[raw_orientation]
            except KeyError:
                logger.warning(
                    "ROI JSON có orientation không hợp lệ %r — dùng fallback HORIZONTAL.",
                    raw_orientation,
                )
                orientation = TextOrientation["HORIZONTAL"]

            return Roi(
                x=int(data["x"]),
                y=int(data["y"]),
                width=int(data["width"]),
                height=int(data["height"]),
                alignment=alignment,
                orientation=orientation,
            )
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
            logger.warning("Không thể parse ROI JSON từ SQLite: %s", exc)
            return None

    def close(self) -> None:
        """[LỖI 2 TRÀN Ổ CỨNG FIX] Dọn dẹp file WAL."""
        with self._lock:
            try:
                with contextlib.closing(self._conn.cursor()) as cursor:
                    cursor.execute("PRAGMA wal_checkpoint(TRUNCATE);")
                self._conn.close()
            except sqlite3.Error as exc:
                logger.warning("Lỗi khi đóng SQLite video state repo: %s.", exc)

    def __del__(self) -> None:
        """Last-resort cleanup."""
        with contextlib.suppress(sqlite3.Error, AttributeError):
            with contextlib.closing(self._conn.cursor()) as cursor:
                cursor.execute("PRAGMA wal_checkpoint(TRUNCATE);")
            self._conn.close()


__all__ = ["SqliteVideoStateRepository"]
