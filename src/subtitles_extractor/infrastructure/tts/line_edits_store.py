"""[v3.23.654] Kho sửa tay từng dòng của trang TTS bằng TỆP (thay registry Windows), có lịch sử phiên bản.

Trước đây mọi sửa tay (người nói, lời, thêm/xoá dòng) của TẤT CẢ phụ đề nằm chung một khoá QSettings — trên Windows là
registry: phình ~14,7 MB, mỗi lần sửa ghi lại cả khối (~0,46 s trên luồng giao diện), và không có lịch sử nên xoá nhầm
rồi đóng app là mất đường lùi.

Nay mỗi phụ đề (theo dấu phụ đề) một tệp ``<thư mục>/<dấu>.json``::

    {"current": <chỉnh sửa hiện tại>, "history": [{"at": "...", "label": "...", "edits": {...}}, ...]}

* Ghi an toàn (tệp tạm + đổi tên).
* Lịch sử: phiên bản cũ được cất khi danh sách dòng đổi (thêm/xoá/gộp/tách) hoặc sau ≥ :data:`HISTORY_MIN_GAP_SEC` giây;
  giữ :data:`MAX_HISTORY` bản gần nhất.
* Giữ tối đa :data:`MAX_FILES` phụ đề gần nhất (tệp cũ nhất bị xoá).
* Lần đầu chạy: chuyển dữ liệu cũ từ QSettings sang (xem :meth:`LineEditsFileStore.import_legacy`).
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from subtitles_extractor.infrastructure.subtitle.atomic_save import atomic_write_text

logger = logging.getLogger(__name__)

MAX_HISTORY = 20
HISTORY_MIN_GAP_SEC = 30.0
MAX_FILES = 60
_SAFE_NAME = re.compile(r"^[0-9a-zA-Z_-]{1,64}$")


@dataclass(frozen=True, slots=True)
class HistoryEntry:
    """Một phiên bản cũ của chỉnh sửa (để "🕘 Lịch sử…")."""

    at: str
    label: str
    edits: dict


def describe_edits(encoded: Mapping[str, object] | None) -> str:
    """Tóm tắt một bản chỉnh sửa đã mã hoá: số dòng, người nói/lời/người nghe sửa tay."""
    if not encoded:
        return "không có sửa tay"
    parts: list[str] = []
    lines = encoded.get("lines")
    if isinstance(lines, list):
        added = sum(1 for line in lines if isinstance(line, list | tuple) and line and line[0] == -1)
        parts.append(f"{len(lines)} dòng" + (f" (+{added} thêm tay)" if added else ""))
    for key, label in (("speakers", "người nói"), ("texts", "lời"), ("addressees", "nói với")):
        value = encoded.get(key)
        if isinstance(value, Mapping) and value:
            parts.append(f"{len(value)} {label}")
    return ", ".join(parts) or "không có sửa tay"


class LineEditsFileStore:
    """Kho tệp sửa tay theo dấu phụ đề."""

    def __init__(self, directory: Path, clock=time.time) -> None:
        self._dir = Path(directory)
        self._clock = clock
        self._last_push: dict[str, float] = {}

    @property
    def directory(self) -> Path:
        return self._dir

    def _path(self, fingerprint: str) -> Path | None:
        if not fingerprint or not _SAFE_NAME.match(fingerprint):
            return None
        return self._dir / f"{fingerprint}.json"

    def _read(self, fingerprint: str) -> dict:
        path = self._path(fingerprint)
        if path is None or not path.is_file():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("Tệp sửa tay TTS %s hỏng — bỏ qua: %s", path.name, exc)
            return {}
        return data if isinstance(data, dict) else {}

    def load(self, fingerprint: str) -> dict | None:
        """Chỉnh sửa hiện tại (dạng mã hoá) của phụ đề; ``None`` nếu chưa có."""
        current = self._read(fingerprint).get("current")
        return current if isinstance(current, dict) and current else None

    def history(self, fingerprint: str) -> list[HistoryEntry]:
        """Các phiên bản cũ, MỚI NHẤT trước."""
        entries: list[HistoryEntry] = []
        for raw in reversed(self._read(fingerprint).get("history") or []):
            if isinstance(raw, dict) and isinstance(raw.get("edits"), dict):
                entries.append(HistoryEntry(str(raw.get("at") or ""), str(raw.get("label") or ""), raw["edits"]))
        return entries

    def save(self, fingerprint: str, encoded: dict | None) -> bool:
        """Ghi chỉnh sửa hiện tại; cất bản cũ vào lịch sử khi cần. ``True`` nếu có ghi.

        Args:
            fingerprint: Dấu phụ đề.
            encoded: Chỉnh sửa đã mã hoá (``None``/rỗng = không còn sửa tay — lịch sử vẫn giữ).
        """
        path = self._path(fingerprint)
        if path is None:
            return False
        data = self._read(fingerprint)
        old = data.get("current") or None
        new = encoded or None
        if old == new:
            return False
        history = [entry for entry in data.get("history") or [] if isinstance(entry, dict)]
        now = self._clock()
        structural = _line_count(old) != _line_count(new)
        # Xoá hết sửa tay / bớt nhiều sửa tay: luôn cất bản cũ (đúng ca "lỡ tay rồi đóng app" mà lịch sử phải cứu).
        shrinking = new is None or _edit_count(new) < _edit_count(old)
        if old and (structural or shrinking or now - self._last_push.get(fingerprint, 0.0) >= HISTORY_MIN_GAP_SEC):
            history.append({
                "at": datetime.fromtimestamp(now).isoformat(timespec="seconds"),
                "label": describe_edits(old),
                "edits": old,
            })
            self._last_push[fingerprint] = now
        payload = {"current": new, "history": history[-MAX_HISTORY:]}
        try:
            atomic_write_text(path, json.dumps(payload, ensure_ascii=False), fsync_timeout_sec=0)
        except OSError as exc:
            logger.warning("Không lưu được sửa tay TTS (%s): %s", path, exc)
            return False
        self._prune()
        return True

    def _prune(self) -> None:
        try:
            files = sorted(self._dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
        except OSError:
            return
        for stale in files[:-MAX_FILES] if len(files) > MAX_FILES else ():
            try:
                stale.unlink()
            except OSError:
                pass

    def import_legacy(self, store: Mapping[str, object]) -> tuple[int, int]:
        """Chuyển kho cũ ``{dấu: chỉnh sửa}`` (QSettings) sang tệp; bỏ qua dấu đã có tệp.

        Returns:
            ``(số phụ đề đã chuyển, số phụ đề KHÔNG ghi được)`` — bên gọi chỉ xoá kho cũ khi số sau bằng 0.
        """
        moved = failed = 0
        for fingerprint, encoded in store.items():
            path = self._path(str(fingerprint))
            if path is None or path.exists() or not isinstance(encoded, dict) or not encoded:
                continue
            try:
                atomic_write_text(path, json.dumps({"current": encoded, "history": []}, ensure_ascii=False),
                                  fsync_timeout_sec=0)
                moved += 1
            except OSError as exc:
                failed += 1
                logger.warning("Không chuyển được sửa tay %s sang tệp: %s", fingerprint, exc)
        return moved, failed


def _edit_count(encoded: object) -> int:
    if not isinstance(encoded, dict):
        return 0
    return sum(len(value) for key, value in encoded.items() if key != "lines" and isinstance(value, Mapping))


def _line_count(encoded: object) -> int | None:
    if isinstance(encoded, dict) and isinstance(encoded.get("lines"), list):
        return len(encoded["lines"])
    return None


__all__ = ["HISTORY_MIN_GAP_SEC", "MAX_HISTORY", "HistoryEntry", "LineEditsFileStore", "describe_edits"]
