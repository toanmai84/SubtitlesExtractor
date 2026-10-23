"""Lưu trữ PHIÊN DỊCH (translation session) theo từng video, trong SQLite.

Mục tiêu (Bước 2 tái thiết khâu dịch): khi mở lại một video đã xử lý, ứng dụng nhận
biết qua khoá video và KHÔI PHỤC được mọi kết quả trung gian thay vì làm lại từ đầu:

* Kết quả PHÂN TÍCH ngữ cảnh (nhân vật / tóm tắt / ngôn ngữ).
* Bản DỊCH theo TỪNG GIAI ĐOẠN (preprocess → literal → style → localize).
* Danh sách FILE CLOUD đã upload (remote_name + khoảng thời gian + thời điểm upload),
  để tái dùng hoặc xoá khi cần.

Nguyên tắc nhận biết "còn dùng được":
* Mỗi thành phần lưu kèm HASH của ĐẦU VÀO tạo ra nó. Khi đầu vào đổi (vd phụ đề gốc
  được sửa) → hash khác → biết là cũ, cần làm lại. Việc so hash do tầng gọi quyết
  định; store chỉ lưu & trả lại trung thực.

Store này BỔ SUNG cho :class:`SqliteTranslationContextStore` (chỉ lưu ngữ cảnh).
Ở đây quản lý toàn bộ vòng đời phiên dịch một cách có cấu trúc (JSON trong cột TEXT).
"""

from __future__ import annotations

import contextlib
import json
import sqlite3

from subtitles_extractor.infrastructure.database.db_guard import prepare_database
import threading
from dataclasses import asdict, dataclass, field, replace
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class CloudVideoFile:
    """Một file video đã upload lên cloud (Gemini Files API)."""

    remote_name: str
    start_sec: float
    end_sec: float
    uploaded_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CloudVideoFile":
        return cls(
            remote_name=str(data.get("remote_name", "")),
            start_sec=float(data.get("start_sec", 0.0)),
            end_sec=float(data.get("end_sec", 0.0)),
            uploaded_at=str(data.get("uploaded_at", "")),
        )


CLOUD_FILE_MAX_AGE = timedelta(hours=48)
"""Gemini Files API tự xoá file sau 48 giờ — bản ghi cũ hơn chắc chắn đã không còn."""


def _uploaded_before(cloud_file: CloudVideoFile, cutoff: datetime) -> bool:
    """True nếu file được tải trước ``cutoff`` (không rõ thời điểm → False)."""
    if not cloud_file.uploaded_at:
        return False
    try:
        uploaded = datetime.fromisoformat(cloud_file.uploaded_at)
    except ValueError:
        return False
    if uploaded.tzinfo is None:
        uploaded = uploaded.replace(tzinfo=UTC)
    return uploaded < cutoff


def _earlier_iso(left: str, right: str) -> bool:
    """True nếu mốc ISO ``left`` sớm hơn ``right`` (thiếu múi giờ coi là UTC)."""
    try:
        left_dt = datetime.fromisoformat(left)
        right_dt = datetime.fromisoformat(right)
    except ValueError:
        return False
    if left_dt.tzinfo is None:
        left_dt = left_dt.replace(tzinfo=UTC)
    if right_dt.tzinfo is None:
        right_dt = right_dt.replace(tzinfo=UTC)
    return left_dt < right_dt


def merge_cloud_files(
    existing: Sequence[CloudVideoFile],
    uploaded: Iterable[CloudVideoFile],
    now: datetime,
    max_age: timedelta = CLOUD_FILE_MAX_AGE,
) -> list[CloudVideoFile]:
    """[v3.23.609] Thêm các file vừa tải vào danh sách đã ghi (hàm thuần).

    Trước đây mỗi lần phân tích GHI ĐÈ danh sách, nên file của lần trước (vẫn còn trên
    cloud tới 48 giờ) bị quên và nút "Xoá file cloud" không xoá được nữa.

    Args:
        existing: Danh sách đang lưu trong phiên.
        uploaded: Các file vừa tải.
        now: Thời điểm hiện tại (UTC).
        max_age: Bản ghi cũ hơn mức này bị bỏ (file đã bị Gemini xoá).

    Returns:
        Danh sách mới: bản ghi cũ còn hạn, rồi các file vừa tải; mỗi tên file một lần.

    [v3.23.615] File được TÁI DÙNG (đã có trên cloud) được ghi lại mỗi lần dùng; giữ mốc
    tải lên CŨ của nó, vì Gemini đếm 48 giờ từ lúc tải chứ không từ lúc dùng. Chẩn đoán
    thật 14/10: ``files/6iip7kcgg24f`` tải lúc 08:27 nhưng ghi ``uploaded_at`` 09:23 (lần
    tái dùng khi dịch) — bản ghi sống quá hạn thật gần một giờ.
    """
    cutoff = now - max_age
    first_seen = {
        cf.remote_name: cf.uploaded_at
        for cf in existing
        if cf.remote_name and cf.uploaded_at and not _uploaded_before(cf, cutoff)
    }
    fresh = [
        replace(cf, uploaded_at=first_seen[cf.remote_name])
        if cf.remote_name in first_seen
        and _earlier_iso(first_seen[cf.remote_name], cf.uploaded_at)
        else cf
        for cf in uploaded
        if cf.remote_name
    ]
    fresh_names = {cf.remote_name for cf in fresh}
    kept = [
        cf for cf in existing
        if cf.remote_name
        and cf.remote_name not in fresh_names
        and not _uploaded_before(cf, cutoff)
    ]
    merged: list[CloudVideoFile] = []
    seen: set[str] = set()
    for cf in [*kept, *fresh]:
        if cf.remote_name not in seen:
            seen.add(cf.remote_name)
            merged.append(cf)
    return merged


@dataclass(frozen=True)
class StageResult:
    """Kết quả dịch của MỘT giai đoạn.

    Attributes:
        stage_id: Định danh giai đoạn (vd 'preprocess', 'literal', 'style', 'localize').
        input_hash: Hash của đầu vào giai đoạn (để biết khi nào cần dịch lại).
        lines_json: Chuỗi JSON các dòng kết quả (index, text, speaker, description…).
        completed_at: Thời điểm hoàn tất (ISO-8601 UTC).
    """

    stage_id: str
    input_hash: str
    lines_json: str
    completed_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StageResult":
        return cls(
            stage_id=str(data.get("stage_id", "")),
            input_hash=str(data.get("input_hash", "")),
            lines_json=str(data.get("lines_json", "")),
            completed_at=str(data.get("completed_at", "")),
        )


@dataclass(frozen=True)
class TranslationSession:
    """Toàn bộ trạng thái có thể khôi phục của một phiên dịch cho một video."""

    video_key: str
    source_hash: str = ""
    analysis_characters: str = ""
    analysis_overview: str = ""
    analysis_glossary: str = ""
    analysis_visual_cues: str = ""
    analysis_source_lang: str = ""
    analysis_target_lang: str = ""
    analysis_input_hash: str = ""
    stages: tuple[StageResult, ...] = field(default_factory=tuple)
    cloud_files: tuple[CloudVideoFile, ...] = field(default_factory=tuple)
    updated_at: str = ""

    def stage(self, stage_id: str) -> StageResult | None:
        """Trả về kết quả của một giai đoạn theo id, hoặc None nếu chưa có."""
        for stage in self.stages:
            if stage.stage_id == stage_id:
                return stage
        return None

    def has_valid_analysis(self, current_input_hash: str) -> bool:
        """True nếu đã có phân tích VÀ khớp hash đầu vào hiện tại (dùng lại được)."""
        return bool(self.analysis_overview or self.analysis_characters) and (
            self.analysis_input_hash == current_input_hash
        )


class SqliteTranslationSessionStore:
    """CRUD phiên dịch theo khoá video, lưu trong SQLite (WAL, thread-safe)."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._lock: threading.RLock = threading.RLock()
        # [v3.23.649] Kiểm tra / sao lưu / tự phục hồi tệp trước khi kết nối.
        prepare_database(self._db_path, background=True)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with contextlib.suppress(sqlite3.Error):
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_db()

    def _init_db(self) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS translation_session (
                    video_key            TEXT PRIMARY KEY,
                    source_hash          TEXT DEFAULT '',
                    analysis_characters  TEXT DEFAULT '',
                    analysis_overview    TEXT DEFAULT '',
                    analysis_glossary    TEXT DEFAULT '',
                    analysis_visual_cues TEXT DEFAULT '',
                    analysis_source_lang TEXT DEFAULT '',
                    analysis_target_lang TEXT DEFAULT '',
                    analysis_input_hash  TEXT DEFAULT '',
                    stages_json          TEXT DEFAULT '[]',
                    cloud_files_json     TEXT DEFAULT '[]',
                    updated_at           TEXT
                )
                """
            )
            # [v3.23.27] Migration cho DB cũ chưa có cột analysis_glossary. ALTER TABLE
            # ADD COLUMN an toàn (idempotent qua kiểm cột hiện có) — giữ dữ liệu cũ.
            self._migrate_add_column_if_missing(
                "translation_session", "analysis_glossary", "TEXT DEFAULT ''"
            )
            self._migrate_add_column_if_missing(
                "translation_session", "analysis_visual_cues", "TEXT DEFAULT ''"
            )

    def _migrate_add_column_if_missing(
        self, table: str, column: str, decl: str
    ) -> None:
        """Thêm cột nếu DB cũ chưa có (an toàn, không mất dữ liệu)."""
        cursor = self._conn.execute(f"PRAGMA table_info({table})")
        existing = {row[1] for row in cursor.fetchall()}
        if column not in existing:
            with contextlib.suppress(sqlite3.Error):
                self._conn.execute(
                    f"ALTER TABLE {table} ADD COLUMN {column} {decl}"
                )

    @staticmethod
    def _normalize_key(video_key: str | Path) -> str:
        return str(video_key).strip()

    # ── Đọc / ghi toàn phiên ─────────────────────────────────────────────
    def get(self, video_key: str | Path) -> TranslationSession | None:
        """Đọc phiên dịch của một video, hoặc None nếu chưa có."""
        key = self._normalize_key(video_key)
        if not key:
            return None
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM translation_session WHERE video_key = ?", (key,)
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_session(row)

    def _row_to_session(self, row: sqlite3.Row) -> TranslationSession:
        stages = tuple(
            StageResult.from_dict(item)
            for item in self._safe_json_list(row["stages_json"])
        )
        cloud_files = tuple(
            CloudVideoFile.from_dict(item)
            for item in self._safe_json_list(row["cloud_files_json"])
        )
        return TranslationSession(
            video_key=row["video_key"],
            source_hash=row["source_hash"] or "",
            analysis_characters=row["analysis_characters"] or "",
            analysis_overview=row["analysis_overview"] or "",
            analysis_glossary=self._row_get(row, "analysis_glossary"),
            analysis_visual_cues=self._row_get(row, "analysis_visual_cues"),
            analysis_source_lang=row["analysis_source_lang"] or "",
            analysis_target_lang=row["analysis_target_lang"] or "",
            analysis_input_hash=row["analysis_input_hash"] or "",
            stages=stages,
            cloud_files=cloud_files,
            updated_at=row["updated_at"] or "",
        )

    @staticmethod
    def _row_get(row: sqlite3.Row, key: str) -> str:
        """Đọc cột an toàn (trả '' nếu DB cũ thiếu cột vì lý do nào đó)."""
        try:
            return row[key] or ""
        except (IndexError, KeyError):
            return ""

    @staticmethod
    def _safe_json_list(text: str | None) -> list[dict[str, Any]]:
        if not text:
            return []
        try:
            data = json.loads(text)
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, TypeError):
            return []

    def _ensure_row(self, key: str) -> None:
        """Đảm bảo có một hàng cho video_key (tạo rỗng nếu chưa có)."""
        self._conn.execute(
            "INSERT OR IGNORE INTO translation_session (video_key, updated_at) "
            "VALUES (?, ?)",
            (key, _utc_now_iso()),
        )

    # ── Cập nhật từng phần (giữ nguyên các phần khác) ─────────────────────
    def save_analysis(
        self,
        video_key: str | Path,
        *,
        characters: str,
        overview: str,
        source_lang: str,
        target_lang: str,
        input_hash: str,
        source_hash: str = "",
        glossary: str = "",
        visual_cues: str = "",
    ) -> None:
        """Lưu kết quả PHÂN TÍCH ngữ cảnh (kèm hash đầu vào để nhận biết hợp lệ)."""
        key = self._normalize_key(video_key)
        if not key:
            raise ValueError("video_key không được rỗng.")
        with self._lock, self._conn:
            self._ensure_row(key)
            self._conn.execute(
                """
                UPDATE translation_session SET
                    analysis_characters = ?,
                    analysis_overview = ?,
                    analysis_glossary = ?,
                    analysis_visual_cues = ?,
                    analysis_source_lang = ?,
                    analysis_target_lang = ?,
                    analysis_input_hash = ?,
                    source_hash = CASE WHEN ? != '' THEN ? ELSE source_hash END,
                    updated_at = ?
                WHERE video_key = ?
                """,
                (characters, overview, glossary, visual_cues, source_lang, target_lang,
                 input_hash, source_hash, source_hash, _utc_now_iso(), key),
            )

    def save_analysis_texts(
        self,
        video_key: str | Path,
        *,
        characters: str,
        overview: str,
        glossary: str,
        visual_cues: str,
    ) -> None:
        """[v3.23.567] Cập nhật PHẦN VĂN BẢN của phân tích, giữ hash/ngôn ngữ như cũ.

        Dùng khi ngữ cảnh đã lưu được sửa tất định (vd phiên âm tên sai âm Hán-Việt) —
        phân tích vẫn hợp lệ với phụ đề hiện tại nên không đụng ``analysis_input_hash``.

        Args:
            video_key: Đường dẫn/khoá video.
            characters: Bảng nhân vật.
            overview: Tóm tắt phim.
            glossary: Bảng thuật ngữ.
            visual_cues: Chuỗi JSON Visual Cues.

        Raises:
            ValueError: Nếu ``video_key`` rỗng.
        """
        key = self._normalize_key(video_key)
        if not key:
            raise ValueError("video_key không được rỗng.")
        with self._lock, self._conn:
            self._ensure_row(key)
            self._conn.execute(
                "UPDATE translation_session SET analysis_characters = ?, "
                "analysis_overview = ?, analysis_glossary = ?, analysis_visual_cues = ?, "
                "updated_at = ? WHERE video_key = ?",
                (characters, overview, glossary, visual_cues, _utc_now_iso(), key),
            )

    def save_visual_cues(self, video_key: str | Path, visual_cues: str) -> None:
        """[v3.23.559] Chỉ cập nhật Visual Cues của phân tích, giữ nguyên mọi phần khác.

        Dùng khi khâu dịch đã quét BỔ SUNG cues cho các dòng mà phân tích ngữ cảnh bỏ
        trống: lưu lại để lần dịch sau và file chuẩn đoán thấy đúng độ phủ thật.
        Không đụng ``analysis_input_hash`` nên phân tích vẫn hợp lệ như trước.

        Args:
            video_key: Đường dẫn/khoá video.
            visual_cues: Chuỗi JSON Visual Cues đã gộp.

        Raises:
            ValueError: Nếu ``video_key`` rỗng.
        """
        key = self._normalize_key(video_key)
        if not key:
            raise ValueError("video_key không được rỗng.")
        with self._lock, self._conn:
            self._ensure_row(key)
            self._conn.execute(
                "UPDATE translation_session SET analysis_visual_cues = ?, updated_at = ? "
                "WHERE video_key = ?",
                (visual_cues, _utc_now_iso(), key),
            )

    def save_stage(self, video_key: str | Path, stage: StageResult) -> None:
        """Lưu/cập nhật kết quả MỘT giai đoạn dịch (theo stage_id)."""
        key = self._normalize_key(video_key)
        if not key:
            raise ValueError("video_key không được rỗng.")
        completed = stage.completed_at or _utc_now_iso()
        stage = StageResult(stage.stage_id, stage.input_hash, stage.lines_json, completed)
        with self._lock, self._conn:
            self._ensure_row(key)
            cursor = self._conn.execute(
                "SELECT stages_json FROM translation_session WHERE video_key = ?", (key,)
            )
            row = cursor.fetchone()
            stages = self._safe_json_list(row["stages_json"] if row else None)
            stages = [s for s in stages if s.get("stage_id") != stage.stage_id]
            stages.append(stage.to_dict())
            self._conn.execute(
                "UPDATE translation_session SET stages_json = ?, updated_at = ? "
                "WHERE video_key = ?",
                (json.dumps(stages, ensure_ascii=False), _utc_now_iso(), key),
            )

    def save_cloud_files(
        self, video_key: str | Path, cloud_files: list[CloudVideoFile]
    ) -> None:
        """Lưu danh sách file cloud đã upload (thay thế toàn bộ danh sách cũ)."""
        key = self._normalize_key(video_key)
        if not key:
            raise ValueError("video_key không được rỗng.")
        payload = json.dumps([cf.to_dict() for cf in cloud_files], ensure_ascii=False)
        with self._lock, self._conn:
            self._ensure_row(key)
            self._conn.execute(
                "UPDATE translation_session SET cloud_files_json = ?, updated_at = ? "
                "WHERE video_key = ?",
                (payload, _utc_now_iso(), key),
            )

    # ── Xoá ──────────────────────────────────────────────────────────────
    def clear_stages(self, video_key: str | Path) -> None:
        """Xoá toàn bộ kết quả dịch theo giai đoạn (giữ phân tích & cloud files)."""
        key = self._normalize_key(video_key)
        if not key:
            return
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE translation_session SET stages_json = '[]', updated_at = ? "
                "WHERE video_key = ?",
                (_utc_now_iso(), key),
            )

    def clear_cloud_files(self, video_key: str | Path) -> None:
        """Xoá danh sách file cloud khỏi phiên (sau khi đã xoá thật trên cloud)."""
        key = self._normalize_key(video_key)
        if not key:
            return
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE translation_session SET cloud_files_json = '[]', updated_at = ? "
                "WHERE video_key = ?",
                (_utc_now_iso(), key),
            )

    def delete(self, video_key: str | Path) -> None:
        """Xoá toàn bộ phiên dịch của một video."""
        key = self._normalize_key(video_key)
        if not key:
            return
        with self._lock, self._conn:
            self._conn.execute(
                "DELETE FROM translation_session WHERE video_key = ?", (key,)
            )

    def close(self) -> None:
        with self._lock, contextlib.suppress(sqlite3.Error):
            self._conn.close()
