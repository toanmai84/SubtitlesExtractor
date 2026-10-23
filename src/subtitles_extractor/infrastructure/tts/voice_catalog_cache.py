"""[v3.23.647] Bộ nhớ đệm trên đĩa cho danh mục giọng preset của VieNeu.

Log thật: mở trang Dịch/TTS khiến app nạp VieNeu-TTS (ONNX/CPU) trong tiến trình chính **36 giây** (09:53:52 →
09:54:28) chỉ để đọc danh sách 25 giọng — kể cả khi chỉ dùng GPU hoặc chưa hề tổng hợp. Danh sách này chỉ đổi khi
đổi chế độ engine hoặc nâng bản ``vieneu``, nên lưu lại sau lần đọc đầu và đọc từ đĩa các lần sau.

Ghi nguyên tử; mọi lỗi đọc/ghi được coi là "chưa có cache" (không bao giờ chặn giao diện).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

CACHE_FILENAME = "vieneu_voice_catalog.json"
_FORMAT = 1


def default_cache_path() -> Path | None:
    """Tệp cache cạnh kho model của ứng dụng; ``None`` nếu không có chỗ ghi."""
    try:
        from subtitles_extractor.infrastructure.model_store import ensure_model_store_root

        return ensure_model_store_root() / CACHE_FILENAME
    except Exception as exc:  # noqa: BLE001 - không có kho model = không cache, không phải lỗi
        logger.debug("Không xác định được thư mục cache danh mục giọng: %s", exc)
        return None


def load_catalog(path: Path | None, mode: str, vieneu_version: str | None) -> list[tuple[str, str]] | None:
    """Danh mục ``(nhãn, ID)`` đã lưu cho ``mode`` và đúng bản ``vieneu``; ``None`` nếu không dùng được."""
    if path is None or not vieneu_version:
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        entry = raw["modes"][mode]
        if raw.get("format") != _FORMAT or entry["vieneu"] != vieneu_version:
            return None
        pairs = [(str(label), str(voice_id)) for label, voice_id in entry["voices"]]
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None
    return pairs or None


def save_catalog(
    path: Path | None, mode: str, vieneu_version: str | None, pairs: list[tuple[str, str]],
) -> bool:
    """Lưu danh mục của ``mode`` (giữ nguyên các chế độ khác). ``False`` nếu không ghi được."""
    if path is None or not vieneu_version or not pairs:
        return False
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        modes = raw["modes"] if raw.get("format") == _FORMAT and isinstance(raw.get("modes"), dict) else {}
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        modes = {}
    modes[mode] = {"vieneu": vieneu_version, "voices": [[label, voice_id] for label, voice_id in pairs]}
    tmp = path.with_name(path.name + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps({"format": _FORMAT, "modes": modes}, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
        return True
    except OSError as exc:
        logger.debug("Không ghi được cache danh mục giọng: %s", exc)
        return False


__all__ = ["CACHE_FILENAME", "default_cache_path", "load_catalog", "save_catalog"]
