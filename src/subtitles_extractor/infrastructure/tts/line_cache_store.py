"""[v3.23.630] Lưu / đọc bộ nhớ đệm TTS từng dòng (``line_cache.json`` cạnh các segment).

Cache nằm TRONG thư mục segment (``<output>_segments/line_cache.json``) nên đi cùng audio dòng:
xoá thư mục segment là mất cache, đúng ý. Ghi nguyên tử (file tạm rồi thay) — mất điện giữa
chừng không để lại JSON dở dang; đọc hỏng thì coi như chưa có cache (chạy cả phim).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, fields
from pathlib import Path

from subtitles_extractor.application.services.tts_line_cache import (
    CACHE_VERSION,
    CacheEntry,
    LineCache,
)
from subtitles_extractor.domain.ports.subtitle_tts_port import TTSSegmentResult

logger = logging.getLogger(__name__)

CACHE_FILENAME = "line_cache.json"
_RESULT_FIELDS = frozenset(f.name for f in fields(TTSSegmentResult))


def cache_path(segments_dir: Path) -> Path:
    """Đường dẫn file cache trong thư mục segment."""
    return segments_dir / CACHE_FILENAME


def audio_ok(path: str) -> bool:
    """File audio dòng còn tồn tại và không rỗng."""
    try:
        return os.path.getsize(path) > 44  # nhỏ hơn đầu WAV = hỏng
    except OSError:
        return False


def save_cache(path: Path, cache: LineCache) -> bool:
    """Ghi cache nguyên tử. ``False`` (kèm log) nếu không ghi được — không chặn TTS chính."""
    payload = {
        "version": CACHE_VERSION,
        "global_key": cache.global_key,
        "sample_rate": cache.sample_rate,
        "complete": cache.complete,
        "entries": {
            str(position): {"sig": entry.signature, "result": asdict(entry.result)}
            for position, entry in sorted(cache.entries.items())
        },
    }
    tmp = path.with_name(path.name + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
        return True
    except OSError as exc:
        logger.warning("Không ghi được cache TTS từng dòng (%s): %s", path, exc)
        return False


def load_cache(path: Path) -> LineCache | None:
    """Đọc cache; ``None`` nếu chưa có, hỏng, hoặc khác phiên bản định dạng."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        logger.warning("Cache TTS từng dòng hỏng (%s) — bỏ qua: %s", path, exc)
        return None
    if not isinstance(raw, dict) or raw.get("version") != CACHE_VERSION:
        return None
    entries: dict[int, CacheEntry] = {}
    try:
        for key, value in raw.get("entries", {}).items():
            data = {k: v for k, v in value["result"].items() if k in _RESULT_FIELDS}
            entries[int(key)] = CacheEntry(str(value["sig"]), TTSSegmentResult(**data))
        return LineCache(
            str(raw["global_key"]), int(raw.get("sample_rate", 0)), entries,
            bool(raw.get("complete", True)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning("Cache TTS từng dòng sai cấu trúc (%s) — bỏ qua: %s", path, exc)
        return None


def discard_cache(path: Path) -> None:
    """Xoá cache (trước khi chạy cả phim: audio dòng sắp bị ghi đè nên cache cũ hết đúng)."""
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        logger.debug("Không xoá được cache TTS từng dòng: %s", exc)


__all__ = [
    "CACHE_FILENAME", "audio_ok", "cache_path", "discard_cache", "load_cache",
    "save_cache",
]
