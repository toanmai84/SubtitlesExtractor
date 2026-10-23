"""[v3.23.567] Nạp bảng âm Hán-Việt theo chữ (``data/hanviet_readings.json``).

File do ``tools/build_hanviet_readings.py`` dựng. Thiếu/hỏng file thì trả bảng RỖNG kèm
cảnh báo: bộ soát phiên âm tên chỉ là lưới an toàn, không được làm hỏng phân tích/dịch.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from subtitles_extractor.application.services.name_reading_check import NameReadings

logger = logging.getLogger(__name__)

HANVIET_READINGS_FILENAME = "hanviet_readings.json"

_SCHEMA = "subtitles_extractor.hanviet_readings/1"


def load_name_readings(path: Path) -> NameReadings:
    """Đọc bảng âm Hán-Việt từ file JSON.

    Args:
        path: Đường dẫn ``hanviet_readings.json``.

    Returns:
        Bảng âm; rỗng nếu file thiếu, hỏng hoặc sai schema.
    """
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.warning("Không có bảng âm Hán-Việt (%s) — bỏ qua soát phiên âm tên.", path)
        return NameReadings()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        logger.warning("Không đọc được bảng âm Hán-Việt (%s): %s", path, exc)
        return NameReadings()
    if not isinstance(payload, dict) or payload.get("schema") != _SCHEMA:
        logger.warning("Bảng âm Hán-Việt sai định dạng (%s) — bỏ qua.", path)
        return NameReadings()
    encoded = payload.get("readings")
    if not isinstance(encoded, dict):
        return NameReadings()
    readings = NameReadings.from_encoded(
        {str(char): str(value) for char, value in encoded.items()}
    )
    logger.debug("Đã nạp âm Hán-Việt cho %d chữ.", len(encoded))
    return readings


__all__ = ["HANVIET_READINGS_FILENAME", "load_name_readings"]
