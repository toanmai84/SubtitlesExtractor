"""[v3.23.610] Nạp bảng giá Gemini (``data/gemini_pricing.json``).

Thiếu hoặc hỏng tệp thì trả ``None`` kèm cảnh báo: chi phí ước tính chỉ là thông tin thêm,
không được làm hỏng lượt dịch hay phân tích.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from subtitles_extractor.application.services.gemini_pricing import (
    PricingFormatError,
    PricingTable,
    parse_pricing,
)

logger = logging.getLogger(__name__)

PRICING_FILENAME = "gemini_pricing.json"

__all__ = ["PRICING_FILENAME", "load_pricing_table"]


def load_pricing_table(path: Path) -> PricingTable | None:
    """Đọc bảng giá từ tệp JSON.

    Args:
        path: Đường dẫn ``gemini_pricing.json``.

    Returns:
        :class:`PricingTable`; ``None`` nếu tệp thiếu, hỏng hoặc sai cấu trúc.
    """
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.warning("Không có bảng giá Gemini (%s) — bỏ ước tính chi phí.", path)
        return None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        logger.warning("Không đọc được bảng giá Gemini (%s): %s", path, exc)
        return None
    try:
        return parse_pricing(payload)
    except PricingFormatError as exc:
        logger.warning("Bảng giá Gemini sai cấu trúc (%s): %s", path, exc)
        return None
