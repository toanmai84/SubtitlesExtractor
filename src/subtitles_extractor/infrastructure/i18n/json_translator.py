"""Adapter dịch chuỗi UI từ các tệp JSON đặt trong thư mục ``data``.

Cấu trúc tệp::

    {
      "_meta": {"language": "vi", "language_name": "Tiếng Việt"},
      "app":   {"title": "..."},
      "nav":   {"extract": "...", "settings": "..."},
      ...
    }

Khoá dạng phân cấp ``"editor.btn_save"`` → ``data["editor"]["btn_save"]``.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class JsonTranslator:
    """Hiện thực :class:`TranslatorPort` bằng các tệp JSON.

    Args:
        data_dir:        Thư mục chứa ``strings_<locale>.json``.
        default_locale:  Locale mặc định khi khởi tạo.
        fallback_locale: Locale dùng khi key thiếu ở locale hiện tại.
    """

    def __init__(
        self,
        data_dir: Path,
        default_locale: str = "vi",
        fallback_locale: str = "vi",
    ) -> None:
        self._data_dir = data_dir
        self._fallback_locale = fallback_locale
        self._lock = threading.Lock()
        self._cached_strings: dict[str, dict[str, Any]] = {}
        self._current_locale = default_locale
        self._ensure_loaded(default_locale)

    # ── Port API ─────────────────────────────────────────────────────────

    @property
    def locale(self) -> str:
        return self._current_locale

    def set_locale(self, locale: str) -> None:
        with self._lock:
            self._ensure_loaded(locale)
            self._current_locale = locale
            logger.info("Đã đổi ngôn ngữ giao diện sang %r.", locale)

    def translate(self, key: str, **placeholders: object) -> str:
        result = self._lookup(self._current_locale, key)
        if result is None and self._current_locale != self._fallback_locale:
            result = self._lookup(self._fallback_locale, key)
        if result is None:
            return key
        if placeholders:
            try:
                return result.format(**placeholders)
            except (KeyError, IndexError, ValueError):
                return result
        return result

    def available_locales(self) -> list[str]:
        """Quét thư mục dữ liệu để liệt kê các locale khả dụng."""
        if not self._data_dir.exists():
            return []
        result = []
        for path in self._data_dir.glob("strings_*.json"):
            stem = path.stem  # "strings_vi"
            if stem.startswith("strings_"):
                result.append(stem[len("strings_"):])
        return sorted(result)

    # ── Helpers nội bộ ──────────────────────────────────────────────────

    def _ensure_loaded(self, locale: str) -> None:
        if locale in self._cached_strings:
            return
        path = self._data_dir / f"strings_{locale}.json"
        if not path.exists():
            logger.warning("Không tìm thấy tệp ngôn ngữ %s — bỏ qua.", path)
            self._cached_strings[locale] = {}
            return
        try:
            with path.open("r", encoding="utf-8") as fh:
                self._cached_strings[locale] = json.load(fh)
            logger.debug("Đã nạp ngôn ngữ %r từ %s.", locale, path)
        except (OSError, json.JSONDecodeError) as exc:
            logger.exception("Lỗi đọc tệp %s: %s", path, exc)
            self._cached_strings[locale] = {}

    def _lookup(self, locale: str, key: str) -> str | None:
        """Tra cứu key phân cấp. Trả ``None`` nếu không tìm thấy."""
        node: Any = self._cached_strings.get(locale, {})
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return None
            node = node[part]
        return node if isinstance(node, str) else None


__all__ = ["JsonTranslator"]
