"""[v3.23.631] Sổ "model đã bỏ qua" của bước tự kiểm tra phân tích — nhớ giữa các lần chạy.

Khiếm khuyết mà model được hỏi vẫn không sửa (``self_address`` hợp lệ như độc thoại, nhãn cố
tình ngoài bảng…) mà cứ gửi lại mỗi lần là tốn lời gọi vô ích: log thật có hai lần tự kiểm tra
liền nhau đều mở đầu bằng "17 khiếm khuyết". Sổ này lưu, theo từng phim, số lần mỗi đơn vị
khiếm khuyết đã bị bỏ qua (xem ``analysis_audit.note_declined``). Ghi nguyên tử, giữ tối đa
:data:`MAX_FILMS` phim (cũ nhất bị bỏ trước); tệp hỏng coi như sổ trống.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

#: Số phim nhớ tối đa.
MAX_FILMS = 30
#: Trần số đơn vị mỗi phim (chống phình tệp).
MAX_UNITS_PER_FILM = 4000


class AuditLedgerStore:
    """Tệp JSON ``{dấu phim: {đơn vị khiếm khuyết: số lần bị bỏ qua}}``."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def _read(self) -> dict[str, dict[str, int]]:
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as exc:
            logger.warning("Sổ tự kiểm tra phân tích hỏng (%s) — bỏ qua: %s", self._path, exc)
            return {}
        if not isinstance(raw, dict):
            return {}
        films: dict[str, dict[str, int]] = {}
        for film, units in raw.items():
            if isinstance(film, str) and isinstance(units, dict):
                films[film] = {
                    str(k): int(v) for k, v in units.items()
                    if isinstance(v, int) and not isinstance(v, bool) and v > 0
                }
        return films

    def load(self, film_id: str) -> dict[str, int]:
        """Sổ của một phim (rỗng nếu chưa có)."""
        return dict(self._read().get(film_id, {}))

    def save(self, film_id: str, declined: dict[str, int]) -> bool:
        """Ghi sổ của một phim (đưa phim lên mới nhất). ``False`` nếu không ghi được."""
        films = self._read()
        films.pop(film_id, None)
        if declined:
            films[film_id] = dict(list(declined.items())[-MAX_UNITS_PER_FILM:])
        while len(films) > MAX_FILMS:
            films.pop(next(iter(films)))
        tmp = self._path.with_name(self._path.name + ".tmp")
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(films, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, self._path)
            return True
        except OSError as exc:
            logger.warning("Không ghi được sổ tự kiểm tra phân tích: %s", exc)
            return False


__all__ = ["MAX_FILMS", "MAX_UNITS_PER_FILM", "AuditLedgerStore"]
