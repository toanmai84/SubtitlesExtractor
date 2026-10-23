"""[v3.23.519] Ghi lại DIỄN BIẾN quá trình dịch để đưa vào file chuẩn đoán.

Vì sao cần: file chuẩn đoán trước đây chỉ có KẾT QUẢ CUỐI của từng giai đoạn. Toàn bộ diễn biến —
lô nào bị model trả sai, thử lại mấy lần, có phải chia nhỏ không, lô nào bị phát hiện lệch dòng,
lô nào cuối cùng đành giữ nguyên văn bản gốc — chỉ nằm trong nhật ký ứng dụng và mất khi đóng
chương trình. Không có dữ liệu đó thì mỗi lần chuẩn đoán lại phải xin thêm ``app_log.txt``, và
không thể đối chiếu "lô bị thử lại" với "dòng dịch kém" trong cùng một tệp.

Thiết kế: một bộ ghi nhỏ, an toàn với đa luồng (các lô dịch chạy song song). Bộ ghi có TRẦN số bản
ghi để không phình bộ nhớ ở phim dài, và luôn nuốt lỗi — ghi nhật ký không bao giờ được phép làm
hỏng việc dịch.
"""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Iterable, Mapping
from typing import Any

MAX_EVENTS = 2000
"""Trần số sự kiện giữ lại — đủ cho phim vài nghìn dòng mà không phình bộ nhớ."""


class TranslationProcessLog:
    """Bộ ghi diễn biến dịch dùng chung cho một lần chạy."""

    def __init__(self, max_events: int = MAX_EVENTS) -> None:
        """Khởi tạo bộ ghi rỗng.

        Args:
            max_events: Số sự kiện tối đa giữ lại.
        """
        self._lock = threading.Lock()
        self._events: list[dict[str, Any]] = []
        self._dropped = 0
        self._max_events = max_events
        self._owner = ""

    def clear(self) -> None:
        """Xoá sạch trước khi bắt đầu một lần dịch mới."""
        self.start_run("")

    def start_run(self, owner: str) -> None:
        """[v3.23.579] Xoá sạch và gắn nhật ký với lần dịch ``owner``.

        Args:
            owner: Dấu vân tay phụ đề nguồn của lần dịch (:func:`run_fingerprint`).
        """
        with self._lock:
            self._events.clear()
            self._dropped = 0
            self._owner = owner

    def record(self, event: str, **fields: Any) -> None:
        """Ghi một sự kiện. Không bao giờ ném lỗi ra ngoài.

        Args:
            event: Mã sự kiện ngắn, vd ``"batch_retry"``, ``"batch_misaligned"``.
            **fields: Thông tin kèm theo (chỉ số dòng, số lần thử, lý do…).
        """
        try:
            with self._lock:
                if len(self._events) >= self._max_events:
                    self._dropped += 1
                    return
                self._events.append({"event": event, **fields})
        except Exception:  # noqa: BLE001 — ghi nhật ký không được làm hỏng việc dịch
            pass

    def count(self) -> int:
        """[v3.23.592] Số sự kiện đang giữ — dùng làm MỐC cho :meth:`events_since`."""
        with self._lock:
            return len(self._events)

    def events_since(self, mark: int) -> list[dict[str, Any]]:
        """[v3.23.592] Bản sao các sự kiện ghi SAU mốc ``mark``.

        Args:
            mark: Giá trị :meth:`count` lấy trước đó.

        Returns:
            Các sự kiện từ vị trí ``mark`` trở đi (rỗng nếu mốc vượt quá số sự kiện).
        """
        with self._lock:
            return [dict(item) for item in self._events[max(0, mark):]]

    def replay(self, events: Iterable[Mapping[str, Any]]) -> None:
        """[v3.23.592] Ghi lại sự kiện của một lần chạy TRƯỚC, đánh dấu nguồn gốc.

        Lượt dịch chạy tiếp từ checkpoint bỏ qua các giai đoạn đã xong, nên sự kiện của
        chúng (lưới số, lưới dải lệch, mốc giai đoạn…) không được sinh ra lần nữa. Phát
        lại từ checkpoint để file chuẩn đoán vẫn kể đủ cả lượt dịch; mỗi sự kiện mang
        ``from_checkpoint: True`` để người đọc biết nó thuộc lần chạy trước.

        Args:
            events: Các sự kiện đã lưu (mục không phải dict hoặc thiếu mã bị bỏ qua).
        """
        for item in events:
            if not isinstance(item, Mapping) or not item.get("event"):
                continue
            fields = {
                str(key): value for key, value in item.items()
                if key not in ("event", "from_checkpoint")
            }
            self.record(str(item["event"]), **fields, from_checkpoint=True)

    def snapshot(self) -> dict[str, Any]:
        """Trả bản sao dữ liệu đã ghi, kèm thống kê gọn theo loại sự kiện.

        Returns:
            Dict gồm ``events`` (danh sách), ``counts`` (đếm theo loại), ``dropped`` và —
            khi nhật ký đã gắn với một lần dịch — ``owner`` (dấu vân tay phụ đề).
        """
        with self._lock:
            events = list(self._events)
            dropped = self._dropped
            owner = self._owner
        counts: dict[str, int] = {}
        for item in events:
            name = str(item.get("event", ""))
            counts[name] = counts.get(name, 0) + 1
        snapshot: dict[str, Any] = {
            "events": events, "counts": counts, "dropped": dropped,
        }
        if owner:
            snapshot["owner"] = owner
        return snapshot


def run_fingerprint(source_texts: Iterable[str]) -> str:
    """[v3.23.579] Dấu vân tay phụ đề nguồn của một lần dịch (hàm thuần).

    Nhật ký là bộ ghi DÙNG CHUNG trong bộ nhớ: dịch phim A rồi mở lại phim B đã dịch từ
    trước và xuất chuẩn đoán thì file của B mang diễn biến của A (ca thật: file 3902
    dòng chứa ``stage_completed`` 5540 dòng và các lô 3421-3480 của phim khác). Dấu vân
    tay giúp lúc xuất biết nhật ký có thuộc đúng phụ đề đang xuất không.

    Args:
        source_texts: Văn bản các dòng phụ đề nguồn theo thứ tự.

    Returns:
        Chuỗi ``"<số dòng>:<16 ký tự sha256>"``.
    """
    digest = hashlib.sha256()
    count = 0
    for text in source_texts:
        digest.update((text or "").encode("utf-8"))
        digest.update(b"\x1f")
        count += 1
    return f"{count}:{digest.hexdigest()[:16]}"


_PROCESS_LOG = TranslationProcessLog()


def get_process_log() -> TranslationProcessLog:
    """Bộ ghi dùng chung cho lần dịch hiện tại."""
    return _PROCESS_LOG


def record_event(event: str, **fields: Any) -> None:
    """Tiện ích ghi nhanh vào bộ ghi dùng chung."""
    _PROCESS_LOG.record(event, **fields)
