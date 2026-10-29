"""[v3.23.655] Giải mã khung ở luồng riêng trong lúc OCR chạy (đường ống sản xuất → hàng đợi → tiêu thụ).

Trước đây giải mã → lọc trùng → OCR chạy NỐI TIẾP trên một luồng: GPU OCR đứng chờ trong lúc CPU giải mã/lọc lô kế
tiếp và ngược lại. Ở đây bộ lặp khung (giải mã + cắt ROI + lọc trùng) chạy trong một luồng nền, đẩy khung vào hàng
đợi có giới hạn; luồng OCR lấy ra theo đúng thứ tự.

* Thứ tự khung giữ nguyên (một luồng sản xuất, hàng đợi FIFO).
* Lỗi của luồng giải mã được ném lại ở luồng OCR (không bị nuốt).
* Bên tiêu thụ dừng sớm (huỷ / lỗi OCR) → luồng giải mã được báo dừng, bộ lặp của nó được đóng (giải phóng bộ giải mã)
  và hàng đợi được xả để nó không kẹt ở ``put``.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable, Iterator
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

#: Số khung tối đa nằm chờ trong hàng đợi.
DEFAULT_PREFETCH = 256
#: Tổng dung lượng ảnh tối đa nằm chờ (ROI nhỏ ~200 KB/khung thì giới hạn số khung chạm trước; không ROI / ROI lớn
#: trên 1080p–4K ~6–25 MB/khung thì giới hạn này chạm trước — 256 khung có thể là 1,5–6 GB RAM).
DEFAULT_PREFETCH_BYTES = 256 * 1024 * 1024
_END = object()


class _Failure:
    __slots__ = ("error",)

    def __init__(self, error: BaseException) -> None:
        self.error = error


def _item_bytes(item: object) -> int:
    image = getattr(item, "image_rgb", None)
    return int(getattr(image, "nbytes", 0) or 0)


def prefetched(
    factory: Callable[[], Iterator[T]], maxsize: int = DEFAULT_PREFETCH,
    max_bytes: int = DEFAULT_PREFETCH_BYTES, size_of: Callable[[object], int] = _item_bytes,
) -> Iterator[T]:
    """Lặp ``factory()`` trong một luồng nền, trả từng phần tử theo đúng thứ tự.

    Args:
        factory: Hàm tạo bộ lặp (gọi TRONG luồng nền — bộ giải mã được tạo và dùng trên cùng một luồng).
        maxsize: Số phần tử tối đa chờ trong hàng đợi (``<= 0`` = không chạy nền, lặp trực tiếp).
        max_bytes: [v3.23.655] Tổng dung lượng tối đa chờ trong hàng đợi (theo ``size_of``); luôn cho ít nhất 1 phần tử.
    """
    if maxsize <= 0:
        yield from factory()
        return
    items: queue.Queue[object] = queue.Queue(maxsize=maxsize)
    stop = threading.Event()
    budget = threading.Condition()
    in_flight = [0]                      # tổng dung lượng các phần tử đang chờ

    def put(item: object) -> bool:
        size = size_of(item) if item is not _END and not isinstance(item, _Failure) else 0
        with budget:
            while size and in_flight[0] > 0 and in_flight[0] + size > max_bytes and not stop.is_set():
                budget.wait(timeout=0.1)
            if stop.is_set():
                return False
            in_flight[0] += size
        while not stop.is_set():
            try:
                items.put((item, size), timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def produce() -> None:
        iterator = None
        try:
            iterator = factory()
            for item in iterator:
                if not put(item):
                    break
        except BaseException as exc:  # noqa: BLE001 — chuyển lỗi sang luồng tiêu thụ
            put(_Failure(exc))
        finally:
            close = getattr(iterator, "close", None)
            if close is not None:
                try:
                    close()
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Đóng bộ lặp khung lỗi: %s", exc)
            put(_END)

    worker = threading.Thread(target=produce, name="frame-decoder", daemon=True)
    worker.start()
    try:
        while True:
            item, size = items.get()  # type: ignore[misc]
            if size:
                with budget:
                    in_flight[0] -= size
                    budget.notify_all()
            if item is _END:
                return
            if isinstance(item, _Failure):
                raise item.error
            yield item  # type: ignore[misc,unused-ignore]
    finally:
        stop.set()
        with budget:
            budget.notify_all()
        while True:  # xả để luồng giải mã không kẹt ở put()
            try:
                items.get_nowait()
            except queue.Empty:
                break
        worker.join(timeout=10)
        if worker.is_alive():
            logger.warning("Luồng giải mã khung chưa dừng sau 10 s — bỏ qua (luồng nền tự kết thúc).")


__all__ = ["DEFAULT_PREFETCH", "DEFAULT_PREFETCH_BYTES", "prefetched"]
