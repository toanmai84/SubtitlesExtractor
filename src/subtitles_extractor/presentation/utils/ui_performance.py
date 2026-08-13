"""Tiện ích hiệu năng giao diện: tiết lưu (throttle) và cập nhật theo lô.

Phần *quyết định* được tách thuần (không phụ thuộc Qt) để kiểm thử dễ dàng bằng
cách tiêm đồng hồ (clock). Widget chỉ việc gọi :meth:`IntervalThrottle.should_run`.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterator, Protocol


class _SupportsUpdatesToggle(Protocol):
    """Đối tượng giống widget Qt có bật/tắt vẽ lại."""

    def setUpdatesEnabled(self, enabled: bool) -> None: ...


class IntervalThrottle:
    """[Editor Throttle] Giới hạn tần suất chạy một hành động xuống mỗi N mili-giây.

    Dùng cho việc bôi đậm dòng hiện tại + cuộn theo video: thay vì chạy 60 lần/giây
    (gây nghẽn CPU, giật khung hình), ép tối đa một lần mỗi ``min_interval_ms``.

    Đồng hồ được tiêm (``clock``) nên kiểm thử không cần thời gian thực.

    Args:
        min_interval_ms: Khoảng cách tối thiểu giữa hai lần chạy (mili-giây).
        clock: Hàm trả về thời điểm hiện tại theo *giây* (mặc định ``time.monotonic``).
    """

    def __init__(
        self, min_interval_ms: float = 100.0, clock: Callable[[], float] | None = None
    ) -> None:
        import time

        self._min_interval_sec = max(0.0, min_interval_ms / 1000.0)
        self._clock = clock or time.monotonic
        self._last_run_at: float | None = None

    def should_run(self) -> bool:
        """True nếu đã đủ thời gian kể từ lần chạy trước (và ghi nhận lần này).

        Lần gọi đầu tiên luôn trả True. Gọi lại trong vòng ``min_interval_ms`` sẽ
        trả False để bỏ qua công việc tốn kém (vẽ lại, cuộn…).
        """
        now = self._clock()
        if self._last_run_at is None or (now - self._last_run_at) >= self._min_interval_sec:
            self._last_run_at = now
            return True
        return False

    def reset(self) -> None:
        """Xoá mốc thời gian để lần ``should_run`` kế tiếp chắc chắn chạy."""
        self._last_run_at = None


@contextmanager
def batched_widget_update(widget: _SupportsUpdatesToggle) -> Iterator[None]:
    """[UI Freeze Prevention] Khoá vẽ lại trong lúc nạp dữ liệu khối lớn.

    Bọc đoạn nạp hàng ngàn dòng vào bảng: tắt ``setUpdatesEnabled(False)`` để
    Qt không vẽ lại sau mỗi dòng (nguyên nhân treo cứng khi nạp 4000+ dòng), nạp
    hết vào bộ nhớ, rồi bật lại để vẽ một lần duy nhất. Luôn bật lại kể cả khi có
    lỗi (``finally``), tránh để giao diện kẹt ở trạng thái không vẽ.

    Ví dụ::

        with batched_widget_update(self.cue_table):
            for cue in cues:
                self._append_row(cue)
    """
    widget.setUpdatesEnabled(False)
    try:
        yield
    finally:
        widget.setUpdatesEnabled(True)
