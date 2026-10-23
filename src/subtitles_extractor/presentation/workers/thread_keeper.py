"""Giữ luồng nền và worker của nó sống tới khi luồng THOÁT HẲN rồi mới thả.

Mẫu "worker chuyển sang QThread" (``worker.moveToThread(thread)``) từng dọn dẹp bằng hai
đường cùng lúc:

* ``thread.finished → worker.deleteLater``: Qt xoá worker NGAY TRONG LUỒNG NỀN, ở bước
  kết thúc luồng;
* slot ``finished`` ở luồng giao diện gán ``self._worker = None``: Python thả tham chiếu
  cuối cùng và cũng xoá chính worker đó.

Hai lần xoá chạy song song ở hai luồng → xoá hai lần, hỏng heap. Đo bằng tua liên tục
trên trang Trích xuất (khung vẽ dự phòng khi không có MPV): 4/5 lần chạy 400 lần tua sập
ứng dụng (Segmentation fault / Bus error).

Ở đây chỉ còn MỘT chủ sở hữu: Python giữ worker tới khi ``finished`` về luồng giao diện,
chờ luồng thoát hẳn (``wait``) rồi mới thả — lúc đó không còn ai khác đụng tới worker.
Không gắn ``worker.deleteLater`` vào ``thread.finished`` khi đã dùng lớp này.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import QObject, Qt, QThread

logger = logging.getLogger(__name__)

_EXIT_WAIT_MS = 5000
"""Chờ tối đa khi luồng đã phát ``finished`` (chỉ còn vài lệnh dọn dẹp là thoát)."""


class WorkerThreadKeeper(QObject):
    """Giữ tham chiếu (luồng, worker) cho tới khi luồng thoát hẳn.

    Đặt một đối tượng cho mỗi trang/view model (tạo ở luồng giao diện) và gọi
    :meth:`keep` ngay khi tạo luồng — kể cả khi nơi gọi còn tự giữ tham chiếu riêng:
    gán đè tham chiếu đó (bắt đầu tác vụ mới khi luồng cũ chưa thoát) cũng không còn
    làm xoá worker dưới chân luồng cũ.
    """

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._kept: dict[int, tuple[QThread, QObject | None]] = {}

    def keep(self, thread: QThread, worker: QObject | None = None) -> None:
        """Giữ ``thread`` (và ``worker`` chạy trong nó) tới khi luồng thoát.

        Args:
            thread: Luồng nền — chưa ``start`` hoặc vừa ``start``.
            worker: Đối tượng đã ``moveToThread(thread)`` (nếu có).
        """
        self._kept[id(thread)] = (thread, worker)
        thread.finished.connect(self._release, Qt.ConnectionType.QueuedConnection)

    @property
    def kept_count(self) -> int:
        """Số luồng đang được giữ (để kiểm thử/chẩn đoán)."""
        return len(self._kept)

    def _release(self) -> None:
        thread = self.sender()
        pair = self._kept.pop(id(thread), None)
        if pair is None:
            return
        # "finished" được phát TRƯỚC khi luồng thoát hẳn; thả worker lúc đó vẫn có thể
        # trùng với bước dọn dẹp cuối của luồng.
        try:
            exited = pair[0].wait(_EXIT_WAIT_MS)
        except RuntimeError:
            exited = True  # QThread đã bị Qt huỷ (deleteLater) → chắc chắn đã thoát
        if not exited:
            logger.warning(
                "Luồng nền chưa thoát sau %d ms kể từ khi báo xong — vẫn giữ worker.",
                _EXIT_WAIT_MS,
            )
            self._kept[id(thread)] = pair


__all__ = ["WorkerThreadKeeper"]
