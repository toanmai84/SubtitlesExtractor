"""Hợp đồng thiết bị phát âm thanh — tách khỏi Qt để kiểm thử được.

VÌ SAO cần cổng này
===================
Đồng bộ A/V chuẩn dùng **âm thanh làm đồng hồ chủ**: phần cứng âm thanh tiêu thụ mẫu
ở tốc độ cố định rất chính xác, nên vị trí phát của nó là mốc thời gian đáng tin nhất.
Hình được kéo theo cho khớp.

Nhưng thiết bị âm thanh thật (``QAudioSink``) không kiểm thử được trong CI/không có
card âm thanh. Tách thành cổng cho phép:
    * Qt hiện thực bằng ``QAudioSink`` (PySide6, LGPL).
    * Test hiện thực bằng thiết bị giả, kiểm soát chính xác "đã phát bao nhiêu giây".

Nhờ vậy toàn bộ **logic đồng bộ** kiểm chứng được, chỉ còn phần nối dây thiết bị thật
là không kiểm được ở đây.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class AudioSinkPort(Protocol):
    """Hợp đồng tối thiểu của một thiết bị phát âm thanh PCM."""

    @property
    def free_bytes(self) -> int:
        """Số byte thiết bị còn nhận thêm được ngay lúc này."""
        ...

    def write(self, pcm: bytes) -> int:
        """Đẩy dữ liệu PCM xuống thiết bị.

        Args:
            pcm: PCM 16-bit xen kẽ kênh.

        Returns:
            Số byte thiết bị đã nhận (có thể ít hơn ``len(pcm)`` nếu đầy).
        """
        ...

    def played_seconds(self) -> float:
        """Thời lượng âm thanh thiết bị ĐÃ PHÁT kể từ lần :meth:`reset` gần nhất."""
        ...

    def reset(self) -> None:
        """Xoá bộ đệm và đưa bộ đếm đã-phát về 0 (dùng khi seek)."""
        ...

    def set_volume(self, volume: float) -> None:
        """Đặt âm lượng trong khoảng ``0.0``–``1.0``."""
        ...

    def close(self) -> None:
        """Đóng thiết bị. Idempotent."""
        ...


__all__ = ["AudioSinkPort"]
