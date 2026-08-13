"""Hợp đồng phát video — phục vụ tab Trích xuất và Editor.

Adapter mpv hiện thực hợp đồng này. Tầng presentation chỉ phụ thuộc
vào port để có thể swap player (mpv ↔ Qt MediaPlayer ↔ VLC) không
phải sửa view.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class VideoPlayerPort(Protocol):
    """Hợp đồng tối thiểu của một video player.

    Các adapter phải an toàn khi gọi ``release`` 2 lần (idempotent).
    """

    @property
    def is_loaded(self) -> bool:
        """``True`` khi có media đang được nạp."""
        ...

    @property
    def is_playing(self) -> bool:
        """``True`` khi đang phát (không pause)."""
        ...

    @property
    def position_sec(self) -> float:
        """Vị trí phát hiện tại (giây)."""
        ...

    @property
    def duration_sec(self) -> float:
        """Tổng thời lượng (giây). 0 nếu chưa nạp."""
        ...

    def load(self, video_path: Path) -> None:
        """Nạp video — không tự phát."""
        ...

    def play(self) -> None:
        """Phát từ vị trí hiện tại."""
        ...

    def pause(self) -> None:
        """Tạm dừng."""
        ...

    def toggle_play_pause(self) -> None:
        """Đảo trạng thái play/pause."""
        ...

    def seek(self, position_sec: float) -> None:
        """Nhảy tới mốc thời gian (giây)."""
        ...

    def step_frame(self, forward: bool = True) -> None:
        """Bước 1 frame (forward/backward)."""
        ...

    def set_volume(self, volume: int) -> None:
        """Đặt âm lượng 0–100."""
        ...

    def set_speed(self, speed: float) -> None:
        """Đặt tốc độ phát (0.25 – 4.0)."""
        ...

    def release(self) -> None:
        """Giải phóng tài nguyên. Idempotent."""
        ...


__all__ = ["VideoPlayerPort"]
