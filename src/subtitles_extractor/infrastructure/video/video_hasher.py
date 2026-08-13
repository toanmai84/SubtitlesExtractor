"""Tính hash định danh duy nhất cho file video, độc lập tên/thư mục.

Mục tiêu: cùng một video (dù đổi tên hay di chuyển thư mục) phải cho cùng một
hash → tránh xử lý lại từ đầu. Với file video lớn (hàng GB), đọc toàn bộ để
băm sẽ rất chậm, nên dùng chiến lược "partial hash": băm kích thước file cộng
một số đoạn mẫu ở đầu, giữa và cuối. Xác suất trùng lặp giả gần như bằng 0 cho
mục đích nhận dạng nội dung phim.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 1024 * 1024  # 1MB mỗi đoạn mẫu
_SAMPLE_POSITIONS = 3      # đầu, giữa, cuối


def compute_video_hash(video_path: str | Path) -> str:
    """Tính hash SHA-256 rút gọn (16 ký tự hex) định danh nội dung video.

    Băm: kích thước file + 3 đoạn 1MB (đầu/giữa/cuối). Nhanh kể cả với file
    nhiều GB vì chỉ đọc tối đa 3MB.

    Args:
        video_path: Đường dẫn tới file video.

    Returns:
        Chuỗi hash 16 ký tự hex định danh duy nhất nội dung video.

    Raises:
        FileNotFoundError: Nếu file không tồn tại.
        OSError: Nếu không đọc được file.
    """
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file video: {path}")

    file_size = path.stat().st_size
    hasher = hashlib.sha256()
    hasher.update(str(file_size).encode("utf-8"))

    with path.open("rb") as handle:
        if file_size <= _CHUNK_SIZE * _SAMPLE_POSITIONS:
            # File nhỏ: băm toàn bộ cho chắc chắn.
            hasher.update(handle.read())
        else:
            positions = (
                0,
                file_size // 2 - _CHUNK_SIZE // 2,
                file_size - _CHUNK_SIZE,
            )
            for offset in positions:
                handle.seek(offset)
                hasher.update(handle.read(_CHUNK_SIZE))

    digest = hasher.hexdigest()[:16]
    logger.debug("Video hash %s ← %s (size=%d)", digest, path.name, file_size)
    return digest


__all__ = ["compute_video_hash"]
