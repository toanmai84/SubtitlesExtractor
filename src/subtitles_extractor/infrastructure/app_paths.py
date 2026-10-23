"""[v3.23.582] Thư mục GỐC của ứng dụng — nơi chung cho các file chuẩn đoán và nhật ký.

Trước đây ba nút xuất mở hộp thoại lưu ở ba chỗ khác nhau: chuẩn đoán phân tích theo
thư mục làm việc hiện tại, chuẩn đoán dịch cạnh file video, nhật ký theo hộp thoại hệ
điều hành (nhớ thư mục lần trước) — người dùng phải tự lần về đúng thư mục mỗi lần xuất.

Bản đóng gói: thư mục chứa file ``.exe`` (cùng chỗ với ``models/``). Chạy từ mã nguồn: thư
mục dự án (chứa ``src/``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import subtitles_extractor


def application_root_dir(
    *,
    frozen: bool | None = None,
    executable: str | None = None,
    package_file: str | None = None,
) -> Path:
    """Thư mục gốc của ứng dụng (hàm thuần khi truyền đủ tham số).

    Args:
        frozen: Đang chạy bản đóng gói không; ``None`` = đọc ``sys.frozen``.
        executable: Đường dẫn file thực thi; ``None`` = ``sys.executable``.
        package_file: ``__file__`` của gói ``subtitles_extractor``; ``None`` = thật.

    Returns:
        Bản đóng gói: thư mục chứa file thực thi. Chạy từ nguồn: thư mục cha của ``src``
        (``<dự án>/src/subtitles_extractor/__init__.py`` → ``<dự án>``).
    """
    is_frozen = getattr(sys, "frozen", False) if frozen is None else frozen
    if is_frozen:
        return Path(executable or sys.executable).resolve().parent
    package = Path(package_file or subtitles_extractor.__file__).resolve()
    return package.parent.parent.parent


__all__ = ["application_root_dir"]
