"""Định vị thư mục ``vendor/`` tập trung chứa binary native vendored.

Ý tưởng
=======
Mọi binary native mà app phụ thuộc nhưng KHÔNG cài qua pip (libmpv, ffmpeg/ffprobe…)
được gom vào MỘT thư mục ``vendor/`` ở gốc dự án, có cấu trúc con rõ ràng::

    vendor/
        mpv/       libmpv-2.dll  (hoặc mpv-2.dll)   — LGPL
        ffmpeg/    ffmpeg.exe, ffprobe.exe          — PHẢI là bản LGPL

Lợi ích: tập trung một chỗ, dễ thêm/bớt/cập nhật, gốc dự án sạch, dễ .gitignore
binary lớn, và ``.spec`` chỉ cần nhúng nguyên cây ``vendor/``.

Module này KHÔNG có phụ thuộc nặng (chỉ stdlib) để mọi tầng dùng chung được và test
độc lập.

Thứ tự phân giải gốc ``vendor``
------------------------------
1. Biến môi trường ``SUBEXT_VENDOR_DIR`` (escape hatch, trỏ gốc vendor tuỳ ý).
2. Bản đóng gói: ``sys._MEIPASS/vendor`` (do ``.spec`` nhúng).
3. Chạy nguồn: ``<gốc dự án>/vendor`` (cạnh ``main.py``).
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Tên thư mục vendor + biến môi trường ghi đè gốc vendor.
_VENDOR_DIRNAME: str = "vendor"
_VENDOR_DIR_ENV: str = "SUBEXT_VENDOR_DIR"


def vendor_root() -> Path | None:
    """Trả về gốc thư mục ``vendor`` đang dùng, theo thứ tự ưu tiên.

    Returns:
        :class:`~pathlib.Path` tới gốc vendor tồn tại đầu tiên, hoặc ``None`` nếu
        không có ở đâu (khi đó caller tự fallback: PATH / tải runtime).
    """
    override = os.environ.get(_VENDOR_DIR_ENV, "").strip()
    if override:
        override_dir = Path(override)
        if override_dir.is_dir():
            return override_dir
        logger.warning(
            "%s trỏ tới thư mục không tồn tại: %s — bỏ qua.",
            _VENDOR_DIR_ENV,
            override,
        )

    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        bundled = Path(sys._MEIPASS) / _VENDOR_DIRNAME
        if bundled.is_dir():
            return bundled

    # Chạy nguồn: vendor.py ở src/subtitles_extractor/infrastructure/ → parents[3] = gốc.
    try:
        source_root = Path(__file__).resolve().parents[3] / _VENDOR_DIRNAME
    except IndexError:
        return None
    if source_root.is_dir():
        return source_root

    return None


def vendor_subdir(name: str) -> Path | None:
    """Trả về thư mục con ``vendor/<name>`` nếu tồn tại.

    Args:
        name: Tên nhóm binary (vd ``"mpv"``, ``"ffmpeg"``).

    Returns:
        :class:`~pathlib.Path` tới ``vendor/<name>`` nếu có, ngược lại ``None``.
    """
    root = vendor_root()
    if root is None:
        return None
    subdir = root / name
    return subdir if subdir.is_dir() else None


__all__ = ["vendor_root", "vendor_subdir"]
