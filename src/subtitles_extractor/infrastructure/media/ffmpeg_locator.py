"""Định vị binary ``ffmpeg``/``ffprobe``: ưu tiên bản ĐÓNG GÓI, rồi PATH hệ thống.

VÌ SAO tồn tại module này
=========================
Nhiều tính năng shell ra ``ffmpeg``/``ffprobe`` (video-context cho dịch, trích phụ
đề nhúng, waveform, mastering audio). Trước đây mỗi nơi tự gọi ``shutil.which()`` —
chỉ tìm trên PATH hệ thống. Bản đóng gói standalone ở máy người dùng thường KHÔNG
có ffmpeg trên PATH → các tính năng đó fail. Module này gom một chỗ (DRY) và cho
phép dùng ffmpeg ĐÃ NHÚNG trong bản build.

Thứ tự ưu tiên khi tìm
----------------------
1. Biến môi trường ``SUBEXT_FFMPEG_DIR`` (thư mục chứa ffmpeg/ffprobe) — escape
   hatch cho dev/test hoặc khi người dùng muốn chỉ định bản riêng.
2. ``vendor/ffmpeg`` — thư mục binary native tập trung (xem infrastructure/vendor.py),
   hoạt động cả khi chạy nguồn lẫn khi đóng gói (``sys._MEIPASS/vendor/ffmpeg``).
3. PATH hệ thống (``shutil.which``) — hành vi cũ, dùng khi chạy nguồn/dev.

LƯU Ý LICENSE
-------------
Binary nhúng phải là bản **LGPL** (dựng với ``--disable-gpl``, không kèm x264…) để
giữ tính license-clean thương mại. Module này KHÔNG tự tải binary; nó chỉ định vị.
Việc chọn/đặt binary phù hợp là ở khâu build.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

from subtitles_extractor.infrastructure.vendor import vendor_subdir

logger = logging.getLogger(__name__)

# Tên nhóm binary ffmpeg trong thư mục vendor tập trung (vendor/ffmpeg/).
_VENDOR_SUBDIR: str = "ffmpeg"

# Biến môi trường trỏ thư mục ffmpeg tuỳ ý (ưu tiên cao nhất, escape hatch riêng).
_FFMPEG_DIR_ENV: str = "SUBEXT_FFMPEG_DIR"


def _candidate_filenames(tool: str) -> tuple[str, ...]:
    """Tên file khả dĩ của một tool theo hệ điều hành.

    Windows cần đuôi ``.exe``; thử cả hai để an toàn.
    """
    if os.name == "nt":
        return (f"{tool}.exe", tool)
    return (tool,)


def find_executable_in_dir(directory: Path, tool: str) -> str | None:
    """Tìm ``tool`` (ffmpeg/ffprobe) trong một thư mục cụ thể.

    Args:
        directory: Thư mục cần tìm.
        tool: Tên tool không đuôi (vd ``"ffmpeg"``).

    Returns:
        Đường dẫn tuyệt đối (chuỗi) nếu tìm thấy file khả thi; ``None`` nếu không.
    """
    for filename in _candidate_filenames(tool):
        candidate = directory / filename
        # os.access(X_OK) trên Windows luôn True với file tồn tại — vẫn an toàn.
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def _bundled_bin_dir() -> Path | None:
    """Thư mục chứa binary ffmpeg đã vendored/được chỉ định, nếu có.

    Ưu tiên: env ``SUBEXT_FFMPEG_DIR`` > ``vendor/ffmpeg`` (tập trung) .

    Returns:
        :class:`~pathlib.Path` hợp lệ, hoặc ``None`` nếu không có (sẽ dùng PATH).
    """
    override = os.environ.get(_FFMPEG_DIR_ENV, "").strip()
    if override:
        override_dir = Path(override)
        if override_dir.is_dir():
            return override_dir
        logger.warning(
            "%s trỏ tới thư mục không tồn tại: %s — bỏ qua.",
            _FFMPEG_DIR_ENV,
            override,
        )

    return vendor_subdir(_VENDOR_SUBDIR)


def _resolve(tool: str) -> str | None:
    """Định vị ``tool`` theo thứ tự ưu tiên bundle → PATH.

    Args:
        tool: ``"ffmpeg"`` hoặc ``"ffprobe"``.

    Returns:
        Đường dẫn tới binary, hoặc ``None`` nếu không tìm thấy ở đâu cả.
    """
    bundled_dir = _bundled_bin_dir()
    if bundled_dir is not None:
        found = find_executable_in_dir(bundled_dir, tool)
        if found is not None:
            logger.debug("Dùng %s đã nhúng: %s.", tool, found)
            return found

    on_path = shutil.which(tool)
    if on_path is not None:
        return on_path

    return None


def find_ffmpeg() -> str | None:
    """Định vị ``ffmpeg`` (bundle → PATH). ``None`` nếu không có ở đâu."""
    return _resolve("ffmpeg")


def find_ffprobe() -> str | None:
    """Định vị ``ffprobe`` (bundle → PATH). ``None`` nếu không có ở đâu."""
    return _resolve("ffprobe")


# [v3.23.306] Tính năng PHỤ THUỘC ffmpeg CLI — dùng cho thông báo lỗi thống nhất.
# Lõi OCR hardsub + trình phát video KHÔNG nằm trong danh sách này (dùng PyAV/libmpv).
FFMPEG_DEPENDENT_FEATURES: tuple[str, ...] = (
    "Ngữ cảnh video khi dịch (cắt đoạn gửi Gemini)",
    "Trích phụ đề nhúng từ video",
    "Sóng âm (waveform) trong Trình sửa",
    "Nén audio TTS sang định dạng nén (đã có phương án WAV thay thế)",
)


def missing_ffmpeg_message(*, feature: str | None = None) -> str:
    """Thông điệp chuẩn khi KHÔNG tìm thấy ffmpeg — nêu rõ cách khắc phục.

    Vì sao cần: sau khi gỡ ``vendor/ffmpeg`` (tránh bản GPL), máy người dùng cuối
    thường KHÔNG có ffmpeg trên PATH. Thông báo phải nói rõ tính năng nào hỏng và
    cách sửa ĐÚNG LICENSE, thay vì chỉ "không tìm thấy ffmpeg".

    Args:
        feature: Tên tính năng đang cần ffmpeg (để nêu cụ thể). ``None`` = thông
            điệp tổng quát liệt kê tất cả tính năng ảnh hưởng.

    Returns:
        Chuỗi thông báo tiếng Việt, có hướng dẫn hành động.
    """
    head = (
        f"Không tìm thấy ffmpeg — cần cho: {feature}."
        if feature
        else "Không tìm thấy ffmpeg. Các tính năng sau sẽ không dùng được:\n  - "
        + "\n  - ".join(FFMPEG_DEPENDENT_FEATURES)
    )
    return (
        f"{head}\n"
        "Cách khắc phục (giữ license-clean thương mại): tải bản dựng ffmpeg "
        "**LGPL** (tên tệp có 'lgpl', KHÔNG phải 'gpl') rồi đặt ffmpeg.exe và "
        "ffprobe.exe vào thư mục 'vendor/ffmpeg/' của ứng dụng. "
        "Kiểm tra lại bằng: python tools/check_media_licenses.py"
    )


__all__ = ["find_executable_in_dir", "find_ffmpeg", "find_ffprobe"]
