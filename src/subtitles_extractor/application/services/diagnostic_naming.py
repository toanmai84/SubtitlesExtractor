"""[v3.23.428] Đặt tên file xuất chẩn đoán NHẤT QUÁN cho mọi khâu (OCR/Dịch/TTS).

Mục tiêu: tránh các khâu cứ lưu cùng MỘT tên gây ghi đè + khó quản lý. Quy ước thống nhất:

    ``<gốc>.<khâu>.<YYYYMMDD_HHMMSS>.<đuôi>``

- ``<gốc>``: tên tệp nguồn (video/phụ đề) KHÔNG đuôi — giúp truy về đúng tập phim.
- ``<khâu>``: nhãn khâu (``translation_debug``, ``tts_debug``, ``ocr_raw``…).
- timestamp: thời điểm xuất (giờ địa phương) → mỗi lần xuất là MỘT tệp riêng, tự sắp xếp
  theo thời gian, dễ tìm bản mới nhất.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

#: Ký tự không hợp lệ cho tên tệp trên Windows (và ký tự điều khiển) → thay bằng "_".
_INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

#: Độ dài tối đa cho phần GỐC để tránh tên tệp quá dài (đường dẫn Windows ~260 ký tự).
_MAX_BASE_LEN = 80


def _sanitize_base(name: str) -> str:
    """Làm sạch phần gốc: bỏ ký tự cấm, gộp khoảng trắng, cắt bớt nếu quá dài.

    Args:
        name: Tên gốc thô (có thể chứa ký tự cấm/khoảng trắng).

    Returns:
        Chuỗi an toàn để làm tên tệp; ``"subtitle"`` nếu rỗng sau khi làm sạch.
    """
    cleaned = _INVALID_FILENAME_CHARS.sub("_", name).strip().strip(".")
    cleaned = re.sub(r"\s+", " ", cleaned)
    if len(cleaned) > _MAX_BASE_LEN:
        cleaned = cleaned[:_MAX_BASE_LEN].rstrip(" ._")
    return cleaned or "subtitle"


def build_diagnostic_filename(
    source_path: str | None,
    stage: str,
    ext: str,
    *,
    now: datetime | None = None,
) -> str:
    """Dựng tên tệp chẩn đoán nhất quán ``<gốc>.<khâu>.<timestamp>.<đuôi>``.

    Args:
        source_path: Đường dẫn tệp nguồn (video/phụ đề) để lấy tên gốc; ``None`` →
            ``"subtitle"``.
        stage: Nhãn khâu, ví dụ ``"translation_debug"``, ``"tts_debug"``, ``"ocr_raw"``.
        ext: Phần đuôi KHÔNG kèm dấu chấm, ví dụ ``"json"``, ``"csv"``.
        now: Thời điểm (để test tất định); mặc định ``datetime.now()``.

    Returns:
        Tên tệp (chỉ phần tên, không kèm thư mục).
    """
    timestamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    if source_path:
        base = _sanitize_base(Path(source_path).with_suffix("").name)
    else:
        base = "subtitle"
    stage_clean = _sanitize_base(stage) if stage else "diag"
    ext_clean = ext.lstrip(".") or "txt"
    return f"{base}.{stage_clean}.{timestamp}.{ext_clean}"


def build_diagnostic_path(
    source_path: str | None,
    stage: str,
    ext: str,
    *,
    output_dir: str | Path | None = None,
    now: datetime | None = None,
) -> str:
    """Như :func:`build_diagnostic_filename` nhưng trả về ĐƯỜNG DẪN đầy đủ.

    Thư mục ưu tiên: ``output_dir`` nếu có; nếu không, lưu CẠNH tệp nguồn; nếu cũng không
    có nguồn thì trả về tên tệp trơn (caller/hộp thoại tự quyết thư mục).

    Args:
        source_path: Đường dẫn tệp nguồn.
        stage: Nhãn khâu.
        ext: Đuôi tệp (không dấu chấm).
        output_dir: Thư mục đích tuỳ chọn.
        now: Thời điểm (để test).

    Returns:
        Đường dẫn đầy đủ (hoặc tên tệp trơn nếu không xác định được thư mục).
    """
    filename = build_diagnostic_filename(source_path, stage, ext, now=now)
    if output_dir is not None:
        return str(Path(output_dir) / filename)
    if source_path:
        return str(Path(source_path).parent / filename)
    return filename


def build_app_log_filename(*, now: datetime | None = None) -> str:
    """[v3.23.582] Tên file nhật ký ứng dụng: ``app_log.<YYYYMMDD_HHMMSS>.txt``.

    Nhật ký không gắn với một tệp nguồn nên không có phần ``<gốc>``; vẫn có timestamp như
    mọi file chuẩn đoán khác để lần xuất sau không ghi đè lần trước.

    Args:
        now: Thời điểm (để test tất định); mặc định ``datetime.now()``.

    Returns:
        Tên tệp (chỉ phần tên).
    """
    return f"app_log.{(now or datetime.now()).strftime('%Y%m%d_%H%M%S')}.txt"


__all__ = ["build_app_log_filename", "build_diagnostic_filename", "build_diagnostic_path"]
