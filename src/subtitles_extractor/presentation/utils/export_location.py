"""[v3.23.582] Nơi hộp thoại lưu mở ra cho các nút xuất chuẩn đoán và nhật ký.

Ba nút xuất (chuẩn đoán phân tích, chuẩn đoán dịch, nhật ký) vẫn HỎI nơi lưu mỗi lần,
nhưng hộp thoại luôn mở sẵn ở CÙNG một thư mục: thư mục gốc của ứng dụng. Xem
:mod:`subtitles_extractor.infrastructure.app_paths`.
"""

from __future__ import annotations

from subtitles_extractor.infrastructure.app_paths import application_root_dir


def export_dialog_path(filename: str) -> str:
    """Đường dẫn gợi ý cho hộp thoại lưu: ``<thư mục gốc ứng dụng>/<filename>``.

    Args:
        filename: Tên file gợi ý (không kèm thư mục).

    Returns:
        Đường dẫn đầy đủ dạng chuỗi.
    """
    return str(application_root_dir() / filename)


__all__ = ["export_dialog_path"]
