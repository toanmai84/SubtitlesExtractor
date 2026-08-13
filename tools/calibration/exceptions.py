"""Ngoại lệ tùy chỉnh cho khung tự hiệu chuẩn (calibration framework).

Mỗi lỗi nghiệp vụ có một loại ngoại lệ riêng để tầng gọi xử lý chính xác,
tránh ``except Exception`` chung chung.
"""

from __future__ import annotations


class CalibrationError(Exception):
    """Lớp gốc cho mọi lỗi của khung hiệu chuẩn."""


class GroundTruthNotFoundError(CalibrationError):
    """Không tìm thấy hoặc không đọc được file ground-truth (SRT)."""


class PairingAmbiguousError(CalibrationError):
    """Không ghép được cặp seraw↔SRT đủ tin cậy (độ trùng dưới ngưỡng)."""


class EmptyDatasetError(CalibrationError):
    """Bộ dữ liệu hiệu chuẩn rỗng — không có cặp nào để đánh giá."""


class InvalidParameterSpecError(CalibrationError):
    """Đặc tả tham số (ParameterSpec) không hợp lệ (low > high, kind sai...)."""


class ObjectiveEvaluationError(CalibrationError):
    """Lỗi khi đánh giá hàm mục tiêu cho một bộ tham số ứng viên."""


class CalibrationStateError(CalibrationError):
    """Lỗi đọc/ghi trạng thái hiệu chuẩn đã lưu (warm-start JSON)."""
