"""Dò GPU NVIDIA khả dụng THẬT lúc chạy (thuần, không phụ thuộc nặng).

VÌ SAO tồn tại module này
=========================
Bản đóng gói dùng ``paddlepaddle-gpu`` (build kèm CUDA). Vấn đề:
``paddle.is_compiled_with_cuda()`` chỉ cho biết **bản build** có CUDA — nó trả
``True`` trên MỌI máy, kể cả máy KHÔNG có GPU NVIDIA. Nếu chỉ dựa vào hàm đó để
chọn ``device="gpu"``, thì trên máy không NVIDIA việc khởi tạo model sẽ lỗi và
OCR không chạy được.

Để bản standalone chạy được cả trên máy không NVIDIA, ta phải kiểm **phần cứng
thật**: ``paddle.device.cuda.device_count() > 0``. Chỉ khi có ít nhất một GPU
NVIDIA dùng được mới chọn GPU; ngược lại lùi về CPU.

Tách thành module riêng (không import ``paddle``/``cv2``) để kiểm thử độc lập
bằng một đối tượng ``paddle`` giả — không cần cài paddle thật trong CI.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class _CudaNamespace(Protocol):
    """Giao diện tối thiểu của ``paddle.device.cuda`` mà module này cần."""

    def device_count(self) -> int: ...


class _DeviceNamespace(Protocol):
    """Giao diện tối thiểu của ``paddle.device``."""

    cuda: _CudaNamespace


class PaddleLike(Protocol):
    """Giao diện tối thiểu của module ``paddle`` mà :func:`cuda_runtime_available` dùng.

    Cho phép truyền một stub trong test thay vì import paddle thật.
    """

    def is_compiled_with_cuda(self) -> bool: ...

    device: _DeviceNamespace


def _safe_device_count(paddle_module: Any) -> int:
    """Đọc ``paddle.device.cuda.device_count()`` an toàn.

    Trên máy không có driver NVIDIA, lời gọi có thể trả 0 hoặc ném ngoại lệ
    (``RuntimeError``/``OSError``) tuỳ phiên bản paddle — coi mọi trường hợp lỗi
    là "không có GPU".

    Returns:
        Số GPU NVIDIA đếm được (>= 0). Trả 0 nếu không xác định được.
    """
    try:
        count = paddle_module.device.cuda.device_count()
    except (AttributeError, RuntimeError, OSError) as exc:
        logger.debug("device_count() không khả dụng — coi như 0 GPU: %s.", exc)
        return 0
    try:
        return max(0, int(count))
    except (TypeError, ValueError):
        return 0


def cuda_runtime_available(paddle_module: Any) -> bool:
    """True nếu paddle build kèm CUDA **và** máy thực sự có GPU NVIDIA dùng được.

    Args:
        paddle_module: Module ``paddle`` (hoặc stub tương thích :class:`PaddleLike`).

    Returns:
        ``True`` chỉ khi cả hai điều kiện đúng: bản build có CUDA và
        ``device_count() > 0``. Ngược lại ``False`` (nên dùng CPU).
    """
    try:
        compiled = bool(paddle_module.is_compiled_with_cuda())
    except (AttributeError, RuntimeError) as exc:
        logger.debug("is_compiled_with_cuda() lỗi — coi như không CUDA: %s.", exc)
        return False

    if not compiled:
        return False

    device_count = _safe_device_count(paddle_module)
    if device_count <= 0:
        logger.info(
            "Paddle có CUDA nhưng KHÔNG phát hiện GPU NVIDIA dùng được "
            "(device_count=0) — sẽ dùng CPU cho OCR."
        )
        return False

    logger.debug("Phát hiện %d GPU NVIDIA dùng được cho OCR.", device_count)
    return True


def should_use_gpu(
    paddle_module: Any, *, want_gpu: bool, force_cpu: bool
) -> bool:
    """Quyết định CUỐI CÙNG có dùng GPU cho OCR hay không.

    Gộp 3 yếu tố theo thứ tự ưu tiên:
        1. ``force_cpu`` (vd biến môi trường ``SUBEXT_FORCE_CPU``) — ép CPU, thắng tất cả.
        2. ``want_gpu`` — cấu hình người dùng có chọn GPU không.
        3. :func:`cuda_runtime_available` — máy có GPU NVIDIA THẬT không.

    Args:
        paddle_module: Module ``paddle`` (hoặc stub tương thích).
        want_gpu: True nếu cấu hình đặt thiết bị = GPU.
        force_cpu: True nếu có tín hiệu ép CPU (env/escape hatch).

    Returns:
        True chỉ khi ``not force_cpu and want_gpu and cuda_runtime_available(...)``.
    """
    if force_cpu:
        logger.info("SUBEXT_FORCE_CPU bật — ép OCR chạy CPU (bỏ qua GPU).")
        return False
    if not want_gpu:
        return False
    return cuda_runtime_available(paddle_module)


__all__ = ["PaddleLike", "cuda_runtime_available", "should_use_gpu"]
