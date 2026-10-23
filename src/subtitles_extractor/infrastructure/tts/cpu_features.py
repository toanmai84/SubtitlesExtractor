"""[v3.23.595] Dò tập lệnh VNNI của CPU — điều kiện để VieNeu chạy int8 không méo tiếng.

VieNeu 3.8 (``vieneu.v3turbo``) cho chọn ``precision="int8"`` ở đường ONNX/CPU: "nhanh ~3x
mỗi frame, nhỏ 4x; cần CPU hỗ trợ VNNI để không bị méo" (chú thích trong mã nguồn SDK),
mặc định vẫn là ``fp32``. VNNI có hai dạng: AVX-512 VNNI (Intel Xeon/Core đời 10+ có
AVX-512, AMD Zen 4+) và AVX-VNNI (Intel Core đời 12+ không có AVX-512).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

#: Tên cờ VNNI trong /proc/cpuinfo (Linux).
_PROC_FLAGS = frozenset({"avx512_vnni", "avx_vnni"})
#: Tên tính năng VNNI trong bảng dò CPU của NumPy.
_NUMPY_FEATURES = ("AVX512VNNI", "AVXVNNI")


def vnni_from_cpu_flags(flags: set[str]) -> bool:
    """CPU có VNNI không, theo tập cờ kiểu /proc/cpuinfo (hàm thuần).

    Args:
        flags: Cờ CPU viết thường (vd ``{"avx2", "avx_vnni"}``).

    Returns:
        ``True`` nếu có một trong hai dạng VNNI.
    """
    return bool(flags & _PROC_FLAGS)


def vnni_from_numpy_features(features: dict[str, bool]) -> bool | None:
    """CPU có VNNI không, theo bảng ``__cpu_features__`` của NumPy (hàm thuần).

    NumPy có mục AVX-512 VNNI nhưng nhiều bản không có mục AVX-VNNI, nên "không thấy" chỉ
    là KHÔNG RÕ, không phải "không có".

    Args:
        features: Bảng tính năng CPU của NumPy.

    Returns:
        ``True`` nếu thấy VNNI; ``False`` nếu NumPy có dò AVX-VNNI và cả hai đều không có;
        ``None`` nếu không kết luận được.
    """
    if any(features.get(name) for name in _NUMPY_FEATURES):
        return True
    if all(name in features for name in _NUMPY_FEATURES):
        return False
    return None


def cpu_vnni_support() -> bool | None:
    """Dò VNNI trên máy đang chạy.

    Returns:
        ``True``/``False`` khi biết chắc; ``None`` khi không dò được.
    """
    cpuinfo = Path("/proc/cpuinfo")
    if sys.platform.startswith("linux") and cpuinfo.is_file():
        try:
            text = cpuinfo.read_text(encoding="utf-8", errors="replace")
            for line in text.splitlines():
                if line.startswith("flags"):
                    return vnni_from_cpu_flags(set(line.split(":", 1)[-1].split()))
        except OSError as exc:
            logger.debug("Không đọc được /proc/cpuinfo: %s", exc)
    try:
        from numpy._core._multiarray_umath import __cpu_features__ as features
    except ImportError:
        try:
            from numpy.core._multiarray_umath import __cpu_features__ as features
        except ImportError:
            return None
    return vnni_from_numpy_features(dict(features))


__all__ = ["cpu_vnni_support", "vnni_from_cpu_flags", "vnni_from_numpy_features"]
