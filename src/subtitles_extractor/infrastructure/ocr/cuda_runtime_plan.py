"""[v3.23.375] Lập kế hoạch TẢI CUDA runtime cho GPU OCR *lúc chạy*.

Bản đóng gói mặc định KHÔNG nhúng thư viện CUDA (~2.3GB) để đủ nhỏ up lên GitHub. App vẫn
chạy ngay bằng CPU (paddle tự lùi CPU khi thiếu CUDA). Người dùng bấm "Bật tăng tốc GPU (OCR)"
để tải các gói ``nvidia-*-cu12`` về thư mục ``models/cuda_runtime/`` — khớp cơ chế tải-lúc-chạy
của WhisperX/VieNeu, KHÔNG đụng tới môi trường đóng gói.

Tách phần quyết định (thuần, test được) khỏi phần chạy pip (worker).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

#: Các gói wheel chứa DLL CUDA mà paddlepaddle-gpu cần nạp lúc chạy (khớp CUDA 12.x).
CUDA_RUNTIME_PACKAGES: Final[tuple[str, ...]] = (
    "nvidia-cudnn-cu12",
    "nvidia-cublas-cu12",
    "nvidia-cufft-cu12",
    "nvidia-curand-cu12",
    "nvidia-cusolver-cu12",
    "nvidia-cusparse-cu12",
    "nvidia-cuda-runtime-cu12",
    "nvidia-cuda-nvrtc-cu12",
    "nvidia-nvjitlink-cu12",
)

#: Tên thư mục con của kho model dùng cho CUDA runtime tải về.
CUDA_RUNTIME_DIRNAME: Final[str] = "cuda_runtime"

#: Ước tính dung lượng tải (GB) để hiện cho người dùng trước khi tải.
CUDA_RUNTIME_DOWNLOAD_GB: Final[float] = 2.3

#: Thư mục con ``nvidia/<component>`` mà pip trải ra khi cài các wheel trên.
_NVIDIA_COMPONENTS: Final[tuple[str, ...]] = (
    "cudnn", "cublas", "cufft", "curand", "cusolver", "cusparse",
    "cuda_runtime", "cuda_nvrtc", "nvjitlink",
)


class CudaRuntimeStatus(StrEnum):
    """Trạng thái CUDA runtime cho GPU OCR."""

    BUNDLED = "bundled"
    """Đã nhúng sẵn trong bản đóng gói (build SUBEXT_BUNDLE_CUDA=1) — không cần tải."""

    INSTALLED = "installed"
    """Đã tải về ``models/cuda_runtime/`` từ lần trước — dùng được ngay."""

    NEEDS_DOWNLOAD = "needs_download"
    """Chưa có — cần tải để bật GPU (app vẫn chạy CPU)."""


@dataclass(frozen=True, slots=True)
class CudaRuntimePlan:
    """Kế hoạch bật GPU OCR.

    Attributes:
        status: Trạng thái hiện tại.
        cuda_dir: Thư mục sẽ tải CUDA runtime vào (``models/cuda_runtime``).
        download_estimate_gb: Ước tính dung lượng tải.
    """

    status: CudaRuntimeStatus
    cuda_dir: Path
    download_estimate_gb: float = CUDA_RUNTIME_DOWNLOAD_GB

    @property
    def needs_download(self) -> bool:
        return self.status is CudaRuntimeStatus.NEEDS_DOWNLOAD


def cuda_runtime_dll_dirs(cuda_dir: Path) -> list[Path]:
    """Danh sách thư mục ``.../nvidia/<component>/bin`` cần thêm vào DLL search path.

    Chỉ trả về thư mục THỰC SỰ tồn tại (đã tải xong), để bootstrap thêm an toàn.
    """
    base = cuda_dir / "nvidia"
    dirs = [base / comp / "bin" for comp in _NVIDIA_COMPONENTS]
    return [d for d in dirs if d.is_dir()]


def is_cuda_runtime_installed(cuda_dir: Path) -> bool:
    """CUDA runtime đã tải xong chưa? (có ít nhất một DLL cuDNN/cuBLAS)."""
    return bool(cuda_runtime_dll_dirs(cuda_dir))


def evaluate_cuda_runtime(
    cuda_dir: Path, *, bundled: bool
) -> CudaRuntimePlan:
    """Đánh giá trạng thái CUDA runtime (hàm thuần).

    Args:
        cuda_dir: Thư mục ``models/cuda_runtime``.
        bundled: ``True`` nếu bản build đã nhúng sẵn CUDA (SUBEXT_BUNDLE_CUDA=1).
    """
    if bundled:
        return CudaRuntimePlan(CudaRuntimeStatus.BUNDLED, cuda_dir)
    if is_cuda_runtime_installed(cuda_dir):
        return CudaRuntimePlan(CudaRuntimeStatus.INSTALLED, cuda_dir)
    return CudaRuntimePlan(CudaRuntimeStatus.NEEDS_DOWNLOAD, cuda_dir)


def build_cuda_install_command(python_exe: str, cuda_dir: Path) -> tuple[str, ...]:
    """Lệnh pip tải CUDA runtime vào ``cuda_dir`` (không đụng site-packages nào khác).

    Dùng ``--target`` để trải wheel ra đúng thư mục đích; ``--no-deps`` vì các gói nvidia-*
    độc lập nhau, tránh kéo thêm phụ thuộc thừa.
    """
    return (
        python_exe, "-m", "pip", "install",
        "--target", str(cuda_dir),
        "--no-deps", "--upgrade",
        *CUDA_RUNTIME_PACKAGES,
    )


__all__ = [
    "CUDA_RUNTIME_DIRNAME",
    "CUDA_RUNTIME_DOWNLOAD_GB",
    "CUDA_RUNTIME_PACKAGES",
    "CudaRuntimePlan",
    "CudaRuntimeStatus",
    "build_cuda_install_command",
    "cuda_runtime_dll_dirs",
    "evaluate_cuda_runtime",
    "is_cuda_runtime_installed",
]
