"""[v3.23.386] Lập kế hoạch TẢI ``paddlepaddle-gpu`` (lõi binary ~810MB) *lúc chạy*.

Bối cảnh
========
``paddlepaddle-gpu`` là thành phần NẶNG NHẤT của bản đóng gói (~810MB lõi binary). Để bản
build đủ nhỏ mà dễ chia sẻ (mục tiêu one-file), ta KHÔNG nhúng lõi paddle mà tải-lúc-chạy —
khớp cơ chế đã dùng cho CUDA runtime / WhisperX / VieNeu.

Tải lõi ``paddlepaddle-gpu`` KÈM phụ thuộc RIÊNG của nó
------------------------------------------------------
Bản build nhỏ LOẠI paddle khỏi bundle → các phụ thuộc RIÊNG của paddle (astor, decorator,
opt-einsum, networkx…) cũng KHÔNG được nhúng (paddleocr/paddlex không import chúng). Vì vậy
KHÔNG dùng ``--no-deps`` — phải tải paddle kèm deps để ``import paddle`` chạy được.

An toàn với deps DÙNG CHUNG (numpy/opencv/protobuf…): thêm ``paddle_dir`` vào CUỐI
``sys.path`` nên bản NHÚNG (đã kiểm thử cùng paddle) thắng; bản tải kèm chỉ dự phòng. Chỉ deps
RIÊNG của paddle (chỉ có ở ``paddle_dir``) mới được dùng từ đó → tránh xung đột phiên bản mà
vẫn đủ phụ thuộc.

Lõi paddle được cài vào ``models/paddle_runtime/`` (kho runtime CỐ ĐỊNH cạnh exe — xem
``model_store._frozen_runtime_root``), bền qua các lần chạy kể cả bản one-file.

Tách phần quyết định (thuần, test được) khỏi phần chạy pip (worker).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

#: Gói lõi paddle GPU cần tải (ghim version khớp bản build hiện tại — xem build_windows.bat).
PADDLE_RUNTIME_PACKAGE: Final[str] = "paddlepaddle-gpu==3.3.1"

#: Index CHÍNH của Paddle (bản CUDA 12.9). paddlepaddle-gpu chỉ có ở đây, không có trên PyPI.
PADDLE_INDEX_URL: Final[str] = "https://www.paddlepaddle.org.cn/packages/stable/cu129/"

#: Index phụ (PyPI) — phòng khi pip cần phân giải metadata phụ trợ.
PADDLE_EXTRA_INDEX_URL: Final[str] = "https://pypi.org/simple"

#: Tên thư mục con của kho runtime dùng cho lõi paddle tải về.
PADDLE_RUNTIME_DIRNAME: Final[str] = "paddle_runtime"

#: Ước tính dung lượng tải (GB) để hiện cho người dùng trước khi tải.
PADDLE_RUNTIME_DOWNLOAD_GB: Final[float] = 0.9

#: Thư mục gói mà pip trải ra khi cài ``paddlepaddle-gpu`` (dùng để dò đã cài xong chưa).
_PADDLE_PACKAGE_DIRNAME: Final[str] = "paddle"


class PaddleRuntimeStatus(StrEnum):
    """Trạng thái lõi ``paddlepaddle-gpu`` runtime."""

    BUNDLED = "bundled"
    """Đã nhúng sẵn trong bản đóng gói (build SUBEXT_BUNDLE_PADDLE=1) — không cần tải."""

    INSTALLED = "installed"
    """Đã tải về ``models/paddle_runtime/`` từ lần trước — dùng được ngay."""

    NEEDS_DOWNLOAD = "needs_download"
    """Chưa có — cần tải mới chạy được OCR (không có lõi paddle thì không nhận diện được)."""


@dataclass(frozen=True, slots=True)
class PaddleRuntimePlan:
    """Kế hoạch chuẩn bị lõi paddle.

    Attributes:
        status: Trạng thái hiện tại.
        paddle_dir: Thư mục sẽ tải lõi paddle vào (``models/paddle_runtime``).
        download_estimate_gb: Ước tính dung lượng tải.
    """

    status: PaddleRuntimeStatus
    paddle_dir: Path
    download_estimate_gb: float = PADDLE_RUNTIME_DOWNLOAD_GB

    @property
    def needs_download(self) -> bool:
        return self.status is PaddleRuntimeStatus.NEEDS_DOWNLOAD


def paddle_runtime_sys_path(paddle_dir: Path) -> list[Path]:
    """Thư mục cần THÊM VÀO CUỐI ``sys.path`` để ``import paddle`` thấy lõi đã tải.

    Thêm vào CUỐI (không phải đầu) để các phụ thuộc dùng chung (numpy, opencv…) vẫn ưu tiên
    bản NHÚNG đã kiểm thử; chỉ ``paddle`` (chỉ có trong thư mục này) mới lấy từ đây.

    Returns:
        ``[paddle_dir]`` nếu lõi paddle đã tải xong (tồn tại ``paddle_dir/paddle/``); ngược
        lại danh sách rỗng (để bootstrap không thêm đường dẫn vô ích).
    """
    if (paddle_dir / _PADDLE_PACKAGE_DIRNAME).is_dir():
        return [paddle_dir]
    return []


def is_paddle_runtime_installed(paddle_dir: Path) -> bool:
    """Lõi paddle đã tải xong chưa? (tồn tại thư mục gói ``paddle/``)."""
    return bool(paddle_runtime_sys_path(paddle_dir))


def evaluate_paddle_runtime(
    paddle_dir: Path, *, bundled: bool
) -> PaddleRuntimePlan:
    """Đánh giá trạng thái lõi paddle (hàm thuần).

    Args:
        paddle_dir: Thư mục ``models/paddle_runtime``.
        bundled: ``True`` nếu bản build đã nhúng sẵn paddle (SUBEXT_BUNDLE_PADDLE=1).
    """
    if bundled:
        return PaddleRuntimePlan(PaddleRuntimeStatus.BUNDLED, paddle_dir)
    if is_paddle_runtime_installed(paddle_dir):
        return PaddleRuntimePlan(PaddleRuntimeStatus.INSTALLED, paddle_dir)
    return PaddleRuntimePlan(PaddleRuntimeStatus.NEEDS_DOWNLOAD, paddle_dir)


def build_paddle_install_command(
    python_exe: str, paddle_dir: Path, target_python_version: str
) -> tuple[str, ...]:
    """Lệnh pip tải lõi ``paddlepaddle-gpu`` + PHỤ THUỘC RIÊNG cho ĐÚNG phiên bản Python bundled.

    QUAN TRỌNG — khớp phiên bản Python: wheel ``paddlepaddle-gpu`` phụ thuộc phiên bản CPython
    (cp311/cp312…). Bản đóng gói chạy Python BUNDLED, nhưng pip lại chạy bằng Python HỆ THỐNG
    (có thể khác phiên bản). Nếu tải nhầm wheel cho phiên bản khác → ``import paddle`` lỗi ABI.
    Dùng ``--python-version {target} --only-binary=:all:`` để pip tải ĐÚNG wheel cho Python
    bundled bất kể pip chạy bằng Python nào — thiết yếu để bản .exe chia sẻ chạy được trên máy
    người khác (Python hệ thống của họ thường khác phiên bản).

    KHÔNG dùng ``--no-deps``: khi bản build nhỏ LOẠI paddle khỏi bundle, các phụ thuộc RIÊNG
    của paddle (astor, decorator, opt-einsum, networkx…) cũng KHÔNG được nhúng. Phải tải kèm.
    Bootstrap thêm ``paddle_dir`` vào CUỐI ``sys.path`` nên deps DÙNG CHUNG vẫn ưu tiên bản
    nhúng; chỉ deps RIÊNG mới lấy từ đây.

    Args:
        python_exe: Python hệ thống dùng để CHẠY pip.
        paddle_dir: Thư mục đích ``models/paddle_runtime``.
        target_python_version: Phiên bản Python BUNDLED dạng ``"major.minor"`` (vd ``"3.11"``)
            — thường lấy từ ``f"{sys.version_info.major}.{sys.version_info.minor}"`` của app.
    """
    return (
        python_exe, "-m", "pip", "install",
        "--target", str(paddle_dir),
        # Tải wheel cho ĐÚNG phiên bản Python bundled (không phải phiên bản của pip đang chạy).
        "--python-version", target_python_version,
        "--only-binary=:all:",
        "--upgrade",
        PADDLE_RUNTIME_PACKAGE,
        "--index-url", PADDLE_INDEX_URL,
        "--extra-index-url", PADDLE_EXTRA_INDEX_URL,
    )


__all__ = [
    "PADDLE_EXTRA_INDEX_URL",
    "PADDLE_INDEX_URL",
    "PADDLE_RUNTIME_DIRNAME",
    "PADDLE_RUNTIME_DOWNLOAD_GB",
    "PADDLE_RUNTIME_PACKAGE",
    "PaddleRuntimePlan",
    "PaddleRuntimeStatus",
    "build_paddle_install_command",
    "evaluate_paddle_runtime",
    "is_paddle_runtime_installed",
    "paddle_runtime_sys_path",
]
