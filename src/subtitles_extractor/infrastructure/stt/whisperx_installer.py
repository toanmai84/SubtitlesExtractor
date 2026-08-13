"""Cài WhisperX vào MÔI TRƯỜNG RIÊNG ngay từ trong ứng dụng.

VÌ SAO an toàn (khác hẳn nút cài cũ)
====================================
v3.23.335 vừa gỡ bỏ nút cài cũ vì nó chạy ``pip install whisperx`` bằng
``sys.executable`` — tức cài thẳng vào môi trường CHÍNH, làm hạ cấp ``huggingface-hub``
từ 1.x xuống <1.0 và có thể hỏng VieNeu-TTS/PaddleOCR.

Module này khác ở chỗ **mọi lệnh pip đều chạy bằng trình thông dịch của môi trường
riêng** ``whisperx_env``. Môi trường chính không bị đụng tới. Có kiểm tra bất biến để
chắc chắn điều đó — xem :func:`assert_targets_isolated_env`.

Các bước cài
------------
1. Tạo venv ``whisperx_env`` bằng Python hệ thống.
2. Nâng cấp pip trong venv đó.
3. Cài ``torch/torchaudio/torchvision`` từ **index CUDA 12.9** (khớp paddle ``cu129``).
   Bỏ qua bước này thì PyPI sẽ trả bản CPU và WhisperX chạy rất chậm.
4. Cài ``whisperx``.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from subtitles_extractor.infrastructure.stt.whisperx_adapter import (
    WHISPERX_ENV_DIRNAME,
)

logger = logging.getLogger(__name__)

#: Index chứa bản torch biên dịch cho CUDA 12.9 — khớp paddlepaddle-gpu cu129.
TORCH_CUDA_INDEX: Final[str] = "https://download.pytorch.org/whl/cu129"

#: Dung lượng tải ước tính (GB) — để cảnh báo trước khi bắt đầu.
ESTIMATED_DOWNLOAD_GB: Final[float] = 3.0


class WhisperXInstallError(Exception):
    """Không cài được WhisperX."""


@dataclass(frozen=True, slots=True)
class InstallStep:
    """Một bước trong quá trình cài.

    Attributes:
        label: Mô tả hiển thị cho người dùng.
        command: Lệnh đầy đủ để chạy.
        weight: Trọng số thời gian (dùng để ước lượng tiến độ).
    """

    label: str
    command: list[str]
    weight: float = 1.0


def find_system_python() -> str | None:
    """Tìm Python hệ thống để tạo venv.

    Bản đóng gói không có Python bên trong (``sys.executable`` là chính tệp ``.exe``),
    nên phải tìm Python cài ngoài.

    Returns:
        Đường dẫn Python, hoặc ``None`` nếu không tìm thấy.
    """
    if not getattr(sys, "frozen", False):
        return sys.executable

    for name in ("python", "python3", "py"):
        found = shutil.which(name)
        if found:
            return found
    return None


def env_python_path(project_root: Path) -> Path:
    """Đường dẫn trình thông dịch bên trong môi trường riêng.

    Args:
        project_root: Thư mục gốc dự án (nơi đặt ``whisperx_env``).

    Returns:
        Đường dẫn tới ``python.exe`` (Windows) hoặc ``bin/python`` (Linux/macOS).
    """
    env_dir = project_root / WHISPERX_ENV_DIRNAME
    if sys.platform == "win32":
        return env_dir / "Scripts" / "python.exe"
    return env_dir / "bin" / "python"


def assert_targets_isolated_env(command: list[str], project_root: Path) -> None:
    """Chặn cứng: mọi lệnh ``pip install`` phải chạy bằng Python của môi trường riêng.

    Đây là bất biến quan trọng nhất của module. Nếu một lệnh cài lọt qua với trình
    thông dịch của môi trường chính, ứng dụng sẽ bị hỏng đúng như sự cố v3.23.335.

    Args:
        command: Lệnh sắp chạy.
        project_root: Thư mục gốc dự án.

    Raises:
        WhisperXInstallError: Khi lệnh cài không trỏ vào môi trường riêng.
    """
    if "install" not in command:
        return
    expected = str(env_python_path(project_root))
    if command[0] != expected:
        raise WhisperXInstallError(
            "Từ chối chạy lệnh cài không trỏ vào môi trường riêng "
            f"({WHISPERX_ENV_DIRNAME}). Lệnh: {command[0]}"
        )


def build_install_steps(project_root: Path, system_python: str) -> list[InstallStep]:
    """Dựng danh sách bước cài (hàm thuần — kiểm thử được, không chạy gì).

    Args:
        project_root: Thư mục gốc dự án.
        system_python: Python hệ thống dùng để tạo venv.

    Returns:
        Các bước theo đúng thứ tự thực hiện.
    """
    env_python = str(env_python_path(project_root))
    env_dir = str(project_root / WHISPERX_ENV_DIRNAME)

    return [
        InstallStep(
            "Tạo môi trường riêng…",
            [system_python, "-m", "venv", env_dir],
            weight=0.5,
        ),
        InstallStep(
            "Nâng cấp pip…",
            [env_python, "-m", "pip", "install", "--upgrade", "pip"],
            weight=0.5,
        ),
        InstallStep(
            f"Tải torch bản CUDA 12.9 (~{ESTIMATED_DOWNLOAD_GB:.0f}GB, lâu nhất)…",
            [env_python, "-m", "pip", "install",
             "torch", "torchaudio", "torchvision",
             "--index-url", TORCH_CUDA_INDEX],
            weight=8.0,
        ),
        InstallStep(
            "Cài WhisperX…",
            [env_python, "-m", "pip", "install", "whisperx"],
            weight=1.0,
        ),
    ]


def install_whisperx(
    project_root: Path,
    *,
    on_progress: object = None,
    is_cancelled: object = None,
    timeout_per_step: float = 3600.0,
) -> Path:
    """Thực hiện cài WhisperX vào môi trường riêng.

    Args:
        project_root: Thư mục gốc dự án.
        on_progress: Callback ``(ratio: float, label: str)`` báo tiến độ.
        is_cancelled: Callback trả ``True`` để dừng.
        timeout_per_step: Giới hạn thời gian mỗi bước (giây).

    Returns:
        Đường dẫn Python của môi trường vừa tạo.

    Raises:
        WhisperXInstallError: Khi thiếu Python hệ thống, bị huỷ, hoặc một bước thất bại.
    """
    system_python = find_system_python()
    if system_python is None:
        raise WhisperXInstallError(
            "Không tìm thấy Python trên máy. Hãy cài Python 3.10–3.13 từ python.org "
            "(nhớ tick “Add to PATH”) rồi thử lại."
        )

    steps = build_install_steps(project_root, system_python)
    total_weight = sum(step.weight for step in steps)
    completed_weight = 0.0

    for step in steps:
        if is_cancelled is not None and is_cancelled():  # type: ignore[operator]
            raise WhisperXInstallError("Đã huỷ cài đặt theo yêu cầu.")

        assert_targets_isolated_env(step.command, project_root)
        if on_progress is not None:
            on_progress(completed_weight / total_weight, step.label)  # type: ignore[operator]

        logger.info("Cài WhisperX — %s", step.label)
        try:
            result = subprocess.run(
                step.command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_per_step,
                check=False,
                **_no_window_kwargs(),  # type: ignore[arg-type]
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise WhisperXInstallError(f"{step.label} thất bại: {exc}") from exc

        if result.returncode != 0:
            tail = (result.stderr or result.stdout or "").strip().splitlines()[-3:]
            raise WhisperXInstallError(
                f"{step.label} thất bại (mã {result.returncode}):\n" + "\n".join(tail)
            )

        completed_weight += step.weight

    env_python = env_python_path(project_root)
    if not env_python.is_file():
        raise WhisperXInstallError(
            "Cài xong nhưng không thấy trình thông dịch trong môi trường riêng."
        )

    if on_progress is not None:
        on_progress(1.0, "Hoàn tất.")  # type: ignore[operator]
    logger.info("Đã cài WhisperX vào %s", env_python)
    return env_python


def _no_window_kwargs() -> dict[str, object]:
    """Ẩn cửa sổ console trên Windows (không ảnh hưởng nền tảng khác)."""
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NO_WINDOW}
    return {}


__all__ = [
    "ESTIMATED_DOWNLOAD_GB",
    "TORCH_CUDA_INDEX",
    "InstallStep",
    "WhisperXInstallError",
    "assert_targets_isolated_env",
    "build_install_steps",
    "env_python_path",
    "find_system_python",
    "install_whisperx",
]
