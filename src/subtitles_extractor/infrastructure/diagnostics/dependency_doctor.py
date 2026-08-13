"""Chẩn đoán & sửa phụ thuộc tuỳ chọn (Dependency Doctor).

Triết lý: KHÔNG tự ý ``pip install`` ngầm khi app chạy (rủi ro phá môi trường,
xung đột phiên bản torch CUDA/CPU, cần quyền admin, tải GB bất ngờ). Thay vào đó:
  * CHẨN ĐOÁN rõ cái gì thiếu/hỏng + nguyên nhân.
  * Đưa LỆNH cài cụ thể để người dùng tự chạy (hoặc bấm nút cài có xác nhận).
  * Phân biệt "thiếu package" (pip sửa được) vs "thiếu DLL hệ thống" (pip KHÔNG sửa).

Dùng cho các nguồn tuỳ chọn: WhisperX (STT). PaddleOCR/MPV là bắt buộc, kiểm ở
bootstrap riêng.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
from subtitles_extractor.infrastructure.process.hidden_process import no_window_kwargs
import sys
from dataclasses import dataclass
from enum import Enum


class DependencyStatus(Enum):
    """Trạng thái một phụ thuộc."""

    OK = "ok"
    MISSING_PACKAGE = "missing_package"   # chưa cài → pip install được
    BROKEN_RUNTIME = "broken_runtime"     # cài rồi nhưng lỗi nạp (vd thiếu DLL)
    NOT_CHECKED = "not_checked"


@dataclass(frozen=True, slots=True)
class DependencyReport:
    """Kết quả chẩn đoán một phụ thuộc.

    Attributes:
        name:         Tên hiển thị (vd "WhisperX").
        status:       Trạng thái.
        detail:       Mô tả ngắn cho người dùng.
        install_hint: Lệnh/cách khắc phục (rỗng nếu OK).
        pip_args:     Đối số pip để cài tự động (rỗng nếu không nên auto-install).
    """

    name: str
    status: DependencyStatus
    detail: str = ""
    install_hint: str = ""
    pip_args: tuple[str, ...] = ()

    @property
    def is_ok(self) -> bool:
        return self.status == DependencyStatus.OK


def _module_installed(module_name: str) -> bool:
    """Kiểm module đã cài chưa mà KHÔNG import (tránh side-effect/lỗi nạp)."""
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def check_ffmpeg_cli() -> DependencyReport:
    """Kiểm ffmpeg.exe trong PATH (bắt buộc cho trích audio/video)."""
    from subtitles_extractor.infrastructure.media import (
        FFMPEG_DEPENDENT_FEATURES,
        find_ffmpeg,
    )

    if find_ffmpeg() is not None:
        return DependencyReport("FFmpeg (CLI)", DependencyStatus.OK)
    # [v3.23.306] Nêu RÕ tính năng nào hỏng + cách sửa ĐÚNG LICENSE. Bản dựng ffmpeg
    # phổ biến là GPL — bundle kèm sẽ phá license-clean thương mại của ứng dụng.
    return DependencyReport(
        "FFmpeg (CLI)",
        DependencyStatus.BROKEN_RUNTIME,
        detail=(
            "Không tìm thấy ffmpeg (cả bản trong vendor/ffmpeg lẫn PATH). "
            "Ảnh hưởng: " + "; ".join(FFMPEG_DEPENDENT_FEATURES) + ". "
            "Lõi OCR hardsub và trình phát video KHÔNG bị ảnh hưởng."
        ),
        install_hint=(
            "Tải bản dựng ffmpeg LGPL (tên tệp có 'lgpl', KHÔNG phải 'gpl') rồi đặt "
            "ffmpeg.exe + ffprobe.exe vào thư mục 'vendor/ffmpeg/'. Xác minh bằng "
            "tools/check_media_licenses.py."
        ),
    )


def check_whisperx() -> DependencyReport:
    """Chẩn đoán WhisperX cho tính năng phiên âm giọng nói.

    [v3.23.335] SỬA HAI LỖI NGHIÊM TRỌNG của bản trước:

    1. **Kiểm sai chỗ.** Trước đây dùng ``_module_installed("whisperx")`` — chỉ xét
       tiến trình HIỆN TẠI. Nhưng từ v3.23.333 WhisperX cố ý được cài ở môi trường
       RIÊNG ``whisperx_env``, nên tạo môi trường xong vẫn bị báo "chưa cài".
    2. **Gợi ý cài nguy hiểm.** Trước đây đề xuất ``pip install whisperx`` kèm
       ``pip_args``, mà :func:`install_package` chạy pip bằng ``sys.executable`` —
       tức cài thẳng vào môi trường CHÍNH. Làm vậy sẽ hạ cấp ``huggingface-hub`` từ
       1.x xuống <1.0 và có thể làm hỏng VieNeu-TTS/PaddleOCR. Nay BỎ HẲN ``pip_args``
       để không có nút tự cài, chỉ hướng dẫn tạo môi trường riêng.
    """
    from subtitles_extractor.infrastructure.stt.whisperx_adapter import (
        WHISPERX_ENV_DIRNAME,
        WHISPERX_PYTHON_ENV_VAR,
        resolve_whisperx_python,
    )

    python_exe = resolve_whisperx_python()
    if python_exe is None:
        return DependencyReport(
            "WhisperX (phiên âm giọng nói)",
            DependencyStatus.MISSING_PACKAGE,
            detail=(
                "Chưa có môi trường WhisperX — tính năng phiên âm chưa dùng được. "
                "WhisperX phải cài RIÊNG vì nó yêu cầu huggingface-hub < 1.0 trong khi "
                "ứng dụng dùng bản 1.x; cài chung sẽ làm hỏng VieNeu-TTS/PaddleOCR."
            ),
            install_hint=(
                f"Chạy trong thư mục dự án:\n"
                f"  python -m venv {WHISPERX_ENV_DIRNAME}\n"
                f"  {WHISPERX_ENV_DIRNAME}\\Scripts\\python -m pip install "
                f"torch torchaudio torchvision "
                f"--index-url https://download.pytorch.org/whl/cu129\n"
                f"  {WHISPERX_ENV_DIRNAME}\\Scripts\\python -m pip install whisperx\n"
                f"Hoặc đặt biến {WHISPERX_PYTHON_ENV_VAR} trỏ tới python.exe có WhisperX."
            ),
            # KHÔNG đặt pip_args: cài vào môi trường chính sẽ làm hỏng ứng dụng.
        )

    return DependencyReport(
        "WhisperX (phiên âm giọng nói)",
        DependencyStatus.OK,
        detail=f"Dùng môi trường riêng: {python_exe}",
    )


def install_package(pip_args: tuple[str, ...], timeout_sec: float = 1800.0) -> tuple[bool, str]:
    """Cài package qua pip (CHỈ gọi sau khi người dùng xác nhận tường minh).

    Args:
        pip_args:    Đối số truyền cho ``pip install`` (vd ``("whisperx",)``).
        timeout_sec: Giới hạn thời gian (mặc định 30 phút — model/torch rất nặng).

    Returns:
        ``(thành_công, log_output)``.
    """
    # [v3.23.335] Chặn cứng các gói KHÔNG được cài vào môi trường chính.
    _FORBIDDEN = {"whisperx", "torch", "torchaudio", "torchvision"}
    for arg in pip_args:
        name = arg.split("=")[0].split(">")[0].split("<")[0].strip().lower()
        if name in _FORBIDDEN:
            return False, (
                f"Từ chối cài '{name}' vào môi trường chính: nó sẽ đổi phiên bản các "
                "gói đang dùng (huggingface-hub/CUDA) và có thể làm hỏng ứng dụng. "
                "Hãy tạo môi trường riêng 'whisperx_env' theo hướng dẫn."
            )

    if not pip_args:
        return False, "Không có gói nào để cài."
    command = [sys.executable, "-m", "pip", "install", *pip_args]
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout_sec,
            **no_window_kwargs(),
        )
    except subprocess.TimeoutExpired:
        return False, "Quá thời gian cài đặt (>30 phút)."
    except OSError as exc:
        return False, f"Không chạy được pip: {exc}"
    output = (completed.stdout or "") + (completed.stderr or "")
    return completed.returncode == 0, output[-4000:]


__all__ = [
    "DependencyStatus",
    "DependencyReport",
    "check_ffmpeg_cli",
    "check_whisperx",
    "run_full_diagnosis",
    "install_package",
]
