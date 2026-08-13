"""Dò khả năng chạy VieNeu-TTS trên GPU và lập kế hoạch cài đặt (thuần, testable).

PHÁT HIỆN QUAN TRỌNG TỪ MÃ NGUỒN VieNeu 3.2.3
==============================================
Đọc thẳng ``vieneu/v3turbo.py`` và ``_v3_turbo_engine/onnx_runtime_lite.py``::

    backend: str = "auto"   # "auto" → ONNX on CPU, PyTorch on GPU
    use_onnx = backend == "onnx" or (backend == "auto" and dev_type == "cpu")
    ...
    prov = ["CPUExecutionProvider"]        # <-- GHI CỨNG trong engine ONNX

Hệ quả:

1. **``onnxruntime-gpu`` KHÔNG giúp gì.** Engine ONNX của VieNeu ghi cứng
   ``CPUExecutionProvider``; nó không nhận tham số provider nào. Cài
   ``onnxruntime-gpu`` chỉ làm nặng thêm mà không đổi được gì — và còn xung đột với
   ``onnxruntime`` bản CPU mà ứng dụng đang dùng (cùng tên module).
2. **GPU buộc đi đường PyTorch** (``VieNeuTTSv3Turbo``), cần ``torch``, ``torchaudio``,
   ``transformers``, ``safetensors``.
3. Đường PyTorch còn có **gộp lô tĩnh** (``max_batch_size=32``) — với tập 56 câu thì
   đây là nguồn tăng tốc lớn hơn cả việc chạy trên GPU.

VÌ SAO PHẢI DÙNG MÔI TRƯỜNG RIÊNG
=================================
``torch`` nạp CUDA runtime riêng, đụng ``paddlepaddle-gpu`` trong môi trường chính —
đúng lỗi mà adapter đang né bằng ``force_cpu=True`` (xem chú thích ``WinError 127``).

**Điểm thuận lợi:** ``whisperx_env`` (tạo từ v3.23.333) ĐÃ có sẵn cả bốn gói vì WhisperX
cần chúng. Nên bật GPU cho TTS chỉ cần thêm ``vieneu`` vào môi trường đó — **không phải
tải lại ~3GB torch**.
"""

from __future__ import annotations

import logging
import subprocess
from subtitles_extractor.infrastructure.process.hidden_process import no_window_kwargs
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from subtitles_extractor.infrastructure.stt.whisperx_adapter import (
    WHISPERX_ENV_DIRNAME,
    resolve_whisperx_python,
)

logger = logging.getLogger(__name__)

#: Gói mà đường PyTorch của VieNeu cần (đã đối chiếu import trong mã nguồn gói).
TORCH_PATH_REQUIREMENTS: Final[tuple[str, ...]] = (
    "torch",
    "torchaudio",
    "transformers",
    "safetensors",
)

#: Gói cần thêm vào môi trường riêng để chạy được VieNeu ở đó.
#:
#: [v3.23.346] Dùng ``vieneu[legacy]`` chứ KHÔNG phải ``vieneu`` trần. Metadata của gói
#: khai báo ``transformers`` chỉ ở extra ``legacy``, nên bản trần KHÔNG cài transformers
#: — mà đường PyTorch (đường DUY NHẤT chạy được GPU) lại cần nó. Log thực tế cho thấy
#: hậu quả: worker thất bại 55 lần liên tiếp với "Try: pip install transformers -U".
#:
#: [v3.23.373] GHIM TRẦN ``transformers``. WhisperX cài ``torch`` bản CUDA rồi bị chính
#: whisperx ghim xuống bản cũ hơn; nếu ``vieneu[legacy]`` kéo ``transformers`` MỚI NHẤT thì
#: nó import các symbol torch quá mới (``ScalingType`` từ ``torch.nn.functional``) → worker
#: GPU chết bằng ImportError và rơi về ONNX/CPU. Trần này giữ ``transformers`` khớp torch
#: mà whisperx_env thực có, để đường PyTorch/GPU import được.
VIENEU_TRANSFORMERS_SPEC: Final[str] = "transformers<4.56"
VIENEU_INSTALL_SPECS: Final[tuple[str, ...]] = (
    "vieneu[legacy]",
    "sea-g2p",
    VIENEU_TRANSFORMERS_SPEC,
)

#: Tên gói dùng để KIỂM đã cài hay chưa (bỏ phần extra trong ngoặc vuông).
VIENEU_EXTRA_PACKAGES: Final[tuple[str, ...]] = ("vieneu", "sea-g2p")


class GpuTtsStatus(StrEnum):
    """Trạng thái khả năng chạy TTS trên GPU."""

    READY = "ready"
    """Môi trường riêng có đủ torch và vieneu — dùng được ngay."""

    NEEDS_VIENEU = "needs_vieneu"
    """Có torch nhưng thiếu vieneu — chỉ cần thêm gói, KHÔNG phải tải lại torch."""

    NO_ENVIRONMENT = "no_environment"
    """Chưa có môi trường riêng nào."""


@dataclass(frozen=True, slots=True)
class GpuTtsPlan:
    """Kế hoạch bật GPU cho VieNeu-TTS.

    Attributes:
        status: Trạng thái hiện tại.
        python_exe: Trình thông dịch của môi trường riêng (``None`` nếu chưa có).
        missing_packages: Gói còn thiếu.
        install_commands: Lệnh cần chạy để hoàn tất.
        download_estimate_gb: Dung lượng cần tải thêm (GB).
        note: Giải thích cho người dùng.
    """

    status: GpuTtsStatus
    python_exe: str | None
    missing_packages: tuple[str, ...]
    install_commands: tuple[tuple[str, ...], ...]
    download_estimate_gb: float
    note: str

    @property
    def is_ready(self) -> bool:
        """``True`` nếu dùng được GPU ngay."""
        return self.status is GpuTtsStatus.READY


def installed_packages(python_exe: str) -> frozenset[str]:
    """Đọc danh sách gói của một môi trường.

    Args:
        python_exe: Trình thông dịch cần kiểm.

    Returns:
        Tập tên gói (chuẩn hoá về chữ thường, gạch ngang).
    """
    try:
        result = subprocess.run(
            [python_exe, "-m", "pip", "list", "--format=freeze"],
            capture_output=True, text=True, timeout=120, check=False,
            **no_window_kwargs(),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.debug("Không đọc được gói của %s: %s", python_exe, exc)
        return frozenset()

    names: set[str] = set()
    for line in result.stdout.splitlines():
        if "==" in line:
            names.add(line.partition("==")[0].strip().lower().replace("_", "-"))
    return frozenset(names)


def build_gpu_tts_plan(installed: frozenset[str] | None = None) -> GpuTtsPlan:
    """Lập kế hoạch bật GPU cho VieNeu-TTS.

    Args:
        installed: Gói đang có trong môi trường riêng. ``None`` = tự đọc.

    Returns:
        Kế hoạch kèm lệnh cần chạy và dung lượng phải tải.
    """
    python_exe = resolve_whisperx_python()

    if python_exe is None:
        return GpuTtsPlan(
            status=GpuTtsStatus.NO_ENVIRONMENT,
            python_exe=None,
            missing_packages=TORCH_PATH_REQUIREMENTS + VIENEU_EXTRA_PACKAGES,
            install_commands=(),
            download_estimate_gb=3.0,
            note=(
                f"Chưa có môi trường riêng '{WHISPERX_ENV_DIRNAME}'. Bấm "
                "“⬇️ Cài WhisperX tự động” trong nhóm Giọng nói (STT) để tạo — nó cài "
                "torch bản CUDA, thứ mà TTS trên GPU cũng cần."
            ),
        )

    if installed is None:
        installed = installed_packages(python_exe)

    missing_torch = tuple(
        name for name in TORCH_PATH_REQUIREMENTS if name not in installed
    )
    missing_vieneu = tuple(
        name for name in VIENEU_EXTRA_PACKAGES if name not in installed
    )

    if not missing_torch and not missing_vieneu:
        return GpuTtsPlan(
            status=GpuTtsStatus.READY,
            python_exe=python_exe,
            missing_packages=(),
            install_commands=(),
            download_estimate_gb=0.0,
            note=f"Môi trường riêng đã đủ: {python_exe}",
        )

    missing = missing_torch + missing_vieneu
    commands: list[tuple[str, ...]] = []
    if missing_torch:
        commands.append(
            (python_exe, "-m", "pip", "install", "torch", "torchaudio",
             "--index-url", "https://download.pytorch.org/whl/cu129")
        )
    if missing_vieneu:
        commands.append((python_exe, "-m", "pip", "install", *VIENEU_INSTALL_SPECS))

    # torch là phần nặng; nếu đã có thì chỉ còn vài chục MB.
    estimate = 3.0 if missing_torch else 0.05

    if not missing_torch:
        note = (
            "Môi trường riêng đã có torch (WhisperX cài sẵn) — chỉ cần thêm "
            f"{', '.join(missing_vieneu)}, khoảng {estimate * 1024:.0f} MB. "
            "KHÔNG phải tải lại torch."
        )
    else:
        note = (
            f"Còn thiếu: {', '.join(missing)}. Cần tải khoảng {estimate:.0f} GB."
        )

    return GpuTtsPlan(
        status=GpuTtsStatus.NEEDS_VIENEU if not missing_torch else GpuTtsStatus.NO_ENVIRONMENT,
        python_exe=python_exe,
        missing_packages=missing,
        install_commands=tuple(commands),
        download_estimate_gb=estimate,
        note=note,
    )


def repair_command(python_exe: str) -> tuple[str, ...]:
    """Lệnh sửa môi trường đã cài bằng bản ``vieneu`` TRẦN (thiếu transformers).

    Người dùng cài bằng nút của v3.23.343/344 sẽ có ``vieneu`` nhưng KHÔNG có
    ``transformers`` đúng bản — worker báo "Try: pip install transformers -U" và hỏng
    mọi câu. Chạy lại lệnh này là đủ, không cần xoá môi trường.

    Args:
        python_exe: Trình thông dịch của môi trường riêng.

    Returns:
        Lệnh pip cần chạy.
    """
    return (python_exe, "-m", "pip", "install", "--upgrade", *VIENEU_INSTALL_SPECS)


def why_onnxruntime_gpu_does_not_help() -> str:
    """Giải thích vì sao ``onnxruntime-gpu`` KHÔNG bật được GPU cho VieNeu.

    Câu hỏi này rất tự nhiên (VieNeu mặc định chạy ONNX, nên đổi sang bản GPU của
    onnxruntime nghe hợp lý) nhưng câu trả lời là không — và lý do nằm trong mã nguồn.

    Returns:
        Giải thích nhiều dòng.
    """
    return (
        "onnxruntime-gpu KHÔNG bật được GPU cho VieNeu, vì hai lý do đo được từ mã "
        "nguồn gói:\n"
        "• Engine ONNX của VieNeu GHI CỨNG providers = [\"CPUExecutionProvider\"] "
        "(onnx_runtime_lite.py) và không nhận tham số provider nào.\n"
        "• Chính VieNeu ghi trong chú thích: backend \"auto\" nghĩa là ONNX trên CPU, "
        "PyTorch trên GPU — tức GPU chỉ đi qua đường PyTorch.\n"
        "Ngoài ra onnxruntime-gpu dùng CÙNG tên module với onnxruntime bản CPU mà ứng "
        "dụng đang dùng, nên cài chung sẽ thay thế nó và ảnh hưởng cả fastembed."
    )


def summarise_plan(plan: GpuTtsPlan) -> str:
    """Tóm tắt kế hoạch thành thông điệp hiển thị.

    Args:
        plan: Kế hoạch đã lập.

    Returns:
        Chuỗi nhiều dòng.
    """
    header = {
        GpuTtsStatus.READY: "✓ TTS trên GPU dùng được ngay.",
        GpuTtsStatus.NEEDS_VIENEU: "⚠ Gần đủ — chỉ thiếu gói VieNeu.",
        GpuTtsStatus.NO_ENVIRONMENT: "⛔ Chưa sẵn sàng.",
    }[plan.status]

    lines = [header, plan.note]
    if plan.install_commands:
        lines.append("Lệnh cần chạy:")
        lines.extend(f"  {' '.join(command)}" for command in plan.install_commands)
    return "\n".join(lines)


__all__ = [
    "TORCH_PATH_REQUIREMENTS",
    "VIENEU_EXTRA_PACKAGES",
    "VIENEU_INSTALL_SPECS",
    "GpuTtsPlan",
    "GpuTtsStatus",
    "build_gpu_tts_plan",
    "installed_packages",
    "repair_command",
    "summarise_plan",
    "why_onnxruntime_gpu_does_not_help",
]
