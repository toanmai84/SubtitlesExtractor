"""Test kế hoạch bật GPU cho VieNeu-TTS — v3.23.341.

PHÁT HIỆN TỪ MÃ NGUỒN VieNeu 3.2.3 (đọc trực tiếp, không đoán):

* ``v3turbo.py``: ``backend "auto"`` = ONNX trên CPU, PyTorch trên GPU.
* ``onnx_runtime_lite.py``: ``prov = ["CPUExecutionProvider"]`` — **ghi cứng**.

Nên ``onnxruntime-gpu`` KHÔNG bật được GPU (câu hỏi rất tự nhiên nhưng câu trả lời là
không), và GPU buộc đi đường PyTorch.

Điểm thuận lợi được canh giữ ở đây: ``whisperx_env`` đã có sẵn torch/torchaudio/
transformers/safetensors, nên bật GPU cho TTS **không cần tải lại ~3GB**.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from subtitles_extractor.infrastructure.stt.whisperx_adapter import (
    WHISPERX_ENV_DIRNAME,
    WHISPERX_PYTHON_ENV_VAR,
)
from subtitles_extractor.infrastructure.tts.vieneu_gpu_plan import (
    TORCH_PATH_REQUIREMENTS,
    VIENEU_EXTRA_PACKAGES,
    GpuTtsStatus,
    build_gpu_tts_plan,
    summarise_plan,
    why_onnxruntime_gpu_does_not_help,
)

_TORCH_READY = frozenset(TORCH_PATH_REQUIREMENTS)
_ALL_READY = _TORCH_READY | frozenset(VIENEU_EXTRA_PACKAGES)


@pytest.fixture
def isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Giả lập đã có môi trường riêng."""
    python_exe = tmp_path / "python.exe"
    python_exe.write_text("", encoding="utf-8")
    monkeypatch.setenv(WHISPERX_PYTHON_ENV_VAR, str(python_exe))
    return python_exe


@pytest.fixture
def no_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(WHISPERX_PYTHON_ENV_VAR, raising=False)


# ── Trạng thái ───────────────────────────────────────────────────────────────
def test_no_environment_points_to_whisperx_installer(no_env: None) -> None:
    """Chưa có môi trường -> hướng người dùng tới nút cài WhisperX (cùng torch)."""
    plan = build_gpu_tts_plan()
    assert plan.status is GpuTtsStatus.NO_ENVIRONMENT
    assert WHISPERX_ENV_DIRNAME in plan.note
    assert not plan.is_ready


def test_torch_present_needs_only_vieneu(isolated_env: Path) -> None:
    """ĐIỂM THUẬN LỢI CHÍNH: có torch rồi thì chỉ cần thêm vieneu."""
    plan = build_gpu_tts_plan(installed=_TORCH_READY)
    assert plan.status is GpuTtsStatus.NEEDS_VIENEU
    assert set(plan.missing_packages) == set(VIENEU_EXTRA_PACKAGES)


def test_torch_present_avoids_redownloading_gigabytes(isolated_env: Path) -> None:
    """Không được đề xuất tải lại torch khi nó đã có — đó là ~3GB."""
    plan = build_gpu_tts_plan(installed=_TORCH_READY)
    assert plan.download_estimate_gb < 0.1
    assert "KHÔNG phải tải lại torch" in plan.note
    # Lệnh cài không được chứa torch.
    for command in plan.install_commands:
        assert "torch" not in command


def test_everything_present_is_ready(isolated_env: Path) -> None:
    plan = build_gpu_tts_plan(installed=_ALL_READY)
    assert plan.is_ready
    assert plan.missing_packages == ()
    assert plan.install_commands == ()
    assert plan.download_estimate_gb == 0.0


def test_missing_torch_estimates_full_download(isolated_env: Path) -> None:
    plan = build_gpu_tts_plan(installed=frozenset())
    assert plan.download_estimate_gb >= 3.0


# ── Lệnh cài ─────────────────────────────────────────────────────────────────
def test_install_commands_target_isolated_python(isolated_env: Path) -> None:
    """Mọi lệnh phải chạy bằng Python của môi trường RIÊNG, không phải môi trường chính."""
    plan = build_gpu_tts_plan(installed=_TORCH_READY)
    assert plan.install_commands
    for command in plan.install_commands:
        assert command[0] == str(isolated_env)


def test_torch_install_uses_cuda_index(isolated_env: Path) -> None:
    """Thiếu torch thì phải cài từ index CUDA, không phải PyPI (bản CPU)."""
    plan = build_gpu_tts_plan(installed=frozenset())
    torch_commands = [c for c in plan.install_commands if "torch" in c]
    assert torch_commands
    assert any("cu129" in part for part in torch_commands[0])


# ── Yêu cầu đúng theo mã nguồn VieNeu ────────────────────────────────────────
def test_requirements_match_vieneu_pytorch_imports() -> None:
    """Bốn gói này là đúng những gì đường PyTorch của VieNeu import."""
    assert set(TORCH_PATH_REQUIREMENTS) == {
        "torch", "torchaudio", "transformers", "safetensors"
    }


def test_vieneu_extras_include_g2p() -> None:
    """``sea-g2p`` cần cho chuyển chữ sang âm tiếng Việt."""
    assert "vieneu" in VIENEU_EXTRA_PACKAGES
    assert "sea-g2p" in VIENEU_EXTRA_PACKAGES


# ── Giải thích onnxruntime-gpu ───────────────────────────────────────────────
def test_explains_hardcoded_cpu_provider() -> None:
    """Phải nêu BẰNG CHỨNG từ mã nguồn, không chỉ khẳng định."""
    text = why_onnxruntime_gpu_does_not_help()
    assert "CPUExecutionProvider" in text
    assert "onnx_runtime_lite" in text


def test_explains_module_name_collision() -> None:
    """onnxruntime-gpu thay thế onnxruntime CPU — ảnh hưởng cả fastembed."""
    text = why_onnxruntime_gpu_does_not_help()
    assert "fastembed" in text


# ── Trình bày ────────────────────────────────────────────────────────────────
def test_summary_includes_commands_when_needed(isolated_env: Path) -> None:
    summary = summarise_plan(build_gpu_tts_plan(installed=_TORCH_READY))
    assert "pip install" in summary


def test_summary_is_short_when_ready(isolated_env: Path) -> None:
    summary = summarise_plan(build_gpu_tts_plan(installed=_ALL_READY))
    assert "dùng được ngay" in summary
    assert "pip install" not in summary
