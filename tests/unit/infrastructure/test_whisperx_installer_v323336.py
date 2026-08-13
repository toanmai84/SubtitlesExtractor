"""Test trình cài WhisperX vào môi trường riêng — v3.23.336.

BỐI CẢNH: v3.23.335 vừa GỠ BỎ nút cài cũ vì nó chạy ``pip install whisperx`` bằng
``sys.executable`` — cài thẳng vào môi trường CHÍNH, làm hạ cấp ``huggingface-hub`` và
có thể hỏng VieNeu-TTS/PaddleOCR.

Nút mới an toàn vì **mọi lệnh pip đều chạy bằng Python của ``whisperx_env``**. Bất biến
đó là thứ quan trọng nhất cần canh giữ — nếu vỡ, sự cố cũ tái diễn.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from subtitles_extractor.infrastructure.stt.whisperx_adapter import (
    WHISPERX_ENV_DIRNAME,
)
from subtitles_extractor.infrastructure.stt.whisperx_installer import (
    TORCH_CUDA_INDEX,
    WhisperXInstallError,
    assert_targets_isolated_env,
    build_install_steps,
    env_python_path,
    find_system_python,
)

_ROOT = Path("D:/Du An") if sys.platform == "win32" else Path("/du-an")
_SYSTEM_PYTHON = "C:/Python312/python.exe"


@pytest.fixture
def steps() -> list:
    return build_install_steps(_ROOT, _SYSTEM_PYTHON)


# ── BẤT BIẾN AN TOÀN (quan trọng nhất) ───────────────────────────────────────
def test_every_install_command_targets_isolated_env(steps: list) -> None:
    """Mọi lệnh ``pip install`` PHẢI chạy bằng Python của môi trường riêng.

    Vỡ bất biến này là tái diễn đúng sự cố v3.23.335.
    """
    expected = str(env_python_path(_ROOT))
    for step in steps:
        if "install" in step.command:
            assert step.command[0] == expected, step.label


def test_rejects_command_pointing_at_main_environment() -> None:
    """Chặn cứng lệnh cài trỏ vào môi trường chính."""
    with pytest.raises(WhisperXInstallError, match=WHISPERX_ENV_DIRNAME):
        assert_targets_isolated_env(
            [sys.executable, "-m", "pip", "install", "whisperx"], _ROOT
        )


def test_allows_non_install_commands() -> None:
    """Lệnh tạo venv dùng Python hệ thống là hợp lệ — nó không cài gì."""
    assert_targets_isolated_env(
        [_SYSTEM_PYTHON, "-m", "venv", str(_ROOT / WHISPERX_ENV_DIRNAME)], _ROOT
    )


def test_venv_is_created_inside_project_root(steps: list) -> None:
    venv_step = steps[0]
    assert "venv" in venv_step.command
    assert WHISPERX_ENV_DIRNAME in venv_step.command[-1]


# ── Nội dung các bước ────────────────────────────────────────────────────────
def test_torch_uses_cuda_index(steps: list) -> None:
    """Thiếu ``--index-url`` thì PyPI trả bản CPU — WhisperX sẽ chạy rất chậm."""
    torch_steps = [s for s in steps if "torch" in s.command]
    assert len(torch_steps) == 1
    command = torch_steps[0].command
    assert "--index-url" in command
    assert command[command.index("--index-url") + 1] == TORCH_CUDA_INDEX
    assert "cu129" in TORCH_CUDA_INDEX


def test_installs_all_three_torch_packages(steps: list) -> None:
    torch_command = next(s.command for s in steps if "torch" in s.command)
    for package in ("torch", "torchaudio", "torchvision"):
        assert package in torch_command


def test_whisperx_installed_after_torch(steps: list) -> None:
    """torch phải cài TRƯỚC, nếu không pip sẽ tự kéo bản CPU từ PyPI."""
    torch_index = next(i for i, s in enumerate(steps) if "torch" in s.command)
    whisperx_index = next(i for i, s in enumerate(steps) if "whisperx" in s.command)
    assert torch_index < whisperx_index


def test_steps_have_vietnamese_labels(steps: list) -> None:
    for step in steps:
        assert step.label
        assert step.label.strip() == step.label


def test_torch_step_has_largest_weight(steps: list) -> None:
    """Bước tải ~3GB phải có trọng số lớn nhất để tiến độ không nhảy vọt."""
    torch_step = next(s for s in steps if "torch" in s.command)
    assert torch_step.weight == max(s.weight for s in steps)


def test_step_count_is_stable(steps: list) -> None:
    assert len(steps) == 4


# ── Đường dẫn môi trường ─────────────────────────────────────────────────────
def test_env_python_path_matches_platform() -> None:
    path = env_python_path(_ROOT)
    assert WHISPERX_ENV_DIRNAME in str(path)
    if sys.platform == "win32":
        assert path.name == "python.exe"
        assert path.parent.name == "Scripts"
    else:
        assert path.name == "python"
        assert path.parent.name == "bin"


def test_env_python_path_is_stable() -> None:
    assert env_python_path(_ROOT) == env_python_path(_ROOT)


def test_installer_env_matches_adapter_lookup() -> None:
    """Trình cài phải tạo đúng nơi adapter đi tìm — lệch là cài xong vẫn báo chưa có."""
    from subtitles_extractor.infrastructure.stt import whisperx_adapter

    assert WHISPERX_ENV_DIRNAME == whisperx_adapter.WHISPERX_ENV_DIRNAME


# ── Tìm Python hệ thống ──────────────────────────────────────────────────────
def test_uses_current_interpreter_when_not_frozen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    assert find_system_python() == sys.executable


def test_frozen_build_searches_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bản đóng gói: ``sys.executable`` là .exe nên phải tìm Python ngoài."""
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    result = find_system_python()
    assert result != sys.executable or result is None
