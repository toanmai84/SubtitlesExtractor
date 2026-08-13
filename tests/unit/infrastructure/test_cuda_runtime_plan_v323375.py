"""[v3.23.375] Kiểm thử plan tải CUDA runtime cho GPU OCR."""
from __future__ import annotations

from pathlib import Path

from subtitles_extractor.infrastructure.ocr.cuda_runtime_plan import (
    CUDA_RUNTIME_PACKAGES,
    CudaRuntimeStatus,
    build_cuda_install_command,
    cuda_runtime_dll_dirs,
    evaluate_cuda_runtime,
    is_cuda_runtime_installed,
)


def test_needs_download_when_absent(tmp_path: Path) -> None:
    plan = evaluate_cuda_runtime(tmp_path / "cuda_runtime", bundled=False)
    assert plan.status is CudaRuntimeStatus.NEEDS_DOWNLOAD
    assert plan.needs_download


def test_bundled_takes_priority(tmp_path: Path) -> None:
    plan = evaluate_cuda_runtime(tmp_path / "cuda_runtime", bundled=True)
    assert plan.status is CudaRuntimeStatus.BUNDLED
    assert not plan.needs_download


def test_installed_when_dll_dir_present(tmp_path: Path) -> None:
    cuda = tmp_path / "cuda_runtime"
    (cuda / "nvidia" / "cublas" / "bin").mkdir(parents=True)
    assert is_cuda_runtime_installed(cuda)
    assert evaluate_cuda_runtime(cuda, bundled=False).status is CudaRuntimeStatus.INSTALLED


def test_dll_dirs_only_existing(tmp_path: Path) -> None:
    cuda = tmp_path / "cuda_runtime"
    (cuda / "nvidia" / "cudnn" / "bin").mkdir(parents=True)
    (cuda / "nvidia" / "cufft" / "bin").mkdir(parents=True)
    dirs = cuda_runtime_dll_dirs(cuda)
    assert len(dirs) == 2
    assert all(d.is_dir() for d in dirs)


def test_install_command_uses_target_and_no_deps(tmp_path: Path) -> None:
    cmd = build_cuda_install_command("C:\\Python\\python.exe", tmp_path / "cuda")
    assert cmd[:4] == ("C:\\Python\\python.exe", "-m", "pip", "install")
    assert "--target" in cmd
    assert "--no-deps" in cmd
    # Tất cả gói nvidia-* đều có trong lệnh.
    for pkg in CUDA_RUNTIME_PACKAGES:
        assert pkg in cmd
