"""[v3.23.386] Test thuần cho kế hoạch tải paddlepaddle-gpu runtime."""

from __future__ import annotations

from pathlib import Path

from subtitles_extractor.infrastructure.ocr.paddle_runtime_plan import (
    PADDLE_INDEX_URL,
    PADDLE_RUNTIME_PACKAGE,
    PaddleRuntimeStatus,
    build_paddle_install_command,
    evaluate_paddle_runtime,
    is_paddle_runtime_installed,
    paddle_runtime_sys_path,
)


def test_needs_download_when_empty(tmp_path: Path) -> None:
    plan = evaluate_paddle_runtime(tmp_path / "paddle_runtime", bundled=False)
    assert plan.status is PaddleRuntimeStatus.NEEDS_DOWNLOAD
    assert plan.needs_download is True
    assert is_paddle_runtime_installed(tmp_path / "paddle_runtime") is False
    assert paddle_runtime_sys_path(tmp_path / "paddle_runtime") == []


def test_installed_when_paddle_package_present(tmp_path: Path) -> None:
    paddle_dir = tmp_path / "paddle_runtime"
    (paddle_dir / "paddle").mkdir(parents=True)
    plan = evaluate_paddle_runtime(paddle_dir, bundled=False)
    assert plan.status is PaddleRuntimeStatus.INSTALLED
    assert plan.needs_download is False
    assert is_paddle_runtime_installed(paddle_dir) is True
    assert paddle_runtime_sys_path(paddle_dir) == [paddle_dir]


def test_bundled_takes_priority(tmp_path: Path) -> None:
    plan = evaluate_paddle_runtime(tmp_path / "paddle_runtime", bundled=True)
    assert plan.status is PaddleRuntimeStatus.BUNDLED
    assert plan.needs_download is False


def test_install_command_pulls_deps_from_paddle_index(tmp_path: Path) -> None:
    paddle_dir = tmp_path / "paddle_runtime"
    cmd = build_paddle_install_command("py.exe", paddle_dir, "3.11")
    assert cmd[0] == "py.exe"
    # KHÔNG --no-deps: phải kéo phụ thuộc RIÊNG của paddle (astor/decorator/…).
    assert "--no-deps" not in cmd
    assert "--target" in cmd
    assert str(paddle_dir) in cmd
    assert PADDLE_RUNTIME_PACKAGE in cmd
    # Phải trỏ kho Paddle (nơi duy nhất có paddlepaddle-gpu) + PyPI cho deps thuần.
    assert "--index-url" in cmd
    assert PADDLE_INDEX_URL in cmd


def test_install_command_targets_bundled_python_version(tmp_path: Path) -> None:
    """Phải tải wheel cho ĐÚNG phiên bản Python bundled (không phải phiên bản pip đang chạy)."""
    cmd = build_paddle_install_command("py.exe", tmp_path / "pr", "3.12")
    # Cặp cờ này bắt buộc để pip tải cross-version wheel.
    assert "--python-version" in cmd
    idx = cmd.index("--python-version")
    assert cmd[idx + 1] == "3.12"
    assert "--only-binary=:all:" in cmd


def test_sys_path_appends_only_existing(tmp_path: Path) -> None:
    # Chưa có -> rỗng.
    assert paddle_runtime_sys_path(tmp_path / "nope") == []
    # Có -> đúng 1 đường dẫn.
    d = tmp_path / "paddle_runtime"
    (d / "paddle").mkdir(parents=True)
    assert paddle_runtime_sys_path(d) == [d]
