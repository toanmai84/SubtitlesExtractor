"""[v3.23.613] Tải CUDA/paddle thất bại: đuôi log pip phải VÀO được tệp log.

Hai worker dùng ``logging`` chuẩn nhưng viết chỗ trống kiểu loguru (``{}``) → ``logging``
báo "not all arguments converted" và bỏ cả dòng: đúng dòng duy nhất chứa lý do pip thất
bại (bản v3.23.547 thêm vào để chẩn đoán) chưa từng được ghi.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("PySide6")

from subtitles_extractor.presentation.workers import (
    install_cuda_runtime_worker,
    install_paddle_runtime_worker,
)

PIP_TAIL = "ERROR: No matching distribution found for nvidia-cudnn-cu12==9.9.9"


def _failed_pip(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr=PIP_TAIL)


@pytest.mark.parametrize(
    ("module", "worker_class"),
    [
        (install_cuda_runtime_worker, "InstallCudaRuntimeWorker"),
        (install_paddle_runtime_worker, "InstallPaddleRuntimeWorker"),
    ],
)
def test_pip_tail_reaches_the_log(
    module: Any,
    worker_class: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(module.subprocess, "run", _failed_pip)
    worker = getattr(module, worker_class)("python", tmp_path / "runtime")
    failures: list[str] = []
    worker.failed.connect(failures.append)
    with caplog.at_level(logging.ERROR, logger=module.__name__):
        worker.run()
    messages = [record.getMessage() for record in caplog.records]
    assert any("mã thoát 1" in m and PIP_TAIL in m for m in messages), messages
    assert failures and PIP_TAIL in failures[0]
