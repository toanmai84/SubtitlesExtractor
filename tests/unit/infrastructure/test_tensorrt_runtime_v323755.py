"""[v3.23.755] Bật TensorRT: "No module named 'tensorrt'" (log 19/11 10:26) — bản paddle (3.3.1 cu129, biên dịch với
TensorRT 10.5, ``with_pip_tensorrt='OFF'``) không kèm TensorRT. Ứng dụng tải TensorRT 10.9 (bản thấp nhất có Python 3.13
trên Windows) từ kho NVIDIA vào models/tensorrt_runtime, tự ghi mô-đun ``tensorrt`` và đưa vào sys.path + DLL."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from subtitles_extractor.infrastructure.ocr import tensorrt_runtime_plan as plan


def _fake_install(trt_dir: Path) -> None:
    (trt_dir / "tensorrt_libs").mkdir(parents=True)
    (trt_dir / "tensorrt_libs" / "nvinfer_10.dll").write_bytes(b"dll")
    (trt_dir / "tensorrt_bindings").mkdir()
    (trt_dir / "tensorrt_bindings" / "__init__.py").write_text("__version__ = '10.9.0.34'\nBuilder = object\n")


def test_install_command_uses_nvidia_index_and_wheels_only(tmp_path: Path) -> None:
    command = plan.build_tensorrt_install_command("py.exe", tmp_path)
    assert command[:4] == ("py.exe", "-m", "pip", "install")
    assert command[command.index("--target") + 1] == str(tmp_path)
    assert command[command.index("--extra-index-url") + 1] == "https://pypi.nvidia.com"
    assert "--only-binary" in command and "--no-deps" in command
    assert {"tensorrt-cu12-libs==10.9.0.34", "tensorrt-cu12-bindings==10.9.0.34"} <= set(command)


def test_shim_exposes_bindings_as_tensorrt(tmp_path: Path) -> None:
    _fake_install(tmp_path)
    assert not plan.is_tensorrt_installed(tmp_path)                 # chưa có mô-đun tensorrt
    plan.write_tensorrt_shim(tmp_path)
    assert plan.is_tensorrt_installed(tmp_path)
    assert plan.tensorrt_dll_dirs(tmp_path) == [tmp_path / "tensorrt_libs"]
    out = subprocess.run([sys.executable, "-I", "-c", "import sys; sys.path.insert(0, sys.argv[1]); import tensorrt; "
                          "print(tensorrt.__version__, tensorrt.Builder)", str(tmp_path)],
                         capture_output=True, text=True, check=True).stdout
    assert out.startswith("10.9.0.34")


def test_worker_writes_shim_after_pip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("PySide6")
    from subtitles_extractor.presentation.workers import install_tensorrt_runtime_worker as mod

    def fake_run(command, **_k):  # noqa: ANN001, ANN202
        _fake_install(Path(command[command.index("--target") + 1]))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    worker = mod.InstallTensorRtRuntimeWorker("py.exe", tmp_path / "trt")
    done: list[object] = []
    worker.finished.connect(done.append)
    worker.failed.connect(lambda m: pytest.fail(m))
    worker.run()
    assert done == [tmp_path / "trt"] and plan.is_tensorrt_installed(tmp_path / "trt")


def test_bootstrap_and_settings_wired() -> None:
    root = Path(__file__).resolve().parents[3] / "src/subtitles_extractor"
    boot = (root / "composition/bootstrap.py").read_text(encoding="utf-8")
    assert "tensorrt_dll_dirs(trt_dir)" in boot and "sys.path.append(str(trt_dir))" in boot
    page = (root / "presentation/pages/settings_page.py").read_text(encoding="utf-8")
    assert "InstallTensorRtRuntimeWorker" in page and "self._refresh_tensorrt_status()" in page
    for lang in ("vi", "en"):
        strings = json.loads((root / f"data/strings_{lang}.json").read_text(encoding="utf-8"))["settings"]
        assert {"trt_btn", "trt_tip", "trt_row", "trt_needs", "trt_installed", "trt_confirm_body"} <= strings.keys()


# ── [v3.23.756] log 19/11 11:15: paddle/tensorrt/converter.py gọi cứng ctypes.CDLL('libnvinfer_plugin.so') ─────────
def test_paddle_converter_gets_windows_plugin_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import types

    _fake_install(tmp_path)
    (tmp_path / "tensorrt_libs" / "nvinfer_plugin_10.dll").write_bytes(b"dll")
    loaded: list[str] = []
    converter = types.ModuleType("paddle.tensorrt.converter")
    import ctypes as real_ctypes

    converter.ctypes = real_ctypes
    monkeypatch.setitem(sys.modules, "paddle.tensorrt.converter", converter)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(real_ctypes, "CDLL", lambda name, *a, **k: loaded.append(str(name)) or "lib")
    monkeypatch.delattr(real_ctypes, "WinDLL", raising=False)
    monkeypatch.setattr(plan, "_LOADED_PLUGINS", [])
    monkeypatch.setenv("PATH", "")
    assert plan.patch_paddle_tensorrt_plugin_loader(tmp_path)
    # [v3.23.763] Cùng tệp bí danh ``nvinfer_plugin.dll`` mà phần C++ của Paddle nạp lúc chạy engine (nạp sẵn 1 lần).
    alias = str(tmp_path / "tensorrt_libs" / "nvinfer_plugin.dll")
    assert loaded == [alias]
    assert converter.ctypes.CDLL("libnvinfer_plugin.so") == "lib"           # như dòng 91 của converter.py
    converter.ctypes.CDLL("other.dll")
    assert loaded == [alias, alias, "other.dll"]
    assert converter.ctypes.c_int is real_ctypes.c_int                     # phần còn lại của ctypes giữ nguyên
    assert real_ctypes.CDLL is not converter.ctypes.CDLL                   # không đụng ctypes toàn cục


def test_patch_skipped_off_windows_or_without_plugin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    assert not plan.patch_paddle_tensorrt_plugin_loader(tmp_path)
    monkeypatch.setattr(sys, "platform", "win32")
    assert not plan.patch_paddle_tensorrt_plugin_loader(tmp_path)          # chưa tải TensorRT


def test_adapter_prepares_tensorrt_before_building_trt_model() -> None:
    import inspect

    from subtitles_extractor.infrastructure.ocr.paddle_ocr_adapter import (
        PaddleOcrAdapter,
    )

    source = inspect.getsource(PaddleOcrAdapter._create_model_with_device_fallback)
    assert "if use_trt:\n            self._trt_dir = _prepare_paddle_tensorrt()" in source  # [v3.23.764] trả thư mục TensorRT
