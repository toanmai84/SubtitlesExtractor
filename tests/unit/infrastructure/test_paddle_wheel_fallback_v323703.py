"""[v3.23.703] Build / tải lõi OCR khi trang chỉ mục Paddle sập.

Ca thật 10/11/2026: paddlepaddle.org.cn trả 502 → pip lặng lẽ bỏ kho Paddle, chỉ thấy paddlepaddle-gpu 2.6.x trên PyPI:
"No matching distribution found for paddlepaddle-gpu==3.3.1". File wheel vẫn tải thẳng được từ paddle-whl.bj.bcebos.com.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from subtitles_extractor.infrastructure.ocr.paddle_runtime_plan import (
    PADDLE_EXTRA_INDEX_URL,
    PADDLE_INDEX_URL,
    build_paddle_install_command,
    paddle_direct_wheel_url,
)

ROOT = Path(__file__).resolve().parents[3]


def test_direct_wheel_url_matches_python_version() -> None:
    assert paddle_direct_wheel_url("3.13") == (
        "https://paddle-whl.bj.bcebos.com/stable/cu129/paddlepaddle-gpu/"
        "paddlepaddle_gpu-3.3.1-cp313-cp313-win_amd64.whl")
    assert paddle_direct_wheel_url("3.12").endswith("-cp312-cp312-win_amd64.whl")
    assert paddle_direct_wheel_url("abc") == ""


def test_fallback_command_installs_the_wheel_file_with_pypi_deps() -> None:
    normal = build_paddle_install_command("py", Path("t"), "3.13")
    direct = build_paddle_install_command("py", Path("t"), "3.13", direct_wheel=True)
    assert PADDLE_INDEX_URL in normal
    assert PADDLE_INDEX_URL not in direct and paddle_direct_wheel_url("3.13") in direct
    assert direct[direct.index("--index-url") + 1] == PADDLE_EXTRA_INDEX_URL
    assert "--python-version" in direct and "--only-binary=:all:" in direct


def _batch_tool():
    path = ROOT / "tools" / "check_batch_syntax.py"
    spec = importlib.util.spec_from_file_location("_batch_tool_703", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_batch_tool_703"] = module
    spec.loader.exec_module(module)
    return module


def test_build_script_falls_back_to_direct_wheel_and_is_clean() -> None:
    text = (ROOT / "build_windows.bat").read_text(encoding="utf-8")
    assert "goto :paddle_direct_wheel" in text and ":paddle_done" in text
    assert "paddle-whl.bj.bcebos.com/stable/cu129/paddlepaddle-gpu/paddlepaddle_gpu-3.3.1-%PYTAG%-%PYTAG%" in text
    assert _batch_tool().check_batch_file(ROOT / "build_windows.bat") == []


def test_batch_checker_flags_unescaped_paren_in_echo_inside_block(tmp_path: Path) -> None:
    bad = tmp_path / "bad.bat"
    bad.write_text('@echo off\nif errorlevel 1 (\n    echo [LOI] that bai (ca 2 duong).\n    exit /b 1\n)\n', "utf-8")
    good = tmp_path / "good.bat"
    good.write_text('@echo off\nif errorlevel 1 (\n    echo [LOI] that bai ^(ca 2 duong^).\n    exit /b 1\n)\necho xong (ok)\n',
                    "utf-8")
    tool = _batch_tool()
    assert tool.check_batch_file(bad) and tool.check_batch_file(good) == []
