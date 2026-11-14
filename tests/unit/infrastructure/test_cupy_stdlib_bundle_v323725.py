"""[v3.23.725] Bản đóng gói 3.23.724: "thiếu CuPy dùng được (chưa cài CuPy ... No module named 'graphlib')".

Gốc lỗi: ``cupy/cuda/function`` là mô-đun Cython đã biên dịch (.pyd) có ``import graphlib``. PyInstaller không đọc được
import trong .pyd nên không gom ``graphlib`` vào gói, khiến ``import cupy`` hỏng và NVDEC lùi về PyAV. Tái hiện bằng
PyInstaller 6.22 với CuPy 14.2: gói chỉ ``--collect-all cupy`` → ``ModuleNotFoundError: graphlib``; thêm danh sách do
spec tính → ``import cupy`` thành công.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]

#: ``import cupy`` 14.2 (Python 3.13, đo thật 14/11/2026): mô-đun chuẩn công khai được nạp thêm.
_MEASURED_CUPY_14_2 = {
    "ast", "atexit", "binascii", "bisect", "bz2", "collections", "contextlib", "contextvars", "copy", "copyreg",
    "ctypes", "dataclasses", "datetime", "dis", "email", "enum", "fnmatch", "functools", "gc", "glob", "graphlib",
    "hashlib", "importlib", "inspect", "itertools", "json", "keyword", "linecache", "locale", "lzma", "math", "ntpath",
    "numbers", "opcode", "operator", "pathlib", "pickle", "platform", "random", "re", "reprlib", "select",
    "selectors", "shutil", "signal", "string", "struct", "subprocess", "sysconfig", "tempfile", "textwrap",
    "threading", "token", "tokenize", "traceback", "types", "typing", "weakref", "zipfile", "zlib",
}
_POSIX_ONLY = {"fcntl", "grp", "pwd"}


def _spec_scope() -> dict:
    source = (ROOT / "SubtitlesExtractor.spec").read_text(encoding="utf-8")
    block = source[source.index("_CUPY_STDLIB_FLOOR = ("):source.index('if _importlib_util.find_spec("cupy") is not None:')]
    scope = {"_importlib_util": importlib.util}
    exec(block, scope)  # noqa: S102 — chạy đúng đoạn mã của spec
    return scope


def test_floor_covers_measured_cupy_imports() -> None:
    floor = set(_spec_scope()["_CUPY_STDLIB_FLOOR"])
    assert "graphlib" in floor
    assert floor >= _MEASURED_CUPY_14_2 - _POSIX_ONLY


def test_spec_hiddenimports_add_graphlib_even_when_probe_fails() -> None:
    compute = _spec_scope()["_cupy_stdlib_hiddenimports"]
    names = compute(str(ROOT / "khong-co-python.exe"))     # dò hỏng → vẫn còn danh sách sàn
    assert "graphlib" in names and all(not n.startswith("_") for n in names)


def test_spec_probe_finds_stdlib_imported_by_installed_cupy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Bản CuPy mới import thêm mô-đun chuẩn khác → lúc build tự dò ra (không cần sửa danh sách sàn)."""
    package = tmp_path / "cupy"
    package.mkdir()
    (package / "__init__.py").write_text("import graphlib, mailbox\n", encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    names = _spec_scope()["_cupy_stdlib_hiddenimports"](sys.executable)
    assert {"graphlib", "mailbox"} <= set(names)


def test_spec_wires_the_list_into_hiddenimports() -> None:
    source = (ROOT / "SubtitlesExtractor.spec").read_text(encoding="utf-8")
    start = source.index('if _importlib_util.find_spec("cupy") is not None:')
    wired = source.index("    hiddenimports.extend(_cupy_std)", start)
    assert wired < source.index("a = Analysis(")
    assert "_cupy_std = _cupy_stdlib_hiddenimports(_sys.executable)" in source[start:wired]


class _CupyMissingGraphlib(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """``import cupy`` như trong gói 3.23.724: CuPy bọc ``ModuleNotFoundError('graphlib')`` bằng ``ImportError``."""

    def find_spec(self, name, path=None, target=None):
        return importlib.util.spec_from_loader(name, self) if name == "cupy" else None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        try:
            raise ModuleNotFoundError("No module named 'graphlib'", name="graphlib")
        except ImportError as exc:
            raise ImportError(textwrap.dedent("""
                Failed to import CuPy.
                Original error:
                  ModuleNotFoundError: No module named 'graphlib'
            """)) from exc


def test_loader_names_the_missing_module_instead_of_not_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    from subtitles_extractor.infrastructure.ocr.preprocessing import gpu_image_filters

    monkeypatch.delitem(sys.modules, "cupy", raising=False)
    monkeypatch.setattr(sys, "meta_path", [_CupyMissingGraphlib(), *sys.meta_path])
    module, usable, reason = gpu_image_filters._load_cupy()
    assert module is None and not usable
    assert "graphlib" in reason and "chưa cài" not in reason       # mã cũ: "chưa cài CuPy (...khối chẩn đoán...)"


def test_missing_dependency_ignores_cupy_itself() -> None:
    from subtitles_extractor.infrastructure.ocr.preprocessing.gpu_image_filters import (
        _missing_dependency,
    )

    assert _missing_dependency(ModuleNotFoundError("No module named 'cupy'", name="cupy")) == ""
    assert _missing_dependency(ModuleNotFoundError("x", name="cupy_backends.cuda")) == ""
    assert _missing_dependency(OSError("dll")) == ""
    outer = ImportError("wrapped")
    outer.__cause__ = ModuleNotFoundError("x", name="graphlib")
    assert _missing_dependency(outer) == "graphlib"


def test_not_installed_reason_is_short(monkeypatch: pytest.MonkeyPatch) -> None:
    from subtitles_extractor.infrastructure.ocr.preprocessing import gpu_image_filters

    class NoCupy(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name == "cupy":
                raise ModuleNotFoundError("No module named 'cupy'", name="cupy")
            return None

    monkeypatch.delitem(sys.modules, "cupy", raising=False)
    monkeypatch.setattr(sys, "meta_path", [NoCupy(), *sys.meta_path])
    assert gpu_image_filters._load_cupy() == (None, False, "chưa cài CuPy")
