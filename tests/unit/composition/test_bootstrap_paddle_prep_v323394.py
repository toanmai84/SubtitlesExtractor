"""[v3.23.394] Khóa hồi quy: CẢ HAI đường khởi động (GUI/CLI) phải chuẩn bị paddle.

Trước v3.23.394, ``bootstrap_for_cli`` thiếu bước thêm lõi paddle tải-lúc-chạy vào ``sys.path``
(+ cuDNN + chặn torch) → chạy OCR headless sẽ lỗi ``import paddle``. Test này đọc THẲNG mã
nguồn (không import module — tránh phụ thuộc PySide6) để đảm bảo cả hai hàm gọi helper dùng
chung ``_prepare_process_for_paddle``.
"""

from __future__ import annotations

import ast
from pathlib import Path

_BOOTSTRAP = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "subtitles_extractor"
    / "composition"
    / "bootstrap.py"
)


def _calls_prepare(func: ast.FunctionDef) -> bool:
    for node in ast.walk(func):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_prepare_process_for_paddle"
        ):
            return True
    return False


def _get_func(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"Không tìm thấy hàm {name} trong bootstrap.py")


def test_both_bootstraps_prepare_paddle() -> None:
    tree = ast.parse(_BOOTSTRAP.read_text(encoding="utf-8"))
    assert _calls_prepare(_get_func(tree, "bootstrap_for_gui"))
    assert _calls_prepare(_get_func(tree, "bootstrap_for_cli"))


def test_helper_defined() -> None:
    tree = ast.parse(_BOOTSTRAP.read_text(encoding="utf-8"))
    names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert "_prepare_process_for_paddle" in names
    assert "_ensure_paddle_runtime_on_syspath" in names
