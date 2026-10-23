"""[v3.23.498] Quét TOÀN DỰ ÁN: mọi lời gọi ``self._method(...)`` phải khớp chữ ký hàm nhận.

Bối cảnh: lớp lỗi ``TypeError: ... unexpected keyword argument`` đã làm hỏng khâu dịch HAI lần
(v3.23.483 và v3.23.496). Bộ quét v3.23.497 chỉ phủ một file adapter — không đủ, vì lỗi kiểu này có
thể xảy ra ở bất kỳ module nào khi sửa chữ ký mà quên nơi gọi.

Test này quét mọi file trong ``src`` và kiểm ba loại sai lệch:

1. Truyền tham số theo TÊN mà hàm nhận không có.
2. Truyền quá nhiều tham số VỊ TRÍ.
3. THIẾU tham số bắt buộc.

Chỉ xét lời gọi tới phương thức định nghĩa trong CÙNG lớp (``self.x``/``cls.x``) — nơi có thể đối
chiếu chữ ký chắc chắn; lời gọi động hoặc kế thừa được bỏ qua để tránh báo nhầm.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import subtitles_extractor

_SRC_ROOT = Path(subtitles_extractor.__file__).parent


def _method_kind(func: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    names = {d.id for d in func.decorator_list if isinstance(d, ast.Name)}
    if "staticmethod" in names:
        return "static"
    if "classmethod" in names:
        return "class"
    return "instance"


def _check_call(
    class_name: str,
    caller: ast.FunctionDef | ast.AsyncFunctionDef,
    callee: ast.FunctionDef | ast.AsyncFunctionDef,
    node: ast.Call,
    *,
    is_method: bool = True,
) -> list[str]:
    """Trả về danh sách mô tả sai lệch giữa lời gọi và chữ ký (rỗng nếu hợp lệ)."""
    issues: list[str] = []
    args = callee.args
    prefix = f"{class_name}.{caller.name} → {callee.name}"

    accepted = {a.arg for a in args.args + args.kwonlyargs + args.posonlyargs}
    passed = {kw.arg for kw in node.keywords if kw.arg is not None}
    has_star_star = any(kw.arg is None for kw in node.keywords)

    if args.kwarg is None:
        unexpected = sorted(passed - accepted)
        if unexpected:
            issues.append(f"{prefix}: tham số KHÔNG được nhận {unexpected}")

    positional_slots = len(args.posonlyargs) + len(args.args)
    # Hàm cấp module KHÔNG có self/cls → không trừ chỗ (trừ nhầm gây dương tính giả hàng loạt).
    if is_method and _method_kind(callee) != "static":
        positional_slots -= 1
    if args.vararg is None and len(node.args) > positional_slots:
        issues.append(
            f"{prefix}: truyền {len(node.args)} tham số vị trí, chỉ nhận {positional_slots}"
        )

    required = [a.arg for a in args.posonlyargs + args.args]
    skip = 1 if (is_method and _method_kind(callee) != "static") else 0
    required = required[skip:]
    if args.defaults:
        required = required[: -len(args.defaults)]
    supplied = set(required[: len(node.args)]) | passed
    missing = [name for name in required if name not in supplied]
    if missing and not has_star_star:
        issues.append(f"{prefix}: THIẾU tham số bắt buộc {missing}")
    return issues


def _scan_file(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    findings: list[str] = []

    # (1) Phương thức trong lớp: lời gọi self.x(...) / cls.x(...)
    for class_node in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
        methods = {
            n.name: n
            for n in class_node.body
            if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)
        }
        for caller in methods.values():
            for node in ast.walk(caller):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                if not (
                    isinstance(func, ast.Attribute)
                    and isinstance(func.value, ast.Name)
                    and func.value.id in {"self", "cls"}
                ):
                    continue
                callee = methods.get(func.attr)
                if callee is None:
                    continue  # phương thức kế thừa/động — không đối chiếu được
                for issue in _check_call(class_node.name, caller, callee, node):
                    findings.append(f"{path.name}:{node.lineno} — {issue}")

    # (2) [v3.23.498] Hàm cấp MODULE: lời gọi ten_ham(...) trong cùng file. Bỏ sót nhóm này từng
    # khiến bộ quét không phát hiện lỗi ở các service thuần (vốn toàn hàm module-level).
    module_functions = {
        n.name: n
        for n in tree.body
        if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)
    }
    if module_functions:
        shadowed = {
            n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)
        } | _imported_names(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            name = node.func.id
            callee = module_functions.get(name)
            if callee is None or name in shadowed:
                continue
            for issue in _check_call("<module>", callee, callee, node, is_method=False):
                findings.append(f"{path.name}:{node.lineno} — {issue}")
    return findings


def _imported_names(tree: ast.Module) -> set[str]:
    """Tên được import (có thể trùng tên hàm cục bộ) — bỏ qua để tránh báo nhầm."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom | ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
    return names


_PY_FILES = sorted(_SRC_ROOT.rglob("*.py"))


def test_source_files_found() -> None:
    """Tránh test rỗng luôn xanh nếu đường dẫn nguồn thay đổi."""
    assert len(_PY_FILES) > 100


@pytest.mark.parametrize("path", _PY_FILES, ids=lambda p: p.name)
def test_internal_call_signatures(path: Path) -> None:
    findings = _scan_file(path)
    assert not findings, "Sai lệch chữ ký (sẽ gây TypeError khi chạy):\n" + "\n".join(findings)
