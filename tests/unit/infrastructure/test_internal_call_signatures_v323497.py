"""[v3.23.497] Quét TOÀN BỘ lời gọi phương thức nội bộ và đối chiếu với chữ ký hàm nhận.

Bối cảnh: lỗi ``TypeError: ... got an unexpected keyword argument`` đã xảy ra HAI lần liên tiếp
(v3.23.483 với ``_translate_single_batch``, v3.23.496 với ``_patch_missing_with_context``), mỗi lần
đều làm HỎNG toàn bộ khâu dịch và chỉ lộ ra khi người dùng chạy thật.

Test v3.23.495 quá yếu: chỉ ĐẾM số lần xuất hiện của một tham số, không kiểm tra lời gọi đó đi tới
hàm nào. Test này dùng AST để duyệt mọi biểu thức ``self._method(...)`` trong adapter, tra chữ ký
của chính phương thức đó, và bắt lỗi ngay tại kiểm thử — không chờ tới lúc chạy.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from subtitles_extractor.infrastructure.translation import gemini_translation_adapter

_ADAPTER_PATH = Path(inspect.getfile(gemini_translation_adapter))
_TREE = ast.parse(_ADAPTER_PATH.read_text(encoding="utf-8"))


def _class_node(name: str) -> ast.ClassDef:
    for node in ast.walk(_TREE):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"không tìm thấy lớp {name}")


_CLASS = _class_node("GeminiSubtitleTranslator")

_METHODS: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {
    node.name: node
    for node in _CLASS.body
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
}


def _accepted_keywords(func: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str] | None:
    """Tập tên tham số nhận được; None nếu hàm có ``**kwargs`` (nhận mọi thứ)."""
    args = func.args
    if args.kwarg is not None:
        return None
    names = {a.arg for a in args.args} | {a.arg for a in args.kwonlyargs}
    names |= {a.arg for a in args.posonlyargs}
    return names


def _internal_calls() -> list[tuple[str, str, set[str], int]]:
    """Mọi lời gọi ``self._method(...)`` → (hàm gọi, hàm nhận, tên tham số truyền, dòng)."""
    calls: list[tuple[str, str, set[str], int]] = []
    for caller_name, caller in _METHODS.items():
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
            callee_name = func.attr
            if callee_name not in _METHODS:
                continue  # phương thức kế thừa/động — bỏ qua
            passed = {kw.arg for kw in node.keywords if kw.arg is not None}
            calls.append((caller_name, callee_name, passed, node.lineno))
    return calls


_CALLS = _internal_calls()


def test_scan_found_calls() -> None:
    """Bảo đảm bộ quét thực sự tìm được lời gọi (tránh test rỗng luôn xanh)."""
    assert len(_CALLS) >= 10


@pytest.mark.parametrize(
    ("caller", "callee", "passed", "lineno"),
    [pytest.param(*c, id=f"{c[0]}->{c[1]}:{c[3]}") for c in _CALLS],
)
def test_keyword_arguments_accepted(
    caller: str, callee: str, passed: set[str], lineno: int
) -> None:
    accepted = _accepted_keywords(_METHODS[callee])
    if accepted is None:
        return  # hàm nhận **kwargs
    unexpected = sorted(passed - accepted)
    assert not unexpected, (
        f"{caller} (dòng {lineno}) truyền tham số {unexpected} mà {callee} không nhận "
        f"→ sẽ gây TypeError khi chạy"
    )


def test_video_refs_reaches_call_gemini_paths() -> None:
    """Các hàm trong chuỗi dịch phải cùng nhận ``video_refs`` để heal 403 hoạt động xuyên suốt."""
    for name in ("_translate_single_batch", "_patch_missing_with_context", "_call_gemini"):
        accepted = _accepted_keywords(_METHODS[name])
        assert accepted is None or "video_refs" in accepted, name
