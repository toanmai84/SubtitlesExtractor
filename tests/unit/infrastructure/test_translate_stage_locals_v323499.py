"""[v3.23.499] Chống hồi quy UnboundLocalError trong `translate_stage`.

`covering` (các đoạn video phủ khoảng thời gian của batch) từng chỉ được gán BÊN TRONG nhánh
"có đính video", nhưng lại được dùng ở lời gọi phía dưới → nhánh KHÔNG đính video (vd vòng dọn
CJK sót, dịch lại từ nguồn) văng `UnboundLocalError` và hỏng cả tác vụ dịch.

Test dùng AST nên bắt được cả khi mã được sắp xếp lại, không phụ thuộc thứ tự dòng cụ thể.
"""
from __future__ import annotations

import ast
import inspect
import textwrap

from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
)

_TREE = ast.parse(textwrap.dedent(inspect.getsource(GeminiSubtitleTranslator.translate_stage)))
_FUNC = next(
    n for n in ast.walk(_TREE) if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)
)


def _assignments_of(name: str) -> list[ast.AST]:
    found: list[ast.AST] = []
    for node in ast.walk(_FUNC):
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        for target in targets:
            if any(isinstance(n, ast.Name) and n.id == name for n in ast.walk(target)):
                found.append(node)
    return found


def _is_inside_if(node: ast.AST) -> bool:
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(_FUNC):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    current = parents.get(node)
    while current is not None and current is not _FUNC:
        if isinstance(current, ast.If):
            return True
        current = parents.get(current)
    return False


def test_covering_is_used() -> None:
    """Bảo vệ chống test rỗng: nếu `covering` không còn được dùng thì test này phải được xem lại."""
    used = any(
        isinstance(n, ast.Name) and n.id == "covering" and isinstance(n.ctx, ast.Load)
        for n in ast.walk(_FUNC)
    )
    assert used


def test_covering_has_unconditional_assignment() -> None:
    assignments = _assignments_of("covering")
    assert assignments, "không tìm thấy phép gán cho 'covering'"
    assert any(not _is_inside_if(node) for node in assignments), (
        "'covering' chỉ được gán bên trong nhánh if → nhánh không đính video sẽ "
        "văng UnboundLocalError"
    )


def test_batch_video_vars_have_unconditional_assignment() -> None:
    """Các biến cùng nhóm cũng phải khởi tạo vô điều kiện."""
    for name in ("batch_video_handles", "batch_video_tokens"):
        assignments = _assignments_of(name)
        assert assignments, name
        assert any(not _is_inside_if(node) for node in assignments), name
