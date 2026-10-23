"""[v3.23.613] Quét TOÀN DỰ ÁN ba lớp lỗi chỉ lộ ra khi chạy đúng nhánh đó.

Đợt rà lỗi 14/10 tìm thấy cả ba lớp ở mã đang chạy thật:

1. **Thuộc tính ``self.X`` không tồn tại** — ``ExtractPage._update_action_states``,
   ``ExtractPage.current_roi``, ``EditorPage.error_occurred``. PySide6 nuốt exception
   trong slot nên nút chỉ "không làm gì", không ai thấy lỗi.
2. **Kiểu định dạng log lệch thư viện** — ``logging`` chuẩn dùng ``%s``, loguru dùng
   ``{}``. Lệch kiểu thì loguru ghi nguyên "%d", còn ``logging`` ném lỗi định dạng và
   MẤT dòng log (ở đây là đuôi log pip khi tải CUDA/paddle thất bại — đúng thứ cần để
   chẩn đoán).
3. **Chuỗi giao diện** — khoá thiếu ở một ngôn ngữ, hoặc thay ``"{{n}}"`` trong khi
   chuỗi ghi ``"{n}"`` → người dùng thấy "{n}" thô.
"""

from __future__ import annotations

import ast
import enum
import importlib
import inspect
import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

import subtitles_extractor

pytest.importorskip("PySide6")

_SRC_ROOT = Path(subtitles_extractor.__file__).parent
_LOCALES = ("vi", "en")
_LOG_METHODS = frozenset({
    "debug", "info", "warning", "error", "exception", "critical", "trace", "success",
})
_PERCENT_FIELD = re.compile(
    r"%(?:\([^)]*\))?[-#0 +]*(?:\*|\d+)?(?:\.(?:\*|\d+))?[sdrifgexXoca]"
)
_BRACE_FIELD = re.compile(r"(?<!\{)\{(?!\{)[^{}]*\}(?!\})")
_UI_PLACEHOLDER = re.compile(r"(?<!\{)\{([A-Za-z_]\w*)\}(?!\})")


def _source_files() -> Iterator[tuple[str, Path, ast.Module]]:
    for path in sorted(_SRC_ROOT.rglob("*.py")):
        relative = path.relative_to(_SRC_ROOT.parent).with_suffix("")
        yield ".".join(relative.parts), path, ast.parse(path.read_text(encoding="utf-8"))


_FILES = list(_source_files())


def _where(path: Path, node: ast.AST) -> str:
    return f"{path.relative_to(_SRC_ROOT.parent)}:{getattr(node, 'lineno', 0)}"


# ── 1. self.X phải tồn tại ──────────────────────────────────────────────────


def _defined_in_class(class_node: ast.ClassDef) -> set[str]:
    """Tên lớp tự định nghĩa: phương thức, biến lớp, các ``self.X =``, ``cls.….X =``."""
    names: set[str] = set()
    for item in class_node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(item.name)
        elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            names.add(item.target.id)
        elif isinstance(item, ast.Assign):
            names.update(t.id for t in item.targets if isinstance(t, ast.Name))
    for node in ast.walk(class_node):
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, (ast.Store, ast.Del)):
            root = node.value
            while isinstance(root, ast.Attribute):
                root = root.value
            # cls._instance.X = … (singleton gán thuộc tính khi tạo) cũng là định nghĩa.
            if isinstance(root, ast.Name) and root.id in ("self", "cls"):
                names.add(node.attr)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "setattr"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
        ):
            names.add(str(node.args[1].value))
    return names


_CLASS_NODES: dict[tuple[str, str], ast.ClassDef] = {
    (module, node.name): node
    for module, _path, tree in _FILES
    for node in tree.body
    if isinstance(node, ast.ClassDef)
}


def _known_attributes(cls: type) -> set[str]:
    names: set[str] = set()
    for base in inspect.getmro(cls):
        node = _CLASS_NODES.get((base.__module__, base.__name__))
        if node is not None:
            names |= _defined_in_class(node)
    names |= set(getattr(cls, "__dataclass_fields__", {}) or {})
    model_fields = getattr(cls, "model_fields", None)
    if isinstance(model_fields, dict):
        names |= set(model_fields)
    if issubclass(cls, enum.Enum):
        names |= {"value", "name"}
    return names


def _self_reads(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
) -> Iterator[ast.Attribute]:
    """Mọi ``self.X`` được ĐỌC trong phương thức (không đi vào lớp lồng bên trong)."""
    stack: list[ast.AST] = list(ast.iter_child_nodes(method))
    while stack:
        node = stack.pop()
        if isinstance(node, ast.ClassDef):
            continue
        stack.extend(ast.iter_child_nodes(node))
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.ctx, ast.Load)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
        ):
            yield node


def test_every_self_attribute_exists() -> None:
    missing: list[str] = []
    for module_name, path, tree in _FILES:
        module = importlib.import_module(module_name)
        for class_node in (n for n in tree.body if isinstance(n, ast.ClassDef)):
            cls = getattr(module, class_node.name, None)
            if not inspect.isclass(cls):
                continue
            known = _known_attributes(cls)
            for item in class_node.body:
                if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                for read in _self_reads(item):
                    if read.attr not in known and not hasattr(cls, read.attr):
                        missing.append(f"{_where(path, read)} {cls.__name__}.{read.attr}")
    report = "\n".join(sorted(set(missing)))
    assert not missing, "Đọc thuộc tính không tồn tại:\n" + report


# ── 2. Kiểu định dạng log khớp thư viện ─────────────────────────────────────


def _logger_kinds(tree: ast.Module) -> dict[str, str]:
    """Tên biến logger trong module → ``"std"`` (logging) hoặc ``"loguru"``."""
    kinds: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "loguru":
            kinds.update({(a.asname or a.name): "loguru" for a in node.names})
        elif (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "getLogger"
        ):
            kinds[node.targets[0].id] = "std"
    return kinds


def _format_mismatch(kind: str, template: str, argument_count: int) -> str:
    if kind == "std":
        fields = len(_PERCENT_FIELD.findall(template.replace("%%", "")))
        expected = "%s"
    else:
        fields = len(_BRACE_FIELD.findall(template))
        expected = "{}"
    if fields == argument_count:
        return ""
    return f"{kind}: {fields} chỗ {expected} cho {argument_count} tham số"


def test_log_format_matches_logger_library() -> None:
    problems: list[str] = []
    for _module, path, tree in _FILES:
        kinds = _logger_kinds(tree)
        if not kinds:
            continue
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in _LOG_METHODS
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in kinds
                and node.args
            ):
                continue
            template = node.args[0]
            arguments = node.args[1:]
            if not arguments or any(isinstance(a, ast.Starred) for a in arguments):
                continue
            if not (
                isinstance(template, ast.Constant) and isinstance(template.value, str)
            ):
                continue
            problem = _format_mismatch(
                kinds[node.func.value.id], template.value, len(arguments)
            )
            if problem:
                shown = template.value[:60]
                problems.append(f"{_where(path, node)} {problem}: {shown!r}")
    assert not problems, "Định dạng log lệch:\n" + "\n".join(problems)


# ── 3. Chuỗi giao diện ──────────────────────────────────────────────────────


def _strings(locale: str) -> dict[str, Any]:
    path = _SRC_ROOT / "data" / f"strings_{locale}.json"
    return json.loads(path.read_text(encoding="utf-8"))


_UI_STRINGS = {locale: _strings(locale) for locale in _LOCALES}


def _ui_text(locale: str, key: str) -> str | None:
    node: Any = _UI_STRINGS[locale]
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node if isinstance(node, str) else None


def _translate_calls() -> Iterator[tuple[Path, ast.Call, str, list[str]]]:
    """``translate("khoá", …)`` kèm các chuỗi được ``.replace(...)`` nối ngay sau."""
    for _module, path, tree in _FILES:
        parents = {
            child: parent
            for parent in ast.walk(tree)
            for child in ast.iter_child_nodes(parent)
        }
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "translate"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
                and re.fullmatch(r"[a-z_]+(?:\.[A-Za-z0-9_]+)+", node.args[0].value)
            ):
                continue
            replaced: list[str] = []
            current: ast.AST = node
            while True:
                attribute = parents.get(current)
                call = parents.get(attribute) if attribute is not None else None
                if not (
                    isinstance(attribute, ast.Attribute)
                    and attribute.attr == "replace"
                    and isinstance(call, ast.Call)
                    and call.args
                    and isinstance(call.args[0], ast.Constant)
                ):
                    break
                replaced.append(str(call.args[0].value))
                current = call
            yield path, node, node.args[0].value, replaced


_TRANSLATE_CALLS = list(_translate_calls())


def test_ui_keys_exist_in_every_language() -> None:
    missing = [
        f"{_where(path, node)} {key!r} thiếu ở {locale}"
        for path, node, key, _replaced in _TRANSLATE_CALLS
        for locale in _LOCALES
        if _ui_text(locale, key) is None
    ]
    assert not missing, "\n".join(missing)


def test_ui_placeholders_match() -> None:
    problems: list[str] = []
    for path, node, key, replaced in _TRANSLATE_CALLS:
        keywords = {kw.arg for kw in node.keywords if kw.arg}
        for locale in _LOCALES:
            text = _ui_text(locale, key)
            if text is None:
                continue
            fields = set(_UI_PLACEHOLDER.findall(text))
            problems.extend(
                f"{_where(path, node)} {key!r}[{locale}] không có {old!r} để thay"
                for old in replaced
                if old not in text
            )
            if keywords and keywords != fields:
                problems.append(
                    f"{_where(path, node)} {key!r}[{locale}] chỗ trống {sorted(fields)} "
                    f"≠ tham số {sorted(keywords)}"
                )
    assert not problems, "\n".join(problems)
