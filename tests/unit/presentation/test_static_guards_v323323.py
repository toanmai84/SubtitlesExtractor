"""Bộ canh giữ TĨNH — bắt các lỗi chỉ lộ ra lúc chạy (v3.23.323).

Ba lớp lỗi dưới đây đều ĐÃ XẢY RA THẬT trên máy người dùng và KHÔNG bị compileall bắt:

1. ``NameError: name '_os' is not defined`` trong ``SubtitlesExtractor.spec`` — dùng
   biến ở dòng 164 nhưng import ở dòng 231 → **build thất bại hoàn toàn**.
2. ``setIcon(FluentIcon.PLAY)`` — ``FluentIcon`` là ``Enum``, phải gọi ``.icon()``.
   Xảy ra ở 4 chỗ trên 3 trang, đổ traceback mỗi lần bấm play/pause.

Các test này đọc thẳng mã nguồn nên chạy được không cần Qt/màn hình.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest


def _project_root() -> Path:
    """Gốc dự án, suy từ module domain (import được mà không cần Qt)."""
    import subtitles_extractor.domain.entities.project_record as anchor

    return Path(anchor.__file__).resolve().parents[4]


def _presentation_files() -> list[Path]:
    root = _project_root() / "src" / "subtitles_extractor" / "presentation"
    return sorted(root.rglob("*.py"))


# ── Lỗi 1: dùng tên trước khi định nghĩa (cấp module) ────────────────────────
def _find_use_before_define(source: str) -> list[tuple[int, str, int]]:
    """Tìm tên riêng tư dùng ở dòng TRƯỚC nơi nó được định nghĩa."""
    tree = ast.parse(source)
    defined: dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                name = (alias.asname or alias.name).split(".")[0]
                defined.setdefault(name, node.lineno)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    defined.setdefault(target.id, node.lineno)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            defined.setdefault(node.name, node.lineno)

    problems: list[tuple[int, str, int]] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id.startswith("_")
            and node.id in defined
            and node.lineno < defined[node.id]
        ):
            problems.append((node.lineno, node.id, defined[node.id]))
    return sorted(set(problems))


def test_spec_has_no_use_before_define() -> None:
    """File .spec phải không dùng tên nào trước khi định nghĩa.

    Lỗi này làm BUILD CHẾT HOÀN TOÀN mà ``compileall`` không phát hiện được, vì cú
    pháp vẫn hợp lệ — chỉ sai ở thứ tự thực thi.
    """
    spec = _project_root() / "SubtitlesExtractor.spec"
    if not spec.is_file():
        pytest.skip("Không tìm thấy SubtitlesExtractor.spec")
    problems = _find_use_before_define(spec.read_text(encoding="utf-8"))
    assert not problems, f"Dùng trước khi định nghĩa: {problems}"


def test_spec_imports_os_before_first_use() -> None:
    """Kiểm cụ thể ``_os`` — đây chính là biến gây lỗi build thật."""
    spec = _project_root() / "SubtitlesExtractor.spec"
    if not spec.is_file():
        pytest.skip("Không tìm thấy SubtitlesExtractor.spec")
    source = spec.read_text(encoding="utf-8")

    import_line = next(
        (i for i, line in enumerate(source.splitlines(), 1)
         if line.strip().startswith("import os as _os")),
        None,
    )
    assert import_line is not None, "spec phải import os as _os"

    first_use = next(
        (i for i, line in enumerate(source.splitlines(), 1)
         if "_os." in line and not line.strip().startswith("#")),
        None,
    )
    assert first_use is not None
    assert import_line < first_use, (
        f"import _os ở dòng {import_line} nhưng dùng từ dòng {first_use}"
    )


# ── Lỗi 2: truyền Enum vào setIcon thay vì QIcon ─────────────────────────────
_BAD_SET_ICON = re.compile(r"setIcon\([^)]*FluentIcon\.[A-Z_]+(?!\s*\.icon\(\))")


def test_no_enum_passed_to_set_icon() -> None:
    """``FluentIcon`` là Enum — ``setIcon`` cần ``QIcon``, phải gọi ``.icon()``.

    Thiếu ``.icon()`` gây ``TypeError`` mỗi lần bấm nút, không làm sập app nhưng đổ
    traceback liên tục và nút mất biểu tượng.
    """
    offenders: list[str] = []
    for path in _presentation_files():
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "setIcon(" not in line or "FluentIcon." not in line:
                continue
            if ".icon()" in line:
                continue
            offenders.append(f"{path.name}:{lineno}")
    assert not offenders, f"setIcon nhận Enum thay vì QIcon tại: {offenders}"


def test_fluent_icon_exposes_icon_method() -> None:
    """``FluentIcon`` phải có ``.icon()`` — hợp đồng mà các trang dựa vào."""
    icons_file = (
        _project_root() / "src" / "subtitles_extractor" / "presentation"
        / "fluent_compat" / "icons.py"
    )
    source = icons_file.read_text(encoding="utf-8")
    assert "def icon(" in source


# ── Lỗi 3: nút được tạo nhưng quên thêm vào layout ───────────────────────────
def test_created_buttons_are_added_to_a_layout() -> None:
    """Nút tạo mà quên ``addWidget`` sẽ KHÔNG BAO GIỜ hiện trên giao diện.

    Đây là lỗi mình từng mắc ở v3.23.319 (nút hàng loạt) — bắt bằng kiểm tĩnh vì
    không có màn hình để nhìn thấy.
    """
    button_pattern = re.compile(r"self\.(_\w*button\w*)\s*=\s*\w*PushButton\(")
    offenders: list[str] = []

    for path in _presentation_files():
        source = path.read_text(encoding="utf-8")
        for match in button_pattern.finditer(source):
            name = match.group(1)
            # Nút phải xuất hiện trong ít nhất một lời gọi thêm-vào-layout.
            added = re.search(rf"(addWidget|addButton|insertWidget)\(\s*self\.{name}\b", source)
            if not added:
                offenders.append(f"{path.name}:{name}")

    assert not offenders, f"Nút tạo nhưng không thêm vào layout: {offenders}"
