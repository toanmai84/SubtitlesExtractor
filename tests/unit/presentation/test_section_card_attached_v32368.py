"""[v3.23.73] Guard tĩnh CHỐNG THẺ RỖNG cho mọi trang đã Card hoá.

Mọi ``SectionCard`` tạo trong một hàm phải được GẮN nội dung bằng
``add_layout(...)``/``add_widget(...)`` *trong cùng hàm đó*, nếu không thẻ sẽ rỗng (chỉ
hiện tiêu đề) và — với thẻ chưa có cha — có thể nổi thành cửa sổ rời.

Vì sao quét THEO PHẠM VI HÀM (per-function): các hàm builder hay tái sử dụng cùng một tên
biến cục bộ (vd ``g`` lặp lại cho nhiều thẻ). Kiểm theo tập tên toàn cục sẽ bỏ lọt khi
một ``g`` thiếu ``add_layout`` còn ``g`` khác thì có. Quét trong từng hàm khử được điểm mù
đó — chính lỗi đã làm các thẻ "Cấu hình chung/File WAV/Thực thi/Kết quả" của TTS bị rỗng.

Phân tích AST nguồn → chạy headless, không cần dựng Qt.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_PAGES_DIR = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "subtitles_extractor"
    / "presentation"
    / "pages"
)

# Mọi trang đã Card hoá — đều phải sạch thẻ rỗng.
_CARDIFIED_PAGES = (
    "debug_page.py",
    "tts_page.py",
    "translate_page.py",
    "extract_page.py",
    "settings_page.py",
    "editor_page.py",
)

_ATTACH_METHODS = {"add_layout", "add_widget"}


def _target_name(node: ast.AST) -> str | None:
    """Tên định danh: ``x`` -> 'x'; ``self._x`` -> '_x'."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _is_section_card_call(call_func: ast.AST) -> bool:
    return (isinstance(call_func, ast.Name) and call_func.id == "SectionCard") or (
        isinstance(call_func, ast.Attribute) and call_func.attr == "SectionCard"
    )


def _orphan_cards_in_function(func: ast.FunctionDef) -> list[tuple[str, int]]:
    """Danh sách (tên_thẻ, dòng) các ``SectionCard`` chưa được gắn nội dung trong hàm."""
    created: dict[str, int] = {}
    attached: set[str] = set()

    for node in ast.walk(func):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and _is_section_card_call(node.value.func)
        ):
            for target in node.targets:
                name = _target_name(target)
                if name is not None:
                    created[name] = node.lineno

        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _ATTACH_METHODS
        ):
            name = _target_name(node.func.value)
            if name is not None:
                attached.add(name)

    return [(var, line) for var, line in created.items() if var not in attached]


def _all_section_card_names(source: str) -> list[str]:
    """Tên mọi biến/thuộc tính được gán ``= SectionCard(...)``."""
    tree = ast.parse(source)
    names: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and _is_section_card_call(node.value.func)
        ):
            for target in node.targets:
                name = _target_name(target)
                if name is not None:
                    names.append(name)
    return names


@pytest.mark.parametrize("page_file", _CARDIFIED_PAGES)
def test_no_orphan_section_cards(page_file: str) -> None:
    source = (_PAGES_DIR / page_file).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=page_file)

    orphans: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for var, line in _orphan_cards_in_function(node):
                orphans.append(f"{node.name}():'{var}'@{line}")

    assert not orphans, (
        f"{page_file}: SectionCard chưa gắn nội dung (thiếu add_layout/add_widget) → "
        f"thẻ rỗng / nguy cơ nổi cửa sổ rời: {sorted(orphans)}"
    )


def test_debug_page_has_four_cards() -> None:
    """Chốt: 4 nhóm chính của trang Gỡ lỗi đã Card hoá."""
    source = (_PAGES_DIR / "debug_page.py").read_text(encoding="utf-8")
    found = set(_all_section_card_names(source))
    expected = {"_group_video", "_group_img", "_group_tuning", "_group_data"}
    assert expected <= found, f"Thiếu thẻ: {sorted(expected - found)}"
