"""[v3.23.71 — Giai đoạn 6 Accessibility] Guard tĩnh: các nút CHỈ-ICON phải có

``accessibleName`` (đặt qua :func:`set_accessible_name`) để trình đọc màn hình mô tả được.

Phân tích AST nguồn (chạy headless, không cần dựng Qt). Mỗi nút icon-only đã biết phải
xuất hiện như ĐỐI SỐ ĐẦU của một lời gọi ``set_accessible_name(<nút>, ...)`` trong trang
tương ứng.
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

# Các nút chỉ-icon đã biết theo trang (tên thuộc tính/biến).
# Thêm vào đây khi có nút icon mới.
_CONTROLS_NEEDING_NAME: dict[str, set[str]] = {
    "debug_page.py": {"_btn_play", "_btn_prev", "_btn_next", "_slider"},
    "extract_page.py": {"_btn_play_pause", "_position_slider"},
    "translate_page.py": {"_btn_eye"},
    "tts_page.py": {"_btn_eye"},
    "editor_page.py": {
        "btn_play",
        "_btn_prev_frame",
        "_btn_step_back",
        "_btn_play",
        "_btn_step_forward",
        "_btn_next_frame",
        "timeline_slider",
        "_vol",
        "_position_slider",
        "_waveform_y_zoom",
    },
}


def _first_arg_name(call: ast.Call) -> str | None:
    """Tên thuộc tính/biến của đối số đầu trong lời gọi (self._x -> '_x'; x -> 'x')."""
    if not call.args:
        return None
    arg = call.args[0]
    if isinstance(arg, ast.Attribute):
        return arg.attr
    if isinstance(arg, ast.Name):
        return arg.id
    return None


def _named_by_set_accessible_name(source: str) -> set[str]:
    """Thu thập tên các widget được truyền vào ``set_accessible_name(<widget>, ...)``."""
    tree = ast.parse(source)
    named: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "set_accessible_name"
        ):
            name = _first_arg_name(node)
            if name is not None:
                named.add(name)
    return named


@pytest.mark.parametrize("page_file", sorted(_CONTROLS_NEEDING_NAME))
def test_controls_have_accessible_name(page_file: str) -> None:
    source = (_PAGES_DIR / page_file).read_text(encoding="utf-8")
    named = _named_by_set_accessible_name(source)
    missing = sorted(_CONTROLS_NEEDING_NAME[page_file] - named)
    assert not missing, (
        f"{page_file}: các nút chỉ-icon CHƯA có accessibleName "
        f"(thiếu set_accessible_name): {missing}"
    )


def test_helper_sets_name_and_tooltip() -> None:
    """Hành vi helper: đặt accessibleName; đặt tooltip nếu chưa có (không ghi đè)."""
    pytest.importorskip("PyQt6.QtWidgets")
    from PySide6.QtWidgets import QApplication, QPushButton

    app = QApplication.instance() or QApplication([])
    assert app is not None

    from subtitles_extractor.presentation.utils.accessibility import set_accessible_name

    btn = QPushButton()
    set_accessible_name(btn, "Phát/Tạm dừng")
    assert btn.accessibleName() == "Phát/Tạm dừng"
    assert btn.toolTip() == "Phát/Tạm dừng"

    btn2 = QPushButton()
    btn2.setToolTip("Tooltip cũ")
    set_accessible_name(btn2, "Tên mới")
    assert btn2.accessibleName() == "Tên mới"
    assert btn2.toolTip() == "Tooltip cũ"  # không ghi đè

    btn3 = QPushButton()
    set_accessible_name(btn3, "Chỉ tên", set_tooltip=False)
    assert btn3.accessibleName() == "Chỉ tên"
    assert btn3.toolTip() == ""
