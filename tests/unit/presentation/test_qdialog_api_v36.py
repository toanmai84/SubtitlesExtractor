"""Test bảo vệ PyQt6 QInputDialog.getDouble() keyword argument names.

PyQt6 đổi tên: minValue/maxValue (PyQt5) → min/max (PyQt6).
Dùng wrong args gây TypeError ngay khi user click "Tách câu".
"""
from pathlib import Path

_PAGE = (
    Path(__file__).resolve().parents[3]
    / "src/subtitles_extractor/presentation/pages/editor_page.py"
)


def test_get_double_uses_pyqt6_min_max_args() -> None:
    """getDouble phải dùng min= và max= (PyQt6), KHÔNG dùng minValue/maxValue (PyQt5)."""
    source = _PAGE.read_text(encoding="utf-8")
    assert "minValue=" not in source, (
        "Tìm thấy 'minValue=' — PyQt6 dùng 'min=' thay vì 'minValue='. "
        "Sẽ gây TypeError: 'minValue' is not a valid keyword argument."
    )
    assert "maxValue=" not in source or "cv2" in source.split("maxValue=")[0].rsplit("\n", 1)[-1], (
        "Tìm thấy 'maxValue=' không phải từ cv2 — PyQt6 dùng 'max=' thay vì 'maxValue='."
    )


def test_get_double_split_has_correct_args() -> None:
    """_on_split_clicked phải gọi getDouble với min= và max=."""
    source = _PAGE.read_text(encoding="utf-8")
    split_start = source.find("def _on_split_clicked(self)")
    split_end = source.find("\n    def ", split_start + 1)
    snippet = source[split_start:split_end]

    assert "min=" in snippet, "getDouble trong _on_split_clicked phải có min="
    assert "max=" in snippet, "getDouble trong _on_split_clicked phải có max="
    assert "minValue=" not in snippet
    assert "maxValue=" not in snippet
