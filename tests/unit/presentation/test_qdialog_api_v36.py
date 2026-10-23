"""Test bảo vệ cách gọi QInputDialog.getDouble() ở nút "Tách câu".

[v3.23.613] Bản v3.6 giả định PyQt6 (tên tham số ``min``/``max``) nhưng ứng dụng chạy
PySide6, nơi tên là ``minValue``/``maxValue`` → ``min=`` làm getDouble ném TypeError ngay
khi bấm "Tách câu" (đo trên PySide6 6.11.2). Tên tham số khác nhau giữa các binding nên
cận dưới/cận trên phải truyền THEO VỊ TRÍ — đúng với mọi binding. Hành vi thật được kiểm
ở ``test_ui_bug_sweep_v323613.py::TestEditorSplit``.
"""
from pathlib import Path

_PAGE = (
    Path(__file__).resolve().parents[3]
    / "src/subtitles_extractor/presentation/pages/editor_page.py"
)


def _split_snippet() -> str:
    source = _PAGE.read_text(encoding="utf-8")
    split_start = source.find("def _on_split_clicked(self)")
    split_end = source.find("\n    def ", split_start + 1)
    return source[split_start:split_end]


def test_get_double_bounds_are_positional() -> None:
    """Không dùng tên tham số phụ thuộc binding cho cận dưới/cận trên."""
    snippet = _split_snippet()
    call = snippet[snippet.find("QInputDialog.getDouble("):]
    call = call[: call.find("\n        )") + 10]
    for keyword in ("min=", "max=", "minValue=", "maxValue="):
        assert keyword not in call, f"getDouble không được dùng {keyword!r}"
    assert "start + 0.001" in call and "end - 0.001" in call
