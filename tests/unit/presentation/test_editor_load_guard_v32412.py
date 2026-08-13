"""[v3.23.112] Test guard chống mất dữ liệu khi nạp phụ đề từ luồng khác vào Editor.

Gọi ``EditorPage.load_events`` dạng unbound với một stub nhẹ -> kiểm logic guard mà KHÔNG
phải dựng toàn bộ trang (tránh khởi tạo native gây segfault trong sandbox headless).
"""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")


class _StubVM:
    def __init__(self) -> None:
        self.loaded_events: list | None = None

    def load_from_events(self, events: list) -> None:
        self.loaded_events = events


class _StubEditor:
    """Stub tối thiểu mô phỏng các thuộc tính mà load_events đụng tới."""

    def __init__(self, confirm_result: bool) -> None:
        self._confirm_result = confirm_result
        self._view_model = _StubVM()
        self.confirm_called = 0

    def _confirm_discard_unsaved(self) -> bool:
        self.confirm_called += 1
        return self._confirm_result


def _load_events():
    from subtitles_extractor.presentation.pages.editor_page import EditorPage

    return EditorPage.load_events


def test_load_proceeds_when_no_unsaved() -> None:
    load_events = _load_events()
    stub = _StubEditor(confirm_result=True)  # sạch -> confirm trả True
    result = load_events(stub, [1, 2, 3])
    assert result is True
    assert stub._view_model.loaded_events == [1, 2, 3]
    assert stub.confirm_called == 1


def test_load_aborts_when_user_keeps_unsaved() -> None:
    load_events = _load_events()
    stub = _StubEditor(confirm_result=False)  # người dùng chọn GIỮ bản đang sửa
    result = load_events(stub, [1, 2, 3])
    assert result is False
    assert stub._view_model.loaded_events is None  # KHÔNG ghi đè
    assert stub.confirm_called == 1


def test_confirm_false_skips_guard() -> None:
    load_events = _load_events()
    stub = _StubEditor(confirm_result=False)
    # Nơi gọi đã tự hỏi xác nhận -> confirm=False, không hỏi lại, vẫn nạp.
    result = load_events(stub, [9], confirm=False)
    assert result is True
    assert stub._view_model.loaded_events == [9]
    assert stub.confirm_called == 0
