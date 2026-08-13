"""[v3.23.113] Test cảnh báo chống mất việc khi đóng app + cờ chưa lưu của Editor.

Dùng kỹ thuật gọi method dạng unbound với stub nhẹ -> kiểm logic mà không dựng GUI thật
(tránh khởi tạo native gây segfault headless).
"""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")


# --- has_unsaved_changes ---------------------------------------------------

class _Snap:
    def __init__(self, dirty: bool) -> None:
        self.is_dirty = dirty


class _Service:
    def __init__(self, dirty: bool) -> None:
        self._dirty = dirty

    def snapshot_state(self) -> _Snap:
        return _Snap(self._dirty)


class _VM:
    def __init__(self, dirty: bool) -> None:
        self._service = _Service(dirty)


class _Model:
    def __init__(self, n: int) -> None:
        self.events = list(range(n))


class _EditorStub:
    def __init__(self, n_events: int, dirty: bool) -> None:
        self._table_model = _Model(n_events)
        self._view_model = _VM(dirty)


def _has_unsaved():
    from subtitles_extractor.presentation.pages.editor_page import EditorPage
    return EditorPage.has_unsaved_changes


def test_unsaved_true_when_dirty_with_events() -> None:
    fn = _has_unsaved()
    assert fn(_EditorStub(3, dirty=True)) is True


def test_unsaved_false_when_clean() -> None:
    fn = _has_unsaved()
    assert fn(_EditorStub(3, dirty=False)) is False


def test_unsaved_false_when_empty() -> None:
    fn = _has_unsaved()
    assert fn(_EditorStub(0, dirty=True)) is False


# --- _collect_close_warnings ----------------------------------------------

class _PageVM:
    def __init__(self, busy: bool) -> None:
        self._view_model = type("_X", (), {"is_busy": busy})()


class _EditorWithFlag:
    def __init__(self, unsaved: bool) -> None:
        self._unsaved = unsaved

    def has_unsaved_changes(self) -> bool:
        return self._unsaved


def _build():
    from subtitles_extractor.presentation.main_window import _build_close_warnings
    return _build_close_warnings


def test_no_warnings_when_idle_and_saved() -> None:
    fn = _build()
    out = fn(_EditorWithFlag(False), _PageVM(False), _PageVM(False), _PageVM(False))
    assert out == []


def test_warns_for_unsaved_and_busy_tasks() -> None:
    fn = _build()
    warnings = fn(
        _EditorWithFlag(True),  # editor chưa lưu
        _PageVM(True),          # trích xuất bận
        _PageVM(False),         # dịch rảnh
        _PageVM(True),          # tts bận
    )
    text = "\n".join(warnings)
    assert "CHƯA LƯU" in text
    assert "Trích xuất" in text
    assert "TTS" in text
    assert "Dịch" not in text
    assert len(warnings) == 3
