"""[v3.23.682] Phím tắt duyệt nhanh trang Chỉnh sửa: F8 dòng nghi ngờ OCR kế tiếp, Ctrl+[ / Ctrl+] đặt mốc theo vị trí phát, F1."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QAbstractItemView, QApplication


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_next_suspect_row_wraps() -> None:
    from subtitles_extractor.presentation.pages.editor_page import next_suspect_row

    assert next_suspect_row([], 3) is None
    assert next_suspect_row({4, 9, 15}, 4) == 9
    assert next_suspect_row({4, 9, 15}, 15) == 4              # vòng lại đầu
    assert next_suspect_row({4, 9, 15}, -1) == 4               # chưa chọn dòng
    assert next_suspect_row({4, 9, 15}, 9, forward=False) == 4
    assert next_suspect_row({4, 9, 15}, 4, forward=False) == 15


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def __getattr__(self, name):
        return lambda *a, **k: self.calls.append((name, a, k))


def _fake_page():
    from subtitles_extractor.presentation.pages.editor_page import EditorPage

    rec = _Recorder()
    page = SimpleNamespace(
        _table=SimpleNamespace(state=lambda: QAbstractItemView.State.NoState),
        _jump_to_suspect=lambda forward=True: rec.calls.append(("suspect", forward)),
        _set_time_at_playhead=lambda is_start: rec.calls.append(("playhead", is_start)),
        _on_show_shortcuts=lambda: rec.calls.append(("help",)),
        _change_speed=lambda d: rec.calls.append(("speed", d)),
        _find=lambda forward=True: rec.calls.append(("find", forward)),
    )
    return page, rec, EditorPage._handle_global_keys


def test_keys_route_to_new_actions(app: QApplication) -> None:
    page, rec, handle = _fake_page()
    none, ctrl, shift = (Qt.KeyboardModifier.NoModifier, Qt.KeyboardModifier.ControlModifier,
                         Qt.KeyboardModifier.ShiftModifier)
    assert handle(page, Qt.Key.Key_F8, none)
    assert handle(page, Qt.Key.Key_F8, shift)
    assert handle(page, Qt.Key.Key_BracketLeft, ctrl)
    assert handle(page, Qt.Key.Key_BracketRight, ctrl)
    assert handle(page, Qt.Key.Key_F1, none)
    assert handle(page, Qt.Key.Key_BracketLeft, none)          # [ không Ctrl vẫn là chỉnh tốc độ
    assert rec.calls == [
        ("suspect", True), ("suspect", False), ("playhead", True), ("playhead", False), ("help",), ("speed", -0.25),
    ]


def _timing_page(position: float, start: float = 10.0, end: float = 12.0):
    from subtitles_extractor.presentation.pages.editor_page import EditorPage

    updates: list[tuple] = []
    warnings: list[str] = []
    page = SimpleNamespace(
        _current_row=lambda: 0,
        _table_model=SimpleNamespace(events=[SimpleNamespace(start_sec=start, end_sec=end)]),
        _video_widget=SimpleNamespace(position_sec=position),
        _view_model=SimpleNamespace(update_timing=lambda i, s, e: updates.append((i, s, e))),
        _translator=SimpleNamespace(translate=lambda key, **k: key),
    )
    return page, updates, warnings, EditorPage._set_time_at_playhead


def test_set_time_at_playhead(app: QApplication, monkeypatch: pytest.MonkeyPatch) -> None:
    from subtitles_extractor.presentation.pages import editor_page

    warned: list[str] = []
    monkeypatch.setattr(editor_page.InfoBar, "warning", lambda *a, **k: warned.append(a[1]))
    page, updates, _w, set_at = _timing_page(position=9.5)
    set_at(page, is_start=True)
    page2, updates2, _w2, _ = _timing_page(position=12.75)
    set_at(page2, is_start=False)
    assert updates == [(0, 9.5, 12.0)] and updates2 == [(0, 10.0, 12.75)]

    page3, updates3, _w3, _ = _timing_page(position=12.5)    # mốc đầu sau mốc cuối → không đổi
    set_at(page3, is_start=True)
    assert updates3 == [] and warned


def test_help_table_lists_new_keys() -> None:
    from subtitles_extractor.presentation.pages.editor_page import editor_shortcuts_html

    class _Tr:
        def translate(self, key: str, **_k: object) -> str:
            return key

    html = editor_shortcuts_html(_Tr())
    for key in ("F8 / Shift+F8", "Ctrl+[ / Ctrl+]", "F1"):
        assert key in html


# ── Trang Lồng tiếng ──
@pytest.fixture
def tts(app: QApplication, tmp_path, monkeypatch: pytest.MonkeyPatch):
    from pathlib import Path

    from PySide6.QtCore import QSettings

    from subtitles_extractor.composition.container import ApplicationContainer
    from subtitles_extractor.infrastructure.settings.json_file_repository import (
        JsonFileRepository,
    )
    from subtitles_extractor.presentation.pages import tts_page as tts_page_module

    ini = tmp_path / "page.ini"
    monkeypatch.setattr(tts_page_module, "QSettings", lambda *_a: QSettings(str(ini), QSettings.Format.IniFormat))
    container = ApplicationContainer(
        settings_repository=JsonFileRepository(file_path=tmp_path / "config.json"),
        i18n_data_dir=Path(tts_page_module.__file__).resolve().parents[2] / "data",
        user_data_dir=tmp_path,
    )
    page = tts_page_module.TTSPage(container)
    yield page
    page.deleteLater()


def test_tts_review_shortcuts(tts, monkeypatch: pytest.MonkeyPatch) -> None:
    from PySide6.QtWidgets import QTableWidgetItem

    keys = {s.key().toString() for s in tts._review_shortcuts}
    assert keys == {"Space", "Ctrl+R", "F8", "Shift+F8"}
    assert all(s.context() == Qt.ShortcutContext.WidgetShortcut for s in tts._review_shortcuts)

    table = tts._table
    table.setRowCount(6)
    for row in range(6):
        item = QTableWidgetItem(str(row))
        item.setData(Qt.ItemDataRole.UserRole, 100 + row)
        table.setItem(row, 0, item)
    monkeypatch.setattr(tts._pronunciation, "wrong_indices", lambda: {101, 104})
    monkeypatch.setattr(tts, "_abnormal_event_indices", lambda: [103])
    table.setCurrentCell(0, 0)
    seen = []
    for _ in range(4):
        tts._jump_to_review_line(forward=True)
        seen.append(table.currentRow())
    assert seen == [1, 3, 4, 1]                      # vòng lại đầu
    tts._jump_to_review_line(forward=False)
    assert table.currentRow() == 4

    monkeypatch.setattr(tts._pronunciation, "wrong_indices", lambda: set())
    monkeypatch.setattr(tts, "_abnormal_event_indices", lambda: [])
    tts._jump_to_review_line()
    assert tts._stage_lbl.text() == tts._translator.translate("tts.review_none")


# ── "Việc cần duyệt" trên thanh điều hướng ──
def test_nav_badge(app: QApplication) -> None:
    from PySide6.QtWidgets import QWidget

    from subtitles_extractor.presentation.fluent_compat.window import FluentWindow

    window = FluentWindow()
    page, other = QWidget(), QWidget()
    window.addSubInterface(page, None, "✏️  Chỉnh sửa")
    window.set_nav_badge(page, 12, "12 dòng nghi")
    button = window._button_for_widget[page]
    assert button.text() == "✏️  Chỉnh sửa  ● 12" and button.toolTip() == "12 dòng nghi"
    assert window.nav_badge(page) == 12
    window.set_nav_badge(page, 0, "x")
    assert button.text() == "✏️  Chỉnh sửa" and button.toolTip() == "" and window.nav_badge(page) == 0
    window.set_nav_badge(other, 5)                  # trang chưa đăng ký: bỏ qua, không lỗi
    assert window.nav_badge(other) == 0
    window.deleteLater()


def test_tts_review_count_signal(tts, qtbot, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tts._pronunciation, "wrong_indices", lambda: {1, 2})
    monkeypatch.setattr(tts, "_abnormal_event_indices", lambda: [2, 3])
    monkeypatch.setattr(tts._line_edits, "_review_count", 4)
    assert tts.review_count() == 3 + 4
    with qtbot.waitSignal(tts.review_count_changed, timeout=3000) as blocker:
        tts._schedule_review_count()
    assert blocker.args == [7]


def test_editor_review_count_counts_suspects(app: QApplication) -> None:
    from subtitles_extractor.presentation.pages.editor_page import EditorPage

    emitted: list[int] = []
    page = SimpleNamespace(
        _proxy_model=SimpleNamespace(suspects=lambda: {3: ["x"], 8: ["y"]}),
        review_count_changed=SimpleNamespace(emit=emitted.append),
    )
    page.review_count = lambda: EditorPage.review_count(page)
    EditorPage._emit_review_count(page)
    assert emitted == [2]
