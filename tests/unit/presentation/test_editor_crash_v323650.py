"""[v3.23.650] Chặn tái phát crash native khi sửa chữ trực tiếp trong ô bảng ở trang Chỉnh sửa.

Lỗi gốc: ``RichTextSubtitleDelegate.setModelData`` gọi ``model.setData`` lần 1 (EditRole) → model phát
``text_edit_requested`` đồng bộ → ViewModel → ``_refresh_table`` → ``beginResetModel``/``endResetModel``
NGAY TRONG lúc commit; lệnh ``setData`` lần 2 (UserRole) vẫn dùng QModelIndex của proxy cũ
(use-after-free trong QSortFilterProxyModel → SIGSEGV, chương trình biến mất không để lại log).

Crash native giết cả tiến trình test nên kịch bản chạy trong tiến trình con.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("PySide6")

_PRELUDE = textwrap.dedent(
    """
    import os, sys, tempfile, time
    from pathlib import Path
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from subtitles_extractor.composition.container import ApplicationContainer
    from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
    from subtitles_extractor.domain.value_objects.time_interval import TimeInterval
    from subtitles_extractor.infrastructure.settings.json_file_repository import JsonFileRepository
    from subtitles_extractor.presentation.pages import editor_page as ep

    ApplicationContainer.preload_ocr_engine_async = lambda self: None
    tmp = Path(tempfile.mkdtemp(prefix="ed650_"))
    container = ApplicationContainer(
        settings_repository=JsonFileRepository(file_path=tmp / "c.json"),
        i18n_data_dir=Path(ep.__file__).resolve().parents[2] / "data",
        user_data_dir=tmp,
    )
    page = ep.EditorPage(container)
    page.resize(1400, 900)
    page.show()

    def pump(ms=50):
        end = time.monotonic() + ms / 1000
        while time.monotonic() < end:
            app.processEvents()
            time.sleep(0.003)

    def make_events(n):
        return [
            SubtitleEvent(index=i + 1, text=f"dòng {i + 1}",
                          interval=TimeInterval(start_sec=i * 2.0, end_sec=i * 2.0 + 1.5),
                          confidence=0.9, frame_count=10)
            for i in range(n)
        ]

    page.load_events(make_events(40), confirm=False)
    pump(150)
    """
)


def _run(body: str, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    script = _PRELUDE + textwrap.dedent(body)
    import os
    from pathlib import Path

    src = str(Path(__file__).resolve().parents[3] / "src")
    env = dict(os.environ, PYTHONPATH=src, QT_QPA_PLATFORM="offscreen")
    return subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", script],
        capture_output=True, text=True, timeout=timeout, env=env, check=False,
    )


def test_direct_cell_edit_commit_does_not_segfault_v323650() -> None:
    """Double-click ô Nội dung → gõ → click dòng khác: không được SIGSEGV."""
    result = _run(
        """
        proxy = page._proxy_model
        idx = proxy.index(1, ep.COL_TEXT)
        rect = page._table.visualRect(idx)
        QTest.mouseClick(page._table.viewport(), Qt.MouseButton.LeftButton, pos=rect.center())
        QTest.mouseDClick(page._table.viewport(), Qt.MouseButton.LeftButton, pos=rect.center())
        pump(100)
        editor = QApplication.focusWidget()
        QTest.keyClicks(editor, "abc")
        pump(50)
        rect2 = page._table.visualRect(proxy.index(3, ep.COL_INDEX))
        QTest.mouseClick(page._table.viewport(), Qt.MouseButton.LeftButton, pos=rect2.center())
        pump(300)
        print("TEXT=" + page._table_model.events[1].text)
        """
    )
    assert result.returncode == 0, f"exit={result.returncode}\n{result.stderr[-2000:]}"
    assert "TEXT=" in result.stdout
    assert "abc" in result.stdout.split("TEXT=")[1]


# ---------------------------------------------------------------------------
# Mốc thời gian không hữu hạn (NaN/inf) — fuzzer giao diện: shift_all(NaN) tạo sự kiện end=NaN, mọi lần vẽ
# bảng sau đó ném ValueError trong data() (cannot convert float NaN to integer) → bảng hỏng, đơ giao diện.
# ---------------------------------------------------------------------------
import math  # noqa: E402

from subtitles_extractor.application.services.subtitle_editor_service import SubtitleEditorService  # noqa: E402
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent  # noqa: E402
from subtitles_extractor.domain.exceptions import ConfigurationError  # noqa: E402
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval  # noqa: E402
from subtitles_extractor.presentation.utils.time_format import (  # noqa: E402
    seconds_to_ass,
    seconds_to_display,
)


def _events(n: int = 3) -> list[SubtitleEvent]:
    return [
        SubtitleEvent(index=i + 1, text=f"t{i}", interval=TimeInterval(start_sec=i * 2.0, end_sec=i * 2.0 + 1.0),
                      confidence=0.9, frame_count=5)
        for i in range(n)
    ]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_time_interval_rejects_non_finite_v323650(bad: float) -> None:
    with pytest.raises(ConfigurationError):
        TimeInterval(start_sec=bad, end_sec=1.0)
    with pytest.raises(ConfigurationError):
        TimeInterval(start_sec=0.0, end_sec=bad)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_shift_all_non_finite_leaves_state_and_undo_untouched_v323650(bad: float) -> None:
    service = SubtitleEditorService()
    service.load(_events())
    before = [(e.start_sec, e.end_sec) for e in service.snapshot_state().events]
    with pytest.raises(ConfigurationError):
        service.shift_all(bad)
    assert [(e.start_sec, e.end_sec) for e in service.snapshot_state().events] == before
    assert not service.snapshot_state().can_undo


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_update_timing_non_finite_rejected_without_undo_entry_v323650(bad: float) -> None:
    service = SubtitleEditorService()
    service.load(_events())
    with pytest.raises(ConfigurationError):
        service.update_timing(1, 2.0, bad)
    state = service.snapshot_state()
    assert not state.can_undo
    assert all(math.isfinite(e.end_sec) for e in state.events)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), -5.0])
def test_time_formatters_never_raise_on_non_finite_v323650(bad: float) -> None:
    assert seconds_to_display(bad) == "00:00:00.000"
    assert seconds_to_ass(bad) == "0:00:00.00"


def test_time_cell_edit_commit_does_not_segfault_v323650() -> None:
    """Sửa ô Bắt đầu/Kết thúc (delegate mặc định) rồi click sang dòng khác: không crash, giá trị được áp dụng."""
    result = _run(
        """
        proxy = page._proxy_model
        idx = proxy.index(2, ep.COL_START)
        rect = page._table.visualRect(idx)
        QTest.mouseClick(page._table.viewport(), Qt.MouseButton.LeftButton, pos=rect.center())
        QTest.mouseDClick(page._table.viewport(), Qt.MouseButton.LeftButton, pos=rect.center())
        pump(100)
        editor = QApplication.focusWidget()
        editor.selectAll()
        QTest.keyClicks(editor, "00:00:03.000")
        rect2 = page._table.visualRect(proxy.index(6, ep.COL_INDEX))
        QTest.mouseClick(page._table.viewport(), Qt.MouseButton.LeftButton, pos=rect2.center())
        pump(300)
        print("START=%.3f" % page._table_model.events[2].start_sec)
        """
    )
    assert result.returncode == 0, f"exit={result.returncode}\n{result.stderr[-2000:]}"
    assert "START=3.000" in result.stdout


def test_waveform_click_syncs_edit_panel_and_focus_out_keeps_data_v323650() -> None:
    """Bấm sóng âm sang dòng khác rồi focus vào/ra ô chữ không được ghi đè chữ dòng mới bằng chữ dòng cũ."""
    result = _run(
        """
        page._table.selectRow(page._proxy_model.mapFromSource(page._table_model.index(3, 0)).row())
        pump(100)
        old_text = page._table_model.events[3].text
        page._last_waveform_seek_time = 0.0
        target = page._table_model.events[20]
        page._on_waveform_seek(target.start_sec + 0.2)
        pump(50)
        print("PANEL=" + page._edit_text.toPlainText())
        print("ROW20=" + target.text)
        page._edit_text.setFocus()
        pump(50)
        page._table.setFocus()
        pump(100)
        print("AFTER20=" + page._table_model.events[20].text)
        print("OLD3=" + page._table_model.events[3].text + "|" + old_text)
        """
    )
    assert result.returncode == 0, result.stderr[-2000:]
    out = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line and line.split("=")[0].isupper())
    assert out["PANEL"] == out["ROW20"] == out["AFTER20"] == "dòng 21"
    assert out["OLD3"] == "dòng 4|dòng 4"


def test_waveform_seek_exception_resets_sync_flag_and_reconnects_v323650() -> None:
    """Ngoại lệ trong lúc đồng bộ từ waveform không được để cờ kẹt hay selectionChanged mất kết nối."""
    result = _run(
        """
        page._last_waveform_seek_time = 0.0
        def boom(*a, **k):
            raise RuntimeError("seek boom")
        page._video_widget.seek = boom
        try:
            page._on_waveform_seek(5.0)
        except RuntimeError:
            pass
        print("FLAG=" + str(page._is_syncing_from_waveform))
        page._table.selectRow(7)
        pump(100)
        print("PANEL=" + page._edit_text.toPlainText())
        """
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "FLAG=False" in result.stdout
    assert "PANEL=dòng 8" in result.stdout


def test_select_all_then_edit_keeps_few_selection_ranges_and_current_row_v323650() -> None:
    """Ctrl+A rồi sửa một dòng: vùng chọn phải được khôi phục thành vài range (không phải một range/dòng)."""
    result = _run(
        """
        page.load_events(make_events(2000), confirm=False)
        pump(150)
        page._table.selectAll()
        from PySide6.QtCore import QItemSelectionModel
        page._table.selectionModel().setCurrentIndex(page._proxy_model.index(5, 0), QItemSelectionModel.SelectionFlag.NoUpdate)
        pump(50)
        page._view_model.update_text(10, "đã sửa")
        pump(200)
        print("RANGES=%d" % len(page._table.selectionModel().selection()))
        print("ROWS=%d" % len(page._table.selectionModel().selectedRows()))
        print("CUR=%d" % page._current_row())
        """
    )
    assert result.returncode == 0, result.stderr[-2000:]
    out = dict(line.split("=", 1) for line in result.stdout.splitlines() if line.startswith(("RANGES", "ROWS", "CUR")))
    assert int(out["RANGES"]) <= 3
    assert int(out["ROWS"]) == 2000
    assert int(out["CUR"]) == 1999  # currentIndex được đặt về hàng cuối của vùng chọn sau khôi phục


def test_start_reocr_missing_video_reports_error_instead_of_raising_v323650() -> None:
    """Video đã bị xoá/đổi tên: Re-OCR phải báo lỗi cho người dùng, không ném VideoNotFoundError ra khỏi slot."""
    result = _run(
        """
        from pathlib import Path
        errors = []
        page._view_model.error_occurred.connect(errors.append)
        try:
            page._view_model.start_reocr(Path("/nonexistent/x.mp4"), [0, 1], None)
            print("RAISED=no")
        except Exception as exc:
            print("RAISED=" + type(exc).__name__)
        print("ERRORS=%d" % len(errors))
        """
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "RAISED=no" in result.stdout
    assert "ERRORS=1" in result.stdout


def test_ctrl_down_while_filtered_jumps_to_source_row_v323650() -> None:
    """Ctrl+↓ khi đang lọc: video phải tua tới dòng NGUỒN của hàng proxy mới, không phải số hàng proxy."""
    result = _run(
        """
        page._proxy_model.set_filter_params("dòng 3", 0)
        pump(100)
        jumps = []
        page._jump_to_event_at = lambda r: jumps.append(r)
        print("PROXYROWS=%d" % page._proxy_model.rowCount())
        page._table.selectRow(0)
        pump(50)
        page._step_selected_row(+1)
        pump(50)
        src = page._proxy_model.mapToSource(page._proxy_model.index(1, 0)).row()
        print("JUMPS=" + ",".join(map(str, jumps)))
        print("EXPECT=%d" % src)
        """
    )
    assert result.returncode == 0, result.stderr[-2000:]
    out = dict(line.split("=", 1) for line in result.stdout.splitlines() if line.startswith(("JUMPS", "EXPECT", "PROXYROWS")))
    assert int(out["PROXYROWS"]) > 2
    assert out["JUMPS"] == out["EXPECT"]
    assert int(out["EXPECT"]) != 1  # chắc chắn khác số hàng proxy


def test_play_button_icon_follows_canvas_state_v323650() -> None:
    result = _run(
        """
        seen = []
        page._btn_play.setIcon = lambda icon: seen.append(icon)
        page._on_canvas_state_changed(True)
        page._on_canvas_state_changed(False)
        print("ICONS=%d" % len(seen))
        """
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "ICONS=2" in result.stdout


def test_reocr_result_after_new_data_loaded_is_discarded_v323650() -> None:
    """Re-OCR của video/phụ đề cũ xong SAU khi nạp dữ liệu mới: kết quả không được chèn vào phụ đề mới."""
    result = _run(
        """
        from types import SimpleNamespace
        vm = page._view_model

        class _Thread:
            def isRunning(self): return True
            def requestInterruption(self): pass

        worker = object()
        vm._reocr_worker = worker
        vm._reocr_thread = _Thread()
        vm.sender = lambda: worker
        vm.load_from_events(make_events(5))       # nạp dữ liệu mới → Re-OCR cũ thành "cũ"
        extra = [SubtitleEvent(index=1, text="cũ", interval=TimeInterval(start_sec=1000.0, end_sec=1001.0),
                               confidence=0.9, frame_count=3)]
        response = SimpleNamespace(was_cancelled=False, new_events=extra, replaced_uids=["khong-co"],
                                   elapsed_seconds=1.0)
        vm._on_reocr_finished(response)
        print("COUNT=%d" % len(vm._service.snapshot_state().events))
        """
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "COUNT=5" in result.stdout


def test_insert_after_tight_gap_keeps_events_sorted_v323650() -> None:
    service = SubtitleEditorService()
    service.load([
        SubtitleEvent(index=1, text="a", interval=TimeInterval(0.0, 1.0), confidence=0.9, frame_count=1),
        SubtitleEvent(index=2, text="b", interval=TimeInterval(1.02, 2.0), confidence=0.9, frame_count=1),
    ])
    state = service.insert_after(0, "mới")
    starts = [e.start_sec for e in state.events]
    assert starts == sorted(starts)
    assert [e.text for e in state.events] == ["a", "mới", "b"]


# ---------------------------------------------------------------------------
# mpv: giải phóng lõi khi đóng app, không tạo lõi ngầm, nạp lại khi HWND đổi, tệp lỗi không ném ra slot.
# ---------------------------------------------------------------------------
_FAKE_MPV = """
        import subtitles_extractor.infrastructure.video.mpv_player_adapter as _ad
        from subtitles_extractor.presentation.widgets import mpv_video_widget as mvw

        CREATED = []

        class FakePlayer:
            def __init__(self, wid, mpv_options=None):
                self.wid = wid
                self.released = False
                self.loaded = []
                self.is_loaded = False
                self.is_playing = False
                self.position_sec = 12.5
                self.duration_sec = 100.0
                self.video_width = 720
                self.video_height = 1280
                self.eof_reached = False
                self.fail_load = False
                CREATED.append(self)
            def load(self, path):
                if self.fail_load:
                    raise FileNotFoundError(str(path))
                self.loaded.append(path); self.is_loaded = True
            def pause(self): self.is_playing = False
            def play(self): self.is_playing = True
            def seek(self, s): self.position_sec = s
            def release(self): self.released = True
            def send_command(self, *a): return True
            def set_mute(self, m): pass
            def set_volume(self, v): pass
            def set_speed(self, s): pass
            def observe_property(self, *a): pass

        _ad.MpvPlayerAdapter = FakePlayer
"""


def test_release_all_mpv_players_releases_every_live_widget_v323650() -> None:
    result = _run(
        _FAKE_MPV
        + """
        from pathlib import Path
        w1 = mvw.MpvVideoWidget(); w2 = mvw.MpvVideoWidget()
        for w in (w1, w2):
            w.resize(300, 200); w.show()
        pump(50)
        w1.load(Path("/tmp/a.mp4")); w2.load(Path("/tmp/b.mp4"))
        n = mvw.release_all_mpv_players()
        print("RELEASED=%d" % n)
        print("ALL=" + str(all(p.released for p in CREATED)))
        print("NONE=" + str(w1.peek_player() is None and w2.peek_player() is None))
        """
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "RELEASED=2" in result.stdout and "ALL=True" in result.stdout and "NONE=True" in result.stdout


def test_peek_player_and_preview_do_not_create_mpv_core_v323650() -> None:
    result = _run(
        _FAKE_MPV
        + """
        page._video_widget.player = lambda: (_ for _ in ()).throw(AssertionError("player() không được gọi"))
        page._generate_ass_preview()
        page._jump_to_event_at(1)
        page._cleanup_old_preview_tracks("x")
        print("CREATED=%d" % len(CREATED))
        """
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "CREATED=0" in result.stdout


def test_mpv_rebinds_when_native_window_id_changes_v323650() -> None:
    result = _run(
        _FAKE_MPV
        + """
        from pathlib import Path
        w = mvw.MpvVideoWidget(); w.resize(300, 200); w.show(); pump(50)
        w.load(Path("/tmp/a.mp4"))
        first = w.peek_player(); first.position_sec = 33.0
        print("SAME=" + str(w.rebind_if_window_changed()))          # HWND không đổi → không làm gì
        w._player_wid = w._player_wid + 12345                       # mô phỏng HWND đã bị tạo lại
        print("REBOUND=" + str(w.rebind_if_window_changed()))
        second = w.peek_player()
        print("NEWCORE=" + str(second is not first and first.released and not second.released))
        print("POS=%.1f" % second.position_sec)
        """
    )
    assert result.returncode == 0, result.stderr[-2000:]
    out = result.stdout
    assert "SAME=False" in out and "REBOUND=True" in out and "NEWCORE=True" in out and "POS=33.0" in out


def test_mpv_load_missing_file_does_not_raise_v323650() -> None:
    result = _run(
        _FAKE_MPV
        + """
        from pathlib import Path
        w = mvw.MpvVideoWidget(); w.resize(300, 200); w.show(); pump(50)
        w.player().fail_load = True
        try:
            w.load(Path("/tmp/missing.mp4"))
            print("RAISED=no")
        except Exception as exc:
            print("RAISED=" + type(exc).__name__)
        """
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "RAISED=no" in result.stdout


def test_split_cancelled_when_list_changed_while_dialog_open_v323650() -> None:
    """Tách dòng: nếu danh sách đổi trong lúc hộp thoại mở thì không tách nhầm câu."""
    result = _run(
        """
        from PySide6.QtWidgets import QInputDialog
        calls = []
        page._view_model.split_event = lambda i, v: calls.append((i, v))
        page._table.selectRow(5)
        pump(50)
        def fake_get_double(*a):
            page._view_model.delete_event(0)          # Re-OCR/nạp lại xong trong lúc hộp thoại mở: dồn chỉ số
            return 5.5, True
        QInputDialog.getDouble = staticmethod(fake_get_double)
        page._on_split_clicked()
        print("SPLITS=%d" % len(calls))
        """
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "SPLITS=0" in result.stdout


def test_split_proceeds_when_list_unchanged_v323650() -> None:
    result = _run(
        """
        from PySide6.QtWidgets import QInputDialog
        calls = []
        page._view_model.split_event = lambda i, v: calls.append((i, v))
        page._table.selectRow(5)
        pump(50)
        QInputDialog.getDouble = staticmethod(lambda *a: (10.5, True))
        page._on_split_clicked()
        print("SPLITS=%d" % len(calls))
        """
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "SPLITS=1" in result.stdout


def test_debug_output_label_has_translator_and_paints_empty_without_crash_v323650() -> None:
    """Tab "Output" của trang Gỡ lỗi khi chưa có ảnh: trước đây AttributeError trong paintEvent + QPainter còn active → segfault."""
    result = _run(
        """
        from subtitles_extractor.presentation.pages import debug_page as dp
        label = dp.InteractiveOutputImagePreviewLabel(page._translator)
        label.resize(300, 200); label.show(); pump(100)
        label.grab()
        print("TRANSLATOR=" + str(label._translator is not None))
        label.deleteLater(); pump(100)
        print("DONE")
        """
    )
    assert result.returncode == 0, f"exit={result.returncode}\n{result.stderr[-1500:]}"
    assert "TRANSLATOR=True" in result.stdout and "DONE" in result.stdout


def test_debug_page_output_tab_survives_empty_frame_v323650() -> None:
    result = _run(
        """
        from subtitles_extractor.presentation.pages.debug_page import DebugPage
        dbg = DebugPage(container)
        dbg.resize(1200, 800); dbg.show(); pump(150)
        dbg._tabs_img.setCurrentIndex(1); pump(150)
        dbg._lbl_img_out.grab()
        dbg.close(); dbg.deleteLater(); pump(150)
        print("DONE")
        """
    )
    assert result.returncode == 0, f"exit={result.returncode}\n{result.stderr[-1500:]}"
    assert "DONE" in result.stdout
