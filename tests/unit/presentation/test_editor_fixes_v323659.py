"""[v3.23.659] Rà BUG khâu Chỉnh sửa: chữ đang gõ bị mất, nút Lưu tệp kẹt."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

import os
import tempfile
import time
from pathlib import Path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication
app = QApplication.instance() or QApplication([])
from subtitles_extractor.composition.container import ApplicationContainer
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval
from subtitles_extractor.infrastructure.settings.json_file_repository import JsonFileRepository
from subtitles_extractor.presentation.pages import editor_page as ep
from subtitles_extractor.domain.value_objects.device_kind import SubtitleFormat

def pump(ms=50):
    end = time.monotonic() + ms / 1000
    while time.monotonic() < end:
        app.processEvents(); time.sleep(0.003)

def make_page():
    ApplicationContainer.preload_ocr_engine_async = lambda self: None
    tmp = Path(tempfile.mkdtemp(prefix="probe_"))
    c = ApplicationContainer(settings_repository=JsonFileRepository(file_path=tmp / "c.json"),
        i18n_data_dir=Path(ep.__file__).resolve().parents[2] / "data", user_data_dir=tmp)
    page = ep.EditorPage(c); page.resize(1400, 900); page.show()
    evs = [SubtitleEvent(index=i+1, text=f"dòng {i+1}", interval=TimeInterval(i*2.0, i*2.0+1.5), confidence=0.9, frame_count=10) for i in range(10)]
    page.load_events(evs, confirm=False); pump(150)
    return page, tmp

def test_export_error_leaves_button_disabled():
    page, tmp = make_page()
    assert page._export_button.isEnabled()
    vm = page._view_model
    (tmp/"afile").write_text("x"); target = tmp / "afile" / "x.srt"
    errs=[]; vm.error_occurred.connect(errs.append)
    vm = page._view_model
    # simulate the export button flow
    page._export_button.setEnabled(False)
    assert vm.export_to_file(target, SubtitleFormat.SRT)
    for _ in range(100):
        pump(20)
        if not vm._is_busy: break
    pump(100)
    print("errs", errs, "busy", vm._is_busy, "enabled", page._export_button.isEnabled())
    assert page._export_button.isEnabled(), "export button stuck disabled after export error"

def test_reocr_finish_leaves_export_disabled():
    page, tmp = make_page()
    vm = page._view_model
    vm._set_busy(True)   # Re-OCR started
    st = vm._service.snapshot_state()
    vm.state_changed.emit(st)  # _on_reocr_finished emits while busy
    vm._set_busy(False)  # thread finished
    pump(50)
    assert page._export_button.isEnabled(), "export disabled after re-ocr"

def _type_in_row(page, row, text):
    page._table.selectRow(row); pump(30)
    page.activateWindow(); page._edit_text.setFocus(); pump(30)
    page._edit_text.setPlainText(text); pump(10)
    print("focus edit:", page._edit_text.hasFocus())

def test_merge_button_loses_typed_text():
    page, tmp = make_page()
    _type_in_row(page, 2, "SỬA MỚI")
    page._merge_button.click(); pump(50)
    texts = [e.text for e in page._view_model.current_events]
    print(texts[:4])
    assert any("SỬA MỚI" in t for t in texts)

def test_apply_timing_loses_typed_text():
    page, tmp = make_page()
    _type_in_row(page, 2, "SỬA MỚI")
    page._edit_apply_timing.click(); pump(50)
    texts = [e.text for e in page._view_model.current_events]
    print(texts[:4], repr(page._edit_text.toPlainText()))
    assert any("SỬA MỚI" in t for t in texts) or page._edit_text.toPlainText()=="SỬA MỚI"

def test_async_state_change_overwrites_typed_text():
    page, tmp = make_page()
    _type_in_row(page, 2, "SỬA MỚI")
    vm = page._view_model
    vm.state_changed.emit(vm._service.snapshot_state())  # e.g. Re-OCR finished on other rows
    pump(30)
    print(repr(page._edit_text.toPlainText()))
    assert page._edit_text.toPlainText() == "SỬA MỚI"


def test_reocr_failed_range_keeps_original_lines():
    """[v3.23.659] Một đoạn Re-OCR lỗi → dòng gốc trong đoạn đó phải còn (trước đây bị xoá mà không có bản thay)."""
    from PySide6.QtCore import QObject, Signal
    from subtitles_extractor.application.dtos.reocr_dto import ReOcrResponse, TimeRange

    class _FakeWorker(QObject):
        finished = Signal(object)

    page, _tmp = make_page()
    vm = page._view_model
    events = vm._service.snapshot_state().events
    uids = [events[0].uid, events[3].uid]  # 0.0–1.5s và 6.0–7.5s
    worker = _FakeWorker()
    vm._reocr_worker = worker
    vm._reocr_result_stale = False
    worker.finished.connect(vm._on_reocr_finished)
    new = SubtitleEvent(index=1, text="mới", interval=TimeInterval(0.1, 1.4), confidence=0.9, frame_count=5)
    worker.finished.emit(ReOcrResponse(
        new_events=[new], replaced_uids=uids, elapsed_seconds=0.1, frames_processed=5,
        failed_ranges=(TimeRange(start_sec=5.5, end_sec=8.0),),
    ))
    left = {e.uid for e in vm._service.snapshot_state().events}
    texts = [e.text for e in vm._service.snapshot_state().events]
    assert events[0].uid not in left
    assert events[3].uid in left, "Dòng thuộc đoạn Re-OCR lỗi bị xoá mất."
    assert "mới" in texts


def test_reocr_use_case_reports_failed_ranges():
    from types import SimpleNamespace
    from subtitles_extractor.application.dtos.reocr_dto import TimeRange
    from subtitles_extractor.application.use_cases.reocr import ReOcrUseCase

    class _Meta:
        def execute(self, _p):
            return SimpleNamespace(duration_sec=100.0)

    class _Extract:
        calls = 0

        def execute(self, request, progress):
            _Extract.calls += 1
            if _Extract.calls == 2:
                raise RuntimeError("giải mã lỗi")
            return SimpleNamespace(events=[], frames_processed=1)

    class _Rep:
        def report(self, *a):
            pass

        def is_cancelled(self):
            return False

    sampling = SimpleNamespace(sample_step_sec=0.2, phash_distance_threshold=5, pixel_diff_threshold=10,
        apply_median_blend=False, median_blend_frames=3, vram_upscale_small_text=False,
        vram_upscale_target_height_px=48, vram_add_border=False, vram_border_thickness_px=8,
        vram_sharpen=False, vram_contrast_factor=1.0)
    request = SimpleNamespace(video_path=Path("/tmp/fake.mp4"),
        time_ranges=[TimeRange(start_sec=0.0, end_sec=3.0), TimeRange(start_sec=20.0, end_sec=23.0)],
        merge_window_sec=0.0, replace_uids=["a", "b"], roi=None, sampling=sampling, ocr=SimpleNamespace(),
        builder=SimpleNamespace(), auto_tune_batch=True, save_debug_frames=False, debug_frames_dir=None)
    response = ReOcrUseCase(extract_use_case=_Extract(), load_metadata_use_case=_Meta()).execute(request, progress=_Rep())
    assert len(response.failed_ranges) == 1
    assert response.failed_ranges[0].start_sec >= 15.0
