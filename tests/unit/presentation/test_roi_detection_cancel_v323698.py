"""[v3.23.698] Nút Huỷ dừng được lượt tự dò vùng phụ đề (thường và phim bộ).

Trước bản này nút Huỷ bật trong lúc dò (trang đang bận) nhưng không có tác dụng: lượt dò đầy đủ (~5 phút trên CPU, đo tập 3:
335 s) chạy hết rồi mới trả quyền; hàng loạt bị huỷ cũng phải chờ.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

from subtitles_extractor.presentation.workers.detection_workers import (
    DetectAutoRoiWorker,
    DetectSeriesRoiWorker,
)


class _SlowUseCase:
    """Bộ dò giả: báo tiến độ 100 lần (như vòng lấy mẫu thật); bấm Huỷ ở lần thứ 5."""

    def __init__(self) -> None:
        self.worker = None
        self.reported = 0

    def _loop(self, progress_callback) -> object:
        for i in range(100):
            if i == 5:
                self.worker.request_cancel()
            progress_callback(i, 100, "Đang trích xuất frame mẫu")
            self.reported += 1
        return "ROI"

    def supports_analyze_only(self) -> bool:
        return False

    def execute(self, *, metadata, step_ms, batch_size, vram_flags):
        return self._loop(lambda *a: self.worker._on_progress(*a))

    def detect_series_roi(self, metadata, previous_roi, *, step_ms, batch_size, progress_callback, vram_flags):
        return self._loop(progress_callback)


def _run(worker_cls, **kwargs) -> tuple[_SlowUseCase, dict]:
    use_case = _SlowUseCase()
    worker = worker_cls(use_case=use_case, metadata=SimpleNamespace(), **kwargs)
    use_case.worker = worker
    seen = {"finished": [], "failed": [], "cancelled": 0, "done": 0}
    worker.finished.connect(seen["finished"].append)
    worker.failed.connect(seen["failed"].append)
    worker.cancelled.connect(lambda: seen.__setitem__("cancelled", seen["cancelled"] + 1))
    worker.done.connect(lambda: seen.__setitem__("done", seen["done"] + 1))
    worker.run()
    return use_case, seen


@pytest.mark.parametrize(("worker_cls", "kwargs"), [
    (DetectAutoRoiWorker, {}),
    (DetectSeriesRoiWorker, {"previous_roi": None}),
])
def test_cancel_stops_detection_mid_way(worker_cls, kwargs) -> None:
    use_case, seen = _run(worker_cls, **kwargs)
    assert use_case.reported == 5                      # dừng ngay lần báo tiến độ kế tiếp
    assert seen == {"finished": [], "failed": [], "cancelled": 1, "done": 1}


def test_cancel_after_detection_finished_does_not_apply_result() -> None:
    class _Fast:
        def supports_analyze_only(self) -> bool:
            return False

        def execute(self, **_k):
            worker.request_cancel()            # bấm Huỷ đúng lúc bộ dò vừa xong
            return "ROI"

    worker = DetectAutoRoiWorker(use_case=_Fast(), metadata=SimpleNamespace())
    got: list = []
    cancelled: list = []
    worker.finished.connect(got.append)
    worker.cancelled.connect(lambda: cancelled.append(1))
    worker.run()
    assert got == [] and cancelled == [1]


def test_view_model_cancel_reaches_detection_worker() -> None:
    from subtitles_extractor.presentation.view_models.extract_page_view_model import ExtractPageViewModel

    calls: list[str] = []
    vm = SimpleNamespace(_detect_worker=SimpleNamespace(request_cancel=lambda: calls.append("detect")))
    ExtractPageViewModel.cancel_extraction(vm)
    assert calls == ["detect"]


def test_page_reports_cancelled_detection() -> None:
    from subtitles_extractor.presentation.pages.extract_page import ExtractPage

    statuses: list[str] = []
    page = SimpleNamespace(
        _set_status=statuses.append,
        _translator=SimpleNamespace(translate=lambda key: key),
    )
    ExtractPage._on_detection_cancelled(page)
    assert statuses == ["extract.ss_roi_detect_cancelled"]
