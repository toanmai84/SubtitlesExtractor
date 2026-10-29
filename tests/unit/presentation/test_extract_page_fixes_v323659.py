"""[v3.23.659] Rà BUG khâu Trích xuất (trang + view model)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("PySide6")

from tests.unit.presentation.test_series_pages_v323658 import (  # noqa: F401
    _pump,
    _videos,
    app,
    extract,
    no_dialogs,
)


def test_primary_roi_flag_does_not_stick(extract: Any) -> None:
    """Tự dò ROI chính không chạy được (chưa có video / đang bận) → cờ kẹt, lần "Tự dò ROI" sau bỏ qua hộp kiểm duyệt."""
    vm = extract._view_model
    vm._video = None
    vm.detect_primary_subtitle_roi()
    assert vm._primary_roi_mode is False


def test_batch_preview_follows_current_episode(extract: Any, tmp_path: Path, no_dialogs) -> None:  # noqa: F811
    """Khung xem trước / đọc khung từng giữ video mở đầu tiên trong suốt lượt hàng loạt."""
    from subtitles_extractor.application.services.batch_extraction_plan import build_batch_plan

    extract._pending_video_path = tmp_path / "video_cu.mp4"
    plan = build_batch_plan(_videos(tmp_path, 2), reference_roi=None, reference_size=None)
    extract._batch_queue = [item for item in plan if item.will_run]
    extract._batch_index, extract._batch_failures, extract._batch_cancelled = 0, [], False
    extract._run_next_batch_item()
    assert extract._pending_video_path == tmp_path / "ep1.mp4"
    extract._batch_queue = []


def test_each_roi_gets_its_own_ocr_cache_and_rebuild_merges(app, tmp_path: Path) -> None:  # noqa: F811
    """Nhiều ROI từng ghi CHUNG một cache OCR: ROI sau đè ROI trước, dựng lại ra phụ đề sai + khung lệch toạ độ."""
    from subtitles_extractor.application.dtos.extract_subtitles_dto import SubtitleBuilderConfig
    from subtitles_extractor.domain.value_objects.roi import Roi
    from subtitles_extractor.infrastructure.serializers.raw_ocr_serializer import RawOcrMeta, save_raw_ocr
    from subtitles_extractor.presentation.view_models.extract_page_view_model import (
        RebuildFromCacheWorker,
        _roi_cache_path,
    )
    from tests.unit.application.test_restore_centering_v323657 import WITHOUT, _frames

    base = tmp_path / "ep.seraw.json.gz"
    assert _roi_cache_path(base, 0) == base
    assert _roi_cache_path(base, 1).name == "ep_roi2.seraw.json.gz"

    meta = RawOcrMeta(video_name="ep.mp4", video_duration_sec=60.0, frame_count=20, sample_step_sec=0.04,
                      detection_model="d", recognition_model="r", score_threshold=0.5, saved_at="2026-10-27")
    save_raw_ocr(_frames(), base, meta)
    roi = Roi(x=57, y=819, width=603, height=119)
    worker = RebuildFromCacheWorker([(base, SubtitleBuilderConfig(), roi, roi),
                                     (_roi_cache_path(base, 1), SubtitleBuilderConfig(), roi, roi)],
                                    tmp_path / "ep.mp4")
    got: list = []
    worker.finished.connect(got.append)
    worker.run()
    events = got[0].events
    assert [e.text for e in events] == [WITHOUT]
    box = events[0].bounding_box
    assert box is None or box[0] >= roi.x                               # toạ độ video, không còn tính trong ROI


def test_rebuild_worker_honours_cancel(tmp_path) -> None:
    """Rà độc lập: nút Huỷ bật trong lúc "Dựng lại từ OCR" nhưng worker không dừng."""
    from subtitles_extractor.presentation.view_models.extract_page_view_model import RebuildFromCacheWorker

    missing = tmp_path / "none.json"
    worker = RebuildFromCacheWorker([(missing, None, None, None)], tmp_path / "v.mp4")
    got, failed = [], []
    worker.finished.connect(got.append)
    worker.failed.connect(failed.append)
    worker.request_cancel()
    worker.run()
    assert failed == [] and len(got) == 1 and got[0].events == []
