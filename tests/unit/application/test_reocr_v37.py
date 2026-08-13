"""Tests bảo vệ v3.7 Re-OCR fixes.

REOCR-LEAK:  iter_frames của NVDEC sampler phải giải phóng nv_decoder + CuPy
             pool + gc.collect trong finally. Trước đây chỉ đóng av_container →
             rò NVDEC session (giới hạn 2–8 trên GPU consumer) + VRAM creep qua
             nhiều lần Re-OCR → cộng dồn cạn GPU.

REOCR-CANCEL: Huỷ giữa chừng đa-range KHÔNG được áp dụng thay thế. replaced_uids
              chứa mọi uid được chọn nhưng new_events chỉ có range đã xong → xóa
              phụ đề thuộc range chưa quét mà không có bản thay thế (mất dữ liệu).
"""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

from subtitles_extractor.application.dtos.reocr_dto import ReOcrResponse, TimeRange
from subtitles_extractor.application.use_cases.reocr import ReOcrUseCase

_SAMPLER_PATH = (
    Path(__file__).resolve().parents[3]
    / "src/subtitles_extractor/infrastructure/video/decoders/pynvvideocodec_frame_sampler.py"
)
_VM_PATH = (
    Path(__file__).resolve().parents[3]
    / "src/subtitles_extractor/presentation/view_models/editor_page_view_model.py"
)


# ── REOCR-LEAK: soi mã nguồn iter_frames ─────────────────────────────────────

class TestNvdecLeakFix:
    def test_iter_frames_releases_decoder_and_gpu(self) -> None:
        src = _SAMPLER_PATH.read_text(encoding="utf-8")
        start = src.find("def iter_frames")
        end = src.find("\n    def ", start + 1)
        snippet = src[start:end]
        assert "nv_decoder = None" in snippet, "Phải bỏ tham chiếu nv_decoder để giải phóng NVDEC session."
        assert "gc.collect()" in snippet, "Phải gc.collect() để ép finalizer NVDEC chạy ngay."
        assert "free_all_blocks" in snippet, "Phải trả VRAM CuPy pool về driver."

    def test_cleanup_in_finally_block(self) -> None:
        src = _SAMPLER_PATH.read_text(encoding="utf-8")
        start = src.find("def iter_frames")
        end = src.find("\n    def ", start + 1)
        snippet = src[start:end]
        finally_pos = snippet.rfind("finally:")
        assert finally_pos != -1
        finally_block = snippet[finally_pos:]
        # Cleanup GPU phải nằm trong finally để chạy cả khi huỷ/lỗi giữa chừng.
        assert "nv_decoder = None" in finally_block
        assert "gc.collect()" in finally_block
        assert "av_container.close()" in finally_block


# ── REOCR-CANCEL: hành vi use case + guard view model ────────────────────────

class _FakeMetadata:
    duration_sec = 100.0


class _FakeMetadataUseCase:
    def execute(self, _video_path):
        return _FakeMetadata()


class _FakeExtractResponse:
    def __init__(self, events):
        self.events = events
        self.frames_processed = len(events)


class _FakeExtractUseCase:
    def __init__(self):
        self.call_count = 0

    def execute(self, request, progress):
        self.call_count += 1
        return _FakeExtractResponse([])


class _CancellingReporter:
    """Báo huỷ ngay từ lần kiểm tra đầu tiên."""

    def __init__(self):
        self.reports = []

    def report(self, current, total, message):
        self.reports.append((current, total, message))

    def is_cancelled(self):
        return True


class _NeverCancelReporter:
    def report(self, current, total, message):
        pass

    def is_cancelled(self):
        return False


def _make_request(num_ranges: int):
    ranges = [TimeRange(start_sec=float(i * 5), end_sec=float(i * 5 + 3)) for i in range(num_ranges)]
    sampling = SimpleNamespace(
        sample_step_sec=0.2,
        phash_distance_threshold=5,
        pixel_diff_threshold=10,
        apply_median_blend=False,
        median_blend_frames=3,
        vram_upscale_small_text=False,
        vram_upscale_target_height_px=48,
        vram_add_border=False,
        vram_border_thickness_px=8,
        vram_sharpen=False,
        vram_contrast_factor=1.0,
    )
    return SimpleNamespace(
        video_path=Path("/tmp/fake.mp4"),
        time_ranges=ranges,
        merge_window_sec=0.0,
        replace_uids=["uid-a", "uid-b", "uid-c"],
        roi=None,
        sampling=sampling,
        ocr=SimpleNamespace(),
        builder=SimpleNamespace(),
        auto_tune_batch=True,
        save_debug_frames=False,
        debug_frames_dir=None,
    )


class TestReOcrCancelDataLoss:
    def test_cancelled_sets_flag_and_skips_extract(self) -> None:
        extract = _FakeExtractUseCase()
        use_case = ReOcrUseCase(extract_use_case=extract, load_metadata_use_case=_FakeMetadataUseCase())
        response = use_case.execute(_make_request(2), progress=_CancellingReporter())
        assert response.was_cancelled is True
        assert extract.call_count == 0, "Huỷ ngay đầu → không được chạy extract range nào."
        # replaced_uids vẫn copy đầy đủ — nhưng caller phải bỏ qua khi was_cancelled.
        assert response.replaced_uids == ["uid-a", "uid-b", "uid-c"]

    def test_not_cancelled_flag_false(self) -> None:
        extract = _FakeExtractUseCase()
        use_case = ReOcrUseCase(extract_use_case=extract, load_metadata_use_case=_FakeMetadataUseCase())
        response = use_case.execute(_make_request(1), progress=_NeverCancelReporter())
        assert response.was_cancelled is False
        assert extract.call_count == 1

    def test_response_has_was_cancelled_field(self) -> None:
        resp = ReOcrResponse(new_events=[], replaced_uids=[], elapsed_seconds=0.0, frames_processed=0)
        assert hasattr(resp, "was_cancelled")
        assert resp.was_cancelled is False  # mặc định an toàn

    def test_view_model_guards_was_cancelled(self) -> None:
        """_on_reocr_finished phải bỏ qua thay thế khi was_cancelled=True."""
        src = _VM_PATH.read_text(encoding="utf-8")
        start = src.find("def _on_reocr_finished")
        end = src.find("\n    def ", start + 1)
        snippet = src[start:end]
        guard_pos = snippet.find("was_cancelled")
        replace_pos = snippet.find("replace_events_by_uid")
        assert guard_pos != -1, "Phải kiểm tra was_cancelled trong _on_reocr_finished."
        assert guard_pos < replace_pos, "Guard was_cancelled phải đứng TRƯỚC khi gọi replace_events_by_uid."
        assert "return" in snippet[guard_pos:replace_pos], "Phải return sớm khi was_cancelled."
