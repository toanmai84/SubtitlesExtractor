"""[v3.23.728] Trang Gỡ lỗi OCR mở được bộ "xuất dữ liệu thô" mới (``.seraw.json.gz`` + video vùng ROI); use case gọi
bộ xuất video đúng lúc, chạy GPU khi lượt OCR chạy GPU, và lỗi xuất video không phá lượt trích xuất."""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from subtitles_extractor.domain.entities.ocr_frame_result import OcrFrameResult, OcrTextBox
from subtitles_extractor.domain.value_objects.confidence import Confidence
from subtitles_extractor.domain.value_objects.device_kind import DeviceKind
from subtitles_extractor.domain.value_objects.roi import Roi

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

FFMPEG = shutil.which("ffmpeg")


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


class _Exporter:
    def __init__(self, fail: bool = False) -> None:
        self.calls: list[tuple] = []
        self.fail = fail

    def export(self, video_path, roi, output_path, *, prefer_gpu, progress=None, quality=None):  # noqa: ANN001, ANN201
        self.calls.append((video_path, roi, output_path, prefer_gpu))
        if self.fail:
            raise RuntimeError("NVENC hết phiên")
        assert progress is not None and progress(0.5) is True
        return object()


def _use_case(exporter):  # noqa: ANN001, ANN202
    from subtitles_extractor.application.use_cases.extract_subtitles import ExtractSubtitlesUseCase

    use_case = ExtractSubtitlesUseCase.__new__(ExtractSubtitlesUseCase)
    use_case._roi_clip_exporter = exporter
    return use_case


class _Reporter:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def report(self, current: int, total: int, message: str = "") -> None:
        self.messages.append(message)

    def is_cancelled(self) -> bool:
        return False


@pytest.mark.parametrize(("device", "gpu"), [(DeviceKind.GPU, True), (DeviceKind.CPU, False)])
def test_use_case_exports_clip_next_to_raw_file(tmp_path: Path, device: DeviceKind, gpu: bool) -> None:
    exporter, reporter = _Exporter(), _Reporter()
    request = SimpleNamespace(video_path=tmp_path / "phim.mkv", roi=Roi(0, 774, 720, 136), ocr=SimpleNamespace(device=device))
    metadata = SimpleNamespace(width=720, height=1280)
    _use_case(exporter)._export_roi_clip(request, metadata, tmp_path / "phim.seraw.json.gz", reporter)
    (video, roi, clip, prefer_gpu), = exporter.calls
    assert clip == tmp_path / "phim.roi.mp4" and prefer_gpu is gpu and roi == Roi(0, 774, 720, 136)
    assert any("vùng ROI" in m for m in reporter.messages)


def test_clip_failure_does_not_break_extraction(tmp_path: Path) -> None:
    request = SimpleNamespace(video_path=tmp_path / "phim.mkv", roi=None, ocr=SimpleNamespace(device=DeviceKind.GPU))
    _use_case(_Exporter(fail=True))._export_roi_clip(
        request, SimpleNamespace(width=720, height=1280), tmp_path / "phim.seraw.json.gz", _Reporter())
    _use_case(None)._export_roi_clip(
        request, SimpleNamespace(width=720, height=1280), tmp_path / "phim.seraw.json.gz", _Reporter())


@pytest.mark.skipif(FFMPEG is None, reason="cần ffmpeg")
def test_debug_page_reads_frames_from_roi_clip(app: QApplication, tmp_path: Path) -> None:
    from subtitles_extractor.infrastructure.serializers.raw_ocr_serializer import RawOcrMeta, save_raw_ocr
    from subtitles_extractor.infrastructure.video.roi_clip_exporter import FfmpegRoiClipExporter
    from subtitles_extractor.presentation.view_models.debug_page_view_model import DebugPageViewModel
    from tests.unit.infrastructure.test_roi_clip_export_v323728 import _make_video, _source_frames

    video = tmp_path / "phim.mkv"
    _make_video(video)
    roi = Roi(35, 773, 645, 148)
    raw = tmp_path / "phim.seraw.json.gz"
    source = _source_frames(video)
    frames = [
        OcrFrameResult(frame_index=i, timestamp_sec=ts, text_boxes=[OcrTextBox(
            text=str(i), confidence=Confidence(0.99), polygon=[(165, 27), (485, 27), (485, 107), (165, 107)])])
        for i, (ts, _) in enumerate(source)
    ]
    save_raw_ocr(frames, raw, RawOcrMeta(
        video_name=video.name, video_duration_sec=3.0, frame_count=len(frames), sample_step_sec=0.04,
        detection_model="d", recognition_model="r", score_threshold=0.35, saved_at="2026-11-16T00:00:00",
        roi_xywh=[roi.x, roi.y, roi.width, roi.height]))
    assert raw.read_bytes()[:2] == b"\x1f\x8b"                       # nén gzip như lượt trích xuất thường
    assert FfmpegRoiClipExporter(lambda: FFMPEG).export(video, roi, tmp_path / "phim.roi.mp4", prefer_gpu=False)

    vm = DebugPageViewModel(SimpleNamespace(translator=SimpleNamespace(translate=lambda k: k)))
    loaded, images, changed = [], [], []
    vm.file_loaded.connect(lambda ok, msg: loaded.append(ok))
    vm.live_images_ready.connect(lambda a, b: images.append((a, b)))
    vm.frame_changed.connect(lambda *args: changed.append(args))
    vm.load_raw_file(raw)
    assert loaded == [True] and images, "trang Gỡ lỗi phải hiện khung từ video vùng ROI"
    image = images[-1][0]
    assert (image.width(), image.height()) == (645, 148)              # đúng ROI → hộp chữ khớp từng điểm ảnh
    vm.jump_to_frame(40)
    assert len(images) == 2 and changed[-1][3] == "" and not images[-1][0].isNull()
