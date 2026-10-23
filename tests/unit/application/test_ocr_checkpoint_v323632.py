"""[v3.23.632] OCR có điểm dừng: bị ngắt giữa chừng thì chạy TIẾP, kết quả y hệt chạy một mạch.

Nhật ký thật: một lượt OCR 2 giờ 24 phút chỉ lưu ở cuối → sập ở phút 140 là mất sạch. Bộ giả
dưới đây mô phỏng đúng hợp đồng của bộ lấy mẫu thật: ``skip_intro_sec`` tua tới mốc đó,
``frame_index`` đếm lại từ 0, mốc thời gian là TUYỆT ĐỐI.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from subtitles_extractor.application.dtos.extract_subtitles_dto import (
    ExtractSubtitlesRequest,
    SubtitleBuilderConfig,
)
from subtitles_extractor.application.services.subtitle_builder import SubtitleBuilder
from subtitles_extractor.application.use_cases import extract_subtitles as use_case_module
from subtitles_extractor.application.use_cases.extract_subtitles import (
    ExtractSubtitlesUseCase,
)
from subtitles_extractor.domain.entities.ocr_frame_result import OcrFrameResult, OcrTextBox
from subtitles_extractor.domain.entities.video_metadata import VideoMetadata
from subtitles_extractor.domain.exceptions import SubtitlesExtractorError
from subtitles_extractor.domain.ports.frame_sampler_port import (
    FrameSamplingConfig,
    SampledFrame,
)
from subtitles_extractor.domain.ports.ocr_engine_port import OcrEngineConfig
from subtitles_extractor.domain.value_objects.confidence import Confidence
from subtitles_extractor.domain.value_objects.roi import Roi
from subtitles_extractor.infrastructure.serializers.ocr_checkpoint import (
    OcrCheckpoint,
    checkpoint_path_for,
    run_fingerprint,
)
from subtitles_extractor.infrastructure.serializers.raw_ocr_serializer import load_raw_ocr

STEP = 0.1
DURATION = 10.0


def _text_at(ts: float) -> str:
    return f"câu {int(ts // 2)}"       # đổi câu mỗi 2 s → bản sao khung trùng vẫn đúng chữ


class _Sampler:
    """Tua tới ``skip_intro_sec``, đánh số frame từ 0, mỗi khung thứ 3 là khung trùng."""

    def __init__(self) -> None:
        self.starts: list[float] = []

    def iter_frames(self, *, metadata, roi, config: FrameSamplingConfig) -> Iterator[SampledFrame]:
        self.starts.append(config.skip_intro_sec)
        # Khung video thật nằm trên lưới cố định 0,1 s: lấy khung đầu tiên >= mốc tua.
        ts, idx = math.ceil((config.skip_intro_sec - 1e-4) / 0.1) * 0.1, 0
        while ts < DURATION - 1e-9:
            # [v3.23.644] Trùng theo MỐC THỜI GIAN (không theo chỉ số đếm lại sau mỗi lần tua) — bộ lấy mẫu
            # thật quyết định theo nội dung ảnh nên cũng không phụ thuộc chỗ bắt đầu lượt chạy.
            dup = round(ts / 0.1) % 3 == 2
            yield SampledFrame(
                frame_index=idx, timestamp_sec=round(ts, 6),
                image_rgb=np.empty(0, np.uint8) if dup else np.zeros((20, 80, 3), np.uint8),
                is_duplicate=dup,
            )
            ts += config.sample_step_sec
            idx += 1


class _Engine:
    is_initialized = True

    def __init__(self, crash_after: int | None = None) -> None:
        self.seen: list[float] = []
        self.crash_after = crash_after

    def infer_batch(self, images_rgb, frame_indices, timestamps_sec):
        if self.crash_after is not None and len(self.seen) >= self.crash_after:
            raise RuntimeError("máy sập")
        self.seen.extend(timestamps_sec)
        return [
            OcrFrameResult(
                frame_index=i, timestamp_sec=t,
                text_boxes=[OcrTextBox(
                    text=_text_at(t), confidence=Confidence(0.9),
                    polygon=[(0, 10), (100, 10), (100, 40), (0, 40)],
                )],
            )
            for i, t in zip(frame_indices, timestamps_sec, strict=True)
        ]


class _Reporter:
    def __init__(self, cancel_after_checks: int | None = None) -> None:
        self.messages: list[str] = []
        self._cancel_after = cancel_after_checks
        self._checks = 0

    def report(self, current: int, total: int, message: str) -> None:
        self.messages.append(message)

    def is_cancelled(self) -> bool:
        self._checks += 1
        return self._cancel_after is not None and self._checks > self._cancel_after


@pytest.fixture(autouse=True)
def _write_every_iteration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(use_case_module, "_CHECKPOINT_INTERVAL_SEC", 0.0)


def _metadata(tmp_path: Path) -> VideoMetadata:
    video = tmp_path / "phim.mp4"
    if not video.exists():                 # giữ nguyên mtime giữa các lượt (vân tay dùng mtime)
        video.write_bytes(b"0" * 64)
    return VideoMetadata(path=video, width=720, height=1280, fps=25.0,
                         total_frames=250, duration_sec=DURATION)


def _run(tmp_path: Path, engine: _Engine, sampler: _Sampler | None = None, *,
         reporter: _Reporter | None = None, roi: Roi | None = None,
         step: float = STEP, raw_name: str = "phim.seraw.json.gz"):
    metadata = _metadata(tmp_path)
    reader = type("Reader", (), {"read": lambda self, path: metadata})()
    use_case = ExtractSubtitlesUseCase(
        metadata_reader=reader, frame_sampler=sampler or _Sampler(), ocr_engine=engine,
        builder=SubtitleBuilder(SubtitleBuilderConfig()), exporters={},
    )
    request = ExtractSubtitlesRequest(
        video_path=metadata.path, output_path=tmp_path / "out.srt", roi=roi,
        sampling=FrameSamplingConfig(sample_step_sec=step),
        ocr=OcrEngineConfig(batch_size=4), skip_export=True,
        raw_ocr_output_path=tmp_path / raw_name,
    )
    return use_case.execute(request, reporter), request


def _frames(tmp_path: Path, raw_name: str = "phim.seraw.json.gz"):
    frames, _ = load_raw_ocr(tmp_path / raw_name)
    return [(f.frame_index, f.timestamp_sec, [b.text for b in f.text_boxes]) for f in frames]


def test_uninterrupted_run_leaves_no_checkpoint(tmp_path: Path) -> None:
    _run(tmp_path, _Engine())
    assert not list(tmp_path.glob("*.ckpt"))
    assert len(_frames(tmp_path)) == 100


def test_crash_then_resume_equals_single_run(tmp_path: Path) -> None:
    reference_dir = tmp_path / "ref"
    reference_dir.mkdir()
    _run(reference_dir, _Engine())
    reference = _frames(reference_dir)

    crashed = _Engine(crash_after=40)
    with pytest.raises(SubtitlesExtractorError):
        _run(tmp_path, crashed)
    assert checkpoint_path_for(tmp_path / "phim.seraw.json.gz").is_file()
    assert not (tmp_path / "phim.seraw.json.gz").exists()

    sampler, resumed = _Sampler(), _Engine()
    response, _ = _run(tmp_path, resumed, sampler)
    assert _frames(tmp_path) == reference                    # y hệt: chỉ số, mốc, chữ
    assert response.frames_processed == 100
    assert sampler.starts[0] > 3.0                           # tua tới sau phần đã làm
    assert min(resumed.seen) > max(crashed.seen)             # không OCR lại khung nào đã xong
    assert len(resumed.seen) < 100 - len(crashed.seen) + 8   # không OCR lại phần đã xong
    assert not list(tmp_path.glob("*.ckpt"))                 # xong thì dọn


def test_resume_announces_progress(tmp_path: Path) -> None:
    with pytest.raises(SubtitlesExtractorError):
        _run(tmp_path, _Engine(crash_after=40))
    reporter = _Reporter()
    _run(tmp_path, _Engine(), reporter=reporter)
    assert any("Tiếp tục OCR" in message for message in reporter.messages)


def test_cancel_keeps_checkpoint_and_resume_finishes(tmp_path: Path) -> None:
    reference_dir = tmp_path / "ref"
    reference_dir.mkdir()
    _run(reference_dir, _Engine())

    first = _Engine()
    _run(tmp_path, first, reporter=_Reporter(cancel_after_checks=45))
    assert checkpoint_path_for(tmp_path / "phim.seraw.json.gz").is_file()
    assert not (tmp_path / "phim.seraw.json.gz").exists()    # bị huỷ: không có cache dở

    second = _Engine()
    _run(tmp_path, second)
    assert _frames(tmp_path) == _frames(reference_dir)
    assert len(second.seen) < 100 - len(first.seen) + 8


@pytest.mark.parametrize("change", ["roi", "step"])
def test_different_run_does_not_reuse_checkpoint(tmp_path: Path, change: str) -> None:
    with pytest.raises(SubtitlesExtractorError):
        _run(tmp_path, _Engine(crash_after=40))
    engine = _Engine()
    kwargs = {"roi": Roi(x=0, y=0, width=100, height=50)} if change == "roi" else {"step": 0.2}
    _run(tmp_path, engine, **kwargs)
    # chạy từ đầu: thấy mốc 0.0 và không tua
    assert engine.seen[0] == 0.0
    assert not list(tmp_path.glob("*.ckpt"))


def test_torn_last_line_is_dropped_and_rest_reused(tmp_path: Path) -> None:
    with pytest.raises(SubtitlesExtractorError):
        _run(tmp_path, _Engine(crash_after=40))
    path = checkpoint_path_for(tmp_path / "phim.seraw.json.gz")
    with path.open("ab") as handle:
        handle.write(b'{"fi":999,"ts":9')                    # mất điện giữa lúc ghi
    reference_dir = tmp_path / "ref"
    reference_dir.mkdir()
    _run(reference_dir, _Engine())
    _run(tmp_path, _Engine())
    assert _frames(tmp_path) == _frames(reference_dir)


def test_no_raw_path_means_no_checkpoint(tmp_path: Path) -> None:
    metadata = _metadata(tmp_path)
    reader = type("Reader", (), {"read": lambda self, path: metadata})()
    use_case = ExtractSubtitlesUseCase(
        metadata_reader=reader, frame_sampler=_Sampler(), ocr_engine=_Engine(),
        builder=SubtitleBuilder(SubtitleBuilderConfig()), exporters={},
    )
    use_case.execute(ExtractSubtitlesRequest(
        video_path=metadata.path, output_path=tmp_path / "o.srt",
        sampling=FrameSamplingConfig(sample_step_sec=STEP), skip_export=True,
    ))
    assert not list(tmp_path.glob("*.ckpt"))


class TestCheckpointFile:
    def _frame(self, i: int) -> OcrFrameResult:
        return OcrFrameResult(frame_index=i, timestamp_sec=i * 0.1, text_boxes=[])

    def test_round_trip_and_append(self, tmp_path: Path) -> None:
        path = tmp_path / "a.ckpt"
        first = OcrCheckpoint(path, "fp")
        first.append([self._frame(0), self._frame(1)])
        first.append([self._frame(2)])
        assert first.written == 3
        again = OcrCheckpoint(path, "fp")
        assert [f.frame_index for f in again.resume()] == [0, 1, 2]
        again.append([self._frame(3)])
        assert [f.frame_index for f in OcrCheckpoint(path, "fp").resume()] == [0, 1, 2, 3]

    def test_other_fingerprint_discards(self, tmp_path: Path) -> None:
        path = tmp_path / "a.ckpt"
        OcrCheckpoint(path, "one").append([self._frame(0)])
        assert OcrCheckpoint(path, "two").resume() == []
        assert not path.exists()

    def test_garbage_file_discarded(self, tmp_path: Path) -> None:
        path = tmp_path / "a.ckpt"
        path.write_bytes(b"\xff\xfe not json")
        assert OcrCheckpoint(path, "fp").resume() == []
        assert not path.exists()

    def test_corrupt_middle_line_keeps_prefix(self, tmp_path: Path) -> None:
        path = tmp_path / "a.ckpt"
        cp = OcrCheckpoint(path, "fp")
        cp.append([self._frame(0), self._frame(1)])
        with path.open("ab") as handle:
            handle.write(b"khong phai json\n")
        cp.append([self._frame(5)])                          # sau dòng hỏng: không tin
        assert [f.frame_index for f in OcrCheckpoint(path, "fp").resume()] == [0, 1]

    def test_fingerprint_ignores_speed_only_settings_but_not_models(self, tmp_path: Path) -> None:
        metadata = _metadata(tmp_path)
        base = ExtractSubtitlesRequest(
            video_path=metadata.path, output_path=tmp_path / "o.srt",
            ocr=OcrEngineConfig(batch_size=4, parallel_workers=2),
        )
        faster = ExtractSubtitlesRequest(
            video_path=metadata.path, output_path=tmp_path / "o.srt",
            ocr=OcrEngineConfig(batch_size=64, parallel_workers=8),
        )
        other_model = ExtractSubtitlesRequest(
            video_path=metadata.path, output_path=tmp_path / "o.srt",
            ocr=OcrEngineConfig(batch_size=4, det_thresh=0.5),
        )
        assert run_fingerprint(base, metadata) == run_fingerprint(faster, metadata)
        assert run_fingerprint(base, metadata) != run_fingerprint(other_model, metadata)

    def test_fingerprint_changes_when_video_file_changes(self, tmp_path: Path) -> None:
        metadata = _metadata(tmp_path)
        request = ExtractSubtitlesRequest(video_path=metadata.path, output_path=tmp_path / "o.srt")
        before = run_fingerprint(request, metadata)
        metadata.path.write_bytes(b"1" * 128)
        assert run_fingerprint(request, metadata) != before


@pytest.fixture(scope="module")
def bframe_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """12 s HEVC 25 fps, GOP 50, có B-frame: khung khoá có dts <= mốc nhưng pts > mốc."""
    import shutil
    import subprocess

    if shutil.which("ffmpeg") is None:
        pytest.skip("cần ffmpeg")
    path = tmp_path_factory.mktemp("bframe") / "bframe.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=25",
         "-t", "12", "-c:v", "libx265", "-g", "50", "-bf", "3", "-x265-params", "log-level=error",
         str(path)],
        check=True,
    )
    return path


@pytest.mark.parametrize("step", [0.04, 0.1, 0.25])
def test_pyav_sampler_resume_point_matches_single_pass(bframe_video: Path, step: float) -> None:
    """Điểm tua của lượt chạy tiếp ra ĐÚNG các khung mà lượt một mạch sẽ lấy tiếp theo.

    Lỗi cũ: PyAV seek tới đúng mốc → với B-frame rơi vào khung khoá kế tiếp, mất cả quãng
    giữa mốc và khung khoá (skip_intro=5.85 s cho khung đầu 6.00 s thay vì 5.88 s).
    """
    import dataclasses

    from subtitles_extractor.domain.ports.frame_sampler_port import SAMPLE_TIME_TOLERANCE_SEC
    from subtitles_extractor.infrastructure.video.decoders.pyav_frame_sampler import (
        PyAvFrameSampler,
    )

    metadata = VideoMetadata(path=bframe_video, width=160, height=120, fps=25.0,
                             total_frames=300, duration_sec=12.0)
    config = FrameSamplingConfig(sample_step_sec=step)
    full = [f.timestamp_sec for f in PyAvFrameSampler().iter_frames(
        metadata=metadata, roi=None, config=config)]
    for cut in (len(full) // 3, len(full) // 2, 2 * len(full) // 3):
        resumed_config = dataclasses.replace(
            config, skip_intro_sec=full[cut - 1] + step - SAMPLE_TIME_TOLERANCE_SEC,
        )
        rest = [f.timestamp_sec for f in PyAvFrameSampler().iter_frames(
            metadata=metadata, roi=None, config=resumed_config)]
        assert rest == full[cut:]
