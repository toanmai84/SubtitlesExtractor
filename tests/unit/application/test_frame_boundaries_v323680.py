"""[v3.23.680] Mốc phụ đề đúng khung hình nhờ dải ranh giới (khung không OCR).

Lỗi gốc: bước lấy mẫu mặc định 0,12 s (25 fps → OCR 1/3 khung) làm mốc câu lệch tới 2 khung — đo trên đáp án
tập 1–5: mốc bắt đầu lệch trung bình 45 ms, 210/428 câu lệch > 1 khung. Bản này cho bộ giải mã ghi độ giống chữ của MỌI
khung bị bỏ qua rồi đặt lại mốc, không OCR thêm.
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
from subtitles_extractor.application.services.subtitle_pipeline.boundary_refinement import (
    refine_event_boundaries,
)
from subtitles_extractor.application.use_cases.extract_subtitles import ExtractSubtitlesUseCase
from subtitles_extractor.domain.entities.ocr_frame_result import OcrFrameResult, OcrTextBox
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.entities.video_metadata import VideoMetadata
from subtitles_extractor.domain.ports.frame_sampler_port import (
    FrameSamplingConfig,
    SampledFrame,
    is_sample_due,
)
from subtitles_extractor.domain.ports.ocr_engine_port import OcrEngineConfig
from subtitles_extractor.domain.value_objects.boundary_track import BoundaryProbe, BoundaryTrack
from subtitles_extractor.domain.value_objects.confidence import Confidence
from subtitles_extractor.domain.value_objects.roi import Roi
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval
from subtitles_extractor.infrastructure.serializers.raw_ocr_serializer import load_raw_ocr
from subtitles_extractor.infrastructure.video.decoders.boundary_probe import (
    BoundaryProber,
    expand_studio_luma,
    mask_similarity,
    probing_enabled,
    text_mask,
)

PERIOD = 0.04          # 25 fps
STEP = 0.12            # mặc định của ứng dụng: OCR 1/3 khung
H, W = 40, 120


def _luma(text_id: int | None) -> np.ndarray:
    """Khung độ sáng: ``None`` = không chữ; mỗi ``text_id`` là một khối chữ trắng ở chỗ khác nhau."""
    frame = np.full((H, W), 30, np.uint8)
    if text_id is not None:
        x = 4 + 12 * (text_id % 4)
        frame[8:32, x:x + 64] = 250
    return frame


def _event(start: float, end: float, text: str = "câu") -> SubtitleEvent:
    return SubtitleEvent(index=1, text=text, interval=TimeInterval(start, end))


def _track(schedule: dict[int, int | None], frames: int, step_frames: int = 3) -> BoundaryTrack:
    prober = BoundaryProber(None)
    for k in range(frames):
        luma = _luma(schedule.get(k))
        (prober.sampled if k % step_frames == 0 else prober.skipped)(luma, k * PERIOD)
    return prober.finish()


# ── Mặt nạ chữ ──
def test_text_mask_and_similarity() -> None:
    a, b = text_mask(_luma(0)), text_mask(_luma(0))
    assert a.shape == (H // 4, W // 4) and int(a.sum()) >= 60
    assert mask_similarity(a, b) == 1.0
    assert mask_similarity(text_mask(_luma(None)), text_mask(_luma(None))) == 1.0   # cùng "không chữ"
    assert mask_similarity(a, text_mask(_luma(None))) == 0.0
    assert mask_similarity(a, text_mask(_luma(2))) < 0.75                          # chữ khác chỗ
    assert text_mask(np.zeros((3, 3), np.uint8)).size == 0


def test_expand_studio_luma_matches_pyav_gray() -> None:
    """NVDEC lấy mặt phẳng Y dải hẹp; PyAV ``gray`` là dải đầy — hai đường phải cho cùng mặt nạ."""
    y = np.arange(256, dtype=np.uint8)
    out = expand_studio_luma(y)
    assert out.dtype == np.uint8
    assert out[16] == 0 and out[235] == 255 and out[0] == 0 and out[255] == 255
    expected = np.clip(np.round((y.astype(float) - 16) * 255 / 219), 0, 255)
    assert np.abs(out.astype(int) - expected.astype(int)).max() <= 1


def test_probing_enabled() -> None:
    roi = Roi(x=0, y=0, width=10, height=10)
    assert probing_enabled(roi, 0.12)
    assert not probing_enabled(None, 0.12)
    assert not probing_enabled(roi, 5.0)      # chế độ khung khoá


# ── Đặt lại mốc ──
def test_start_moves_back_and_end_forward_to_exact_frame() -> None:
    # Chữ hiện từ khung 7 (0,28 s) đến hết khung 49 (biến mất ở 2,00 s). Khung mẫu: 0,3,6,9,… → OCR thấy chữ từ khung 9
    # (0,36 s) tới khung 48 (1,92 s).
    track = _track({k: 0 for k in range(7, 50)}, 80)
    [event] = refine_event_boundaries([_event(0.36, 1.98)], track, STEP)
    assert event.start_sec == pytest.approx(0.28)
    assert event.end_sec == pytest.approx(2.00)


def test_step_inferred_from_track() -> None:
    track = _track({k: 0 for k in range(7, 50)}, 80)
    [event] = refine_event_boundaries([_event(0.36, 1.98)], track)
    assert event.start_sec == pytest.approx(0.28)


def test_end_never_passes_refined_start_of_next_similar_line() -> None:
    """Câu sau giống chữ câu trước ("天下无敌" → "什么天下无敌"): câu trước không được lấn sang."""
    schedule = {k: 0 for k in range(3, 20)}
    schedule.update({k: 0 for k in range(20, 40)})   # cùng chữ (mặt nạ giống) nhưng OCR là câu khác
    track = _track(schedule, 60)
    first, second = refine_event_boundaries(
        [_event(0.12, 0.80, "天下无敌"), _event(0.84, 1.62, "什么天下无敌")], track, STEP,
    )
    # Mốc câu sau được chỉnh TRƯỚC; câu trước bị chặn tại đó — không chồng nhau, không lùi quá cuối câu trước − 1 bước.
    assert 0.80 - STEP - 1e-9 <= second.start_sec <= 0.84 + 1e-9
    assert first.end_sec <= second.start_sec + 1e-9
    assert first.end_sec > first.start_sec


def test_grey_or_faint_text_is_left_alone() -> None:
    prober = BoundaryProber(None)
    for k in range(60):
        faint = np.full((H, W), 30, np.uint8)
        if 7 <= k < 50:
            faint[8:32, 4:68] = 200            # dưới ngưỡng chữ sáng
        (prober.sampled if k % 3 == 0 else prober.skipped)(faint, k * PERIOD)
    original = _event(0.36, 1.98)
    assert refine_event_boundaries([original], prober.finish(), STEP) == [original]


def test_missing_frames_stop_the_walk() -> None:
    """Khung hụt (giải mã lỗi / bộ đệm đầy) giữa hai mẫu: không suy mốc qua chỗ không có dữ liệu."""
    probes = (
        BoundaryProbe(0.04, 200, 1.0, 1.0),       # cách mẫu 0,36 s tới 0,32 s — không liền khung
    )
    track = BoundaryTrack((0.0, 0.36), (0, 200), probes, PERIOD)
    [event] = refine_event_boundaries([_event(0.36, 0.5)], track, 0.36)
    assert event.start_sec == pytest.approx(0.36)


def test_no_track_or_dense_step_is_noop() -> None:
    events = [_event(0.36, 1.98)]
    assert refine_event_boundaries(events, None, STEP) == events
    assert refine_event_boundaries(events, BoundaryTrack(), STEP) == events
    track = _track({k: 0 for k in range(7, 50)}, 80, step_frames=1)    # OCR mọi khung → không có gì để chỉnh
    assert refine_event_boundaries(events, track, PERIOD) == events


def test_track_round_trip_and_bad_data() -> None:
    track = _track({k: 0 for k in range(7, 50)}, 30)
    again = BoundaryTrack.from_dict(track.to_dict())
    assert again is not None
    assert again.sample_times == pytest.approx(track.sample_times)
    assert len(again.probes) == len(track.probes)
    assert again.frame_period_sec == pytest.approx(PERIOD)
    assert BoundaryTrack.from_dict(None) is None
    assert BoundaryTrack.from_dict({"st": [1], "sp": []}) is None
    assert BoundaryTrack.from_dict({"st": "x"}) is None


# ── Toàn trình: bộ lấy mẫu → use case → bộ dựng + tệp OCR thô ──
_VISIBLE = (0.28, 2.00)       # chữ thật hiện [0,28; 2,00)


class _ProbingSampler:
    """Bộ lấy mẫu giả theo hợp đồng thật: ghi mọi khung vào bộ dò, chỉ trả khung mẫu."""

    last_boundary_track: BoundaryTrack | None = None

    def iter_frames(self, *, metadata, roi, config: FrameSamplingConfig) -> Iterator[SampledFrame]:
        self.last_boundary_track = None
        prober = BoundaryProber(None)
        next_target, idx = 0.0, 0
        for k in range(100):
            ts = round(k * PERIOD, 6)
            luma = _luma(0 if _VISIBLE[0] - 1e-9 <= ts < _VISIBLE[1] - 1e-9 else None)
            if not is_sample_due(ts, next_target):
                prober.skipped(luma, ts)
                continue
            prober.sampled(luma, ts)
            next_target = ts + config.sample_step_sec
            yield SampledFrame(frame_index=idx, timestamp_sec=ts, image_rgb=np.zeros((20, 80, 3), np.uint8))
            idx += 1
        self.last_boundary_track = prober.finish()


class _Engine:
    is_initialized = True

    def infer_batch(self, images_rgb, frame_indices, timestamps_sec):
        return [
            OcrFrameResult(
                frame_index=i, timestamp_sec=t,
                text_boxes=[OcrTextBox(
                    text="天下无敌", confidence=Confidence(0.95),
                    polygon=[(0, 10), (100, 10), (100, 40), (0, 40)],
                )] if _VISIBLE[0] <= t < _VISIBLE[1] else [],
            )
            for i, t in zip(frame_indices, timestamps_sec, strict=True)
        ]


def _metadata(tmp_path: Path) -> VideoMetadata:
    video = tmp_path / "phim.mp4"
    video.write_bytes(b"0" * 64)
    return VideoMetadata(path=video, width=720, height=1280, fps=25.0, total_frames=100, duration_sec=4.0)


def test_extraction_places_boundaries_on_exact_frames(tmp_path: Path) -> None:
    metadata = _metadata(tmp_path)
    reader = type("Reader", (), {"read": lambda self, path: metadata})()
    use_case = ExtractSubtitlesUseCase(
        metadata_reader=reader, frame_sampler=_ProbingSampler(), ocr_engine=_Engine(),
        builder=SubtitleBuilder(SubtitleBuilderConfig(sample_step_sec=STEP)), exporters={},
    )
    raw = tmp_path / "phim.seraw.json.gz"
    request = ExtractSubtitlesRequest(
        video_path=metadata.path, output_path=tmp_path / "out.srt",
        sampling=FrameSamplingConfig(sample_step_sec=STEP), ocr=OcrEngineConfig(batch_size=4),
        skip_export=True, raw_ocr_output_path=raw,
    )
    response = use_case.execute(request)
    [event] = response.events
    # Trước bản này: bắt đầu ở khung mẫu 0,36 s (lệch 2 khung).
    assert event.start_sec == pytest.approx(_VISIBLE[0])
    assert event.end_sec == pytest.approx(_VISIBLE[1])

    # Dựng lại từ tệp OCR thô (trang Trích xuất "dựng lại", trang Gỡ lỗi) cho cùng mốc.
    frames, meta = load_raw_ocr(raw)
    track = BoundaryTrack.from_dict(meta.boundary)
    assert track is not None and not track.is_empty
    [rebuilt] = SubtitleBuilder(SubtitleBuilderConfig(sample_step_sec=STEP)).build(frames, boundary_track=track)
    assert (rebuilt.start_sec, rebuilt.end_sec) == pytest.approx((event.start_sec, event.end_sec))
    # Không có dải (tệp cũ) → như cũ.
    [legacy] = SubtitleBuilder(SubtitleBuilderConfig(sample_step_sec=STEP)).build(frames)
    assert legacy.start_sec == pytest.approx(0.36)


# ── Bộ lấy mẫu PyAV thật trên video tổng hợp ──
av = pytest.importorskip("av")


@pytest.fixture(scope="module")
def flash_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("video") / "flash.mp4"
    container = av.open(str(path), "w")
    stream = container.add_stream("mpeg4", rate=25)
    stream.width, stream.height, stream.pix_fmt = 128, 64, "yuv420p"
    stream.bit_rate = 4_000_000
    for k in range(60):
        image = np.full((64, 128, 3), 20, np.uint8)
        if 7 <= k < 40:
            image[20:44, 16:112] = 255
        for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return path


def test_pyav_sampler_records_every_frame(flash_video: Path) -> None:
    from subtitles_extractor.infrastructure.video.decoders.pyav_frame_sampler import PyAvFrameSampler

    sampler = PyAvFrameSampler()
    metadata = VideoMetadata(path=flash_video, width=128, height=64, fps=25.0, total_frames=60, duration_sec=2.4)
    roi = Roi(x=0, y=0, width=128, height=64)
    list(sampler.iter_frames(metadata=metadata, roi=roi, config=FrameSamplingConfig(sample_step_sec=STEP)))
    track = sampler.last_boundary_track
    assert track is not None
    assert track.frame_period_sec == pytest.approx(PERIOD, abs=1e-3)
    assert len(track.sample_times) + len(track.probes) == 60
    texty = sorted(p.timestamp_sec for p in track.probes if p.text_pixels >= 60)
    assert texty and math.isclose(texty[0], 0.28, abs_tol=1e-3)     # khung 7 (bỏ qua) được ghi có chữ

    [event] = refine_event_boundaries([_event(0.36, 1.60)], track, STEP)
    assert event.start_sec == pytest.approx(0.28, abs=1e-3)
    assert event.end_sec == pytest.approx(1.60, abs=1e-3)

    # Không ROI → không ghi (mặt nạ chỉ có nghĩa trên vùng phụ đề).
    list(sampler.iter_frames(metadata=metadata, roi=None, config=FrameSamplingConfig(sample_step_sec=STEP)))
    assert sampler.last_boundary_track is None


# ── Đường NVDEC: độ sáng ROI lấy thẳng từ mặt phẳng Y trên GPU (v3.23.723: chạy đúng bộ xử lý khung với NumPy) ──
def test_nvdec_luma_sink_crops_roi_and_expands_range(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from subtitles_extractor.infrastructure.video.decoders import pynvvideocodec_frame_sampler as module

    h, w, pitch = 64, 96, 128                                  # pitch > bề rộng như NVDEC thật
    nv12 = np.full((h + h // 2, pitch), 128, np.uint8)
    nv12[:h, :w] = 16                                          # đen dải hẹp
    nv12[40:56, 20:60] = 235                                   # chữ trắng dải hẹp
    y_plane, uv_plane = module._split_nv12(nv12, h)
    roi = Roi(x=8, y=32, width=80, height=32)

    def processor(sink: list[np.ndarray]):
        prober = SimpleNamespace(sampled=lambda luma, ts: sink.append(luma), skipped=lambda luma, ts: None)
        return module._FrameProcessor(
            config=FrameSamplingConfig(sample_step_sec=0.04), roi=roi, video_width=w, video_height=h, matrix="bt601",
            full_range=False, prober=prober, xp=np, start_sec=0.0,
        )

    sink: list[np.ndarray] = []
    frame = processor(sink).handle_nv12(y_plane[:h, :w], uv_plane[:, :w], 0.0)
    assert frame is not None and frame.image_rgb.shape == (32, 80, 3)        # khung OCR vẫn ra như cũ
    [luma] = sink
    assert luma.shape == (32, 80)
    assert luma.max() == 255 and luma.min() == 0               # 16–235 → 0–255 như PyAV ``gray``
    assert int(text_mask(luma).sum()) == (16 // 4) * (40 // 4)
    assert int(frame.image_rgb.max()) == 255 and int(frame.image_rgb.min()) == 0   # RGB cũng giãn dải như PyAV

    # Lỗi đầu dò → chỉ bỏ đầu dò, KHÔNG làm mất khung OCR.
    monkeypatch.setattr(module, "expand_studio_luma", lambda _y: (_ for _ in ()).throw(TypeError("lạ")))
    sink2: list[np.ndarray] = []
    assert processor(sink2).handle_nv12(y_plane[:h, :w], uv_plane[:, :w], 0.0) is not None
    assert sink2 == []
