"""[v3.23.723] Nhánh GPU (NVDEC) cho RA ĐÚNG các khung của nhánh CPU (PyAV) — chạy thật vòng giải mã NVDEC với một bộ
giải mã giả (giải mã gói bằng PyAV, trả khung NV12 qua DLPack) và NumPy thay CuPy.

Lỗi gốc (đo trên khung thật phim 86 tập, chưa từng lộ ra vì máy chủ dự án thiếu CuPy nên luôn rơi về PyAV):

* nhân NV12→RGB dùng công thức dải đầy BT.601 cho video dải TV → lệch PyAV TB 7–13 mức xám, chữ trắng không bao giờ đạt
  "lõi" ≥ 240 (7 404 so với 50 391 điểm ảnh) → bộ lọc chữ trắng chọn khung OCR khác hẳn nhánh CPU;
* bộ khử trùng xét ảnh ĐÃ tiền xử lý; làm nét / tương phản là nhân riêng khác bản CPU; CLAHE không có bản GPU;
* lỗi GPU bị nuốt (khung bị bỏ im lặng); bước thưa vẫn giải mã mọi khung; 10-bit/HDR bỏ tiền xử lý và bỏ khung trùng.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import av  # noqa: E402

from subtitles_extractor.domain.entities.video_metadata import VideoMetadata  # noqa: E402
from subtitles_extractor.domain.exceptions import VideoDecodeError  # noqa: E402
from subtitles_extractor.domain.ports.frame_sampler_port import FrameSamplingConfig, SampledFrame  # noqa: E402
from subtitles_extractor.domain.value_objects.roi import Roi  # noqa: E402
from subtitles_extractor.infrastructure.ocr.preprocessing import gpu_image_filters as gpu  # noqa: E402
from subtitles_extractor.infrastructure.ocr.preprocessing.image_filters import (  # noqa: E402
    apply_cpu_preprocessing_pipeline,
)
from subtitles_extractor.infrastructure.video.decoders import pynvvideocodec_frame_sampler as nvdec  # noqa: E402
from subtitles_extractor.infrastructure.video.decoders.fallback_frame_sampler import FallbackFrameSampler  # noqa: E402
from subtitles_extractor.infrastructure.video.decoders.pyav_frame_sampler import PyAvFrameSampler  # noqa: E402

W, H, FPS, COUNT = 320, 192, 25, 100
ROI = Roi(x=0, y=128, width=320, height=56)


def _make_video(path: Path) -> Path:
    """Video thử: nền chuyển động + hai câu "phụ đề" chữ trắng viền tối (hiện / tắt), mã hoá mpeg4 (có ở mọi bản PyAV)."""
    container = av.open(str(path), "w")
    stream = container.add_stream("mpeg4", rate=FPS)
    stream.width, stream.height, stream.pix_fmt = W, H, "yuv420p"
    stream.codec_context.gop_size = 12
    stream.codec_context.bit_rate = 4_000_000
    yy, xx = np.mgrid[0:H, 0:W]
    for index in range(COUNT):
        image = np.stack([(xx + index * 3) % 256, (yy * 2 + index) % 256, ((xx + yy) // 2) % 256], -1).astype(np.uint8)
        image = (image * 0.5).astype(np.uint8)
        if 20 <= index < 55 or 62 <= index < 90:
            x0 = 60 if index < 55 else 100
            image[140:176, x0 - 2:x0 + 162] = 20
            image[144:172, x0:x0 + 160:6] = 250
            image[144:172, x0 + 2:x0 + 160:6] = 250
        frame = av.VideoFrame.from_ndarray(image, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return path


def _metadata(path: Path) -> VideoMetadata:
    return VideoMetadata(path=path, width=W, height=H, fps=float(FPS), total_frames=COUNT, duration_sec=COUNT / FPS)


class _DeviceFrame:
    """Khung "trên GPU" giả: mảng NV12 xuất qua DLPack như PyNvVideoCodec."""

    def __init__(self, nv12: np.ndarray, pts: int) -> None:
        self._array = nv12
        self._pts = pts

    def getPTS(self) -> int:  # noqa: N802 — tên API của PyNvVideoCodec
        return self._pts

    def __dlpack__(self, *args, **kwargs):
        if self._array is None:
            raise RuntimeError("cudaErrorLaunchFailure (giả lập)")
        return self._array.__dlpack__(*args, **kwargs)

    def __dlpack_device__(self):
        return self._array.__dlpack_device__()


class _FakeNvDecoder:
    """Bộ giải mã NVDEC giả: giải mã gói bằng PyAV, trả NV12 (pitch rộng hơn khung như NVDEC thật)."""

    def __init__(self, stream, fail_at: float | None = None) -> None:
        self._codec = av.CodecContext.create(stream.codec_context.name, "r")
        self._codec.extradata = stream.codec_context.extradata
        self._time_base = stream.time_base
        self._fail_at = fail_at
        self.fed: list[bool] = []

    def Decode(self, packet):  # noqa: N802 — tên API của PyNvVideoCodec
        if isinstance(packet, SimpleNamespace):          # gói EOS
            return [self._to_device(f) for f in self._codec.decode(None)]
        raw, source = packet
        self.fed.append(bool(source.is_keyframe))
        frames = []
        # Giải mã thẳng gói của bộ tách luồng (``av.Packet(bytes)`` mất dữ liệu phụ của mpeg4 → giải mã sai vài khung P;
        # NVDEC thật tự phân tích dòng byte nên không gặp).
        for frame in self._codec.decode(source):
            device = self._to_device(frame)
            if self._fail_at is not None and float(frame.pts * self._time_base) >= self._fail_at:
                device._array = None                     # lỗi GPU giả lập khi đọc khung (vd NVRTC / CUDA)
            frames.append(device)
        return frames

    @staticmethod
    def _to_device(frame) -> _DeviceFrame:
        nv12 = frame.reformat(format="nv12").to_ndarray()
        padded = np.zeros((nv12.shape[0], nv12.shape[1] + 64), np.uint8)
        padded[:, : nv12.shape[1]] = nv12
        return _DeviceFrame(padded, frame.pts)


class _FakeNvdecSampler(nvdec.PyNvVideoCodecFrameSampler):
    """Chạy ĐÚNG ``_decode_loop`` của nhánh NVDEC, chỉ thay bộ giải mã phần cứng."""

    def __init__(self, fail_at: float | None = None) -> None:
        self.fail_at = fail_at
        self.decoder: _FakeNvDecoder | None = None

    def iter_frames(self, metadata, roi, config):
        container = av.open(str(metadata.path))
        stream = container.streams.video[0]
        self.decoder = _FakeNvDecoder(stream, self.fail_at)
        try:
            yield from self._decode_loop(
                av_container=container, av_stream=stream, nv_decoder=self.decoder,
                video_width=W, video_height=H, time_base=float(stream.time_base),
                is_16bit=False, is_hdr=False, needs_annexb_conversion=False, nal_length_size=4, annexb_header=b"",
                enum_eos=None, class_packet_data=SimpleNamespace, metadata=metadata, roi=roi, config=config,
            )
        finally:
            container.close()


@pytest.fixture()
def fake_gpu(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(nvdec, "cp", np, raising=False)
    monkeypatch.setattr(nvdec, "HAS_CUPY", True)
    monkeypatch.setattr(nvdec, "_create_packet_data", lambda raw, packet: ((raw, packet), None))
    monkeypatch.setattr(gpu, "cp", np, raising=False)
    monkeypatch.setattr(gpu, "HAS_CUPY", True)


@pytest.fixture(scope="module")
def video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _make_video(tmp_path_factory.mktemp("nvdec") / "clip.mp4")


def _summary(frames: list[SampledFrame]) -> list[tuple]:
    return [(round(f.timestamp_sec, 3), bool(f.image_rgb.size), f.is_duplicate, f.is_blank) for f in frames]


GATE = FrameSamplingConfig(sample_step_sec=0.04, white_text_gate=True, white_text_luma=200)
CLASSIC = FrameSamplingConfig(sample_step_sec=0.08)


@pytest.mark.parametrize("config", [GATE, CLASSIC], ids=["white_text_gate", "classic"])
def test_nvdec_branch_samples_the_same_frames_as_pyav(video: Path, fake_gpu, config: FrameSamplingConfig) -> None:
    cpu = list(PyAvFrameSampler().iter_frames(_metadata(video), ROI, config))
    gpu_frames = list(_FakeNvdecSampler().iter_frames(_metadata(video), ROI, config))
    assert _summary(gpu_frames) == _summary(cpu)                    # cùng mốc, cùng phán OCR / trùng / trống
    assert sum(bool(f.image_rgb.size) for f in cpu) >= 4
    for mine, ref in zip(gpu_frames, cpu, strict=True):
        if ref.image_rgb.size:
            assert np.array_equal(mine.image_rgb, ref.image_rgb)   # trùng khít (nhân cũ: lệch TB 7–13, tối đa ~20)


def test_old_full_range_kernel_was_far_from_pyav(video: Path) -> None:
    """Công thức nhân CUDA cũ (dải đầy BT.601, ``Y << 13``) trên cùng khung: lệch xa — chặn quay lại."""
    container = av.open(str(video))
    frame = next(f for i, f in enumerate(container.decode(video=0)) if i == 30)
    ref = frame.to_ndarray(format="rgb24").astype(int)
    nv12 = frame.reformat(format="nv12").to_ndarray()
    container.close()
    y, uv = nvdec._split_nv12(nv12, H)
    u = np.repeat(np.repeat(uv[:, 0::2].astype(int) - 128, 2, 0), 2, 1)
    v = np.repeat(np.repeat(uv[:, 1::2].astype(int) - 128, 2, 0), 2, 1)
    old_red = np.clip(((y.astype(int) << 13) + 11485 * v) >> 13, 0, 255)
    new = gpu.nv12_roi_to_rgb(y, uv, 0, 0, W, H, matrix="bt601", full_range=False, xp=np).astype(int)
    assert np.abs(old_red - ref[..., 0]).mean() > 4
    assert np.array_equal(new, ref)


@pytest.mark.parametrize(
    "colorspace, color_range, matrix, full",
    [(2, 1, "bt601", False), (1, 1, "bt709", False), (9, 1, "bt2020", False), (5, 2, "bt601", True),
     (1, 2, "bt709", True), (6, 1, "bt601", False)],
)
def test_color_conversion_follows_stream_tags_like_pyav(colorspace: int, color_range: int, matrix: str, full: bool) -> None:
    rng = np.random.default_rng(colorspace * 10 + color_range)
    yuv = rng.integers(0, 256, (H * 3 // 2, W), dtype=np.uint8)
    frame = av.VideoFrame.from_ndarray(yuv, format="yuv420p")
    frame.colorspace, frame.color_range = colorspace, color_range
    ref = frame.to_ndarray(format="rgb24").astype(int)
    assert gpu.color_matrix_for(colorspace) == matrix and gpu.is_full_range(color_range) is full
    nv12 = frame.reformat(format="nv12").to_ndarray()
    y, uv = nvdec._split_nv12(nv12, H)
    out = gpu.nv12_roi_to_rgb(y, uv, 3, 10, W - 5, H - 7, matrix=matrix, full_range=full, xp=np)
    assert np.array_equal(out, ref[10:H - 7, 3:W - 5])           # trùng khít swscale (ROI lệch chẵn/lẻ)


def test_nvdec_preprocesses_only_ocr_frames_on_vram_like_the_cpu_branch(video: Path, fake_gpu) -> None:
    """Phóng ×2 + CLAHE + làm nét + viền trên "VRAM" = nhánh CPU (PyAV + pipeline CPU) tới vài mức xám; bộ khử trùng
    xét ảnh GỐC nên mốc / phán quyết y như không tiền xử lý."""
    config = FrameSamplingConfig(
        sample_step_sec=0.04, white_text_gate=True, white_text_luma=200,
        vram_upscale_small_text=True, vram_upscale_target_height_px=112, vram_sharpen=True,
        vram_add_border=True, vram_border_thickness_px=8,
        vram_apply_clahe=True, vram_clahe_clip_limit=3.0, vram_clahe_tile_size=4,
    )
    cpu = list(PyAvFrameSampler().iter_frames(_metadata(video), ROI, config))
    gpu_frames = list(_FakeNvdecSampler().iter_frames(_metadata(video), ROI, config))
    assert _summary(gpu_frames) == _summary(cpu)
    for mine, ref in zip(gpu_frames, cpu, strict=True):
        if not ref.image_rgb.size:
            continue
        expected = apply_cpu_preprocessing_pipeline(
            ref.image_rgb, upscale_small_text=True, upscale_target_height_px=112, apply_clahe_flag=True,
            clahe_clip_limit=3.0, clahe_tile_size=4, apply_sharpen_flag=True, apply_contrast_boost_flag=False,
            contrast_factor=1.0, add_white_border=True, border_thickness_px=8,
        )
        assert mine.image_rgb.shape == expected.shape == (112 + 16, 640 + 16, 3)
        diff = np.abs(mine.image_rgb.astype(int) - expected)
        assert np.percentile(diff, 99) <= 3 and diff.mean() < 1.5


def test_sparse_step_feeds_only_keyframes_to_nvdec(video: Path, fake_gpu) -> None:
    sampler = _FakeNvdecSampler()
    frames = list(sampler.iter_frames(_metadata(video), ROI, FrameSamplingConfig(sample_step_sec=6.0)))
    assert sampler.decoder is not None and sampler.decoder.fed and all(sampler.decoder.fed)   # mã cũ: mọi gói
    assert frames


def test_gpu_failure_mid_stream_resumes_on_pyav_without_losing_frames(video: Path, fake_gpu) -> None:
    """Lỗi GPU giữa chừng: mã cũ BỎ KHUNG im lặng (``_convert_and_crop_cupy`` trả ``None``). Nay PyAV làm tiếp từ đúng
    mốc đó: đủ khung, đúng thứ tự, đánh số nối tiếp, vẫn được tiền xử lý."""
    reference = list(PyAvFrameSampler().iter_frames(_metadata(video), ROI, GATE))
    wrapped = FallbackFrameSampler(_FakeNvdecSampler(fail_at=1.5), PyAvFrameSampler)
    frames = list(wrapped.iter_frames(_metadata(video), ROI, GATE))
    assert wrapped.last_fallback_reason and "1.5" in wrapped.last_fallback_reason
    assert [round(f.timestamp_sec, 3) for f in frames] == [round(f.timestamp_sec, 3) for f in reference]
    assert [f.frame_index for f in frames] == list(range(len(frames)))
    assert sum(bool(f.image_rgb.size) for f in frames) >= sum(bool(f.image_rgb.size) for f in reference)


def test_fallback_from_the_start_applies_vram_steps_on_cpu(video: Path) -> None:
    class Broken:
        last_boundary_track = None

        def iter_frames(self, *args):
            raise VideoDecodeError("NVIDIA NVDEC không hỗ trợ codec")
            yield  # pragma: no cover

    config = FrameSamplingConfig(sample_step_sec=0.04, vram_upscale_small_text=True, vram_upscale_target_height_px=112)
    frames = list(FallbackFrameSampler(Broken(), PyAvFrameSampler).iter_frames(_metadata(video), ROI, config))
    shapes = {f.image_rgb.shape for f in frames if f.image_rgb.size}
    assert shapes == {(112, 640, 3)}                    # bước "VRAM" không bị mất khi chạy PyAV


def test_hdr_path_uses_the_shared_deduper(monkeypatch: pytest.MonkeyPatch, video: Path) -> None:
    """Đường 10-bit/HDR (PyAV + tonemap): trước đây băm thô và BỎ HẲN khung trùng (mất mốc chữ còn hiện). Nay phát đủ
    khung như PyAV, kể cả khung trùng / trống."""
    monkeypatch.setattr(nvdec, "HAS_CUPY", False)
    sampler = object.__new__(nvdec.PyNvVideoCodecFrameSampler)
    monkeypatch.setattr(sampler, "_build_hdr_tonemap_graph", lambda stream: (None, None), raising=False)
    frames = list(sampler._iter_frames_hdr_via_pyav(_metadata(video), ROI, GATE, is_hdr=False))
    reference = list(PyAvFrameSampler().iter_frames(_metadata(video), ROI, GATE))
    assert _summary(frames) == _summary(reference)
    assert any(f.is_duplicate for f in frames) and any(f.is_blank for f in frames)


def test_routing_follows_the_container_decoder_choice(monkeypatch: pytest.MonkeyPatch) -> None:
    """Có CuPy mà thiếu PyNvVideoCodec: container giải mã PyAV → tiền xử lý phải ở CPU (mã cũ giao cho VRAM → mất)."""
    from subtitles_extractor.composition import preprocess_routing
    from subtitles_extractor.infrastructure.settings.application_settings import ApplicationSettings

    settings = ApplicationSettings()
    settings.hardware.frame_decoder_backend = "pynvvideocodec"
    settings.preprocess.apply_sharpen = True
    monkeypatch.setattr(preprocess_routing, "find_spec", lambda name: object() if name == "cupy" else None)
    assert not preprocess_routing.nvdec_sampler_selected(settings)
    vram, cpu = preprocess_routing.route_preprocessing(settings.preprocess, on_gpu=False)
    assert cpu.apply_sharpen and not vram["vram_sharpen"]
    vram, cpu = preprocess_routing.route_preprocessing(settings.preprocess, on_gpu=True)
    assert vram["vram_sharpen"] and not cpu.apply_sharpen


# ── Rà độc lập (v3.23.723) ─────────────────────────────────────────────────────────────────────────────────────────


def test_vram_upscale_cap_matches_cpu_for_very_short_roi() -> None:
    """ROI thấp hơn 1/3 đích: VRAM trước đây trần ×3 (CPU ×4) → ảnh khác cỡ hẳn."""
    image = np.random.default_rng(4).integers(0, 256, (30, 200, 3), dtype=np.uint8)
    out = gpu.apply_gpu_preprocessing_pipeline(image, {}, True, 128, False, 1.0, False, 0, xp=np)
    ref = apply_cpu_preprocessing_pipeline(
        image, upscale_small_text=True, upscale_target_height_px=128, apply_clahe_flag=False, clahe_clip_limit=3.0,
        clahe_tile_size=8, apply_sharpen_flag=False, apply_contrast_boost_flag=False, contrast_factor=1.0,
        add_white_border=False, border_thickness_px=0,
    )
    assert out.shape == ref.shape == (120, 800, 3)               # mã cũ: (90, 600)
    assert int(np.abs(out.astype(int) - ref).max()) <= 1


def _plane(plane, rows: int, cols: int) -> np.ndarray:
    return np.frombuffer(bytes(plane), np.uint8).reshape(-1, plane.line_size)[:rows, :cols]


@pytest.mark.parametrize("width, height", [(321, 192), (853, 480), (320, 193), (321, 193)])
def test_odd_frame_size_keeps_the_last_chroma_column_and_row(width: int, height: int) -> None:
    """Bề rộng / chiều cao lẻ: mã cũ cắt UV ``H//2 × W`` → ROI chạm mép phải / dưới đọc ngoài mảng (NumPy: IndexError →
    cả lượt lùi PyAV; CuPy: quấn về cột 0). Bề rộng lẻ: trùng khít PyAV; chiều cao lẻ (swscale đổi đường tính): không lỗi."""
    rgb = np.random.default_rng(width + height).integers(0, 256, (height, width, 3), dtype=np.uint8)
    frame = av.VideoFrame.from_ndarray(rgb, format="rgb24").reformat(format="yuv420p")
    ref = frame.to_ndarray(format="rgb24")
    chroma_w, chroma_h = (width + 1) // 2, (height + 1) // 2
    nv12 = np.zeros((height + chroma_h, 2 * chroma_w + 32), np.uint8)               # pitch rộng như NVDEC
    nv12[:height, :width] = _plane(frame.planes[0], height, width)
    nv12[height:, 0:2 * chroma_w:2] = _plane(frame.planes[1], chroma_h, chroma_w)
    nv12[height:, 1:2 * chroma_w:2] = _plane(frame.planes[2], chroma_h, chroma_w)
    y_plane, uv_plane = nvdec._crop_nv12(*nvdec._split_nv12(nv12, height), width, height)
    out = gpu.nv12_roi_to_rgb(y_plane, uv_plane, 0, 0, width, height, matrix="bt601", full_range=False, xp=np)
    assert out.shape == ref.shape
    if height % 2 == 0:
        assert np.array_equal(out, ref)
    else:
        assert abs(np.abs(out.astype(int) - rgb).mean() - np.abs(ref.astype(int) - rgb).mean()) < 2.0


def test_provenance_records_the_decoder_that_really_ran(video: Path, fake_gpu) -> None:
    from types import SimpleNamespace as NS

    from subtitles_extractor.application.services.run_provenance import collect_run_info

    def record(sampler) -> dict:
        request = NS(video_path=video, roi=ROI, sampling=GATE, ocr=None)
        return collect_run_info(run_id="t", app_version="x", request=request, metadata=_metadata(video),
                                builder_config=None, frame_sampler=sampler, ocr_engine=None)["environment"]

    whole = FallbackFrameSampler(_FakeNvdecSampler(), PyAvFrameSampler)
    list(whole.iter_frames(_metadata(video), ROI, GATE))
    assert record(whole)["frame_sampler"] == "_FakeNvdecSampler"
    assert record(whole)["frame_sampler_fallback_reason"] is None
    broken = FallbackFrameSampler(_FakeNvdecSampler(fail_at=1.5), PyAvFrameSampler)
    list(broken.iter_frames(_metadata(video), ROI, GATE))
    env = record(broken)
    assert env["frame_sampler"].startswith("_FakeNvdecSampler→PyAvFrameSampler@1.4")        # mã cũ: "FallbackFrameSampler"
    assert "cudaErrorLaunchFailure" in env["frame_sampler_fallback_reason"]


@pytest.mark.parametrize("step", [0.2, 0.12])
def test_fallback_resume_stays_on_the_sampling_grid(video: Path, fake_gpu, step: float) -> None:
    """Mã cũ tiếp tục ở ``mốc cuối + 1 ms`` → thêm một khung ngay sau và lệch lưới lấy mẫu về sau."""
    config = FrameSamplingConfig(sample_step_sec=step)
    reference = [round(f.timestamp_sec, 3) for f in PyAvFrameSampler().iter_frames(_metadata(video), ROI, config)]
    wrapped = FallbackFrameSampler(_FakeNvdecSampler(fail_at=1.5), PyAvFrameSampler)
    got = [round(f.timestamp_sec, 3) for f in wrapped.iter_frames(_metadata(video), ROI, config)]
    assert got == reference


def test_auto_roi_engine_config_does_not_redo_vram_steps_on_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tự dò vùng gọi OCR không kèm cấu hình → cấu hình mặc định của container. Giải mã NVDEC: CLAHE / làm nét đã chạy
    trên VRAM, CPU không được làm lại (mã cũ: CLAHE hai lần)."""
    from subtitles_extractor.composition import preprocess_routing
    from subtitles_extractor.composition.container import ApplicationContainer
    from subtitles_extractor.infrastructure.settings.application_settings import ApplicationSettings

    settings = ApplicationSettings()
    settings.preprocess.apply_clahe = True
    settings.preprocess.apply_sharpen = True
    monkeypatch.setattr(preprocess_routing, "nvdec_sampler_selected", lambda snapshot: True)
    cpu = ApplicationContainer._routed_cpu_preprocess(settings, settings.preprocess)
    assert not cpu.apply_clahe and not cpu.apply_sharpen
    monkeypatch.setattr(preprocess_routing, "nvdec_sampler_selected", lambda snapshot: False)
    cpu = ApplicationContainer._routed_cpu_preprocess(settings, settings.preprocess)
    assert cpu.apply_clahe and cpu.apply_sharpen


def test_hdr_path_sparse_step_decodes_keyframes_only(monkeypatch: pytest.MonkeyPatch, video: Path) -> None:
    monkeypatch.setattr(nvdec, "HAS_CUPY", False)
    sampler = object.__new__(nvdec.PyNvVideoCodecFrameSampler)
    monkeypatch.setattr(sampler, "_build_hdr_tonemap_graph", lambda stream: (None, None), raising=False)
    sparse = FrameSamplingConfig(sample_step_sec=6.0)
    frames = list(sampler._iter_frames_hdr_via_pyav(_metadata(video), ROI, sparse, is_hdr=False))
    reference = list(PyAvFrameSampler().iter_frames(_metadata(video), ROI, sparse))
    assert _summary(frames) == _summary(reference)


def test_hdr_path_keeps_going_on_cpu_when_the_gpu_fails(monkeypatch: pytest.MonkeyPatch, video: Path) -> None:
    """GPU lỗi trên video HDR: làm tiếp bằng NumPy (giữ tonemap) thay vì lùi PyAV thường (không tonemap)."""
    class FlakyGpu:
        __name__ = "flaky"

        def __getattr__(self, name):
            def boom(*args, **kwargs):
                raise RuntimeError("cudaErrorIllegalAddress (giả lập)")
            return boom

    monkeypatch.setattr(nvdec, "HAS_CUPY", True)
    monkeypatch.setattr(nvdec, "cp", FlakyGpu(), raising=False)
    sampler = object.__new__(nvdec.PyNvVideoCodecFrameSampler)
    monkeypatch.setattr(sampler, "_build_hdr_tonemap_graph", lambda stream: (None, None), raising=False)
    frames = list(sampler._iter_frames_hdr_via_pyav(_metadata(video), ROI, GATE, is_hdr=True))
    reference = list(PyAvFrameSampler().iter_frames(_metadata(video), ROI, GATE))
    assert _summary(frames) == _summary(reference)


@pytest.mark.parametrize("length", [1, 2, 3, 4, 9])
def test_reflect101_matches_opencv_for_tiny_images(length: int) -> None:
    import cv2

    expected = [cv2.borderInterpolate(i, length, cv2.BORDER_REFLECT_101) for i in range(-3, length + 3)]
    assert gpu._reflect101(length, 3, 3).tolist() == expected
