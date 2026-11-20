"""[v3.23.728] "Xuất dữ liệu thô OCR": tệp nén ``.seraw.json.gz`` như lượt thường + VIDEO vùng ROI thay cho ảnh JPG.

Ảnh JPG từng khung (hai ảnh mỗi khung có chữ) của phim nối 86 tập là hàng trăm nghìn tệp, nhiều GB — không gửi đi gỡ
lỗi được. Video ROI tập 24 (64 s, 720×136, HEVC CRF 26): 1,4 MB, PSNR độ chói 39,5 dB so với khung gốc.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from subtitles_extractor.domain.value_objects.roi import Roi
from subtitles_extractor.infrastructure.video.roi_clip_exporter import (
    FfmpegRoiClipExporter,
    build_attempts,
    clip_paths_for,
    even_crop,
)
from subtitles_extractor.infrastructure.video.roi_clip_reader import RoiClipReader, find_roi_clip
from subtitles_extractor.presentation.view_models.extract_page_view_model import compressed_raw_path

FFMPEG = shutil.which("ffmpeg")
needs_ffmpeg = pytest.mark.skipif(FFMPEG is None, reason="cần ffmpeg")


# ── tên tệp ──────────────────────────────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(("chosen", "expected"), [
    ("phim.seraw.json", "phim.seraw.json.gz"),
    ("phim.seraw.json.gz", "phim.seraw.json.gz"),
    ("phim.json", "phim.seraw.json.gz"),
    ("phim", "phim.seraw.json.gz"),
    ("PHIM.SERAW.JSON", "PHIM.seraw.json.gz"),
])
def test_raw_export_is_always_compressed(chosen: str, expected: str) -> None:
    assert compressed_raw_path(Path("/x") / chosen).name == expected


def test_clip_paths_follow_raw_file() -> None:
    clip, sidecar = clip_paths_for(Path("/x/phim_roi2.seraw.json.gz"))
    assert clip.name == "phim_roi2.roi.mp4" and sidecar.name == "phim_roi2.roi.json"


# ── vùng cắt ─────────────────────────────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(("roi", "expected"), [
    (Roi(0, 774, 720, 136), (0, 774, 720, 136)),
    (Roi(35, 773, 645, 148), (34, 772, 646, 150)),          # lẻ → chẵn, vẫn phủ trọn ROI
    (Roi(701, 1275, 19, 5), (700, 1274, 20, 6)),             # sát mép: không vượt khung
    (None, (0, 0, 720, 1280)),
])
def test_even_crop_covers_roi(roi: Roi | None, expected: tuple[int, int, int, int]) -> None:
    x, y, w, h = even_crop(roi, 720, 1280)
    assert (x, y, w, h) == expected and x % 2 == y % 2 == w % 2 == h % 2 == 0
    assert x + w <= 720 and y + h <= 1280
    if roi is not None:
        assert x <= roi.x and y <= roi.y and x + w >= min(720, roi.x + roi.width) and y + h >= min(1280, roi.y + roi.height)


# ── thang lệnh ffmpeg ────────────────────────────────────────────────────────────────────────────────────────────────
def _attempts(hardware: str | None, codec: str = "hevc"):  # noqa: ANN202
    return build_attempts(
        "ffmpeg", Path("in.mkv"), (0, 774, 720, 136), (720, 1280), codec, Path("out.roi.mp4"),
        hardware_encoder=hardware, software_encoder="libx265", quality_args=lambda encoder: ["-q", encoder],
    )


def test_gpu_ladder_starts_zero_copy_on_vram() -> None:
    attempts = _attempts("hevc_nvenc")
    assert [a.pipeline for a in attempts] == ["gpu_zero_copy", "gpu_hybrid", "cpu"]
    zero = attempts[0].command
    # NVDEC giải mã + cắt ROI trong bộ giải mã, khung ở lại VRAM, NVENC mã hoá — không bộ lọc CPU nào.
    assert zero[zero.index("-c:v") + 1] == "hevc_cuvid"
    assert zero[zero.index("-crop") + 1] == "774x370x0x0"          # trên x dưới x trái x phải
    assert zero[zero.index("-hwaccel_output_format") + 1] == "cuda"
    assert "-vf" not in zero and "hevc_nvenc" in zero
    # [v3.23.735] luôn đưa khung về cỡ gốc trước khi cắt (video đổi cỡ giữa chừng); scale tự bỏ qua khi đúng cỡ.
    assert attempts[1].command[attempts[1].command.index("-vf") + 1] == "scale=720:1280,crop=720:136:0:774"
    assert "libx265" in attempts[2].command


def test_unknown_codec_skips_zero_copy_and_cpu_only_without_gpu() -> None:
    assert [a.pipeline for a in _attempts("hevc_nvenc", codec="prores")] == ["gpu_hybrid", "cpu"]
    assert [a.pipeline for a in _attempts(None)] == ["cpu"]
    assert [a.pipeline for a in _attempts("hevc_qsv")] == ["gpu_hybrid", "cpu"]


def test_every_attempt_keeps_every_frame_and_drops_audio() -> None:
    for attempt in _attempts("hevc_nvenc"):
        command = attempt.command
        assert command[command.index("-fps_mode") + 1] == "passthrough" and "-an" in command
        assert command[-1] == "out.roi.mp4"


# ── chạy thật (CPU) ──────────────────────────────────────────────────────────────────────────────────────────────────
def _make_video(path: Path, start_offset: float = 1.0) -> None:
    """3 s, 25 fps, 720×1280, khung có số thứ tự sáng trong vùng phụ đề; mốc đầu tệp ``start_offset`` s."""
    subprocess.run([
        FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
        "testsrc2=size=720x1280:rate=25:duration=3", "-vf",
        "drawbox=x=200:y=800:w=320:h=80:color=white:t=fill,"
        "drawtext=text='%{frame_num}':x=300:y=810:fontsize=60:fontcolor=black",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-output_ts_offset", str(start_offset), str(path),
    ], check=True, capture_output=True)


def _source_frames(path: Path) -> list[tuple[float, np.ndarray]]:
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        return [(float(f.pts * stream.time_base), f.to_ndarray(format="rgb24")) for f in container.decode(stream)]


@needs_ffmpeg
@pytest.mark.parametrize("prefer_gpu", [False, True])
def test_export_and_read_back_every_frame(tmp_path: Path, prefer_gpu: bool) -> None:
    from subtitles_extractor.infrastructure.video.video_encoder_selector import reset_encoder_cache

    reset_encoder_cache()
    video = tmp_path / "phim.mkv"
    _make_video(video)
    roi = Roi(35, 773, 645, 148)
    clip, sidecar = clip_paths_for(tmp_path / "phim.seraw.json.gz")
    progress: list[float] = []
    result = FfmpegRoiClipExporter(lambda: FFMPEG).export(
        video, roi, clip, prefer_gpu=prefer_gpu, progress=lambda f: progress.append(f) or True,
    )
    assert result is not None and clip.is_file() and sidecar.is_file()
    assert result.crop_xywh == (34, 772, 646, 150)
    if not prefer_gpu:
        assert result.pipeline == "cpu"                      # máy không GPU: lùi về CPU vẫn ra video
    info = json.loads(sidecar.read_text(encoding="utf-8"))
    assert info["roi_xywh"] == [35, 773, 645, 148] and info["source_start_sec"] == pytest.approx(1.0, abs=0.05)
    assert progress and max(progress) <= 1.0

    reader = RoiClipReader(*find_roi_clip(tmp_path / "phim.seraw.json.gz"))
    assert reader.box_offset == (1, 1)
    source = _source_frames(video)
    assert source[0][0] == pytest.approx(1.0, abs=0.05)   # bộ giải mã của ứng dụng thấy mốc thật (không trừ start)
    for ts, full in source[::7] + [source[-1]]:
        image = reader.frame_at(ts)
        assert image is not None and image.shape[:2] == (150, 646)
        crop = full[772:922, 34:680].astype(np.int16)
        assert float(np.abs(image.astype(np.int16) - crop).mean()) < 6.0, ts    # đúng khung, nén nhẹ
    reader.close()


@needs_ffmpeg
def test_cancel_removes_partial_clip(tmp_path: Path) -> None:
    video = tmp_path / "phim.mkv"
    _make_video(video)
    clip = tmp_path / "phim.roi.mp4"
    result = FfmpegRoiClipExporter(lambda: FFMPEG).export(
        video, Roi(0, 774, 720, 136), clip, prefer_gpu=False, progress=lambda f: False,
    )
    assert result is None and not clip.exists()


def test_missing_ffmpeg_or_video_is_not_fatal(tmp_path: Path) -> None:
    assert FfmpegRoiClipExporter(lambda: None).export(
        tmp_path / "x.mkv", None, tmp_path / "x.roi.mp4", prefer_gpu=True) is None
    assert FfmpegRoiClipExporter(lambda: "ffmpeg").export(
        tmp_path / "khong_co.mkv", None, tmp_path / "x.roi.mp4", prefer_gpu=True) is None
