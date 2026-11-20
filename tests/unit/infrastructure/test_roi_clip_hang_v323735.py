"""[v3.23.735] Cắt video vùng ROI treo ở 39 % (phim "合欢宗…" 80 tập nối, 2 giờ 49 phút) — không báo gì.

Nguyên nhân (đo trên video thật):

* đoạn 3979,67–3992,72 s (39,3 %) là một tập 480×854 / 30 fps chèn giữa phim 720×1280 / 25 fps. Vùng cắt (cố định trong
  ``h264_cuvid -crop`` hoặc bộ lọc ``crop`` theo 720×1280) vượt khỏi khung 480×854 → ffmpeg báo lỗi liên tục / dừng;
* ứng dụng mở ffmpeg với ``stderr=PIPE`` nhưng KHÔNG đọc stderr, cũng không canh "ffmpeg ngừng tiến triển": ống stderr
  đầy (Windows ≈ 4 KB) → ffmpeg đứng chờ ghi, ứng dụng đứng chờ dòng tiến độ → treo vĩnh viễn, không lỗi, không log.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

from subtitles_extractor.domain.value_objects.roi import Roi
from subtitles_extractor.infrastructure.video import roi_clip_exporter as rce
from subtitles_extractor.infrastructure.video.roi_clip_exporter import (
    FfmpegRoiClipExporter,
    build_attempts,
    detect_size_changes,
)

FFMPEG = shutil.which("ffmpeg")
needs_ffmpeg = pytest.mark.skipif(FFMPEG is None, reason="cần ffmpeg")


def _run_in_thread(command: list[str], timeout: float) -> str | None:
    out: list[str] = []
    worker = threading.Thread(target=lambda: out.append(FfmpegRoiClipExporter._run(command, 10.0, lambda f: True)),
                              daemon=True)
    worker.start()
    worker.join(timeout)
    return out[0] if out else None


def test_stderr_flood_does_not_deadlock() -> None:
    # "ffmpeg" ghi 2 MB lỗi ra stderr (vượt mọi bộ đệm ống) rồi mới ra tiến độ và thoát — mã cũ đứng chờ mãi.
    script = (
        "import sys\n"
        "for i in range(40000): sys.stderr.write('[h264_cuvid] crop vượt khung %d\\n' % i)\n"
        "sys.stderr.flush()\n"
        "print('out_time_us=5000000', flush=True)\n"
        "print('progress=end', flush=True)\n"
    )
    assert _run_in_thread([sys.executable, "-c", script], timeout=30) == "ok"


def test_stalled_ffmpeg_is_killed(monkeypatch: pytest.MonkeyPatch) -> None:
    # ffmpeg ra một dòng tiến độ rồi đứng im (bộ giải mã kẹt) → phải huỷ và báo "stalled" để thử cách kế.
    monkeypatch.setattr(rce, "STALL_TIMEOUT_SEC", 1.5)
    script = "import time\nprint('out_time_us=1000000', flush=True)\ntime.sleep(600)\n"
    assert _run_in_thread([sys.executable, "-c", script], timeout=30) == "stalled"


def test_cancel_works_while_ffmpeg_is_silent(monkeypatch: pytest.MonkeyPatch) -> None:
    # Người dùng bấm Huỷ khi ffmpeg im lặng: không được chờ tới dòng tiến độ kế (có thể không bao giờ tới).
    script = "import time\ntime.sleep(600)\n"
    out: list[str] = []
    worker = threading.Thread(
        target=lambda: out.append(FfmpegRoiClipExporter._run([sys.executable, "-c", script], 10.0, lambda f: False)),
        daemon=True,
    )
    worker.start()
    worker.join(30)
    assert out == ["cancelled"]


# ── video đổi kích thước giữa chừng ──────────────────────────────────────────────────────────────────────────────────
def _attempts(size_changes: bool):  # noqa: ANN202
    return build_attempts(
        "ffmpeg", Path("in.mp4"), (0, 795, 720, 120), (720, 1280), "h264", Path("out.roi.mp4"),
        hardware_encoder="hevc_nvenc", software_encoder="libx265", quality_args=lambda encoder: [],
        size_changes=size_changes,
    )


def test_mixed_size_skips_decoder_crop_and_scales_before_crop() -> None:
    assert [a.pipeline for a in _attempts(False)] == ["gpu_zero_copy", "gpu_hybrid", "cpu"]
    attempts = _attempts(True)
    assert [a.pipeline for a in attempts] == ["gpu_hybrid", "cpu"]        # cắt trong bộ giải mã không theo kịp
    for attempt in attempts:
        vf = attempt.command[attempt.command.index("-vf") + 1]
        assert vf.startswith("scale=720:1280") and "crop=720:120:0:795" in vf, vf


def _segment(path: Path, size: str, rate: int, seconds: int, colour: str) -> None:
    assert FFMPEG is not None
    subprocess.run([
        FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
        f"color=c={colour}:size={size}:rate={rate}:duration={seconds}", "-vf",
        "drawbox=x=iw*0.25:y=ih*0.65:w=iw*0.5:h=ih*0.05:color=white:t=fill",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-x264-params", "repeat-headers=1:keyint=25",
        "-video_track_timescale", "90000", str(path),
    ], check=True, capture_output=True)


def _mixed_video(tmp_path: Path) -> Path:
    """Như phim thật: 720×1280/25 fps → 480×854/30 fps → 720×1280/25 fps, nối bằng chép luồng (SPS mới trong luồng)."""
    parts = [tmp_path / "a.mp4", tmp_path / "b.mp4", tmp_path / "c.mp4"]
    _segment(parts[0], "720x1280", 25, 2, "navy")
    _segment(parts[1], "480x854", 30, 1, "darkgreen")
    _segment(parts[2], "720x1280", 25, 2, "maroon")
    listing = tmp_path / "list.txt"
    listing.write_text("".join(f"file '{p.name}'\n" for p in parts), encoding="utf-8")
    video = tmp_path / "phim.mp4"
    assert FFMPEG is not None
    subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i",
                    str(listing), "-c", "copy", str(video)], check=True, capture_output=True, cwd=tmp_path)
    return video


@needs_ffmpeg
def test_detect_size_changes_from_packets(tmp_path: Path) -> None:
    changes = detect_size_changes(_mixed_video(tmp_path))
    assert [(w, h) for _, w, h in changes] == [(480, 854), (720, 1280)]
    assert changes[0][0] == pytest.approx(2.0, abs=0.1) and changes[1][0] == pytest.approx(3.0, abs=0.1)


@needs_ffmpeg
def test_uniform_video_has_no_size_change(tmp_path: Path) -> None:
    video = tmp_path / "u.mp4"
    _segment(video, "720x1280", 25, 2, "navy")
    assert detect_size_changes(video) == []


@needs_ffmpeg
def test_export_mixed_size_video_keeps_every_frame_at_source_size(tmp_path: Path) -> None:
    import av

    from subtitles_extractor.infrastructure.video.video_encoder_selector import reset_encoder_cache

    reset_encoder_cache()
    video = _mixed_video(tmp_path)
    clip = tmp_path / "phim.roi.mp4"
    progress: list[float] = []
    result = FfmpegRoiClipExporter(lambda: FFMPEG).export(
        video, Roi(0, 795, 720, 120), clip, prefer_gpu=False, progress=lambda f: progress.append(f) or True,
    )
    assert result is not None and clip.is_file()
    info = json.loads(result.sidecar_path.read_text(encoding="utf-8"))
    assert info["size_changes"] == [[pytest.approx(2.0, abs=0.1), 480, 854], [pytest.approx(3.0, abs=0.1), 720, 1280]]
    with av.open(str(clip)) as container:
        stream = container.streams.video[0]
        frames = [(float(f.pts * stream.time_base), f.to_ndarray(format="gray")) for f in container.decode(stream)]
    assert len(frames) == 50 + 30 + 50                                   # đủ mọi khung, kể cả đoạn 30 fps
    _, y, _, h = result.crop_xywh
    assert {img.shape for _, img in frames} == {(h, 720)}
    # Vạch trắng (65 %–70 % chiều cao) phóng về khung gốc → nằm cùng chỗ trong vùng cắt ở cả ba đoạn.
    top, bottom = round(0.65 * 1280) - y, round(0.70 * 1280) - y
    for ts, img in frames[::10]:
        rows = np.flatnonzero(img.mean(axis=1) > 120)
        assert rows.size and abs(rows[0] - top) <= 3 and abs(rows[-1] - bottom) <= 3, (ts, rows[:3], rows[-3:])


REAL_CLIP = Path(__file__).resolve().parents[3] / "test data" / "hh_v3_23_734" / "size_change.mp4"


@needs_ffmpeg
@pytest.mark.skipif(not REAL_CLIP.is_file(), reason="thiếu dữ liệu test data/hh_v3_23_734")
def test_real_film_ad_segment(tmp_path: Path) -> None:
    """Đoạn thật quanh 39 %: quảng cáo 480×854/30 fps chèn giữa phim 720×1280/25 fps."""
    changes = detect_size_changes(REAL_CLIP)
    assert [(w, h) for _, w, h in changes] == [(480, 854), (720, 1280)]
    assert changes[0][0] == pytest.approx(4.66, abs=0.05) and changes[1][0] == pytest.approx(17.71, abs=0.05)
    clip = tmp_path / "phim.roi.mp4"
    result = FfmpegRoiClipExporter(lambda: FFMPEG).export(REAL_CLIP, Roi(0, 795, 719, 119), clip, prefer_gpu=False)
    assert result is not None
    import av

    with av.open(str(clip)) as out, av.open(str(REAL_CLIP)) as src:
        clip_frames = sum(1 for _ in out.decode(video=0))
        source_frames = sum(1 for _ in src.decode(video=0))
        assert (out.streams.video[0].width, out.streams.video[0].height) == (720, 120)
    assert clip_frames == source_frames                                   # không mất / thêm khung nào
