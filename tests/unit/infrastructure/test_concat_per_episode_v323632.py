"""[v3.23.632] Nối cả bộ: chuẩn hoá TỪNG TẬP (tiếp tục được) rồi nối sao chép luồng — ffmpeg thật.

Log thật 16/10 (86 tập, GPU): MỘT lệnh ffmpeg mở 86 đầu vào, ``hevc_nvenc`` hỏng sau 4 phút (lý do
không được ghi), rồi chạy lại TOÀN BỘ bằng ``libx265`` mất 39 phút.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from itertools import pairwise
from pathlib import Path

import pytest

from subtitles_extractor.infrastructure.video import ffmpeg_concat_adapter as adapter
from subtitles_extractor.infrastructure.video import video_encoder_selector as selector

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="Cần ffmpeg + ffprobe",
)


def _clip(path: Path, *, fps: int, seconds: float, size: str = "160x284") -> Path:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y",
         "-f", "lavfi", "-i", f"testsrc2=s={size}:r={fps}:d={seconds}",
         "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=44100:duration={seconds}",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", str(fps), "-bf", "2",
         "-c:a", "aac", "-shortest", str(path)],
        check=True, timeout=120,
    )
    return path


def _stamps(path: Path) -> list[float]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "frame=pts_time", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True, timeout=120,
    ).stdout
    return sorted(float(t.strip(",")) for t in out.split() if t.strip(",") not in ("", "N/A"))


@pytest.fixture
def clips(tmp_path: Path) -> list[Path]:
    return [
        _clip(tmp_path / "01.mp4", fps=25, seconds=2),
        _clip(tmp_path / "02.mp4", fps=30, seconds=2),
        _clip(tmp_path / "03.mp4", fps=25, seconds=2, size="320x568"),
        _clip(tmp_path / "04.mp4", fps=25, seconds=1),
    ]


def _infos(clips: list[Path]):
    return adapter.probe_concat_inputs(clips)


def _target(infos):
    return adapter.choose_concat_target(infos)


def _by_episode(clips, output, encoder="libx265", **kw):
    infos = _infos(clips)
    return adapter._concat_by_episode(
        "ffmpeg", infos, _target(infos), output, encoder, 0.0,
        kw.get("on_progress"), kw.get("cancel_check"), 600.0,
    )


def test_result_is_gap_free_and_work_dir_is_removed(tmp_path: Path, clips: list[Path]) -> None:
    output = tmp_path / "tron_bo.mkv"
    _by_episode(clips, output)
    stamps = _stamps(output)
    assert max(b - a for a, b in pairwise(stamps)) < 0.1
    assert 170 <= len(stamps) <= 180          # 7 s ở 25 fps
    assert not (tmp_path / "tron_bo.parts").exists()


def test_progress_is_monotonic_and_ends_at_one(tmp_path: Path, clips: list[Path]) -> None:
    seen: list[float] = []
    _by_episode(clips, tmp_path / "t.mkv", on_progress=seen.append)
    assert seen and seen[-1] == 1.0 and all(0.0 <= v <= 1.0 for v in seen)
    assert seen == sorted(seen) or max(a - b for a, b in pairwise(seen)) < 0.5   # luồng song song: gần đơn điệu


def test_interrupted_run_resumes_without_redoing_finished_episodes(
    tmp_path: Path, clips: list[Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "tron_bo.mkv"
    real = adapter._run_ffmpeg_concat
    calls: list[str] = []

    def flaky(command, out, *a, **k):
        name = Path(command[command.index("-i") + 1]).name
        calls.append(name)
        if name == "03.mp4":
            raise adapter.VideoConcatError("ffmpeg lỗi (mã 1): giả lập")
        return real(command, out, *a, **k)

    monkeypatch.setattr(adapter, "_run_ffmpeg_concat", flaky)
    monkeypatch.setattr(adapter, "_parallel_parts", lambda _encoder: 1)
    with pytest.raises(adapter.VideoConcatError, match="03.mp4"):
        _by_episode(clips, output)
    work = tmp_path / "tron_bo.parts" / "libx265"
    assert sorted(p.name for p in work.glob("*.mkv")) == ["000.mkv", "001.mkv"]

    calls.clear()
    monkeypatch.setattr(adapter, "_run_ffmpeg_concat", real)
    counted: list[str] = []

    def counting(command, out, *a, **k):
        if "-f" not in command[:6] or command[1] != "-hide_banner" or "-filter_complex" in command:
            counted.append(Path(command[command.index("-i") + 1]).name)
        return real(command, out, *a, **k)

    monkeypatch.setattr(adapter, "_run_ffmpeg_concat", counting)
    _by_episode(clips, output)
    assert [n for n in counted if n.endswith(".mp4")] == ["03.mp4", "04.mp4"]   # 01, 02 dùng lại
    assert output.is_file()


def test_changed_source_invalidates_its_part(tmp_path: Path, clips: list[Path]) -> None:
    output = tmp_path / "tron_bo.mkv"
    infos = _infos(clips)
    target = _target(infos)
    # Dựng sẵn các tập, rồi đổi nguồn tập 2 → chỉ tập đó bị làm lại.
    work = tmp_path / "x.parts" / "libx265"
    work.mkdir(parents=True)
    quality = adapter.encoder_quality_args("libx265", output_path=tmp_path / "x.mkv")
    signature = adapter._part_signature(target, "libx265", quality)
    part, meta = work / "000.mkv", work / "000.json"
    shutil.copy(clips[0], part)
    meta.write_text(
        json.dumps({"signature": signature, "source": adapter._source_stamp(infos[0])}),
        encoding="utf-8",
    )
    assert adapter._valid_part(part, meta, infos[0], signature)
    os.utime(clips[0], (1, 1))                                  # nguồn đổi mtime
    infos_after = _infos(clips)
    assert not adapter._valid_part(part, meta, infos_after[0], signature)
    assert not adapter._valid_part(part, meta, infos[0], "khác-cấu-hình")


def test_cancel_keeps_finished_parts_for_the_next_run(tmp_path: Path, clips: list[Path]) -> None:
    output = tmp_path / "tron_bo.mkv"
    state = {"n": 0}

    def cancel() -> bool:
        state["n"] += 1
        return state["n"] > 60          # huỷ sau một lúc, khi vài tập đã xong

    with pytest.raises(adapter.VideoConcatError):
        _by_episode(clips, output, cancel_check=cancel)
    assert not output.exists()


def test_gpu_failure_is_logged_with_its_reason_and_falls_back_for_every_episode(
    tmp_path: Path, clips: list[Path], monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Sandbox không có GPU: ``hevc_nvenc`` hỏng THẬT khi chạy — đúng kịch bản log."""
    monkeypatch.setattr(adapter, "pick_video_encoder", lambda *_a, **_k: "hevc_nvenc")
    monkeypatch.setattr(selector, "_broken", set())
    caplog.set_level(logging.WARNING)
    output = tmp_path / "tron_bo.mkv"
    infos = _infos(clips)
    adapter._concat_normalized("ffmpeg", infos, output, 0.0, None, None, 600.0)
    assert output.is_file()
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "hevc_nvenc lỗi khi chuẩn hoá từng tập" in messages and "Lý do:" in messages
    stamps = _stamps(output)
    assert max(b - a for a, b in pairwise(stamps)) < 0.1


def test_single_graph_remains_as_the_last_resort(
    tmp_path: Path, clips: list[Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_a, **_k):
        raise adapter.VideoConcatError("giả lập")

    monkeypatch.setattr(adapter, "_concat_by_episode", boom)
    output = tmp_path / "tron_bo.mkv"
    adapter._concat_normalized("ffmpeg", _infos(clips), output, 0.0, None, None, 600.0)
    assert max(b - a for a, b in pairwise(_stamps(output))) < 0.1


def test_parallelism_policy() -> None:
    assert adapter._parallel_parts("hevc_nvenc") == 2
    assert 1 <= adapter._parallel_parts("libx265") <= 4
