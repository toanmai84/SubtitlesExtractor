"""[v3.23.161] Test CUDA device TƯỜNG MINH cho filter + per-source cả cặp tầng CUDA.

Log thực tế: CẢ HAI tầng CUDA (thuần + qua-upload) cùng chết -40 với danh sách format
của encoder -> filter CUDA không có device context khi hwaccel decoder init thất bại
với stream đó. ffmpeg trên máy (gyan git 2026, --disable-autodetect) CÓ đủ filter ->
fix đúng tài liệu HWAccelIntro: ``-init_hw_device cuda=cu -filter_hw_device cu`` cấp
device cho filter ĐỘC LẬP với hwaccel. Kèm: tầng qua-upload fail cũng được nhớ theo
video để đoạn sau bỏ cả cặp (không thử-fail lặp).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from subtitles_extractor.infrastructure.translation.gemini_video_context import (
    GeminiVideoContextProvider,
    VideoChunk,
)


@pytest.fixture()
def provider(tmp_path: Path) -> GeminiVideoContextProvider:
    return GeminiVideoContextProvider(
        cache_db_path=tmp_path / "cache.db", work_dir=tmp_path
    )


def _cmds(provider: GeminiVideoContextProvider, tmp_path: Path, source: Path):
    provider._nvenc_available = True
    chunk = VideoChunk(0, tmp_path / "o.mp4", 0.0, 100.0, False)
    return provider._build_encode_commands("ffmpeg", source, chunk, tmp_path / "t.mp4")


def _has_pair(cmd: list[str]) -> bool:
    return (
        "-init_hw_device" in cmd
        and "cuda=cu" in cmd
        and "-filter_hw_device" in cmd
        and "cu" in cmd
    )


def test_cuda_tiers_carry_explicit_filter_device(provider, tmp_path) -> None:
    cmds = _cmds(provider, tmp_path, Path("movie.mkv"))
    cuda_tiers = [cmd for _g, cmd in cmds if any("scale_cuda" in str(a) for a in cmd)]
    assert len(cuda_tiers) == 2  # thuần-GPU + qua-upload
    assert all(_has_pair(cmd) for cmd in cuda_tiers)
    # Các tầng KHÔNG dùng filter CUDA thì không cần device tường minh.
    non_cuda = [cmd for _g, cmd in cmds if not any("scale_cuda" in str(a) for a in cmd)]
    assert all(not _has_pair(cmd) for cmd in non_cuda)


def test_upload_failed_source_skips_upload_tier(provider, tmp_path) -> None:
    provider._gpu_upload_failed_sources.add(str(Path("movie.mkv")))
    cmds = _cmds(provider, tmp_path, Path("movie.mkv"))
    assert not any(
        any("hwupload_cuda" in str(a) for a in cmd) for _g, cmd in cmds
    )


def test_both_cuda_tiers_fail_remembered_and_counted_once(
    provider, tmp_path, monkeypatch
) -> None:
    provider._nvenc_available = True
    provider._detect_nvenc = lambda _f: True  # type: ignore[method-assign]
    source = tmp_path / "movie.mkv"
    chunk = VideoChunk(0, tmp_path / "movie.ctxpart00.h360f1.mp4", 0.0, 60.0, False)

    def fake_run(cmd, **_kwargs):
        if any("scale_cuda" in str(a) for a in cmd):
            raise subprocess.CalledProcessError(-40, cmd, stderr=b"reinit filters")
        out = Path(cmd[-1])
        out.write_bytes(b"0" * 2048)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    provider._encode_low_res_chunk("ffmpeg", source, chunk)

    assert str(source) in provider._gpu_full_failed_sources
    assert str(source) in provider._gpu_upload_failed_sources
    assert provider._scale_cuda_failures == 1  # MỘT nhịp/đoạn dù 2 lệnh fail
    assert chunk.path.exists()  # tầng scale-CPU cứu thành công
