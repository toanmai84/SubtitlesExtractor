"""[v3.23.160] Test ghi nhớ GPU-THUẦN fail theo VIDEO (skip thử-fail lặp lại mỗi đoạn).

Với file kiểu The Hot Spot, tầng GPU-thuần (giữ frame trên CUDA) fail vì ĐẶC TÍNH
STREAM — giống nhau cho mọi đoạn của cùng video. Trước đây 11 đoạn là 11 lần thử-fail
(~1.5s + spam log mỗi lần) dù tầng GPU-qua-upload cứu được ngay. Nay: fail lần đầu ->
nhớ theo str(source_path) -> các đoạn sau của CÙNG video vào thẳng tầng upload; video
KHÁC vẫn được thử GPU-thuần bình thường.
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


def _has_gpu_full(cmds: list[tuple[bool, list[str]]]) -> bool:
    return any("-hwaccel_output_format" in cmd for _gpu, cmd in cmds)


def test_failed_source_skips_gpu_full_tier(provider, tmp_path) -> None:
    provider._nvenc_available = True
    chunk = VideoChunk(0, tmp_path / "o.mp4", 0.0, 100.0, False)
    provider._gpu_full_failed_sources.add(str(Path("bad_stream.mkv")))

    cmds_bad = provider._build_encode_commands(
        "ffmpeg", Path("bad_stream.mkv"), chunk, tmp_path / "t.mp4"
    )
    assert not _has_gpu_full(cmds_bad)  # video này bỏ hẳn tầng GPU-thuần
    # Tầng GPU-qua-upload vẫn ĐỨNG ĐẦU (vẫn tận dụng GPU tối đa cho scale+encode).
    first_vf = cmds_bad[0][1][cmds_bad[0][1].index("-vf") + 1]
    assert "hwupload_cuda" in first_vf and "scale_cuda" in first_vf

    cmds_other = provider._build_encode_commands(
        "ffmpeg", Path("good_stream.mkv"), chunk, tmp_path / "t.mp4"
    )
    assert _has_gpu_full(cmds_other)  # video khác không bị ảnh hưởng


def test_gpu_full_failure_remembered_per_source(provider, tmp_path, monkeypatch) -> None:
    provider._nvenc_available = True
    provider._detect_nvenc = lambda _f: True  # type: ignore[method-assign]
    source = tmp_path / "movie.mkv"
    chunk = VideoChunk(0, tmp_path / "movie.ctxpart00.h360f1.mp4", 0.0, 60.0, False)

    def fake_run(cmd, **_kwargs):
        if "-hwaccel_output_format" in cmd:
            raise subprocess.CalledProcessError(-40, cmd, stderr=b"reinit filters")
        # Tầng upload thành công: tạo file ra đủ lớn.
        out = Path(cmd[-1])
        out.write_bytes(b"0" * 2048)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    provider._encode_low_res_chunk("ffmpeg", source, chunk)

    assert str(source) in provider._gpu_full_failed_sources
    # Biến thể scale_cuda (tầng upload) THÀNH CÔNG -> không cộng nhịp tắt scale_cuda.
    assert provider._scale_cuda_failures == 0
    assert chunk.path.exists()

    # Đoạn kế của CÙNG video: lệnh GPU-thuần không còn trong danh sách.
    cmds_next = provider._build_encode_commands(
        "ffmpeg", source, VideoChunk(1, tmp_path / "p1.mp4", 60.0, 120.0, False),
        tmp_path / "t2.mp4",
    )
    assert not _has_gpu_full(cmds_next)
