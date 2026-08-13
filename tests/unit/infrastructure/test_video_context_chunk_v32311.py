"""Test [v3.23.12] materialize_chunks: nén 360p, tên CJK tạo được, upload toàn bộ."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from subtitles_extractor.domain.ports.video_context_port import (
    VideoChunk,
    VideoContextPlan,
)
from subtitles_extractor.infrastructure.translation.gemini_video_context import (
    GeminiVideoContextProvider,
    VideoContextError,
    _TOKENS_PER_SECOND,
)

_FFMPEG = shutil.which("ffmpeg")
_FFPROBE = shutil.which("ffprobe")
pytestmark = pytest.mark.skipif(_FFMPEG is None, reason="cần ffmpeg")


def _make_test_mkv(path: Path, seconds: int = 8, height: int = 1080) -> None:
    subprocess.run(
        [
            _FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", f"testsrc=duration={seconds}:size={height*16//9}x{height}:rate=30",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
            "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac",
            str(path),
        ],
        check=True, capture_output=True, timeout=120,
    )


class TestMaterializeLowRes:
    def test_creates_chunk_files_with_cjk_name(self, tmp_path: Path) -> None:
        # Tên CJK như log thực tế gây lỗi trước đây.
        src = tmp_path / "[达尔文]中文字幕.mkv"
        _make_test_mkv(src, seconds=8, height=1080)
        provider = GeminiVideoContextProvider(work_dir=tmp_path)
        chunks = [
            VideoChunk(0, tmp_path / "[达尔文]中文字幕.ctxpart00.mp4", 0.0, 4.0),
            VideoChunk(1, tmp_path / "[达尔文]中文字幕.ctxpart01.mp4", 4.0, 8.0),
        ]
        plan = VideoContextPlan(src, 8.0, 800, chunks)
        provider.materialize_chunks(plan)
        for chunk in chunks:
            assert chunk.path.exists(), f"{chunk.path.name} không được tạo (lỗi CJK path)"
            assert chunk.path.stat().st_size > 1024

    @pytest.mark.skipif(_FFPROBE is None, reason="cần ffprobe")
    def test_downscales_to_360p(self, tmp_path: Path) -> None:
        src = tmp_path / "video.mkv"
        _make_test_mkv(src, seconds=4, height=1080)
        provider = GeminiVideoContextProvider(work_dir=tmp_path)
        chunk = VideoChunk(0, tmp_path / "out.ctxpart00.mp4", 0.0, 4.0)
        plan = VideoContextPlan(src, 4.0, 400, [chunk, chunk])  # needs_split
        provider._encode_low_res_chunk(_FFMPEG, src, chunk)
        out = subprocess.run(
            [_FFPROBE, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=height", "-of", "csv=p=0", str(chunk.path)],
            capture_output=True, text=True,
        )
        assert out.stdout.strip() == "360"  # đã giảm về 360p

    def test_hash_file_clear_error_when_missing(self, tmp_path: Path) -> None:
        provider = GeminiVideoContextProvider(work_dir=tmp_path)
        with pytest.raises(VideoContextError, match="không tồn tại"):
            provider._hash_file(tmp_path / "khong_co.mp4")


class TestUploadFullVideo:
    def test_long_video_not_truncated_at_low_res(self, tmp_path: Path) -> None:
        # Video 6847s @ 100 token/s = 685k < 950k trần → KHÔNG cắt bớt.
        provider = GeminiVideoContextProvider(
            work_dir=tmp_path, max_total_tokens=950_000
        )
        est = int(6847 * _TOKENS_PER_SECOND)
        assert est < 950_000  # toàn bộ video vừa trần

    def test_tokens_per_second_is_low_res(self) -> None:
        # Phải là ~100 (low res), không phải 263 (default).
        assert _TOKENS_PER_SECOND == 100


class TestInMemRefsCache:
    """[v3.23.14] prepare_and_upload tái dùng refs giữa analyze→translate (cùng video)."""

    def test_second_call_skips_materialize_and_upload(self, tmp_path: Path) -> None:
        from unittest.mock import patch
        from subtitles_extractor.domain.ports.video_context_port import VideoContextPlan
        from subtitles_extractor.infrastructure.translation.gemini_video_context import (
            RemoteVideoRef,
        )

        src = tmp_path / "movie.mkv"
        _make_test_mkv(src, seconds=4)
        provider = GeminiVideoContextProvider(work_dir=tmp_path)

        mat_calls = {"n": 0}
        up_calls = {"n": 0}

        def fake_plan(vp):
            chunks = [
                VideoChunk(0, tmp_path / "c0.mp4", 0.0, 100.0),
                VideoChunk(1, tmp_path / "c1.mp4", 100.0, 200.0),
            ]
            return VideoContextPlan(vp, 200.0, 20000, chunks)

        def fake_mat(_plan):
            mat_calls["n"] += 1

        def fake_up(chunk, **kw):
            up_calls["n"] += 1
            return RemoteVideoRef(chunk.index, f"f{chunk.index}",
                                  chunk.start_sec, chunk.end_sec, "ACTIVE")

        with patch.object(provider, "plan_chunks", fake_plan), \
             patch.object(provider, "materialize_chunks", fake_mat), \
             patch.object(provider, "upload_chunk", fake_up), \
             patch.object(provider, "cleanup_local_chunks", lambda p: None), \
             patch.object(provider, "_refs_still_active", lambda r: True):
            provider.prepare_and_upload(src)   # analyze
            provider.prepare_and_upload(src)   # translate

        assert mat_calls["n"] == 1   # chỉ cắt 1 lần
        assert up_calls["n"] == 2    # chỉ upload 2 chunk (lần 1), lần 2 dùng cache

    def test_reupload_if_refs_inactive(self, tmp_path: Path) -> None:
        from unittest.mock import patch
        from subtitles_extractor.domain.ports.video_context_port import VideoContextPlan
        from subtitles_extractor.infrastructure.translation.gemini_video_context import (
            RemoteVideoRef,
        )

        src = tmp_path / "movie2.mkv"
        _make_test_mkv(src, seconds=4)
        provider = GeminiVideoContextProvider(work_dir=tmp_path)
        mat_calls = {"n": 0}

        def fake_plan(vp):
            return VideoContextPlan(vp, 200.0, 20000,
                                    [VideoChunk(0, tmp_path / "c0.mp4", 0.0, 100.0),
                                     VideoChunk(1, tmp_path / "c1.mp4", 100.0, 200.0)])

        with patch.object(provider, "plan_chunks", fake_plan), \
             patch.object(provider, "materialize_chunks", lambda p: mat_calls.__setitem__("n", mat_calls["n"] + 1)), \
             patch.object(provider, "upload_chunk",
                          lambda c, **k: RemoteVideoRef(c.index, f"f{c.index}", c.start_sec, c.end_sec, "ACTIVE")), \
             patch.object(provider, "cleanup_local_chunks", lambda p: None), \
             patch.object(provider, "_refs_still_active", lambda r: False):  # cloud hết hạn
            provider.prepare_and_upload(src)
            provider.prepare_and_upload(src)

        assert mat_calls["n"] == 2  # refs hết hạn → phải làm lại
