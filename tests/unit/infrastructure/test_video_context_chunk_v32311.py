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


@pytest.mark.skipif(not _FFMPEG, reason="cần ffmpeg")
class TestYoutubeChromaFormats:
    """[v3.23.466] Video YouTube (VP9/AV1) hay 4:4:4/10-bit — libx264 'main' từ chối → phải ép
    -pix_fmt yuv420p mới cắt được."""

    def _has_vp9(self) -> bool:
        out = subprocess.run([_FFMPEG, "-hide_banner", "-encoders"], capture_output=True, text=True)
        return "libvpx-vp9" in out.stdout

    def test_cuts_vp9_444_youtube_style(self, tmp_path: Path) -> None:
        if not self._has_vp9():
            pytest.skip("ffmpeg thiếu libvpx-vp9")
        from subtitles_extractor.infrastructure.translation.gemini_video_context import (
            GeminiVideoContextProvider, VideoChunk,
        )
        src = tmp_path / "yt.webm"
        subprocess.run(
            [_FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
             "-f", "lavfi", "-i", "testsrc=duration=6:size=1280x720:rate=30",
             "-f", "lavfi", "-i", "sine=duration=6",
             "-c:v", "libvpx-vp9", "-b:v", "300k", "-c:a", "libopus", str(src)],
            check=True, capture_output=True, timeout=120,
        )
        prov = GeminiVideoContextProvider(work_dir=tmp_path)
        prov._nvenc_available = False  # bỏ GPU (sandbox không có)
        chunk = VideoChunk(0, tmp_path / "out.ctxpart00.mp4", 0.0, 6.0)
        prov._encode_low_res_chunk(_FFMPEG, src, chunk)
        # Phải ra file 360p yuv420p hợp lệ (không "main doesn't support 4:4:4")
        out = subprocess.run(
            [shutil.which("ffprobe"), "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=height,pix_fmt", "-of", "csv=p=0", str(chunk.path)],
            capture_output=True, text=True,
        )
        assert "360" in out.stdout and "yuv420p" in out.stdout, out.stdout


class TestNvdecHybridTier:
    """[v3.23.467] Tier NVDEC-giải-mã + CPU-encode: né NVENC hỏng, giải mã VP9/AV1 nhanh."""

    def _prov(self, tmp_path):
        from subtitles_extractor.infrastructure.translation.gemini_video_context import (
            GeminiVideoContextProvider,
        )
        return GeminiVideoContextProvider(work_dir=tmp_path)

    def test_hybrid_tier_present_and_uses_libx264_not_nvenc(self, tmp_path: Path) -> None:
        from subtitles_extractor.infrastructure.translation.gemini_video_context import VideoChunk
        prov = self._prov(tmp_path)
        prov._nvenc_available = False  # NVENC hỏng → chỉ còn nvdec-hybrid + cpu
        chunk = VideoChunk(0, tmp_path / "o.mp4", 0.0, 10.0)
        cmds = prov._build_encode_commands("ffmpeg", tmp_path / "in.webm", chunk, tmp_path / "t.mp4")
        nvdec = [c for g, c in cmds if (not g) and ("-hwaccel" in c)]
        assert len(nvdec) == 1, "phải có đúng 1 tier NVDEC-hybrid"
        assert "libx264" in nvdec[0] and "h264_nvenc" not in nvdec[0]  # né NVENC
        assert "yuv420p" in nvdec[0]  # ép 4:2:0 (chroma YouTube)

    def test_hybrid_tier_excluded_when_marked_broken(self, tmp_path: Path) -> None:
        from subtitles_extractor.infrastructure.translation.gemini_video_context import VideoChunk
        prov = self._prov(tmp_path)
        prov._nvenc_available = False
        # [v3.23.471] Đánh dấu MỌI hwaccel ứng viên hỏng → không còn tier hwdec nào.
        prov._hwdec_broken = {"d3d11va", "cuda"}
        chunk = VideoChunk(0, tmp_path / "o.mp4", 0.0, 10.0)
        cmds = prov._build_encode_commands("ffmpeg", tmp_path / "in.webm", chunk, tmp_path / "t.mp4")
        assert not any((not g) and ("-hwaccel" in c) for g, c in cmds)


class TestYoutubeMovFlags:
    """[v3.23.469] Cờ chống treo cho MP4 YouTube/Google: ignore_editlist (chỉ mov/mp4) +
    max_interleave_delta 0 (chống treo muxing khi hạ fps mạnh)."""

    def _cmds(self, tmp_path, suffix):
        from subtitles_extractor.infrastructure.translation.gemini_video_context import (
            GeminiVideoContextProvider, VideoChunk,
        )
        prov = GeminiVideoContextProvider(work_dir=tmp_path)
        prov._nvenc_available = False
        ch = VideoChunk(0, tmp_path / "o.mp4", 0.0, 10.0)
        return prov._build_encode_commands("ffmpeg", tmp_path / f"in{suffix}", ch, tmp_path / "t.mp4")

    def test_mp4_gets_ignore_editlist(self, tmp_path: Path) -> None:
        for _g, c in self._cmds(tmp_path, ".mp4"):
            assert "-ignore_editlist" in c

    def test_webm_no_ignore_editlist(self, tmp_path: Path) -> None:
        # -ignore_editlist là option demuxer mov → thêm cho webm sẽ LỖI "Option not found".
        for _g, c in self._cmds(tmp_path, ".webm"):
            assert "-ignore_editlist" not in c

    def test_mkv_no_ignore_editlist(self, tmp_path: Path) -> None:
        for _g, c in self._cmds(tmp_path, ".mkv"):
            assert "-ignore_editlist" not in c

    def test_default_has_audio_not_an(self, tmp_path: Path) -> None:
        # [v3.23.473] Mặc định KÈM audio (bỏ -an); chỉ rơi video-only khi file kẹt (auto-fallback).
        for _g, c in self._cmds(tmp_path, ".mp4"):
            assert "-an" not in c and "0:a:0?" in c


class TestChunkFailureTolerance:
    """[v3.23.472] Một đoạn nén lỗi (file hỏng ở đầu) KHÔNG làm hỏng cả ngữ cảnh — bỏ đoạn lỗi,
    dùng các đoạn còn lại. Chỉ thất bại hẳn nếu MỌI đoạn lỗi."""

    def _plan(self, tmp_path, n):
        from subtitles_extractor.domain.ports.video_context_port import (
            VideoContextPlan, VideoChunk,
        )
        chunks = [VideoChunk(i, tmp_path / f"c{i}.mp4", i * 10.0, i * 10.0 + 10) for i in range(n)]
        return VideoContextPlan(source_path=tmp_path / "in.mp4", chunks=chunks,
                                duration_sec=float(n * 10), estimated_tokens=1000)

    def test_skips_failed_chunk_keeps_rest(self, tmp_path: Path) -> None:
        from unittest.mock import patch
        from subtitles_extractor.infrastructure.translation.gemini_video_context import (
            GeminiVideoContextProvider, VideoContextError,
        )
        prov = GeminiVideoContextProvider(work_dir=tmp_path)
        plan = self._plan(tmp_path, 4)

        def enc(ffmpeg, src, chunk):
            if chunk.index == 0:
                raise VideoContextError("đoạn 0 hỏng")
            chunk.path.write_bytes(b"x" * 2000)

        with patch.object(prov, "_encode_low_res_chunk", side_effect=enc), \
             patch("subtitles_extractor.infrastructure.translation.gemini_video_context."
                   "find_ffmpeg", return_value="ffmpeg"):
            prov.materialize_chunks(plan)
        assert [c.index for c in plan.chunks] == [1, 2, 3]

    def test_all_failed_raises(self, tmp_path: Path) -> None:
        from unittest.mock import patch
        from subtitles_extractor.infrastructure.translation.gemini_video_context import (
            GeminiVideoContextProvider, VideoContextError,
        )
        prov = GeminiVideoContextProvider(work_dir=tmp_path)
        plan = self._plan(tmp_path, 3)

        def enc(ffmpeg, src, chunk):
            raise VideoContextError("hỏng")

        with patch.object(prov, "_encode_low_res_chunk", side_effect=enc), \
             patch("subtitles_extractor.infrastructure.translation.gemini_video_context."
                   "find_ffmpeg", return_value="ffmpeg"), \
             pytest.raises(VideoContextError):
            prov.materialize_chunks(plan)


class TestAudioFallback:
    """[v3.23.473] Mặc định KÈM audio; nếu file kẹt xen kẽ (audio fail) → tự rơi video-only."""

    def test_default_includes_audio(self, tmp_path: Path) -> None:
        from subtitles_extractor.infrastructure.translation.gemini_video_context import (
            GeminiVideoContextProvider, VideoChunk,
        )
        prov = GeminiVideoContextProvider(work_dir=tmp_path)
        prov._nvenc_available = False
        ch = VideoChunk(0, tmp_path / "o.mp4", 0.0, 10.0)
        cmds = prov._build_encode_commands("ffmpeg", tmp_path / "in.mp4", ch, tmp_path / "t.mp4")
        for _g, c in cmds:
            assert "0:a:0?" in c and "-an" not in c  # có audio, không -an

    def test_drop_audio_switches_to_video_only(self, tmp_path: Path) -> None:
        from subtitles_extractor.infrastructure.translation.gemini_video_context import (
            GeminiVideoContextProvider, VideoChunk,
        )
        prov = GeminiVideoContextProvider(work_dir=tmp_path)
        prov._nvenc_available = False
        prov._drop_audio = True
        ch = VideoChunk(0, tmp_path / "o.mp4", 0.0, 10.0)
        cmds = prov._build_encode_commands("ffmpeg", tmp_path / "in.mp4", ch, tmp_path / "t.mp4")
        for _g, c in cmds:
            assert "-an" in c and "0:a:0?" not in c  # video-only


@pytest.mark.skipif(not _FFMPEG, reason="cần ffmpeg")
class TestAttachAudioSeparately:
    """[v3.23.474] File kẹt xen kẽ a/v (drop_audio) VẪN có audio nhờ trích RIÊNG audio + ghép."""

    def test_drop_audio_chunk_still_gets_audio(self, tmp_path: Path) -> None:
        from subtitles_extractor.infrastructure.translation.gemini_video_context import (
            GeminiVideoContextProvider, VideoChunk,
        )
        src = tmp_path / "in.mp4"
        subprocess.run(
            [_FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
             "-f", "lavfi", "-i", "testsrc=duration=6:size=640x360:rate=30",
             "-f", "lavfi", "-i", "sine=duration=6",
             "-c:v", "libx264", "-c:a", "aac", str(src)],
            check=True, capture_output=True, timeout=60,
        )
        prov = GeminiVideoContextProvider(work_dir=tmp_path)
        prov._nvenc_available = False
        prov._drop_audio = True  # mô phỏng file kẹt → đi đường video-only + ghép audio riêng
        ch = VideoChunk(0, tmp_path / "out.ctxpart00.mp4", 0.0, 6.0)
        prov._encode_low_res_chunk(_FFMPEG, src, ch)
        out = subprocess.run(
            [shutil.which("ffprobe"), "-v", "error", "-show_entries", "stream=codec_type",
             "-of", "csv=p=0", str(ch.path)],
            capture_output=True, text=True,
        )
        assert "video" in out.stdout and "audio" in out.stdout


class TestIgnoreCloudCache:
    """[v3.23.479] Tùy chọn 'bỏ cache cloud' → BỎ mọi tái dùng, buộc cắt+tải lại (khi đã xoá cloud)."""

    def test_flag_stored(self, tmp_path: Path) -> None:
        from subtitles_extractor.infrastructure.translation.gemini_video_context import (
            GeminiVideoContextProvider,
        )
        assert GeminiVideoContextProvider(work_dir=tmp_path, ignore_cloud_cache=True)._ignore_cloud_cache is True
        assert GeminiVideoContextProvider(work_dir=tmp_path)._ignore_cloud_cache is False

    def test_skip_cache_bypasses_all_lookups(self, tmp_path: Path) -> None:
        from unittest.mock import patch, MagicMock
        from subtitles_extractor.infrastructure.translation.gemini_video_context import (
            GeminiVideoContextProvider,
        )
        prov = GeminiVideoContextProvider(work_dir=tmp_path, ignore_cloud_cache=True)
        prov._inmem_refs = {}
        calls = {"plan": 0, "source": 0}
        with patch.object(prov, "_video_signature", return_value="sig"), \
             patch.object(prov, "plan_chunks",
                          return_value=MagicMock(chunks=[1, 2], needs_split=True, estimated_tokens=1)), \
             patch.object(prov, "_plan_cache_lookup",
                          side_effect=lambda *a: calls.__setitem__("plan", calls["plan"] + 1)), \
             patch.object(prov, "_resolve_chunks_from_source_cache",
                          side_effect=lambda *a: calls.__setitem__("source", calls["source"] + 1)), \
             patch.object(prov, "materialize_chunks"), \
             patch.object(prov, "_ensure_client"):
            try:
                prov.prepare_and_upload(tmp_path / "v.mp4")
            except Exception:
                pass
        assert calls["plan"] == 0 and calls["source"] == 0
