"""Test [v3.23.21] cache plan video bền vững (DB) + nén GPU NVENC."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from subtitles_extractor.domain.ports.video_context_port import VideoChunk
from subtitles_extractor.infrastructure.translation.gemini_video_context import (
    GeminiVideoContextProvider,
    RemoteVideoRef,
)


@pytest.fixture()
def provider(tmp_path: Path) -> GeminiVideoContextProvider:
    return GeminiVideoContextProvider(cache_db_path=tmp_path / "cache.db", work_dir=tmp_path)


class TestPlanCachePersistent:
    def test_roundtrip(self, provider: GeminiVideoContextProvider) -> None:
        refs = [RemoteVideoRef(0, "files/a", 0, 1610, "ACTIVE"),
                RemoteVideoRef(1, "files/b", 1610, 3220, "ACTIVE")]
        provider._plan_cache_store("sig1", refs)
        got = provider._plan_cache_lookup("sig1", 2)
        assert got is not None
        assert [r.remote_name for r in got] == ["files/a", "files/b"]

    def test_chunk_count_mismatch_returns_none(self, provider) -> None:
        provider._plan_cache_store("sig1", [RemoteVideoRef(0, "files/a", 0, 1, "ACTIVE")])
        assert provider._plan_cache_lookup("sig1", 3) is None

    def test_unknown_signature_returns_none(self, provider) -> None:
        assert provider._plan_cache_lookup("nope", 2) is None

    def test_store_replaces_existing(self, provider) -> None:
        provider._plan_cache_store("sig", [RemoteVideoRef(0, "old", 0, 1, "ACTIVE")])
        provider._plan_cache_store("sig", [RemoteVideoRef(0, "new", 0, 1, "ACTIVE")])
        got = provider._plan_cache_lookup("sig", 1)
        assert got[0].remote_name == "new"

    def test_prepare_skips_materialize_on_cache_hit(self, provider, tmp_path) -> None:
        refs = [RemoteVideoRef(0, "files/x", 0, 1610, "ACTIVE"),
                RemoteVideoRef(1, "files/y", 1610, 3220, "ACTIVE")]
        vsig = provider._video_signature(Path("movie.mkv"))
        provider._plan_cache_store(vsig, refs)

        class FakePlan:
            chunks = [VideoChunk(0, tmp_path / "a", 0, 1610, False),
                      VideoChunk(1, tmp_path / "b", 1610, 3220, False)]
            needs_split = True
            estimated_tokens = 300_000

        materialized = [False]
        with patch.object(provider, "plan_chunks", lambda p: FakePlan()), \
             patch.object(provider, "_refs_still_active", lambda r: True), \
             patch.object(provider, "materialize_chunks",
                          lambda p: materialized.__setitem__(0, True)), \
             patch.object(provider, "upload_chunk",
                          lambda c: RemoteVideoRef(c.index, "NEW", 0, 1, "ACTIVE")):
            result = provider.prepare_and_upload(Path("movie.mkv"))
        assert materialized[0] is False  # KHÔNG cắt+nén lại
        assert [r.remote_name for r in result] == ["files/x", "files/y"]


class TestNvencEncoding:
    def test_detect_nvenc_present(self, provider) -> None:
        def fake_run(cmd, **kw):
            m = MagicMock()
            m.stdout = "... h264_nvenc ..." if "-encoders" in cmd else ""
            return m
        with patch("subprocess.run", fake_run):
            assert provider._detect_nvenc("ffmpeg") is True
            assert provider._nvenc_available is True  # cached

    def test_detect_nvenc_absent(self, provider) -> None:
        def fake_run(cmd, **kw):
            m = MagicMock()
            m.stdout = "libx264 libx265"
            return m
        with patch("subprocess.run", fake_run):
            assert provider._detect_nvenc("ffmpeg") is False

    def test_build_commands_gpu_first(self, provider, tmp_path) -> None:
        provider._nvenc_available = True
        chunk = VideoChunk(0, tmp_path / "o.mp4", 0.0, 100.0, False)
        cmds = provider._build_encode_commands("ffmpeg", Path("s.mkv"), chunk, tmp_path / "t.mp4")
        # [v3.23.159] 5 cấp: GPU hoàn toàn → GPU qua upload → GPU giải mã+scale CPU
        # → NVENC-only → CPU.
        assert len(cmds) == 5
        # Cấp 1: GPU hoàn toàn (hwaccel cuda + scale_cuda + nvenc).
        assert cmds[0][0] is True
        assert "-hwaccel" in cmds[0][1] and "cuda" in cmds[0][1]
        assert any("scale_cuda" in c for c in cmds[0][1])
        assert "h264_nvenc" in cmds[0][1]
        # Cấp 1.25: GPU qua upload (hwaccel cuda, KHÔNG output_format, có hwupload).
        assert cmds[1][0] is True and "h264_nvenc" in cmds[1][1]
        assert any("hwupload_cuda" in c for c in cmds[1][1])
        assert "-hwaccel_output_format" not in cmds[1][1]
        # Cấp 1.5: GPU giải mã (hwaccel cuda) + scale CPU + nvenc (KHÔNG scale_cuda).
        assert cmds[2][0] is True and "h264_nvenc" in cmds[2][1]
        assert "-hwaccel" in cmds[2][1] and not any("scale_cuda" in c for c in cmds[2][1])
        # Cấp 2: NVENC-only (không hwaccel).
        assert cmds[3][0] is True and "h264_nvenc" in cmds[3][1]
        assert "-hwaccel" not in cmds[3][1]
        # Cấp 3: CPU.
        assert cmds[4][0] is False and "libx264" in cmds[4][1]

    def test_build_commands_cpu_only(self, provider, tmp_path) -> None:
        provider._nvenc_available = False
        chunk = VideoChunk(0, tmp_path / "o.mp4", 0.0, 100.0, False)
        cmds = provider._build_encode_commands("ffmpeg", Path("s.mkv"), chunk, tmp_path / "t.mp4")
        assert len(cmds) == 1
        assert cmds[0][0] is False


class TestEncodeModeLabel:
    def test_full_gpu_label(self, provider, tmp_path) -> None:
        provider._nvenc_available = True
        chunk = VideoChunk(0, tmp_path / "o.mp4", 0.0, 100.0, False)
        cmds = provider._build_encode_commands("ffmpeg", Path("s.mkv"), chunk, tmp_path / "t.mp4")
        assert "GPU hoàn toàn" in provider._encode_mode_label(cmds[0][1])
        assert "GPU qua upload" in provider._encode_mode_label(cmds[1][1])
        assert "scale CPU" in provider._encode_mode_label(cmds[2][1])
        assert "GPU một phần" in provider._encode_mode_label(cmds[3][1])
        assert "CPU" in provider._encode_mode_label(cmds[4][1])


class TestGpuHangFix:
    def test_gpu_commands_have_vsync_genpts(self, provider, tmp_path) -> None:
        # [v3.23.30] Lệnh GPU phải có -fflags +genpts + max_muxing_queue (chống treo).
        # KHÔNG dùng -vsync (deprecated, không tương thích ffmpeg cũ).
        provider._nvenc_available = True
        chunk = VideoChunk(0, tmp_path / "o.mp4", 0.0, 1075.0, False)
        cmds = provider._build_encode_commands("ffmpeg", Path("s.mkv"), chunk, tmp_path / "t.mp4")
        for use_gpu, cmd in cmds:
            assert "-max_muxing_queue_size" in cmd
            assert "-vsync" not in cmd  # đã gỡ (deprecated)
            if use_gpu:
                assert "+genpts" in cmd

    def test_gpu_timeout_triggers_fallback(self, provider, tmp_path, monkeypatch) -> None:
        # GPU treo (TimeoutExpired) → fallback sang CPU, KHÔNG raise.
        import subprocess as sp
        provider._nvenc_available = True
        chunk = VideoChunk(0, tmp_path / "o.mp4", 0.0, 100.0, False)

        calls = {"n": 0}

        def fake_run(cmd, **kwargs):
            calls["n"] += 1
            is_gpu = "h264_nvenc" in cmd
            if is_gpu:
                raise sp.TimeoutExpired(cmd, kwargs.get("timeout", 0))
            # CPU: tạo file giả hợp lệ.
            out = Path(cmd[-1])
            out.write_bytes(b"x" * 2048)
            return sp.CompletedProcess(cmd, 0)

        monkeypatch.setattr(sp, "run", fake_run)
        # Không raise = thành công fallback.
        provider._encode_low_res_chunk("ffmpeg", Path("s.mkv"), chunk)
        # GPU bị tắt sau khi mọi lệnh GPU treo.
        assert provider._nvenc_available is False


class TestGpuPartialFallback:
    def test_extra_hw_frames_in_gpu_full(self, provider, tmp_path) -> None:
        # [v3.23.33] GPU full phải có -extra_hw_frames (chống "No decoder surfaces left").
        provider._nvenc_available = True
        chunk = VideoChunk(0, tmp_path / "o.mp4", 0.0, 816.0, False)
        cmds = provider._build_encode_commands("ffmpeg", Path("s.mkv"), chunk, tmp_path / "t.mp4")
        gpu_full = cmds[0][1]
        assert "-extra_hw_frames" in gpu_full
        # [v3.23.159] Cấp 1 thuần-GPU: fps ĐẦU + scale_cuda, KHÔNG hwupload (legacy
        # hwupload_cuda chặn frame CUDA). Cấp 1.25 kế tiếp mới là đường upload.
        vf_value = gpu_full[gpu_full.index("-vf") + 1]
        assert vf_value.startswith("fps=")
        assert "hwupload" not in vf_value
        assert "scale_cuda=w=-2:h=360:format=nv12" in vf_value
        assert "hwdownload" in vf_value  # [v3.23.167] kéo frame VRAM->RAM
        gpu_upload = cmds[1][1]
        vf_upload = gpu_upload[gpu_upload.index("-vf") + 1]
        assert "hwupload_cuda" in vf_upload
        assert "scale_cuda" in vf_upload
        assert "-hwaccel_output_format" not in gpu_upload  # frame về RAM rồi upload

    def test_keep_gpu_when_nvenc_only_works(self, provider, tmp_path, monkeypatch) -> None:
        # Cấp 1 (scale_cuda) lỗi nhưng cấp 2 (NVENC-only) chạy được → KHÔNG tắt GPU.
        import subprocess as sp
        provider._nvenc_available = True
        chunk = VideoChunk(0, tmp_path / "o.mp4", 0.0, 816.0, False)

        def fake_run(cmd, **kwargs):
            is_scale_cuda = any("scale_cuda" in str(x) for x in cmd)
            if is_scale_cuda:
                raise sp.CalledProcessError(-40, cmd)  # cấp 1 crash
            # cấp 2 NVENC-only hoặc CPU: tạo file giả.
            Path(cmd[-1]).write_bytes(b"x" * 2048)
            return sp.CompletedProcess(cmd, 0)

        monkeypatch.setattr(sp, "run", fake_run)
        provider._encode_low_res_chunk("ffmpeg", Path("s.mkv"), chunk)
        # NVENC vẫn dùng được (cấp 2) → KHÔNG bị tắt.
        assert provider._nvenc_available is True

    def test_disable_gpu_only_when_all_gpu_fail(self, provider, tmp_path, monkeypatch) -> None:
        # Mọi cấp GPU lỗi, chỉ CPU chạy → tắt GPU.
        import subprocess as sp
        provider._nvenc_available = True
        chunk = VideoChunk(0, tmp_path / "o.mp4", 0.0, 816.0, False)

        def fake_run(cmd, **kwargs):
            is_nvenc = "h264_nvenc" in cmd
            if is_nvenc:
                raise sp.CalledProcessError(-40, cmd)  # mọi GPU lỗi
            Path(cmd[-1]).write_bytes(b"x" * 2048)  # CPU OK
            return sp.CompletedProcess(cmd, 0)

        monkeypatch.setattr(sp, "run", fake_run)
        provider._encode_low_res_chunk("ffmpeg", Path("s.mkv"), chunk)
        assert provider._nvenc_available is False
