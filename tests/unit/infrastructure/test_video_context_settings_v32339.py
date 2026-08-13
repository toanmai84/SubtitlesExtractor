"""Test [v3.23.39] nhóm cài đặt Video ngữ cảnh + Dịch + áp vào provider."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from subtitles_extractor.infrastructure.settings.application_settings import (
    ApplicationSettings,
    TranslationSettings,
    VideoContextSettings,
)
from subtitles_extractor.infrastructure.translation.gemini_video_context import (
    GeminiVideoContextProvider,
)


class TestSettingsModel:
    def test_defaults_match_legacy_hardcode(self) -> None:
        # Mặc định phải khớp giá trị hardcode cũ (tương thích ngược).
        vc = VideoContextSettings()
        assert vc.resolution_height == 360
        assert vc.fps == 1.0
        assert vc.nvenc_cq == 32
        assert vc.cpu_crf == 30
        # [v3.23.141] tokens_per_chunk hạ 300K -> 230K (< TPM free-tier 250K) để mỗi đoạn
        # không tự vượt TPM ở medium/high. Thay đổi CÓ CHỦ ĐÍCH, không phải hardcode cũ.
        assert vc.tokens_per_chunk == 230_000
        assert vc.tokens_per_second == 100
        tr = TranslationSettings()
        assert tr.default_batch_size == 50
        assert tr.retry_count == 3
        assert tr.request_timeout_sec == 120

    def test_in_application_settings(self) -> None:
        s = ApplicationSettings()
        assert s.video_context.resolution_height == 360
        assert s.translation.request_timeout_sec == 120

    def test_validation_bounds(self) -> None:
        import pytest
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            VideoContextSettings(resolution_height=50)  # < 144
        with pytest.raises(ValidationError):
            TranslationSettings(retry_count=0)  # < 1


class TestProviderUsesSettings:
    def test_resolution_fps_applied_to_encode(self, tmp_path) -> None:
        prov = GeminiVideoContextProvider(
            cache_db_path=tmp_path / "c.db", work_dir=tmp_path,
            resolution_height=480, fps=2.0, nvenc_cq=28, cpu_crf=26,
        )
        prov._nvenc_available = True
        from subtitles_extractor.domain.ports.video_context_port import VideoChunk
        chunk = VideoChunk(0, tmp_path / "o.mp4", 0.0, 100.0, False)
        cmds = prov._build_encode_commands("ffmpeg", Path("s.mkv"), chunk, tmp_path / "t.mp4")
        gpu_full = " ".join(cmds[0][1])
        assert "h=480" in gpu_full          # độ phân giải áp dụng
        assert "-cq 28" in gpu_full          # CQ áp dụng
        cpu = " ".join(cmds[-1][1])
        assert "scale=-2:480" in cpu
        assert "-crf 26" in cpu              # CRF áp dụng
        assert "fps=2" in cpu                # fps áp dụng

    def test_tokens_per_second_affects_chunking(self, tmp_path) -> None:
        # tokens_per_second cao hơn → mỗi đoạn ngắn hơn (nhiều đoạn hơn).
        fake = tmp_path / "v.mp4"
        fake.write_bytes(b"x")
        prov_lo = GeminiVideoContextProvider(
            cache_db_path=tmp_path / "a.db", work_dir=tmp_path,
            max_tokens_per_chunk=300_000, tokens_per_second=100, max_chunk_minutes=60.0,
        )
        prov_hi = GeminiVideoContextProvider(
            cache_db_path=tmp_path / "b.db", work_dir=tmp_path,
            max_tokens_per_chunk=300_000, tokens_per_second=300, max_chunk_minutes=60.0,
        )
        with patch.object(prov_lo, "_ffprobe_duration_sec", return_value=11901.0), \
             patch.object(prov_hi, "_ffprobe_duration_sec", return_value=11901.0):
            plan_lo = prov_lo.plan_chunks(fake)
            plan_hi = prov_hi.plan_chunks(fake)
        assert len(plan_hi.chunks) > len(plan_lo.chunks)
