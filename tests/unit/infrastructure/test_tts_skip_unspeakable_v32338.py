"""Test [v3.23.38] bỏ qua văn bản không đọc được + plan giữ hết đoạn."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from subtitles_extractor.infrastructure.tts.edge_tts_adapter import _has_speakable_content
from subtitles_extractor.infrastructure.translation.gemini_video_context import (
    GeminiVideoContextProvider,
)


class TestSpeakableContent:
    def test_speakable_true(self) -> None:
        for t in ("(nói tiếng Quan Thoại)", "Xin chào", "林昆", "123", "[âm thanh] vo ve"):
            assert _has_speakable_content(t) is True, t

    def test_speakable_false(self) -> None:
        for t in ("♪♪", "♪", "...", "!!!", "—", "   ", "(...)"):
            assert _has_speakable_content(t) is False, t


class TestPlanKeepsAllChunks:
    def _provider(self, tmp_path):
        return GeminiVideoContextProvider(
            cache_db_path=tmp_path / "c.db", work_dir=tmp_path,
            max_tokens_per_chunk=300_000, max_total_tokens=950_000,
        )

    def test_default_keeps_all_chunks(self, tmp_path) -> None:
        prov = self._provider(tmp_path)
        fake = tmp_path / "v.mp4"
        fake.write_bytes(b"x")
        with patch.object(prov, "_ffprobe_duration_sec", return_value=11901.0):
            plan = prov.plan_chunks(fake)  # mặc định KHÔNG truncate
        assert plan.is_truncated is False
        # Các đoạn phủ KÍN toàn bộ thời lượng (không thiếu giây nào).
        total = sum(c.duration_sec for c in plan.chunks)
        assert abs(total - 11901.0) < 1.0
        assert len(plan.chunks) >= 4

    def test_truncation_opt_in(self, tmp_path) -> None:
        prov = self._provider(tmp_path)
        fake = tmp_path / "v.mp4"
        fake.write_bytes(b"x")
        with patch.object(prov, "_ffprobe_duration_sec", return_value=11901.0):
            full = prov.plan_chunks(fake)
            truncated = prov.plan_chunks(fake, allow_truncation=True)
        # allow_truncation giữ ÍT đoạn hơn (hoặc bằng nếu vừa token).
        assert len(truncated.chunks) <= len(full.chunks)

    def test_short_video_single_chunk(self, tmp_path) -> None:
        prov = self._provider(tmp_path)
        fake = tmp_path / "v.mp4"
        fake.write_bytes(b"x")
        with patch.object(prov, "_ffprobe_duration_sec", return_value=600.0):
            plan = prov.plan_chunks(fake)
        assert len(plan.chunks) == 1
        assert plan.chunks[0].is_full_video is True
