"""Unit test cho các sửa lỗi hạ tầng đám mây video (v3.14.1, Nhóm 1).

Phủ: cô lập cache theo API key, tên ASCII an toàn (hardlink), nhận diện lỗi cloud
tự chữa, và Smart Video Truncation (chọn đoạn đại diện).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from subtitles_extractor.infrastructure.translation.gemini_video_context import (
    GeminiVideoContextProvider,
)


class TestApiKeyIsolation:
    def test_cache_key_differs_per_api_key(self) -> None:
        provider_a = GeminiVideoContextProvider(api_key="KEY_A")
        provider_b = GeminiVideoContextProvider(api_key="KEY_B")
        file_hash = "abc123" * 5
        assert provider_a._cache_key(file_hash) != provider_b._cache_key(file_hash)

    def test_cache_key_stable_same_key(self) -> None:
        provider = GeminiVideoContextProvider(api_key="KEY_A")
        file_hash = "deadbeef" * 4
        assert provider._cache_key(file_hash) == provider._cache_key(file_hash)


class TestAsciiSafeSource:
    def test_cjk_filename_becomes_ascii(self, tmp_path: Path) -> None:
        cjk_video = tmp_path / "玄幻修仙_第一集.mp4"
        cjk_video.write_bytes(b"FAKE_VIDEO_CONTENT")
        safe_path, is_temp = GeminiVideoContextProvider._make_ascii_safe_source(
            cjk_video, "a1b2c3d4e5", 0
        )
        try:
            assert safe_path.name.isascii()
            assert safe_path.exists()
            assert safe_path.read_bytes() == b"FAKE_VIDEO_CONTENT"
            assert is_temp is True
        finally:
            if is_temp and safe_path != cjk_video:
                safe_path.unlink(missing_ok=True)

    def test_ascii_name_pattern(self, tmp_path: Path) -> None:
        video = tmp_path / "视频.mp4"
        video.write_bytes(b"x")
        safe_path, is_temp = GeminiVideoContextProvider._make_ascii_safe_source(
            video, "ffeeddcc", 3
        )
        try:
            assert safe_path.name.startswith("video_ffeeddcc_part03")
        finally:
            if is_temp and safe_path != video:
                safe_path.unlink(missing_ok=True)


class TestCloudHealing:
    @pytest.mark.parametrize(
        "message",
        ["404 NOT_FOUND", "PERMISSION_DENIED", "File was not found", "403 Forbidden",
         "resource has expired"],
    )
    def test_recoverable_errors(self, message: str) -> None:
        assert GeminiVideoContextProvider._is_recoverable_remote_error(Exception(message))

    @pytest.mark.parametrize("message", ["500 internal server error", "timeout", "quota"])
    def test_non_recoverable_errors(self, message: str) -> None:
        assert not GeminiVideoContextProvider._is_recoverable_remote_error(Exception(message))


class TestSmartTruncation:
    def test_select_representative_includes_endpoints(self) -> None:
        indices = GeminiVideoContextProvider._select_representative_indices(100, 3)
        assert indices[0] == 0
        assert indices[-1] == 99
        assert len(indices) == 3

    def test_select_keeps_all_when_budget_allows(self) -> None:
        indices = GeminiVideoContextProvider._select_representative_indices(5, 10)
        assert indices == [0, 1, 2, 3, 4]

    def test_select_single_when_tiny_budget(self) -> None:
        assert GeminiVideoContextProvider._select_representative_indices(50, 1) == [0]

    def test_plan_truncates_long_video(self, tmp_path: Path, monkeypatch) -> None:
        video = tmp_path / "long_movie.mp4"
        video.write_bytes(b"x")
        # Phim 80 tập giả lập: 8 giờ → ~7.5 triệu token, vượt xa trần 600K.
        monkeypatch.setattr(
            GeminiVideoContextProvider, "_ffprobe_duration_sec",
            staticmethod(lambda _p: 8 * 3600.0),
        )
        provider = GeminiVideoContextProvider(
            max_tokens_per_chunk=300_000, max_total_tokens=600_000, work_dir=tmp_path
        )
        # [v3.23.38] Truncation giờ là OPT-IN; mặc định giữ hết đoạn (phân tích tuần tự).
        plan = provider.plan_chunks(video, allow_truncation=True)
        assert plan.is_truncated is True
        from subtitles_extractor.infrastructure.translation.gemini_video_context import (
            _TOKENS_PER_SECOND,
        )

        kept_tokens = sum(c.duration_sec for c in plan.chunks) * _TOKENS_PER_SECOND
        assert kept_tokens <= 600_000 * 1.05  # trong trần (biên làm tròn nhỏ)
        # Đoạn đầu phải bắt đầu ~0, đoạn cuối phải gần hết phim (Đầu/Giữa/Cuối).
        assert plan.chunks[0].start_sec < 60.0
        assert plan.chunks[-1].end_sec > 8 * 3600.0 - 2000.0

    def test_default_keeps_all_chunks_long_video(self, tmp_path: Path, monkeypatch) -> None:
        # [v3.23.38] Mặc định (không truncate): giữ TẤT CẢ đoạn, phủ kín toàn phim.
        video = tmp_path / "long_movie.mp4"
        video.write_bytes(b"x")
        monkeypatch.setattr(
            GeminiVideoContextProvider, "_ffprobe_duration_sec",
            staticmethod(lambda _p: 8 * 3600.0),
        )
        provider = GeminiVideoContextProvider(
            max_tokens_per_chunk=300_000, max_total_tokens=600_000, work_dir=tmp_path
        )
        plan = provider.plan_chunks(video)  # mặc định
        assert plan.is_truncated is False
        total = sum(c.duration_sec for c in plan.chunks)
        assert abs(total - 8 * 3600.0) < 1.0  # phủ kín, không thiếu

    def test_short_video_not_truncated(self, tmp_path: Path, monkeypatch) -> None:
        video = tmp_path / "short.mp4"
        video.write_bytes(b"x")
        monkeypatch.setattr(
            GeminiVideoContextProvider, "_ffprobe_duration_sec",
            staticmethod(lambda _p: 120.0),
        )
        plan = GeminiVideoContextProvider(work_dir=tmp_path).plan_chunks(video)
        assert plan.is_truncated is False
        assert len(plan.chunks) == 1
        assert plan.chunks[0].is_full_video is True
