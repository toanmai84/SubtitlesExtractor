"""Test các sửa lỗi critical + cải tiến (v3.14.8): #3 TTS srt, sibling video."""

from __future__ import annotations

from pathlib import Path

from subtitles_extractor.application.services.sibling_video_finder import (
    find_sibling_video,
)
from subtitles_extractor.application.use_cases.generate_tts import GenerateTTSUseCase
from subtitles_extractor.domain.ports.subtitle_tts_port import TTSRequest, TTSSegmentResult


class TestSiblingVideoFinder:
    def test_exact_name_match(self, tmp_path: Path) -> None:
        (tmp_path / "phim.mp4").write_bytes(b"x")
        srt = tmp_path / "phim.srt"; srt.write_text("")
        assert find_sibling_video(srt) == tmp_path / "phim.mp4"

    def test_strips_subtitle_suffix(self, tmp_path: Path) -> None:
        (tmp_path / "phim.mkv").write_bytes(b"x")
        srt = tmp_path / "phim.translate.vi.srt"; srt.write_text("")
        assert find_sibling_video(srt) == tmp_path / "phim.mkv"

    def test_original_suffix(self, tmp_path: Path) -> None:
        (tmp_path / "movie 玄幻.mp4").write_bytes(b"x")
        srt = tmp_path / "movie 玄幻.original.srt"; srt.write_text("")
        assert find_sibling_video(srt) == tmp_path / "movie 玄幻.mp4"

    def test_none_when_no_video(self, tmp_path: Path) -> None:
        srt = tmp_path / "phim.srt"; srt.write_text("")
        assert find_sibling_video(srt) is None

    def test_extension_priority(self, tmp_path: Path) -> None:
        (tmp_path / "phim.mp4").write_bytes(b"x")
        (tmp_path / "phim.mkv").write_bytes(b"x")
        # mp4 đứng trước mkv trong danh sách ưu tiên.
        assert find_sibling_video(tmp_path / "phim.srt") == tmp_path / "phim.mp4"


class TestTtsExportSrtNoNameError:
    """[#3] _export_adjusted_srt nhận request → không còn NameError."""

    def test_export_with_request_language(self, tmp_path: Path) -> None:
        wav = tmp_path / "movie.wav"
        results = [
            TTSSegmentResult(
                event_index=1, start_sec=0.0, end_sec=1.0, text="Xin chào",
                adjusted_start_sec=0.0, adjusted_end_sec=1.2,
            ),
            TTSSegmentResult(
                event_index=2, start_sec=1.5, end_sec=2.5, text="Tạm biệt",
                adjusted_start_sec=1.5, adjusted_end_sec=2.6,
            ),
        ]
        request = TTSRequest(events=[], language="vi-VN", speaker="x")
        # Không được ném NameError; phải tạo file .tts.vi.srt cạnh wav.
        srt_path = GenerateTTSUseCase._export_adjusted_srt(results, wav, request)
        assert srt_path is not None
        assert srt_path.name == "movie.tts.vi.srt"
        assert srt_path.exists()
