"""[v3.23.77] Hồi quy: file SRT xuất kèm audio TTS phải GIỮ tag người nói "[Tên:]".

Bug trước đây: ``_build_results_from_map`` gán ``TTSSegmentResult.text`` bằng văn bản ĐÃ
bỏ tag (để giọng đọc không đọc tên), và ``_export_adjusted_srt`` dùng chính ``text`` đó →
file phụ đề mất sạch tag người nói. Sửa bằng trường ``subtitle_text`` (giữ tag).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from subtitles_extractor.application.use_cases.generate_tts import GenerateTTSUseCase
from subtitles_extractor.domain.ports.subtitle_tts_port import TTSSegmentResult


def _export(results: list[TTSSegmentResult], tmp_path: Path) -> str:
    request = SimpleNamespace(language="vi")
    wav_path = tmp_path / "movie.flac"
    srt_path = GenerateTTSUseCase._export_adjusted_srt(results, wav_path, request)
    assert srt_path is not None and srt_path.exists()
    return srt_path.read_text(encoding="utf-8")


def test_srt_keeps_speaker_tag_for_processed_line(tmp_path: Path) -> None:
    results = [
        TTSSegmentResult(
            event_index=0, start_sec=0.0, end_sec=2.0,
            text="Xin chào mọi người",  # text ĐỌC (đã bỏ tag)
            subtitle_text="[Lâm Hằng:] Xin chào mọi người",  # text PHỤ ĐỀ (giữ tag)
            adjusted_start_sec=0.10, adjusted_end_sec=2.0,
        ),
    ]
    content = _export(results, tmp_path)
    assert "[Lâm Hằng:]" in content
    assert "Xin chào mọi người" in content


def test_srt_falls_back_to_text_when_subtitle_text_empty(tmp_path: Path) -> None:
    # Dòng mô tả âm thanh (skipped) vốn đã chứa tag trong ``text`` (subtitle_text rỗng).
    # Cần ít nhất một dòng thường (không skipped) để kích hoạt xuất SRT.
    results = [
        TTSSegmentResult(
            event_index=0, start_sec=0.0, end_sec=1.5,
            text="Một câu thoại", subtitle_text="[An:] Một câu thoại",
            adjusted_start_sec=0.0, adjusted_end_sec=1.5,
        ),
        TTSSegmentResult(
            event_index=1, start_sec=2.0, end_sec=3.5,
            text="[Hệ thống:] (tiếng còi báo động)",
            subtitle_text="",
            was_skipped=True,
            adjusted_start_sec=2.0, adjusted_end_sec=3.5,
        ),
    ]
    content = _export(results, tmp_path)
    assert "[An:]" in content  # subtitle_text (giữ tag)
    assert "[Hệ thống:]" in content  # fallback về text (cũng có tag)
