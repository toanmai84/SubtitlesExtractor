"""Test [v3.23.14] phân tích ngữ cảnh MAP-REDUCE: nhiều video → từng đoạn riêng
(≤1 video/request, tránh 400) rồi gộp."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
)
from subtitles_extractor.domain.ports.subtitle_translator_port import TranslationLine


class _Ref:
    def __init__(self, idx: int, start: float, end: float) -> None:
        self.chunk_index = idx
        self.remote_name = f"file{idx}"
        self.start_sec = start
        self.end_sec = end
        self.state = "ACTIVE"


def _make_adapter() -> GeminiSubtitleTranslator:
    adapter = GeminiSubtitleTranslator.__new__(GeminiSubtitleTranslator)
    adapter._types_module = MagicMock()
    return adapter


class TestContextMapReduce:
    def test_sequential_no_reduce_step(self) -> None:
        # [v3.23.23] Phân tích TUẦN TỰ: N đoạn = N request, KHÔNG có bước reduce thừa.
        adapter = _make_adapter()
        refs = [_Ref(0, 0, 2282), _Ref(1, 2282, 4565), _Ref(2, 4565, 6847)]
        video_counts: list[int] = []
        prompts: list[str] = []

        def fake_call(model, prompt, config, validator, cancel_cb=None,
                      video_files=None, est_tokens=0, video_refs=None):
            video_counts.append(len(video_files) if video_files else 0)
            prompts.append(prompt)
            return {"source_lang": "zh", "characters": "林昆 (Lâm Côn)",
                    "overview": f"đoạn {len(video_counts)}", "glossary": ""}

        lines = [TranslationLine(index=i, start_ms=0, end_ms=1, text=f"c{i}")
                 for i in range(1, 6)]
        with patch.object(adapter, "_resolve_video_handles", lambda r: ["h0", "h1", "h2"]), \
             patch.object(adapter, "_build_config", lambda *a, **k: MagicMock()), \
             patch.object(adapter, "_call_gemini", fake_call), \
             patch.object(adapter, "_ensure_available", lambda: None):
            result = adapter.analyze_global_context(lines, "vi", video_refs=refs)

        # MẤU CHỐT: không request nào có >1 video.
        assert all(n <= 1 for n in video_counts)
        # CHỈ 3 request (mỗi đoạn 1 video) — KHÔNG có request reduce (0 video) thừa.
        assert video_counts.count(1) == 3
        assert video_counts.count(0) == 0
        assert len(video_counts) == 3
        # Đoạn 2,3 phải chứa ngữ cảnh tích luỹ từ đoạn trước.
        assert "TÍCH LUỸ" in prompts[1]
        assert "TÍCH LUỸ" in prompts[2]
        # Kết quả cuối = phân tích đoạn cuối (đã hợp nhất).
        assert result.overview == "đoạn 3"

    def test_single_video_no_mapreduce(self) -> None:
        adapter = _make_adapter()
        refs = [_Ref(0, 0, 1000)]
        video_counts: list[int] = []

        def fake_call(model, prompt, config, validator, cancel_cb=None,
                      video_files=None, est_tokens=0, video_refs=None):
            video_counts.append(len(video_files) if video_files else 0)
            return {"source_lang": "en", "characters": "x", "overview": "y"}

        lines = [TranslationLine(index=1, start_ms=0, end_ms=1, text="hello")]
        with patch.object(adapter, "_resolve_video_handles", lambda r: ["h0"]), \
             patch.object(adapter, "_build_config", lambda *a, **k: MagicMock()), \
             patch.object(adapter, "_call_gemini", fake_call), \
             patch.object(adapter, "_ensure_available", lambda: None):
            adapter.analyze_global_context(lines, "vi", video_refs=refs)

        # 1 video → gửi trực tiếp 1 lần, không map-reduce.
        assert video_counts == [1]

    def test_merge_sequential_keeps_old_when_new_empty(self) -> None:
        # [v3.23.28] _merge_sequential_analysis: ưu tiên bản mới, giữ trường cũ nếu mới rỗng.
        from subtitles_extractor.domain.ports.subtitle_translator_port import (
            SubtitleContextAnalysis,
        )

        adapter = _make_adapter()
        prev = SubtitleContextAnalysis(
            source_lang="zh", characters="林昆", overview="đoạn 1", glossary="内功 => nội công")
        # Đoạn mới bỏ trống glossary → phải giữ glossary cũ.
        current = SubtitleContextAnalysis(
            source_lang="zh", characters="林昆, 王语嫣", overview="đoạn 1+2", glossary="")
        merged = adapter._merge_sequential_analysis(prev, current)
        assert merged.characters == "林昆, 王语嫣"   # bản mới
        assert merged.overview == "đoạn 1+2"        # bản mới
        assert merged.glossary == "内功 => nội công"  # giữ cũ vì mới rỗng

    def test_merge_sequential_none_prev(self) -> None:
        from subtitles_extractor.domain.ports.subtitle_translator_port import (
            SubtitleContextAnalysis,
        )

        adapter = _make_adapter()
        current = SubtitleContextAnalysis(source_lang="zh", characters="a", overview="o")
        assert adapter._merge_sequential_analysis(None, current) is current
