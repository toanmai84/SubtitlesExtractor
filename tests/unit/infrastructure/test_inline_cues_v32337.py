"""Test [v3.23.37] Visual Cues GỘP CHUNG vào phân tích ngữ cảnh (tiết kiệm quota)."""

from __future__ import annotations

from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    _CONTEXT_ANALYSIS_SCHEMA,
    _CONTEXT_WITH_CUES_SCHEMA,
    GeminiSubtitleTranslator,
)


class TestExtractCuesJson:
    def test_valid_cues(self) -> None:
        ext = GeminiSubtitleTranslator._extract_cues_json
        out = ext({"cues": [{"id": 1, "spk": "Lâm Côn", "to": "Sư phụ"},
                            {"id": 3, "spk": "Vương"}]})
        assert '"id":1' in out and "Lâm Côn" in out
        assert '"id":3' in out

    def test_drops_invalid(self) -> None:
        ext = GeminiSubtitleTranslator._extract_cues_json
        # id=0 và cue rỗng đều bị loại.
        assert ext({"cues": [{"id": 0, "spk": "X"}, {"id": 2, "spk": "", "to": ""}]}) == ""

    def test_no_cues_key(self) -> None:
        assert GeminiSubtitleTranslator._extract_cues_json({"characters": "A"}) == ""


class TestSchemaSelection:
    def test_cues_schema_has_cues_array(self) -> None:
        assert "cues" in _CONTEXT_WITH_CUES_SCHEMA["properties"]
        assert "cues" not in _CONTEXT_ANALYSIS_SCHEMA["properties"]

    def test_both_schemas_require_core_fields(self) -> None:
        for sch in (_CONTEXT_WITH_CUES_SCHEMA, _CONTEXT_ANALYSIS_SCHEMA):
            assert set(sch["required"]) == {"source_lang", "characters", "overview"}


class TestUseCaseInlineFlow:
    def test_analysis_requests_inline_cues(self) -> None:
        from subtitles_extractor.application.use_cases.analyze_subtitle_context import (
            AnalyzeSubtitleContextUseCase,
        )
        from subtitles_extractor.domain.ports.subtitle_translator_port import (
            SubtitleContextAnalysis,
            TranslationLine,
        )

        captured = {}

        class FakeT:
            def analyze_global_context(
                self, *, source_lines, target_lang, model_name,
                cancel_cb=None, video_refs=None, with_visual_cues=False,
                prior_context="",
            ):
                captured["with_visual_cues"] = with_visual_cues
                return SubtitleContextAnalysis(visual_cues='[{"id":1,"spk":"X"}]')

        uc = AnalyzeSubtitleContextUseCase(FakeT())
        lines = [TranslationLine(index=1, start_ms=0, end_ms=1000, text="Hi")]
        # Có video + bật cues → phải yêu cầu inline cues (KHÔNG gọi riêng).
        result = uc.execute(lines, "vi", video_refs=["ref"], enable_visual_cues=True)
        assert captured["with_visual_cues"] is True
        assert result.visual_cues  # cues lấy luôn từ phân tích

    def test_no_inline_when_disabled(self) -> None:
        from subtitles_extractor.application.use_cases.analyze_subtitle_context import (
            AnalyzeSubtitleContextUseCase,
        )
        from subtitles_extractor.domain.ports.subtitle_translator_port import (
            SubtitleContextAnalysis,
            TranslationLine,
        )

        captured = {}

        class FakeT:
            def analyze_global_context(
                self, *, source_lines, target_lang, model_name,
                cancel_cb=None, video_refs=None, with_visual_cues=False,
                prior_context="",
            ):
                captured["with_visual_cues"] = with_visual_cues
                return SubtitleContextAnalysis()

        uc = AnalyzeSubtitleContextUseCase(FakeT())
        lines = [TranslationLine(index=1, start_ms=0, end_ms=1000, text="Hi")]
        uc.execute(lines, "vi", video_refs=["ref"])  # không bật
        assert captured["with_visual_cues"] is False
