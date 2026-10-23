"""[v3.23.612] Tuỳ chọn xưng hô thuần Việt đi tới đủ nơi: cài đặt, container, prompt
phân tích + dịch, khoá tiến độ phân tích, use case phân tích, bước cuối của lượt dịch."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from subtitles_extractor.application.services.translation_process_log import (
    get_process_log,
)
from subtitles_extractor.application.use_cases.analyze_subtitle_context import (
    AnalyzeSubtitleContextUseCase,
)
from subtitles_extractor.application.use_cases.translate_subtitles import (
    TranslateSubtitlesRequest,
    TranslateSubtitlesUseCase,
)
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    SubtitleContextAnalysis,
    TranslationContext,
    TranslationStageConfig,
    TranslationStageKind,
)
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval
from subtitles_extractor.infrastructure.settings.application_settings import (
    TranslationSettings,
)
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    NATIVE_ADDRESS_ANALYSIS_RULE,
    NATIVE_ADDRESS_TRANSLATION_RULE,
    GeminiSubtitleTranslator,
)
from tests.unit.application.test_native_address_v323612 import KIEM, ROSTER, TU
from tests.unit.infrastructure.gemini_call_harness import bare_analysis_adapter

# Dòng 360, 621, 837 của lượt thật 13/10 (câu gốc → bản dịch thô thật).
SOURCES = ["劍哥", "我還進去看看劍哥", "洪大嫂我不想吃"]
LITERAL = {
    "劍哥": "Kiếm ca.",
    "我還進去看看劍哥": "Tôi vào xem Kiếm ca thế nào đã.",
    "洪大嫂我不想吃": "Hồng đại tẩu, tôi không muốn ăn.",
}
CUES = json.dumps([
    {"id": 1, "spk": "秦可秀", "to": "甄劍聲"},
    {"id": 2, "spk": "秦可秀", "to": "朱正軍"},
    {"id": 3, "spk": "秦可秀", "to": "洪大嫂"},
], ensure_ascii=False)


class _Literal:
    """Bộ dịch giả: trả bản dịch thô thật theo câu gốc."""

    def translate_stage(self, *, input_lines: list[Any], **_kw: Any) -> list[Any]:
        return [replace(line, text=LITERAL[line.text]) for line in input_lines]


def _request() -> TranslateSubtitlesRequest:
    return TranslateSubtitlesRequest(
        events=[
            SubtitleEvent(index=i, text=text, interval=TimeInterval(i, i + 0.9))
            for i, text in enumerate(SOURCES, start=1)
        ],
        stages=[
            TranslationStageConfig(
                kind=TranslationStageKind.LITERAL, model_name="gemini-3.6-flash"
            )
        ],
        context=TranslationContext(
            target_lang="Vietnamese", source_lang="zh", characters=ROSTER,
            visual_cues=CUES,
        ),
    )


@pytest.fixture(autouse=True)
def _fresh_log() -> None:
    get_process_log().clear()


class TestTranslateUseCase:
    def test_final_text_is_native_when_enabled(self) -> None:
        response = TranslateSubtitlesUseCase(
            translator=_Literal(), native_address=True  # type: ignore[arg-type]
        ).execute(_request())
        assert [event.text for event in response.events] == [
            "Anh Kiếm.",
            "Tôi vào xem anh Kiếm thế nào đã.",
            "Chị Hồng, tôi không muốn ăn.",
        ]

    def test_off_by_default(self) -> None:
        response = TranslateSubtitlesUseCase(
            translator=_Literal()  # type: ignore[arg-type]
        ).execute(_request())
        assert [event.text for event in response.events] == [
            LITERAL[text] for text in SOURCES
        ]

    def test_change_is_recorded_in_process_log(self) -> None:
        TranslateSubtitlesUseCase(
            translator=_Literal(), native_address=True  # type: ignore[arg-type]
        ).execute(_request())
        events = [
            event for event in get_process_log().snapshot()["events"]
            if event["event"] == "native_address_applied"
        ]
        assert events and events[-1]["lines"] == [1, 2, 3]


class _Analyzer:
    def analyze_global_context(self, **_kw: Any) -> SubtitleContextAnalysis:
        return SubtitleContextAnalysis(
            source_lang="zh", characters=f"{TU}\n{KIEM}",
            overview="Kiếm ca trở về nhà.", glossary="劍哥 = Kiếm ca",
        )


class TestAnalyzeUseCase:
    def test_result_is_native_when_enabled(self) -> None:
        result = AnalyzeSubtitleContextUseCase(
            _Analyzer(), native_address=True  # type: ignore[arg-type]
        ).execute([MagicMock(text="劍哥")], "Vietnamese")
        assert "(Chân Kiếm Thanh / anh Kiếm)" in result.characters
        assert "(danh xưng: anh Kiếm)" in result.characters
        assert result.overview == "Anh Kiếm trở về nhà."
        assert result.glossary == "劍哥 = anh Kiếm"

    def test_off_by_default(self) -> None:
        result = AnalyzeSubtitleContextUseCase(
            _Analyzer()  # type: ignore[arg-type]
        ).execute([MagicMock(text="劍哥")], "Vietnamese")
        assert "Kiếm ca" in result.characters


def _translation_prompt(kind: TranslationStageKind, native: bool) -> str:
    adapter = bare_analysis_adapter()
    adapter._native_address = native
    stage = TranslationStageConfig(kind=kind, model_name="gemini-3.6-flash")
    context = TranslationContext(target_lang="Vietnamese", characters=ROSTER)
    has_prior = kind is not TranslationStageKind.LITERAL
    return adapter._system_instruction(stage, context, has_prior)


class TestAdapterPrompts:
    @pytest.mark.parametrize(
        "kind",
        [TranslationStageKind.LITERAL, TranslationStageKind.STYLE,
         TranslationStageKind.LOCALIZE],
    )
    def test_translation_rule_when_enabled(self, kind: TranslationStageKind) -> None:
        assert NATIVE_ADDRESS_TRANSLATION_RULE in _translation_prompt(kind, True)
        assert NATIVE_ADDRESS_TRANSLATION_RULE not in _translation_prompt(kind, False)

    def test_preprocess_never_gets_the_rule(self) -> None:
        prompt = _translation_prompt(TranslationStageKind.PREPROCESS, True)
        assert NATIVE_ADDRESS_TRANSLATION_RULE not in prompt

    def _analysis_instruction(self, native: bool) -> str:
        adapter = bare_analysis_adapter()
        adapter._native_address = native
        captured: list[str] = []

        def fake_config(
            _temperature: float, _schema: Any, system: str, **_kw: Any
        ) -> Any:
            captured.append(system)
            return MagicMock()

        with (
            patch.object(adapter, "_build_config", fake_config),
            patch.object(adapter, "_call_gemini", lambda *a, **k: {
                "source_lang": "zh", "characters": [], "overview": "x",
            }),
            patch.object(adapter, "_ensure_available", lambda: None),
        ):
            adapter.analyze_global_context(
                [MagicMock(index=1, text="劍哥")], "Vietnamese"
            )
        return captured[0]

    def test_analysis_rule_when_enabled(self) -> None:
        assert NATIVE_ADDRESS_ANALYSIS_RULE in self._analysis_instruction(True)
        assert NATIVE_ADDRESS_ANALYSIS_RULE not in self._analysis_instruction(False)

    def test_analysis_progress_key_changes_only_when_enabled(self) -> None:
        adapter = bare_analysis_adapter()
        refs: list[Any] = []
        legacy = adapter._analysis_checkpoint_key("m", "p", refs, True)
        adapter._native_address = False
        assert adapter._analysis_checkpoint_key("m", "p", refs, True) == legacy
        adapter._native_address = True
        assert adapter._analysis_checkpoint_key("m", "p", refs, True) != legacy

    def test_constructor_keeps_the_flag(self) -> None:
        assert GeminiSubtitleTranslator(api_key="k", native_address=True)._native_address
        assert not GeminiSubtitleTranslator(api_key="k")._native_address


class TestSettingsAndContainer:
    def test_setting_defaults_to_off(self) -> None:
        assert TranslationSettings().native_address is False

    def test_container_passes_the_setting(self, tmp_path: Path) -> None:
        from subtitles_extractor.composition.container import ApplicationContainer
        from subtitles_extractor.infrastructure.settings.json_file_repository import (
            JsonFileRepository,
        )

        container = ApplicationContainer(
            settings_repository=JsonFileRepository(file_path=tmp_path / "c.json"),
            i18n_data_dir=tmp_path, user_data_dir=tmp_path,
        )
        container.settings_service.update(translation={"native_address": True})
        assert container.make_subtitle_translator("k")._native_address
        assert container.make_translate_subtitles_use_case("k")._native_address
        assert container.make_analyze_context_use_case("k")._native_address
