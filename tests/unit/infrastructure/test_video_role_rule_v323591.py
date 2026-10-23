"""[v3.23.591] Hồi quy: đính video khi dịch mà KHÔNG nói video dùng để làm gì.

Phim "灭门惨案II" nói tiếng Quảng Đông, phụ đề cứng theo văn viết: lời nói và chữ khác
nhau, và trong tiếng còn có những câu không hề có phụ đề. Cả ba giai đoạn dịch đều đính
1-3 đoạn video cho mỗi lô, nhưng chỉ thị hệ thống không có chữ nào về vai trò của video.
Trên 16 lượt dịch thật của phim này:

    dòng  câu gốc             bản thô đúng   bản cuối mang lời NGHE được
    68    不要生氣             16/16          8/16  "Mở cửa mau!", "Khai mau.", …
    70    這麼久               16/16          4/16  "Nhanh lên!", …
    77    你沒有事嗎            16/16          3/16  "Mọi người vỗ tay hoan nghênh!"
    93    改革開放成功了         16/16          6/16  "MƯỜI NĂM cải cách mở cửa…"

Dòng 93 là dấu vết rõ nhất: câu gốc không có con số nào, bản thô của cả 16 lượt cũng
không — "mười năm" chỉ có trong tiếng.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from subtitles_extractor.domain.ports.subtitle_translator_port import (
    TranslationContext,
    TranslationLine,
    TranslationStageConfig,
    TranslationStageKind,
)
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    VIDEO_ROLE_RULE,
    GeminiSubtitleTranslator,
)
from subtitles_extractor.infrastructure.translation.gemini_video_context import (
    RemoteVideoRef,
)

TRANSLATION_STAGES = (
    TranslationStageKind.LITERAL,
    TranslationStageKind.STYLE,
    TranslationStageKind.LOCALIZE,
)

#: Câu gốc và bản thô chép nguyên từ lượt chạy thật v3.23.590 (bản thô đúng cả bốn dòng;
#: giai đoạn tinh chỉnh của chính lượt này đổi dòng 68 thành "Khai mau." và thêm "mười
#: năm" vào dòng 93).
REAL_LINES = [
    (68, "不要生氣", "Đừng giận."),
    (70, "這麼久", "Lâu thế?"),
    (77, "你沒有事嗎", "Cậu không sao chứ?"),
    (93, "改革開放成功了", "cải cách mở cửa đã thành công rồi."),
]


def _stub() -> SimpleNamespace:
    return SimpleNamespace(
        _canonical_names_directive=GeminiSubtitleTranslator._canonical_names_directive
    )


def _instruction(
    kind: TranslationStageKind, *, with_video: bool, has_prior: bool = True
) -> str:
    stage = TranslationStageConfig(kind=kind, model_name="gemini-3.5-flash-lite")
    context = TranslationContext(target_lang="Vietnamese", source_lang="zh", overview="x")
    return GeminiSubtitleTranslator._system_instruction(
        _stub(), stage, context, has_prior, with_video=with_video
    )


class TestRuleText:
    def test_video_is_context_only_and_text_wins(self) -> None:
        assert "VIDEO KÈM THEO (nếu có) CHỈ LÀ NGỮ CẢNH" in VIDEO_ROLE_RULE
        assert "NGUỒN ĐỂ DỊCH luôn là CHỮ của từng dòng phụ đề" in VIDEO_ROLE_RULE
        assert "Chữ và tiếng khác nhau → LUÔN theo CHỮ" in VIDEO_ROLE_RULE

    def test_rule_names_the_three_real_failures(self) -> None:
        """Dịch theo lời nghe (68), thay câu của dòng (77), thêm con số (93)."""
        assert "KHÔNG dịch theo lời nghe được" in VIDEO_ROLE_RULE
        assert "KHÔNG thêm từ ngữ hay con số chỉ có trong âm thanh" in VIDEO_ROLE_RULE
        assert "KHÔNG thay nội dung của một dòng bằng câu nghe thấy" in VIDEO_ROLE_RULE

    def test_rule_explains_why_speech_can_differ(self) -> None:
        assert "lồng tiếng lại hoặc nói tiếng địa phương" in VIDEO_ROLE_RULE
        assert "câu thoại không có phụ đề" in VIDEO_ROLE_RULE

    def test_emotion_tags_are_not_forbidden_by_it(self) -> None:
        """Tag cảm xúc (tuỳ chọn) là thứ DUY NHẤT được lấy từ âm thanh."""
        assert "Tag cảm xúc" in VIDEO_ROLE_RULE

    def test_rule_is_one_bullet_line(self) -> None:
        assert VIDEO_ROLE_RULE.startswith("- ")
        assert VIDEO_ROLE_RULE.endswith("\n")
        assert VIDEO_ROLE_RULE.count("\n") == 1


class TestSystemInstruction:
    @pytest.mark.parametrize("kind", TRANSLATION_STAGES)
    def test_stage_with_video_carries_the_rule(self, kind: TranslationStageKind) -> None:
        assert VIDEO_ROLE_RULE in _instruction(kind, with_video=True)

    @pytest.mark.parametrize("kind", TRANSLATION_STAGES)
    def test_stage_without_video_is_unchanged(self, kind: TranslationStageKind) -> None:
        text = _instruction(kind, with_video=False)
        assert "VIDEO KÈM THEO" not in text
        assert text == _instruction(kind, with_video=True).replace(VIDEO_ROLE_RULE, "")

    @pytest.mark.parametrize("kind", TRANSLATION_STAGES)
    def test_default_is_without_video(self, kind: TranslationStageKind) -> None:
        stage = TranslationStageConfig(kind=kind, model_name="m")
        context = TranslationContext(target_lang="Vietnamese", overview="x")
        text = GeminiSubtitleTranslator._system_instruction(_stub(), stage, context, True)
        assert "VIDEO KÈM THEO" not in text

    @pytest.mark.parametrize("kind", TRANSLATION_STAGES)
    def test_rule_sits_in_the_rules_section_before_line_alignment(
        self, kind: TranslationStageKind
    ) -> None:
        text = _instruction(kind, with_video=True)
        rules, video, alignment = (
            text.index("## QUY TẮC"),
            text.index(VIDEO_ROLE_RULE),
            text.index("- CĂN DÒNG 1:1"),
        )
        assert rules < video < alignment < text.index("## ĐỊNH DẠNG")

    def test_preprocess_never_gets_the_rule(self) -> None:
        """Hiệu đính phụ đề gốc không phải khâu dịch và không đính video."""
        text = _instruction(
            TranslationStageKind.PREPROCESS, with_video=True, has_prior=False
        )
        assert "VIDEO KÈM THEO" not in text

    @pytest.mark.parametrize("has_prior", [True, False])
    def test_rule_does_not_depend_on_prior_translation(self, has_prior: bool) -> None:
        text = _instruction(
            TranslationStageKind.STYLE, with_video=True, has_prior=has_prior
        )
        assert VIDEO_ROLE_RULE in text


def _lines() -> list[TranslationLine]:
    return [
        TranslationLine(
            index=index,
            start_ms=index * 1000,
            end_ms=index * 1000 + 900,
            text=literal,
            original_text=source,
        )
        for index, source, literal in REAL_LINES
    ]


def _run_stage(
    kind: TranslationStageKind, *, attach_video: bool, video_refs: list[Any]
) -> list[str]:
    """Chạy ``translate_stage`` thật, trả các chỉ thị hệ thống đã dựng cho giai đoạn."""
    adapter = GeminiSubtitleTranslator(api_key="KEY1")
    adapter._ensure_available = lambda: None  # type: ignore[assignment]
    adapter._resolve_video_handles = lambda refs: [  # type: ignore[assignment]
        SimpleNamespace(name=ref.remote_name) for ref in refs or []
    ]
    built: list[str] = []

    def _build_config(
        _temperature: float, _schema: Any, system_instruction: str, **_kwargs: Any
    ) -> SimpleNamespace:
        built.append(system_instruction)
        return SimpleNamespace(system_instruction=system_instruction)

    def _single_batch(
        *, batch: list[TranslationLine], **_kwargs: Any
    ) -> list[TranslationLine]:
        return [replace(line, text=line.text) for line in batch]

    adapter._build_config = _build_config  # type: ignore[assignment]
    adapter._translate_single_batch = _single_batch  # type: ignore[assignment]
    lines = _lines()
    adapter.translate_stage(
        stage=TranslationStageConfig(kind=kind, model_name="m", batch_size=10),
        context=TranslationContext(target_lang="Vietnamese", source_lang="zh"),
        source_lines=lines,
        input_lines=lines,
        has_prior_translation=kind is not TranslationStageKind.LITERAL,
        video_refs=video_refs,
        attach_video=attach_video,
    )
    assert built, "translate_stage không dựng cấu hình nào"
    return built


def _refs() -> list[RemoteVideoRef]:
    return [RemoteVideoRef(0, "files/SEG0", 0.0, 344.0, "ACTIVE")]


class TestTranslateStageWiring:
    @pytest.mark.parametrize("kind", TRANSLATION_STAGES)
    def test_real_setup_all_three_stages_attach_video(
        self, kind: TranslationStageKind
    ) -> None:
        """Đúng cài đặt của người dùng: cả ba giai đoạn đều đính video."""
        for instruction in _run_stage(kind, attach_video=True, video_refs=_refs()):
            assert VIDEO_ROLE_RULE in instruction

    @pytest.mark.parametrize("kind", TRANSLATION_STAGES)
    def test_stage_not_attaching_video_has_no_rule(
        self, kind: TranslationStageKind
    ) -> None:
        for instruction in _run_stage(kind, attach_video=False, video_refs=_refs()):
            assert "VIDEO KÈM THEO" not in instruction

    def test_attach_flag_without_any_video_has_no_rule(self) -> None:
        """Bật đính video nhưng không có đoạn nào (chuẩn bị video hỏng) → không nhắc."""
        instructions = _run_stage(
            TranslationStageKind.STYLE, attach_video=True, video_refs=[]
        )
        assert all("VIDEO KÈM THEO" not in text for text in instructions)
