"""[v3.23.605] Giai đoạn tinh chỉnh chỉ chèn từ đệm mà vượt ngân sách đọc → giữ bản trước.

Chuẩn đoán thật 11/10 21:10 (灭门惨案II): dịch thô gemini-3.7-flash để 122 dòng vượt
``max_chars``; tinh chỉnh gemini-3.5-flash-lite đẩy lên 194 (bản cuối 202). TTS: số dòng
nén trên 1.6 lần tăng 44 → 59 so với lần chạy trước. Mốc thời gian, câu gốc, bản dịch thô
và bản tinh chỉnh dưới đây lấy nguyên từ file chuẩn đoán đó.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from subtitles_extractor.application.services.padding_regression_guard import (
    find_padding_regressions,
    inserted_words,
)
from subtitles_extractor.application.use_cases.translate_subtitles import (
    TranslateSubtitlesUseCase,
)
from subtitles_extractor.domain.ports.subtitle_translator_port import TranslationLine
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
)

#: (dòng, bắt đầu ms, kết thúc ms, bắt đầu dòng kế ms, câu gốc, dịch thô, tinh chỉnh).
REAL_LINES = [
    (46, 257467, 258400, 258533, "你真是混蛋呀你",
     "Đúng là đồ khốn nạn!", "Anh đúng là đồ khốn nạn!"),
    (142, 625533, 626933, 627000, "應付這麼多人而已",
     "đối phó nhiều người thế này.", "đối phó với nhiều người thế này."),
    (171, 712533, 713333, 713400, "跟著我的拍子", "Theo nhịp tôi,", "Theo nhịp của tôi,"),
    (176, 767800, 769000, 769067, "我還有一段時間才行呀",
     "Tôi còn lâu mới ra được!", "Tôi còn lâu mới ra được đấy!"),
    (200, 841733, 841799, 841800, "慢慢吃吧", "Ăn đi.", "Từ từ ăn đi."),
    (384, 1605400, 1606800, 1606867, "你呀太過份了，現在怎麼辦",  # noqa: RUF001
     "Mày quá đáng lắm, giờ tính sao?",
     "Mày quá đáng lắm rồi đấy, giờ tính sao đây?"),
    (400, 1634533, 1636467, 1636533, "你把她的地方都撞散了呀",
     "Mày tông nát chỗ người ta rồi kìa!",
     "Mày tông nát hết cả chỗ của người ta rồi kìa!"),
    (410, 1648400, 1649533, 1649600, "我不會賠給你的",
     "Tôi không đền cho anh nữa đâu.", "Tôi sẽ không đền cho anh nữa đâu."),
    (416, 1658733, 1659733, 1660400, "洪震，你幹甚麼",  # noqa: RUF001
     "Hồng Chấn, mày làm cái gì đấy?", "Hồng Chấn, mày đang làm cái gì đấy?"),
    (602, 2558667, 2559800, 2559867, "我沒有事，不用扶我",  # noqa: RUF001
     "Anh không sao, khỏi đỡ.", "Anh không sao, khỏi cần đỡ."),
    (617, 2603533, 2604333, 2604467, "是我媽留給我",
     "mẹ để lại cho tôi,", "là mẹ để lại cho tôi,"),
    (771, 3613400, 3614933, 3615000, "我們奉公守法",
     "Bọn tao tuân thủ pháp luật,", "Bọn tao đây luôn tuân thủ pháp luật,"),
    (967, 4255133, 4256600, 4256800, "怎麼了？我老婆最近怎麼了？",  # noqa: RUF001
     "Sao thế? Dạo này vợ tôi thế nào?", "Sao thế? Dạo này vợ tôi thế nào rồi?"),
    (1075, 4546067, 4546867, 4546933, "一定是阿軍來了",
     "Chắc A Quân tới.", "Chắc A Quân tới rồi."),
    (1243, 5789667, 5791333, 5791400, "分明可以破一宗滅門大案",
     "Rõ ràng phá được thảm án diệt môn,",
     "Rõ ràng có thể phá được một vụ thảm án diệt môn,"),
]

#: Dòng chỉ chèn hư từ/trợ từ ("với", "của", "đấy", "rồi", "sẽ", "đang"…).
PADDED = [142, 171, 176, 384, 410, 416, 617, 771, 967, 1075]
#: Chèn đại từ (46), từ mang nghĩa (200 "Từ từ" = 慢慢, 400 "hết cả", 602 "cần",
#: 1243 "có thể … vụ").
KEPT = [46, 200, 400, 602, 1243]


def _source_lines() -> list[TranslationLine]:
    lines: list[TranslationLine] = []
    for index, start, end, next_start, source, _literal, _style in REAL_LINES:
        lines.append(
            TranslationLine(
                index=index, start_ms=start, end_ms=end, text=source, original_text=source
            )
        )
        # Dòng kế tiếp thật chỉ để ngân sách tính đúng khung hiệu dụng như khi dịch.
        lines.append(
            TranslationLine(
                index=index + 10_000,
                start_ms=next_start,
                end_ms=next_start + 500,
                text="",
            )
        )
    return lines


@pytest.fixture(scope="module")
def adapter() -> GeminiSubtitleTranslator:
    # 21 = mặc định "reading_cps_budget" của Cài đặt (đúng như container truyền vào).
    return GeminiSubtitleTranslator(api_key="KEY", reading_cps_budget=21.0)


class TestFindPaddingRegressions:
    def test_real_filler_only_rewrites_are_reverted(
        self, adapter: GeminiSubtitleTranslator
    ) -> None:
        budgets = adapter.line_char_budgets(_source_lines())
        previous = {row[0]: row[5] for row in REAL_LINES}
        current = {row[0]: row[6] for row in REAL_LINES}
        assert find_padding_regressions(previous, current, budgets) == PADDED

    def test_lines_within_budget_are_never_touched(self) -> None:
        previous = {1: "Chắc A Quân tới."}
        current = {1: "Chắc A Quân tới rồi."}
        assert find_padding_regressions(previous, current, {1: 40}) == []
        assert find_padding_regressions(previous, current, {}) == []

    def test_statement_turned_question_is_kept(self) -> None:
        # Ca thật (傻子落水…, v3.23.5xx): câu kể thành câu hỏi chỉ nhờ trợ từ "chứ".
        previous = {1: "Dật Tử của chúng ta không có tiền"}
        current = {1: "Dật Tử của chúng ta không có tiền chứ?"}
        assert find_padding_regressions(previous, current, {1: 30}) == []

    def test_inserted_words_reports_only_pure_insertions(self) -> None:
        assert inserted_words("Theo nhịp tôi,", "Theo nhịp của tôi,") == ["của"]
        assert inserted_words("Phá đám tao!", "Dám phá chuyện tốt của tao à.") is None


class _BudgetTranslator:
    def __init__(self, adapter: GeminiSubtitleTranslator) -> None:
        self._adapter = adapter

    def line_char_budgets(self, lines: list[TranslationLine]) -> dict[int, int]:
        return self._adapter.line_char_budgets(lines)


class TestUseCaseStageGuard:
    def test_stage_output_keeps_previous_text_for_padded_lines(
        self, adapter: GeminiSubtitleTranslator
    ) -> None:
        use_case = TranslateSubtitlesUseCase(translator=_BudgetTranslator(adapter))
        sources = _source_lines()
        real = [line for line in sources if line.index < 10_000]
        literal = {row[0]: row[5] for row in REAL_LINES}
        style = {row[0]: row[6] for row in REAL_LINES}
        previous = [replace(line, text=literal[line.index]) for line in real]
        output = [replace(line, text=style[line.index]) for line in real]

        result = use_case._revert_padding_regressions(output, previous, sources, "GĐ3")

        by_index = {line.index: line.text for line in result}
        assert {index for index in by_index if by_index[index] == literal[index]
                and literal[index] != style[index]} == set(PADDED)
        assert all(by_index[index] == style[index] for index in KEPT)

    def test_translator_without_budgets_changes_nothing(self) -> None:
        use_case = TranslateSubtitlesUseCase(translator=object())  # type: ignore[arg-type]
        lines = [
            TranslationLine(index=1, start_ms=0, end_ms=500, text="Chắc A Quân tới rồi.")
        ]
        previous = [replace(lines[0], text="Chắc A Quân tới.")]
        result = use_case._revert_padding_regressions(lines, previous, lines, "GĐ3")
        assert result is lines


class TestPipeline:
    def test_padding_from_style_stage_never_reaches_the_final(
        self, adapter: GeminiSubtitleTranslator
    ) -> None:
        from subtitles_extractor.application.use_cases.translate_subtitles import (
            TranslateSubtitlesRequest,
        )
        from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
        from subtitles_extractor.domain.ports.subtitle_translator_port import (
            TranslationContext,
            TranslationStageConfig,
            TranslationStageKind,
        )
        from subtitles_extractor.domain.value_objects.time_interval import TimeInterval

        chosen = [row for row in REAL_LINES if row[0] in (384, 1075)]
        events: list[SubtitleEvent] = []
        literal: list[str] = []
        style: list[str] = []
        for _index, start, end, next_start, source, literal_text, style_text in chosen:
            # Dòng thật cùng dòng kế tiếp (mốc thật): ngân sách tính đúng khung hiệu dụng.
            events.append(SubtitleEvent(
                index=len(events) + 1, text=source,
                interval=TimeInterval(start / 1000, end / 1000),
            ))
            events.append(SubtitleEvent(
                index=len(events) + 1, text="走吧",
                interval=TimeInterval(next_start / 1000, next_start / 1000 + 0.8),
            ))
            literal += [literal_text, "Đi thôi."]
            style += [style_text, "Đi thôi."]

        class _Pipeline:
            def translate_stage(self, *, stage, input_lines, **_kwargs):
                texts = literal if stage.kind is TranslationStageKind.LITERAL else style
                return [replace(ln, text=texts[ln.index - 1]) for ln in input_lines]

            def line_char_budgets(self, lines: list[TranslationLine]) -> dict[int, int]:
                return adapter.line_char_budgets(lines)

        request = TranslateSubtitlesRequest(
            events=events,
            stages=[
                TranslationStageConfig(kind=TranslationStageKind.LITERAL, model_name="m"),
                TranslationStageConfig(kind=TranslationStageKind.STYLE, model_name="m"),
            ],
            context=TranslationContext(target_lang="Vietnamese"),
        )
        use_case = TranslateSubtitlesUseCase(translator=_Pipeline())  # type: ignore[arg-type]
        response = use_case.execute(request)
        assert [event.text for event in response.events] == literal
