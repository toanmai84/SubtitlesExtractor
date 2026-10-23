"""[v3.23.605] Dòng thoại không được bị giai đoạn tinh chỉnh biến thành chỉ còn "[cười]".

Chuẩn đoán thật 11/10 21:10 (灭门惨案II): dòng 263 "當當" — bản tinh chỉnh
"Đương Đương...", giai đoạn bản địa hoá (bật sinh tag cảm xúc) trả về "[cười]". Khâu
TTS bỏ trống dòng chỉ có tag ("Bị bỏ (không đọc): 1 — Cửa sổ quá ngắn / text rỗng") và
phụ đề xuất ra cũng lọc mất tag → dòng thoại biến mất. Cùng dòng ở v3.23.586:
"Trời ạ." → "[cười]".
"""

from __future__ import annotations

from subtitles_extractor.application.services.spoken_line_guard import (
    find_unspoken_regressions,
    has_spoken_words,
    is_non_verbal_source,
)
from subtitles_extractor.application.use_cases.translate_subtitles import (
    TranslateSubtitlesUseCase,
)
from subtitles_extractor.domain.ports.subtitle_translator_port import TranslationLine

#: Dòng 262-264 thật: (dòng, bắt đầu ms, kết thúc ms, câu gốc, tinh chỉnh, bản địa hoá).
SCENE_263 = [
    (262, 1028267, 1030133, "因為她還是處女呀",
     "Vì cô ấy vẫn còn là trinh nữ!", "Vì cô ấy vẫn còn là trinh nữ!"),
    (263, 1030533, 1031100, "當當", "Đương Đương...", "[cười]"),
    (264, 1034000, 1034800, "我明白了", "Tôi hiểu rồi!", "Tôi hiểu rồi!"),
]


class TestSpokenWords:
    def test_tag_only_line_has_no_spoken_words(self) -> None:
        assert not has_spoken_words("[cười]")
        assert not has_spoken_words("[cười] [thở dài]!")
        assert not has_spoken_words("")
        assert has_spoken_words("[cười] Thật sự rất tuyệt đấy!")

    def test_laughter_and_sighs_are_non_verbal_sources(self) -> None:
        assert is_non_verbal_source("哈哈哈哈哈哈哈")
        assert is_non_verbal_source("唉")
        assert is_non_verbal_source("Hahaha")
        assert is_non_verbal_source("（笑）")  # noqa: RUF001 — chữ thật

    def test_words_are_verbal_sources(self) -> None:
        assert not is_non_verbal_source("當當")
        assert not is_non_verbal_source("哎呀")
        assert not is_non_verbal_source("哈哈，你好")  # noqa: RUF001 — chữ thật


class TestFindUnspokenRegressions:
    def test_real_line_263_is_reverted(self) -> None:
        sources = {row[0]: row[3] for row in SCENE_263}
        previous = {row[0]: row[4] for row in SCENE_263}
        current = {row[0]: row[5] for row in SCENE_263}
        assert find_unspoken_regressions(sources, previous, current) == [263]

    def test_v323586_variant_is_reverted(self) -> None:
        sources = {263: "當當"}
        found = find_unspoken_regressions(sources, {263: "Trời ạ."}, {263: "[cười]"})
        assert found == [263]

    def test_laughter_may_become_a_tag(self) -> None:
        sources = {2679: "哈哈哈哈哈哈哈"}
        previous = {2679: "Ha ha ha!"}
        assert find_unspoken_regressions(sources, previous, {2679: "[cười]"}) == []

    def test_previous_without_words_is_not_restored(self) -> None:
        # Dòng nhiễu OCR thật "L上上": mọi giai đoạn đều trả tag — không có lời để giữ.
        sources = {523: "L上上"}
        previous = {523: "[thở dài]"}
        assert find_unspoken_regressions(sources, previous, {523: "[Music]"}) == []


class TestUseCaseStageGuard:
    def test_stage_output_keeps_previous_spoken_text(self) -> None:
        sources = [
            TranslationLine(index=i, start_ms=s, end_ms=e, text=src, original_text=src)
            for i, s, e, src, _prev, _cur in SCENE_263
        ]
        previous = [
            TranslationLine(index=i, start_ms=s, end_ms=e, text=prev)
            for i, s, e, _src, prev, _cur in SCENE_263
        ]
        output = [
            TranslationLine(index=i, start_ms=s, end_ms=e, text=cur)
            for i, s, e, _src, _prev, cur in SCENE_263
        ]
        result = TranslateSubtitlesUseCase._revert_unspoken_regressions(
            output, previous, sources, "GĐ4"
        )
        assert [line.text for line in result] == [
            "Vì cô ấy vẫn còn là trinh nữ!", "Đương Đương...", "Tôi hiểu rồi!",
        ]


class TestQualityFlag:
    def test_final_tag_only_line_is_flagged_but_laughter_is_not(self) -> None:
        from types import SimpleNamespace

        from subtitles_extractor.application.services.translation_diagnostics import (
            detect_quality_flags,
        )

        def _event(index: int, text: str) -> SimpleNamespace:
            return SimpleNamespace(
                index=index, text=text, start_sec=index, end_sec=index + 1
            )

        sources = [
            _event(263, "當當"), _event(2050, "哎呀"), _event(2679, "哈哈哈哈哈哈哈"),
        ]
        finals = [_event(263, "[cười]"), _event(2050, "[cười]"), _event(2679, "[cười]")]
        flags = detect_quality_flags(sources, finals)
        assert flags["unspoken_lines"] == [263, 2050]
        assert flags["counts"]["unspoken_lines"] == 2


class TestPipeline:
    def test_tag_only_localize_output_never_reaches_the_final(self) -> None:
        from dataclasses import replace

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

        events = [
            SubtitleEvent(index=pos, text=src, interval=TimeInterval(s / 1000, e / 1000))
            for pos, (_i, s, e, src, _prev, _cur) in enumerate(SCENE_263, start=1)
        ]
        style = [row[4] for row in SCENE_263]
        localize = [row[5] for row in SCENE_263]

        class _Pipeline:
            literal_calls = 0

            def translate_stage(self, *, stage, input_lines, **_kwargs):
                texts = localize if stage.kind is TranslationStageKind.LOCALIZE else style
                if stage.kind is TranslationStageKind.LITERAL:
                    self.literal_calls += 1
                    if self.literal_calls > 1:
                        # Lượt dịch trọng tài cho dòng lẻ bị viết lại: lần chạy thật
                        # trọng tài KHÔNG đứng về bản trước ("… → giữ"); v3.23.603
                        # dịch "Tèn ten!".
                        texts = [*style[:1], "Tèn ten!", *style[2:]]
                return [replace(ln, text=texts[ln.index - 1]) for ln in input_lines]

        request = TranslateSubtitlesRequest(
            events=events,
            stages=[
                TranslationStageConfig(kind=kind, model_name="m")
                for kind in (
                    TranslationStageKind.LITERAL,
                    TranslationStageKind.STYLE,
                    TranslationStageKind.LOCALIZE,
                )
            ],
            context=TranslationContext(target_lang="Vietnamese"),
        )
        use_case = TranslateSubtitlesUseCase(translator=_Pipeline())  # type: ignore[arg-type]
        response = use_case.execute(request)
        assert [event.text for event in response.events] == style
