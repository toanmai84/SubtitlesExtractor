"""[v3.23.570] Hồi quy: dòng chỉ GỌI MỘT CÁI TÊN bị dịch thành tên người khác.

Chuẩn đoán thật v3.23.569: dòng 2532 gốc chỉ có ``柳慕云`` dịch thành "Thẩm Minh Nguyệt,",
dòng 3075 gốc ``徐俊豪`` dịch thành "Tống Tiền," — lặp ở 5-6 lượt dịch của phim này; trên
100 bản dịch thật có 28 dòng như vậy. Nguyên nhân: nhãn Visual Cues ghi NGƯỢC người nói/
người nghe (``spk=柳慕云, to=沈明月``) và model tin nhãn hơn câu gốc.
"""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

from subtitles_extractor.application.services.context_diagnostics import (
    build_context_diagnostics,
)
from subtitles_extractor.application.services.visual_cue_serializer import (
    deserialize_visual_cues,
)
from subtitles_extractor.application.services.vocative_guard import (
    find_vocative_name_fixes,
    fix_vocative_cues,
    vocative_callee,
)
from subtitles_extractor.application.use_cases.translate_subtitles import (
    TranslateSubtitlesRequest,
    TranslateSubtitlesUseCase,
)
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    TranslationContext,
    TranslationLine,
    TranslationStageConfig,
    TranslationStageKind,
    VisualCue,
)

ROSTER = (
    "宋钱 / 宋神医 (Tống Tiền) [nam, ~23] — nhân vật chính\n"
    "许桃花 (Hứa Đào Hoa) [nữ, ~24] — chị dâu\n"
    "春花 / 许春花 (Hứa Xuân Hoa) [nữ, ~28] — vợ Trương Thiết Trụ\n"
    "徐俊豪 (Từ Tuấn Hào) [nam, ~25] — phản diện\n"
    "柳慕云 (Liễu Mộ Vân) [nữ, ~25] — tổng giám đốc\n"
    "沈明月 (Thẩm Minh Nguyệt) [nữ, ~25] — chủ tịch khách sạn\n"
    "吴二狗 / 土狗 / 狗哥 (Ngô Nhị Cẩu / Thổ Cẩu / Cẩu Ca) [nam, ~30] — côn đồ"
)

# Nguyên văn ca thật (số dòng, câu gốc, nhãn người nói, nhãn người nghe, bản dịch).
REAL = {
    2531: ("我明白了", "柳慕云", "沈明月", "Tôi hiểu rồi,"),
    2532: ("柳慕云", "柳慕云", "沈明月", "Thẩm Minh Nguyệt,"),
    2533: (
        "你是不是喜欢宋神医啊?",
        "柳慕云",
        "沈明月",
        "có phải cậu thích Tống thần y không?",
    ),
    3075: ("徐俊豪", "徐俊豪", "宋钱 / 宋神医", "Tống Tiền,"),
    3076: ("你有句话说的很对", "宋钱 / 宋神医", "徐俊豪", "mày nói một câu rất đúng,"),
    7: ("宋钱", "宋钱 / 宋神医", "许桃花", "Tống Tiền."),
    8: ("宋钱", "许桃花", "宋钱 / 宋神医", "Tống Tiền."),
}
SOURCES = {index: row[0] for index, row in REAL.items()}
TRANSLATIONS = {index: row[3] for index, row in REAL.items()}


def _cues() -> list[VisualCue]:
    return [
        VisualCue(line_no=index, speaker=row[1], addressee=row[2], scene="")
        for index, row in sorted(REAL.items())
    ]


class TestVocativeCallee:
    def _index(self) -> dict:
        from subtitles_extractor.application.services.roster_format import parse_roster
        from subtitles_extractor.application.services.vocative_guard import _alias_index

        return _alias_index(parse_roster(ROSTER))

    def test_interjections_and_punctuation_are_stripped(self) -> None:
        index = self._index()
        first = vocative_callee("哎呀许桃花\uff01", index)
        second = vocative_callee("宋钱啊\uff0c", index)
        assert first is not None and first.alias == "许桃花"
        assert second is not None and second.alias == "宋钱"

    def test_name_inside_a_sentence_is_not_a_vocative(self) -> None:
        # [v3.23.571] "宋钱哥哥" nay là tiếng gọi kèm vai vế (test_kin_vocative_v323571).
        index = self._index()
        assert vocative_callee("你是不是喜欢宋神医啊?", index) is None
        assert vocative_callee("宋钱你别去", index) is None


class TestFixVocativeCues:
    def test_reversed_labels_are_swapped(self) -> None:
        fixed, swapped = fix_vocative_cues(_cues(), SOURCES, ROSTER)
        assert swapped == [7, 2532, 3075]
        by_line = {cue.line_no: cue for cue in fixed}
        assert (by_line[2532].speaker, by_line[2532].addressee) == ("沈明月", "柳慕云")
        assert (by_line[3075].speaker, by_line[3075].addressee) == (
            "宋钱 / 宋神医",
            "徐俊豪",
        )

    def test_correct_and_non_vocative_labels_are_untouched(self) -> None:
        fixed, _ = fix_vocative_cues(_cues(), SOURCES, ROSTER)
        by_line = {cue.line_no: cue for cue in fixed}
        assert (by_line[8].speaker, by_line[8].addressee) == ("许桃花", "宋钱 / 宋神医")
        assert by_line[2533].speaker == "柳慕云"

    def test_nothing_to_fix_returns_same_list(self) -> None:
        cues = [VisualCue(line_no=8, speaker="许桃花", addressee="宋钱", scene="")]
        assert fix_vocative_cues(cues, SOURCES, ROSTER) == (cues, [])

    def test_self_addressed_label_is_left_alone(self) -> None:
        cues = [VisualCue(line_no=1, speaker="狗哥", addressee="吴二狗", scene="")]
        assert fix_vocative_cues(cues, {1: "狗哥"}, ROSTER)[1] == []


class TestVocativeNameFixes:
    def test_real_swaps_are_fixed(self) -> None:
        found = find_vocative_name_fixes(SOURCES, TRANSLATIONS, ROSTER)
        fixes = {fix.index: fix for fix in found}
        assert fixes[2532].text_after == "Liễu Mộ Vân,"
        assert fixes[3075].text_after == "Từ Tuấn Hào,"
        assert set(fixes) == {2532, 3075}

    def test_particles_are_kept(self) -> None:
        fixes = find_vocative_name_fixes(
            {1: "柳慕云啊"}, {1: "Thẩm Minh Nguyệt à..."}, ROSTER
        )
        assert fixes[0].text_after == "Liễu Mộ Vân à..."

    def test_alias_gets_its_own_reading(self) -> None:
        fixes = find_vocative_name_fixes({1: "狗哥"}, {1: "Tống Tiền ơi,"}, ROSTER)
        assert fixes[0].text_after == "Cẩu Ca ơi,"

    def test_kinship_title_means_a_shifted_line(self) -> None:
        """Ca thật dòng 175: gốc ``宋钱`` mang câu dòng sau "Chị dâu Xuân Hoa..."."""
        roster = ROSTER.replace("(Hứa Xuân Hoa)", "(Xuân Hoa)")
        translations = {1: "Chị dâu Xuân Hoa..."}
        assert find_vocative_name_fixes({1: "宋钱"}, translations, roster) == []

    def test_whole_sentence_is_not_touched(self) -> None:
        text = "mà dám cho Tống Tiền mượn hai vạn tệ cơ à?"
        assert find_vocative_name_fixes({1: "徐俊豪"}, {1: text}, ROSTER) == []

    def test_own_name_in_any_form_is_accepted(self) -> None:
        assert find_vocative_name_fixes({1: "宋钱"}, {1: "Tiền em à,"}, ROSTER) == []


def _lines(texts: dict[int, str]) -> list[TranslationLine]:
    return [
        TranslationLine(
            index=i, start_ms=i * 1000, end_ms=i * 1000 + 900,
            text=texts[i], original_text=SOURCES[i],
        )
        for i in sorted(texts)
    ]


def _request(cues: str = "") -> TranslateSubtitlesRequest:
    return TranslateSubtitlesRequest(
        events=[],
        stages=[
            TranslationStageConfig(kind=TranslationStageKind.LITERAL, model_name="m")
        ],
        context=TranslationContext(
            target_lang="Vietnamese", characters=ROSTER, visual_cues=cues
        ),
    )


def _cues_json() -> str:
    return json.dumps(
        [{"id": i, "spk": row[1], "to": row[2]} for i, row in sorted(REAL.items())],
        ensure_ascii=False,
    )


class TestUseCase:
    def test_request_cues_are_fixed(self) -> None:
        request = _request(_cues_json())
        fixed = TranslateSubtitlesUseCase._fix_vocative_cues(request, _lines(SOURCES))
        assert fixed is not request
        cues = {c.line_no: c for c in deserialize_visual_cues(fixed.context.visual_cues)}
        assert cues[2532].addressee == "柳慕云"

    def test_request_without_cues_is_untouched(self) -> None:
        request = _request()
        assert TranslateSubtitlesUseCase._fix_vocative_cues(request, _lines(SOURCES)) is (
            request
        )

    def test_stage_output_is_fixed(self) -> None:
        result = TranslateSubtitlesUseCase._enforce_vocative_names(
            _lines(TRANSLATIONS), _lines(SOURCES), _request().context, "Dịch thô"
        )
        texts = {line.index: line.text for line in result}
        assert texts[2532] == "Liễu Mộ Vân," and texts[3075] == "Từ Tuấn Hào,"
        assert texts[2531] == TRANSLATIONS[2531]

    def test_execute_fixes_and_returns_cues_to_persist(self) -> None:
        from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
        from subtitles_extractor.domain.value_objects.time_interval import TimeInterval

        order = sorted(REAL)
        events = [
            SubtitleEvent(
                index=pos, text=SOURCES[i], interval=TimeInterval(pos, pos + 0.9)
            )
            for pos, i in enumerate(order, start=1)
        ]
        cues = json.dumps(
            [
                {"id": pos, "spk": REAL[i][1], "to": REAL[i][2]}
                for pos, i in enumerate(order, start=1)
            ],
            ensure_ascii=False,
        )

        class _Translator:
            def translate_stage(self, *, input_lines, **_kwargs):
                return [
                    replace(line, text=TRANSLATIONS[order[line.index - 1]])
                    for line in input_lines
                ]

        translator = _Translator()
        use_case = TranslateSubtitlesUseCase(translator=translator)  # type: ignore[arg-type]
        request = replace(_request(cues), events=events)
        response = use_case.execute(request)
        final = {order[pos - 1]: e.text for pos, e in enumerate(response.events, 1)}
        assert final[2532] == "Liễu Mộ Vân," and final[3075] == "Từ Tuấn Hào,"
        persisted = {c.line_no: c for c in deserialize_visual_cues(response.visual_cues)}
        assert persisted[order.index(2532) + 1].addressee == "柳慕云"


class TestContextDiagnostics:
    def test_reversed_vocatives_are_counted(self) -> None:
        order = sorted(REAL)
        cues = json.dumps(
            [
                {"id": pos, "spk": REAL[i][1], "to": REAL[i][2]}
                for pos, i in enumerate(order, start=1)
            ],
            ensure_ascii=False,
        )
        context = SimpleNamespace(
            target_lang="Vietnamese", source_lang="zh", overview="Tóm tắt.",
            characters=ROSTER, glossary="", visual_cues=cues,
            enable_tags=False, include_desc=True,
        )
        events = [SimpleNamespace(index=pos, text=SOURCES[i]) for pos, i in enumerate(
            order, start=1
        )]
        report = build_context_diagnostics(context=context, source_events=events)
        reversed_lines = report["visual_cues"]["reversed_vocative_lines"]
        assert [order[pos - 1] for pos in reversed_lines] == [7, 2532, 3075]
