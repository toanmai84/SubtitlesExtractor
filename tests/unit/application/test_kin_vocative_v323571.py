"""[v3.23.571] Hồi quy: tiếng gọi kèm VAI VẾ (``宋钱弟弟``) mang nhãn người nói trái vai.

Chuẩn đoán thật v3.23.570 (lặp ở 5 lượt dịch): dòng 3197/3221/3286 gốc ``宋钱弟弟`` ("em
Tống Tiền") có nhãn người nói ``张依贝`` — người mà bảng nhân vật ghi "gọi 'anh Tống
Tiền'" — nên cả lượt lời ra "Anh Tống Tiền, anh đi trước đi!" (12/15 dòng sai vai). Dòng
413 cùng câu gốc có nhãn ngược ``spk=宋钱`` nên ra "Chị Đào Hoa ơi,". Chỉ sửa từ "Anh"
thành "Em" sẽ ra câu lai "Em Tống Tiền, anh đi trước đi!" — phải sửa ở NHÃN, trước khi
dịch.
"""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from subtitles_extractor.application.services.context_diagnostics import (
    build_context_diagnostics,
)
from subtitles_extractor.application.services.roster_format import parse_roster
from subtitles_extractor.application.services.visual_cue_serializer import (
    deserialize_visual_cues,
)
from subtitles_extractor.application.services.vocative_guard import (
    KIN_RELABEL_MAX_LINES,
    OLDER,
    YOUNGER,
    _alias_index,
    address_rank,
    find_vocative_name_fixes,
    fix_kin_vocative_speakers,
    fix_vocative_cue_labels,
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

# Rút gọn từ bảng nhân vật thật của lượt dịch v3.23.570 (quy tắc xưng hô giữ nguyên văn).
ROSTER = (
    "宋钱 / 宋神医 (Tống Tiền) [nam, ~23] — nhân vật chính. Xưng hô: với 沈明月 tự xưng"
    " 'tôi' gọi 'chị Minh Nguyệt'; với 张依贝 tự xưng 'anh' gọi 'em'; với 许桃花 tự xưng"
    " 'em' gọi 'chị' (danh xưng: chị Đào Hoa); với 张铁柱 tự xưng 'em' gọi"
    " 'anh Thiết Trụ'\n"
    "许桃花 (Hứa Đào Hoa) [nữ, ~24] — chị dâu. Xưng hô: với 宋钱 tự xưng 'chị' gọi"
    " 'Tiền em'\n"
    "张依贝 (Trương Y Bối) [nữ, ~23] — bạn học. Xưng hô: với 宋钱 tự xưng 'em' gọi"
    " 'anh Tống Tiền' / 'sư phụ' (danh xưng: anh Tống Tiền)\n"
    "沈明月 (Thẩm Minh Nguyệt) [nữ, ~25] — chủ tịch khách sạn. Xưng hô: với 宋钱 tự xưng"
    " 'chị' gọi 'em Tiền' (danh xưng: Tống thần y)\n"
    "向城 (Hướng Thành) [nam, ~45] — võ sư. Xưng hô: với 宋钱 tự xưng 'ta' gọi"
    " 'thằng nhãi'\n"
    "安雪漫 (An Tuyết Mạn) [nữ, ~23] — bạn gái cũ. Xưng hô: với 宋钱 tự xưng 'tôi'"
    " gọi 'anh'\n"
    "张铁柱 / 铁柱 (Trương Thiết Trụ) [nam, ~30] — dân làng"
)
SONG = "宋钱 / 宋神医"

# Nguyên văn ca thật: số dòng → (câu gốc, người nói, người nghe, bản dịch v3.23.570).
REAL = {
    412: ("但我实在不愿意", "许桃花", SONG, "Nhưng chị thật sự không muốn."),
    413: ("宋钱弟弟", SONG, "许桃花", "Chị Đào Hoa ơi,"),
    414: ("你愿意帮帮我吗\uff1f", "许桃花", SONG, "em có bằng lòng giúp chị không?"),
    3195: ("宋钱你别去", "沈明月", SONG, "em Tiền, em đừng qua đó."),
    3196: ("你会被他打死的", "沈明月", SONG, "Hắn sẽ đánh chết em mất."),
    3197: ("宋钱弟弟", "张依贝", SONG, "Anh Tống Tiền,"),
    3198: ("你先走吧", "张依贝", SONG, "anh đi trước đi!"),
    3199: ("毕竟我们家大业大", "张依贝", SONG, "Dù gì nhà chúng ta cũng có thế lực lớn,"),
    3200: ("就算他是大宗师", "张依贝", SONG, "cho dù hắn là đại tông sư,"),
    3201: (
        "也不敢拿我们怎么样的",
        "张依贝",
        "向城",
        "thì cũng không dám làm gì chúng ta đâu.",
    ),
    3202: ("没事", SONG, "张依贝", "Không sao đâu,"),
    3220: ("就能打败大宗师", "沈明月", "张依贝", "mà đã đánh bại được đại tông sư."),
    3221: ("宋钱弟弟", "张依贝", "沈明月", "Anh Tống Tiền,"),
    3222: ("也太厉害了\uff01", "张依贝", "沈明月", "anh lợi hại quá đi mất!"),
    3223: ("宋钱", "安雪漫", SONG, "Tống Tiền,"),
}
SOURCES = {index: row[0] for index, row in REAL.items()}


def _cues(rows: dict[int, tuple[str, str, str, str]] = REAL) -> list[VisualCue]:
    return [
        VisualCue(line_no=index, speaker=row[1], addressee=row[2], scene="")
        for index, row in sorted(rows.items())
    ]


def _labels(cues: list[VisualCue]) -> dict[int, tuple[str, str]]:
    return {cue.line_no: (cue.speaker, cue.addressee) for cue in cues}


class TestAddressRank:
    @pytest.mark.parametrize(
        ("rule", "expected"),
        [
            ("tự xưng 'em' gọi 'anh Tống Tiền' / 'sư phụ'", OLDER),
            ("tự xưng 'chị' gọi 'em Tiền' (danh xưng: Tống thần y)", YOUNGER),
            ("tự xưng 'chị' gọi 'Tiền em'", YOUNGER),
            ("tự xưng 'em' gọi 'Cẩu ca'", OLDER),
            # Cách gọi không có vai vế → suy từ cách tự xưng (ca thật lượt 60b5385c).
            ("tự xưng 'chị Minh Nguyệt' gọi 'Tiểu Tiền'", YOUNGER),
            ("tự xưng 'ta' gọi 'thằng nhãi'", None),
            # Hai giai đoạn trái vai nhau → không rõ.
            (
                "ban đầu tự xưng 'tôi' gọi 'anh', sau xưng 'chị Mộ Vân' gọi 'em Tiền'",
                None,
            ),
            # "Anh" viết hoa là một phần của tên (英), không phải vai vế.
            ("tự xưng 'tôi' gọi 'Tú Anh'", None),
            ("", None),
        ],
    )
    def test_rank(self, rule: str, expected: str | None) -> None:
        assert address_rank(rule) == expected


class TestKinVocativeCallee:
    def test_kin_suffix_is_read(self) -> None:
        index = _alias_index(parse_roster(ROSTER))
        younger = vocative_callee("宋钱弟弟", index)
        older = vocative_callee("哎呀\uff0c铁柱哥\uff01", index)
        assert younger is not None and younger.alias == "宋钱"
        assert younger.kinship == ("em", "đệ")
        assert older is not None and older.alias == "铁柱"
        assert older.kinship == ("anh", "ca")

    def test_alias_ending_in_kin_character_is_not_split(self) -> None:
        roster = ROSTER + "\n吴二狗 / 狗哥 (Ngô Nhị Cẩu / Cẩu Ca)"
        index = _alias_index(parse_roster(roster))
        callee = vocative_callee("狗哥", index)
        assert callee is not None and callee.alias == "狗哥" and callee.kinship is None


class TestFixKinVocativeSpeakers:
    def test_real_conflicts_are_relabeled(self) -> None:
        fixed, relabeled = fix_kin_vocative_speakers(_cues(), SOURCES, ROSTER)
        assert relabeled == [413, 3197, 3221]
        labels = _labels(fixed)
        assert labels[413] == ("许桃花", SONG)
        assert labels[3197] == ("沈明月", SONG)
        assert labels[3221] == ("沈明月", SONG)

    def test_whole_utterance_follows_the_vocative(self) -> None:
        labels = _labels(fix_kin_vocative_speakers(_cues(), SOURCES, ROSTER)[0])
        for line in (3198, 3199, 3200):
            assert labels[line] == ("沈明月", SONG)
        # Người nghe trong nhãn là người nói mới → người nghe thật là người được gọi.
        assert labels[3222] == ("沈明月", SONG)

    def test_lines_outside_the_utterance_are_untouched(self) -> None:
        labels = _labels(fix_kin_vocative_speakers(_cues(), SOURCES, ROSTER)[0])
        assert labels[3201] == ("张依贝", "向城")
        assert labels[3202] == (SONG, "张依贝")
        assert labels[3223] == ("安雪漫", SONG)
        # Nhãn ghi chính người được gọi: chỉ đổi dòng gọi tên, dòng sau có thể là lời đáp.
        assert labels[414] == ("许桃花", SONG)

    def test_speaker_label_of_callee_changes_only_the_vocative_line(self) -> None:
        rows = {
            412: REAL[412],
            413: REAL[413],
            414: ("当然", SONG, "许桃花", "Đương nhiên..."),
        }
        labels = _labels(fix_kin_vocative_speakers(_cues(rows), SOURCES, ROSTER)[0])
        assert labels[413] == ("许桃花", SONG)
        assert labels[414] == (SONG, "许桃花")

    def test_matching_speaker_is_left_alone(self) -> None:
        cues = [
            VisualCue(line_no=1, speaker="沈明月", addressee=SONG, scene=""),
            VisualCue(line_no=2, speaker="张依贝", addressee=SONG, scene=""),
        ]
        sources = {1: "宋钱弟弟", 2: "宋钱哥哥"}
        assert fix_kin_vocative_speakers(cues, sources, ROSTER) == (cues, [])

    def test_unknown_rank_is_left_alone(self) -> None:
        cues = [
            VisualCue(line_no=1, speaker="沈明月", addressee=SONG, scene=""),
            VisualCue(line_no=2, speaker="向城", addressee=SONG, scene=""),
        ]
        assert fix_kin_vocative_speakers(cues, {2: "宋钱弟弟"}, ROSTER) == (cues, [])

    def test_no_matching_neighbour_leaves_labels(self) -> None:
        cues = [
            VisualCue(line_no=1, speaker="向城", addressee=SONG, scene=""),
            VisualCue(line_no=2, speaker="张依贝", addressee=SONG, scene=""),
            VisualCue(line_no=3, speaker="安雪漫", addressee=SONG, scene=""),
        ]
        assert fix_kin_vocative_speakers(cues, {2: "宋钱弟弟"}, ROSTER) == (cues, [])

    def test_utterance_is_capped(self) -> None:
        total = KIN_RELABEL_MAX_LINES + 3
        cues = [VisualCue(line_no=1, speaker="沈明月", addressee=SONG, scene="")] + [
            VisualCue(line_no=line, speaker="张依贝", addressee=SONG, scene="")
            for line in range(2, total + 1)
        ]
        fixed, _ = fix_kin_vocative_speakers(cues, {2: "宋钱弟弟"}, ROSTER)
        speakers = [cue.speaker for cue in fixed]
        assert speakers.count("沈明月") == 1 + KIN_RELABEL_MAX_LINES
        assert speakers[-1] == "张依贝"

    def test_roster_without_address_rules_is_a_no_op(self) -> None:
        roster = "宋钱 (Tống Tiền) [nam]\n张依贝 (Trương Y Bối) [nữ]"
        cues = _cues()
        assert fix_kin_vocative_speakers(cues, SOURCES, roster) == (cues, [])


class TestFixVocativeCueLabels:
    def test_kin_relabel_runs_before_the_swap(self) -> None:
        rows = dict(REAL)
        rows[7] = ("宋钱", SONG, "许桃花", "Tống Tiền.")
        sources = {**SOURCES, 7: "宋钱"}
        fixes = fix_vocative_cue_labels(_cues(rows), sources, ROSTER)
        assert fixes.changed
        assert fixes.kin_relabeled == [413, 3197, 3221]
        assert fixes.swapped == [7]
        labels = _labels(fixes.cues)
        assert labels[7] == ("许桃花", SONG)
        assert labels[3197] == ("沈明月", SONG)

    def test_nothing_to_fix(self) -> None:
        cues = [VisualCue(line_no=1, speaker="沈明月", addressee=SONG, scene="")]
        fixes = fix_vocative_cue_labels(cues, {1: "宋钱弟弟"}, ROSTER)
        assert not fixes.changed and fixes.cues is cues


class TestKinVocativeNameFixes:
    def test_wrong_name_and_its_kin_word_are_fixed(self) -> None:
        """Ca thật dòng 413: "Chị Đào Hoa ơi," / "Chị Đào Hoa," cho gốc 宋钱弟弟."""
        translations = {1: "Chị Đào Hoa ơi,", 2: "Chị Đào Hoa,"}
        sources = {1: "宋钱弟弟", 2: "宋钱弟弟"}
        fixes = find_vocative_name_fixes(sources, translations, ROSTER)
        assert [fix.text_after for fix in fixes] == ["Em Tống Tiền ơi,", "Em Tống Tiền,"]

    def test_right_name_with_wrong_kin_word_is_not_patched_alone(self) -> None:
        """Sửa riêng "Anh" → "Em" sẽ ra câu lai với dòng nói tiếp "anh đi trước đi!"."""
        sources = {3197: "宋钱弟弟", 3198: "你先走吧"}
        translations = {3197: "Anh Tống Tiền,", 3198: "anh đi trước đi!"}
        assert find_vocative_name_fixes(sources, translations, ROSTER) == []

    def test_hán_việt_kin_word_is_kept_in_style(self) -> None:
        roster = ROSTER.replace("(Hứa Đào Hoa)", "(Hứa Đào Hoa / Đào Hoa)")
        fixes = find_vocative_name_fixes({1: "宋钱弟弟"}, {1: "Đào Hoa tỷ,"}, roster)
        assert fixes[0].text_after == "Tống Tiền đệ,"


def _lines(order: list[int], texts: dict[int, str]) -> list[TranslationLine]:
    return [
        TranslationLine(
            index=pos, start_ms=pos * 1000, end_ms=pos * 1000 + 900,
            text=texts[line], original_text=SOURCES[line],
        )
        for pos, line in enumerate(order, start=1)
    ]


def _cues_json(order: list[int]) -> str:
    return json.dumps(
        [
            {"id": pos, "spk": REAL[line][1], "to": REAL[line][2]}
            for pos, line in enumerate(order, start=1)
        ],
        ensure_ascii=False,
    )


def _request(cues: str) -> TranslateSubtitlesRequest:
    return TranslateSubtitlesRequest(
        events=[],
        stages=[
            TranslationStageConfig(kind=TranslationStageKind.LITERAL, model_name="m")
        ],
        context=TranslationContext(
            target_lang="Vietnamese", characters=ROSTER, visual_cues=cues
        ),
    )


class TestUseCase:
    def test_request_cues_are_relabeled_before_translation(self) -> None:
        order = [3195, 3196, 3197, 3198, 3199, 3200, 3201, 3202]
        request = _request(_cues_json(order))
        fixed = TranslateSubtitlesUseCase._fix_vocative_cues(
            request, _lines(order, SOURCES)
        )
        assert fixed is not request
        cues = {c.line_no: c for c in deserialize_visual_cues(fixed.context.visual_cues)}
        assert cues[order.index(3197) + 1].speaker == "沈明月"
        assert cues[order.index(3198) + 1].speaker == "沈明月"
        assert cues[order.index(3201) + 1].speaker == "张依贝"

    def test_translator_sees_the_matching_speaker(self) -> None:
        """Model dịch theo nhãn: nhãn đúng vai thì cả lượt lời ra cùng một vai."""
        from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
        from subtitles_extractor.domain.value_objects.time_interval import TimeInterval

        order = [3196, 3197, 3198]
        events = [
            SubtitleEvent(
                index=pos, text=SOURCES[line], interval=TimeInterval(pos, pos + 0.9)
            )
            for pos, line in enumerate(order, start=1)
        ]
        by_speaker = {
            "张依贝": {2: "Anh Tống Tiền,", 3: "anh đi trước đi!"},
            "沈明月": {2: "Em Tống Tiền,", 3: "em đi trước đi!"},
        }

        class _Translator:
            def translate_stage(self, *, input_lines, context, **_kwargs):
                cues = deserialize_visual_cues(context.visual_cues)
                texts = by_speaker[{c.line_no: c for c in cues}[2].speaker]
                return [
                    replace(line, text=texts.get(line.index, "Hắn sẽ đánh chết em mất."))
                    for line in input_lines
                ]

        use_case = TranslateSubtitlesUseCase(translator=_Translator())  # type: ignore[arg-type]
        request = replace(_request(_cues_json(order)), events=events)
        response = use_case.execute(request)
        assert [event.text for event in response.events][1:] == [
            "Em Tống Tiền,",
            "em đi trước đi!",
        ]
        persisted = {c.line_no: c for c in deserialize_visual_cues(response.visual_cues)}
        assert persisted[2].speaker == "沈明月" and persisted[3].speaker == "沈明月"


class TestContextDiagnostics:
    def test_kin_conflicts_are_reported(self) -> None:
        order = sorted(REAL)
        context = SimpleNamespace(
            target_lang="Vietnamese", source_lang="zh", overview="Tóm tắt.",
            characters=ROSTER, glossary="", visual_cues=_cues_json(order),
            enable_tags=False, include_desc=True,
        )
        events = [
            SimpleNamespace(index=pos, text=SOURCES[line])
            for pos, line in enumerate(order, start=1)
        ]
        report = build_context_diagnostics(context=context, source_events=events)
        cues_report = report["visual_cues"]
        kin_lines = [order[pos - 1] for pos in cues_report["kin_speaker_conflict_lines"]]
        assert kin_lines == [413, 3197, 3221]
        assert cues_report["reversed_vocative_lines"] == []
