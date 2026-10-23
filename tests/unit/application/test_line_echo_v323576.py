"""[v3.23.576] Hồi quy: dòng dịch NÓI LẶP nội dung dòng kề.

Chuẩn đoán thật phim "傻子落水…" (3902 dòng, dịch bằng gemini-3.5-flash-lite, v3.23.575):
model gộp trọn câu của hai dòng vào một dòng rồi VẪN dịch dòng kế bên — giọng thuyết minh
đọc một ý hai lần ("Mọi người, mọi người bình tĩnh nào," rồi "Mọi người bình tĩnh nào,").
Cờ ``duplicate_adjacent`` chỉ bắt hai dòng GIỐNG HỆT nên cả loại lỗi này lọt lưới ở mọi
lần dịch (2-5 dòng/lần). Dữ liệu dưới đây là nguyên văn các ca thật.

Kèm hai lỗi ống dẫn: cờ ``needs_expansion`` bị mất khi lô bị chia đôi; và lô 517-529
(phim "灭门惨案II借种") trả dư 60 mục, đủ 12/13 dòng, chỉ dòng rác OCR "L上上" để trống →
bị coi là sai cấu trúc, gọi lại 5 lần rồi chia đôi thay vì vá đúng một dòng.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace

import pytest

from subtitles_extractor.application.services.line_echo_guard import (
    LineEcho,
    find_echo_reverts,
    find_line_echoes,
    group_still_repeats,
    repeated_line_groups,
)
from subtitles_extractor.application.services.translation_diagnostics import (
    detect_quality_flags,
)
from subtitles_extractor.application.services.translation_process_log import (
    get_process_log,
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
)

# (dòng, câu gốc, bản cuối) — phim "傻子落水…", v3.23.575.
EVERYONE = [
    (2819, "退钱退钱", "Trả tiền đi! Trả tiền!"),
    (2820, "各位各位", "Mọi người, mọi người bình tĩnh nào,"),
    (2821, "大家稍安勿躁", "Mọi người bình tĩnh nào,"),
    (2822, "稍安勿躁", "bình tĩnh nào,"),
]
BUMPKIN = [
    (2726, "儿子你来的正好", "Con đến vừa vặn lắm."),
    (2727, "你怎么能请他", "Sao con lại mời"),
    (2728, "来参加婚礼呢?", "loại đồ nhà quê này đến dự đám cưới thế?"),
    (2729, "这种山炮", "Loại đồ nhà quê này"),
    (2730, "他来我都嫌脏了空气", "thấy hắn xuất hiện làm bẩn cả bầu không khí."),
]
# Dòng MẤT luôn nghĩa của chính nó (lần dịch v3.23.571).
LOST = [
    (2879, "真是丟人", "Đúng là mất mặt chết đi được."),
    (2880, "本想着和徐家攀点关系",
     "Sớm biết thế này thì tao đã chẳng thèm mừng tiền rồi chuồn cho rồi."),
    (2881, "早知道这样就退钱走了", "Sớm biết thế này thì đã đòi tiền rồi chuồn cho rồi."),
    (2882, "大家别当真啊", "Mọi người đừng tin làm gì."),
]


def _split(rows: list[tuple]) -> tuple[dict[int, str], dict[int, str]]:
    return {r[0]: r[1] for r in rows}, {r[0]: r[2] for r in rows}


class TestFindLineEchoes:
    @pytest.mark.parametrize(
        ("rows", "expected"),
        [
            (EVERYONE, [LineEcho(2820, (2821,), 1.0)]),
            (BUMPKIN, [LineEcho(2728, (2729,), 1.0)]),
            (LOST, [LineEcho(2880, (2881,), 0.92)]),
        ],
    )
    def test_real_echoes(self, rows: list[tuple], expected: list[LineEcho]) -> None:
        sources, translations = _split(rows)
        assert find_line_echoes(sources, translations) == expected

    def test_echo_of_several_following_lines(self) -> None:
        """Ca thật (bản địa hoá): 1060 ra trọn câu của 1061-1063."""
        sources = {1060: "这是元神碎片。", 1061: "没想到", 1062: "第四块元神碎片",
                   1063: "居然在她身上"}
        translations = {
            1060: "Không ngờ mảnh nguyên thần thứ tư lại ở trên người cô ấy.",
            1061: "Không ngờ", 1062: "mảnh nguyên thần thứ tư",
            1063: "lại ở trên người cô ấy.",
        }
        sources[1060] = "这是那个"  # câu gốc KHÔNG chứa câu gốc các dòng kề
        echo = find_line_echoes(sources, translations)[0]
        assert (echo.index, echo.neighbors) == (1060, (1061, 1062, 1063))
        assert echo.lines == (1060, 1061, 1062, 1063)

    @pytest.mark.parametrize(
        "rows",
        [
            # Câu gốc cũng lặp lời dòng kề ("手镯") — lời thoại lặp thật.
            [(683, "哎我的手镯", "Ấy, vòng tay của em!"),
             (684, "这手镯是你的", "Vòng tay này là của em à?")],
            [(1451, "我我什么都愿意付出。", "tôi sẵn sàng đánh đổi mọi thứ."),
             (1452, "你什么都愿意?", "Cô sẵn sàng đánh đổi mọi thứ á?")],
            [(2835, "退钱退钱", "Trả tiền! Trả tiền!"),
             (2836, "退钱退钱", "Trả tiền! Trả tiền!")],
            # Dòng kề quá ngắn để làm bằng chứng.
            [(1, "小钱", "Tống Tiền,"), (2, "宋钱哥", "Anh Tống Tiền,")],
            # Phim nguồn chữ Latin: câu gốc lặp thật (phim "Along with the Gods").
            [(1337, "Vậy chưa đủ...", "Vậy chưa đủ..."),
             (1338, "Vậy chưa đủ,", "Vậy chưa đủ,")],
        ],
    )
    def test_repetition_present_in_source_is_not_an_echo(self, rows: list[tuple]) -> None:
        sources, translations = _split(rows)
        assert find_line_echoes(sources, translations) == []

    def test_blank_translations_are_ignored(self) -> None:
        assert find_line_echoes({1: "你好", 2: "再见"}, {1: "", 2: ""}) == []


class TestFindEchoReverts:
    def test_refinement_merged_a_cleanly_split_sentence(self) -> None:
        """Ca thật v3.23.565: dịch thô tách đúng, tinh chỉnh gộp rồi lặp."""
        sources = {2007: "还有你呢\uff1f", 2008: "爷爷\uff0c孙女", 2009: "知道错了",
                   2010: "爷爷\uff0c爷爷"}
        literal = {2007: "Còn mày thì sao?", 2008: "Ông nội, cháu gái",
                   2009: "biết sai rồi.", 2010: "Ông nội, ông nội,"}
        style = {
            **literal, 2008: "Ông nội, cháu gái biết sai rồi.", 2009: "Biết sai rồi."
        }
        assert find_echo_reverts(sources, literal, style) == {2008: "Ông nội, cháu gái"}

    def test_localize_stage_echo_is_reverted(self) -> None:
        """Ca thật v3.23.573, dòng 3526."""
        sources = {3525: "你现在变得有钱了", 3526: "而是因为", 3527: "我真的很喜欢你"}
        style = {3525: "bây giờ anh đã trở nên giàu có,", 3526: "mà vì",
                 3527: "em thật sự rất thích anh."}
        localize = {**style, 3526: "mà là vì em thật sự rất thích anh."}
        assert find_echo_reverts(sources, style, localize) == {3526: "mà vì"}

    def test_echo_caused_by_a_changed_neighbour_is_not_reverted(self) -> None:
        """Ca thật dòng 2535: dịch thô đã gộp; tinh chỉnh chỉ sửa dòng KỀ (đang lệch)."""
        sources = {2535: "哎呀沈总你就", 2536: "别为难慕云姐了", 2537: "叫我明月姐就行"}
        literal = {2535: "Ôi chị Minh Nguyệt đừng làm khó chị Mộ Vân nữa.",
                   2536: "Gọi chị là Minh Nguyệt tỷ là được.",
                   2537: "Gọi em là Minh Nguyệt tỷ là được."}
        style = {**literal, 2536: "Đừng làm khó chị Mộ Vân nữa."}
        assert [e.index for e in find_line_echoes(sources, style)] == [2535]
        assert find_echo_reverts(sources, literal, style) == {}

    def test_restoring_must_not_make_the_neighbour_echo(self) -> None:
        """Ca thật v3.23.571 dòng 3081-3083: bản địa hoá viết lại CẢ dòng 3081."""
        sources = {3081: "就凭你找来", 3082: "这几个蝼蚁啊", 3083: "还不足以对付"}
        style = {3081: "Chỉ dựa vào mấy con lũ kiến cỏ", 3082: "mà mày gọi tới đây,",
                 3083: "vẫn chưa đủ để đấu lại tao đâu."}
        localize = {3081: "Chỉ dựa vào mấy kẻ mà mày gọi tới",
                    3082: "vẫn chưa đủ để đối phó với tao đâu.",
                    3083: "vẫn chưa đủ để đối phó đâu."}
        assert find_echo_reverts(sources, style, localize) == {}

    def test_restoring_must_not_duplicate_a_neighbour(self) -> None:
        """Ca thật (bản cũ): bản trước lệch một dòng nên câu cũ trùng hệt dòng kề."""
        sources = {48: "这这是哪儿?", 49: "此地为本帝的识海", 50: "宋钱"}
        literal = {48: "Nơi này là Thức hải của bản đế.", 49: "Tống Tiền,",
                   50: "bản đế đã đem toàn bộ"}
        style = {48: "Nơi này là Thức hải của bản đế.",
                 49: "Nơi này là Thức hải của bản đế,", 50: "Tống Tiền,"}
        assert find_echo_reverts(sources, literal, style) == {}

    def test_restoring_must_not_bring_back_untranslated_cjk(self) -> None:
        sources = {1: "胖子你跟林逸", 2: "不会是看强哥", 3: "定在了中山酒店"}
        previous = {1: "hai người các cậu", 2: "có phải thấy Cường哥",
                    3: "đặt tiệc ở Khách sạn Trung Sơn,"}
        current = {**previous, 2: "hai người các cậu, có phải thấy anh Cường"}
        assert find_echo_reverts(sources, previous, current) == {}

    def test_echo_already_in_previous_stage_is_left_to_the_cleanup(self) -> None:
        sources, final = _split(EVERYONE)
        assert find_echo_reverts(sources, final, dict(final)) == {}


class TestRepeatedLineGroups:
    def test_identical_pair_and_echo_are_both_groups(self) -> None:
        sources = {1859: "厂里困难", 1860: "连大家的工资都", 1861: "发不了",
                   **_split(EVERYONE)[0]}
        translations = {1859: "Nhà máy khó khăn,", 1860: "không phát nổi.",
                        1861: "không phát nổi.", **_split(EVERYONE)[1]}
        assert repeated_line_groups(sources, translations) == [
            (1860, 1861), (2820, 2821)
        ]

    def test_overlapping_groups_are_merged(self) -> None:
        sources = {1: "一二", 2: "三四", 3: "五六"}
        translations = {1: "a b c d e f", 2: "a b c", 3: "a b c"}
        assert repeated_line_groups(sources, translations) == [(1, 2, 3)]

    def test_still_repeats(self) -> None:
        sources, translations = _split(EVERYONE)
        assert group_still_repeats((2820, 2821), sources, translations)
        fixed = {**translations, 2820: "Mọi người, mọi người,"}
        assert not group_still_repeats((2820, 2821), sources, fixed)


def _line(index: int, text: str, source: str = "") -> TranslationLine:
    return TranslationLine(
        index=index, start_ms=index * 1000, end_ms=index * 1000 + 900, text=text,
        original_text=source or None,
    )


def _request() -> TranslateSubtitlesRequest:
    return TranslateSubtitlesRequest(
        events=[],
        stages=[
            TranslationStageConfig(kind=TranslationStageKind.LITERAL, model_name="m"),
            TranslationStageConfig(kind=TranslationStageKind.STYLE, model_name="m"),
        ],
        context=TranslationContext(target_lang="Vietnamese"),
    )


class _Retranslator:
    def __init__(self, answer: dict[int, str]) -> None:
        self.answer = answer
        self.calls: list[list[int]] = []

    def translate_stage(self, *, stage, source_lines, **_kwargs):
        assert stage.kind is TranslationStageKind.LITERAL
        self.calls.append([line.index for line in source_lines])
        return [
            replace(line, text=self.answer.get(line.index, "")) for line in source_lines
        ]


def _film(rows: list[tuple]) -> tuple[list[TranslationLine], list[TranslationLine]]:
    current = [_line(i, text, source) for i, source, text in rows]
    sources = [_line(i, source, source) for i, source, _ in rows]
    return current, sources


class TestCleanupRetranslatesEchoGroups:
    def test_echo_group_is_retranslated_from_source(self) -> None:
        current, sources = _film(BUMPKIN + EVERYONE)
        referee = _Retranslator({
            2820: "Mọi người, mọi người,", 2821: "xin hãy bình tĩnh,",
            2728: "đến dự đám cưới thế?", 2729: "Loại đồ nhà quê này",
        })
        get_process_log().clear()
        use_case = TranslateSubtitlesUseCase(translator=referee)  # type: ignore[arg-type]
        result = use_case._cleanup_duplicate_adjacent(current, sources, _request(), None)
        texts = {line.index: line.text for line in result}
        assert referee.calls == [[2728, 2729, 2820, 2821]]
        assert texts[2820] == "Mọi người, mọi người,"
        assert texts[2728] == "đến dự đám cưới thế?"
        assert texts[2727] == "Sao con lại mời"  # ngoài nhóm: giữ nguyên
        events = [e for e in get_process_log().snapshot()["events"]
                  if e["event"] == "repeated_lines_retranslated"]
        assert events[-1]["fixed"] == [[2728, 2729], [2820, 2821]]

    def test_each_group_is_accepted_on_its_own(self) -> None:
        """Nhóm dịch lại vẫn lặp thì giữ bản cũ — không kéo theo nhóm đã sửa được."""
        current, sources = _film(BUMPKIN + EVERYONE)
        referee = _Retranslator({
            2820: "Mọi người, mọi người bình tĩnh nào,", 2821: "Mọi người bình tĩnh nào,",
            2728: "đến dự đám cưới thế?", 2729: "Loại đồ nhà quê này",
        })
        use_case = TranslateSubtitlesUseCase(translator=referee)  # type: ignore[arg-type]
        result = use_case._cleanup_duplicate_adjacent(current, sources, _request(), None)
        texts = {line.index: line.text for line in result}
        assert texts[2820] == "Mọi người, mọi người bình tĩnh nào,"
        assert texts[2728] == "đến dự đám cưới thế?"

    def test_nothing_fixed_keeps_the_same_list(self) -> None:
        current, sources = _film(EVERYONE)
        referee = _Retranslator({2820: "Mọi người, mọi người bình tĩnh nào,"})
        use_case = TranslateSubtitlesUseCase(translator=referee)  # type: ignore[arg-type]
        assert use_case._cleanup_duplicate_adjacent(
            current, sources, _request(), None
        ) is current


class TestStageRevertInPipeline:
    def test_refinement_echo_never_reaches_the_final(self) -> None:
        from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
        from subtitles_extractor.domain.value_objects.time_interval import TimeInterval

        sources = ["还有你呢\uff1f", "爷爷\uff0c孙女", "知道错了", "爷爷\uff0c爷爷"]
        literal = ["Còn mày thì sao?", "Ông nội, cháu gái", "biết sai rồi.",
                   "Ông nội, ông nội,"]
        style = ["Còn mày thì sao?", "Ông nội, cháu gái biết sai rồi.", "Biết sai rồi.",
                 "Ông nội, ông nội,"]

        class _Pipeline:
            def translate_stage(self, *, stage, input_lines, **_kwargs):
                texts = literal if stage.kind is TranslationStageKind.LITERAL else style
                return [replace(ln, text=texts[ln.index - 1]) for ln in input_lines]

        events = [
            SubtitleEvent(index=pos, text=text, interval=TimeInterval(pos, pos + 0.9))
            for pos, text in enumerate(sources, start=1)
        ]
        get_process_log().clear()
        use_case = TranslateSubtitlesUseCase(translator=_Pipeline())  # type: ignore[arg-type]
        response = use_case.execute(replace(_request(), events=events))
        assert [event.text for event in response.events] == [
            "Còn mày thì sao?", "Ông nội, cháu gái", "Biết sai rồi.", "Ông nội, ông nội,"
        ]
        reverts = [e for e in get_process_log().snapshot()["events"]
                   if e["event"] == "echo_revert"]
        assert reverts and reverts[-1]["lines"] == [2]


class _Event:
    def __init__(self, index: int, text: str) -> None:
        self.index = index
        self.text = text


class TestDiagnostics:
    def test_remaining_echoes_are_reported(self) -> None:
        sources = [_Event(i, s) for i, s, _ in EVERYONE]
        final = [_Event(i, t) for i, _, t in EVERYONE]
        flags = detect_quality_flags(sources, final)
        assert flags["counts"]["echo_lines"] == 1
        assert flags["echo_lines"] == [{
            "index": 2820, "neighbors": [2821],
            "text": "Mọi người, mọi người bình tĩnh nào,",
        }]
        assert "echo_lines" in flags["note"]


# ── Ống dẫn giữa các lô (adapter) ─────────────────────────────────────────────


def _adapter():
    from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (  # noqa: E501
        GeminiSubtitleTranslator,
    )

    translator = GeminiSubtitleTranslator.__new__(GeminiSubtitleTranslator)
    translator._retry_count = 1
    return translator


class TestExpansionFlagSurvivesHalving:
    def test_halves_still_carry_needs_expansion(self) -> None:
        from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (  # noqa: E501
            _BatchCountMismatchError,
        )

        translator = _adapter()
        sent: list[list[dict]] = []

        def fake_call(_model, prompt, _config, validator, **_kwargs):
            block = re.search(r"<current_batch>(.*?)</current_batch>", prompt, re.S)
            items = json.loads(block.group(1))
            sent.append(items)
            if len(items) > 2:
                raise _BatchCountMismatchError("Model trả sai số dòng")
            data = {"subtitles": [
                {"line_no": item["line_no"], "text": f"dịch {item['line_no']}"}
                for item in items
            ]}
            validator(data)
            return data

        translator._call_gemini = fake_call
        lines = [
            TranslationLine(index=i, start_ms=i * 1000, end_ms=i * 1000 + 900,
                            text=f"bản nháp {i}", original_text=f"原文{i}")
            for i in range(1, 5)
        ]
        result = translator._translate_single_batch(
            batch=lines, source_before=[], source_after=[], history_before=[],
            start_idx=0, is_preprocess=False, is_literal=False, config=None,
            model_name="m", cancel_cb=None, dual_payload=True,
            expansion_indices=frozenset({3}),
        )
        assert [line.text for line in result] == [f"dịch {i}" for i in range(1, 5)]
        flagged = [
            [item["line_no"] for item in items if item.get("needs_expansion")]
            for items in sent
        ]
        assert flagged == [[3], [], [3]]


class TestOverflowWithOneBlankLine:
    def test_real_batch_517_529_is_patched_not_rejected(self) -> None:
        from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (  # noqa: E501
            _BatchPartialError,
        )

        returned = {517: "Trong lòng em,", 518: "anh mãi là người chồng tuyệt vời nhất.",
                    519: "A Tú,", 520: "đừng khóc nữa, anh không sao.",
                    521: "Thôi, khuya rồi, ngủ sớm đi.", 522: "Ngủ đi.", 523: "",
                    524: "Chào buổi sáng.", 525: "Chị dâu.", 526: "Để tôi giúp chị.",
                    527: "Xin lỗi chú.", 528: "Có chuyện gì thế?", 529: "Chuyện gì ạ?"}
        context = [{"line_no": n, "text": f"ngữ cảnh {n}"} for n in range(490, 517)]
        context += [{"line_no": n, "text": f"ngữ cảnh {n}"} for n in range(530, 550)]
        payload = {"subtitles": context + [
            {"line_no": n, "text": text} for n, text in returned.items()
        ]}
        assert len(payload["subtitles"]) == 60
        with pytest.raises(_BatchPartialError) as caught:
            _adapter()._validate_batch(payload, expected_count=13, start_idx=516)
        assert caught.value.missing_line_nos == [523]
        assert [item["line_no"] for item in payload["subtitles"]] == [
            n for n in returned if n != 523
        ]
