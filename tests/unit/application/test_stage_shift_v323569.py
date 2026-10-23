"""[v3.23.569] Hồi quy: giai đoạn tinh chỉnh đẩy lệch dòng và làm sai con số.

Chuẩn đoán thật v3.23.568 (dịch ba giai đoạn bằng gemini-3.5-flash-lite):

1. Dịch thô căn đúng 3393-3423; tinh chỉnh phong cách trả bản mà mỗi dòng mang câu của
   dòng thứ HAI kế tiếp. Tương quan độ dài vẫn 0,6-0,9 (câu ngắn dài xấp xỉ nhau) nên bộ
   dò khối im lặng; bản cuối còn 9 dòng lệch và mất một câu. Dữ liệu dưới đây là nguyên
   văn ca thật; cột trọng tài là bản dịch ĐỘC LẬP của chính phim này ở một lượt khác.
2. Dịch thô đọc đúng ``一千万`` là "mười triệu" cả 6 lần; tinh chỉnh đổi 4 lần thành
   "một nghìn vạn" và 2 lần thành "một triệu" (SAI GIÁ TRỊ). Lưới số chỉ chọn cách đọc
   trong CHÍNH giai đoạn đó — không còn bản đúng nào nên không sửa được gì.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from subtitles_extractor.application.services.number_consistency_guard import (
    find_number_fixes,
)
from subtitles_extractor.application.services.stage_shift_guard import (
    VERDICT_CURRENT,
    VERDICT_PREVIOUS,
    VERDICT_UNDECIDED,
    StageShift,
    find_stage_shifts,
    judge_shift,
    shift_span,
)
from subtitles_extractor.application.use_cases.translate_subtitles import (
    TranslateSubtitlesRequest,
    TranslateSubtitlesUseCase,
)
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    SubtitleTranslationError,
    TranslationCancelledError,
    TranslationContext,
    TranslationLine,
    TranslationStageConfig,
    TranslationStageKind,
)

# (dòng, câu gốc, dịch thô, tinh chỉnh phong cách, bản dịch độc lập làm trọng tài)
REAL = [
    (3393, "哎", "Hừm,", "Mới là gì đâu,", "Thôi đi,"),
    (3394, "这才哪到哪啊", "mới chỉ là bắt đầu thôi.", "sau này còn nhiều.",
     "đây đã là gì."),
    (3395, "这种情况", "Sau này", "Tiền đối với anh", "Kiểu này"),
    (3396, "以后会更多", "sẽ còn nhiều hơn thế.", "chỉ là những con số mà thôi.",
     "sau này sẽ còn nhiều hơn."),
    (3397, "钱对于我来说", "Tiền đối với anh", "Đúng rồi Tiểu Dao,", "Tiền đối với tôi"),
    (3398, "只是一串数字而已", "cũng chỉ là những con số mà thôi.",
     "quay về đập nát nhà chúng ta đi,", "chỉ là những con số mà thôi."),
    (3399, "对了小瑶", "Đúng rồi Tiểu Dao,", "xây lại một căn nhà khác,",
     "Đúng rồi Tống Dao,"),
    (3400, "回头把咱们家砸了", "hôm nào quay về đập nát cái nhà của chúng ta đi,",
     "anh chỉ có một yêu cầu duy nhất,", "quay về đập nhà chúng ta đi,"),
    (3401, "重新建一座", "xây lại cái nhà thật to vào.", "đó chính là",
     "xây lại một tòa"),
    (3402, "我只有一个要求", "Anh chỉ có một yêu cầu duy nhất,",
     "Anh chỉ có một yêu cầu duy nhất,", "Tôi chỉ có một yêu cầu,"),
    (3403, "那就是", "đó chính là", "vừa tốt,", "đó chính là"),
    (3404, "又大", "vừa to,", "lại vừa đắt đỏ.", "vừa lớn,"),
    (3405, "又好", "vừa đẹp,", "Còn tiền", "vừa tốt,"),
    (3406, "又贵", "lại vừa sang trọng.", "thì không phải vấn đề.", "vừa đắt."),
    (3407, "钱", "Còn tiền", "Được thôi!", "Còn tiền"),
    (3408, "不是问题", "thì không thành vấn đề.", "Thế còn Từ Tuấn Hào thì sao?",
     "không thành vấn đề."),
    (3409, "好嘞", "Vâng ạ!", "Chẳng phải anh", "Được ạ."),
    (3410, "那徐俊豪呢", "Thế còn Từ Tuấn Hào thì sao?",
     "đã đi dự đám cưới của hắn rồi cơ à?", "Thế còn Từ Tuấn Hào?"),
    (3411, "你不是", "Chẳng phải anh", "Kết quả thế nào rồi?", "Cậu không phải"),
    (3412, "去参加他婚礼了吗", "đã đi dự đám cưới của hắn rồi cơ mà?",
     "Anh đã phá nát đám cưới của Từ Tuấn Hào,", "đã đi dự đám cưới của hắn ta à?"),
    (3413, "情况怎么样了", "Kết quả thế nào rồi?", "khiến hắn và An Tuyết Mạn",
     "Tình hình thế nào rồi?"),
    (3414, "我毁了徐俊豪的婚礼", "Anh đã phá nát đám cưới của Từ Tuấn Hào,",
     "phải chịu sự trừng phạt thích đáng.", "Tôi đã hủy đám cưới của Từ Tuấn Hào,"),
    (3415, "让他和安雪漫", "khiến hắn và An Tuyết Mạn", "Từ nay về sau,",
     "khiến hắn và An Tuyết Mạn"),
    (3416, "得到应有的惩罚", "phải chịu sự trừng phạt thích đáng.",
     "bọn chúng sẽ không bao giờ", "phải chịu sự trừng phạt thích đáng."),
    (3417, "从今往后", "Từ nay về sau,", "dám đến gây sự với chúng ta nữa.",
     "Từ nay về sau,"),
    (3418, "他们再也不敢", "bọn chúng sẽ không bao giờ",
     "Em đã phá nát đám cưới của bọn chúng luôn á?", "bọn chúng sẽ không bao giờ"),
    (3419, "来针对我们了", "dám đến gây sự với chúng ta nữa đâu.", "Tiền em,",
     "dám đến gây sự với chúng ta nữa."),
    (3420, "你毁了他们的婚礼", "Em đã phá nát đám cưới của bọn chúng luôn á?",
     "em đúng là càng ngày càng lợi hại rồi.", "Cậu hủy đám cưới của bọn chúng á?"),
    (3421, "你毁了他们的婚礼", "Cậu phá nát đám cưới của bọn chúng luôn á?",
     "Cậu phá nát đám cưới của bọn chúng luôn á?",
     "Cậu phá nát đám cưới của bọn chúng á?"),
    (3422, "小钱", "Tiền em,", "Tiền em,", "Tống Tiền,"),
    (3423, "你可真是越来越厉害了", "dạo này em càng ngày càng lợi hại rồi đấy.",
     "em đúng là càng ngày càng lợi hại rồi đấy.",
     "em đúng là càng lúc càng lợi hại rồi đấy."),
]
SOURCE = {row[0]: row[1] for row in REAL}
LITERAL = {row[0]: row[2] for row in REAL}
STYLE = {row[0]: row[3] for row in REAL}
REFERENCE = {row[0]: row[4] for row in REAL}


class TestFindStageShifts:
    def test_real_shift_is_found(self) -> None:
        shifts = find_stage_shifts(LITERAL, STYLE)
        # Hai dải cùng lệch +2 đứt ở 3402-3404 được gộp thành một vùng.
        assert [(s.from_index, s.to_index, s.offset) for s in shifts] == [(3395, 3418, 2)]

    def test_in_place_rephrasing_is_not_a_shift(self) -> None:
        assert find_stage_shifts(LITERAL, REFERENCE) == []

    def test_identical_stage_is_clean(self) -> None:
        assert find_stage_shifts(LITERAL, dict(LITERAL)) == []

    def test_short_lines_are_not_evidence(self) -> None:
        previous = {i: f"Dạ{i}." for i in range(1, 20)}
        current = {i: previous.get(i + 1, "") for i in range(1, 20)}
        assert find_stage_shifts(previous, current) == []

    def test_two_stray_lines_are_not_a_run(self) -> None:
        current = dict(LITERAL)
        current[3400], current[3412] = LITERAL[3401], LITERAL[3413]
        assert find_stage_shifts(LITERAL, current) == []


RUN = [i for i in SOURCE if 3405 <= i <= 3418]


class TestJudgeShift:
    def test_independent_reference_sides_with_the_aligned_version(self) -> None:
        assert judge_shift(RUN, LITERAL, STYLE, REFERENCE) == VERDICT_PREVIOUS

    def test_refinement_that_fixed_a_shifted_previous_is_kept(self) -> None:
        """Ca thật khác: bản dịch thô lệch, bản tinh chỉnh căn lại đúng — phải giữ."""
        assert judge_shift(RUN, STYLE, LITERAL, REFERENCE) == VERDICT_CURRENT

    def test_unusable_reference_decides_nothing(self) -> None:
        assert judge_shift(RUN, LITERAL, STYLE, {}) == VERDICT_UNDECIDED

    def test_narrow_win_is_not_enough_to_revert(self) -> None:
        """Bản trước thắng 2 dòng, bản mới thắng 1: chưa đủ gấp 3 lần để lùi."""
        previous = {1: "một hai ba", 2: "bốn năm sáu", 3: "chẳng liên quan gì"}
        current = {1: "khác hẳn nhau", 2: "hoàn toàn khác", 3: "bảy tám chín mười"}
        reference = {1: "một hai ba", 2: "bốn năm sáu", 3: "bảy tám chín mười"}
        assert judge_shift([1, 2, 3], previous, current, reference) == VERDICT_UNDECIDED

    def test_span_padding(self) -> None:
        ordered = sorted(SOURCE)
        span = shift_span(ordered, StageShift(3405, 3418, 2, 13), 1)
        assert span[0] == 3404 and span[-1] == 3419


def _lines(texts: dict[int, str]) -> list[TranslationLine]:
    return [
        TranslationLine(
            index=index,
            start_ms=index * 1000,
            end_ms=index * 1000 + 900,
            text=texts[index],
            original_text=SOURCE[index],
        )
        for index in sorted(texts)
    ]


class _Referee:
    """Dịch trọng tài giả: trả bản dịch độc lập cho vùng được hỏi."""

    def __init__(self, answer: dict[int, str], error: Exception | None = None) -> None:
        self.answer = answer
        self.error = error
        self.calls: list[tuple[int, int]] = []

    def translate_stage(self, *, stage, source_lines, **_kwargs):
        assert stage.kind is TranslationStageKind.LITERAL
        self.calls.append((source_lines[0].index, source_lines[-1].index))
        if self.error is not None:
            raise self.error
        return [replace(line, text=self.answer[line.index]) for line in source_lines]


def _request() -> TranslateSubtitlesRequest:
    return TranslateSubtitlesRequest(
        events=[],
        stages=[
            TranslationStageConfig(kind=TranslationStageKind.LITERAL, model_name="m"),
            TranslationStageConfig(kind=TranslationStageKind.STYLE, model_name="m"),
        ],
        context=TranslationContext(target_lang="Vietnamese"),
    )


def _repair(referee: _Referee, cancel=None) -> dict[int, str]:
    use_case = TranslateSubtitlesUseCase(translator=referee)  # type: ignore[arg-type]
    result = use_case._repair_stage_shifts(
        _lines(STYLE), _lines(LITERAL), _lines(SOURCE), _request(), cancel, "Phong cách"
    )
    return {line.index: line.text for line in result}


class TestUseCaseRepair:
    def test_real_case_is_reverted_to_the_aligned_stage(self) -> None:
        referee = _Referee(REFERENCE)
        fixed = _repair(referee)
        assert len(referee.calls) == 1
        for index in range(3393, 3421):
            assert fixed[index] == LITERAL[index]
        assert fixed[3423] == STYLE[3423]  # ngoài vùng lệch: giữ bản tinh chỉnh

    def test_reference_agreeing_with_refinement_keeps_it(self) -> None:
        assert _repair(_Referee(dict(STYLE))) == STYLE

    def test_referee_failure_keeps_the_stage(self) -> None:
        referee = _Referee(REFERENCE, error=SubtitleTranslationError("503"))
        assert _repair(referee) == STYLE

    def test_user_cancel_is_propagated(self) -> None:
        referee = _Referee(REFERENCE, error=TranslationCancelledError("huỷ"))
        with pytest.raises(TranslationCancelledError):
            _repair(referee, cancel=lambda: True)

    def test_spurious_cancel_is_ignored(self) -> None:
        referee = _Referee(REFERENCE, error=TranslationCancelledError("huỷ giả"))
        assert _repair(referee, cancel=lambda: False) == STYLE

    def test_clean_stage_costs_no_request(self) -> None:
        referee = _Referee(REFERENCE)
        use_case = TranslateSubtitlesUseCase(translator=referee)  # type: ignore[arg-type]
        stage = _lines(LITERAL)
        result = use_case._repair_stage_shifts(
            stage, _lines(LITERAL), _lines(SOURCE), _request(), None, "Phong cách"
        )
        assert result is stage and referee.calls == []


MONEY_SOURCES = {
    2338: "这张卡里有一千万", 3377: "给了我一千万", 3378: "一千万?",
    3380: "这卡里居然有一千万", 3392: "又多了一千万", 3440: "一千万长什么样",
}
MONEY_LITERAL = {
    2338: "Trong thẻ này có mười triệu.", 3377: "còn cho anh mười triệu cơ.",
    3378: "Mười triệu?", 3380: "Trong thẻ này lại có mười triệu,",
    3392: "lại có thêm mười triệu.", 3440: "mười triệu trông như thế nào đã.",
}
MONEY_STYLE = {
    2338: "Trong thẻ này có một triệu.", 3377: "còn cho anh một nghìn vạn cơ.",
    3378: "Một nghìn vạn?", 3380: "Trong thẻ này lại có một nghìn vạn,",
    3392: "lại có thêm một nghìn vạn.", 3440: "một triệu trông như thế nào đã.",
}


class TestNumbersFromPreviousStage:
    def test_stage_alone_reads_every_line_the_same_way(self) -> None:
        # [v3.23.581] Không có cách đọc mẫu nào: dòng SAI GIÁ TRỊ vẫn được đọc chuẩn từ
        # chính giá trị (để nguyên là giọng đọc nói sai số tiền).
        # [v3.23.590] Các dòng "một nghìn vạn" (đúng giá trị, trộn hai lối) cũng được đọc
        # lại theo lối của tiền triệu — trước đây một mình giai đoạn này không thống nhất
        # được vì phim chưa có dòng "mười triệu" nào để làm mẫu.
        fixes = find_number_fixes(MONEY_SOURCES, MONEY_STYLE)
        assert {fix.index: fix.text_after for fix in fixes} == MONEY_LITERAL
        assert {fix.index for fix in fixes if fix.reason == "sai giá trị"} == {2338, 3440}

    def test_previous_stage_reading_restores_every_line(self) -> None:
        fixes = find_number_fixes(
            MONEY_SOURCES, MONEY_STYLE, prior_translations=MONEY_LITERAL
        )
        fixed = {fix.index: fix.text_after for fix in fixes}
        assert fixed == MONEY_LITERAL
        assert {fix.index for fix in fixes if fix.reason == "sai giá trị"} == {2338, 3440}

    def test_use_case_passes_the_previous_stage(self) -> None:
        sources = [
            TranslationLine(index=i, start_ms=0, end_ms=1, text=t, original_text=t)
            for i, t in MONEY_SOURCES.items()
        ]
        stage = [replace(line, text=MONEY_STYLE[line.index]) for line in sources]
        previous = [replace(line, text=MONEY_LITERAL[line.index]) for line in sources]
        result = TranslateSubtitlesUseCase._enforce_number_consistency(
            stage, sources, "Phong cách", previous_lines=previous
        )
        assert {line.index: line.text for line in result} == MONEY_LITERAL


class TestPipelineIntegration:
    """Cả lượt dịch hai giai đoạn: kết quả cuối phải căn đúng dòng như bản dịch thô."""

    def test_execute_returns_aligned_result(self) -> None:
        from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
        from subtitles_extractor.domain.value_objects.time_interval import TimeInterval

        order = sorted(SOURCE)
        by_position = {
            name: {pos: texts[index] for pos, index in enumerate(order, start=1)}
            for name, texts in (
                ("literal", LITERAL), ("style", STYLE), ("reference", REFERENCE)
            )
        }

        class _Pipeline:
            def __init__(self) -> None:
                self.calls: list[tuple[str, int]] = []

            def translate_stage(self, *, stage, source_lines, input_lines, **_kwargs):
                full = len(source_lines) == len(order)
                if stage.kind is TranslationStageKind.LITERAL:
                    texts = by_position["literal" if full else "reference"]
                else:
                    texts = by_position["style"]
                self.calls.append((stage.kind.value, len(source_lines)))
                return [replace(line, text=texts[line.index]) for line in input_lines]

        events = [
            SubtitleEvent(
                index=pos, text=SOURCE[index], interval=TimeInterval(pos, pos + 0.9)
            )
            for pos, index in enumerate(order, start=1)
        ]
        pipeline = _Pipeline()
        use_case = TranslateSubtitlesUseCase(translator=pipeline)  # type: ignore[arg-type]
        request = replace(_request(), events=events)
        response = use_case.execute(request)
        final = [event.text for event in response.events]
        # Dải lệch 3395-3418 (vị trí 3-26) + 2 dòng đệm mỗi bên lùi về bản dịch thô.
        assert final == [LITERAL[i] for i in order[:28]] + [STYLE[i] for i in order[28:]]
        assert [kind for kind, _ in pipeline.calls].count("literal") == 2


class TestDiagnosticsReport:
    def test_final_vs_literal_shift_is_reported(self) -> None:
        from subtitles_extractor.application.services.translation_diagnostics import (
            final_vs_literal_shifts,
        )

        stages = [
            {
                "stage_id": "literal",
                "lines": [{"index": i, "text": t} for i, t in LITERAL.items()],
            }
        ]
        final = [{"index": i, "text": t} for i, t in STYLE.items()]
        # [v3.23.573] 20 dòng: dòng 3404 ("lại vừa đắt đỏ." — câu của dòng 3406 bị cắt
        # lại) nay được tính là bằng chứng yếu nối vào dải.
        assert final_vs_literal_shifts(stages, final) == [
            {"from_index": 3395, "to_index": 3418, "offset": 2, "lines": 20}
        ]

    def test_without_literal_stage_nothing_is_reported(self) -> None:
        from subtitles_extractor.application.services.translation_diagnostics import (
            final_vs_literal_shifts,
        )

        final = [{"index": i, "text": t} for i, t in STYLE.items()]
        assert final_vs_literal_shifts([], final) == []
