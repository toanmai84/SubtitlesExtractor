"""[v3.23.573] Hồi quy: lệch kiểu CẮT LẠI CÂU lọt bộ dò; báo nhầm lệch KHỐI tốn lượt gọi.

Chuẩn đoán thật v3.23.572:

1. Giai đoạn bản địa hoá đẩy dòng 3618-3621 xuống một dòng ("我" ra "từ Liễu Mộ Vân,",
   "柳慕云" ra "chị Đào Hoa,"…) và cắt lại câu, nên chỉ HAI dòng giống hẳn dòng kề; bộ dò
   v3.23.569 đòi ba dòng nên im lặng, bản cuối giữ nguyên lỗi. Trên 23 lượt dịch thật có
   7 dải như vậy.
2. Bộ dò lệch KHỐI theo tương quan độ dài bắt nhầm đoạn câu ngắn 371-390 ở 10/23 lượt
   dịch, mỗi lần bắt dịch lại cả lô. Ở giai đoạn tinh chỉnh/bản địa hoá, 7 lần báo nhầm
   đều căn đúng từng dòng với bản trước — 13 lần lệch thật thì không.
"""

from __future__ import annotations

from dataclasses import replace

from subtitles_extractor.application.services.alignment_check import (
    find_low_correlation_blocks,
)
from subtitles_extractor.application.services.stage_shift_guard import (
    PRIOR_ALIGNED_MIN_LINES,
    SHIFT_MIN_STRICT_LINES,
    aligned_with_previous,
    find_stage_shifts,
)
from subtitles_extractor.application.services.translation_diagnostics import (
    final_vs_literal_shifts,
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
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
)

# Nguyên văn lượt dịch thật: (dòng, câu gốc, dịch thô, tinh chỉnh, bản địa hoá).
RESPLIT = [
    (3612, "想来占走一部分", "muốn chiếm lấy một phần.", "muốn chiếm lấy một phần.",
     "muốn chiếm lấy một phần."),
    (3613, "所以", "Cho nên,", "Cho nên,", "Cho nên,"),
    (3614, "我只占有", "em chỉ chiếm", "em chỉ chiếm", "em chỉ chiếm"),
    (3615, "你一部分就好啦", "một phần của anh là được rồi.",
     "một phần của anh là được rồi.", "một phần của anh là được rồi."),
    (3616, "你说的是哪跟哪啊\uff1f", "Em đang nói cái quái gì thế hả?",
     "Em đang nói cái quái gì thế hả?", "Em đang nói cái quái gì thế hả?"),
    (3617, "你别以为我看不出来", "Anh đừng tưởng em không nhận ra,",
     "Anh đừng tưởng em không nhận ra,", "Anh đừng tưởng em không nhận ra,"),
    (3618, "我", "em,", "em,", "từ Liễu Mộ Vân,"),
    (3619, "柳慕云", "Liễu Mộ Vân,", "Liễu Mộ Vân,", "chị Đào Hoa,"),
    (3620, "桃花嫂子", "chị Đào Hoa,", "chị Đào Hoa,", "cho tới chị Xuân Hoa,"),
    (3621, "春花嫂子", "chị Xuân Hoa", "chị Xuân Hoa", "ai nấy đều đang nhìn chằm chằm"),
    (3622, "都对你虎视眈眈呢", "đều đang nhìn chằm chằm như hổ đói vào anh đấy.",
     "đều đang nhìn chằm chằm như hổ đói vào anh đấy.", "như hổ đói vào anh đấy."),
    (3623, "哎呀呀", "Ơ hay.", "Ơ hay.", "Ơ hay."),
    (3624, "打住打住", "Thôi đi em.", "Thôi đi em.", "Thôi đi em."),
    (3625, "其他人就算了", "Người khác thì không nói làm gì,",
     "Người khác thì không nói làm gì,", "Người khác thì không nói làm gì,"),
    (3626, "春花嫂子", "còn chị Xuân Hoa", "còn chị Xuân Hoa", "còn chị Xuân Hoa"),
]
SOURCE = {row[0]: row[1] for row in RESPLIT}
LITERAL = {row[0]: row[2] for row in RESPLIT}
STYLE = {row[0]: row[3] for row in RESPLIT}
LOCALIZE = {row[0]: row[4] for row in RESPLIT}


class TestResplitShift:
    def test_real_resplit_is_found(self) -> None:
        shifts = find_stage_shifts(STYLE, LOCALIZE)
        assert [(s.from_index, s.to_index, s.offset, s.lines) for s in shifts] == [
            (3618, 3621, 1, 4)
        ]

    def test_two_strong_lines_alone_are_not_a_shift(self) -> None:
        """Hai dòng lệch mà không dòng kề nào mang dấu vết cắt lại câu: vẫn chưa đủ."""
        current = dict(STYLE)
        current[3618], current[3619] = LOCALIZE[3618], LOCALIZE[3619]
        assert find_stage_shifts(STYLE, current) == []

    def test_one_strong_line_is_never_extended(self) -> None:
        assert SHIFT_MIN_STRICT_LINES == 2
        current = dict(STYLE)
        current[3619] = LOCALIZE[3619]
        current[3620], current[3621] = LOCALIZE[3620], LOCALIZE[3621]
        assert find_stage_shifts(STYLE, current) == []

    def test_weak_line_with_another_offset_is_not_joined(self) -> None:
        current = dict(LOCALIZE)
        current[3620] = "chị Đào Hoa, cho tới"  # giống dòng 3620 của bản trước hơn
        current[3621] = STYLE[3621]
        assert find_stage_shifts(STYLE, current) == []

    def test_final_vs_literal_report_now_shows_it(self) -> None:
        stages = [
            {
                "stage_id": "literal",
                "lines": [{"index": i, "text": t} for i, t in LITERAL.items()],
            }
        ]
        final = [{"index": i, "text": t} for i, t in LOCALIZE.items()]
        assert final_vs_literal_shifts(stages, final) == [
            {"from_index": 3618, "to_index": 3621, "offset": 1, "lines": 4}
        ]


def _lines(texts: dict[int, str]) -> list[TranslationLine]:
    return [
        TranslationLine(
            index=index, start_ms=index * 1000, end_ms=index * 1000 + 900,
            text=texts[index], original_text=SOURCE[index],
        )
        for index in sorted(texts)
    ]


class _Referee:
    """Dịch trọng tài giả: bản dịch thô mới cho vùng tranh chấp (ở đây: bản thô thật)."""

    def __init__(self) -> None:
        self.calls = 0

    def translate_stage(self, *, stage, source_lines, **_kwargs):
        assert stage.kind is TranslationStageKind.LITERAL
        self.calls += 1
        return [replace(line, text=LITERAL[line.index]) for line in source_lines]


class TestUseCaseRepair:
    def test_resplit_stage_is_reverted(self) -> None:
        referee = _Referee()
        request = TranslateSubtitlesRequest(
            events=[],
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
        use_case = TranslateSubtitlesUseCase(translator=referee)  # type: ignore[arg-type]
        result = use_case._repair_stage_shifts(
            _lines(LOCALIZE), _lines(STYLE), _lines(SOURCE), request, None, "Bản địa hoá"
        )
        fixed = {line.index: line.text for line in result}
        assert referee.calls == 1
        for index in range(3618, 3622):
            assert fixed[index] == STYLE[index]


# Lô 368-394 của lượt dịch thật (giai đoạn bản địa hoá bị báo lệch khối 374-390):
# (dòng, câu gốc, bản tinh chỉnh = đầu vào, bản model trả về).
FALSE_ALARM = [
    (368, "我们家小宋钱最棒了", "tiểu Tống Tiền nhà chúng ta là tuyệt nhất.",
     "tiểu Tống Tiền nhà chúng ta là tuyệt nhất."),
    (369, "这是?", "Đây là...?", "Đây là...?"),
    (370, "嫂子", "Chị Đào Hoa,", "Chị Đào Hoa,"),
    (371, "我能不能看看你", "em có thể xem", "Em có thể xem"),
    (372, "脖子上的玉佩", "ngọc bội trên cổ chị không?", "ngọc bội trên cổ chị không?"),
    (373, "玉佩?", "Ngọc bội?", "Ngọc bội?"),
    (374, "是说这个吗", "Ý em là cái này à?", "Ý em là cái này à?"),
    (375, "果然没错", "Quả nhiên không sai!", "Quả nhiên không sai!"),
    (376, "没想到这么快", "Không ngờ nhanh thế", "Không ngờ nhanh thế"),
    (377, "就遇到第二块", "đã gặp được mảnh", "đã gặp được mảnh"),
    (378, "元神碎片了", "nguyên thần thứ hai.", "nguyên thần thứ hai rồi."),
    (379, "嫂子", "Chị Đào Hoa,", "Chị Đào Hoa,"),
    (380, "你能不能把这个玉佩", "chị có thể nhượng lại", "chị có thể nhượng lại"),
    (381, "卖给我", "miếng ngọc bội này cho em không?",
     "miếng ngọc bội này cho em không?"),
    (382, "你要这玉佩干啥", "Em cần ngọc bội này làm gì?",
     "Em muốn ngọc bội này làm gì?"),
    (383, "不过这玉佩啊", "Nhưng ngọc bội này", "Nhưng ngọc bội này"),
    (384, "是我早年前", "là từ nhiều năm trước", "là từ nhiều năm trước"),
    (385, "在山间捡到的", "chị nhặt được trên núi.", "chị nhặt được trên núi."),
    (386, "倒也不值钱", "Cũng chẳng đáng tiền đâu,", "Cũng chẳng đáng tiền đâu,"),
    (387, "你要是想要", "nếu em muốn", "nếu em muốn"),
    (388, "嫂子送给你都行", "thì chị tặng luôn cho em cũng được.",
     "thì chị tặng luôn cho em cũng được."),
    (389, "真的吗太好了", "Thật á? Tốt quá rồi!", "Thật á? Tốt quá rồi!"),
    (390, "谢谢嫂子", "Cảm ơn chị Đào Hoa.", "Cảm ơn chị Đào Hoa."),
    (391, "想要玉佩的话", "Nhưng nếu muốn có ngọc bội,", "Nhưng nếu muốn có ngọc bội,"),
    (392, "那你就自己来拿吧", "thì em tự đến mà lấy đi.", "thì tự mình đến lấy đi."),
    (393, "嫂子", "Chị Đào Hoa,", "Chị Đào Hoa..."),
    (394, "别开玩笑了", "đừng đùa nữa.", "đừng đùa nữa."),
]

# Lô 1888-1917 của lượt dịch thật (giai đoạn tinh chỉnh LỆCH THẬT: mỗi dòng mang câu của
# dòng sau): (dòng, câu gốc, bản dịch thô = đầu vào, bản model trả về).
REAL_SHIFT = [
    (1888, "对她说", "Đúng vậy,", "Đúng vậy,"),
    (1889, "她跟我道谢", "cô ấy gọi đến cảm ơn anh,", "tiện thể"),
    (1890, "顺便", "tiện thể", "cùng Viện trưởng Tôn"),
    (1891, "和江城医院的", "đi cùng Viện trưởng Tôn", "của Bệnh viện Giang Thành."),
    (1892, "孙院长一起", "của Bệnh viện Giang Thành.", "Ha ha ha ha ha!"),
    (1893, "哈哈哈哈哈哈", "Ha ha ha ha ha!", "Tống Tiền ơi là Tống Tiền,"),
    (1894, "宋钱啊宋钱", "Tống Tiền ơi là Tống Tiền,",
     "bọn tao vừa nhắc đến đại tiểu thư nhà họ Liễu xong,"),
    (1895, "我们刚聊完柳大小姐", "bọn tao vừa nhắc đến đại tiểu thư nhà họ Liễu xong,",
     "mày liền giả vờ gọi điện"),
    (1896, "你就假装和柳大小姐", "mày liền giả vờ gọi điện",
     "với đại tiểu thư nhà họ Liễu luôn cơ đấy."),
    (1897, "打电话", "với đại tiểu thư nhà họ Liễu luôn cơ đấy.",
     "Mày tưởng tụi tao ngu lắm chắc?"),
    (1898, "真当我们俩傻啊\uff1f", "Mày tưởng tụi tao ngu lắm chắc?",
     "Còn bày đặt nói Viện trưởng Tôn cùng đến cơ à,"),
    (1899, "还说什么孙院长一起来", "Còn bày đặt nói Viện trưởng Tôn cùng đến cơ à,",
     "đến làm gì?"),
    (1900, "来干嘛?", "đến làm gì?", "Để xem tụi mày bị"),
    (1901, "看你们怎么被", "Để xem tụi mày bị", "đánh ra nông nỗi gì à?"),
    (1902, "打成残废啊", "đánh ra nông nỗi gì à?", "Các người biết cái gì chứ!"),
    (1903, "你们懂什么\uff01", "Các người biết cái gì chứ!", "Anh tôi mới vừa cứu"),
    (1904, "我哥可是刚救了", "Anh tôi vừa mới cứu", "mạng của Liễu Mộ Vân đấy."),
    (1905, "柳慕云的命", "mạng của Liễu Mộ Vân đấy.", "Cô ta đến thì có gì mà phải lạ?"),
    (1906, "她过来有什么奇怪的吗\uff1f", "Cô ta đến thì có gì mà phải lạ?",
     "Mày nói cái gì?"),
    (1907, "你说什么?", "Mày nói cái gì?", "Mày nói cái gì cơ?"),
    (1908, "你说什么?", "Mày nói cái gì cơ?",
     "Mày là ân nhân cứu mạng của Liễu Mộ Vân á?"),
    (1909, "你是柳慕云的救命恩人\uff1f", "Mày là ân nhân cứu mạng của Liễu Mộ Vân á?",
     "Đúng vậy đấy."),
    (1910, "对啊", "Đúng vậy đấy.", "Anh tôi"),
    (1911, "我哥", "Anh tôi", "bây giờ chính là"),
    (1912, "现在可是柳慕云的", "bây giờ chính là ân nhân",
     "ân nhân cứu mạng của Liễu Mộ Vân."),
    (1913, "救命恩人", "cứu mạng của Liễu Mộ Vân.", "Đợi cô ta tới xem,"),
    (1914, "等她过来了", "Đợi cô ta tới xem,", "tôi nhất định sẽ bắt cô ta"),
    (1915, "我一定要让她", "tôi nhất định sẽ bắt cô ta",
     "cho các người một bài học nhớ đời."),
    (1916, "好好收拾你们", "cho các người một trận ra trò.",
     "cho các người một trận ra trò."),
    (1917, "哈哈哈", "Ha ha ha.", "Ha ha ha."),
]


def _batch(rows: list[tuple[int, str, str, str]]) -> tuple[list, list]:
    batch = [
        TranslationLine(
            index=row[0], start_ms=row[0] * 1000, end_ms=row[0] * 1000 + 900,
            text=row[2], original_text=row[1],
        )
        for row in rows
    ]
    result = [replace(line, text=row[3]) for line, row in zip(batch, rows, strict=True)]
    return batch, result


def _translator() -> GeminiSubtitleTranslator:
    # Bảng tên/thuật ngữ rỗng: chỉ còn bộ dò KHỐI theo tương quan độ dài — đúng lớp kiểm
    # đã bắt nhầm trong ca thật.
    translator = GeminiSubtitleTranslator.__new__(GeminiSubtitleTranslator)
    translator._active_characters = ""
    translator._active_glossary = ""
    return translator


def _texts(rows: list[tuple[int, str, str, str]], column: int) -> dict[int, str]:
    return {row[0]: row[column] for row in rows}


class TestBlockFalseAlarm:
    def test_length_correlation_still_raises_the_alarm(self) -> None:
        """Tiền đề: tương quan độ dài vẫn sụp ở đoạn này như ca thật (-0,036)."""
        blocks = find_low_correlation_blocks(
            _texts(FALSE_ALARM, 1), _texts(FALSE_ALARM, 3)
        )
        assert [(b["from_index"], b["to_index"]) for b in blocks] == [(374, 390)]

    def test_aligned_with_previous_stage_is_dismissed(self) -> None:
        batch, result = _batch(FALSE_ALARM)
        assert _translator()._detect_batch_misalignment(batch, result) == []

    def test_real_block_shift_is_still_caught(self) -> None:
        batch, result = _batch(REAL_SHIFT)
        findings = _translator()._detect_batch_misalignment(batch, result)
        assert findings and findings[0]["index"] == 1890

    def test_literal_stage_has_nothing_to_compare_with(self) -> None:
        """Dịch thô: đầu vào là câu gốc chữ Hán — giữ nguyên cách cũ (vẫn dịch lại)."""
        rows = [(row[0], row[1], row[1], row[3]) for row in FALSE_ALARM]
        batch, result = _batch(rows)
        assert _translator()._detect_batch_misalignment(batch, result)

    def test_dismissal_is_logged_as_a_process_event(self) -> None:
        from subtitles_extractor.application.services.translation_process_log import (
            get_process_log,
        )

        get_process_log().clear()
        try:
            batch, result = _batch(FALSE_ALARM)
            _translator()._detect_batch_misalignment(batch, result)
            events = get_process_log().snapshot()["events"]
        finally:
            get_process_log().clear()
        assert [e["event"] for e in events] == ["batch_misaligned_dismissed"]
        assert events[0]["lines"] == [374, 390]


class TestAlignedWithPrevious:
    def test_real_cases(self) -> None:
        false_alarm = list(range(374, 391))
        real = list(range(1890, 1916))
        assert aligned_with_previous(
            _texts(FALSE_ALARM, 2), _texts(FALSE_ALARM, 3), false_alarm
        )
        assert not aligned_with_previous(
            _texts(REAL_SHIFT, 2), _texts(REAL_SHIFT, 3), real
        )

    def test_too_little_evidence_decides_nothing(self) -> None:
        indices = [374, 375, 376, 377][: PRIOR_ALIGNED_MIN_LINES - 1]
        assert not aligned_with_previous(
            _texts(FALSE_ALARM, 2), _texts(FALSE_ALARM, 3), indices
        )
