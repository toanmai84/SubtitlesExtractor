"""[v3.23.589] Hồi quy: giai đoạn tinh chỉnh VIẾT LẠI cả một dải dòng thành lời khác.

Chuẩn đoán thật v3.23.588, phim "灭门惨案II" (ba giai đoạn bằng gemini-3.5-flash-lite),
cảnh dân làng xếp hàng ở đồn công an, dòng 68-77: dịch thô đúng ("不要生氣" ra "Đừng
giận."), giai đoạn phong cách đổi thành lời đám đông ("Mở cửa!", "Nhanh lên!"). Bản
cuối sai 7 dòng. Trên 14 lượt dịch phim này: bản thô đúng cả dải ở cả 14 lượt, bản cuối
có dòng bị viết lại trong dải ở 8 lượt (4 lượt từ ba dòng trở lên).

Bộ dò lệch dòng (v3.23.569) không thấy: nó tìm dòng giống một dòng KỀ của bản trước
(≥ 0,6), còn ở đây lời mới không giống dòng nào. Tín hiệu mới: độ giống với CHÍNH dòng
đó của bản trước dưới 0,2 ở ít nhất ba dòng gần nhau (cách nhau không quá ba dòng).

Viết lại chưa nói ai đúng: trên 106 giai đoạn tinh chỉnh thật có 23 dải như vậy, trong đó
12 dải là giai đoạn tinh chỉnh SỬA một bản trước lệch dòng, 8 dải làm hỏng bản trước
đúng. Vì vậy dải viết lại đi qua cùng trọng tài với dải lệch. Mọi dữ liệu dưới đây là
nguyên văn; cột trọng tài là bản dịch thô của CHÍNH phim đó ở một lượt chạy khác.
"""

# ruff: noqa: RUF001 — dấu câu fullwidth CJK thuộc câu gốc chép từ phim

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from subtitles_extractor.application.services.stage_shift_guard import (
    REWRITE_MAX_GAP,
    REWRITE_MIN_RUN,
    SHIFT_MAX_GAP,
    VERDICT_CURRENT,
    VERDICT_PREVIOUS,
    StageShift,
    find_stage_rewrites,
    find_stage_shifts,
    judge_shift,
)
from subtitles_extractor.application.services.translation_diagnostics import (
    build_diagnostics_bundle,
    final_vs_literal_rewrites,
    final_vs_literal_shifts,
)
from subtitles_extractor.application.services.translation_process_log import (
    get_process_log,
)
from subtitles_extractor.application.use_cases.translate_subtitles import (
    MAX_SHIFT_ARBITRATIONS_PER_STAGE,
    TranslateSubtitlesRequest,
    TranslateSubtitlesUseCase,
)
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    SubtitleTranslationError,
    TranslationContext,
    TranslationLine,
    TranslationStageConfig,
    TranslationStageKind,
)

# Lượt v3.23.588: (dòng, câu gốc, dịch thô, phong cách, dịch thô của lượt v3.23.587).
QUEUE_SCENE = [
    (64, "簡直是英雄本色，靈氣迫人呀",
     "Đúng là anh hùng bản sắc, khí phách ngút trời!",
     "Đúng là anh hùng bản sắc, khí phách ngút trời!",
     "Quả đúng là anh hùng bản sắc, khí phách ngút trời!"),
    (65, "加上小小淫賤，小白臉",
     "Thêm chút dâm tiện, đúng chất tiểu bạch kiểm.",
     "Thêm chút dâm tiện, đúng chất tiểu bạch kiểm.",
     "Lại thêm chút dâm tiện, lãng tử..."),
    (66, "你中了進口的錄像帶毒嗎",
     "Anh trúng độc băng hình nhập khẩu rồi à?",
     "Anh trúng độc băng hình nhập khẩu rồi à?",
     "Trưởng đồn nhiễm độc băng hình nhập khẩu rồi à?"),
    (67, "叫了你不要看那麼多進口港產片的啊",
     "Đã bảo cậu đừng xem lắm phim Hồng Kong rồi cơ mà.",
     "Đã bảo cậu đừng xem lắm phim Hồng Kong rồi cơ mà.",
     "Đã bảo cậu đừng xem nhiều phim Hồng Kong rồi mà."),
    (68, "不要生氣", "Đừng giận.", "Mở cửa!", "Đừng giận."),
    (69, "排好隊，一個跟一個",
     "Xếp hàng đi, từng người một!", "Xếp hàng đi, từng người một!",
     "Xếp hàng đi, từng người một!"),
    (70, "這麼久", "Lâu thế nhỉ.", "Nhanh lên!", "Chờ lâu thế"),
    (71, "還關門呢", "Lại còn đóng cửa nữa.", "Nhanh lên!", "mà còn đóng cửa nữa."),
    (72, "這次一定派很多豬肉了",
     "Lần này chắc chắn chia rất nhiều thịt heo đây.",
     "Lần này chắc chắn chia rất nhiều thịt heo đây.",
     "Lần này chắc chắn chia được rất nhiều thịt heo."),
    (73, "攪甚麼呀", "Làm cái gì thế hả?", "Sao lại lâu thế này chứ?",
     "Làm cái gì thế?"),
    (74, "你有甚麼理由要分多一點？",
     "Sao ông lại đòi phần nhiều hơn chứ?", "Đến lúc đó đợi thế nào được?",
     "Mấy người có lý do gì mà đòi chia nhiều hơn?"),
    (75, "全部肅靜唔好嘈!唔好嘈!",
     "Tất cả trật tự, câm mồm! Không được ồn!",
     "Cửa cũng đóng rồi, chắc chắn chia nhiều thịt heo đây!",
     "Tất cả trật tự, không được ồn! Không được ồn!"),
    (76, "我們大隊長現在有話對你們說",
     "Đại đội trưởng của chúng ta có lời muốn nói với các người.",
     "Lần này tôi phải được nhiều hơn.",
     "Đại đội trưởng của chúng ta có lời muốn nói với các vị."),
    (77, "你沒有事嗎", "Cậu không sao chứ?", "Tính lão già thế nào?",
     "Cậu không sao chứ?"),
    (78, "各位，我今天騙大家來的目的呢",
     "Bà con ơi, mục đích tôi lừa mọi người tới đây...",
     "Bà con ơi, mục đích tôi lừa mọi người tới đây...",
     "Thưa bà con, mục đích tôi lừa mọi người tới đây..."),
    (79, "是想找出那個是殺洪震一家的凶手",
     "là muốn tìm ra kẻ sát hại cả nhà Hồng Chấn.",
     "là muốn tìm ra kẻ sát hại cả nhà Hồng Chấn.",
     "là muốn tìm ra hung thủ sát hại cả nhà Hồng Chấn."),
    (80, "關我們甚麼事呀",
     "Thì liên quan gì đến chúng tôi chứ?", "Thì liên quan gì đến chúng tôi chứ?",
     "Thì liên quan gì đến chúng tôi?"),
    (81, "我要收集大家的精液拿去化驗",
     "Tôi muốn thu thập tinh dịch của mọi người đi xét nghiệm.",
     "Tôi muốn thu thập tinh dịch của mọi người đi xét nghiệm.",
     "Tôi muốn thu thập tinh dịch của tất cả để mang đi xét nghiệm."),
]
SOURCE = {row[0]: row[1] for row in QUEUE_SCENE}
LITERAL = {row[0]: row[2] for row in QUEUE_SCENE}
STYLE = {row[0]: row[3] for row in QUEUE_SCENE}
OTHER_RUN_LITERAL = {row[0]: row[4] for row in QUEUE_SCENE}

# Bản CUỐI của lượt v3.23.588 (sau cả giai đoạn địa hoá) — file chuẩn đoán không có cờ
# nào cho dải này: final_vs_literal_shifts = 0.
FINAL = {
    64: "Đúng là anh hùng bản sắc, khí phách ngút trời!",
    65: "Lại thêm chút dâm tiện, đúng chất tiểu bạch kiểm.",
    66: "Cậu trúng độc băng hình nhập khẩu rồi à?",
    67: "Đã bảo cậu đừng xem lắm phim Hồng Kong rồi cơ mà.",
    68: "Khai mau lên!",
    69: "Xếp hàng đi, một người theo một người!",
    70: "Nhanh nhanh nhanh lên!",
    71: "Nhanh lên!",
    72: "Sao đợi lâu thế này?",
    73: "Làm cái gì thế hả?",
    74: "Cái này đợi đến khi nào mới tới lượt?",
    75: "Tất cả im lặng, không được ồn, không được ồn!",
    76: "Cửa cũng đóng rồi, rốt cuộc chia thêm chút thịt heo gì chứ?",
    77: "Đến lượt tôi rồi, tôi được nhiều hơn đấy nhé!",
    78: "Bà con ơi, mục đích tôi lừa mọi người tới đây...",
    79: "là muốn tìm ra kẻ sát hại cả nhà Hồng Chấn.",
    80: "Liên quan gì đến chúng tôi chứ?",
    81: "Tôi muốn thu thập tinh dịch của mọi người để đem đi xét nghiệm.",
}

# Lượt v3.23.587, cùng cảnh: phong cách còn đúng, bản ĐỊA HOÁ mới viết lại 68-77.
# (dòng, phong cách, địa hoá) — câu gốc và trọng tài (dịch thô v3.23.588) lấy ở trên.
LOCALIZE_CASE = [
    (64, "Đúng là anh hùng bản sắc, khí phách ngút trời!",
     "Đúng là anh hùng bản sắc, khí phách ngút trời!"),
    (65, "Lại thêm chút dâm tiện, mặt búng ra sữa...",
     "Lại còn mang chút dâm tiện, trông búng ra sữa..."),
    (66, "Trưởng đồn xem nhiều phim ngoại nhập quá nên nhiễm độc rồi à?",
     "Trưởng đồn, anh nhiễm độc băng hình nhập khẩu từ Hong Kong rồi đấy à?"),
    (67, "Đã bảo cậu đừng xem lắm phim Hồng Kong rồi mà.",
     "Đã bảo cậu đừng xem lắm phim Hong Kong rồi mà."),
    (68, "Đừng giận.", "Mở cửa mau!"),
    (69, "Xếp hàng đi, từng người một!", "Xếp hàng đi, từng người một!"),
    (70, "Sao lâu thế", "Nhanh lên, nhanh lên!"),
    (71, "mà còn đóng cửa nữa.", "Xếp hàng đi!"),
    (72, "Lần này chắc chắn chia được khối thịt heo.",
     "Sao thế này, sao làm ăn lâu thế nhỉ, chờ đến bao giờ hả?"),
    (73, "Làm gì thế?", "Cửa đóng kín hết rồi,"),
    (74, "Các người có lý do gì mà đòi chia nhiều hơn?",
     "hôm nay chắc chia thêm ít thịt heo đây."),
    (75, "Tất cả trật tự, đừng ồn! Đừng ồn!", "Tất cả trật tự, đừng ồn! Đừng ồn!"),
    (76, "Đội trưởng của chúng ta có lời muốn nói với mọi người.",
     "Đội trưởng của chúng ta có lời muốn nói, mọi người hoan nghênh nào!"),
    (77, "Cậu không sao chứ?", "Trưởng đồn..."),
    (78, "Bà con ơi, mục đích tôi lừa mọi người tới đây...",
     "Bà con ơi, tôi lừa mọi người đến đây mục đích là..."),
    (79, "là muốn tìm ra hung thủ sát hại cả nhà Hồng Chấn.",
     "muốn tìm ra kẻ đã sát hại cả nhà Hồng Chấn."),
    (80, "Thì liên quan gì đến chúng tôi?", "Thì liên quan gì đến chúng tôi?"),
    (81, "Tôi muốn thu thập tinh dịch của tất cả để mang đi xét nghiệm.",
     "Tôi muốn thu thập tinh dịch của tất cả đem đi xét nghiệm!"),
]
STYLE_587 = {row[0]: row[1] for row in LOCALIZE_CASE}
LOCALIZE_587 = {row[0]: row[2] for row in LOCALIZE_CASE}

# Chiều NGƯỢC LẠI, phim khác (lượt v3.23.578): dịch thô lệch dòng 3401-3403 ("重新建一座"
# mang câu "Anh chỉ có một yêu cầu thôi,"), giai đoạn phong cách SỬA lại đúng.
# (dòng, câu gốc, dịch thô, phong cách, dịch thô của lượt v3.23.570).
REBUILD_SCENE = [
    (3397, "钱对于我来说", "Tiền đối với em", "Tiền đối với em", "Tiền đối với em"),
    (3398, "只是一串数字而已", "cũng chỉ là những con số.", "cũng chỉ là những con số.",
     "chỉ là những con số mà thôi."),
    (3399, "对了小瑶", "Đúng rồi Tiểu Dao,", "Đúng rồi Tiểu Dao,", "Đúng rồi Tống Dao,"),
    (3400, "回头把咱们家砸了",
     "hôm nào đập nhà mình đi xây lại biệt thự đi.",
     "hôm nào đập nhà mình đi xây lại biệt thự đi.",
     "quay về đập cái nhà của chúng ta đi,"),
    (3401, "重新建一座", "Anh chỉ có một yêu cầu thôi,", "Xây lại một căn khác,",
     "xây lại cái khác."),
    (3402, "我只有一个要求", "đó là", "anh chỉ có một yêu cầu thôi,",
     "Anh chỉ có một yêu cầu,"),
    (3403, "那就是", "phải thật to,", "đó là", "đó chính là"),
    (3404, "又大", "Vừa to", "vừa to", "phải thật lớn,"),
    (3405, "又好", "vừa đẹp", "vừa đẹp", "thật đẹp,"),
    (3406, "又贵", "vừa đắt.", "vừa đắt.", "thật đắt đỏ."),
    (3407, "钱", "Tiền", "Tiền", "Tiền"),
]
REBUILD_SOURCE = {row[0]: row[1] for row in REBUILD_SCENE}
REBUILD_LITERAL = {row[0]: row[2] for row in REBUILD_SCENE}
REBUILD_STYLE = {row[0]: row[3] for row in REBUILD_SCENE}
REBUILD_OTHER_RUN = {row[0]: row[4] for row in REBUILD_SCENE}


class TestFindStageRewrites:
    def test_shift_detector_is_blind_to_the_real_case(self) -> None:
        """Lý do cần bộ dò mới: lời viết lại không giống dòng kề nào của bản trước."""
        assert find_stage_shifts(LITERAL, STYLE) == []

    def test_real_case_is_one_region(self) -> None:
        # 8 dòng viết lại; 69 và 72 nằm giữa dải nhưng còn nguyên.
        assert find_stage_rewrites(LITERAL, STYLE) == [StageShift(68, 77, 0, 8)]

    def test_region_is_marked_as_rewrite_not_shift(self) -> None:
        (region,) = find_stage_rewrites(LITERAL, STYLE)
        assert region.is_rewrite
        assert not StageShift(3405, 3418, 2, 13).is_rewrite

    def test_localize_stage_case(self) -> None:
        # Dòng 77 ("Cậu không sao chứ?" thành "Trưởng đồn...") cách dòng 74 ba dòng:
        # vẫn thuộc dải — lời bịa hay chen giữa hai dòng còn nguyên.
        assert find_stage_rewrites(STYLE_587, LOCALIZE_587) == [
            StageShift(68, 77, 0, 7)
        ]

    def test_gap_is_one_line_wider_than_for_shifts(self) -> None:
        """Ca thật v3.23.584: phong cách viết lại 68, 71, 73 — khe 3 rồi khe 2."""
        current = dict(LITERAL)
        for index in (68, 71, 73):
            current[index] = STYLE[index]
        assert REWRITE_MAX_GAP == SHIFT_MAX_GAP + 1 == 3
        assert find_stage_rewrites(LITERAL, current) == [StageShift(68, 73, 0, 3)]

    def test_ordinary_refinement_is_not_a_rewrite(self) -> None:
        """Gọt câu, đổi xưng hô thật của lượt v3.23.587 (dịch thô → phong cách)."""
        assert find_stage_rewrites(OTHER_RUN_LITERAL, STYLE_587) == []

    def test_unchanged_stage(self) -> None:
        assert find_stage_rewrites(LITERAL, dict(LITERAL)) == []

    def test_isolated_rewritten_lines_are_left_alone(self) -> None:
        """Hai dòng viết lại chưa thành dải: dòng lẻ có cả sửa lẫn hỏng, không phân xử."""
        current = dict(LITERAL)
        current[68], current[70] = STYLE[68], STYLE[70]
        assert REWRITE_MIN_RUN == 3
        assert find_stage_rewrites(LITERAL, current) == []

    def test_far_apart_rewritten_lines_are_not_one_region(self) -> None:
        current = dict(LITERAL)
        for index in (68, 73, 77):
            current[index] = STYLE[index]
        assert find_stage_rewrites(LITERAL, current) == []

    def test_one_word_lines_are_not_evidence(self) -> None:
        previous = {1: "Được.", 2: "Ừ.", 3: "Hả?", 4: "Đi."}
        current = {1: "Vâng.", 2: "Phải.", 3: "Gì?", 4: "Nào."}
        assert find_stage_rewrites(previous, current) == []

    def test_emptied_lines_are_not_evidence(self) -> None:
        current = dict.fromkeys(LITERAL, "")
        assert find_stage_rewrites(LITERAL, current) == []

    def test_region_inside_a_known_shift_is_skipped(self) -> None:
        """Dải đã có phân xử riêng của bộ dò lệch: không hỏi trọng tài lần hai."""
        known = [StageShift(66, 70, 1, 3)]
        assert find_stage_rewrites(LITERAL, STYLE, known) == []

    def test_fixing_a_shifted_previous_also_looks_like_a_rewrite(self) -> None:
        """Bộ dò KHÔNG biết ai đúng — vì vậy không được tự lùi, phải qua trọng tài."""
        assert find_stage_rewrites(REBUILD_LITERAL, REBUILD_STYLE) == [
            StageShift(3401, 3403, 0, 3)
        ]


class TestJudgeRewrite:
    def test_real_case_previous_wins(self) -> None:
        region = list(range(68, 78))
        assert (
            judge_shift(region, LITERAL, STYLE, OTHER_RUN_LITERAL) == VERDICT_PREVIOUS
        )

    def test_localize_case_previous_wins(self) -> None:
        region = list(range(68, 78))
        assert judge_shift(region, STYLE_587, LOCALIZE_587, LITERAL) == VERDICT_PREVIOUS

    def test_refinement_that_fixed_a_shifted_previous_wins(self) -> None:
        region = [3401, 3402, 3403]
        verdict = judge_shift(region, REBUILD_LITERAL, REBUILD_STYLE, REBUILD_OTHER_RUN)
        assert verdict == VERDICT_CURRENT


def _lines(texts: dict[int, str], sources: dict[int, str]) -> list[TranslationLine]:
    return [
        TranslationLine(
            index=index,
            start_ms=index * 1000,
            end_ms=index * 1000 + 900,
            text=texts[index],
            original_text=sources[index],
        )
        for index in sorted(texts)
    ]


class _Referee:
    """Dịch trọng tài giả: trả bản dịch độc lập cho đúng vùng được hỏi."""

    def __init__(self, answer: dict[int, str], error: Exception | None = None) -> None:
        self.answer = answer
        self.error = error
        self.calls: list[tuple[int, int]] = []
        self.attach_video: list[bool] = []

    def translate_stage(
        self, *, stage: Any, source_lines: list[TranslationLine], **kwargs: Any
    ) -> list[TranslationLine]:
        assert stage.kind is TranslationStageKind.LITERAL
        self.calls.append((source_lines[0].index, source_lines[-1].index))
        self.attach_video.append(kwargs["attach_video"])
        if self.error is not None:
            raise self.error
        return [replace(line, text=self.answer[line.index]) for line in source_lines]


def _request(with_literal: bool = True) -> TranslateSubtitlesRequest:
    kinds = [TranslationStageKind.STYLE, TranslationStageKind.LOCALIZE]
    if with_literal:
        kinds.insert(0, TranslationStageKind.LITERAL)
    return TranslateSubtitlesRequest(
        events=[],
        stages=[TranslationStageConfig(kind=kind, model_name="m") for kind in kinds],
        context=TranslationContext(target_lang="Vietnamese"),
    )


def _repair(
    referee: _Referee,
    previous: dict[int, str],
    current: dict[int, str],
    sources: dict[int, str],
    request: TranslateSubtitlesRequest | None = None,
) -> dict[int, str]:
    use_case = TranslateSubtitlesUseCase(translator=referee)  # type: ignore[arg-type]
    result = use_case._repair_stage_shifts(
        _lines(current, sources), _lines(previous, sources), _lines(sources, sources),
        request or _request(), None, "Phong cách",
    )
    return {line.index: line.text for line in result}


def _events(name: str) -> list[dict[str, Any]]:
    return [e for e in get_process_log().snapshot()["events"] if e["event"] == name]


@pytest.fixture(autouse=True)
def _fresh_log() -> None:
    get_process_log().clear()


class TestUseCaseRepair:
    def test_real_case_is_reverted_to_the_literal_stage(self) -> None:
        referee = _Referee(OTHER_RUN_LITERAL)
        fixed = _repair(referee, LITERAL, STYLE, SOURCE)
        assert fixed == LITERAL
        assert fixed[68] == "Đừng giận." and fixed[77] == "Cậu không sao chứ?"

    def test_one_text_only_request_covering_the_region_and_its_context(self) -> None:
        referee = _Referee(OTHER_RUN_LITERAL)
        _repair(referee, LITERAL, STYLE, SOURCE)
        assert referee.calls == [(66, 79)]  # dải 68-77 kèm hai dòng ngữ cảnh mỗi bên
        assert referee.attach_video == [False]

    def test_localize_case_reverts_the_region_only(self) -> None:
        """Dòng ngoài dải là bản tinh chỉnh ĐÚNG — không được lùi theo."""
        fixed = _repair(_Referee(LITERAL), STYLE_587, LOCALIZE_587, SOURCE)
        for index in range(68, 78):
            assert fixed[index] == STYLE_587[index]
        assert fixed[77] == "Cậu không sao chứ?"  # bản địa hoá: "Trưởng đồn..."
        for index in (65, 66, 67, 78, 79, 81):
            assert fixed[index] == LOCALIZE_587[index] != STYLE_587[index]

    def test_refinement_that_fixed_a_shifted_literal_is_kept(self) -> None:
        referee = _Referee(REBUILD_OTHER_RUN)
        fixed = _repair(referee, REBUILD_LITERAL, REBUILD_STYLE, REBUILD_SOURCE)
        assert fixed == REBUILD_STYLE
        assert len(referee.calls) == 1

    def test_referee_siding_with_neither_keeps_the_stage(self) -> None:
        unrelated = dict.fromkeys(SOURCE, "Hoàn toàn không dính dáng gì tới câu nào cả.")
        assert _repair(_Referee(unrelated), LITERAL, STYLE, SOURCE) == STYLE

    def test_referee_failure_keeps_the_stage(self) -> None:
        referee = _Referee(OTHER_RUN_LITERAL, error=SubtitleTranslationError("503"))
        assert _repair(referee, LITERAL, STYLE, SOURCE) == STYLE

    def test_without_a_literal_stage_nothing_is_arbitrated(self) -> None:
        referee = _Referee(OTHER_RUN_LITERAL)
        fixed = _repair(referee, LITERAL, STYLE, SOURCE, _request(with_literal=False))
        assert fixed == STYLE and referee.calls == []

    def test_clean_stage_costs_no_request(self) -> None:
        referee = _Referee(OTHER_RUN_LITERAL)
        assert _repair(referee, OTHER_RUN_LITERAL, STYLE_587, SOURCE) == STYLE_587
        assert referee.calls == []

    def test_request_budget_is_shared_with_shift_arbitration(self) -> None:
        """Nhiều dải viết lại trong một giai đoạn vẫn không vượt trần request."""
        previous: dict[int, str] = {}
        current: dict[int, str] = {}
        sources: dict[int, str] = {}
        for block in range(MAX_SHIFT_ARBITRATIONS_PER_STAGE + 2):
            for index, source in SOURCE.items():
                key = block * 100 + index
                sources[key] = source
                previous[key], current[key] = LITERAL[index], STYLE[index]
        referee = _Referee(previous)
        fixed = _repair(referee, previous, current, sources)
        assert len(referee.calls) == MAX_SHIFT_ARBITRATIONS_PER_STAGE
        assert fixed[68] == LITERAL[68]
        last_block = (MAX_SHIFT_ARBITRATIONS_PER_STAGE + 1) * 100
        assert fixed[last_block + 68] == STYLE[68]


class TestProcessLog:
    def test_rewrite_and_verdict_are_recorded(self) -> None:
        _repair(_Referee(OTHER_RUN_LITERAL), LITERAL, STYLE, SOURCE)
        assert _events("stage_rewrite") == [{
            "event": "stage_rewrite",
            "label": "Phong cách",
            "from_index": 68,
            "to_index": 77,
            "offset": 0,
            "verdict": VERDICT_PREVIOUS,
        }]
        assert _events("stage_shift") == []

    def test_kept_rewrite_is_recorded_too(self) -> None:
        _repair(
            _Referee(REBUILD_OTHER_RUN), REBUILD_LITERAL, REBUILD_STYLE, REBUILD_SOURCE
        )
        (event,) = _events("stage_rewrite")
        assert (event["from_index"], event["to_index"]) == (3401, 3403)
        assert event["verdict"] == VERDICT_CURRENT


def _stage_rows(texts: dict[int, str]) -> list[dict[str, Any]]:
    return [{"index": index, "text": text} for index, text in texts.items()]


class TestDiagnosticsReport:
    """Dải viết lại còn tới bản cuối phải hiện trong file chuẩn đoán dịch."""

    STAGES = ({"stage_id": "literal", "lines": _stage_rows(LITERAL)},)

    def test_real_final_is_flagged(self) -> None:
        # Bảy dòng: 73 và 75 đã được giai đoạn địa hoá trả về đúng lời bản thô.
        assert final_vs_literal_rewrites(list(self.STAGES), _stage_rows(FINAL)) == [
            {"from_index": 68, "to_index": 77, "offset": 0, "lines": 7}
        ]
        assert final_vs_literal_shifts(list(self.STAGES), _stage_rows(FINAL)) == []

    def test_faithful_final_is_not_flagged(self) -> None:
        final = _stage_rows(STYLE_587)
        stages = [{"stage_id": "literal", "lines": _stage_rows(OTHER_RUN_LITERAL)}]
        assert final_vs_literal_rewrites(stages, final) == []

    def test_without_literal_stage_nothing_is_reported(self) -> None:
        assert final_vs_literal_rewrites([], _stage_rows(FINAL)) == []

    def test_exported_file_carries_the_flag_and_its_count(self) -> None:
        def events(texts: dict[int, str]) -> list[SimpleNamespace]:
            return [
                SimpleNamespace(
                    index=index, text=text, start_sec=index, end_sec=index + 0.9,
                    confidence=1.0,
                )
                for index, text in texts.items()
            ]

        literal = SimpleNamespace(
            stage_id="literal", input_hash="", completed_at="",
            lines_json=json.dumps(_stage_rows(LITERAL), ensure_ascii=False),
        )
        bundle = build_diagnostics_bundle(
            app_version="3.23.589", exported_at="2026-10-04T00:00:00Z",
            video_path="/x/phim.mp4", source_events=events(SOURCE),
            translated_events=events(FINAL), session=SimpleNamespace(stages=[literal]),
        )
        flags = bundle["quality_flags"]
        assert flags["final_vs_literal_rewrites"] == [
            {"from_index": 68, "to_index": 77, "offset": 0, "lines": 7}
        ]
        assert flags["counts"]["final_vs_literal_rewrites"] == 1
        assert flags["counts"]["final_vs_literal_shifts"] == 0
        assert "final_vs_literal_rewrites" in flags["note"]
