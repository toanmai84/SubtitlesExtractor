# ruff: noqa: RUF001 — dấu câu fullwidth CJK thuộc câu gốc chép từ phim
"""[v3.23.592] Hồi quy: dòng LẺ bị giai đoạn tinh chỉnh viết lại sai, lưới dải không thấy.

Lượt dịch thật v3.23.591 (phim "灭门惨案II", nói tiếng Quảng Đông, phụ đề theo văn viết;
ba giai đoạn đều đính video):

    dòng 68  不要生氣   bản thô "Đừng giận chứ."  →  bản địa hoá "Mau mở cửa!"

"Mau mở cửa!" là lời đám đông NGHE được trong video. Quy tắc "video chỉ là ngữ cảnh" thêm
vào prompt ở v3.23.591 không ngăn được model; lưới dải viết lại (v3.23.589) cần từ ba
dòng liền nhau nên không thấy. Dòng này sai ở bản cuối của 9/17 lượt dịch phim này.

Các dòng của phim này chép nguyên văn từ lượt chạy đó (``0d20d817``); cột "trọng tài" là
bản dịch THÔ của lượt chạy trước (``5049a479``, v3.23.590) cho đúng dòng ấy. Bốn nhóm dòng
cuối lấy từ các lượt dịch thật cũ hơn (ghi ở từng nhóm): những ca mà lùi về bản trước mới
là làm hỏng.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from subtitles_extractor.application.services.lone_rewrite_guard import (
    MAX_LONE_REWRITES_PER_STAGE,
    MOVED_MIN_SHARED_WORDS,
    MOVED_MIN_SIMILARITY,
    REFEREE_MIN_AGREEMENT,
    REFEREE_MIN_MARGIN,
    find_lone_rewrites,
    reference_lines_needed,
    rewrites_to_revert,
)
from subtitles_extractor.application.services.stage_shift_guard import (
    shared_word_count,
    text_similarity,
)
from subtitles_extractor.application.services.translation_process_log import (
    get_process_log,
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

#: (dòng, câu gốc, bản trước, bản mới, trọng tài)
Row = tuple[int, str, str, str, str]

# Giai đoạn ĐỊA HOÁ của lượt thật (bản trước = bản phong cách).
QUEUE_SCENE: list[Row] = [
    (
        67, "叫了你不要看那麼多進口港產片的啊",
        "Đã bảo cậu đừng xem lắm phim Hồng Kông nhập khẩu cơ mà.",
        "Đã bảo cậu đừng xem lắm phim Hồng Kông nhập khẩu cơ mà.",
        "Đã bảo cậu đừng xem lắm phim Hồng Kông rồi.",
    ),
    (68, "不要生氣", "Đừng giận chứ.", "Mau mở cửa!", "Đừng giận."),
    (
        69, "排好隊，一個跟一個", "Xếp hàng đi, từng người một.",
        "Xếp hàng đi, từng người một.", "Xếp hàng đi, từng người một.",
    ),
    (70, "這麼久", "Sao lâu thế.", "Sao lâu thế.", "Lâu thế?"),
    (
        71, "還關門呢", "Còn đóng cửa nữa chứ.",
        "Cửa đóng then cài thế này, chắc là chia nhiều thịt heo lắm.",
        "Còn đóng cửa nữa chứ?",
    ),
    (
        72, "這次一定派很多豬肉了", "Lần này chắc được chia nhiều thịt heo lắm.",
        "Lần này chắc được chia nhiều thịt heo lắm.",
        "Lần này chắc chắn chia được nhiều thịt lợn lắm.",
    ),
]
# Cùng giai đoạn, lô đầu phim: dòng 4 mang lời nghe được; 18 và 19 là hai dòng LIỀN nhau
# cùng bị viết lại, mỗi dòng một kiểu.
OPENING: list[Row] = [
    (
        3, "都算官運亨通、龍馬精神", "Đúng là quan vận hanh thông, phơi phới tinh thần",
        "Đúng là quan vận hanh thông, phơi phới tinh thần,",
        "đúng là quan vận hanh thông, phơi phới phong độ!",
    ),
    (
        4, "了不起嗎", "Có gì mà ghê gớm", "thăng quan phát tài, xui xẻo đến rồi!",
        "Có gì ghê gớm chứ?",
    ),
    (
        5, "本年度十大優秀公安獎", "Giải thưởng 10 công an xuất sắc năm nay",
        "Bản năm nay được giải mười công an xuất sắc nhất...",
        "Giải thưởng công an ưu tú top ten trong năm,",
    ),
    (17, "來，跟我來", "Mau, đi theo tôi.", "Mau, đi theo tôi!", "Mau, theo anh!"),
    (18, "甚麼事呀", "Chuyện gì vậy chứ?", "Thế ạ?", "Có chuyện gì thế?"),
    (19, "幹甚麼", "Làm gì thế?", "Có thật không?", "Làm gì đấy?"),
    (20, "你白痴呀", "Mày đần à?", "Mày điên à?", "Mày đần à?"),
]
# Cùng giai đoạn: bản trước SAI ("Đừng đánh."), giai đoạn địa hoá sửa đúng — phải giữ.
FIXED_BY_LOCALIZE: list[Row] = [
    (1113, "有事慢慢說", "Có gì từ từ nói.", "Có gì từ từ nói.", "Có gì từ từ nói."),
    (1114, "誰來惹麻煩", "Đừng đánh.", "Ai gây rắc rối thế?", "Ai gây chuyện trước hả!"),
    (1115, "爬過來", "Bò lại đây!", "Bò lại đây!", "Lê lại đây cho tao!"),
]
# Giai đoạn PHONG CÁCH của lượt thật (bản trước = bản thô).
SUMMONS: list[Row] = [
    (
        185, "不許離開本村", "không được rời khỏi bổn thôn.",
        "không được rời khỏi bổn thôn,", "không được rời khỏi bản thôn,",
    ),
    (
        186, "隨傳隨到", "Gọi lúc nào có mặt lúc đó.", "phải tùy tùng theo sát.",
        "gọi đâu có đó.",
    ),
    (
        187, "阿積有結果沒有", "A Tích, có kết quả chưa?", "A Tích, có kết quả chưa?",
        "A Tích, có kết quả chưa?",
    ),
]
# Cùng giai đoạn: bản thô SAI ("Xin anh tha mạng."), giai đoạn phong cách sửa đúng.
FIXED_BY_STYLE: list[Row] = [
    (
        1119, "為什麼要害我老婆，為什麼？", "Tại sao phải hại vợ tao, tại sao?",
        "Tại sao phải hại vợ tao, tại sao?", "Tại sao mày hại vợ tao, tại sao?!",
    ),
    (1120, "去你的混蛋", "Xin anh tha mạng.", "Đồ khốn!", "Đồ khốn nạn!"),
    (
        1121, "為什麼要殺我兒子", "Tại sao phải giết con tao?",
        "Tại sao phải giết con tao?", "Tại sao mày lại giết con tao?!",
    ),
]

# Phim "灭门惨案II", lượt fe0a10d9 (v3.23.578), giai đoạn PHONG CÁCH; trọng tài = bản thô
# lượt c4e1c1c3. Giai đoạn tinh chỉnh CHIA LẠI câu cho hai dòng theo đúng thứ tự câu gốc;
# lùi riêng dòng 212 thì "bí truyền" bị nói hai lần còn "chè xoài sago" biến mất.
RESPLIT_DESSERT: list[Row] = [
    (
        211, "當然，這是我家傳秘製的", "Tất nhiên rồi, đây là chè xoài sago",
        "Tất nhiên, đây là món bí truyền của anh,",
        "Đương nhiên rồi, đây là chè xoài sago",
    ),
    (
        212, "西米露芒果", "bí truyền nhà tôi đấy.", "chè xoài sago đấy.",
        "bí truyền nhà tôi đấy.",
    ),
    (
        213, "是嗎甚麼時候可以再吃呀", "Thật á? Khi nào mới được ăn nữa đây?",
        "Thật ạ? Khi nào mới được ăn nữa đây?", "Thật ạ? Bao giờ mới được ăn nữa đây?",
    ),
]
# Phim "he thong sung the", lượt b2e5dad9 (v3.23.579), giai đoạn PHONG CÁCH; trọng tài =
# bản thô lượt fed6ba08. Cùng kiểu: hai vế đổi chỗ cho nhau giữa hai dòng.
RESPLIT_CAR: list[Row] = [
    (
        3570, "这车是今天早上", "Chiếc xe này mới tậu", "Chiếc xe này là sáng nay",
        "Chiếc xe này mới tậu",
    ),
    (3571, "才提的", "vào sáng nay đấy.", "mới tậu đấy.", "vào sáng nay đấy."),
    (
        3572, "以逸子的实力", "Với gia thế của Dật Tử,", "Với gia thế của Dật Tử,",
        "Với thực lực của Dật Tử,",
    ),
]
# Phim "than y xuong nui", lượt 9bc25515 (v3.23.558), giai đoạn PHONG CÁCH; trọng tài =
# bản thô lượt 4270b658 — sót đúng mấy chữ gốc ấy. Bản trước của dòng 2745 CHƯA dịch xong.
LEFTOVER: list[Row] = [
    (2744, "废话", "Đương nhiên,", "Nói nhảm,", "Nói nhảm."),
    (
        2745, "你也不看看是谁的婚礼", "mày也不看看是谁的婚礼",
        "mày không nhìn xem đây là đám cưới của ai à?", "Mày也不看看是谁的婚礼.",
    ),
    (
        2746, "我徐俊豪的婚礼", "Đám cưới của Từ Tuấn Hào tao đây,",
        "Đám cưới của Từ Tuấn Hào tao đây,", "Đây là đám cưới của Từ Tuấn Hào tao,",
    ),
]
# Phim "he thong sung the", lượt 217d710a (v3.23.582), giai đoạn ĐỊA HOÁ; trọng tài = bản
# thô lượt 259d577a. Trọng tài chỉ chung với bản trước mỗi đại từ "Chị".
PRONOUN_ONLY: list[Row] = [
    (
        857, "这个就叫外贸", "cái này gọi là ngoại thương.",
        "cái này gọi là ngoại thương.", "Cái này gọi là ngoại thương.",
    ),
    (858, "你放心", "Chị xem,", "Em đảm bảo,", "Chị yên tâm,"),
    (
        859, "我挣的钱", "em kiếm được tiền", "số tiền em kiếm được",
        "số tiền em kiếm được",
    ),
]


def _columns(rows: list[Row]) -> tuple[dict[int, str], dict[int, str], dict[int, str]]:
    return (
        {row[0]: row[2] for row in rows},
        {row[0]: row[3] for row in rows},
        {row[0]: row[4] for row in rows},
    )


class TestFindLoneRewrites:
    def test_real_case_lines_heard_from_the_video(self) -> None:
        previous, current, _ = _columns(QUEUE_SCENE)
        assert find_lone_rewrites(previous, current) == [68, 71]

    def test_real_case_single_calque_line(self) -> None:
        previous, current, _ = _columns(SUMMONS)
        assert find_lone_rewrites(previous, current) == [186]

    def test_real_case_neighbour_that_also_changed_does_not_hide_the_line(self) -> None:
        """Dòng 5 kề dòng 4 cũng bị sửa nhiều — không vì thế mà bỏ qua dòng 4."""
        previous, current, _ = _columns(OPENING)
        assert previous[5] != current[5]
        assert 4 in find_lone_rewrites(previous, current)

    def test_real_case_two_adjacent_rewrites_are_both_candidates(self) -> None:
        """18 và 19 liền nhau cùng bị viết lại, nhưng câu cũ của dòng này không nằm ở
        dòng kia — không phải đảo vế câu."""
        previous, current, _ = _columns(OPENING)
        assert find_lone_rewrites(previous, current) == [4, 18, 19]

    def test_unchanged_stage_has_no_candidates(self) -> None:
        previous, _, _ = _columns(QUEUE_SCENE)
        assert find_lone_rewrites(previous, dict(previous)) == []

    def test_rewording_that_keeps_the_words_is_not_a_candidate(self) -> None:
        previous = {1: "Anh đi đâu đấy?", 2: "Tôi về nhà đây.", 3: "Mai gặp lại nhé."}
        current = {1: "Anh đi đâu đấy?", 2: "Tôi về nhà thôi.", 3: "Mai gặp lại nhé."}
        assert find_lone_rewrites(previous, current) == []

    def test_clauses_swapped_between_two_lines_are_left_alone(self) -> None:
        """Đảo vế câu giữa hai dòng liền nhau cho thuận tiếng Việt: câu cũ của dòng này
        nằm ở dòng kề của bản mới — lùi một dòng sẽ làm một vế bị nói hai lần."""
        previous = {
            1: "Nếu ngày mai trời mưa to,", 2: "chúng ta sẽ ở nhà.", 3: "Được thôi."
        }
        current = {
            1: "Chúng ta sẽ ở nhà", 2: "nếu ngày mai trời mưa to.", 3: "Được thôi."
        }
        assert find_lone_rewrites(previous, current) == []

    def test_sentence_pushed_down_one_line_is_left_alone(self) -> None:
        """Câu của dòng 1 chạy xuống dòng 2 trong khi dòng 1 cũng đổi: lùi riêng dòng nào
        cũng làm mất hoặc lặp một vế — để lưới dải lệch lo."""
        previous = {1: "Đừng giận chứ.", 2: "Xếp hàng đi, từng người một.", 3: "Lâu thế."}
        current = {1: "Này mọi người,", 2: "đừng giận chứ.", 3: "Lâu thế."}
        assert find_lone_rewrites(previous, current) == []

    def test_real_case_sentence_resplit_between_two_lines_is_left_alone(self) -> None:
        """Dòng 212 lấy vế "chè xoài sago" từ dòng 211, dòng 211 nhận lại vế "bí truyền".
        Trọng tài trùng khít bản trước của dòng 212 — đem phân xử là sẽ bị lùi."""
        previous, current, referee = _columns(RESPLIT_DESSERT)
        assert referee[212] == previous[212]
        assert find_lone_rewrites(previous, current) == []

    def test_real_case_swapped_halves_are_left_alone(self) -> None:
        previous, current, referee = _columns(RESPLIT_CAR)
        assert referee[3571] == previous[3571]
        assert find_lone_rewrites(previous, current) == []

    def test_copying_an_unchanged_neighbour_is_still_a_candidate(self) -> None:
        """Ca thật dòng 71: mang cả câu của dòng 72 trong khi dòng 72 KHÔNG đổi — đó là
        viết hỏng (một vế bị nói hai lần), không phải chia lại câu."""
        previous, current, _ = _columns(QUEUE_SCENE)
        assert previous[72] == current[72]
        assert "chia nhiều thịt heo lắm" in current[71]
        assert 71 in find_lone_rewrites(previous, current)

    def test_one_shared_word_in_short_lines_is_not_a_moved_clause(self) -> None:
        """Ca thật dòng 19: "Làm gì thế?" và bản mới "Thế ạ?" của dòng 18 chỉ chung một
        từ — vẫn đem phân xử."""
        previous, current, _ = _columns(OPENING)
        assert 19 in find_lone_rewrites(previous, current)

    def test_real_case_leftover_source_text_finished_by_the_stage(self) -> None:
        """Bản trước của dòng 2745 còn nguyên chữ gốc, bản mới đã dịch nốt: không đem
        phân xử (trọng tài của ca này cũng sót đúng mấy chữ ấy)."""
        previous, current, referee = _columns(LEFTOVER)
        assert find_lone_rewrites(previous, current) == [2744]
        assert rewrites_to_revert([2744], previous, current, referee) == []

    def test_leftover_on_both_sides_is_still_judged(self) -> None:
        previous = {1: "Một.", 2: "mày也不看看 đi", 3: "Ba."}
        current = {1: "Một.", 2: "Đám cưới của 谁 thế?", 3: "Ba."}
        assert find_lone_rewrites(previous, current) == [2]

    def test_first_and_last_lines_have_only_one_neighbour(self) -> None:
        previous = {1: "Đừng giận chứ.", 2: "Xếp hàng đi.", 3: "Còn đóng cửa nữa chứ."}
        current = {1: "Mau mở cửa!", 2: "Xếp hàng đi.", 3: "Lần này chia nhiều thịt."}
        assert find_lone_rewrites(previous, current) == [1, 3]

    def test_empty_line_on_either_side_is_ignored(self) -> None:
        previous = {1: "Một.", 2: "", 3: "Ba.", 4: "Đừng giận chứ.", 5: "Năm."}
        current = {1: "Một.", 2: "Mau mở cửa!", 3: "Ba.", 4: "", 5: "Năm."}
        assert find_lone_rewrites(previous, current) == []

    def test_neighbours_follow_line_order_not_numbering(self) -> None:
        """Tập dòng thưa (dịch lại vài dòng được chọn): dòng kề là dòng đứng cạnh."""
        previous = {10: "Xếp hàng đi.", 40: "Đừng giận chứ.", 90: "Sao lâu thế."}
        current = {10: "Xếp hàng đi.", 40: "Mau mở cửa!", 90: "Sao lâu thế."}
        assert find_lone_rewrites(previous, current) == [40]

    def test_thresholds(self) -> None:
        assert MOVED_MIN_SIMILARITY == 0.4
        assert MOVED_MIN_SHARED_WORDS == 2
        assert REFEREE_MIN_MARGIN == 0.3
        assert REFEREE_MIN_AGREEMENT == 0.5


class TestSharedWordCount:
    def test_short_lines_sharing_one_word(self) -> None:
        assert text_similarity("Làm gì thế?", "Thế ạ?") == pytest.approx(0.4)
        assert shared_word_count("Làm gì thế?", "Thế ạ?") == 1

    def test_real_case_moved_half_shares_two_words(self) -> None:
        previous, current, _ = _columns(RESPLIT_CAR)
        assert shared_word_count(previous[3571], current[3570]) == 2
        assert text_similarity(previous[3571], current[3570]) == pytest.approx(0.4)

    def test_blank_text_shares_nothing(self) -> None:
        assert shared_word_count("", "Thế ạ?") == 0
        assert shared_word_count("Thế ạ?", "…") == 0


class TestRewritesToRevert:
    def test_real_case_heard_lines_go_back(self) -> None:
        previous, current, referee = _columns(QUEUE_SCENE)
        assert rewrites_to_revert([68, 71], previous, current, referee) == [68, 71]

    def test_real_case_calque_goes_back(self) -> None:
        previous, current, referee = _columns(SUMMONS)
        assert rewrites_to_revert([186], previous, current, referee) == [186]

    def test_real_case_opening_lines(self) -> None:
        """Dòng 4 và 19 về bản trước; dòng 18 trọng tài lưỡng lự nên giữ bản mới."""
        previous, current, referee = _columns(OPENING)
        assert rewrites_to_revert([4, 18, 19], previous, current, referee) == [4, 19]

    def test_real_case_localize_fix_is_kept(self) -> None:
        """Bản trước "Đừng đánh." sai; trọng tài không đứng về phía nó."""
        previous, current, referee = _columns(FIXED_BY_LOCALIZE)
        assert find_lone_rewrites(previous, current) == [1114]
        assert rewrites_to_revert([1114], previous, current, referee) == []

    def test_real_case_style_fix_is_kept(self) -> None:
        previous, current, referee = _columns(FIXED_BY_STYLE)
        assert find_lone_rewrites(previous, current) == [1120]
        assert rewrites_to_revert([1120], previous, current, referee) == []

    def test_real_case_referee_sharing_only_a_pronoun_does_not_revert(self) -> None:
        """Bản mới "Em đảm bảo," gần nghĩa "你放心" hơn bản trước "Chị xem,". Trọng tài
        "Chị yên tâm," giống bản trước hơn bản mới 0,4 — nhưng chỉ nhờ một đại từ."""
        previous, current, referee = _columns(PRONOUN_ONLY)
        assert find_lone_rewrites(previous, current) == [858]
        assert rewrites_to_revert([858], previous, current, referee) == []

    def test_referee_must_agree_with_the_previous_text(self) -> None:
        previous, current = {1: "Chị xem,"}, {1: "Em đảm bảo,"}
        assert rewrites_to_revert([1], previous, current, {1: "Chị xem nhé,"}) == [1]
        assert rewrites_to_revert([1], previous, current, {1: "Chị cứ yên tâm,"}) == []

    def test_undecided_referee_keeps_the_new_text(self) -> None:
        previous, current = {1: "Đừng giận chứ."}, {1: "Mau mở cửa!"}
        assert rewrites_to_revert([1], previous, current, {1: "Bình tĩnh nào."}) == []

    def test_missing_or_blank_referee_keeps_the_new_text(self) -> None:
        previous, current = {1: "Đừng giận chứ."}, {1: "Mau mở cửa!"}
        assert rewrites_to_revert([1], previous, current, {}) == []
        assert rewrites_to_revert([1], previous, current, {1: "  "}) == []

    def test_only_candidates_are_judged(self) -> None:
        previous, current, referee = _columns(QUEUE_SCENE)
        assert rewrites_to_revert([71], previous, current, referee) == [71]


class TestReferenceLinesNeeded:
    def test_candidate_brings_one_neighbour_each_side(self) -> None:
        ordered = [row[0] for row in QUEUE_SCENE]
        assert reference_lines_needed([68, 71], ordered) == [67, 68, 69, 70, 71, 72]

    def test_edges_and_unknown_lines(self) -> None:
        assert reference_lines_needed([1, 9], [1, 2, 3]) == [1, 2]
        assert reference_lines_needed([], [1, 2, 3]) == []


def _lines(rows: list[Row], column: int) -> list[TranslationLine]:
    return [
        TranslationLine(
            index=row[0], start_ms=row[0] * 1000, end_ms=row[0] * 1000 + 900,
            text=row[column], original_text=row[1],
        )
        for row in rows
    ]


class _Referee:
    """Bộ dịch giả: trả cột "trọng tài" cho đúng các dòng được hỏi."""

    def __init__(self, rows: list[Row], error: Exception | None = None) -> None:
        self.answer = {row[0]: row[4] for row in rows}
        self.error = error
        self.calls: list[list[int]] = []
        self.attach_video: list[bool] = []

    def translate_stage(
        self, *, stage: Any, source_lines: list[TranslationLine], **kwargs: Any
    ) -> list[TranslationLine]:
        assert stage.kind is TranslationStageKind.LITERAL
        self.calls.append([line.index for line in source_lines])
        self.attach_video.append(kwargs["attach_video"])
        if self.error is not None:
            raise self.error
        return [replace(line, text=self.answer[line.index]) for line in source_lines]


def _request(with_literal: bool = True) -> TranslateSubtitlesRequest:
    kinds = [TranslationStageKind.LOCALIZE]
    if with_literal:
        kinds.insert(0, TranslationStageKind.LITERAL)
    return TranslateSubtitlesRequest(
        events=[],
        stages=[TranslationStageConfig(kind=kind, model_name="m") for kind in kinds],
        context=TranslationContext(target_lang="Vietnamese"),
        video_refs=("files/SEG0",),
        attach_video_stages=frozenset({TranslationStageKind.LOCALIZE}),
    )


def _run(
    rows: list[Row],
    referee: _Referee,
    *,
    with_literal: bool = True,
    cancel_cb: Any = None,
) -> dict[int, str]:
    use_case = TranslateSubtitlesUseCase(translator=referee)  # type: ignore[arg-type]
    result = use_case._revert_lone_rewrites(
        _lines(rows, 3), _lines(rows, 2), _lines(rows, 1), _request(with_literal),
        cancel_cb, "Giai đoạn 4: Bản địa hoá",
    )
    return {line.index: line.text for line in result}


def _events(name: str) -> list[dict[str, Any]]:
    return [e for e in get_process_log().snapshot()["events"] if e["event"] == name]


@pytest.fixture(autouse=True)
def _fresh_log() -> None:
    get_process_log().clear()


ALL_LOCALIZE = QUEUE_SCENE + FIXED_BY_LOCALIZE


class TestStageGuard:
    def test_real_case_heard_lines_are_reverted(self) -> None:
        fixed = _run(ALL_LOCALIZE, _Referee(ALL_LOCALIZE))
        assert fixed[68] == "Đừng giận chứ."
        assert fixed[71] == "Còn đóng cửa nữa chứ."

    def test_real_case_good_rewrite_and_other_lines_are_untouched(self) -> None:
        _, current, _ = _columns(ALL_LOCALIZE)
        fixed = _run(ALL_LOCALIZE, _Referee(ALL_LOCALIZE))
        assert fixed[1114] == "Ai gây rắc rối thế?"
        assert {i for i in fixed if fixed[i] != current[i]} == {68, 71}

    def test_one_text_only_request_for_candidates_and_their_neighbours(self) -> None:
        referee = _Referee(ALL_LOCALIZE)
        _run(ALL_LOCALIZE, referee)
        assert referee.calls == [[67, 68, 69, 70, 71, 72, 1113, 1114, 1115]]
        # Video là nơi lời "nghe được" lọt vào — trọng tài không được nhìn video.
        assert referee.attach_video == [False]

    def test_stage_without_lone_rewrites_costs_nothing(self) -> None:
        rows = [(i, s, before, before, ref) for i, s, before, _, ref in QUEUE_SCENE]
        referee = _Referee(rows)
        assert _run(rows, referee) == {row[0]: row[2] for row in rows}
        assert referee.calls == []
        assert _events("lone_rewrites_judged") == []

    def test_event_lists_candidates_and_reverted(self) -> None:
        _run(ALL_LOCALIZE, _Referee(ALL_LOCALIZE))
        assert _events("lone_rewrites_judged") == [
            {
                "event": "lone_rewrites_judged",
                "label": "Giai đoạn 4: Bản địa hoá",
                "candidates": [68, 71, 1114],
                "reverted": [68, 71],
            }
        ]

    def test_log_names_the_reverted_lines(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("WARNING"):
            _run(ALL_LOCALIZE, _Referee(ALL_LOCALIZE))
        assert "2/3 dòng lẻ bị viết lại khác hẳn bản trước" in caplog.text
        assert "68, 71" in caplog.text

    def test_no_literal_stage_means_no_referee(self) -> None:
        referee = _Referee(ALL_LOCALIZE)
        _, current, _ = _columns(ALL_LOCALIZE)
        assert _run(ALL_LOCALIZE, referee, with_literal=False) == current
        assert referee.calls == []

    def test_failed_referee_keeps_the_stage_output(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        _, current, _ = _columns(ALL_LOCALIZE)
        referee = _Referee(ALL_LOCALIZE, SubtitleTranslationError("503 UNAVAILABLE"))
        with caplog.at_level("WARNING"):
            assert _run(ALL_LOCALIZE, referee) == current
        assert "dịch trọng tài cho 3 dòng lẻ bị viết lại thất bại" in caplog.text

    def test_spurious_cancel_keeps_the_stage_output(self) -> None:
        _, current, _ = _columns(ALL_LOCALIZE)
        referee = _Referee(ALL_LOCALIZE, TranslationCancelledError("đã huỷ"))
        assert _run(ALL_LOCALIZE, referee, cancel_cb=lambda: False) == current

    def test_real_cancel_is_propagated(self) -> None:
        referee = _Referee(ALL_LOCALIZE, TranslationCancelledError("đã huỷ"))
        with pytest.raises(TranslationCancelledError):
            _run(ALL_LOCALIZE, referee, cancel_cb=lambda: True)

    def test_too_many_candidates_are_capped(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Mỗi dòng thứ ba bị viết lại: quá trần thì chỉ phân xử các dòng đầu."""
        rows: list[Row] = []
        heard = ("Đừng giận chứ.", "Mau mở cửa!")
        for n in range(1, 3 * (MAX_LONE_REWRITES_PER_STAGE + 5) + 1):
            if n % 3 == 2:
                rows.append((n, "不要生氣", heard[0], heard[1], "Đừng giận."))
            else:
                rows.append((n, "排好隊", *(["Xếp hàng đi."] * 3)))
        referee = _Referee(rows)
        with caplog.at_level("WARNING"):
            fixed = _run(rows, referee)
        reverted = [n for n, text in fixed.items() if n % 3 == 2 and text == heard[0]]
        assert len(reverted) == MAX_LONE_REWRITES_PER_STAGE
        assert len(referee.calls) == 1
        assert f"chỉ phân xử {MAX_LONE_REWRITES_PER_STAGE} dòng đầu" in caplog.text


class _Pipeline:
    """Bộ dịch giả cho cả lượt dịch: dịch thô, địa hoá, và lượt trọng tài."""

    def __init__(self, rows: list[Row]) -> None:
        self.rows = rows
        self.calls: list[tuple[str, int, bool]] = []

    def translate_stage(
        self,
        *,
        stage: Any,
        source_lines: list[TranslationLine],
        input_lines: list[TranslationLine],
        attach_video: bool = False,
        **_kwargs: Any,
    ) -> list[TranslationLine]:
        self.calls.append((stage.kind.value, len(source_lines), attach_video))
        # Lượt dịch thô ĐẦU là của cả phim; lượt dịch thô sau đó là trọng tài.
        literal_calls = [call for call in self.calls if call[0] == "literal"]
        if stage.kind is TranslationStageKind.LITERAL:
            column = 2 if len(literal_calls) == 1 else 4
        else:
            column = 3
        # Sự kiện đánh số 1..N theo thứ tự các dòng của kịch bản.
        return [
            replace(line, text=self.rows[line.index - 1][column]) for line in input_lines
        ]


class TestWholeRun:
    def test_execute_returns_the_literal_text_for_the_heard_lines(self) -> None:
        from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
        from subtitles_extractor.domain.value_objects.time_interval import TimeInterval

        events = [
            SubtitleEvent(index=pos, text=row[1], interval=TimeInterval(pos, pos + 0.9))
            for pos, row in enumerate(QUEUE_SCENE, start=1)
        ]
        pipeline = _Pipeline(QUEUE_SCENE)
        use_case = TranslateSubtitlesUseCase(translator=pipeline)  # type: ignore[arg-type]
        response = use_case.execute(replace(_request(), events=events))
        final = {
            row[0]: event.text
            for row, event in zip(QUEUE_SCENE, response.events, strict=True)
        }
        assert final[68] == "Đừng giận chứ."
        assert final[71] == "Còn đóng cửa nữa chứ."
        assert final[69] == "Xếp hàng đi, từng người một."
        # Dịch thô cả cảnh, địa hoá cả cảnh (kèm video), rồi MỘT lượt trọng tài không
        # video cho hai ứng viên và các dòng kề.
        assert pipeline.calls == [
            ("literal", 6, False),
            ("localize", 6, True),
            ("literal", 6, False),
        ]
        assert _events("lone_rewrites_judged")[0]["reverted"] == [2, 5]
