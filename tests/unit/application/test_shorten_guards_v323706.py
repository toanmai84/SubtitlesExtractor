"""[v3.23.706] Lượt rút gọn khớp giọng đọc: không cắt đôi từ xưng hô, không viết hoa dòng nối tiếp.

Ca thật từ A/B 361 câu (gemini-3.5-flash-lite, bản 3.23.702): 3 câu rút gọn hỏng từ xưng hô và 15/361 dòng nối
tiếp bị viết hoa giữa câu.
"""

from __future__ import annotations

from subtitles_extractor.application.services.sentence_case_guard import restore_continuation_case
from subtitles_extractor.application.use_cases.shorten_for_speech import validate_shortened
from subtitles_extractor.infrastructure.tts.tts_fit import REASON_PREDICTED, LineFit


def _fit(target: int = 4) -> LineFit:
    return LineFit(position=0, syllables=9, target_syllables=target, available_s=1.0, reason=REASON_PREDICTED)


class TestTitleCompounds:
    def test_real_broken_compounds_are_rejected(self) -> None:
        assert validate_shortened(_fit(6), "nể mặt ngươi là đệ tử của sư tỷ,", "nể mặt ngươi là đệ sư tỷ,") == ""
        assert validate_shortened(_fit(3), "xin hãy tìm cho đồ nhi", "tìm cho đồ") == ""
        assert validate_shortened(_fit(4), "Diệp Nhiên là đồ nhi của ta,", "Diệp Nhiên là đồ ta.") == ""

    def test_whole_compound_kept_or_dropped_is_fine(self) -> None:
        assert validate_shortened(_fit(6), "nể mặt ngươi là đệ tử của sư tỷ,", "nể ngươi là đệ tử sư tỷ,") \
            == "nể ngươi là đệ tử sư tỷ,"
        assert validate_shortened(_fit(6), "Thiên Thanh sư huynh, huynh điên rồi sao?", "Sư huynh, huynh điên rồi sao?") == ""  # mất tên riêng (luật cũ)
        assert validate_shortened(_fit(5), "Còn tận hai trăm đạo lôi kiếp,", "Còn hai trăm lôi kiếp,") \
            == "Còn hai trăm lôi kiếp,"

    def test_pronoun_half_already_standing_alone_is_not_blamed(self) -> None:
        before = "sư huynh à, huynh điên thật rồi sao hả?"
        assert validate_shortened(_fit(6), before, "sư huynh, huynh điên rồi sao?") == "sư huynh, huynh điên rồi sao?"


class TestClauseMarkers:
    def test_dropping_the_conditional_is_rejected(self) -> None:
        assert validate_shortened(_fit(5), "Nếu như con không chống đỡ nổi nữa,", "Con không chống nổi nữa,") == ""
        assert validate_shortened(_fit(3), "nếu ngươi không chịu nổi", "không chịu nổi") == ""

    def test_keeping_it_is_fine(self) -> None:
        assert validate_shortened(_fit(6), "Nếu như con không chống đỡ nổi nữa,", "Nếu con không chống nổi,") \
            == "Nếu con không chống nổi,"


class TestLeadingCase:
    def test_continuation_keeps_lowercase(self) -> None:
        assert validate_shortened(_fit(5), "mà đã tăng lên năm mươi đạo lôi kiếp rồi.", "Đã tăng năm mươi lôi kiếp.") \
            == "đã tăng năm mươi lôi kiếp."
        assert validate_shortened(_fit(5), "thì cứ cầu xin Thiên Thanh sư bác", "Cầu xin Thiên Thanh sư bác") \
            == "cầu xin Thiên Thanh sư bác"

    def test_proper_name_start_is_kept(self) -> None:
        assert validate_shortened(_fit(6), "thì hôm nay Diệp Nhiên sẽ phải chết ở đây", "Diệp Nhiên sẽ phải chết ở đây") \
            == "Diệp Nhiên sẽ phải chết ở đây"

    def test_sentence_start_is_untouched(self) -> None:
        assert validate_shortened(_fit(5), "Vẫn còn lại hai trăm đạo lôi kiếp,", "Còn hai trăm lôi kiếp,") \
            == "Còn hai trăm lôi kiếp,"


class TestRestoreContinuationCase:
    def test_real_cases(self) -> None:
        literal = ["Đã vậy,", "ta là sư bác đây", "Diệp sư điệt,", "ngươi sợ là đã đến giới hạn rồi nhỉ."]
        final = ["Đã vậy,", "Ta là sư bác đây,", "Diệp sư điệt,", "Ngươi sợ là đã đến giới hạn rồi nhỉ."]
        assert restore_continuation_case(literal, final) == {
            1: "ta là sư bác đây,", 3: "ngươi sợ là đã đến giới hạn rồi nhỉ.",
        }

    def test_not_after_finished_sentence_and_not_names(self) -> None:
        literal = ["Xem ra.", "đối tượng trọng điểm", "Nam Cung,", "cực Quang kia"]
        final = ["Xem ra.", "Đối tượng trọng điểm", "Nam Cung,", "Cực Quang kia"]
        assert restore_continuation_case(literal, final) == {}

    def test_no_evidence_no_change(self) -> None:
        # Bản dịch thô đã viết hoa (model coi là câu mới) → giữ nguyên dù đứng sau dấu phẩy.
        assert restore_continuation_case(["Sư phụ,", "Đồ nhi xong rồi."], ["Sư phụ,", "Đồ nhi đã xong."]) == {}

    def test_length_mismatch_and_empty(self) -> None:
        assert restore_continuation_case(["a"], ["A", "B"]) == {}
        assert restore_continuation_case([], []) == {}
        assert restore_continuation_case(["", "ta đi"], ["", "Ta đi"]) == {}
