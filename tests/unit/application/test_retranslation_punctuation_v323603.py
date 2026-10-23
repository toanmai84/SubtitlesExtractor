"""[v3.23.603] Dòng dịch lại ở bước dọn cuối kết bằng dấu phẩy khi đã hết câu → đóng câu.

Bước dọn cuối gửi riêng vài dòng cho model, không kèm mạch câu. Trên 93 dòng được dịch lại
ở mọi file chuẩn đoán thật, 26 dòng mất dấu kết câu; lưới sửa 9 dòng có dấu phẩy MỚI ở
ranh giới câu (người nói khác / khoảng lặng ≥ 1,5 s) — soát tay đúng cả 9.
"""

from __future__ import annotations

from subtitles_extractor.application.services.retranslation_punctuation import (
    close_new_trailing_commas,
)

#: Dòng 64-66 thật (灭门惨案II, 11/10): dòng 65 được dịch lại ở bước dọn chữ Hán sót.
BEFORE = {
    64: "Đúng là bản chất anh hùng, khí thế ngút trời.",
    65: "Lại thêm chút dâm tiện, đúng là小白臉.",
    66: "Cậu trúng độc băng hình nhập khẩu à?",
}
AFTER = {**BEFORE, 65: "Thêm chút dâm tiện, mặt trắng,"}
SPANS = {64: (365.8, 368.467), 65: (368.733, 371.8), 66: (372.8, 374.267)}
SPEAKERS = {64: "愛國", 65: "胖公安", 66: "龍傳人"}


def test_dong_65_truoc_loi_nguoi_khac_duoc_dong_cau() -> None:
    assert close_new_trailing_commas(BEFORE, AFTER, SPANS, SPEAKERS) == {
        65: "Thêm chút dâm tiện, mặt trắng.",
    }


def test_dau_dong_cau_lay_theo_ban_truoc() -> None:
    # Dòng 4064-4067 thật (phim "he thong sung the", lượt b2e5dad9).
    before = {
        4065: "Không cần đâu!", 4066: "Không cần đâu!", 4067: "Đồ hèn nhát.",
    }
    after = {
        4065: "Không, không, không,", 4066: "Đừng, đừng mà,", 4067: "Đồ hèn nhát.",
    }
    spans = {
        4065: (6908.067, 6908.8),
        4066: (6908.867, 6909.6),
        4067: (6909.667, 6910.667),
    }
    speakers = {4065: "男同学", 4066: "男同学", 4067: "王大福"}
    assert close_new_trailing_commas(before, after, spans, speakers) == {
        4066: "Đừng, đừng mà!",
    }


def test_cung_nguoi_noi_sat_nhau_thi_giu_dau_phay() -> None:
    # Dòng 3826-3828 thật (lượt 217d710a): câu còn tiếp, dòng 3827 không có nhãn.
    before = {3826: "Đến đây,", 3827: "bước lên đây!", 3828: "Giỏi thì bước lên đây!"}
    after = {3826: "Đến đây,", 3827: "Đến đây,", 3828: "Thằng nào muốn chết cứ bước lên!"}
    spans = {3826: (6556.8, 6557.333), 3827: (6557.4, 6558.067), 3828: (6558.8, 6560.067)}
    speakers = {3826: "王大福", 3828: "王大福"}
    assert close_new_trailing_commas(before, after, spans, speakers) == {}


def test_dau_phay_co_san_tu_ban_truoc_khong_dong() -> None:
    # Dòng 1949 thật (lượt b2e5dad9): bản trước cũng kết bằng dấu phẩy.
    before = {1949: "không có những màn mưu mô đấu đá bẩn thỉu,", 1950: "mưu mô đấu đá,"}
    after = {1949: "Không có mấy thứ bẩn thỉu,", 1950: "mưu mô đấu đá,"}
    spans = {1949: (3492.4, 3494.733), 1950: (3494.8, 3495.867)}
    speakers = {1949: "付金奎", 1950: "林逸"}
    assert close_new_trailing_commas(before, after, spans, speakers) == {}


def test_dong_sau_chu_thuong_la_cau_noi_tiep() -> None:
    before = {1949: "Không có những thứ bẩn thỉu.", 1950: "mưu mô đấu đá,"}
    after = {1949: "Không có mấy thứ bẩn thỉu,", 1950: "mưu mô đấu đá,"}
    spans = {1949: (3492.4, 3494.733), 1950: (3494.8, 3495.867)}
    speakers = {1949: "付金奎", 1950: "林逸"}
    assert close_new_trailing_commas(before, after, spans, speakers) == {}


def test_khoang_lang_dai_va_dong_cuoi() -> None:
    before = {1: "Đã qua rồi.", 2: "Còn ngẩn ra làm gì!", 3: "Hết."}
    after = {1: "Đã qua rồi,", 2: "Còn ngẩn ra làm gì!", 3: "Hết rồi,"}
    spans = {1: (4692.4, 4693.2), 2: (4694.733, 4696.0), 3: (4697.0, 4698.0)}
    assert close_new_trailing_commas(before, after, spans, {}) == {
        1: "Đã qua rồi.",
        3: "Hết rồi.",
    }


def test_use_case_dong_cau_va_ghi_nhat_ky() -> None:
    from subtitles_extractor.application.services.translation_process_log import (
        get_process_log,
    )
    from subtitles_extractor.application.services.visual_cue_serializer import (
        serialize_visual_cues,
    )
    from subtitles_extractor.application.use_cases.translate_subtitles import (
        TranslateSubtitlesUseCase,
    )
    from subtitles_extractor.domain.ports.subtitle_translator_port import (
        TranslationContext,
        TranslationLine,
        VisualCue,
    )

    def _line(index: int, text: str) -> TranslationLine:
        start, end = SPANS[index]
        return TranslationLine(index, int(start * 1000), int(end * 1000), text)

    context = TranslationContext(
        target_lang="Vietnamese",
        visual_cues=serialize_visual_cues(
            [VisualCue(index, speaker, "") for index, speaker in SPEAKERS.items()]
        ),
    )
    current = [_line(index, text) for index, text in AFTER.items()]
    sources = [_line(index, "…") for index in SPANS]
    mark = get_process_log().count()
    result = TranslateSubtitlesUseCase._close_cleanup_sentences(
        BEFORE, current, sources, context
    )
    assert [line.text for line in result][1] == "Thêm chút dâm tiện, mặt trắng."
    events = get_process_log().events_since(mark)
    assert {"event": "retranslated_endings_closed", "lines": [65]} in [
        {key: value for key, value in event.items() if key in ("event", "lines")}
        for event in events
    ]
