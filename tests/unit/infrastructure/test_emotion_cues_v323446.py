"""[v3.23.446] Test emotion cues cho VieNeu v3 Turbo."""
from __future__ import annotations
from subtitles_extractor.infrastructure.tts.emotion_cues import (
    strip_brackets_keep_cues, inject_emotion_cues, EMOTION_CUES,
)


def test_strip_keeps_valid_cues():
    assert strip_brackets_keep_cues("Nói [cười] tiếp") == "Nói [cười] tiếp"
    assert strip_brackets_keep_cues("[hắng giọng] rồi") == "[hắng giọng] rồi"
    assert strip_brackets_keep_cues("[thở dài] mệt") == "[thở dài] mệt"


def test_strip_removes_junk_brackets():
    assert strip_brackets_keep_cues("Xin chào [OCR rác]").strip() == "Xin chào"
    assert "[note]" not in strip_brackets_keep_cues("Hi [note] there")


def test_strip_cue_case_insensitive():
    # tag hoa/khoảng trắng thừa vẫn nhận là cue hợp lệ
    assert strip_brackets_keep_cues("[ Cười ]") == "[ Cười ]"


def test_inject_laughter():
    assert inject_emotion_cues("Haha, đúng vậy") == "[cười], đúng vậy"
    assert inject_emotion_cues("Buồn cười ha ha ha") == "Buồn cười [cười]"
    assert inject_emotion_cues("Khà khà thật") == "[cười] thật"


def test_inject_sigh():
    assert inject_emotion_cues("Haizz, mệt quá") == "[thở dài], mệt quá"


def test_no_false_positive_on_h_words():
    # từ bắt đầu bằng 'h' KHÔNG được nhầm thành cười/thở dài
    for t in ["Chào bạn nhé", "Hàng của tôi", "hôm nay trời đẹp", "Hải Phòng",
              "Hạnh phúc", "hả bạn"]:
        assert inject_emotion_cues(t) == t, f"false positive: {t}"


def test_single_ha_not_matched():
    # "ha" đơn (có thể là "hả") KHÔNG khớp — cần >=2 âm
    assert inject_emotion_cues("Ừ ha") == "Ừ ha"


def test_cues_are_three():
    # Ba tag chính thức luôn có mặt; [v3.23.627] thêm tên tiếng Anh/không dấu SDK 3.8 cũng nhận.
    assert {"[cười]", "[thở dài]", "[hắng giọng]"} <= EMOTION_CUES
    assert EMOTION_CUES == frozenset({
        "[cười]", "[thở dài]", "[hắng giọng]", "[chuckle]", "[sigh]", "[clear throat]",
        "[cuoi]", "[tho dai]", "[hang giong]",
    })


# [v3.23.447] strip_emotion_cues (cho xuất phụ đề)
from subtitles_extractor.infrastructure.tts.emotion_cues import strip_emotion_cues


def test_strip_emotion_cues_removes_cues():
    assert strip_emotion_cues("Đúng vậy [cười] thật") == "Đúng vậy thật"
    assert strip_emotion_cues("[thở dài] mệt quá") == "mệt quá"


def test_strip_emotion_cues_keeps_other_brackets():
    assert "[note]" in strip_emotion_cues("Hi [note] there")


def test_strip_vs_keep_are_opposite():
    from subtitles_extractor.infrastructure.tts.emotion_cues import strip_brackets_keep_cues
    line = "A [cười] B [OCR] C"
    # export bỏ cue, giữ nội dung khác đã strip riêng; TTS giữ cue
    assert "[cười]" not in strip_emotion_cues(line)
    assert "[cười]" in strip_brackets_keep_cues(line)


# [v3.23.456] Hồi quy: dòng chỉ-toàn-cue (VieNeu ngân dài) phải nhận diện được để bỏ.
from subtitles_extractor.infrastructure.tts.emotion_cues import has_real_words_besides_cues


def test_cue_only_lines_have_no_real_words():
    for t in ["[cười]!", "[cười][cười]", "[cười].", "[cười]", "[thở dài]!"]:
        assert has_real_words_besides_cues(t) is False, t


def test_lines_with_words_have_real_words():
    for t in ["Đúng rồi [cười]", "[cười] nghe hay", "Hả?", "36D", "Ừ [thở dài] mệt"]:
        assert has_real_words_besides_cues(t) is True, t


# [v3.23.457] Dòng cười THUẦN (cả dòng là tiếng cười) KHÔNG được chuyển thành cue-only (sẽ bị bỏ
# qua → mất tiếng). Giữ nguyên text để VieNeu đọc được.
def test_laugh_only_line_kept_as_text():
    for t in ["ha ha ha!", "Ha ha ha!", "hahaha", "khà khà", "Ha Ha Ha!"]:
        assert inject_emotion_cues(t) == t, f"dòng cười thuần bị đổi thành cue: {t!r}"


def test_embedded_laugh_still_converted():
    assert inject_emotion_cues("Ừ ha ha ha vui thật") == "Ừ [cười] vui thật"
    assert inject_emotion_cues("Đúng rồi hahaha") == "Đúng rồi [cười]"
