"""[v3.23.137] Test: khử nhãn người nói LẶP ở các dòng liên tiếp cùng người.

Model lite đôi khi chèn '[Tên:]' thẳng vào text mỗi dòng (không qua trường speaker), khiến
nhãn lặp dù cùng người. Lượt hậu xử lý gỡ nhãn trùng người nói dòng ngay trước.
"""

from __future__ import annotations

from subtitles_extractor.application.use_cases.translate_subtitles import (
    TranslateSubtitlesUseCase as UseCase,
)
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval


def _e(index: int, text: str) -> SubtitleEvent:
    return SubtitleEvent(index=index, text=text, interval=TimeInterval(index, index + 1))


def _texts(events: list[SubtitleEvent]) -> list[str]:
    return [e.text for e in events]


def test_consecutive_same_speaker_tagged_once() -> None:
    evs = [
        _e(1, "[Miko Flohr:] Đây là trái tim của đế chế."),
        _e(2, "[Miko Flohr:] Sự giàu có từ khắp nơi"),
        _e(3, "[Miko Flohr:] đổ về Pompeii."),
    ]
    out = _texts(UseCase._suppress_repeated_speaker_tags(evs))
    assert out[0].startswith("[Miko Flohr:]")
    assert out[1] == "Sự giàu có từ khắp nơi"
    assert out[2] == "đổ về Pompeii."


def test_speaker_change_keeps_tag() -> None:
    evs = [
        _e(1, "[Miko Flohr:] Câu một."),
        _e(2, "[Người dẫn chuyện:] Câu hai."),
        _e(3, "[Người dẫn chuyện:] Câu ba."),
    ]
    out = _texts(UseCase._suppress_repeated_speaker_tags(evs))
    assert out[0].startswith("[Miko Flohr:]")
    assert out[1].startswith("[Người dẫn chuyện:]")  # đổi người -> giữ
    assert out[2] == "Câu ba."  # cùng người -> gỡ


def test_sound_effect_resets_speaker() -> None:
    evs = [
        _e(1, "[Người dẫn chuyện:] Trước vụ nổ."),
        _e(2, "(tiếng nổ lớn)"),
        _e(3, "[Người dẫn chuyện:] Sau vụ nổ."),
    ]
    out = _texts(UseCase._suppress_repeated_speaker_tags(evs))
    assert out[0].startswith("[Người dẫn chuyện:]")
    assert out[1] == "(tiếng nổ lớn)"
    # SFX ngắt mạch -> người nói kế được TAG LẠI.
    assert out[2].startswith("[Người dẫn chuyện:]")


def test_case_and_whitespace_variants_match() -> None:
    evs = [
        _e(1, "[Miko Flohr:] Một."),
        _e(2, "[MIKO  FLOHR:] Hai."),  # hoa + thừa khoảng trắng -> cùng người
    ]
    out = _texts(UseCase._suppress_repeated_speaker_tags(evs))
    assert out[0].startswith("[Miko Flohr:]")
    assert out[1] == "Hai."


def test_does_not_create_empty_line() -> None:
    # Nếu gỡ nhãn làm dòng trống thì GIỮ NGUYÊN nhãn (không tạo dòng rỗng).
    evs = [
        _e(1, "[Narrator:] Mở đầu."),
        _e(2, "[Narrator:]"),  # chỉ có nhãn, gỡ ra sẽ trống
    ]
    out = _texts(UseCase._suppress_repeated_speaker_tags(evs))
    assert out[1] == "[Narrator:]"  # giữ nguyên


def test_music_line_resets() -> None:
    evs = [
        _e(1, "[A:] X."),
        _e(2, "♪ ♪"),
        _e(3, "[A:] Y."),
    ]
    out = _texts(UseCase._suppress_repeated_speaker_tags(evs))
    assert out[2].startswith("[A:]")  # nhạc ngắt mạch -> tag lại
