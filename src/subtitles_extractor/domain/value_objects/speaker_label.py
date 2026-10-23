"""[v3.23.600] Nhãn người nói DỰ PHÒNG khi model không nhận ra danh tính người nói.

Trước bản này, phân tích ngữ cảnh và bộ quét Visual Cues được dặn "không chắc thì để
trống người nói". Để trống thì khâu lồng tiếng không biết đọc dòng đó bằng giọng nào
(giọng chung cho mọi dòng trống) — lượt dịch mới nhất của hai phim thật còn 103/3902 và
125/5540 dòng như vậy. Model vẫn NGHE được giọng dòng đó là nam hay nữ, người lớn hay
trẻ con, một người hay đám đông: nhãn dự phòng ghi lại đúng điều nó biết, không bịa danh
tính.

Các nhãn là chuỗi cố định (dùng chung cho prompt, phân vai lồng tiếng và soát người nói).
"""

from __future__ import annotations

from subtitles_extractor.domain.value_objects.voice_profile import (
    VOICE_FEMALE,
    VOICE_MALE,
)

#: Nghe là giọng nam / nữ / trẻ em nhưng không nhận ra là ai.
UNKNOWN_MALE_SPEAKER = "Giọng nam chưa rõ"
UNKNOWN_FEMALE_SPEAKER = "Giọng nữ chưa rõ"
UNKNOWN_CHILD_SPEAKER = "Giọng trẻ em chưa rõ"
#: Nhiều người cùng nói.
CROWD_SPEAKER = "Đám đông"
#: Lời kể / thuyết minh ngoài câu chuyện.
NARRATOR_SPEAKER = "Người dẫn chuyện"

#: Nhãn chỉ ghi GIỌNG, chưa rõ là ai (cần người dùng soát nếu muốn đúng vai).
VOICE_ONLY_SPEAKERS = (
    UNKNOWN_MALE_SPEAKER, UNKNOWN_FEMALE_SPEAKER, UNKNOWN_CHILD_SPEAKER,
)
#: Mọi nhãn dự phòng.
FALLBACK_SPEAKERS = (*VOICE_ONLY_SPEAKERS, CROWD_SPEAKER, NARRATOR_SPEAKER)

_FALLBACK_GENDERS = {
    UNKNOWN_MALE_SPEAKER: VOICE_MALE,
    UNKNOWN_FEMALE_SPEAKER: VOICE_FEMALE,
}
#: Cách model hay viết lệch đi một chút (chữ thường, bỏ "chưa rõ", đổi từ gần nghĩa).
_FALLBACK_SPELLINGS = {
    "giọng nam": UNKNOWN_MALE_SPEAKER,
    "giọng nữ": UNKNOWN_FEMALE_SPEAKER,
    "giọng trẻ em": UNKNOWN_CHILD_SPEAKER,
    "giọng trẻ con": UNKNOWN_CHILD_SPEAKER,
    "giọng trẻ con chưa rõ": UNKNOWN_CHILD_SPEAKER,
    "lời dẫn chuyện": NARRATOR_SPEAKER,
    "người kể chuyện": NARRATOR_SPEAKER,
    **{label.casefold(): label for label in FALLBACK_SPEAKERS},
}


def is_fallback_speaker(label: str) -> bool:
    """Nhãn có phải nhãn dự phòng (không khẳng định danh tính) không."""
    return (label or "").strip() in FALLBACK_SPEAKERS


def canonical_fallback_speaker(label: str) -> str:
    """Nhãn dự phòng chuẩn của ``label`` ("giọng nam" → "Giọng nam chưa rõ"); rỗng nếu
    ``label`` không phải nhãn dự phòng.

    Args:
        label: Nhãn đã bỏ phần trong ngoặc.
    """
    return _FALLBACK_SPELLINGS.get(" ".join((label or "").casefold().split()), "")


def fallback_speaker_gender(label: str) -> str:
    """Giới tính mà nhãn dự phòng ghi lại (rỗng nếu không nói lên giới tính)."""
    return _FALLBACK_GENDERS.get((label or "").strip(), "")


__all__ = [
    "CROWD_SPEAKER",
    "FALLBACK_SPEAKERS",
    "NARRATOR_SPEAKER",
    "UNKNOWN_CHILD_SPEAKER",
    "UNKNOWN_FEMALE_SPEAKER",
    "UNKNOWN_MALE_SPEAKER",
    "VOICE_ONLY_SPEAKERS",
    "canonical_fallback_speaker",
    "fallback_speaker_gender",
    "is_fallback_speaker",
]
