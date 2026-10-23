"""[v3.23.592] Mục thuật ngữ chỉ ĐỌC ÂM Hán-Việt, cách nói dễ hiểu nằm trong ngoặc.

Bản phân tích thật v3.23.591 (phim "灭门惨案II", gemini-3.8-flash):

    所長 = sở trưởng (trưởng đồn công an)
    性自樂 = tính tự nhạc (thuật ngữ hài hước chỉ việc tự sướng/thủ dâm)

Khâu dịch chép phần TRƯỚC ngoặc vào lời thoại, nên bản cuối của lượt đó gọi "Sở trưởng" ở
cả 15 dòng và "tính tự nhạc" ở cả 4 dòng — trong khi chính model phân tích đã ghi cách nói
người Việt hiểu được vào trong ngoặc. Các bản phân tích không mắc lỗi này cho "Trưởng đồn"
và "tự sướng" ở đúng những dòng ấy.

Đo trên kho chuẩn đoán thật: 19 bảng thuật ngữ khác nhau (3 733 mục) có 89 mục kiểu này,
gặp ở 18/19 bảng. Trên 56 lượt dịch, 607 dòng chứa các thuật ngữ đó và 553 dòng (91%) mang
nguyên phần đọc âm vào bản cuối — khoảng 10 dòng mỗi lượt ("hưng sư vấn tội", "thông
gian", "tự độc", "bạch phú mỹ", "tiền công tận khí"…).

Module này chỉ CHỈ RA các mục đó để soát tay, KHÔNG tự đổi: khoảng một phần tư là từ
Hán-Việt người Việt vẫn dùng ("phát tài (trúng mánh)", "nghĩa phụ (cha nuôi)", "bạo kích
(nhân bội số cao)"), đưa phần trong ngoặc ra thay là làm hỏng. Phân biệt hai loại cần biết
từ nào có trong tiếng Việt — việc của người soát hoặc của model phân tích (prompt phân
tích của bản này đã nói rõ hơn về phần trước ngoặc).

Toàn bộ là hàm thuần: không I/O.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from subtitles_extractor.application.services.glossary_options import (
    split_glossary_note,
    term_aliases,
    translation_options,
)
from subtitles_extractor.application.services.name_reading_check import (
    NameReadings,
    syllable_key,
)
from subtitles_extractor.application.services.roster_format import parse_glossary

#: Thuật ngữ thuần chữ Hán, từ hai chữ: một chữ đứng riêng thì âm Hán-Việt thường chính là
#: cách dịch ("定 = định").
_HAN_TERM_RE = re.compile(r"^[㐀-䶿一-鿿]{2,}$")


@dataclass(frozen=True, slots=True)
class ReadingOnlyTerm:
    """Một mục thuật ngữ mà phần dùng trong lời thoại chỉ là âm đọc từng chữ.

    Attributes:
        term: Thuật ngữ gốc.
        reading: Phần trước ngoặc — thứ khâu dịch sẽ chép vào lời thoại.
        note: Phần trong ngoặc — cách nói hoặc lời giải thích của chính bản phân tích.
    """

    term: str
    reading: str
    note: str

    def describe(self) -> str:
        """Mục ở dạng ghi trong bảng thuật ngữ: ``gốc = âm đọc (chú thích)``."""
        return f"{self.term} = {self.reading} ({self.note})"


def _reads_each_character(
    term: str, syllables: list[str], readings: NameReadings
) -> bool:
    """Mỗi âm tiết đúng là một âm Hán-Việt của chữ đứng cùng vị trí."""
    if len(syllables) != len(term):
        return False
    for char, syllable in zip(term, syllables, strict=True):
        accepted = {syllable_key(reading) for reading in readings.accepted(char)}
        if not syllable.isalpha() or syllable_key(syllable) not in accepted:
            return False
    return True


def find_reading_only_terms(
    glossary: str, readings: NameReadings
) -> list[ReadingOnlyTerm]:
    """Các mục chỉ đọc âm Hán-Việt từng chữ mà có chú thích trong ngoặc (hàm thuần).

    Bỏ qua: mục không có chú thích (không có gì để đối chiếu); mục viết HOA chữ đầu (tên
    riêng — "Hắc Long Giang", "Dương Liễu Thanh" — đọc âm là đúng); mục có nhiều phương án
    dịch hoặc nhiều cách viết gốc; thuật ngữ một chữ.

    Args:
        glossary: Bảng thuật ngữ, mỗi dòng ``gốc = dịch``.
        readings: Bảng âm Hán-Việt theo chữ.

    Returns:
        Các mục tìm được, theo thứ tự của bảng; rỗng nếu không có bảng âm.
    """
    if not readings:
        return []
    found: list[ReadingOnlyTerm] = []
    for entry in parse_glossary(glossary or ""):
        term = entry.term.strip()
        if len(term_aliases(term)) != 1 or not _HAN_TERM_RE.match(term):
            continue
        reading, note = split_glossary_note(entry.translation)
        if not note or len(translation_options(entry.translation)) != 1:
            continue
        if not reading[:1].islower():
            continue
        if _reads_each_character(term, reading.split(), readings):
            found.append(ReadingOnlyTerm(term=term, reading=reading, note=note))
    return found


__all__ = ["ReadingOnlyTerm", "find_reading_only_terms"]
