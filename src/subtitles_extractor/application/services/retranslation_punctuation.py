"""[v3.23.603] Dòng DỊCH LẠI ở bước dọn cuối mà kết thúc bằng dấu phẩy ở ranh giới câu.

Bước dọn cuối của khâu dịch (dọn chữ Hán sót, dịch lại dòng trùng/nói lặp) gửi riêng vài
dòng cho model, KHÔNG kèm dòng trước/sau. Thiếu mạch câu, model hay kết dòng bằng dấu phẩy
như thể câu còn tiếp. Đo trên 93 dòng được dịch lại ở mọi file chuẩn đoán thật: 26 dòng
mất dấu kết câu mà bản trước có; ca của chuẩn đoán 11/10 (灭门惨案II): dòng 65 "Thêm chút
dâm tiện, mặt trắng," ngay trước lời của người khác. Giọng đọc gặp dấu phẩy cuối dòng sẽ
đọc như câu bỏ lửng.

Chỉ đóng câu khi chắc chắn là ranh giới câu: dòng sau là NGƯỜI NÓI KHÁC (Visual Cues)
hoặc cách xa từ :data:`SENTENCE_BREAK_GAP_S`, hoặc không còn dòng sau — và dòng sau không
mở đầu bằng chữ thường (câu nói tiếp). Dấu phẩy phải là MỚI (bản trước không kết bằng
dấu phẩy), dấu đóng câu lấy theo bản trước ("..." / "!" / "?"), không có thì dùng ".".

Mô phỏng trên 93 dòng: lưới sửa 9 dòng, soát tay đúng cả 9 ("Đừng, đừng mà," → "Đừng,
đừng mà!" trước lời "Đồ hèn nhát." của người khác); dòng "Không có mấy thứ bẩn thỉu," có
dòng sau "mưu mô đấu đá," (chữ thường — câu còn tiếp dù nhãn ghi người khác) được giữ
nguyên.

Toàn bộ là hàm thuần: không I/O.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

#: Khoảng lặng (giây) từ mức này trở lên giữa hai dòng coi là ranh giới câu.
SENTENCE_BREAK_GAP_S = 1.5

_TRAILING_COMMA_RE = re.compile("\\s*[,;:\uff0c\u3001\uff1b\uff1a]\\s*$")
_TERMINAL_RE = re.compile("([.!?\u2026]+)[\"'\u201d\u2019)\\]]*\\s*$")
_LEADING_NOISE_RE = re.compile("^[\\s\"'\u201c\u2018(\\[\\-\u2013\u2014.\u2026]+")


def _starts_lowercase(text: str) -> bool:
    """Câu bắt đầu bằng chữ thường (bỏ ngoặc/dấu gạch/dấu lửng đứng đầu)."""
    core = _LEADING_NOISE_RE.sub("", text or "")
    return bool(core) and core[0].isalpha() and core[0].islower()


def _is_sentence_break(
    index: int,
    texts: Mapping[int, str],
    spans: Mapping[int, tuple[float, float]],
    speakers: Mapping[int, str],
    gap_s: float,
) -> bool:
    """Giữa dòng ``index`` và dòng sau có phải ranh giới câu không."""
    following = index + 1
    if following not in texts:
        return True
    if _starts_lowercase(texts[following].strip()):
        return False
    speaker, next_speaker = speakers.get(index, ""), speakers.get(following, "")
    if speaker and next_speaker and speaker != next_speaker:
        return True
    if index in spans and following in spans:
        return spans[following][0] - spans[index][1] >= gap_s
    return False


def close_new_trailing_commas(
    before: Mapping[int, str],
    after: Mapping[int, str],
    spans: Mapping[int, tuple[float, float]],
    speakers: Mapping[int, str],
    gap_s: float = SENTENCE_BREAK_GAP_S,
) -> dict[int, str]:
    """Đóng câu ở các dòng vừa đổi chữ mà kết thúc bằng dấu phẩy MỚI ở ranh giới câu.

    Args:
        before: ``index → bản dịch`` trước bước dọn cuối.
        after: ``index → bản dịch`` sau bước dọn cuối (mọi dòng).
        spans: ``index → (giây bắt đầu, giây kết thúc)`` của câu gốc.
        speakers: ``index → người nói`` theo Visual Cues (rỗng nếu không có).
        gap_s: Khoảng lặng tối thiểu coi là ranh giới câu.

    Returns:
        ``index → bản dịch đã đóng câu`` của các dòng được sửa (rỗng nếu không có).
    """
    fixed: dict[int, str] = {}
    for index, text in after.items():
        old = (before.get(index) or "").strip()
        new = (text or "").strip()
        if not new or new == old or not _TRAILING_COMMA_RE.search(new):
            continue
        if _TRAILING_COMMA_RE.search(old):
            continue
        if not _is_sentence_break(index, after, spans, speakers, gap_s):
            continue
        terminal = _TERMINAL_RE.search(old)
        body = _TRAILING_COMMA_RE.sub("", new)
        if body:
            fixed[index] = body + (terminal.group(1) if terminal else ".")
    return fixed


__all__ = ["SENTENCE_BREAK_GAP_S", "close_new_trailing_commas"]
