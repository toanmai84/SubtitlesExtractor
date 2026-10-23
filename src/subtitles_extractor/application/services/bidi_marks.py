"""[v3.23.610] Dấu định hướng cho dòng phụ đề chữ phải-sang-trái (hàm thuần).

Trình phát coi mỗi dòng SRT/ASS là đoạn trái-sang-phải nếu không có dấu định hướng, nên
dấu câu của câu tiếng Ả Rập/Do Thái nhảy sang sai đầu dòng. Học từ gemini-srt-translator
(``_dominant_strong_direction`` + bọc RLE … PDF), khác ở hai điểm:

* bọc TỪNG dòng: ký tự xuống dòng là dấu ngắt đoạn, kết thúc mọi lớp định hướng theo
  chuẩn Unicode Bidi (UAX #9), nên bọc cả khối chỉ có tác dụng ở dòng đầu;
* chỉ bọc khi XUẤT tệp và gỡ khi NẠP tệp, để văn bản trong ứng dụng (sửa, đếm ký tự,
  đọc TTS) không mang ký tự ẩn.
"""

from __future__ import annotations

import unicodedata

__all__ = ["dominant_direction", "mark_rtl_lines", "strip_rtl_marks"]

RLE = "‫"  # RIGHT-TO-LEFT EMBEDDING
PDF = "‬"  # POP DIRECTIONAL FORMATTING
_RTL_CLASSES = frozenset({"R", "AL", "RLE", "RLI", "RLO"})
_LTR_CLASSES = frozenset({"L", "LRE", "LRI", "LRO"})
#: Dòng đã có dấu mở đầu chiều phải-sang-trái thì không bọc thêm.
_RTL_OPENERS = (RLE, "‏", "⁧", "‮")  # RLE, RLM, RLI, RLO


def dominant_direction(text: str) -> str:
    """Chiều của các ký tự mạnh chiếm đa số: ``"rtl"`` hoặc ``"ltr"``.

    Args:
        text: Một dòng chữ.

    Returns:
        ``"rtl"`` nếu ký tự phải-sang-trái nhiều hơn, ngược lại ``"ltr"``.
    """
    rtl = ltr = 0
    for char in text:
        direction = unicodedata.bidirectional(char)
        if direction in _RTL_CLASSES:
            rtl += 1
        elif direction in _LTR_CLASSES:
            ltr += 1
    return "rtl" if rtl > ltr else "ltr"


def _mark_line(line: str) -> str:
    if not line.strip() or line.startswith(_RTL_OPENERS):
        return line
    if dominant_direction(line) != "rtl":
        return line
    return f"{RLE}{line}{PDF}"


def mark_rtl_lines(text: str) -> str:
    """Bọc RLE … PDF quanh từng dòng có chữ phải-sang-trái chiếm ưu thế.

    Args:
        text: Nội dung một câu phụ đề (có thể nhiều dòng, ngăn bởi ``\\n``).

    Returns:
        Nội dung đã bọc; dòng trái-sang-phải và dòng đã có dấu giữ nguyên.
    """
    return "\n".join(_mark_line(line) for line in text.split("\n"))


def _strip_line(line: str) -> str:
    if line.startswith(RLE) and line.endswith(PDF):
        return line[len(RLE):-len(PDF)]
    return line


def strip_rtl_marks(text: str) -> str:
    """Gỡ cặp RLE … PDF do :func:`mark_rtl_lines` thêm (từng dòng).

    Args:
        text: Nội dung đọc từ tệp phụ đề.

    Returns:
        Nội dung không còn cặp dấu bọc dòng.
    """
    return "\n".join(_strip_line(line) for line in text.split("\n"))
