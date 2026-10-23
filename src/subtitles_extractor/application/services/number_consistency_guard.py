"""[v3.23.556] Số lớn: bắt SAI GIÁ TRỊ và ép NHẤT QUÁN cách đọc.

Bối cảnh đo được (chuẩn đoán thật v3.23.555, phim 3902 dòng, 43 dòng có 万/亿):

* **Sai giá trị**: ``百万年薪`` (lương năm MỘT triệu) ra "mức lương **năm triệu** tệ" —
  người xem nghe thành 5 triệu. Phim thuyết minh sai số tiền còn nặng hơn vụng câu chữ.
* **Không nhất quán**: cùng ``一千万`` khi thì "mười triệu" (dòng 2338), khi thì "một
  nghìn vạn" (5 dòng đoạn 3377-3440) — một khoản tiền, hai cách đọc trong cùng phim.

Vì sao xử bằng MÃ chứ không bằng chỉ thị: số là thứ **tính được**. Tiếng Trung gom theo
10^4 (万) và 10^8 (亿), tiếng Việt gom theo 10^3 (nghìn/triệu/tỉ) — mỗi lô dịch model lại
chọn một kiểu quy đổi. Không chỉ thị nào ép nổi sự nhất quán đó trên 3902 dòng, còn phép
tính thì luôn đúng.

Module chỉ can thiệp khi **chắc chắn**: câu gốc có đúng một cụm số lớn, câu dịch có
đúng một cụm số, và cả hai phân tích được. Dạng rút gọn khó đoán (``一万四`` = 1,4 vạn)
và số ước lượng (``数十亿``, "hàng chục") đều trả về ``None`` và được bỏ qua.

[v3.23.588] "Nhất quán" là theo LỐI ĐỌC CỦA CẢ PHIM, xét riêng số dưới một triệu và số từ
một triệu (xem :class:`_FilmReadings`). Trước đây mỗi con số được xét riêng và lối hiện
đại luôn thắng: chỉ cần MỘT dòng viết "hai trăm nghìn" là năm dòng "hai mươi vạn" bị đổi
theo, trong khi "một vạn", "hai vạn" của cùng phim giữ nguyên.

[v3.23.590] Lối đọc của phim nay áp cho MỌI số tiền cùng hạng, kể cả số tiền phim chưa đọc
theo lối đó lần nào: cách đọc được dựng từ chính giá trị (xem :func:`_reading_in_style`).
Bản 588 chỉ đổi dòng khi phim đã có mẫu cho đúng con số ấy, nên mỗi số tiền một lối — đo
trên bản cuối của 54 lượt dịch thật: 17 lượt đọc lẫn hai lối trong hạng dưới một triệu,
107 dòng lệch. Bằng phiếu thì theo quy ước của hạng (dưới một triệu: 'vạn'). Thêm chữ số
phồn thể ``兩`` và coi hai chữ số kề nhau ("一兩萬", "một hai vạn") là số ước lượng.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction

__all__ = [
    "NumberFinding",
    "NumberFix",
    "chinese_number_value",
    "find_number_findings",
    "find_number_fixes",
    "format_vietnamese_digits",
    "is_malformed_reading",
    "repair_malformed_reading",
    "vietnamese_number_value",
    "vietnamese_number_words",
]

logger = logging.getLogger(__name__)

#: [v3.23.590] Kèm ``兩`` (phồn thể của ``两``). Thiếu chữ này thì "拿兩萬塊錢來贖" chỉ
#: bắt được cụm "萬" không có chữ số nên bị bỏ qua: ca thật v3.23.589, dòng 747 dịch thô
#: ra "hai mươi vạn tệ" (200.000 thay vì 20.000) mà lưới không thấy.
_CN_DIGIT = {
    "零": 0, "一": 1, "二": 2, "两": 2, "兩": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}
_CN_SECTION = {"十": 10, "百": 100, "千": 1000}
_CN_BIG = {"万": 10**4, "萬": 10**4, "亿": 10**8, "億": 10**8}

#: Cụm số tiếng Trung CÓ đơn vị lớn — chỉ nhóm này mới hay bị quy đổi sai.
#: [v3.23.579] Kèm phần THẬP PHÂN ("三十四点五万" = 34,5 vạn, "3.5万"). Trước đây cụm bắt
#: được chỉ là "五万" (chữ sau dấu "点") nên giá trị kỳ vọng thành 50.000 thay vì 345.000.
_CN_NUMBER_RE = re.compile(
    r"[零一二两兩三四五六七八九十百千\d]*"
    r"(?:[点點.\uff0e][零一二两兩三四五六七八九\d]+)?[万萬亿億]"
)
_CN_DECIMAL_RE = re.compile(
    r"(?P<whole>[零一二两兩三四五六七八九十百千\d]+)[点點.\uff0e]"
    r"(?P<fraction>[零一二两兩三四五六七八九\d]+)(?P<unit>[万萬亿億])"
)
#: [v3.23.590] Hai chữ số Hán KỀ NHAU trong phần nguyên là số ƯỚC LƯỢNG: "一兩萬" (một hai
#: vạn), "七八万" (bảy tám vạn), "三四十万". Trước đây cụm này được tính như số chính xác
#: theo chữ số SAU ("一两万" = 20.000) — bản dịch đúng "một hai vạn" sẽ bị coi là sai giá
#: trị.
_CN_ADJACENT_DIGITS_RE = re.compile(r"[一二两兩三四五六七八九]{2}")
_CN_DECIMAL_MARK_RE = re.compile(r"[点點.\uff0e]")

_VI_DIGIT = {
    "không": 0, "một": 1, "mốt": 1, "hai": 2, "ba": 3, "bốn": 4, "tư": 4,
    "năm": 5, "lăm": 5, "sáu": 6, "bảy": 7, "tám": 8, "chín": 9,
}
#: Đơn vị lớn tiếng Việt. 'ức' (10^8) là lối đọc CŨ, tiếng Việt nay dùng 'tỉ'.
_VI_BIG = {
    "tỉ": 10**9, "tỷ": 10**9, "ức": 10**8, "triệu": 10**6,
    "vạn": 10**4, "nghìn": 1000, "ngàn": 1000,
}
#: Đơn vị chỉ dùng trong lối đọc Hán-Việt; tiếng Việt hiện đại gom theo 10^3.
NON_STANDARD_UNITS = frozenset({"vạn", "ức"})
_STANDARD_BIG_UNITS = frozenset(_VI_BIG) - NON_STANDARD_UNITS

#: Từ chỉ số ƯỚC LƯỢNG — không có giá trị chính xác để đối chiếu.
_VI_APPROXIMATE = frozenset({"hàng", "mấy", "vài", "khoảng", "chừng", "độ", "dăm"})

#: Ký tự đứng TRƯỚC cụm số Hán khiến nó thành ước lượng (数十亿 = "vài tỉ").
_CN_APPROXIMATE_PREFIX = frozenset("数數几幾多上余餘来來好近约約将將")

#: [v3.23.556] Cụm KHÔNG PHẢI SỐ dù nhìn như số. ``千万/千萬`` gần như luôn là TRẠNG TỪ
#: ("千万不能" = tuyệt đối không), ``万一`` = "lỡ như". Ca thật: "大家可千万别当真" dịch
#: đúng là "ngàn lần đừng tưởng thật" — nếu coi là 10 triệu thì lưới sẽ phá thành
#: "mười triệu lần đừng tưởng thật".
_CN_NOT_A_NUMBER = frozenset({"千万", "千萬", "万一", "萬一"})

#: Chữ số Hán đứng NGAY SAU cụm số là lối rút gọn (``一万四`` = 1,4 vạn = 14.000) — cụm
#: bắt được ("一万") KHÔNG phải giá trị thật nên phải bỏ qua.
_CN_TRAILING_DIGIT = frozenset("一二两兩三四五六七八九1234567890")

#: Chuỗi ĐƠN VỊ liên tiếp không có chữ số là bài đếm hàng ("个十百千万"), không phải số.
_CN_UNIT_CHARS = frozenset("十百千万萬亿億")

#: [v3.23.579] Chữ số kèm dấu nhóm: "." ngăn nghìn ("3.450"), "," là dấu thập phân
#: ("34,5 vạn", "1,5 triệu") — trước đây "34,5" bị tách thành 34 và 5 rồi CỘNG lại.
_VI_WORD_RE = re.compile(r"(?i)\d+(?:[.,]\d+)*|[a-zà-ỹđ]+")
_VI_DIGITS = r"\d+(?:[.,]\d+)*"
#: Cụm số tiếng Việt: các từ số liên tiếp, kết thúc bằng một đơn vị lớn (có thể kèm
#: "rưỡi": "ba mươi bốn vạn rưỡi" = 345.000).
#: [v3.23.582] Chỉ khớp TRỌN TỪ: trước đây "mức lương năm triệu" ra hai cụm ("ức" trong
#: "mức" là đơn vị 10^8 lối cũ, cũng như "tỉ" trong "tỉnh") → câu bị coi là có hai con số
#: và lưới bỏ qua lỗi "百万年薪" → "năm triệu" ở 15 lượt dịch thật.
_VI_NUMBER_RE = re.compile(
    rf"(?i)(?<!\w)(?:(?:{_VI_DIGITS}|không|một|mốt|hai|ba|bốn|tư|năm|lăm|sáu|bảy|tám|"
    r"chín|mười|mươi|chục|trăm|phẩy|nghìn|ngàn|vạn|triệu|tỉ|tỷ|ức)\s+)*"
    rf"(?:{_VI_DIGITS}\s*)?(?:nghìn|ngàn|vạn|triệu|tỉ|tỷ|ức)(?:\s+rưỡi)?(?!\w)"
)
#: [v3.23.579] "một mươi" không có trong tiếng Việt (10 = "mười") — ca thật: khâu bản
#: địa hoá ghi "một mươi triệu NDT" ở 4 dòng; ĐÚNG giá trị nên lưới số cũ bỏ qua.
_MALFORMED_TENS_RE = re.compile(r"(?i)\b(một)\s+mươi\b")


@dataclass(frozen=True, slots=True)
class NumberFinding:
    """Một dòng có con số đáng ngờ."""

    index: int
    source_expression: str
    expected_value: int
    found_phrase: str
    found_value: int | None
    kind: str  # "mismatch" | "malformed" | "non_standard_unit" | "standard_unit"


@dataclass(frozen=True, slots=True)
class NumberFix:
    """Một dòng sẽ được sửa lại cách đọc số."""

    index: int
    source_expression: str
    phrase_before: str
    phrase_after: str
    text_before: str
    text_after: str
    reason: str


def _cn_digit_string(digits: str) -> str | None:
    """Chuỗi chữ số Hán/Ả Rập → chuỗi chữ số Ả Rập (``"五"`` → ``"5"``)."""
    converted: list[str] = []
    for char in digits:
        if char in _CN_DIGIT:
            converted.append(str(_CN_DIGIT[char]))
        elif char.isdigit():
            converted.append(char)
        else:
            return None
    return "".join(converted)


def _chinese_decimal_value(expression: str) -> int | None:
    """Giá trị cụm số thập phân ``<nguyên>点<lẻ><đơn vị>`` ("三十四点五万" = 345000)."""
    match = _CN_DECIMAL_RE.fullmatch(expression)
    if match is None:
        return None
    whole = chinese_number_value(match.group("whole"))
    fraction = _cn_digit_string(match.group("fraction"))
    if whole is None or not fraction:
        return None
    value = Fraction(f"{whole}.{fraction}") * _CN_BIG[match.group("unit")]
    return int(value) if value.denominator == 1 else None


def chinese_number_value(expression: str) -> int | None:
    """Giá trị của một cụm số tiếng Trung (hàm thuần).

    Args:
        expression: Cụm số, vd ``"一千万"``, ``"十八万"``, ``"三十四点五万"``.

    Returns:
        Giá trị nguyên, hoặc ``None`` nếu cụm chứa ký tự lạ (không phân tích được) hoặc
        phần thập phân không ra số nguyên.
    """
    if any(separator in expression for separator in "点點.\uff0e"):
        return _chinese_decimal_value(expression)
    total = 0
    section = 0
    number = 0
    seen = False
    for char in expression:
        if char in _CN_DIGIT:
            number = _CN_DIGIT[char]
            seen = True
        elif char.isdigit():
            number = number * 10 + int(char)
            seen = True
        elif char in _CN_SECTION:
            section += (number or 1) * _CN_SECTION[char]
            number = 0
            seen = True
        elif char in _CN_BIG:
            section = (section + number) * _CN_BIG[char]
            total += section
            section = 0
            number = 0
            seen = True
        else:
            return None
    return total + section + number if seen else None


def vietnamese_number_value(phrase: str) -> int | None:
    """Giá trị của một cụm số tiếng Việt (hàm thuần).

    Xử lý được lối gom LỒNG NHAU kiểu Hán ("một nghìn vạn" = 1000 * 10^4) — chính nó gây
    ra bản dịch không nhất quán — và lối đọc hàng chục tiếng Việt ("tám mươi lăm").

    Args:
        phrase: Cụm số, vd ``"mười triệu"``, ``"một trăm tám mươi nghìn"``.

    Returns:
        Giá trị nguyên, hoặc ``None`` nếu là số ƯỚC LƯỢNG ("hàng chục", "mấy triệu") hoặc
        không nhận ra chữ số nào.
    """
    tokens = [word.lower() for word in _VI_WORD_RE.findall(phrase or "")]
    if not tokens or set(tokens) & _VI_APPROXIMATE:
        return None
    total = Fraction(0)
    current = Fraction(0)
    last_big = 0
    seen = False
    position = 0
    after_bare_digit = False
    while position < len(tokens):
        token = tokens[position]
        position += 1
        follows_bare_digit, after_bare_digit = after_bare_digit, False
        if token[0].isdigit():
            digits = _vietnamese_digits_value(token)
            if digits is None:
                return None
            current += digits
            seen = True
            continue
        big = _VI_BIG.get(token)
        if big is not None:
            if last_big and big > last_big:
                # "một nghìn vạn": đơn vị sau LỚN hơn → nhân toàn bộ phần đã gom.
                total = (total + current) * big if current else total * big
            else:
                total += (current or 1) * big
            last_big = big
            current = Fraction(0)
            seen = True
            continue
        if token == "phẩy":
            # [v3.23.580] Dấu thập phân ĐỌC THÀNH CHỮ: "ba mươi bốn phẩy năm vạn" = 34,5
            # vạn. Ca thật: lưới chỉ thấy "năm vạn" (50.000) và báo sai giá trị ở hai
            # dòng dịch đúng 345.000.
            digits, position = _spoken_fraction(tokens, position)
            if not digits:
                return None
            current += Fraction(int(digits), 10 ** len(digits))
            continue
        if token == "rưỡi" and last_big and not current:
            # "ba mươi bốn vạn rưỡi" = 34,5 vạn; "một triệu rưỡi" = 1,5 triệu.
            total += Fraction(last_big, 2)
            continue
        if token == "trăm":
            # [v3.23.590] "không trăm" là 0 trăm ("một triệu không trăm hai mươi nghìn");
            # trước đây được tính như "một trăm" nên cách đọc chuẩn của 1.020.000 đọc
            # ngược lại ra 1.120.000 và bị coi là không chắc.
            zero_hundreds = position >= 2 and tokens[position - 2] == "không"
            current = current * 100 if zero_hundreds else (current or 1) * 100
            seen = True
            continue
        if token == "chục":
            # "một chục triệu" = mười triệu.
            current = (current or 1) * 10
            seen = True
            continue
        if token == "mười":
            current += 10
            seen = True
            continue
        if token in _VI_DIGIT:
            if follows_bare_digit:
                # [v3.23.590] Hai chữ số đọc kề nhau ("một hai vạn", "bảy tám vạn") là
                # số ước lượng; trước đây được CỘNG lại ("một hai vạn" = 30.000).
                return None
            digit = _VI_DIGIT[token]
            if position < len(tokens) and tokens[position] == "mươi":
                current += digit * 10
                position += 1
            else:
                current += digit
                after_bare_digit = True
            seen = True
            continue
    value = total + current
    if not seen or value.denominator != 1:
        return None
    return int(value)


def _spoken_fraction(tokens: list[str], position: int) -> tuple[str, int]:
    """Các chữ số đọc sau "phẩy" ("năm", "hai lăm") → ``("5", vị trí kế tiếp)``.

    Trả chuỗi rỗng khi sau "phẩy" là hàng chục/trăm ("phẩy mười lăm") — lối đọc đó
    không rõ giá trị, bỏ qua cho an toàn.
    """
    digits: list[str] = []
    while position < len(tokens) and tokens[position] in _VI_DIGIT:
        digits.append(str(_VI_DIGIT[tokens[position]]))
        position += 1
    if position < len(tokens) and tokens[position] in {"mười", "mươi", "trăm", "chục"}:
        return "", position
    return "".join(digits), position


def _vietnamese_digits_value(token: str) -> Fraction | None:
    """Chữ số viết kiểu Việt: "." ngăn nghìn, "," thập phân ("3.450" = 3450, "34,5")."""
    whole, _, fraction = token.partition(",")
    groups = whole.split(".")
    if len(groups) > 1 and all(len(group) == 3 for group in groups[1:]):
        whole = "".join(groups)
    elif len(groups) == 2 and not fraction:
        # "1.5 triệu": dấu chấm kiểu Anh làm dấu thập phân.
        whole, fraction = groups
    elif len(groups) > 1:
        return None
    if not whole.isdigit() or (fraction and not fraction.isdigit()):
        return None
    return Fraction(f"{whole}.{fraction}") if fraction else Fraction(int(whole))


_VI_DIGIT_WORDS = ("không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín")
#: Nhóm ba chữ số theo đơn vị tiếng Việt hiện đại (không dùng 'vạn').
_VI_GROUP_UNITS = ((10**9, "tỷ"), (10**6, "triệu"), (1000, "nghìn"))


def _hundreds_reading(value: int, leading: bool) -> list[str]:
    """Đọc nhóm 1..999 ("ba trăm bốn mươi lăm"); nhóm không đứng đầu đọc đủ hàng."""
    hundreds, rest = divmod(value, 100)
    tens, units = divmod(rest, 10)
    words: list[str] = []
    if hundreds or not leading:
        words += [_VI_DIGIT_WORDS[hundreds], "trăm"]
    if tens == 0:
        if units:
            if words:
                words.append("lẻ")
            words.append(_VI_DIGIT_WORDS[units])
        return words
    words += ["mười"] if tens == 1 else [_VI_DIGIT_WORDS[tens], "mươi"]
    if units == 1 and tens > 1:
        words.append("mốt")
    elif units == 5:
        words.append("lăm")
    elif units:
        words.append(_VI_DIGIT_WORDS[units])
    return words


def vietnamese_number_words(value: int) -> str:
    """[v3.23.581] Cách đọc CHUẨN tiếng Việt của một số nguyên dương (hàm thuần).

    Args:
        value: Số cần đọc, vd ``3450000``.

    Returns:
        Chuỗi đọc, vd ``"ba triệu bốn trăm năm mươi nghìn"``; rỗng nếu ``value`` ≤ 0.
    """
    if value <= 0:
        return ""
    words: list[str] = []
    remainder = value
    for unit, name in _VI_GROUP_UNITS:
        group, remainder = divmod(remainder, unit)
        if group:
            words += [*_hundreds_reading(group, leading=not words), name]
    if remainder:
        words += _hundreds_reading(remainder, leading=not words)
    return " ".join(words)


#: Cụm số viết bằng chữ số kèm MỘT đơn vị ("3.455 vạn", "34,5 triệu").
_VI_DIGIT_PHRASE_RE = re.compile(rf"(?i)^({_VI_DIGITS})\s+([a-zà-ỹđ]+)$")
#: Tối đa ngần này chữ số thập phân khi viết lại ("3,45 triệu"); lẻ hơn thì bỏ qua.
_MAX_REWRITTEN_DECIMALS = 3


def format_vietnamese_digits(amount: Fraction) -> str | None:
    """[v3.23.582] Viết số kiểu Việt: "." ngăn nghìn, "," thập phân (hàm thuần).

    Args:
        amount: Số dương, vd ``Fraction(345)`` hay ``Fraction(69, 20)``.

    Returns:
        ``"345"``, ``"3.450"``, ``"3,45"``; ``None`` nếu âm/0 hoặc lẻ quá
        :data:`_MAX_REWRITTEN_DECIMALS` chữ số thập phân.
    """
    scale = 10**_MAX_REWRITTEN_DECIMALS
    if amount <= 0 or (amount * scale).denominator != 1:
        return None
    whole, fraction = divmod(int(amount * scale), scale)
    written = f"{whole:,}".replace(",", ".")
    decimals = str(fraction).rjust(_MAX_REWRITTEN_DECIMALS, "0").rstrip("0")
    return f"{written},{decimals}" if decimals else written


def _digit_reading(phrase: str, value: int) -> str | None:
    """Viết lại cụm "chữ số + đơn vị" cho đúng ``value``, GIỮ đơn vị của chính cụm.

    Ca thật v3.23.581 (phim "he thong sung the", dòng 667 "三百四十五万"): khâu văn phong
    gõ nhầm "3.455 vạn tệ" (34,55 triệu) trong khi dòng 665 ngay trên là "345 vạn tệ" —
    sửa thành "345 vạn" là sửa ít nhất và khớp lối viết của đoạn phim.
    """
    match = _VI_DIGIT_PHRASE_RE.match(phrase.strip())
    unit = _VI_BIG.get(match.group(2).lower()) if match else None
    if match is None or unit is None:
        return None
    written = format_vietnamese_digits(Fraction(value, unit))
    if written is None:
        return None
    reading = f"{written} {match.group(2)}"
    return reading if vietnamese_number_value(reading) == value else None


def _standard_reading(phrase: str, value: int) -> str | None:
    """Cách đọc chuẩn khi cả phim không có cách đọc đúng nào; ``None`` khi không chắc.

    Cụm viết bằng chữ: đọc chuẩn từ giá trị. [v3.23.582] Cụm viết bằng chữ số ("3.455
    vạn"): sửa phần chữ số, giữ đơn vị — không đổi lối viết của phim. Kết quả luôn phải
    đọc ngược lại ra đúng ``value`` (lối "lẻ" ngoài bộ đọc thì bỏ qua).
    """
    if phrase[:1].isdigit():
        return _digit_reading(phrase, value)
    reading = vietnamese_number_words(value)
    return reading if reading and vietnamese_number_value(reading) == value else None


#: Lối 'vạn' chỉ dựng cho số dưới 1000 vạn — từ đó trở lên phải mượn "nghìn" ("một nghìn
#: vạn"), tức là trộn hai lối.
_MAX_VAN_COUNT = 1000
_VAN = 10**4


def _van_reading_of(phrase: str, value: int) -> str | None:
    """[v3.23.590] Cách đọc lối 'vạn' của ``value``, cùng kiểu viết với ``phrase``.

    Chữ số → chữ số ("200 nghìn" → "20 vạn"), chữ → chữ ("hai trăm nghìn" → "hai mươi
    vạn"; lẻ nửa vạn đọc "… vạn rưỡi"). Số lẻ hơn thế, hoặc từ 1000 vạn trở lên, không
    có cách đọc thuần 'vạn' gọn gàng nên trả ``None``.
    """
    count, rest = divmod(value, _VAN)
    if not 0 < count < _MAX_VAN_COUNT:
        return None
    if phrase[:1].isdigit():
        written = format_vietnamese_digits(Fraction(value, _VAN))
        return f"{written} vạn" if written else None
    if rest not in (0, _VAN // 2):
        return None
    return f"{vietnamese_number_words(count)} vạn" + (" rưỡi" if rest else "")


def _modern_reading_of(phrase: str, value: int) -> str | None:
    """[v3.23.590] Cách đọc lối hiện đại của ``value``, cùng kiểu viết với ``phrase``.

    Chữ số → chữ số với đơn vị hợp độ lớn ("345 vạn" → "3,45 triệu"), chữ → cách đọc
    chuẩn ("hai mươi vạn" → "hai trăm nghìn").
    """
    if not phrase[:1].isdigit():
        return vietnamese_number_words(value) or None
    unit, name = next(
        ((unit, name) for unit, name in _VI_GROUP_UNITS if value >= unit),
        (0, ""),
    )
    written = format_vietnamese_digits(Fraction(value, unit)) if unit else None
    return f"{written} {name}" if written else None


def _reading_in_style(phrase: str, value: int, in_van: bool) -> str | None:
    """Cách đọc DỰNG từ chính giá trị theo một lối đọc; ``None`` khi không chắc.

    Dùng khi phim chưa có mẫu nào theo lối của nó cho con số này. Kết quả luôn phải đọc
    ngược lại ra đúng ``value`` và đúng lối (không trộn 'vạn' với nghìn/triệu).
    """
    build = _van_reading_of if in_van else _modern_reading_of
    reading = build(phrase, value)
    if not reading or vietnamese_number_value(reading) != value:
        return None
    if _is_pure_van(reading) != in_van:
        return None
    return reading


def is_malformed_reading(phrase: str) -> bool:
    """[v3.23.579] Cụm số đọc SAI NGỮ PHÁP dù đúng giá trị ("một mươi triệu")."""
    return _MALFORMED_TENS_RE.search(phrase or "") is not None


def repair_malformed_reading(phrase: str) -> str:
    """Sửa "một mươi" → "mười" (giữ chữ hoa đầu cụm)."""
    def _tens(match: re.Match[str]) -> str:
        return "Mười" if match.group(1)[:1].isupper() else "mười"

    return _MALFORMED_TENS_RE.sub(_tens, phrase)


def _touches_unit(text: str, start: int, end: int) -> bool:
    """Ký tự ngay trước/sau đoạn ``text[start:end]`` là một đơn vị số Hán."""
    before = text[start - 1] if start > 0 else ""
    after = text[end] if end < len(text) else ""
    return bool(before and before in _CN_UNIT_CHARS) or bool(
        after and after in _CN_UNIT_CHARS
    )


def _single_source_number(text: str) -> tuple[str, int] | None:
    """Cụm số lớn DUY NHẤT của câu gốc kèm giá trị; ``None`` nếu không có/có nhiều."""
    found = list(_CN_NUMBER_RE.finditer(text or ""))
    if len(found) != 1:
        return None
    match = found[0]
    expression = match.group()
    if expression in _CN_NOT_A_NUMBER:
        return None
    if match.start() > 0 and text[match.start() - 1] in _CN_APPROXIMATE_PREFIX:
        return None
    if match.end() < len(text) and text[match.end()] in _CN_TRAILING_DIGIT:
        return None
    # [v3.23.582] Đơn vị dính liền ngoài cụm bắt được = bài đếm hàng bị cắt khúc. Ca thật:
    # "个十百千万" rồi "十万百千" (dịch "Trăm vạn, nghìn vạn...") — cụm chỉ bắt "十万" nên
    # lưới "sửa" thành "100 nghìn".
    if _touches_unit(text, match.start(), match.end()):
        return None
    if _CN_ADJACENT_DIGITS_RE.search(_CN_DECIMAL_MARK_RE.split(expression)[0]):
        return None
    # [v3.23.579] Chỉ loại bài đếm hàng KHÔNG có chữ số nào, đúng như mô tả của hằng số —
    # trước đây mọi cụm ≥ 3 đơn vị đều bị bỏ, kể cả số thật "三千四百三十万" (34,3 triệu).
    if sum(1 for char in expression if char in _CN_UNIT_CHARS) >= 3 and not any(
        char in _CN_DIGIT or char.isdigit() for char in expression
    ):
        return None
    value = chinese_number_value(expression)
    return (expression, value) if value else None


#: Từ số đứng trước đơn vị lớn — cụm chỉ có đơn vị ("triệu") không phải con số.
_VI_NUMERAL_WORDS = frozenset(_VI_DIGIT) | {"mười", "mươi", "chục", "trăm"}


def _has_numeral(phrase: str) -> bool:
    """[v3.23.582] Cụm có ít nhất một chữ số hay từ số, không chỉ có đơn vị.

    Ca thật v3.23.581: "cho mục tiêu Triệu Hồng Quyên" (họ 赵 = Triệu) bị nhận là con số
    "triệu" (1.000.000) — sai giá trị so với "三十四点五万" nên lưới đọc chuẩn thay họ
    người thành "Ba trăm bốn mươi lăm nghìn Hồng Quyên". "triệu phú" cũng vậy.
    """
    return any(
        word[:1].isdigit() or word.lower() in _VI_NUMERAL_WORDS
        for word in _VI_WORD_RE.findall(phrase)
    )


def _single_target_number(text: str) -> tuple[str, int] | None:
    """Cụm số DUY NHẤT của câu dịch kèm giá trị; ``None`` nếu không có hoặc có nhiều."""
    matches = [
        phrase
        for phrase in (m.group().strip() for m in _VI_NUMBER_RE.finditer(text or ""))
        if _has_numeral(phrase)
    ]
    if len(matches) != 1:
        return None
    value = vietnamese_number_value(matches[0])
    return (matches[0], value) if value else None


def _pairs(
    sources: dict[int, str], translations: dict[int, str]
) -> list[tuple[int, str, int, str, int]]:
    """Dòng có ĐÚNG một cụm số hai phía: ``(index, gốc, trị_gốc, cụm_dịch, trị_dịch)``."""
    rows: list[tuple[int, str, int, str, int]] = []
    for index, source in sources.items():
        found_source = _single_source_number(source)
        if found_source is None:
            continue
        found_target = _single_target_number(translations.get(index, ""))
        if found_target is None:
            continue
        expression, expected = found_source
        phrase, actual = found_target
        rows.append((index, expression, expected, phrase, actual))
    return rows


def _uses_non_standard_unit(phrase: str) -> bool:
    """Cụm số dùng đơn vị Hán-Việt ('vạn', 'ức') thay vì nghìn/triệu/tỉ."""
    words = {word.lower() for word in _VI_WORD_RE.findall(phrase)}
    return bool(words & NON_STANDARD_UNITS)


#: Dòng ``(chỉ số, cụm số gốc, giá trị gốc, cụm số dịch, giá trị dịch)``.
_Row = tuple[int, str, int, str, int]

#: Đơn vị của CÂU GỐC mà tiếng Việt đọc được theo cả hai lối: "二十萬" là "hai mươi vạn"
#: hay "hai trăm nghìn". Con số gốc không có các đơn vị này thì không có gì để chọn.
_CN_VAN_UNITS = frozenset("万萬亿億")


#: Ranh giới hai HẠNG độ lớn có lối đọc riêng. Đo trên bản cuối của 52 lượt dịch thật
#: (ba phim): số DƯỚI một triệu đọc 'vạn' ở 395 dòng, lối hiện đại ở 132 dòng; số TỪ
#: một triệu đọc lối hiện đại ở 1057 dòng, thuần 'vạn' ở 10 dòng — "năm mươi vạn tệ"
#: nhưng "hai mươi triệu tệ" là quy ước tự nhiên của bản dịch, không phải hai lối lẫn lộn.
_MILLION = 10**6


@dataclass(frozen=True, slots=True)
class _FilmReadings:
    """Các cách đọc số bộ phim đã dùng, kèm LỐI ĐỌC chủ đạo theo hạng độ lớn.

    Attributes:
        by_value: Giá trị → các cụm phim đã dùng cho giá trị đó kèm số lần.
        van_below_million: Số dưới một triệu được phim đọc theo lối 'vạn' — trong các
            con số gốc có 万/亿 thuộc hạng đó, số dòng đọc thuần 'vạn' NHIỀU HƠN số dòng
            đọc lối hiện đại (nghìn/triệu/tỉ).
        van_from_million: Như trên, cho số từ một triệu trở lên.
    """

    by_value: dict[int, Counter[str]]
    van_below_million: bool
    van_from_million: bool

    def reads_in_van(self, value: int) -> bool:
        """Phim đọc các con số cùng hạng độ lớn với ``value`` theo lối 'vạn'."""
        return self.van_below_million if value < _MILLION else self.van_from_million


def _mixes_unit_styles(phrase: str) -> bool:
    """Cụm số trộn đơn vị Hán-Việt với đơn vị hiện đại ("ba triệu bốn mươi ba vạn")."""
    words = {word.lower() for word in _VI_WORD_RE.findall(phrase)}
    return bool(words & NON_STANDARD_UNITS) and bool(words & _STANDARD_BIG_UNITS)


def _is_pure_van(phrase: str) -> bool:
    """Cụm số đọc THUẦN lối 'vạn'/'ức' ("hai mươi vạn"), không trộn nghìn/triệu."""
    return _uses_non_standard_unit(phrase) and not _mixes_unit_styles(phrase)


def _film_readings(
    rows: list[_Row], prior_rows: list[_Row] | None = None
) -> _FilmReadings:
    """Gom cách đọc của phim và xác định lối đọc chủ đạo (hàm thuần).

    [v3.23.588] Lối đọc là của CẢ PHIM, không của từng con số. Ca thật (phim
    "灭门惨案II", v3.23.587): "一萬" → "một vạn" (3 dòng), "兩萬" → "hai vạn", "二十萬" →
    "hai mươi vạn" (5 dòng), riêng dòng 966 model viết "hai trăm nghìn". Xét riêng con số
    200.000 thì lối hiện đại "thắng" và cả năm dòng đúng bị đổi sang "hai trăm nghìn",
    trong khi "một vạn", "hai vạn" giữ nguyên. Đếm trên cả phim: 9 dòng 'vạn', 1 dòng
    lối hiện đại.

    Args:
        rows: Các dòng số của bản dịch hiện tại.
        prior_rows: Các dòng số của giai đoạn trước — cũng là cách đọc của chính phim.

    Returns:
        Cách đọc theo giá trị và lối đọc chủ đạo của từng hạng độ lớn. [v3.23.590] Bằng
        phiếu (kể cả chưa có phiếu nào) thì theo quy ước chung của hạng đó: dưới một
        triệu đọc 'vạn', từ một triệu đọc lối hiện đại — xem :data:`_MILLION`.
    """
    by_value: dict[int, Counter[str]] = {}
    # Phiếu (lối 'vạn' trừ lối hiện đại) của hạng dưới triệu [True], từ triệu [False].
    lead = {True: 0, False: 0}
    for _index, expression, expected, phrase, actual in (*rows, *(prior_rows or ())):
        by_value.setdefault(expected, Counter())[phrase] += 1
        if actual != expected or is_malformed_reading(phrase):
            continue
        if _is_pure_van(phrase):
            lead[expected < _MILLION] += 1
        elif not _uses_non_standard_unit(phrase) and _CN_VAN_UNITS & set(expression):
            lead[expected < _MILLION] -= 1
    # [v3.23.590] Ca thật v3.23.589 (phim "灭门惨案II"): bản thô đọc 'vạn' ở 4 dòng đúng
    # ("một vạn" ba dòng, "hai mươi vạn" dòng 966), lối hiện đại ở 4 dòng ("hai trăm
    # nghìn"). Bằng phiếu → trước đây về lối hiện đại: dòng 966 bị đổi thành "hai trăm
    # nghìn", bản cuối đọc "một vạn tệ", "hai vạn tệ" nhưng "hai trăm nghìn". Theo quy ước
    # của hạng dưới một triệu thì cả phim về một lối 'vạn'.
    return _FilmReadings(by_value, lead[True] >= 0, lead[False] > 0)


def _canonical_phrase(
    candidates: Counter[str],
    expected: int,
    rejects: Callable[[str], bool] = _uses_non_standard_unit,
) -> str | None:
    """Cách đọc CHUẨN cho một giá trị, chọn trong các cách chính bộ phim đã dùng.

    Ưu tiên: (1) đọc đúng giá trị; (2) đơn vị tiếng Việt hiện đại (không 'vạn'/'ức');
    (3) xuất hiện nhiều lần nhất. Không tự CHẾ cách đọc mới — chỉ chọn trong các cách
    bản dịch đã dùng, để không áp đặt văn phong.

    Args:
        candidates: Các cụm bộ phim đã dùng cho con số này kèm số lần.
        expected: Giá trị đúng.
        rejects: Loại cụm không được chọn làm mẫu. Mặc định loại lối 'vạn'/'ức'.

    Returns:
        Cụm (chữ thường) dùng nhiều nhất trong các cụm hợp lệ; ``None`` nếu không có.
    """
    standard: Counter[str] = Counter()
    for phrase, count in candidates.items():
        if vietnamese_number_value(phrase) != expected:
            continue
        if rejects(phrase) or is_malformed_reading(phrase):
            continue
        standard[phrase.lower()] += count
    if not standard:
        return None
    return max(standard, key=lambda phrase: (standard[phrase], -len(phrase)))


def _same_form_first(
    candidates: Counter[str],
    expected: int,
    like: str,
    rejects: Callable[[str], bool],
) -> str | None:
    """Cách đọc của phim cho ``expected``, ưu tiên cùng kiểu viết với cụm ``like``.

    Chữ ↔ chữ, chữ số ↔ chữ số: phim có cả "hai vạn tệ" lẫn "2 vạn tệ" thì dòng viết
    chữ được sửa bằng cụm viết chữ.

    Args:
        candidates: Các cụm bộ phim đã dùng cho con số này kèm số lần.
        expected: Giá trị đúng.
        like: Cụm sắp bị thay.
        rejects: Loại cụm không được chọn làm mẫu.

    Returns:
        Cụm (chữ thường) để thay; ``None`` nếu phim không có cụm nào hợp lệ.
    """
    in_digits = like[:1].isdigit()

    def _other_form(candidate: str) -> bool:
        return rejects(candidate) or candidate[:1].isdigit() != in_digits

    return _canonical_phrase(candidates, expected, rejects=_other_form) or (
        _canonical_phrase(candidates, expected, rejects=rejects)
    )


def _van_reading(candidates: Counter[str], expected: int, like: str) -> str | None:
    """Cách đọc thuần lối 'vạn' mà phim đã dùng cho ``expected``; ``None`` nếu chưa có."""
    return _same_form_first(
        candidates, expected, like, lambda candidate: not _is_pure_van(candidate)
    )


def _style_target(film: _FilmReadings, expected: int, phrase: str) -> str | None:
    """Cách đọc nên đổi sang cho một cụm ĐÚNG giá trị nhưng lệch lối đọc của phim.

    * Phim đọc lối hiện đại: cụm 'vạn' → cách đọc hiện đại của con số đó.
    * Phim đọc lối 'vạn': cụm hiện đại → cách đọc 'vạn' của con số đó.
    * Cụm trộn hai lối ("một nghìn vạn"): về lối của phim, không được thì về lối hiện đại.

    Ưu tiên cách đọc phim ĐÃ dùng cho chính con số đó. [v3.23.590] Phim chưa có mẫu thì
    DỰNG cách đọc từ giá trị theo lối của phim (:func:`_reading_in_style`). Trước đây
    dòng không có mẫu bị để nguyên, nên mỗi SỐ TIỀN một lối: đo trên bản cuối của 54
    lượt dịch thật, 17 lượt đọc lẫn hai lối trong hạng dưới một triệu (107 dòng lệch) —
    "hai vạn tệ" nhưng "hai trăm nghìn", "năm mươi vạn" nhưng "tám trăm nghìn".

    Args:
        film: Cách đọc và lối đọc của phim.
        expected: Giá trị của cụm.
        phrase: Cụm số đúng giá trị trong bản dịch.

    Returns:
        Cụm (chữ thường) để thay; ``None`` nếu cụm đã đúng lối hoặc không dựng được
        cách đọc chắc chắn.
    """
    candidates = film.by_value[expected]
    reads_in_van = film.reads_in_van(expected)
    if _is_pure_van(phrase):
        if reads_in_van:
            return None
        return _canonical_phrase(candidates, expected) or _reading_in_style(
            phrase, expected, in_van=False
        )
    in_van = None
    if reads_in_van:
        in_van = _van_reading(candidates, expected, phrase) or _reading_in_style(
            phrase, expected, in_van=True
        )
    if _mixes_unit_styles(phrase):
        return (
            in_van
            or _canonical_phrase(candidates, expected)
            or _reading_in_style(phrase, expected, in_van=False)
        )
    return in_van


def _value_target(film: _FilmReadings, expected: int, phrase: str) -> str | None:
    """Cách đọc để sửa một cụm SAI giá trị: theo lối của phim, rồi mới tới đọc chuẩn.

    Args:
        film: Cách đọc và lối đọc của phim.
        expected: Giá trị đúng (theo câu gốc).
        phrase: Cụm số sai giá trị trong bản dịch.

    Returns:
        Cụm (chữ thường) để thay; ``None`` khi không dựng được cách đọc chắc chắn.
    """
    candidates = film.by_value[expected]
    if film.reads_in_van(expected):
        # [v3.23.590] Phim đọc 'vạn' mà chưa có mẫu cho con số này → dựng cách đọc 'vạn'
        # (ca thật: "拿兩萬塊錢來贖" ra "hai mươi vạn tệ" → "hai vạn tệ"), rồi mới tới
        # cách đọc hiện đại phim đã dùng.
        from_film = (
            _van_reading(candidates, expected, phrase)
            or _reading_in_style(phrase, expected, in_van=True)
            or _canonical_phrase(candidates, expected)
        )
    else:
        # Dựng theo lối hiện đại trước khi về cách đọc chuẩn: cụm chữ số "3.455 vạn" phải
        # ra "3,45 triệu" như dòng "345 vạn" cùng số tiền (vừa được đổi lối), không dừng
        # ở "345 vạn" để rồi lượt sau lại đổi tiếp.
        from_film = _canonical_phrase(candidates, expected) or _reading_in_style(
            phrase, expected, in_van=False
        )
    # [v3.23.581] Cả phim không có cách đọc đúng nào (ca thật: "三百四十五万" → "ba triệu
    # bốn trăm năm mươi vạn" = 7,5 triệu ở cả hai dòng của số tiền đó) → đọc chuẩn từ
    # chính giá trị, vì để nguyên là giọng đọc nói sai số tiền.
    return from_film or _standard_reading(phrase, expected)


def find_number_findings(
    sources: dict[int, str], translations: dict[int, str]
) -> list[NumberFinding]:
    """Các dòng có con số dịch SAI GIÁ TRỊ hoặc lệch lối đọc của phim (hàm thuần).

    ``kind`` của mỗi phát hiện: ``mismatch`` (sai giá trị), ``malformed`` (sai cách
    đọc), ``non_standard_unit`` (dùng 'vạn'/'ức' trong phim đọc lối hiện đại) hoặc
    ``standard_unit`` (dùng lối hiện đại trong phim đọc lối 'vạn').

    Args:
        sources: ``index → câu gốc``.
        translations: ``index → bản dịch``.

    Returns:
        Danh sách phát hiện, sắp theo chỉ số dòng.
    """
    rows = _pairs(sources, translations)
    film = _film_readings(rows)
    findings: list[NumberFinding] = []
    for index, expression, expected, phrase, actual in rows:
        if actual != expected:
            kind = "mismatch"
        elif is_malformed_reading(phrase):
            kind = "malformed"
        elif _style_target(film, expected, phrase):
            # Phim đọc nhất quán một lối thì đó là VĂN PHONG, không phải lỗi; chỉ báo khi
            # DÙNG LẪN hai lối cho cùng một con số.
            in_van = _uses_non_standard_unit(phrase)
            kind = "non_standard_unit" if in_van else "standard_unit"
        else:
            continue
        findings.append(
            NumberFinding(
                index=index,
                source_expression=expression,
                expected_value=expected,
                found_phrase=phrase,
                found_value=actual,
                kind=kind,
            )
        )
    return sorted(findings, key=lambda item: item.index)


def find_number_fixes(
    sources: dict[int, str],
    translations: dict[int, str],
    prior_translations: dict[int, str] | None = None,
) -> list[NumberFix]:
    """Các dòng sửa được cách đọc số về đúng và nhất quán với chính bộ phim (hàm thuần).

    Chỉ sửa khi CHÍNH BỘ PHIM đã có cách đọc đúng cho con số đó ở dòng khác — không tự chế
    cách đọc mới. Hai loại được sửa:

    * **Sai giá trị**: "lương năm triệu" trong khi chỗ khác đọc đúng là "một triệu".
    * **Lệch lối đọc**: "một nghìn vạn" trong khi chỗ khác đọc "mười triệu"; hoặc
      [v3.23.588] "hai trăm nghìn" trong một phim đọc tiền theo lối 'vạn'.

    Args:
        sources: ``index → câu gốc``.
        translations: ``index → bản dịch hiện tại``.
        prior_translations: [v3.23.569] Bản của giai đoạn TRƯỚC — cách đọc ở đó cũng được
            dùng làm ứng viên, để giai đoạn sau làm hỏng MỌI chỗ vẫn sửa lại được.

    Returns:
        Danh sách sửa đổi, sắp theo chỉ số dòng.
    """
    rows = _pairs(sources, translations)
    prior_rows = _pairs(sources, prior_translations) if prior_translations else None
    film = _film_readings(rows, prior_rows)

    fixes: list[NumberFix] = []
    for index, expression, expected, phrase, actual in rows:
        text = translations.get(index, "")
        if actual == expected and is_malformed_reading(phrase):
            # Đúng giá trị, sai cách đọc: sửa thẳng, không cần cách đọc mẫu của phim.
            replacement = repair_malformed_reading(phrase)
            reason = "sai cách đọc"
        else:
            # Chỉ sửa khi SAI GIÁ TRỊ hoặc lệch lối đọc — không sửa khác biệt chỉ là chữ
            # hoa/thường hay chữ số/chữ viết (đó là văn phong, không phải lỗi).
            if actual == expected:
                canonical = _style_target(film, expected, phrase)
                reason = "lệch lối đọc"
            else:
                canonical = _value_target(film, expected, phrase)
                reason = "sai giá trị"
            if canonical is None or canonical == phrase.lower():
                continue
            replacement = canonical.capitalize() if phrase[:1].isupper() else canonical
        replaced = text.replace(phrase, replacement, 1)
        if replaced == text:
            continue
        fixes.append(
            NumberFix(
                index=index,
                source_expression=expression,
                phrase_before=phrase,
                phrase_after=replacement,
                text_before=text,
                text_after=replaced,
                reason=reason,
            )
        )
    return sorted(fixes, key=lambda fix: fix.index)
