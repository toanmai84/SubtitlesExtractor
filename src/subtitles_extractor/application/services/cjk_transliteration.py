"""[v3.23.673] Chữ Hán sót trong bản dịch tiếng Việt → âm Hán-Việt (bước cứu cuối, hàm thuần).

Chạy thật (tập 3 phim mẫu, gemini-3.5-flash-lite): model viết "九转烈阳蛊" thành
"Cửu Chuyển Liệt Dương C蛊" — gõ chữ đầu của "Cổ" rồi chép nguyên chữ 蛊. Hậu quả:
* bảng thuật ngữ loại hẳn mục đó (dịch dở) → khâu dịch mất luôn cách dịch chuẩn của thuật ngữ;
* dòng 16 vẫn còn "C蛊" sau khi dịch lại → giọng đọc gặp chữ Hán, người xem thấy chữ lạ.

Bước này đổi mỗi chữ Hán còn sót sang âm Hán-Việt CHÍNH của nó (bảng ``hanviet_readings.json``
18 nghìn chữ). Khi ngay trước chữ Hán là phần đầu đã gõ của chính âm đó ("C" + 蛊 → "Cổ") thì
ghép lại thành một âm, giữ chữ hoa. Chữ không có trong bảng (và chữ Nhật/Hàn) được giữ nguyên.
"""

from __future__ import annotations

import re
import unicodedata

from subtitles_extractor.application.services.name_reading_check import NameReadings

_HAN_RUN_RE = re.compile(r"[㐀-䶿一-鿿]+")
_TRAILING_LETTERS_RE = re.compile(r"([^\W\d_]+)$")
_LEADING_LETTERS_RE = re.compile(r"^[^\W\d_]")
#: Ngoặc / nháy MỞ — chữ ngay sau không cần dấu cách.
_OPENING_PUNCT = "([{“‘«「『【〈《—-/"  # (không gồm " ' thẳng: vừa mở vừa đóng)
_TRAILING_WORD_RE = re.compile(r"([^\W\d_]+)[^\w]*$")
_HAN_CHAR_RE = re.compile(r"[㐀-䶿一-鿿]")
_LATIN_LETTER_RE = re.compile(r"[A-Za-zÀ-ỹ]")
#: Số chữ Hán tối đa coi là "phần sót" (thuật ngữ/tên 4 chữ như 九转烈阳 còn nhận).
MAX_LEFTOVER_HAN = 4
#: Phần đầu đã gõ dài tối đa bao nhiêu chữ cái thì mới ghép với âm của chữ Hán ngay sau.
MAX_TYPED_PREFIX = 2


def _fold(text: str) -> str:
    """Bỏ dấu, viết thường (so phần đầu đã gõ với âm đầy đủ)."""
    decomposed = unicodedata.normalize("NFD", text.lower().replace("đ", "d"))
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _reading(char: str, readings: NameReadings) -> str:
    options = readings.suggestable(char) or readings.accepted(char)
    return options[0] if options else ""


def is_vietnamese_target(target_lang: str) -> bool:
    """Ngôn ngữ đích là tiếng Việt (âm Hán-Việt chỉ có nghĩa với bản dịch tiếng Việt).

    Đích tiếng Trung/Nhật: chữ Hán là chữ ĐÚNG của bản dịch; đích tiếng Anh/Pháp…: âm Hán-Việt
    là chữ lạ — cả hai đều KHÔNG được đổi.
    """
    lowered = " ".join((target_lang or "").lower().split())
    code = lowered.replace("_", "-")  # [v3.23.676] "vi_VN" như "vi-VN"
    return code in {"vi", "vie", "vi-vn"} or "vietnam" in lowered or "việt" in lowered


def is_short_leftover(text: str, max_han: int = MAX_LEFTOVER_HAN) -> bool:
    """Chữ Hán chỉ là phần SÓT nhỏ trong câu đã dịch (không phải cả câu chưa dịch).

    Câu chưa dịch ("我爱你") đổi âm từng chữ thành "Ngã Ái Nễ" vô nghĩa và còn che mất cờ QC
    ``cjk_leftover`` — câu đó phải để lại cho người soát.
    """
    han = len(_HAN_CHAR_RE.findall(text or ""))
    latin = len(_LATIN_LETTER_RE.findall(text or ""))
    return 0 < han <= max_han and latin >= 2 * han


def _capitalize(word: str) -> str:
    return word[:1].upper() + word[1:]


def _starts_sentence(before: str) -> bool:
    stripped = before.rstrip()
    return not stripped or stripped[-1] in ".!?…:\"“«-—"


def transliterate_leftover_cjk(text: str, readings: NameReadings) -> str:
    """Đổi chữ Hán còn sót trong ``text`` sang âm Hán-Việt.

    Viết hoa theo ngữ cảnh: dính liền chữ trước ("dương之气") thì theo kiểu chữ trước; đứng sau
    chữ viết hoa (thuật ngữ "Liệt Dương 蛊") hoặc đầu câu thì viết hoa; cụm nhiều chữ đứng riêng
    sau chữ thường thường là tên riêng ("gặp 秦雅" → "gặp Tần Nhã").

    Args:
        text: Câu dịch (tiếng Việt) còn lẫn chữ Hán.
        readings: Bảng âm Hán-Việt theo chữ.

    Returns:
        Câu đã đổi; chữ không có âm trong bảng giữ nguyên (người gọi tự kiểm còn sót không).
    """
    if not text or not readings.primary and not readings.extra:
        return text
    out: list[str] = []
    pos = 0
    for match in _HAN_RUN_RE.finditer(text):
        before = text[pos:match.start()]
        words = [_reading(ch, readings) for ch in match.group()]
        if not all(words):
            # Một chữ thiếu âm → giữ nguyên cả cụm (không đọc dở một nửa tên).
            out.append(before + match.group())
            pos = match.end()
            continue
        full_before = "".join(out) + before
        prefix = _TRAILING_LETTERS_RE.search(before)
        merged = False
        if (prefix and len(prefix.group(1)) <= MAX_TYPED_PREFIX
                and _fold(words[0]).startswith(_fold(prefix.group(1)))):
            # "C" + 蛊 → "Cổ": phần đầu đã gõ chính là âm này (chỉ 1-2 chữ cái — "con公" không
            # phải "công").
            typed = prefix.group(1)
            before = before[: prefix.start()]
            full_before = "".join(out) + before
            merged = True
            upper = typed[:1].isupper()
        elif prefix:
            # Dính liền chữ trước ("dương之气") → cùng kiểu chữ với chữ đó.
            upper = prefix.group(1)[:1].isupper()
            before += " "
        else:
            prev_word = _TRAILING_WORD_RE.search(full_before)
            upper = (
                _starts_sentence(full_before)
                or bool(prev_word and prev_word.group(1)[:1].isupper())
                or len(words) >= 2
            )
            # [v3.23.676] Chỉ chèn cách sau chữ/số/dấu đóng — không sau ngoặc/nháy MỞ ("(蛊)" → "(cổ)", không "( cổ)").
            if before and not before[-1].isspace() and before[-1] not in _OPENING_PUNCT:
                before += " "
        if upper:
            words = [_capitalize(w) for w in words]
        elif merged:
            words = [w.lower() for w in words]
        tail = text[match.end():match.end() + 1]
        joined = " ".join(words)
        if tail and _LEADING_LETTERS_RE.match(tail):
            joined += " "
        out.append(before + joined)
        pos = match.end()
    out.append(text[pos:])
    return "".join(out)


__all__ = ["is_short_leftover", "is_vietnamese_target", "transliterate_leftover_cjk"]
