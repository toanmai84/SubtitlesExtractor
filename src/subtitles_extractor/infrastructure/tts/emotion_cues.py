"""[v3.23.446] Emotion cues (tag cảm xúc) cho VieNeu-TTS v3 Turbo.

VieNeu v3 Turbo (48kHz) hỗ trợ chèn tag phi ngôn ngữ NGAY trong text để giọng đọc biểu cảm hơn
(tài liệu chính thức, THỬ NGHIỆM): ``[cười]``, ``[thở dài]``, ``[hắng giọng]``.
VD: ``tts.infer("Nghe hay quá đi [cười]. Để mình nói tiếp [hắng giọng].")``.

Module này (thuần tính toán) làm 2 việc, CHỈ tác động lên text gửi TTS — KHÔNG đụng phụ đề hiển
thị (người xem không thấy tag):
1. :func:`strip_brackets_keep_cues` — xoá ``[...]`` rác NHƯNG GIỮ 3 tag hợp lệ.
2. :func:`inject_emotion_cues` — chuyển interjection cười/thở dài RÕ RÀNG trong bản dịch tiếng Việt
   thành tag tương ứng (thay thế), để giọng đọc phát ra tiếng cười thật thay vì "đọc" chữ "haha".

Bảo thủ theo thiết kế: chỉ khớp interjection ĐỘC LẬP, rõ ràng — thà bỏ sót còn hơn chèn sai (tag
sai nghe tệ hơn không có). Chỉ dùng cho engine v3 Turbo.
"""

from __future__ import annotations

import re

#: Ba tag cảm xúc VieNeu v3 Turbo hỗ trợ chính thức (chuẩn hoá chữ thường, không dấu cách thừa).
EMOTION_CUES: frozenset[str] = frozenset({
    "[cười]", "[thở dài]", "[hắng giọng]",
    # [v3.23.627] VieNeu 3.8 (vieneu_utils/phonemize_text.py) nhận cả tên tiếng Anh và dạng
    # không dấu của ba tag — trước đây app xoá chúng như "tag rác".
    "[chuckle]", "[sigh]", "[clear throat]", "[cuoi]", "[tho dai]", "[hang giong]",
})

#: Regex một tag ``[...]`` bất kỳ.
_ANY_BRACKET_RE = re.compile(r"\[[^\]]*\]")

#: Chuẩn hoá nội dung tag để so khớp whitelist (thường hoá + gộp khoảng trắng).
def _normalize_cue(bracket: str) -> str:
    inner = bracket[1:-1].strip().lower()
    inner = re.sub(r"\s+", " ", inner)
    return f"[{inner}]"


#: Interjection TIẾNG CƯỜI rõ ràng → ``[cười]``. Yêu cầu ≥2 âm để tránh khớp nhầm ("ha" đơn có
#: thể là "hả"). Khớp cả lặp ("ha ha ha", "hahaha").
_LAUGH_RE = re.compile(
    r"(?<![\w])"
    r"(?:ha(?:\s*ha)+"                     # ha ha, hahaha
    r"|kh[àả](?:\s*kh[àả])+"               # khà khà
    r"|h[ôó](?:\s*h[ôó])+"                 # hô hô
    r"|h[êế](?:\s*h[êế])+"                 # hê hê
    r"|h[ìí](?:\s*h[ìí])+)"               # hì hì, hí hí
    r"(?![\w])",
    re.IGNORECASE,
)

#: Interjection THỞ DÀI rõ ràng → ``[thở dài]``.
_SIGH_RE = re.compile(
    r"(?<![\w])(?:haiz+|hầy|haizz+)(?![\w])",
    re.IGNORECASE,
)


def strip_brackets_keep_cues(text: str) -> str:
    """Xoá mọi ``[...]`` NHƯNG giữ nguyên 3 tag cảm xúc hợp lệ (thay các tag rác bằng khoảng trắng).

    Args:
        text: Text đầu vào (có thể chứa tag rác OCR lẫn tag cảm xúc).

    Returns:
        Text đã xoá tag rác, giữ lại ``[cười]``/``[thở dài]``/``[hắng giọng]``.
    """
    def _repl(match: re.Match[str]) -> str:
        return match.group(0) if _normalize_cue(match.group(0)) in EMOTION_CUES else " "

    return _ANY_BRACKET_RE.sub(_repl, text)


def strip_emotion_cues(text: str) -> str:
    """Xoá HẲN 3 tag cảm xúc khỏi text (dùng khi XUẤT PHỤ ĐỀ — người xem không nên thấy tag).

    Ngược với :func:`strip_brackets_keep_cues` (giữ cue cho TTS): hàm này bỏ cue để file phụ đề
    sạch. Chỉ xoá đúng 3 tag hợp lệ, KHÔNG đụng ``[...]`` khác. Gộp khoảng trắng dư sau khi xoá.

    Args:
        text: Text có thể chứa ``[cười]``/``[thở dài]``/``[hắng giọng]``.

    Returns:
        Text đã bỏ tag cảm xúc.
    """
    def _repl(match: re.Match[str]) -> str:
        return "" if _normalize_cue(match.group(0)) in EMOTION_CUES else match.group(0)

    return re.sub(r"[ \t]{2,}", " ", _ANY_BRACKET_RE.sub(_repl, text)).strip()


def has_real_words_besides_cues(text: str) -> bool:
    """[v3.23.456] True nếu bỏ hết tag cảm xúc mà VẪN còn CHỮ/SỐ thật (không chỉ dấu câu).

    VieNeu KHÔNG tổng hợp được dòng CHỈ có tag (vd ``[cười]!``, ``[cười][cười]``) → ngân dài/méo
    (app_log: audio ~4.5s cho một tag). Dùng để phát hiện dòng chỉ-toàn-cue và bỏ tag đi (→ dòng
    rỗng → bỏ qua) thay vì để model ngân.
    """
    return any(ch.isalnum() for ch in strip_emotion_cues(text))


def inject_emotion_cues(text: str) -> str:
    """Chuyển interjection cười/thở dài RÕ RÀNG thành tag VieNeu (chỉ nên gọi cho text TTS).

    Thay thế (không thêm) để giọng đọc phát tiếng cười/thở dài thật thay vì đọc chữ. Nếu text đã
    có sẵn tag cảm xúc thì giữ nguyên (không nhân đôi).

    Args:
        text: Bản dịch tiếng Việt (một dòng) sắp đưa vào TTS.

    Returns:
        Text đã thay interjection rõ ràng bằng tag tương ứng.
    """
    result = _LAUGH_RE.sub("[cười]", text)
    result = _SIGH_RE.sub("[thở dài]", result)
    # Gộp khoảng trắng có thể sinh ra quanh tag.
    result = re.sub(r"\s+", " ", result).strip()
    # [v3.23.457] Nếu việc thay khiến dòng CHỈ còn tag (không còn lời thật, vd "ha ha ha!" →
    # "[cười]!"), GIỮ NGUYÊN text gốc: VieNeu KHÔNG đọc được tag đứng một mình (ngân dài rồi bị
    # bỏ qua → mất tiếng), còn đọc "ha ha ha" thì vẫn nghe được. Tag chỉ hữu ích khi NHÚNG trong
    # câu có lời (vd "Ừ ha ha ha vui thật" → "Ừ [cười] vui thật").
    if result != text and not has_real_words_besides_cues(result):
        return text
    return result


__all__ = [
    "EMOTION_CUES",
    "strip_brackets_keep_cues",
    "has_real_words_besides_cues",
    "strip_emotion_cues",
    "inject_emotion_cues",
]
