"""Tiền xử lý văn bản phụ đề trước khi tổng hợp giọng — DÙNG CHUNG cho mọi engine TTS.

[v3.23.220] Trước đây các hàm này nằm trong ``edge_tts_adapter`` và bị ``vieneu`` +
``gemini`` import ngược (adapter phụ thuộc adapter). Tách về module thuần: chỉ phụ thuộc
``domain.ports.subtitle_tts_port`` (DTO), không phụ thuộc SDK/engine nào.

Trách nhiệm: biến một dòng phụ đề THÔ (có thể chứa tag ASS/HTML, tag người nói, ký hiệu
nhạc, chú thích trong ngoặc) thành văn bản SẠCH để đọc + cờ ``is_dialog`` (dòng có gạch
đầu dòng — dùng để chèn khoảng nghỉ hội thoại).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from subtitles_extractor.domain.ports.subtitle_tts_port import TTSRequest

__all__ = [
    "SkipOptions",
    "dem_am_tiet",
    "has_speakable_content",
    "preprocess_tts_text",
    "skip_from_request",
    "wrap_transcript_for_tts",
]

_ASS_TAG_RE = re.compile(r"\{.*?\}")
_HTML_TAG_RE = re.compile(r"<[^>]+>")

_PAREN_RE = re.compile(r"\([^)]*\)")
_SQUARE_RE = re.compile(r"\[[^\]]*\]")
_CURLY_RE = re.compile(r"\{[^}]*\}")
_MUSIC_PAIR_RE = re.compile(r"\u266a[^\u266a]*\u266a")
_MULTISPACE_RE = re.compile(r"\s{2,}")
_DIALOG_DASHES = ("-", "\u2013", "\u2014")
_MUSIC_NOTE = "\u266a"

# Ký tự "phát âm được" = chữ cái (mọi bảng chữ, kể cả có dấu tiếng Việt) hoặc chữ số.
# ``[^\W_]`` = ký tự word TRỪ dấu gạch dưới -> loại sạch dấu câu, ký hiệu (♪, …), ô
# vuông "□" rác từ OCR.
_SPEAKABLE_CHAR_RE = re.compile(r"[^\W_]", re.UNICODE)

# Tag người nói ở đầu dòng, vd "[Lâm Thần:]" — nhãn HIỂN THỊ, không đọc thành tiếng.
_SPEAKER_TAG_RE = re.compile(r"^\s*\[[^\[\]]*[:：]\]\s*")  # noqa: RUF001 (dấu hai chấm CJK là CỐ Ý)


def has_speakable_content(text: str) -> bool:
    """True nếu ``text`` có ít nhất một ký tự phát âm được (chữ hoặc số).

    [v3.23.220] MỘT nguồn sự thật cho cả ba engine. Trước đây tồn tại hai bản cài đặt
    song song (Edge dùng regex ``[^\\W_]``, VieNeu dùng ``any(ch.isalnum())``) — đã đối
    chiếu và chứng minh tương đương trên toàn bộ ca thực tế (chữ Việt có dấu, CJK, Hangul,
    chữ số Ả Rập/Devanagari, ký hiệu nhạc, ô vuông OCR "□", dấu câu, gạch đầu dòng).

    Văn bản chỉ gồm dấu câu, ký hiệu (♪, …) hoặc "□" khiến engine trả về NoAudioReceived
    (Edge) hoặc sinh audio im lặng rồi retry vô ích (VieNeu/Gemini) -> phải chặn TRƯỚC
    khi gọi model.

    Args:
        text: Văn bản đã tiền xử lý.

    Returns:
        True nếu có ít nhất một chữ cái hoặc chữ số.
    """
    return bool(_SPEAKABLE_CHAR_RE.search(text))


@dataclass(frozen=True)
class SkipOptions:
    """Cấu hình bỏ qua nội dung KHÔNG đọc thành tiếng (do người dùng bật trong UI)."""

    paren: bool = False
    square: bool = False
    curly: bool = False
    music_pair: bool = False
    music_line: bool = False

    @property
    def any_enabled(self) -> bool:
        """True nếu có ít nhất một tuỳ chọn bỏ qua được bật."""
        return any(
            (self.paren, self.square, self.curly, self.music_pair, self.music_line)
        )


def skip_from_request(request: TTSRequest) -> SkipOptions:
    """Rút các tuỳ chọn "bỏ qua" từ yêu cầu TTS (chịu được DTO thiếu trường).

    Args:
        request: Yêu cầu TTS từ tầng application.

    Returns:
        :class:`SkipOptions` tương ứng.
    """
    return SkipOptions(
        paren=getattr(request, "skip_paren", False),
        square=getattr(request, "skip_square", False),
        curly=getattr(request, "skip_curly", False),
        music_pair=getattr(request, "skip_music_pair", False),
        music_line=getattr(request, "skip_music_line", False),
    )


def preprocess_tts_text(
    raw_text: str,
    clean_tags: bool,
    skip: SkipOptions | None = None,
    strip_speaker_tag: bool = False,
) -> tuple[str, bool]:
    """Làm sạch một dòng phụ đề cho TTS (hàm thuần).

    Args:
        raw_text: Dòng phụ đề thô (có thể chứa ``\\N``, tag ASS/HTML, tag người nói).
        clean_tags: True để gỡ tag ASS ``{...}`` và HTML ``<...>``.
        skip: Tuỳ chọn bỏ qua nội dung (ngoặc, ký hiệu nhạc…); None = không bỏ gì.
        strip_speaker_tag: True (khi tổng hợp AUDIO) để bỏ tag người nói đầu dòng; False
            (khi xuất FILE PHỤ ĐỀ) để giữ nhãn cho người xem.

    Returns:
        Cặp ``(văn_bản_sạch, là_dòng_hội_thoại)``. ``là_dòng_hội_thoại`` = True khi dòng
        bắt đầu bằng gạch đầu dòng (dùng để chèn khoảng nghỉ tách lượt thoại).
    """
    text = raw_text.replace("\\N", " ").replace("\n", " ")
    if clean_tags:
        text = _HTML_TAG_RE.sub("", text)
        text = _ASS_TAG_RE.sub("", text)
    text = text.strip()

    # [v3.23.41/45] Tag người nói "[Tên:]" ở đầu dòng là NHÃN HIỂN THỊ. Khi tổng hợp
    # AUDIO (strip_speaker_tag=True) thì bỏ để giọng đọc không đọc cả tên; còn với FILE
    # PHỤ ĐỀ đi kèm (strip_speaker_tag=False, mặc định) thì GIỮ để người xem biết ai nói.
    if strip_speaker_tag:
        text = _SPEAKER_TAG_RE.sub("", text).strip()
        # [v3.23.364] CHỈ cho AUDIO: dòng VIẾT HOA TOÀN BỘ (credit/tiêu đề phim, tên nước
        # ngoài) khiến VieNeu "ngân dài" (token hoa hiếm trong huấn luyện) → retry 10 lần
        # rất tốn. Chuyển sang Title Case để giọng đọc tự nhiên hơn. File phụ đề HIỂN THỊ
        # (strip_speaker_tag=False) GIỮ NGUYÊN chữ hoa cho người xem.
        text = _soften_all_caps(text)

    if skip is not None and skip.any_enabled:
        if skip.music_line and text.lstrip().startswith(_MUSIC_NOTE):
            return "", False
        if skip.music_pair:
            text = _MUSIC_PAIR_RE.sub(" ", text)
        if skip.paren:
            text = _PAREN_RE.sub(" ", text)
        if skip.square:
            text = _SQUARE_RE.sub(" ", text)
        if skip.curly:
            text = _CURLY_RE.sub(" ", text)
        text = _MULTISPACE_RE.sub(" ", text).strip()

    # [Dialog Pause Trim Order] Sau khi xoá dấu câu rác (ngoặc cười, ký hiệu…),
    # phải .strip() để đưa dấu '-' về đầu chuỗi, nếu không '(Cười) - Xin chào'
    # → ' - Xin chào' khiến .startswith('-') trượt và mất khoảng lặng hội thoại.
    text = text.strip()
    is_dialog = text.startswith(_DIALOG_DASHES)
    if is_dialog:
        text = text.lstrip("-\u2013\u2014").strip()
    return text, is_dialog


#: Từ viết tắt phổ biến cần GIỮ NGUYÊN chữ hoa (TTS đọc từng chữ cái, không đọc thành từ).
_KNOWN_ACRONYMS: frozenset[str] = frozenset({
    "FBI", "CIA", "KGB", "IBM", "NASA", "USA", "UK", "EU", "UN", "NATO", "WHO",
    "WTO", "CEO", "CFO", "CTO", "GDP", "UNESCO", "UNICEF", "ASEAN", "OPEC", "NBA",
    "NFL", "BBC", "CNN", "USB", "GPS", "ATM", "DNA", "HIV", "AIDS", "LED", "SUV",
    "FIFA", "UEFA", "NGO", "VIP", "SOS", "UFO", "PIN", "TNHH", "CLB",
})


def _is_acronym(word: str) -> bool:
    """True nếu ``word`` là từ VIẾT TẮT cần GIỮ NGUYÊN chữ hoa (TTS đọc từng chữ) — thuần.

    Chỉ dựa trên DANH SÁCH viết tắt phổ biến (không dùng heuristic độ dài) để KHÔNG nhận
    nhầm từ tiếng Việt ngắn viết hoa như "TY", "TA", "EM"… thành viết tắt. Danh sách có
    thể mở rộng khi cần.
    """
    core = word.strip(".,!?;:\"'()[]{}").upper()
    return bool(core) and core in _KNOWN_ACRONYMS


def _soften_all_caps(text: str) -> str:
    """Chuyển dòng VIẾT HOA TOÀN BỘ sang Title Case cho TTS đọc tự nhiên (hàm thuần).

    VieNeu (và TTS neural nói chung) hay "ngân dài"/đọc sai trên chuỗi in hoa vì token
    chữ hoa hiếm trong dữ liệu huấn luyện. Chỉ can thiệp khi ≥80% chữ cái là HOA và có
    tối thiểu 4 chữ cái (dấu hiệu credit/tiêu đề), giữ nguyên câu thường và viết-hoa-đầu.
    Từ VIẾT TẮT (FBI, KGB, IBM, NASA…) được GIỮ NGUYÊN để TTS đọc đúng từng chữ cái.

    Args:
        text: Văn bản đã làm sạch (dùng cho AUDIO).

    Returns:
        Title Case (giữ acronym) nếu là dòng in hoa toàn bộ; ngược lại trả nguyên văn.
    """
    letters = [char for char in text if char.isalpha()]
    if len(letters) < 4:
        return text
    upper_ratio = sum(1 for char in letters if char.isupper()) / len(letters)
    if upper_ratio < 0.8:
        return text
    return " ".join(
        word if _is_acronym(word) else (word[:1].upper() + word[1:].lower()) if word else word
        for word in text.split(" ")
    )


def dem_am_tiet(text: str) -> int:
    """Đếm số âm tiết trong văn bản tiếng Việt (hàm thuần).

    Tiếng Việt là ngôn ngữ **đơn âm tiết**: mỗi "tiếng" viết rời là đúng một âm tiết, và
    thời gian đọc tỉ lệ với SỐ ÂM TIẾT chứ không phải số ký tự. "nghiêng" có 7 ký tự nhưng
    đọc mất đúng bằng "ta" (2 ký tự) — một nhịp.

    Vì sao điều này quan trọng (đo trên 95 câu Gemini thật):

    ==================================  ========
    cách mô hình hoá độ dài giọng       R²
    ==================================  ========
    trung bình theo KÝ TỰ               0.069
    trung bình theo ÂM TIẾT             0.059
    **BIÊN DƯỚI theo ÂM TIẾT**          **0.894**
    ==================================  ========

    Số ký tự mỗi âm tiết dao động từ 2 tới 6 (median 4.2), nên dùng ký tự làm biến là tự
    đưa nhiễu vào mô hình.

    Args:
        text: Văn bản tiếng Việt.

    Returns:
        Số âm tiết (>= 0). Chữ số được tính là một âm tiết cho mỗi cụm.
    """
    if not text or not text.strip():
        return 0
    # Dấu câu và ký hiệu không phát âm -> thay bằng khoảng trắng để tách tiếng.
    sach = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return len([tieng for tieng in sach.split() if tieng])


# [v3.23.245] Preamble bọc transcript cho Gemini native-audio TTS. Theo tài liệu
# chính thức (Limitations - "Prompt classifier false rejections"): prompt/câu quá trơ
# (vd "Haiz.", "Tuổi:") có thể KHÔNG kích hoạt speech-synthesis classifier -> model đọc
# lệch, ngân dài, hoặc đọc cả chỉ thị. Google khuyến nghị: thêm preamble rõ ràng yêu cầu
# model tổng hợp giọng, và ĐÁNH DẤU rõ chỗ transcript thật bắt đầu.
#
# Đo được trên FLAC thật: 12 câu hallucination (đọc 3-8x mức tối thiểu) phần lớn là câu
# NGẮN trơ trọi — đúng nhóm mà classifier dễ bỏ sót. Preamble này nhắm đúng nguyên nhân
# gốc thay vì chỉ chữa triệu chứng bằng retry.
_TTS_PREAMBLE = (
    "Đọc to đoạn thoại sau bằng giọng tự nhiên, "
    "chỉ đọc đúng nội dung trong dấu ngoặc kép:"
)


def wrap_transcript_for_tts(text: str, preamble: str = _TTS_PREAMBLE) -> str:
    """[v3.23.245] Bọc transcript bằng preamble để classifier nhận đúng là lời cần đọc.

    Theo khuyến nghị tài liệu Gemini TTS: câu trơ trọi (nhất là câu ngắn) dễ khiến model
    không nhận ra đây là transcript -> đọc lệch/ngân dài. Preamble + nhãn ngoặc kép giúp
    phân định rõ "phần chỉ thị" và "phần cần đọc".

    Hàm thuần, idempotent về mặt ngữ nghĩa: chuỗi rỗng trả về nguyên trạng (để lớp skip
    xử lý), không tự bọc câu đã rỗng.

    Args:
        text: Transcript gốc (một dòng phụ đề đã tiền xử lý).
        preamble: Câu chỉ thị đứng trước transcript.

    Returns:
        Chuỗi đã bọc ``<preamble> "<text>"``; hoặc chuỗi gốc nếu rỗng.
    """
    goc = text.strip()
    if not goc:
        return text
    return f'{preamble} "{goc}"'
