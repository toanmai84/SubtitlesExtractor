"""[v3.23.629] Rút gọn các câu dịch quá dài so với khung giọng đọc (khép vòng dịch ↔ TTS).

Khâu dịch đã gửi ngân sách âm tiết cho model, nhưng vẫn còn dòng phải nén giọng mạnh — và dòng
người dùng tự sửa / thêm ở trang TTS thì không có ngân sách. Use case này nhận danh sách dòng không
vừa khung (:class:`~subtitles_extractor.infrastructure.tts.tts_fit.LineFit`), nhờ AI viết lại NGẮN
HƠN mà giữ nghĩa / xưng hô, rồi kiểm từng câu trả về trước khi cho áp dụng.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from subtitles_extractor.domain.ports.subtitle_translator_port import (
    SubtitleTranslationError,
    TranslationCancelledError,
)
from subtitles_extractor.infrastructure.tts.text_prep import dem_am_tiet
from subtitles_extractor.infrastructure.tts.tts_fit import LineFit

logger = logging.getLogger(__name__)

#: Tag người nói ở đầu dòng ("[Tên:] lời"). Phải giữ nguyên để lồng tiếng còn nhận ra người nói.
_SPEAKER_TAG_RE = re.compile(r"^\s*(\[[^\]\n]{1,40}:\])\s*")
_CJK_RE = re.compile(r"[぀-ヿ㐀-鿿豈-﫿가-힯]")
#: Câu mới phải rút bớt ít nhất ngần này phần âm tiết nếu chưa đạt mục tiêu mới đáng nhận.
MIN_USEFUL_REDUCTION = 0.15
#: Tỉ lệ âm tiết của câu mới phải có mặt trong câu cũ. Rút gọn đúng nghĩa chủ yếu là BỎ chữ, không
#: thay chữ: đo thật (Gemini 3.1 flash-lite, 24 dòng tập 2) có ca "Hiện giải phóng cho ký chủ" bị trả
#: thành "Mở toàn lực một ngày" (lấy nhầm nội dung dòng kề) — trùng 0/5 âm tiết.
MIN_WORD_OVERLAP = 0.7
#: Số từ MỚI tối đa (từ không có trong câu cũ): rút gọn không được thêm ý — đo: "Sư thúc luôn chăm
#: sóc" (bịa "chăm sóc"), "Chúng ta cứ gọi kiểu cũ" (thêm "cứ", "cũ").
MAX_NEW_WORDS = 1
#: Số dòng kề (trước / sau) gửi kèm để model hiểu mạch.
CONTEXT_LINES = 2


@dataclass(frozen=True, slots=True)
class ShortenedLine:
    """Một câu đã được rút gọn và hợp lệ.

    Attributes:
        position: Vị trí dòng (0-based).
        before: Câu cũ.
        after: Câu mới.
        syllables_before: Số âm tiết cũ.
        syllables_after: Số âm tiết mới.
    """

    position: int
    before: str
    after: str
    syllables_before: int
    syllables_after: int


def _words(text: str) -> list[str]:
    return [w for w in re.findall(r"\w+", _SPEAKER_TAG_RE.sub("", text).casefold()) if w]


def _new_words(before: str, after: str) -> int:
    old = set(_words(before))
    return sum(1 for w in _words(after) if w not in old)


def _word_overlap(before: str, after: str) -> float:
    """Tỉ lệ âm tiết của ``after`` có trong ``before`` (0..1)."""
    new, old = _words(after), set(_words(before))
    return sum(1 for w in new if w in old) / len(new) if new else 0.0


#: [v3.23.702] Từ phủ định: số lượng phải giữ nguyên ("không qua nổi" → "qua nổi" đảo nghĩa — ca thật).
_NEGATIONS = frozenset({"không", "chưa", "chẳng", "chả", "đừng", "chớ", "khỏi", "miễn"})
#: Đại từ xưng hô: rút gọn không được THÊM đại từ mới (ca thật: "đồ nhi của ta" → "đồ tôi" đổi giọng, "Diệp Nhiên" →
#: "Nó" đổi vai).
_PRONOUNS = frozenset({
    "tôi", "ta", "tao", "mày", "ngươi", "anh", "em", "chị", "ông", "bà", "cô", "chú", "bác", "con", "cháu", "nó",
    "hắn", "họ", "ngài", "nàng", "chàng", "huynh", "đệ", "tỷ", "muội", "mình", "cậu", "bạn", "lão", "y",
})


#: [v3.23.706] Từ xưng hô / danh xưng Hán-Việt HAI tiếng: rút gọn không được giữ một nửa ("đệ tử" → "đệ").
TITLE_COMPOUNDS = frozenset({
    ("đệ", "tử"), ("đồ", "nhi"), ("đồ", "đệ"), ("sư", "phụ"), ("sư", "tôn"), ("sư", "huynh"), ("sư", "đệ"),
    ("sư", "tỷ"), ("sư", "muội"), ("sư", "thúc"), ("sư", "bá"), ("sư", "bác"), ("sư", "điệt"), ("sư", "tổ"),
    ("sư", "nương"), ("vi", "sư"), ("chưởng", "giáo"), ("chưởng", "môn"), ("trưởng", "lão"), ("tiền", "bối"),
    ("vãn", "bối"), ("tại", "hạ"), ("các", "hạ"), ("thuộc", "hạ"), ("bổn", "tọa"), ("lão", "phu"),
    ("tiểu", "thư"), ("công", "tử"), ("cô", "nương"), ("thiếu", "gia"), ("phu", "nhân"), ("đại", "nhân"),
    ("bệ", "hạ"), ("điện", "hạ"), ("hoàng", "thượng"), ("túc", "chủ"), ("ký", "chủ"), ("tông", "chủ"),
    ("môn", "chủ"), ("tông", "môn"), ("đại", "ca"), ("đại", "tẩu"), ("huynh", "đệ"), ("tỷ", "muội"),
    ("tiểu", "nhân"), ("đạo", "hữu"),
})


def _standalone_counts(words: list[str]) -> dict[str, int]:
    """Số lần mỗi từ đứng NGOÀI mọi từ ghép trong :data:`TITLE_COMPOUNDS`."""
    counts: dict[str, int] = {}
    for position, word in enumerate(words):
        joined = (
            (position > 0 and (words[position - 1], word) in TITLE_COMPOUNDS)
            or (position + 1 < len(words) and (word, words[position + 1]) in TITLE_COMPOUNDS)
        )
        if not joined:
            counts[word] = counts.get(word, 0) + 1
    return counts


def _breaks_title_compound(before: str, after: str) -> bool:
    """[v3.23.706] Câu rút gọn CẮT ĐÔI một từ xưng hô hai tiếng: bỏ một nửa, nửa kia trơ trọi (hàm thuần).

    Ca thật (A/B 361 câu, 3.5-flash-lite): "nể mặt ngươi là đệ tử của sư tỷ" → "Nể tình ngươi là đệ sư tỷ";
    "xin hãy tìm cho đồ nhi" → "Tìm nơi cho đồ". Bộ kiểm cũ cho qua vì mọi từ còn lại đều có trong câu cũ.
    Đã thử nhận diện từ ghép bằng thống kê trên chính bản dịch (cặp đi liền ≥ 2 lần): trên 361 câu nó bỏ sót cả hai
    ca trên và chặn nhầm việc bỏ lượng từ ("hai trăm đạo lôi kiếp" → "hai trăm lôi kiếp") nên dùng danh sách cố định.
    """
    old_words, new_words = _words(before), _words(after)
    old_pairs = list(zip(old_words, old_words[1:], strict=False))
    new_pairs = list(zip(new_words, new_words[1:], strict=False))
    old_alone, new_alone = _standalone_counts(old_words), _standalone_counts(new_words)
    for pair in set(old_pairs) & TITLE_COMPOUNDS:
        if new_pairs.count(pair) >= old_pairs.count(pair):
            continue
        if any(new_alone.get(half, 0) > old_alone.get(half, 0) for half in pair):
            return True
    return False


def _keep_leading_case(before: str, text: str) -> str:
    """[v3.23.706] Câu cũ mở bằng chữ thường (dòng nối tiếp) thì câu rút gọn cũng mở bằng chữ thường (hàm thuần).

    Giữ nguyên khi từ đầu là tên riêng: từng viết hoa giữa câu cũ, hoặc từ thứ hai cũng viết hoa ("Diệp Nhiên").
    """
    tag = _SPEAKER_TAG_RE.match(text)
    head = tag.group(0) if tag else ""
    body = text[len(head):]
    old_first = _SPEAKER_TAG_RE.sub("", before).lstrip()[:1]
    stripped = body.lstrip()
    if not (old_first.isalpha() and old_first.islower() and stripped[:1].isupper()):
        return text
    tokens = stripped.split()
    first = tokens[0].strip(".,!?…:;\"'()“”‘’«»—–-~") if tokens else ""
    if not first or any(ch.isupper() for ch in first[1:]) or len(first[:1].lower()) != 1:
        return text
    if len(tokens) > 1 and tokens[1][:1].isupper():
        return text
    old_tokens = _SPEAKER_TAG_RE.sub("", before).split()
    if any(token.strip(".,!?…:;\"'()“”‘’«»—–-~") == first for token in old_tokens[1:]):
        return text
    return head + body[: len(body) - len(stripped)] + stripped[:1].lower() + stripped[1:]


#: [v3.23.706] Từ mở mệnh đề điều kiện / nhượng bộ: bỏ đi là đổi nghĩa câu (ca thật: "Nếu như con không chống đỡ nổi
#: nữa," → "Con không chống nổi nữa,"; "nếu ngươi không chịu nổi" → "Không chịu nổi").
_CLAUSE_MARKERS = frozenset({"nếu", "dù", "tuy"})


def _changes_meaning_markers(before: str, after: str) -> bool:
    """[v3.23.702] Câu rút gọn đổi dấu hiệu nghĩa: phủ định, kiểu câu hỏi, thêm đại từ, mất tên riêng (hàm thuần).

    Đo trên A/B 361 câu (3.5-flash-lite): bộ kiểm cũ cho qua "thậm chí một đạo lôi kiếp cũng không qua nổi" →
    "qua nổi một kiếp", "Không lẽ ngay cả đạo lôi kiếp đầu tiên" → "Đạo lôi kiếp đầu à?", "Diệp Nhiên là đồ nhi của
    ta" → "Diệp Nhiên là đồ tôi".
    """
    old_words, new_words = _words(before), _words(after)
    if sum(w in _NEGATIONS for w in old_words) != sum(w in _NEGATIONS for w in new_words):
        return True
    if ("?" in before) != ("?" in after):
        return True
    if any(marker in old_words and marker not in new_words for marker in _CLAUSE_MARKERS):
        return True
    if any(w in _PRONOUNS and w not in old_words for w in new_words):
        return True
    strip = ".,!?…:;\"'()“”‘’«»—–-~"
    body = _SPEAKER_TAG_RE.sub("", before).split()
    # Chữ hoa giữa câu = tên riêng; chữ hoa ngay sau dấu kết câu chỉ là đầu câu mới.
    names = {
        word.strip(strip).casefold() for prev, word in zip(body, body[1:])
        if word[:1].isupper() and not prev.rstrip("\"'”’»)").endswith((".", "!", "?", "…"))
    }
    after_tokens = {w.strip(strip).casefold() for w in _SPEAKER_TAG_RE.sub("", after).split()}
    return any(name and name not in after_tokens for name in names)


def validate_shortened(fit: LineFit, before: str, candidate: str) -> str:
    """Câu rút gọn hợp lệ (đã chuẩn hoá) hoặc chuỗi rỗng nếu phải bỏ (hàm thuần).

    Quy tắc: không rỗng, khác bản cũ; giữ tag người nói đầu dòng; không chèn chữ Hán/Nhật/Hàn nếu
    bản cũ không có; ngắn hơn bản cũ và hoặc đạt mục tiêu (cho dư 1 âm tiết) hoặc rút được ≥
    :data:`MIN_USEFUL_REDUCTION`; không ngắn hơn quá mức (giữ ≥ 40% âm tiết — tránh cắt mất nghĩa).
    """
    text = " ".join((candidate or "").split())
    if not text or text == " ".join(before.split()):
        return ""
    tag = _SPEAKER_TAG_RE.match(before)
    if tag and not _SPEAKER_TAG_RE.match(text):
        text = f"{tag.group(1)} {text}"
    if _CJK_RE.search(text) and not _CJK_RE.search(before):
        return ""
    old_syl = dem_am_tiet(_SPEAKER_TAG_RE.sub("", before).strip())
    new_syl = dem_am_tiet(_SPEAKER_TAG_RE.sub("", text).strip())
    if _word_overlap(before, text) < MIN_WORD_OVERLAP or _new_words(before, text) > MAX_NEW_WORDS:
        return ""
    if _changes_meaning_markers(before, text):
        return ""
    if _breaks_title_compound(before, text):
        return ""  # [v3.23.706] cắt đôi từ xưng hô ("đệ tử" → "đệ")
    text = _keep_leading_case(before, text)
    if new_syl <= 0 or new_syl >= old_syl:
        return ""
    if new_syl < old_syl * 0.4:
        return ""
    reduced_enough = new_syl <= fit.target_syllables + 1
    useful = (old_syl - new_syl) / old_syl >= MIN_USEFUL_REDUCTION
    return text if (reduced_enough or useful) else ""


class ShortenForSpeechUseCase:
    """Rút gọn các dòng không vừa khung bằng AI rồi kiểm tra từng câu."""

    def __init__(self, translator: Any) -> None:
        """Khởi tạo.

        Args:
            translator: Adapter dịch có ``shorten_for_speech`` (Gemini).
        """
        self._translator = translator

    def execute(
        self,
        fits: Sequence[LineFit],
        texts: Sequence[str],
        target_lang: str,
        model_name: str,
        *,
        source_texts: Sequence[str] = (),
        speakers: Sequence[str] = (),
        addressees: Sequence[str] = (),
        cancel_cb: Callable[[], bool] | None = None,
    ) -> list[ShortenedLine]:
        """Rút gọn các dòng ``fits``.

        Args:
            fits: Dòng không vừa khung (vị trí + mục tiêu âm tiết).
            texts: Lời hiện tại của MỌI dòng (theo vị trí) — dùng làm bản cũ và ngữ cảnh kề.
            target_lang: Ngôn ngữ bản dịch.
            model_name: Model Gemini.
            source_texts: Câu gốc theo vị trí dòng (rỗng nếu không biết).
            speakers: Người nói theo vị trí dòng.
            addressees: Người nghe theo vị trí dòng.
            cancel_cb: Huỷ giữa chừng.

        Returns:
            Các câu đã rút gọn VÀ hợp lệ (dòng model trả hỏng bị bỏ, giữ nguyên bản cũ).

        Raises:
            SubtitleTranslationError: Bộ dịch không hỗ trợ / lỗi API (trừ huỷ).
        """
        shorten = getattr(self._translator, "shorten_for_speech", None)
        if shorten is None:
            raise SubtitleTranslationError("Bộ dịch không hỗ trợ rút gọn câu.")
        if not fits:
            return []

        def _at(values: Sequence[str], position: int) -> str:
            return str(values[position]) if 0 <= position < len(values) else ""

        items: list[dict[str, Any]] = []
        for fit in fits:
            position = fit.position
            item: dict[str, Any] = {
                "id": position + 1,
                "ban_dich": _at(texts, position),
                "toi_da_am_tiet": fit.target_syllables,
            }
            for key, values in (("nguon", source_texts), ("nguoi_noi", speakers),
                                ("nguoi_nghe", addressees)):
                if _at(values, position):
                    item[key] = _at(values, position)
            before = [_at(texts, position - k) for k in range(CONTEXT_LINES, 0, -1)]
            after = [_at(texts, position + k) for k in range(1, CONTEXT_LINES + 1)]
            if any(before):
                item["truoc"] = [b for b in before if b]
            if any(after):
                item["sau"] = [a for a in after if a]
            items.append(item)
        try:
            returned = shorten(items, target_lang, model_name, cancel_cb=cancel_cb)
        except TranslationCancelledError:
            raise
        by_id = {fit.position + 1: fit for fit in fits}
        accepted: list[ShortenedLine] = []
        for line_id, candidate in sorted(returned.items()):
            fit = by_id.get(line_id)
            if fit is None:
                continue
            before_text = _at(texts, fit.position)
            clean = validate_shortened(fit, before_text, candidate)
            if not clean:
                logger.info("Rút gọn dòng %d: bỏ câu AI trả (không hợp lệ): %r", line_id, candidate)
                continue
            accepted.append(ShortenedLine(
                fit.position, before_text, clean,
                dem_am_tiet(_SPEAKER_TAG_RE.sub("", before_text).strip()),
                dem_am_tiet(_SPEAKER_TAG_RE.sub("", clean).strip()),
            ))
        logger.info("Rút gọn để TTS đọc kịp: %d/%d dòng hợp lệ.", len(accepted), len(fits))
        return accepted


__all__ = [
    "MIN_WORD_OVERLAP", "ShortenForSpeechUseCase", "ShortenedLine", "TITLE_COMPOUNDS", "validate_shortened",
]
