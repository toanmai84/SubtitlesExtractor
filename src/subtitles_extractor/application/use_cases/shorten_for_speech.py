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


__all__ = ["MIN_WORD_OVERLAP", "ShortenForSpeechUseCase", "ShortenedLine", "validate_shortened"]
