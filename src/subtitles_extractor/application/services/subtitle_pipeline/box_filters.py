"""Lọc rác cấp **box** trong từng frame OCR.

Phân biệt với :mod:`spatial_filters` (lọc theo vị trí không gian) và
:mod:`event_filters` (lọc cấp event sau khi đã gộp): module này chạy SỚM
NHẤT, dọn rác từng box ngay sau khi OCR trả về để tránh nhiễu lan toả
vào các giai đoạn sau.

Bao gồm:
    * :func:`pre_filter_garbage_boxes` — lọc bỏ box rác đơn thuần (chỉ
      dấu câu, single Latin/digit conf thấp, single CJK conf rất thấp,
      digit-only token dài, Latin lặp ký tự).
    * :func:`is_latin_gibberish` — phát hiện chuỗi Latin rác do OCR sinh
      ra từ logo/watermark (vd 'COZA', 'LKTR', 'ANVSE').
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Sequence

from subtitles_extractor.application.services.cjk_utils import is_predominantly_cjk
from subtitles_extractor.application.services.subtitle_pipeline.constants import (
    DIGIT_ONLY_TOKEN_REGEX,
    LATIN_REPETITIVE_REGEX,
    LATIN_VALID_SHORT_TOKENS,
    LATIN_VOWELS,
    PURE_PUNCTUATION_REGEX,
    SINGLE_ALPHANUMERIC_REGEX,
)
from subtitles_extractor.application.services.subtitle_pipeline.text_correction import (
    correct_hallucination_typos,
)

# Regex biên dịch sẵn (module-level) — box_filters chạy cho MỖI box OCR
# (hàng chục nghìn lần/phim), tránh re-compile mỗi lần gọi.
_DOUBLE_VOWEL_RUN_RE = re.compile(r"(?:A{2,}|I{2,}|U{2,})")
_LONG_CONSONANT_RUN_RE = re.compile(r"[^AEIOUY]{3,}")
_UPPERCASE_RUN_RE = re.compile(r"[A-Z]+")
from subtitles_extractor.domain.entities.ocr_frame_result import (
    OcrFrameResult,
    OcrTextBox,
)


#: [CJK-Context Watermark Drop] Token Latin/digit viết hoa ngắn (watermark động).
UPPER_LATIN_DIGIT_TOKEN_REGEX = re.compile(r"[A-Z0-9]{2,10}")
#: Từ Latin hợp lệ thường gặp trong thoại CJK — không drop.
LATIN_TOKEN_WHITELIST: frozenset[str] = frozenset(
    {"OK", "KTV", "VIP", "NO", "YES", "SOS", "GPS", "ATM", "DNA", "CEO",
     "WIFI", "APP", "AI", "NBA", "MBA", "DJ", "PPT", "GDP", "BGM", "CP"}
)


def _cjk_char_ratio(text: str) -> float:
    """Tỉ lệ ký tự CJK (Hán/Kana/Hangul) trong chuỗi, bỏ whitespace."""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    cjk = sum(
        1 for c in chars
        if 0x4E00 <= ord(c) <= 0x9FFF or 0x3400 <= ord(c) <= 0x4DBF
        or 0xF900 <= ord(c) <= 0xFAFF or 0x3040 <= ord(c) <= 0x30FF
        or 0xAC00 <= ord(c) <= 0xD7AF
    )
    return cjk / len(chars)


def pre_filter_garbage_boxes(
    frames: Sequence[OcrFrameResult],
) -> list[OcrFrameResult]:
    """Lọc bỏ các box rác cấp đơn box trong từng frame.

    Các trường hợp bị loại:
        * Box text rỗng / chỉ whitespace.
        * Box chỉ chứa dấu câu (vd '-', '~').
        * Box 1 ký tự Latin/digit với confidence < 0.95.
        * Box 1 ký tự CJK với confidence < 0.40.
        * Box >= 4 ký tự chỉ chứa chữ số + space, confidence < 0.85
          (rác từ timer/logo).
        * Box >= 4 ký tự Latin có ký tự lặp >= 3 lần, confidence < 0.80
          (rác như 'Moooo', 'AAAAA').

    Args:
        frames: Chuỗi frame OCR đã sắp xếp theo thời gian.

    Returns:
        Danh sách frame mới — các box rác đã được loại bỏ. Frame nào
        không còn box hợp lệ sẽ bị bỏ ra khỏi danh sách.
    """
    cleaned_frames: list[OcrFrameResult] = []

    for frame in frames:
        valid_boxes: list[OcrTextBox] = []
        for box in frame.text_boxes:
            cleaned_text = box.text.strip()
            if not cleaned_text:
                continue

            cleaned_text = correct_hallucination_typos(cleaned_text)

            if PURE_PUNCTUATION_REGEX.fullmatch(cleaned_text):
                continue
            if (
                SINGLE_ALPHANUMERIC_REGEX.fullmatch(cleaned_text)
                and float(box.confidence) < 0.95
            ):
                continue
            if (
                len(cleaned_text) == 1
                and is_predominantly_cjk(cleaned_text)
                and float(box.confidence) < 0.40
            ):
                continue

            box_confidence_value = float(box.confidence)

            if (
                len(cleaned_text) >= 4
                and DIGIT_ONLY_TOKEN_REGEX.fullmatch(cleaned_text)
                and box_confidence_value < 0.85
            ):
                continue

            if (
                len(cleaned_text) >= 4
                and LATIN_REPETITIVE_REGEX.fullmatch(cleaned_text)
                and box_confidence_value < 0.80
            ):
                continue

            valid_boxes.append(dataclasses.replace(box, text=cleaned_text))

        # [CJK-Context Watermark Drop] Frame có phụ đề CJK + box Latin/digit viết
        # hoa ngắn đi kèm (watermark động ``NOVION``/``ARBOR``/``UOK`` trôi qua
        # ROI). Các box này đổi text mỗi frame, phá vỡ grouping (vỡ event, nhân
        # đôi câu) → drop ngay từ box-level. Whitelist từ hợp lệ (OK/VIP/KTV…).
        if len(valid_boxes) >= 2:
            has_cjk_box = any(
                _cjk_char_ratio(b.text) >= 0.5 for b in valid_boxes
            )
            if has_cjk_box:
                valid_boxes = [
                    b for b in valid_boxes
                    if not (
                        UPPER_LATIN_DIGIT_TOKEN_REGEX.fullmatch(b.text.strip())
                        and b.text.strip().upper() not in LATIN_TOKEN_WHITELIST
                        and _cjk_char_ratio(b.text) < 0.5
                    )
                ]

        if valid_boxes:
            cleaned_frames.append(dataclasses.replace(frame, text_boxes=valid_boxes))

    return cleaned_frames


def is_latin_gibberish(text: str, confidence: float, frame_count: int) -> bool:
    """Phát hiện chuỗi Latin "rác" do OCR sinh ra từ logo/watermark.

    Đặc điểm rác Latin OCR:
        * Toàn ký tự ASCII (không lẫn CJK).
        * Độ dài 3-12 ký tự, thường viết hoa.
        * Tỷ lệ nguyên âm cực thấp/cao hoặc pattern bất thường.

    v3.6+: Cải thiện phát hiện watermark cao fc như 'LENCAACA' (fc=15, conf=0.97)
    bằng kiểm tra double-vowel run và long-consonant run.

    Ví dụ rác bị bắt: ``COZA``, ``LKTR``, ``ANVSE``, ``LENCAACA``,
    ``GAPST``, ``GNPSU``.

    Args:
        text: Chuỗi đã loại bỏ ký tự không-word.
        confidence: Điểm tin cậy trung bình của event.
        frame_count: Số khung hình đóng góp.

    Returns:
        ``True`` nếu xác định là rác Latin và nên drop.
    """
    no_space_text = "".join(text.split()).strip()
    if not no_space_text:
        return False

    if no_space_text.upper() in LATIN_VALID_SHORT_TOKENS:
        return False

    text_length = len(no_space_text)
    if text_length > 12:
        return False

    letter_chars = [c for c in no_space_text if c.isalpha()]
    if not letter_chars:
        return text_length <= 6 and confidence < 0.70

    # Confidence >= 0.85 với fc nhỏ (< 7) → văn bản thật, không phải watermark.
    # Với fc >= 7 watermark vẫn có confidence cao → KHÔNG exempt.
    if frame_count < 7 and confidence >= 0.85:
        return False

    vowel_count = sum(1 for c in letter_chars if c in LATIN_VOWELS)
    letter_count = len(letter_chars)
    vowel_ratio = vowel_count / letter_count

    is_consonant_heavy = vowel_ratio <= 0.20
    is_vowel_heavy = vowel_ratio >= 0.75
    is_pattern_suspicious = is_consonant_heavy or is_vowel_heavy

    is_all_uppercase = no_space_text.isupper()

    # ── v3.6+: Tiền tính pattern bất thường để quyết định exempt fc>=7 ──
    # Chỉ bắt AA/II/UU (thực sự bất thường) - không bắt EU, EA, OO, EE (từ thật thông dụng).
    has_double_vowel_run = bool(_DOUBLE_VOWEL_RUN_RE.search(no_space_text.upper()))
    has_long_consonant_run = bool(_LONG_CONSONANT_RUN_RE.search(no_space_text.upper()))
    has_long_uppercase_run = any(
        len(run) >= 4 for run in _UPPERCASE_RUN_RE.findall(no_space_text)
    )
    # Watermark suspicious = bất kỳ pattern bất thường nào trong chuỗi viết hoa.
    watermark_suspicious = (
        is_consonant_heavy
        or has_double_vowel_run
        or has_long_consonant_run
    )

    # ── v3.6+: High-fc exemption — chỉ exempt khi KHÔNG phải watermark suspicious ──
    # Trước đây blanket `frame_count >= 7 → return False` bỏ sót
    # watermark như 'LENCAACA' (có 'AA' = double vowel run → không exempt).
    if frame_count >= 7:
        if is_all_uppercase and watermark_suspicious:
            pass  # Tiếp tục kiểm tra — không exempt watermark
        else:
            # Không phải pattern bất thường + fc >= 7 → có thể là từ thật.
            return False

    if is_pattern_suspicious and confidence < 0.75 and frame_count <= 4:
        return True

    if is_pattern_suspicious and confidence < 0.60 and frame_count <= 10:
        return True

    # ── v3.6+: All-uppercase với pattern bất thường — bắt bất kể fc ──
    if has_long_uppercase_run and is_all_uppercase and watermark_suspicious:
        if text_length <= 10 and frame_count <= 30:
            return True

    if has_long_uppercase_run and confidence < 0.65 and frame_count <= 4:
        return True

    if is_all_uppercase and is_consonant_heavy and confidence < 0.75 and frame_count <= 6:
        return True

    if (
        3 <= text_length <= 5
        and is_all_uppercase
        and confidence < 0.50
        and frame_count <= 4
    ):
        return True

    return False


__all__ = ["is_latin_gibberish", "pre_filter_garbage_boxes"]
