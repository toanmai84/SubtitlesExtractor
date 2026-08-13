"""Sửa lỗi text OCR cấp ký tự và khôi phục ký tự bị drop.

Bao gồm:
    * :func:`correct_hallucination_typos` — sửa lỗi ảo giác OCR phổ biến
      (mapping cố định, vd 涘→埃).
    * :func:`clean_edge_noise` — trim dấu câu rác ở 2 đầu chuỗi.
    * :func:`accumulate_confidence_bucket` — cộng dồn evidence theo
      ngưỡng high/medium cho các restorer.
    * :func:`apply_yi_suffix_restore` — phục hồi ký tự CJK bị OCR drop
      (đặc biệt là chữ '一' đầu/giữa câu và kết tử '了/吗/呢/吧' cuối câu).
    * :func:`restore_dropped_space` — phục hồi space bị drop trong cấu
      trúc `vai trò + lời nói` của phụ đề CJK.

Tất cả các hàm là **pure function** (không state) — dễ unit test.
"""

from __future__ import annotations

from loguru import logger

from subtitles_extractor.application.services.cjk_utils import is_predominantly_cjk
from subtitles_extractor.application.services.subtitle_pipeline.constants import (
    EDGE_NOISE_PUNCTUATION_REGEX,
    HAN_TRADITIONAL_TO_SIMPLIFIED,
    OCR_HALLUCINATION_TYPO_MAP,
)


def normalize_traditional_to_simplified(text: str) -> str:
    """Chuẩn hoá Hán Phồn → Hán Giản trong ``text``.

    Reference phụ đề Trung Quốc hiện đại thường dùng Hán giản; PaddleOCR
    đôi khi output phồn (vd "純陽" → "纯阳") gây mismatch. Hàm này chạy
    char-by-char qua bảng :data:`HAN_TRADITIONAL_TO_SIMPLIFIED` để
    chuẩn hoá toàn bộ ký tự phồn về giản.

    Args:
        text: Chuỗi gốc (có thể chứa cả phồn và giản).

    Returns:
        Chuỗi đã chuẩn hoá hoàn toàn về Hán giản. Các ký tự không nằm
        trong bảng mapping được giữ nguyên.
    """
    if not text:
        return text
    # Char-by-char translate là cách nhanh nhất cho mapping single char.
    # `str.translate` đòi hỏi ord(char) → str, nên ta dùng dict comprehension
    # 1 lần cache ở module-level cũng được, nhưng do chuỗi thường ngắn,
    # join generator đơn giản và đủ nhanh.
    return "".join(HAN_TRADITIONAL_TO_SIMPLIFIED.get(ch, ch) for ch in text)


def correct_hallucination_typos(source_text: str) -> str:
    """Sửa các lỗi ảo giác OCR phổ biến trước khi phân tích nội dung.

    Args:
        source_text: Chuỗi text cần sửa.

    Returns:
        Chuỗi đã sửa lỗi theo bảng :data:`OCR_HALLUCINATION_TYPO_MAP`.
    """
    for wrong_pattern, correct_pattern in OCR_HALLUCINATION_TYPO_MAP.items():
        if wrong_pattern in source_text:
            source_text = source_text.replace(wrong_pattern, correct_pattern)
    return source_text


def clean_edge_noise(text_input: str) -> str:
    """Trim dấu câu rác ở 2 đầu chuỗi.

    [v3.7 NOTE] ĐÃ THỬ wire ``normalize_traditional_to_simplified`` vào đây để
    đồng nhất output về giản-thể. Tuy nhiên đo lường trên test4 cho thấy phụ đề
    "chuẩn" cũng là sản phẩm OCR và dùng LẪN phồn/giản (113 dòng chứa 煙/脈/強/
    輩...). Builder và chuẩn thường ra CÙNG dạng phồn cho cùng frame → normalize
    builder→giản phá vỡ các cặp vốn đã khớp ⇒ net-negative. Vì vậy KHÔNG auto
    normalize ở đây; hàm ``normalize_traditional_to_simplified`` vẫn export để
    dùng có chủ đích (vd video giản-thể thuần) nếu cần.

    Args:
        text_input: Chuỗi đầu vào (có thể chứa dấu câu rác ở biên).

    Returns:
        Chuỗi đã trim biên + strip whitespace.
    """
    return EDGE_NOISE_PUNCTUATION_REGEX.sub("", text_input).strip()


def _max_adjacent_repeat(text: str) -> int:
    """Số lần lặp liên tiếp lớn nhất của một ký tự (vd '一一' → 2, '来来来' → 3)."""
    if not text:
        return 0
    best = run = 1
    for idx in range(1, len(text)):
        run = run + 1 if text[idx] == text[idx - 1] else 1
        if run > best:
            best = run
    return best


def is_single_cjk_char(character: str) -> bool:
    """Kiểm tra ``character`` có phải đúng 1 ký tự CJK (Hán/Hiragana/Katakana).

    Args:
        character: Chuỗi cần kiểm tra.

    Returns:
        ``True`` nếu chuỗi dài 1 ký tự và thuộc dải Unicode CJK/Kana.
    """
    if len(character) != 1:
        return False
    code_point = ord(character)
    return (
        0x4E00 <= code_point <= 0x9FFF
        or 0x3400 <= code_point <= 0x4DBF
        or 0x3040 <= code_point <= 0x30FF
    )


def accumulate_confidence_bucket(
    existing_bucket: tuple[int, int] | None,
    new_confidence: float,
) -> tuple[int, int]:
    """Cộng dồn evidence theo ngưỡng confidence vào bucket (high, medium).

    Args:
        existing_bucket: Cặp ``(high_count, med_count)`` hiện tại, hoặc None.
        new_confidence: Confidence của candidate mới.

    Returns:
        Cặp ``(high_count, med_count)`` đã cộng dồn:
            * high: confidence >= 0.92.
            * medium: 0.85 <= confidence < 0.92.
            * (confidence < 0.85 không được tính.)
    """
    high_count, medium_count = existing_bucket if existing_bucket else (0, 0)
    if new_confidence >= 0.92:
        high_count += 1
    elif new_confidence >= 0.85:
        medium_count += 1
    return (high_count, medium_count)


def apply_yi_suffix_restore(
    voted_text: str,
    texts_with_confidences: list[tuple[str, float]],
) -> str:
    """Phục hồi ký tự CJK bị OCR drop ở đầu/giữa/cuối câu.

    Phát hiện thực nghiệm trên fulltest (3603 câu): PaddleOCR rất hay drop:
        * Ký tự '一' bất kỳ vị trí nào (nét ngang nhỏ dễ nhầm nhiễu).
        * Kết tử cuối câu '了', '个', '吗', '呢', '吧' (font nhỏ ở cuối thường mờ).
        * Ký tự đầu câu bất kỳ CJK — đặc biệt '二', '不', '没', '个', ... (v2.9+).

    Ba pattern được nhận biết:

    * **prefix_prepend**: ``candidate.endswith(voted_text)`` → prepend 1 CJK char
      ở đầu. Bắt ``voted='连灵力'`` candidates ``'个连灵力'``,
      ``voted='师姐'`` candidates ``'二师姐'``, ``voted='愧是美人录里'``
      candidates ``'不愧是美人录里'``, v.v.
    * **suffix_append**: ``candidate.startswith(voted_text)`` → append 1 CJK char
      ở cuối. Bắt drop kết tử ``了/吗/呢/吧/儿`` cuối câu.
    * **yi_insert**: ``'一'`` được chèn vào VỊ TRÍ GIỮA (position > 0) của câu.
      Position 0 đã được xử lý bởi ``prefix_prepend`` nên không cần duplicate.

    Args:
        voted_text: Text đã được ROVER vote.
        texts_with_confidences: Toàn bộ candidate ``(text, conf)`` trong group.

    Returns:
        Text với ký tự được khôi phục nếu thỏa heuristic, ngược lại
        trả ``voted_text`` gốc.
    """
    expected_full_length = len(voted_text) + 1
    candidate_evidence: dict[tuple[str, str, int], tuple[int, int]] = {}

    for candidate_text, candidate_conf in texts_with_confidences:
        if not candidate_text or len(candidate_text) != expected_full_length:
            continue

        # ── Pattern 1: prefix_prepend — voted_text là SUFFIX của candidate ──
        # Bắt drop ký tự đầu câu bất kỳ CJK: '二师姐'→'师姐', '不愧'→'愧',
        # '个连灵力'→'连灵力', '一直盯着'→'直盯着' v.v.
        # Ưu tiên kiểm tra TRƯỚC yi_insert để tránh double-count cho '一' pos-0.
        if candidate_text.endswith(voted_text):
            prepended_char = candidate_text[0]
            if is_single_cjk_char(prepended_char):
                key = ("prefix_prepend", prepended_char, 0)
                candidate_evidence[key] = accumulate_confidence_bucket(
                    candidate_evidence.get(key), candidate_conf
                )
                continue

        # ── Pattern 2: suffix_append — voted_text là PREFIX của candidate ──
        # Bắt drop kết tử cuối câu: '了', '吗', '呢', '吧', '儿', '一'...
        if candidate_text.startswith(voted_text):
            appended_char = candidate_text[-1]
            if is_single_cjk_char(appended_char):
                key = ("suffix_append", appended_char, len(voted_text))
                candidate_evidence[key] = accumulate_confidence_bucket(
                    candidate_evidence.get(key), candidate_conf
                )
                continue

        # ── Pattern 3: yi_insert GIỮA câu (position > 0) ──
        # Chỉ xử lý '一' xuất hiện tại vị trí > 0.  Position-0 đã được
        # xử lý bởi prefix_prepend ở trên để tránh threshold double-count.
        if "一" in candidate_text:
            for yi_idx in range(1, len(candidate_text)):  # bắt đầu từ 1
                if candidate_text[yi_idx] != "一":
                    continue
                stripped = candidate_text[:yi_idx] + candidate_text[yi_idx + 1 :]
                if stripped == voted_text:
                    key = ("yi_insert", "一", yi_idx)
                    candidate_evidence[key] = accumulate_confidence_bucket(
                        candidate_evidence.get(key), candidate_conf
                    )
                    break

    if not candidate_evidence:
        return voted_text

    best_key, (best_high, best_med) = max(
        candidate_evidence.items(),
        key=lambda item: item[1][0] * 2 + item[1][1],
    )

    operation_type, inserted_char, position = best_key

    if operation_type == "prefix_prepend":
        # '一' đầu câu drop rất phổ biến (举两得→一举两得) → 1 high đủ.
        # Ký tự khác ('公','心','二','和'...) ở pos-0 thường là ARTIFACT rìa trái
        # (logo/watermark OCR) hơn là drop thật → cần evidence mạnh hơn (2 high
        # hoặc 4 med) để giảm chèn bậy. Đo trên test4 (cặp ground-truth đáng tin).
        if inserted_char == "一":
            is_strong_evidence = best_high >= 1 or best_med >= 2
        else:
            is_strong_evidence = best_high >= 2 or best_med >= 4
    elif operation_type == "yi_insert":
        # '一' giữa câu: cần evidence chắc hơn vì ít phổ biến hơn đầu câu.
        is_strong_evidence = best_high >= 1 or best_med >= 2
    elif operation_type == "suffix_append":
        # '一' cuối câu (vd '第一', '统一') — khá phổ biến.
        # Các kết tử khác ('了', '儿', ...) cần evidence mạnh hơn.
        if inserted_char == "一":
            is_strong_evidence = best_high >= 1 or best_med >= 2
        else:
            is_strong_evidence = best_high >= 2 or best_med >= 4
    else:
        is_strong_evidence = False

    if not is_strong_evidence:
        return voted_text

    if operation_type == "suffix_append":
        restored = voted_text + inserted_char
    elif operation_type == "yi_insert":
        # Yi-insert có thể ở vị trí bất kỳ giữa câu.
        restored = voted_text[:position] + inserted_char + voted_text[position:]
    elif operation_type == "prefix_prepend":
        # Luôn prepend ở đầu câu.
        restored = inserted_char + voted_text
    else:
        return voted_text

    # [GUARD] Chặn over-correction tạo ký tự CJK lặp liền kề bất thường mà voted gốc
    # KHÔNG có. Ví dụ '不惜一切' + insert '一' → '不惜一一切' (cặp '一一' do OCR
    # double-detect rìa, không phải drop thật). Mẫu lặp 3 lần ('来来来') hợp lệ nên
    # chỉ chặn khi phép chèn LÀM TĂNG số ký tự lặp liền kề so với voted gốc.
    if _max_adjacent_repeat(restored) > _max_adjacent_repeat(voted_text):
        return voted_text

    logger.debug(
        "Char-Restorer: '{}' → '{}' (op={}, char={}, pos={}, high={}, med={})",
        voted_text,
        restored,
        operation_type,
        inserted_char,
        position,
        best_high,
        best_med,
    )
    return restored


def restore_dropped_space(
    voted_text: str,
    texts_with_confidences: list[tuple[str, float]],
) -> str:
    """Phục hồi space bị OCR drop trong câu phụ đề CJK.

    Pattern phổ biến: phụ đề CJK có cấu trúc ``'vai trò + lời nói'`` như
    ``'阿姨 来不及了'``, ``'妈 我先走了'``. PaddleOCR đôi khi detect space,
    đôi khi không. ROVER vote thường chọn version không space.

    Heuristic an toàn:
        * ``voted_text`` predominantly CJK, không có space sẵn.
        * ``voted_text`` >= 5 ký tự (tránh case ngắn 2-4 chars).
        * Trong group có candidate text bằng ``voted_text`` khi bỏ space.
        * Candidate có space đó có >= 2 frames conf >= 0.92 (high) HOẶC
          >= 4 frames conf >= 0.85 (med).

    Args:
        voted_text: Text đã được ROVER vote (thường không có space).
        texts_with_confidences: Toàn bộ candidate ``(text, conf)`` trong group.

    Returns:
        Text với space được restore tại các vị trí evidence-strong,
        ngược lại trả ``voted_text`` gốc.
    """
    if not voted_text or " " in voted_text:
        return voted_text
    if len(voted_text) < 5:
        return voted_text
    if not is_predominantly_cjk(voted_text):
        return voted_text

    space_positions_evidence: dict[tuple[int, ...], tuple[int, int]] = {}

    for candidate_text, candidate_conf in texts_with_confidences:
        if not candidate_text or " " not in candidate_text:
            continue
        candidate_stripped = candidate_text.replace(" ", "")
        if candidate_stripped != voted_text:
            continue
        space_positions: list[int] = []
        characters_seen = 0
        for char in candidate_text:
            if char == " ":
                space_positions.append(characters_seen)
            else:
                characters_seen += 1
        position_key = tuple(space_positions)
        if not position_key:
            continue
        space_positions_evidence[position_key] = accumulate_confidence_bucket(
            space_positions_evidence.get(position_key), candidate_conf
        )

    if not space_positions_evidence:
        return voted_text

    best_positions, (best_high, best_med) = max(
        space_positions_evidence.items(),
        key=lambda item: item[1][0] * 2 + item[1][1],
    )

    is_strong_evidence = best_high >= 2 or best_med >= 4
    if not is_strong_evidence:
        return voted_text

    restored_chars: list[str] = []
    for char_idx, char in enumerate(voted_text):
        if char_idx in best_positions:
            restored_chars.append(" ")
        restored_chars.append(char)
    restored_text = "".join(restored_chars)

    logger.debug(
        "Space-Restorer: '{}' → '{}' (positions={}, high={}, med={})",
        voted_text, restored_text, best_positions, best_high, best_med,
    )
    return restored_text


def restore_dropped_yi_prefix(
    voted_text: str,
    texts_with_confidences: list[tuple[str, float]],
) -> str:
    """Orchestrator: chạy Yi-Restorer rồi Space-Restorer (đúng thứ tự).

    Args:
        voted_text: Text đã được ROVER vote.
        texts_with_confidences: Toàn bộ candidate trong group.

    Returns:
        Text đã được khôi phục ký tự + space (nếu có evidence).
    """
    if not voted_text or len(voted_text) < 2:
        return voted_text

    if not is_predominantly_cjk(voted_text):
        return voted_text

    yi_restored_text = apply_yi_suffix_restore(voted_text, texts_with_confidences)
    return restore_dropped_space(yi_restored_text, texts_with_confidences)


__all__ = [
    "accumulate_confidence_bucket",
    "apply_yi_suffix_restore",
    "clean_edge_noise",
    "correct_hallucination_typos",
    "is_single_cjk_char",
    "normalize_traditional_to_simplified",
    "restore_dropped_space",
    "restore_dropped_yi_prefix",
]
