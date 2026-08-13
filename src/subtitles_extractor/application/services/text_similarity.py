"""Module tính độ tương đồng giữa các chuỗi text — Backend Hybrid + Linguistic Guard.

CẢI TIẾN ĐỘT PHÁ (Linguistic Intelligence):
    - Tích hợp bộ từ điển 'Tử Huyệt' Tiếng Trung (Antonyms / Subject reversals).
    - Phân biệt tức thời những câu sai 1 chữ nhưng trái ngược hoàn toàn ý nghĩa
      (VD: "我爱你" vs "他不爱你" -> Similarity ép về 0.0).
    - [CRITICAL BUG FIX] Ngăn chặn CJK Substring Boost bơm điểm sai sự thật cho các câu phụ đề
      siêu ngắn ("恭喜呀" vs "恭喜").
"""

from __future__ import annotations

import re
from functools import lru_cache

from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

from subtitles_extractor.application.services.cjk_utils import contains_cjk
from subtitles_extractor.infrastructure.nlp.fastembed_adapter import FastEmbedAdapter

__all__ = [
    "clear_caches",
    "contains_cjk",
    "hybrid_semantic_similarity",
    "jaro_winkler_similarity",
    "raw_fuzz_ratio",
    "text_similarity",
    "viterbi_similarity",
    "viterbi_similarity_with_cutoff",
]

_PUNCTUATION_REGEX = re.compile(r'[^\w\s]', flags=re.UNICODE)

# Danh sách "Tử huyệt" - Đảo ngược ý nghĩa, Chủ thể, Hành động
_CJK_CRITICAL_REVERSALS = frozenset({"不", "没", "你", "我", "他", "她", "它", "是", "非", "有", "无", "男", "女", "去", "来", "要", "别", "好", "坏", "买", "卖", "死", "活", "多", "少", "大", "小"})

# Cache thể hiện của Adapter
_nlp_adapter_instance: FastEmbedAdapter | None = None

def _get_nlp_adapter() -> FastEmbedAdapter:
    global _nlp_adapter_instance
    if _nlp_adapter_instance is None:
        _nlp_adapter_instance = FastEmbedAdapter()
    return _nlp_adapter_instance


def raw_fuzz_ratio(text_a: str, text_b: str) -> float:
    if text_a == text_b:
        # Bằng nhau (kể cả cả 2 rỗng) ⇒ similarity tối đa.
        return 1.0
    if not text_a or not text_b:
        return 0.0
    return fuzz.ratio(text_a, text_b) / 100.0


def jaro_winkler_similarity(text_a: str, text_b: str) -> float:
    if text_a == text_b:
        # Bằng nhau (kể cả cả 2 rỗng) ⇒ similarity tối đa.
        return 1.0
    if not text_a or not text_b:
        return 0.0
    return JaroWinkler.normalized_similarity(text_a, text_b)


def hybrid_semantic_similarity(text_a: str, text_b: str) -> float:
    lexical_score = raw_fuzz_ratio(text_a, text_b)
    if lexical_score > 0.90 or lexical_score < 0.20:
        return lexical_score

    try:
        nlp_adapter = _get_nlp_adapter()
        if not nlp_adapter.enabled:
            return lexical_score

        semantic_score = nlp_adapter.cosine_similarity(text_a, text_b)

        if nlp_adapter.mode == "semantic":
            return semantic_score

    except RuntimeError:
        return lexical_score

    is_cjk = contains_cjk(text_a) or contains_cjk(text_b)
    if is_cjk:
        return (lexical_score * 0.4) + (semantic_score * 0.6)
    return (lexical_score * 0.7) + (semantic_score * 0.3)


def _check_cjk_critical_reversal(text_a: str, text_b: str) -> bool:
    """Kiểm tra xem 2 câu ngắn có bị ngược nghĩa hoàn toàn do khác 1 chữ không."""
    len_a, len_b = len(text_a), len(text_b)

    if max(len_a, len_b) > 6:
        return False

    edit_ops = Levenshtein.editops(text_a, text_b)

    if 0 < len(edit_ops) <= 2:
        for op in edit_ops:
            tag, src_pos, dest_pos = op.tag, op.src_pos, op.dest_pos
            diff_char = ""
            if tag in {"replace", "delete"}:
                diff_char = text_a[src_pos]
            elif tag == "insert":
                diff_char = text_b[dest_pos]

            if diff_char in _CJK_CRITICAL_REVERSALS:
                return True
    return False


@lru_cache(maxsize=16384)
def _builder_similarity_cached(text_a: str, text_b: str) -> float:
    if text_a == text_b:
        return 1.0
    if not text_a or not text_b:
        return 0.0

    clean_a = _PUNCTUATION_REGEX.sub('', text_a).replace(' ', '').lower()
    clean_b = _PUNCTUATION_REGEX.sub('', text_b).replace(' ', '').lower()

    if clean_a == clean_b:
        return 0.95 if clean_a else 0.0

    if contains_cjk(clean_a) and contains_cjk(clean_b):
        if _check_cjk_critical_reversal(clean_a, clean_b):
            return 0.0

    len_a, len_b = len(clean_a), len(clean_b)
    max_len = max(len_a, len_b)
    min_len = min(len_a, len_b)

    if max_len > 4 and (len_a > 2 * len_b or len_b > 2 * len_a):
        return 0.0

    has_cjk = contains_cjk(clean_a) or contains_cjk(clean_b)
    is_cjk_substring = False

    if has_cjk and min_len >= 2:
        shorter = clean_a if len_a <= len_b else clean_b
        longer = clean_b if len_a <= len_b else clean_a
        if shorter in longer:
            is_cjk_substring = True

    base_ratio = hybrid_semantic_similarity(text_a, text_b)

    if not has_cjk:
        jw = jaro_winkler_similarity(clean_a, clean_b)
        base_ratio = max(base_ratio, jw)

    length_diff = abs(len_a - len_b)
    if length_diff > 0:
        diff_ratio = length_diff / max_len
        penalty = diff_ratio * 0.40
        base_ratio -= penalty
    else:
        if not has_cjk and base_ratio >= 0.60:
            base_ratio += 0.10

    if is_cjk_substring:
        #[CRITICAL FIX]: Gỡ bỏ án tử hình cho việc gộp sai các câu siêu ngắn.
        # "恭喜" nằm lọt thỏm trong "恭喜呀" nhưng max_len chỉ bằng 3,
        # Nên nó sẽ KHÔNG BỊ BOOST lên 85%, qua đó bảo vệ nó khỏi việc bị gộp nuốt sai lầm.
        if max_len >= 4 and length_diff <= 2:
            base_ratio = max(base_ratio, 0.85)

    return max(0.0, min(1.0, base_ratio))


def text_similarity(left: str, right: str) -> float:
    if left > right:
        left, right = right, left
    return _builder_similarity_cached(left, right)


def viterbi_similarity(text_a: str, text_b: str) -> float:
    if text_a > text_b:
        text_a, text_b = text_b, text_a
    return _builder_similarity_cached(text_a, text_b)


def viterbi_similarity_with_cutoff(
    text_a: str, text_b: str, score_cutoff: float
) -> float:
    if text_a > text_b:
        text_a, text_b = text_b, text_a

    score = _builder_similarity_cached(text_a, text_b)
    return score if score >= score_cutoff else 0.0


def clear_caches() -> None:
    _builder_similarity_cached.cache_clear()
