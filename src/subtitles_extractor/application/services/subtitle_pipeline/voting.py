"""Bỏ phiếu ROVER trên các candidate text trong cùng một :class:`FrameGroup`.

ROVER (Recognition Output Voting Error Reduction) — kỹ thuật từ ASR
research, áp dụng cho OCR: gộp nhiều bản đọc của cùng một câu để bỏ
phiếu chọn ký tự tốt nhất ở mỗi vị trí.

Pipeline:
    1. **Majority short-circuit** — nếu 1 text xuất hiện >= 45% group,
       chọn luôn (không cần vote phức tạp).
    2. **Anchor selection** — chọn text "tốt nhất" làm khung xương dựa
       trên composite score = 0.6*frequency + 0.4*confidence - 0.15*z_len.
    3. **Per-position voting** — với mỗi vị trí trong anchor, dùng
       Levenshtein opcodes cho tất cả candidates khác, tích luỹ
       confidence cho từng ký tự ứng viên, chọn cao nhất.
    4. **Insertion handling** — chèn ký tự ngoài anchor nếu evidence
       vượt 50% tổng confidence pool.
    5. **Post-processing** — chạy Yi-Restorer và Space-Restorer.
"""

from __future__ import annotations

from collections import Counter

import rapidfuzz.distance.Levenshtein as lev
import rapidfuzz.fuzz as fuzz

from subtitles_extractor.application.services.cjk_utils import is_predominantly_cjk
from subtitles_extractor.application.services.outlier_detection import mad_score
from subtitles_extractor.application.services.subtitle_pipeline.constants import (
    CJK_CRITICAL_REVERSAL_KEYWORDS,
    WHITESPACE_RUN_REGEX,
)
from subtitles_extractor.application.services.subtitle_pipeline.text_correction import (
    clean_edge_noise,
    correct_hallucination_typos,
    restore_dropped_yi_prefix,
)
from subtitles_extractor.application.services.text_similarity import (
    text_similarity,
)
from subtitles_extractor.domain.entities.ocr_frame_result import OcrFrameResult

# Trọng số multiplier cho anchor trong voting matrix.
_ANCHOR_WEIGHT_MULTIPLIER: float = 2.5
# Ngưỡng frequency để short-circuit majority.
_MAJORITY_SHORT_CIRCUIT_RATIO: float = 0.45
# Ngưỡng evidence cho insertion (so với tổng confidence pool).
_INSERTION_ACCEPTANCE_RATIO: float = 0.50

# ---------------------------------------------------------------------------
# Build-level frame cache — loại bỏ O(N²) recomputation trong greedy loop.
# Khoá: id(frame_result) — an toàn vì frame là frozen/immutable object.
# Xoá bằng clear_frame_build_cache() ở đầu mỗi lần SubtitleBuilder.build().
# ---------------------------------------------------------------------------
_build_frame_text_cache: dict[int, str] = {}
_build_frame_conf_cache: dict[int, float] = {}


def clear_frame_build_cache() -> None:
    """Xoá cache frame giữa các lần build độc lập.

    Gọi từ :meth:`SubtitleBuilder.build` trước mỗi lần xây dựng mới để
    tránh memory leak và dữ liệu cũ giữa các video khác nhau.
    """
    _build_frame_text_cache.clear()
    _build_frame_conf_cache.clear()


def prewarm_frame_cache(
    frames: list[OcrFrameResult],
    y_tolerance_ratio: float,
    y_tolerance_min_px: float,
) -> None:
    """Tính trước toàn bộ joined_text và mean_confidence cho danh sách frame.

    Một lần O(N) tính toán upfront thay vì O(N²) tính lại trong hot loop.
    Cải thiện hiệu năng 5-15× cho video dài (10k+ frame).

    Args:
        frames: Danh sách :class:`OcrFrameResult` sẽ xử lý.
        y_tolerance_ratio: Tỷ lệ tolerance cho Y-clustering.
        y_tolerance_min_px: Sàn pixel tolerance cho Y-clustering.
    """
    for frame in frames:
        fid = id(frame)
        if fid not in _build_frame_text_cache:
            _build_frame_text_cache[fid] = frame.get_joined_text(
                y_tolerance_ratio=y_tolerance_ratio,
                y_tolerance_min_px=y_tolerance_min_px,
            )
        if fid not in _build_frame_conf_cache:
            _build_frame_conf_cache[fid] = float(frame.mean_confidence)


def extract_joined_text_from_frame(
    frame_result: OcrFrameResult,
    y_tolerance_ratio: float,
    y_tolerance_min_px: float,
) -> str:
    """Wrapper cho :meth:`OcrFrameResult.get_joined_text` với build-level cache.

    Lần đầu truy cập tính toán và lưu vào ``_build_frame_text_cache``.
    Mọi lần gọi tiếp theo trong cùng một lần build là O(1) dict lookup —
    loại bỏ bottleneck O(N²) trong greedy grouping loop (mỗi khi ROVER vote
    chạy lại trên toàn bộ frame trong group).

    Args:
        frame_result: Frame OCR.
        y_tolerance_ratio: Tỷ lệ height cho Y-clustering.
        y_tolerance_min_px: Sàn pixel cho Y-clustering.

    Returns:
        Text đã join các box theo line (\\n giữa các line).
    """
    fid = id(frame_result)
    cached = _build_frame_text_cache.get(fid)
    if cached is not None:
        return cached
    result = frame_result.get_joined_text(
        y_tolerance_ratio=y_tolerance_ratio,
        y_tolerance_min_px=y_tolerance_min_px,
    )
    _build_frame_text_cache[fid] = result
    return result


def get_frame_confidence_cached(frame_result: OcrFrameResult) -> float:
    """Trả về mean_confidence của frame với build-level cache.

    Args:
        frame_result: Frame OCR.

    Returns:
        Mean confidence ∈ [0.0, 1.0].
    """
    fid = id(frame_result)
    cached = _build_frame_conf_cache.get(fid)
    if cached is not None:
        return cached
    conf = float(frame_result.mean_confidence)
    _build_frame_conf_cache[fid] = conf
    return conf


def select_anchor_text(texts_with_confidences: list[tuple[str, float]]) -> str:
    """Chọn anchor text — text tốt nhất làm khung xương cho ROVER vote.

    Composite score = 0.6*frequency_ratio + 0.4*confidence - 0.15*z_score_length.
    Z-score length giúp phạt các text có độ dài lệch mạnh so với median
    (tránh chọn outlier dài/ngắn bất thường làm anchor).

    Args:
        texts_with_confidences: Danh sách ``(text, confidence)`` đầy đủ.

    Returns:
        Anchor text được chọn. Rỗng nếu input rỗng.
    """
    if not texts_with_confidences:
        return ""
    if len(texts_with_confidences) == 1:
        return texts_with_confidences[0][0]

    text_frequencies: Counter[str] = Counter(
        text for text, _ in texts_with_confidences
    )
    total_unique_texts = len(texts_with_confidences)
    measured_lengths = [float(len(text)) for text, _ in texts_with_confidences]

    def _calculate_composite_score(text_conf_tuple: tuple[str, float]) -> float:
        text_value, confidence_value = text_conf_tuple
        frequency_ratio = text_frequencies[text_value] / total_unique_texts
        z_score_length = mad_score(measured_lengths, float(len(text_value)))
        return (frequency_ratio * 0.6) + (confidence_value * 0.4) - (0.15 * z_score_length)

    sorted_by_composite_score = sorted(
        texts_with_confidences, key=_calculate_composite_score, reverse=True
    )
    return sorted_by_composite_score[0][0]


def vote_best_text_rover(
    target_frames: list[OcrFrameResult],
    y_tolerance_ratio: float,
    y_tolerance_min_px: float,
) -> str:
    """Bỏ phiếu ROVER trên các candidate text trong cùng một group.

    Args:
        target_frames: Danh sách frame trong cùng group.
        y_tolerance_ratio: Tỷ lệ Y-clustering tolerance.
        y_tolerance_min_px: Sàn pixel Y-clustering tolerance.

    Returns:
        Text canonical đã được vote và post-process (Yi-restore + Space-restore + edge cleanup).
    """
    valid_frames = [
        frame
        for frame in target_frames
        if extract_joined_text_from_frame(
            frame, y_tolerance_ratio, y_tolerance_min_px
        ).strip()
    ]
    if not valid_frames:
        return ""

    texts_with_confidences: list[tuple[str, float]] = [
        (
            extract_joined_text_from_frame(frame, y_tolerance_ratio, y_tolerance_min_px),
            get_frame_confidence_cached(frame),
        )
        for frame in valid_frames
    ]

    if len(texts_with_confidences) == 1:
        return clean_edge_noise(texts_with_confidences[0][0])

    text_frequencies: Counter[str] = Counter(
        text for text, _ in texts_with_confidences
    )
    total_valid_frames = len(texts_with_confidences)

    most_common_text, most_common_frequency = text_frequencies.most_common(1)[0]
    if most_common_frequency / total_valid_frames >= _MAJORITY_SHORT_CIRCUIT_RATIO:
        voted_text = most_common_text
        voted_text = correct_hallucination_typos(voted_text)
        voted_text = restore_dropped_yi_prefix(voted_text, texts_with_confidences)
        return clean_edge_noise(voted_text)

    anchor_sequence = select_anchor_text(texts_with_confidences)

    if is_predominantly_cjk(anchor_sequence):
        anchor_sequence = correct_hallucination_typos(anchor_sequence)
        anchor_sequence = restore_dropped_yi_prefix(
            anchor_sequence, texts_with_confidences
        )
        return clean_edge_noise(anchor_sequence)

    # ── Full ROVER voting cho Latin/mixed text ──
    texts_with_confidences.sort(
        key=lambda item: (item[0] == anchor_sequence, item[1]), reverse=True
    )

    anchor_confidence_score = next(
        conf for text, conf in texts_with_confidences if text == anchor_sequence
    )

    voting_matrix: list[Counter[str]] = [
        Counter({char: anchor_confidence_score * _ANCHOR_WEIGHT_MULTIPLIER})
        for char in anchor_sequence
    ]
    insertion_matrix: list[Counter[str]] = [
        Counter() for _ in range(len(anchor_sequence) + 1)
    ]

    for comparison_text, current_confidence_score in texts_with_confidences[1:]:
        for operation_code in lev.opcodes(anchor_sequence, comparison_text):
            tag_name = operation_code.tag
            src_start_idx, src_end_idx = operation_code.src_start, operation_code.src_end
            dest_start_idx, dest_end_idx = (
                operation_code.dest_start,
                operation_code.dest_end,
            )

            if tag_name == "equal":
                for anchor_index in range(src_start_idx, src_end_idx):
                    voting_matrix[anchor_index][
                        anchor_sequence[anchor_index]
                    ] += current_confidence_score
            elif tag_name == "replace":
                for anchor_index, comparison_index in zip(
                    range(src_start_idx, src_end_idx),
                    range(dest_start_idx, dest_end_idx),
                    strict=False,
                ):
                    character_to_vote = comparison_text[comparison_index]
                    # [v3.4 BUG FIX]: Dùng biến local để không mutate
                    # ``current_confidence_score`` của outer loop. Trước đây
                    # ``current_confidence_score *= 0.2`` áp dụng vĩnh viễn
                    # cho mọi opcode tiếp theo của cùng candidate ⇒ skew
                    # ROVER vote khi text candidate có ký tự space lệch.
                    vote_weight = current_confidence_score
                    if (
                        character_to_vote == " "
                        and anchor_sequence[anchor_index] != " "
                    ):
                        vote_weight *= 0.2
                    voting_matrix[anchor_index][
                        character_to_vote
                    ] += vote_weight
            elif tag_name == "insert":
                inserted_segment = comparison_text[dest_start_idx:dest_end_idx]
                # [v3.4 BUG FIX]: Tương tự — biến local cho insertion weight.
                insertion_weight = current_confidence_score
                if not inserted_segment.isspace():
                    is_boundary = (
                        src_start_idx == 0
                        or anchor_sequence[src_start_idx - 1].isspace()
                        or src_start_idx == len(anchor_sequence)
                        or anchor_sequence[src_start_idx].isspace()
                    )
                    if not is_boundary:
                        insertion_weight *= 0.3
                insertion_matrix[src_start_idx][
                    inserted_segment
                ] += insertion_weight

    total_confidence_pool = sum(score for _, score in texts_with_confidences)
    insertion_acceptance_threshold = total_confidence_pool * _INSERTION_ACCEPTANCE_RATIO

    reconstructed_characters: list[str] = []
    for sequence_index in range(len(anchor_sequence)):
        if insertion_matrix[sequence_index]:
            best_insertion_string, best_insertion_score = insertion_matrix[
                sequence_index
            ].most_common(1)[0]
            if best_insertion_score > insertion_acceptance_threshold:
                reconstructed_characters.append(best_insertion_string)
        best_character, _ = voting_matrix[sequence_index].most_common(1)[0]
        reconstructed_characters.append(best_character)

    if insertion_matrix[-1]:
        final_insertion_string, final_insertion_score = insertion_matrix[-1].most_common(
            1
        )[0]
        if final_insertion_score > insertion_acceptance_threshold:
            reconstructed_characters.append(final_insertion_string)

    consensus_result_string = "".join(reconstructed_characters)
    consensus_result_string = WHITESPACE_RUN_REGEX.sub(" ", consensus_result_string)
    # v2.9+: Áp dụng typo correction trên VOTED TEXT — bắt lỗi OCR xuất hiện khi
    # các box riêng lẻ được ghép lại (vd '府开上住' từ box '府' + '开上住').
    # Box-level correction trong pre_filter không bắt được cross-box pattern này.
    consensus_result_string = correct_hallucination_typos(consensus_result_string)
    consensus_result_string = restore_dropped_yi_prefix(
        consensus_result_string, texts_with_confidences
    )
    return clean_edge_noise(consensus_result_string)


# ---------------------------------------------------------------------------
# Similarity utilities
# ---------------------------------------------------------------------------


def is_distinct_cjk_utterance(
    text_alpha: str,
    text_beta: str,
    conf_alpha: float,
    conf_beta: float,
    time_gap_seconds: float = 0.0,
) -> bool:
    """Phát hiện 2 câu CJK là phát ngôn riêng biệt (không phải variant OCR).

    Có 2 trường hợp distinct:
        1. **Critical reversal**: max_length <= 6, edit_ops <= 2, ký tự
           khác biệt nằm trong :data:`CJK_CRITICAL_REVERSAL_KEYWORDS`
           (vd 不/没/是), VÀ confidence cả 2 đều >= 0.93 (loại OCR error).
        2. **Prefix/suffix shorter contains in longer** + gap >= 0.15s +
           conf cả 2 >= 0.80: câu ngắn nhập, câu dài là tiếp diễn.

    Args:
        text_alpha: Text 1.
        text_beta: Text 2.
        conf_alpha: Confidence text 1.
        conf_beta: Confidence text 2.
        time_gap_seconds: Khoảng cách thời gian.

    Returns:
        ``True`` nếu là phát ngôn riêng biệt (không nên merge).
    """
    if not text_alpha or not text_beta or text_alpha == text_beta:
        return False
    if not (is_predominantly_cjk(text_alpha) and is_predominantly_cjk(text_beta)):
        return False

    length_alpha, length_beta = len(text_alpha), len(text_beta)
    max_length = max(length_alpha, length_beta)

    confidence_supports_reversal = min(conf_alpha, conf_beta) >= 0.93

    # ── v3.6+: Prefix-drop OCR guard ──
    # Nếu một text là hậu tố / tiền tố của text kia (|len_diff| == 1) VÀ
    # khoảng cách thời gian nhỏ (< 0.15s), đây rất nhiều khả năng là OCR
    # drop 1 ký tự đầu/cuối (vd '不就可以实现' → '就可以实现') chứ không phải
    # 2 câu thoại riêng biệt. Trả False để allow greedy/Viterbi tiếp tục gộp.
    # Không kiểm tra nếu gap >= 0.15s (dùng cơ chế endswith check bên dưới).
    if length_alpha != length_beta and abs(length_alpha - length_beta) == 1:
        _shorter = text_alpha if length_alpha < length_beta else text_beta
        _longer = text_beta if length_alpha < length_beta else text_alpha
        if (
            _longer.endswith(_shorter) or _longer.startswith(_shorter)
        ) and time_gap_seconds < 0.15:
            return False

    if max_length <= 6 and confidence_supports_reversal:
        edit_operations = lev.editops(text_alpha, text_beta)
        if 0 < len(edit_operations) <= 2:
            for operation in edit_operations:
                if (
                    operation.tag in ("replace", "delete")
                    and text_alpha[operation.src_pos] in CJK_CRITICAL_REVERSAL_KEYWORDS
                ):
                    return True
                if (
                    operation.tag == "insert"
                    and text_beta[operation.dest_pos] in CJK_CRITICAL_REVERSAL_KEYWORDS
                ):
                    return True

    if length_alpha != length_beta:
        shorter_text = text_alpha if length_alpha < length_beta else text_beta
        longer_text = text_beta if length_alpha < length_beta else text_alpha

        if longer_text.startswith(shorter_text) or longer_text.endswith(shorter_text):
            if time_gap_seconds >= 0.15:
                if conf_alpha >= 0.80 and conf_beta >= 0.80:
                    return True

    return False


def calculate_effective_similarity(
    text_alpha: str,
    text_beta: str,
    conf_alpha: float,
    conf_beta: float,
    time_gap_seconds: float = 0.0,
) -> float:
    """Tính similarity hiệu dụng có xét đến đảo nghĩa CJK và shortcut CJK.

    Wrapper quanh :func:`text_similarity`:
        1. Nếu phát hiện 2 text là CJK distinct utterance (đảo nghĩa) thì
           ép similarity = 0.0.
        2. Áp dụng 3 shortcut cho cặp CJK-CJK để gộp các OCR error nhẹ:
            * len 2-3, edit distance = 1 → 1.0 (OCR confuse 1 ký tự).
            * cùng length >= 4, fuzz ratio >= 74 → 1.0.
            * |len_a - len_b| <= 2, max >= 4, fuzz ratio >= 78 → 1.0.
        3. Cuối cùng fallback :func:`text_similarity`.

    Args:
        text_alpha: Text 1.
        text_beta: Text 2.
        conf_alpha: Confidence text 1.
        conf_beta: Confidence text 2.
        time_gap_seconds: Khoảng cách thời gian.

    Returns:
        Similarity ∈ [0.0, 1.0].
    """
    if is_distinct_cjk_utterance(
        text_alpha, text_beta, conf_alpha, conf_beta, time_gap_seconds
    ):
        return 0.0

    is_alpha_cjk = is_predominantly_cjk(text_alpha)
    is_beta_cjk = is_predominantly_cjk(text_beta)

    if is_alpha_cjk and is_beta_cjk:
        length_alpha, length_beta = len(text_alpha), len(text_beta)

        # Shortcut 1: len 2-3, distance=1 → 1.0 (OCR error nhẹ).
        # KHÔNG áp dụng cho len=1 vì '三' vs '二' (lev=1) là hai chữ khác hoàn toàn.
        if 2 <= length_alpha == length_beta <= 3:
            if lev.distance(text_alpha, text_beta) == 1:
                return 1.0

        # Shortcut 2: cùng length >= 4, fuzz ratio >= 74 → 1.0.
        if length_alpha == length_beta and length_alpha >= 4:
            if fuzz.ratio(text_alpha, text_beta) >= 74.0:
                return 1.0

        # Shortcut 3: |len_a - len_b| <= 2, max >= 4, fuzz ratio >= 78 → 1.0.
        elif abs(length_alpha - length_beta) <= 2 and max(length_alpha, length_beta) >= 4:
            if fuzz.ratio(text_alpha, text_beta) >= 78.0:
                return 1.0

    return text_similarity(text_alpha, text_beta)


def evaluate_multi_line_similarity(
    text_alpha: str,
    text_beta: str,
    conf_alpha: float,
    conf_beta: float,
    required_threshold: float,
    time_gap_seconds: float = 0.0,
) -> bool:
    """Đánh giá similarity multi-line: tách dòng và so sánh từng cặp.

    Hai text được coi là tương đồng nếu có cùng số line và similarity
    trung bình của các cặp line >= ngưỡng.

    Args:
        text_alpha: Text 1.
        text_beta: Text 2.
        conf_alpha: Confidence text 1.
        conf_beta: Confidence text 2.
        required_threshold: Ngưỡng similarity trung bình cần đạt.
        time_gap_seconds: Khoảng cách thời gian.

    Returns:
        ``True`` nếu thỏa điều kiện multi-line similarity.
    """
    split_lines_alpha = [line.strip() for line in text_alpha.splitlines() if line.strip()]
    split_lines_beta = [line.strip() for line in text_beta.splitlines() if line.strip()]

    if (
        not split_lines_alpha
        or not split_lines_beta
        or len(split_lines_alpha) != len(split_lines_beta)
    ):
        return False

    total_line_pairs = len(split_lines_alpha)
    computed_average_similarity = (
        sum(
            calculate_effective_similarity(
                split_lines_alpha[pair_index],
                split_lines_beta[pair_index],
                conf_alpha,
                conf_beta,
                time_gap_seconds,
            )
            for pair_index in range(total_line_pairs)
        )
        / total_line_pairs
    )
    return computed_average_similarity >= required_threshold


__all__ = [
    "calculate_effective_similarity",
    "clear_frame_build_cache",
    "evaluate_multi_line_similarity",
    "extract_joined_text_from_frame",
    "get_frame_confidence_cached",
    "is_distinct_cjk_utterance",
    "prewarm_frame_cache",
    "select_anchor_text",
    "vote_best_text_rover",
]
