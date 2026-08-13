"""Hậu xử lý các :class:`FrameGroup` và :class:`SubtitleEvent`.

Pipeline post-processing 6 stage:

1. :func:`merge_adjacent_duplicates` — gộp 2 group liền kề có text tương đồng.
2. :func:`filter_echo_trail_groups` — loại "echo" OCR (group chớp nhoáng
   conf thấp giống neighbor conf cao).
3. :func:`convert_groups_to_events` — chuyển FrameGroup → SubtitleEvent.
4. :func:`filter_short_duration_events` — hấp thụ flicker (events ngắn).
5. :func:`filter_short_text_events` — lọc events có text rác/ngắn.
6. :func:`post_merge_duplicates` — merge lần cuối sau khi đã filter rác.
7. :func:`filter_echo_trail_events` — lớp echo filter cuối ở cấp event.
"""

from __future__ import annotations

import bisect

import rapidfuzz.fuzz as fuzz
from loguru import logger

from subtitles_extractor.application.dtos.extract_subtitles_dto import (
    SubtitleBuilderConfig,
)
from subtitles_extractor.application.services.cjk_utils import (
    adaptive_min_text_chars,
    cjk_char_count,
    contains_cjk,
    is_predominantly_cjk,
)
from subtitles_extractor.application.services.flicker_absorber import absorb_flickers
from subtitles_extractor.application.services.subtitle_pipeline.box_filters import (
    is_latin_gibberish,
)
from subtitles_extractor.application.services.subtitle_pipeline.constants import (
    LATIN_VALID_SHORT_TOKENS,
    NON_WORD_CHARACTER_REGEX,
)
from subtitles_extractor.application.services.subtitle_pipeline.frame_group import (
    FrameGroup,
    normalize_cjk_punctuation,
)
from subtitles_extractor.application.services.subtitle_pipeline.voting import (
    calculate_effective_similarity,
    evaluate_multi_line_similarity,
)
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.value_objects.confidence import Confidence
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval

# ---------------------------------------------------------------------------
# Pass 1: Merge adjacent duplicate groups
# ---------------------------------------------------------------------------


#: [Stable Variant Split] Một "khối ổn định" cần tối thiểu bấy nhiêu frame...
_VARIANT_MIN_FRAMES: int = 6
#: ...và kéo dài tối thiểu bấy nhiêu giây.
_VARIANT_MIN_SPAN_SEC: float = 0.30
#: Run nhiễu (<= số frame này) được hấp thụ vào khối lớn đứng trước.
_VARIANT_NOISE_RUN_FRAMES: int = 2


def _normalize_for_variant(text: str) -> str:
    """Chuẩn hoá text để so khối: bỏ whitespace (OCR dao động space vô nghĩa)."""
    return "".join(text.split())


def _is_cjk_char(ch: str) -> bool:
    cp = ord(ch)
    return (
        0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF
        or 0xF900 <= cp <= 0xFAFF or 0x3040 <= cp <= 0x30FF
        or 0xAC00 <= cp <= 0xD7AF
    )


def _should_split_boundary(text_before: str, text_after: str) -> bool:
    """Quyết định ranh giới 2 khối ổn định có phải 2 CÂU THẬT khác nhau.

    Quy tắc (rút từ dữ liệu thực):
      * Chênh ở CUỐI (prefix quan hệ) + delta toàn CJK >= 1 ký tự → 2 câu
        (progressive utterance: ``舒服吗``→``舒服``; ``不过``→``不过什么``).
      * Chênh ở ĐẦU (suffix quan hệ) → chỉ là 2 câu nếu delta toàn CJK và
        >= 2 ký tự; delta 1 ký tự đầu (vd ``一``) là OCR mất nét mảnh.
      * Delta chứa Latin → biến thể watermark dính (``BOR``/``CIACA``), KHÔNG tách.
      * Không prefix/suffix: similarity < 0.5 → 2 câu khác hẳn.
    """
    a = _normalize_for_variant(text_before)
    b = _normalize_for_variant(text_after)
    if not a or not b or a == b:
        return False

    shorter, longer = (a, b) if len(a) < len(b) else (b, a)
    if longer.startswith(shorter):
        delta = longer[len(shorter):]
        return bool(delta) and all(_is_cjk_char(c) for c in delta)
    if longer.endswith(shorter):
        delta = longer[: len(longer) - len(shorter)]
        return len(delta) >= 2 and all(_is_cjk_char(c) for c in delta)

    return fuzz.ratio(a, b) / 100.0 < 0.5


#: [Stable Variant Split] Một "khối ổn định" cần tối thiểu bấy nhiêu frame...
_VARIANT_MIN_FRAMES: int = 6
#: ...và kéo dài tối thiểu bấy nhiêu giây.
_VARIANT_MIN_SPAN_SEC: float = 0.30
#: Run nhiễu (<= số frame này) được hấp thụ vào khối lớn đứng trước.
_VARIANT_NOISE_RUN_FRAMES: int = 2


def split_stable_variant_groups(
    source_groups: list[FrameGroup],
) -> list[FrameGroup]:
    """Tách group chứa NHIỀU khối text ổn định là các CÂU THẬT khác nhau.

    Failure mode khắc phục: "progressive utterance" — 2+ câu thoại thật liên
    tiếp na ná nhau (``舒服吗``→``舒服``; ``不过``→``不过什么``) bị greedy merge
    thành 1 group. Đặc trưng phân biệt với biến thể lỗi OCR: mỗi câu thật chiếm
    một KHỐI frame thuần nhất dài (>= 6 frame, >= 0.30s); lỗi OCR chỉ là run
    1-2 frame hoặc biến thể space/Latin/mất-1-ký-tự-đầu (xem
    :func:`_should_split_boundary`).

    Args:
        source_groups: Groups sau merge/echo-trail.

    Returns:
        Danh sách groups; group đa-câu đã được tách tại ranh giới hợp lệ.
    """
    result_groups: list[FrameGroup] = []
    for group in source_groups:
        member_texts = getattr(group, "member_texts", None) or []
        if len(member_texts) < 2 * _VARIANT_MIN_FRAMES:
            result_groups.append(group)
            continue

        # Nén run-length theo text đã chuẩn hoá space.
        runs: list[dict] = []
        for ts, text in member_texts:
            key = _normalize_for_variant(text)
            if runs and runs[-1]["key"] == key:
                runs[-1]["end"] = ts
                runs[-1]["count"] += 1
            else:
                runs.append({"key": key, "text": text, "start": ts, "end": ts, "count": 1})

        # Hấp thụ run nhiễu (1-2 frame) vào run đứng trước.
        absorbed: list[dict] = []
        for run in runs:
            if absorbed and (
                run["count"] <= _VARIANT_NOISE_RUN_FRAMES
                or run["key"] == absorbed[-1]["key"]
            ):
                absorbed[-1]["end"] = run["end"]
                absorbed[-1]["count"] += run["count"]
            else:
                absorbed.append(dict(run))

        stable_runs = [
            r for r in absorbed
            if r["count"] >= _VARIANT_MIN_FRAMES
            and (r["end"] - r["start"]) >= _VARIANT_MIN_SPAN_SEC
            and len(r["key"]) >= 2
        ]
        if len(stable_runs) < 2:
            result_groups.append(group)
            continue

        # Chống OCR dao động khối lớn: không tách nếu key lặp lại (A-B-A).
        keys = [r["key"] for r in stable_runs]
        if len(set(keys)) != len(keys):
            result_groups.append(group)
            continue

        # Gom các khối thành segment; chỉ cắt tại ranh giới "2 câu thật".
        segments: list[list[dict]] = [[stable_runs[0]]]
        for prev_run, next_run in zip(stable_runs, stable_runs[1:]):
            if _should_split_boundary(prev_run["text"], next_run["text"]):
                segments.append([next_run])
            else:
                segments[-1].append(next_run)

        if len(segments) < 2:
            result_groups.append(group)
            continue

        per_frame_confidence = (
            group.accumulated_confidence / group.total_frames_count
            if group.total_frames_count > 0 else 0.0
        )
        for segment in segments:
            best_run = max(segment, key=lambda r: r["count"])
            seg_start = segment[0]["start"]
            seg_end = segment[-1]["end"]
            seg_frames = sum(r["count"] for r in segment)
            result_groups.append(
                FrameGroup(
                    reconstructed_text=best_run["text"],
                    start_timestamp_sec=seg_start,
                    end_timestamp_sec=seg_end,
                    accumulated_confidence=per_frame_confidence * seg_frames,
                    total_frames_count=seg_frames,
                    primary_center_position=group.primary_center_position,
                    aggregated_bounding_box=group.aggregated_bounding_box,
                    member_texts=[
                        (ts, tx) for ts, tx in member_texts
                        if seg_start <= ts <= seg_end
                    ],
                )
            )
        logger.debug(
            "Stable-variant split: tách group {!r} thành {} câu.",
            group.reconstructed_text, len(segments),
        )
    return result_groups


def merge_adjacent_duplicates(
    source_frame_groups: list[FrameGroup],
    config: SubtitleBuilderConfig,
) -> list[FrameGroup]:
    """Gộp 2 group liền kề có text tương đồng (sau greedy/Viterbi grouping).

    Lý do tồn tại: thuật toán greedy/Viterbi có thể chia 1 câu thành 2
    group nếu OCR có biến động giữa chừng (vd 1-2 frame mờ). Pass này
    quét lại và gộp.

    Args:
        source_frame_groups: Danh sách FrameGroup từ grouper.
        config: Cấu hình builder.

    Returns:
        Danh sách FrameGroup đã gộp.
    """
    if not source_frame_groups:
        return source_frame_groups

    successfully_merged_results: list[FrameGroup] = [source_frame_groups[0]]
    relaxed_similarity_threshold = config.similarity_threshold
    multi_line_similarity_threshold = config.line_similarity_threshold
    maximum_allowed_gap_duration = config.merge_gap_sec

    for current_frame_group in source_frame_groups[1:]:
        previous_frame_group = successfully_merged_results[-1]
        time_gap_duration_seconds = (
            current_frame_group.start_timestamp_sec
            - previous_frame_group.end_timestamp_sec
        )

        if time_gap_duration_seconds > maximum_allowed_gap_duration:
            successfully_merged_results.append(current_frame_group)
            continue

        normalized_previous_text = normalize_cjk_punctuation(
            previous_frame_group.reconstructed_text
        )
        normalized_current_text = normalize_cjk_punctuation(
            current_frame_group.reconstructed_text
        )

        # [Repeat Utterance Guard] Hai group GIỐNG HỆT text, mỗi bên ổn định
        # >= 0.30s, cách nhau >= 0.35s frame trống → câu lặp, KHÔNG gộp.
        if (
            normalized_previous_text == normalized_current_text
            and time_gap_duration_seconds >= 0.35
            and (previous_frame_group.end_timestamp_sec
                 - previous_frame_group.start_timestamp_sec) >= 0.30
            and (current_frame_group.end_timestamp_sec
                 - current_frame_group.start_timestamp_sec) >= 0.30
        ):
            successfully_merged_results.append(current_frame_group)
            continue

        calculated_prev_confidence = previous_frame_group.calculate_mean_confidence()
        calculated_curr_confidence = current_frame_group.calculate_mean_confidence()

        evaluated_overall_similarity = calculate_effective_similarity(
            normalized_previous_text,
            normalized_current_text,
            calculated_prev_confidence,
            calculated_curr_confidence,
            time_gap_duration_seconds,
        )

        is_lines_substantially_similar = False
        if multi_line_similarity_threshold > 0.0:
            is_lines_substantially_similar = evaluate_multi_line_similarity(
                normalized_previous_text,
                normalized_current_text,
                calculated_prev_confidence,
                calculated_curr_confidence,
                multi_line_similarity_threshold,
                time_gap_duration_seconds,
            )

        if (
            evaluated_overall_similarity >= relaxed_similarity_threshold
            or is_lines_substantially_similar
        ):
            previous_group_score = previous_frame_group.total_score
            current_group_score = current_frame_group.total_score

            previous_frame_group.end_timestamp_sec = current_frame_group.end_timestamp_sec
            # [Stable Variant Split] Nối chuỗi text per-frame để pass tách sau dùng.
            previous_frame_group.member_texts.extend(current_frame_group.member_texts)
            previous_frame_group.accumulated_confidence += (
                current_frame_group.accumulated_confidence
            )
            previous_frame_group.total_frames_count += (
                current_frame_group.total_frames_count
            )

            if current_group_score > previous_group_score:
                previous_frame_group.reconstructed_text = (
                    current_frame_group.reconstructed_text
                )
            continue

        successfully_merged_results.append(current_frame_group)

    return successfully_merged_results


# ---------------------------------------------------------------------------
# Pass 2: Echo trail filter (cấp FrameGroup)
# ---------------------------------------------------------------------------


def filter_echo_trail_groups(
    source_frame_groups: list[FrameGroup],
    config: SubtitleBuilderConfig,
) -> list[FrameGroup]:
    """Loại bỏ "echo trail" — rác OCR còn lại sau khi text thật fade out.

    Hiện tượng: sau câu '吩咐下去' (conf 0.9+), OCR engine vẫn output các
    text "biến thái" trong 1-3 khung kế tiếp: '小分休下去', '分时子去',
    … (conf 0.5-0.6). Chúng không đủ giống để merge nhưng cũng không phải
    câu thực sự — chỉ là dư âm.

    Tiêu chí echo:
        * Group có ``frame_count <= 3``.
        * ``mean_confidence < 0.65``.
        * Group LIỀN KỀ có ``mean_confidence >= 0.80``.
        * Raw fuzz ratio giữa hai group >= 0.20.

    Args:
        source_frame_groups: Danh sách FrameGroup sau merge_adjacent_duplicates.
        config: Cấu hình builder.

    Returns:
        Danh sách FrameGroup mới, đã loại bỏ echo trail.
    """
    if len(source_frame_groups) <= 1:
        return source_frame_groups

    max_neighbor_gap_seconds = config.merge_gap_sec * 1.5
    minimum_echo_ratio = 0.20
    kept_groups: list[FrameGroup] = []
    dropped_echo_count = 0

    for group_index, current_group in enumerate(source_frame_groups):
        current_mean_conf = current_group.calculate_mean_confidence()
        is_echo_candidate = (
            current_group.total_frames_count <= 3 and current_mean_conf < 0.65
        )

        if not is_echo_candidate:
            kept_groups.append(current_group)
            continue

        current_normalized_text = normalize_cjk_punctuation(
            current_group.reconstructed_text
        )
        if not current_normalized_text:
            kept_groups.append(current_group)
            continue

        absorbed_by_previous = False
        absorbed_by_next = False

        if kept_groups:
            prev_group = kept_groups[-1]
            gap_to_prev = (
                current_group.start_timestamp_sec - prev_group.end_timestamp_sec
            )
            prev_mean_conf = prev_group.calculate_mean_confidence()
            if (
                0.0 <= gap_to_prev <= max_neighbor_gap_seconds
                and prev_mean_conf >= 0.80
            ):
                prev_normalized_text = normalize_cjk_punctuation(
                    prev_group.reconstructed_text
                )
                if prev_normalized_text:
                    echo_ratio = (
                        fuzz.ratio(prev_normalized_text, current_normalized_text) / 100.0
                    )
                    if echo_ratio >= minimum_echo_ratio:
                        prev_group.end_timestamp_sec = max(
                            prev_group.end_timestamp_sec,
                            current_group.end_timestamp_sec,
                        )
                        absorbed_by_previous = True

        if not absorbed_by_previous and group_index + 1 < len(source_frame_groups):
            next_group = source_frame_groups[group_index + 1]
            gap_to_next = (
                next_group.start_timestamp_sec - current_group.end_timestamp_sec
            )
            next_mean_conf = next_group.calculate_mean_confidence()
            if (
                0.0 <= gap_to_next <= max_neighbor_gap_seconds
                and next_mean_conf >= 0.80
            ):
                next_normalized_text = normalize_cjk_punctuation(
                    next_group.reconstructed_text
                )
                if next_normalized_text:
                    echo_ratio = (
                        fuzz.ratio(next_normalized_text, current_normalized_text) / 100.0
                    )
                    if echo_ratio >= minimum_echo_ratio:
                        absorbed_by_next = True

        if absorbed_by_previous or absorbed_by_next:
            dropped_echo_count += 1
            continue

        kept_groups.append(current_group)

    if dropped_echo_count:
        logger.debug("Echo trail filter đã loại {} group dư âm OCR.", dropped_echo_count)

    return kept_groups


# ---------------------------------------------------------------------------
# Pass 3: Convert groups → events
# ---------------------------------------------------------------------------


def convert_groups_to_events(
    assembled_frame_groups: list[FrameGroup],
    config: SubtitleBuilderConfig,
) -> list[SubtitleEvent]:
    """Chuyển FrameGroup → SubtitleEvent với temporal padding hợp lý.

    Padding cuối thêm ``temporal_padding_sec`` (giúp xem có chút overlap
    với câu kế nếu cần) nhưng KHÔNG vượt qua start của event kế. Đảm bảo
    không có overlap thời gian giữa các event.

    Args:
        assembled_frame_groups: Danh sách FrameGroup đã hậu xử lý.
        config: Cấu hình builder.

    Returns:
        Danh sách SubtitleEvent có index 1-based.
    """
    final_generated_events: list[SubtitleEvent] = []
    configured_padding_seconds = config.temporal_padding_sec
    minimum_gap_buffer_seconds = 0.001

    for group_index, current_group in enumerate(assembled_frame_groups):
        original_start_time = current_group.start_timestamp_sec
        original_end_time = current_group.end_timestamp_sec

        padded_start_time = max(0.0, original_start_time)
        padded_end_time = original_end_time + configured_padding_seconds

        if final_generated_events:
            previous_event_end = final_generated_events[-1].end_sec
            if padded_start_time <= previous_event_end:
                padded_start_time = previous_event_end + minimum_gap_buffer_seconds

        if group_index + 1 < len(assembled_frame_groups):
            next_group_original_start = assembled_frame_groups[
                group_index + 1
            ].start_timestamp_sec
            if padded_end_time >= next_group_original_start:
                padded_end_time = next_group_original_start - minimum_gap_buffer_seconds

        if padded_end_time <= padded_start_time:
            padded_end_time = padded_start_time + minimum_gap_buffer_seconds

        final_generated_events.append(
            SubtitleEvent(
                index=group_index + 1,
                text=current_group.reconstructed_text,
                interval=TimeInterval(
                    start_sec=max(0.0, padded_start_time),
                    end_sec=max(0.001, padded_end_time),
                ),
                confidence=Confidence(current_group.calculate_mean_confidence()),
                frame_count=current_group.total_frames_count,
                position=current_group.primary_center_position,
                bounding_box=current_group.aggregated_bounding_box,
            )
        )
    return final_generated_events


# ---------------------------------------------------------------------------
# Pass 4: Flicker absorption (events quá ngắn được hấp thụ vào láng giềng)
# ---------------------------------------------------------------------------


def filter_short_duration_events(
    subtitles_events_list: list[SubtitleEvent],
    config: SubtitleBuilderConfig,
) -> list[SubtitleEvent]:
    """Hấp thụ flicker events ngắn (thin wrapper quanh :func:`absorb_flickers`).

    Args:
        subtitles_events_list: Danh sách events.
        config: Cấu hình builder.

    Returns:
        Danh sách events sau khi hấp thụ flicker.
    """
    return absorb_flickers(
        events=subtitles_events_list,
        min_duration_sec=config.min_duration_sec,
        similarity_threshold=config.similarity_threshold,
        merge_gap_sec=config.merge_gap_sec,
    )


# ---------------------------------------------------------------------------
# Pass 5: Filter short text events (rác sau khi đã absorb flicker)
# ---------------------------------------------------------------------------


def _is_cjk_edge_fragment(
    current_event: SubtitleEvent,
    event_index: int,
    all_events: list[SubtitleEvent],
) -> bool:
    """Kiểm tra event có phải là edge-fragment của subtitle animated hợp lệ.

    "Edge-fragment" = 1-2 frame đầu/cuối của subtitle xuất hiện dần dần
    (animated text entrance/exit). Cần giữ lại để không mất timing chính xác.

    Điều kiện edge-fragment:
        * Prev hoặc next event trong vòng 0.4s có text chứa/bắt đầu bằng
          text hiện tại (text hiện tại là subset của subtitle hợp lệ).
        * Dùng ``fuzz.partial_ratio`` >= 80 để chịu được OCR noise nhỏ.

    Args:
        current_event: Event đang kiểm tra.
        event_index: Vị trí trong ``all_events``.
        all_events: Danh sách tất cả events (chưa filter ở pass này).

    Returns:
        ``True`` nếu là edge-fragment hợp lệ, ``False`` nếu là rác.
    """
    total = len(all_events)
    curr_text = current_event.text.replace(" ", "")
    _EDGE_GAP_SEC: float = 0.40

    for offset in (-1, 1):
        neighbor_idx = event_index + offset
        if not (0 <= neighbor_idx < total):
            continue
        neighbor = all_events[neighbor_idx]
        gap = (
            current_event.start_sec - neighbor.end_sec
            if offset == -1
            else neighbor.start_sec - current_event.end_sec
        )
        if gap > _EDGE_GAP_SEC:
            continue
        nbr_text = neighbor.text.replace(" ", "")
        # Edge-fragment: text hiện tại là tiền tố/substring của neighbor
        if not nbr_text:
            continue
        if nbr_text.startswith(curr_text) or curr_text in nbr_text:
            return True
        # partial_ratio: tìm sự hiện diện của curr_text trong nbr_text
        if fuzz.partial_ratio(curr_text, nbr_text) >= 80:
            return True

    return False


def filter_short_text_events(
    subtitles_events_list: list[SubtitleEvent],
    config: SubtitleBuilderConfig,
) -> list[SubtitleEvent]:
    """Lọc events có text rác/ngắn theo các tiêu chí frame_count + confidence.

    Phân nhánh theo loại text:

    * **CJK-dominant**:
        - Lai Latin-CJK ngắn (1-2 CJK + >= 2 Latin), conf < 0.85 → drop.
        - CJK 1-3 chars + >= 1 Latin/digit, frame_count <= 2, conf < 0.75 → drop.
        - CJK đơn lẻ (vd '土', '士'): drop trừ khi là tiền tố animated của
          event kế tiếp (gap <= 0.5s, conf >= 0.60).
        - CJK 2 chars + frame_count <= 1 hoặc conf < 0.65 → drop.

    * **Pure Latin/digit**:
        - Latin gibberish detection (vd 'COZA', 'AAR') → drop.
        - <= 2 chars, frame_count <= 2, conf < 0.90 → drop.

    Cuối cùng kiểm tra adaptive_min_text_chars (CJK=1, Latin tuỳ config).

    Args:
        subtitles_events_list: Danh sách events.
        config: Cấu hình builder.

    Returns:
        Danh sách events đã lọc.
    """
    kept_valid_events: list[SubtitleEvent] = []
    total_events_count = len(subtitles_events_list)

    for event_index, current_event in enumerate(subtitles_events_list):
        original_text = current_event.text.strip()
        text_cleaned_for_length_check = NON_WORD_CHARACTER_REGEX.sub(
            "", original_text
        ).replace("_", "")

        if not text_cleaned_for_length_check:
            continue

        recorded_confidence = float(current_event.confidence)
        recorded_frame_count = current_event.frame_count
        total_clean_char_count = len(text_cleaned_for_length_check)

        if contains_cjk(text_cleaned_for_length_check):
            cjk_only_char_count = cjk_char_count(text_cleaned_for_length_check)
            non_cjk_residue_count = total_clean_char_count - cjk_only_char_count

            # CJK garble cực dài conf thấp fc nhỏ — credit screen overlap nhiều
            # dòng text bị OCR đọc chồng lên thành 1 chuỗi rác dài.
            # Quan sát thực nghiệm trên test2: các event 40+ chars CJK conf
            # 0.68-0.75 fc 1-15 đều là rác.
            if (
                cjk_only_char_count >= 25
                and recorded_confidence < 0.80
                and recorded_frame_count <= 15
            ):
                continue

            is_latin_dominant_mixed = (
                cjk_only_char_count <= 2
                and non_cjk_residue_count >= cjk_only_char_count
                and non_cjk_residue_count >= 2
            )
            if is_latin_dominant_mixed and recorded_confidence < 0.85:
                continue

            is_short_cjk_with_latin_noise = (
                1 <= cjk_only_char_count <= 3
                and non_cjk_residue_count >= 1
                and recorded_frame_count <= 2
                and recorded_confidence < 0.75
            )
            if is_short_cjk_with_latin_noise:
                continue

            if cjk_only_char_count == 1 and non_cjk_residue_count == 0:
                # CJK 1 ký tự: kiểm tra animated intro của event kế tiếp.
                is_animated_intro_of_next_event = False
                if event_index + 1 < total_events_count:
                    next_event = subtitles_events_list[event_index + 1]
                    next_text = next_event.text.strip()
                    gap_to_next = next_event.start_sec - current_event.end_sec
                    if (
                        0.0 <= gap_to_next <= 0.5
                        and next_text
                        and next_text[0] == text_cleaned_for_length_check[0]
                        and len(next_text) >= 3
                        and recorded_confidence >= 0.60
                    ):
                        is_animated_intro_of_next_event = True

                if not is_animated_intro_of_next_event:
                    if recorded_frame_count <= 2:
                        # Flicker rác chỉ 1-2 frame: drop bất kể conf.
                        continue
                    # Adaptive threshold theo độ ổn định (frame_count):
                    #   fc >= 5: cluster ổn định >= 0.4s → 0.72 đủ (bắt
                    #            interjection có conf trung bình như '埃'
                    #            xuất hiện 8 frame conf 0.74).
                    #   fc 3-4:  cluster yếu hơn → cần 0.75.
                    # Lý do chọn 0.72: thực nghiệm 11 file cho thấy 0.72 cứu
                    # được '埃' (chinese_vid2 conf 0.745) và '哦/啊/阿' trong
                    # file test, ít gây EXTRA hơn so với 0.68.
                    minimum_required_conf_for_single_cjk = (
                        0.72 if recorded_frame_count >= 5 else 0.75
                    )
                    if recorded_confidence < minimum_required_conf_for_single_cjk:
                        continue
            elif cjk_only_char_count == 2 and non_cjk_residue_count == 0:
                if recorded_frame_count <= 1:
                    continue
                if recorded_confidence < 0.65:
                    continue

            # ── v2.9+: Flash-noise guard — CJK 3-24 chars, fc≤2, conf<0.78 ──
            # Phát hiện thực nghiệm: 155/276 extra trong test3, ~21/60 trong
            # test2 là garbled OCR 1-2 frame từ credits/transition screen.
            if (
                3 <= cjk_only_char_count <= 24
                and non_cjk_residue_count == 0
                and recorded_frame_count <= 2
                and recorded_confidence < 0.78
            ):
                if not _is_cjk_edge_fragment(
                    current_event, event_index, subtitles_events_list
                ):
                    continue

            # ── v2.9+: Mixed-Latin flash guard — unconditional drop ──
            # Latin chars trong OCR result CJK với fc<=2 LUÔN là artifact.
            # Không cần edge-fragment check vì `partial_ratio` có thể
            # false-positive với CJK suffix (vd 'caanfhittnng三千就三千'
            # có partial_ratio=100% với '三千就三千').
            if (
                cjk_only_char_count >= 3
                and non_cjk_residue_count >= 1
                and recorded_frame_count <= 2
                and recorded_confidence < 0.82
            ):
                continue  # Drop vô điều kiện

            # ── v2.9+: Persistent garbled guard — CJK 4-24 chars, fc 3-15 ──
            # Credits/transition text xuất hiện nhiều frame liên tiếp nhưng
            # conf thấp (0.65-0.75). Phân biệt với dialogue bằng edge-fragment
            # check (dialogue hợp lệ luôn có neighbor text tương đồng).
            if (
                4 <= cjk_only_char_count <= 24
                and non_cjk_residue_count == 0
                and 3 <= recorded_frame_count <= 15
                and recorded_confidence < 0.76
                and current_event.duration_sec < 0.80
            ):
                if not _is_cjk_edge_fragment(
                    current_event, event_index, subtitles_events_list
                ):
                    continue
        else:
            if is_latin_gibberish(
                text_cleaned_for_length_check,
                recorded_confidence,
                recorded_frame_count,
            ):
                continue
            if recorded_frame_count <= 2 and total_clean_char_count <= 2:
                if recorded_confidence < 0.90:
                    continue
            # ── v2.9+: Pure non-CJK flash noise — fc≤2, conf<0.80, len≤10 ──
            # Bắt: 'CuC=2', 'CCA', 'BYL2', '000-00-00' v.v.
            # is_latin_gibberish không bắt được vì dựa vào vowel-ratio.
            if (
                recorded_frame_count <= 2
                and recorded_confidence < 0.80
                and total_clean_char_count <= 10
                and text_cleaned_for_length_check.upper()
                not in LATIN_VALID_SHORT_TOKENS
            ):
                continue

        configured_minimum_characters = config.min_text_chars
        if configured_minimum_characters > 0:
            effective_minimum_required = adaptive_min_text_chars(
                text_cleaned_for_length_check,
                latin_min=configured_minimum_characters,
            )
            if total_clean_char_count < effective_minimum_required:
                continue

        # Tạo event mới chỉ với text đã strip — giữ nguyên các trường khác.
        verified_safe_event = SubtitleEvent(
            index=current_event.index,
            text=original_text,
            interval=current_event.interval,
            confidence=current_event.confidence,
            frame_count=current_event.frame_count,
            position=current_event.position,
            bounding_box=current_event.bounding_box,
            uid=current_event.uid,
        )
        kept_valid_events.append(verified_safe_event)

    return kept_valid_events


# ---------------------------------------------------------------------------
# Pass 6: Post-merge duplicates (sau khi đã filter rác)
# ---------------------------------------------------------------------------


def is_superset_within_one_char(text_a: str, text_b: str) -> bool:
    """Kiểm tra 1 text có là siêu chuỗi của cái kia với chỉ 1 ký tự thêm.

    Args:
        text_a: Text candidate 1.
        text_b: Text candidate 2.

    Returns:
        ``True`` nếu cả 2 đều CJK-dominant, length chênh đúng 1, và bỏ 1
        ký tự CJK trong text dài hơn thì khớp text ngắn hơn.
    """
    if not text_a or not text_b:
        return False
    if not is_predominantly_cjk(text_a) or not is_predominantly_cjk(text_b):
        return False
    if abs(len(text_a) - len(text_b)) != 1:
        return False

    longer_text, shorter_text = (
        (text_a, text_b) if len(text_a) > len(text_b) else (text_b, text_a)
    )
    if len(shorter_text) < 2:
        return False

    for skip_position in range(len(longer_text)):
        removed_char = longer_text[skip_position]
        if not (
            "\u4e00" <= removed_char <= "\u9fff"
            or "\u3400" <= removed_char <= "\u4dbf"
        ):
            continue
        stripped_result = longer_text[:skip_position] + longer_text[skip_position + 1 :]
        if stripped_result == shorter_text:
            return True
    return False


def _execute_superset_merge(
    target_result_list: list[SubtitleEvent],
    base_anchor_event: SubtitleEvent,
    event_to_merge_in: SubtitleEvent,
) -> None:
    """Merge cặp superset/subset — giữ text DÀI hơn (OCR drop > add ký tự)."""
    recalculated_end_time = max(base_anchor_event.end_sec, event_to_merge_in.end_sec)
    combined_total_frames = (
        base_anchor_event.frame_count + event_to_merge_in.frame_count
    )
    weighted_confidence_score = (
        float(base_anchor_event.confidence) * base_anchor_event.frame_count
        + float(event_to_merge_in.confidence) * event_to_merge_in.frame_count
    ) / max(1, combined_total_frames)

    longer_text = (
        base_anchor_event.text
        if len(base_anchor_event.text) >= len(event_to_merge_in.text)
        else event_to_merge_in.text
    )

    target_result_list[-1] = SubtitleEvent(
        index=base_anchor_event.index,
        text=longer_text,
        interval=TimeInterval(base_anchor_event.start_sec, recalculated_end_time),
        confidence=Confidence(weighted_confidence_score),
        frame_count=combined_total_frames,
        position=base_anchor_event.position,
        bounding_box=base_anchor_event.bounding_box,
        uid=base_anchor_event.uid,
    )


def _execute_merge_onto_last(
    target_result_list: list[SubtitleEvent],
    base_anchor_event: SubtitleEvent,
    event_to_merge_in: SubtitleEvent,
) -> None:
    """Merge ``event_to_merge_in`` vào ``base_anchor_event`` (đã ở cuối list).

    Chọn text có weight (conf × frame_count) cao hơn. Tuy nhiên nếu là
    cặp superset chênh 1 ký tự CJK VÀ 1 cluster nhỏ (<=4 frames), ưu tiên
    text dài hơn (do OCR thường drop nhiều hơn add).
    """
    recalculated_end_time = max(base_anchor_event.end_sec, event_to_merge_in.end_sec)
    combined_total_frames = (
        base_anchor_event.frame_count + event_to_merge_in.frame_count
    )
    weighted_confidence_score = (
        float(base_anchor_event.confidence) * base_anchor_event.frame_count
        + float(event_to_merge_in.confidence) * event_to_merge_in.frame_count
    ) / max(1, combined_total_frames)

    is_one_cluster_small_enough_for_superset_override = (
        min(base_anchor_event.frame_count, event_to_merge_in.frame_count) <= 4
    )
    if is_one_cluster_small_enough_for_superset_override and is_superset_within_one_char(
        base_anchor_event.text, event_to_merge_in.text
    ):
        optimal_text_choice = (
            base_anchor_event.text
            if len(base_anchor_event.text) >= len(event_to_merge_in.text)
            else event_to_merge_in.text
        )
    else:
        base_event_weight_score = (
            float(base_anchor_event.confidence) * base_anchor_event.frame_count
        )
        merge_event_weight_score = (
            float(event_to_merge_in.confidence) * event_to_merge_in.frame_count
        )
        optimal_text_choice = (
            base_anchor_event.text
            if base_event_weight_score >= merge_event_weight_score
            else event_to_merge_in.text
        )

    target_result_list[-1] = SubtitleEvent(
        index=base_anchor_event.index,
        text=optimal_text_choice,
        interval=TimeInterval(base_anchor_event.start_sec, recalculated_end_time),
        confidence=Confidence(weighted_confidence_score),
        frame_count=combined_total_frames,
        position=base_anchor_event.position,
        bounding_box=base_anchor_event.bounding_box,
        uid=base_anchor_event.uid,
    )


def post_merge_duplicates(
    finalized_events: list[SubtitleEvent],
    config: SubtitleBuilderConfig,
) -> list[SubtitleEvent]:
    """Pass merge cuối: gộp events còn duplicate sau khi đã filter rác.

    Có 2 strategies:
        1. **Direct merge**: gộp current với previous nếu similarity >= ngưỡng.
        2. **Superset merge**: gộp khi 1 trong 2 là superset chênh 1 ký tự CJK
           VÀ 1 cluster nhỏ (<=4 frames). Tránh case 2 câu phụ đề thật.
        3. **Skip-1 merge**: nếu prev event quá ngắn (<=0.25s, có thể là rác
           đã miss), thử gộp current với prev-2 (loại prev khỏi result).

    Args:
        finalized_events: Events sau filter rác.
        config: Cấu hình builder.

    Returns:
        Events đã merge.
    """
    if len(finalized_events) <= 1:
        return finalized_events

    configured_similarity_threshold = config.similarity_threshold
    configured_maximum_gap = config.merge_gap_sec
    # ── v3.6+: Identical-text extended gap ──
    # Khi 2 event có text HOÀN TOÀN GIỐNG NHAU, cho phép gap rộng hơn
    # (2× merge_gap_sec) để xử lý animated subtitle "nhấp nháy" — phụ đề
    # xuất hiện, mờ 1 giây, rồi hiện lại → nên gộp thành 1 event duy nhất.
    _IDENTICAL_TEXT_GAP_MULTIPLIER: float = 2.0
    post_merged_result: list[SubtitleEvent] = [finalized_events[0]]

    for current_candidate_event in finalized_events[1:]:
        has_been_merged = False

        immediate_previous_event = post_merged_result[-1]
        time_gap_to_previous = (
            current_candidate_event.start_sec - immediate_previous_event.end_sec
        )

        # Xác định ngưỡng gap cho lần merge này: text giống nhau hoàn toàn
        # VÀ cả 2 cluster đều nhỏ (fc <= 8) → dùng gap rộng hơn để bắt
        # animated blink pattern (phụ đề nhấp nháy ngắn giữa 2 fragment).
        # Không mở rộng cho cluster lớn (fc > 8) vì đó là 2 câu thoại riêng biệt.
        effective_max_gap = configured_maximum_gap
        normalized_prev1_text_early = normalize_cjk_punctuation(
            immediate_previous_event.text
        )
        normalized_curr_text_early = normalize_cjk_punctuation(
            current_candidate_event.text
        )
        _BLINK_FC_THRESHOLD: int = 8
        both_small_clusters = (
            immediate_previous_event.frame_count <= _BLINK_FC_THRESHOLD
            and current_candidate_event.frame_count <= _BLINK_FC_THRESHOLD
        )
        if normalized_prev1_text_early == normalized_curr_text_early and both_small_clusters:
            effective_max_gap = configured_maximum_gap * _IDENTICAL_TEXT_GAP_MULTIPLIER

        if time_gap_to_previous <= effective_max_gap:
            # Reuse texts already computed for gap decision (identical path).
            normalized_prev1_text = normalized_prev1_text_early
            normalized_curr_text = normalized_curr_text_early

            # [Repeat Utterance Guard] Text GIỐNG HỆT + cả 2 cluster ổn định
            # (không phải mảnh blink) + phụ đề đã biến mất >= 0.35s giữa chừng
            # → nhân vật nói LẶP câu, giữ 2 event riêng (khớp phụ đề gốc).
            if (
                normalized_prev1_text == normalized_curr_text
                and time_gap_to_previous >= 0.35
                and immediate_previous_event.frame_count > _BLINK_FC_THRESHOLD
                and current_candidate_event.frame_count > _BLINK_FC_THRESHOLD
            ):
                post_merged_result.append(current_candidate_event)
                continue

            calculated_sim_score = calculate_effective_similarity(
                normalized_prev1_text,
                normalized_curr_text,
                float(immediate_previous_event.confidence),
                float(current_candidate_event.confidence),
                time_gap_to_previous,
            )

            # Phân biệt 2 câu khác nhau (cả 2 cluster lớn) vs OCR drop transient.
            are_both_clusters_substantial = (
                immediate_previous_event.frame_count >= 5
                and current_candidate_event.frame_count >= 5
            )
            is_short_superset_pattern = is_superset_within_one_char(
                normalized_prev1_text, normalized_curr_text
            ) and (
                max(len(normalized_prev1_text), len(normalized_curr_text)) <= 4
            )
            should_treat_as_distinct_pair = (
                are_both_clusters_substantial and is_short_superset_pattern
            )

            if (
                calculated_sim_score >= configured_similarity_threshold
                and not should_treat_as_distinct_pair
            ):
                _execute_merge_onto_last(
                    post_merged_result,
                    immediate_previous_event,
                    current_candidate_event,
                )
                has_been_merged = True
            elif (
                is_superset_within_one_char(
                    normalized_prev1_text, normalized_curr_text
                )
                and min(
                    immediate_previous_event.frame_count,
                    current_candidate_event.frame_count,
                )
                <= 4
            ):
                _execute_superset_merge(
                    post_merged_result,
                    immediate_previous_event,
                    current_candidate_event,
                )
                has_been_merged = True

        if not has_been_merged and len(post_merged_result) >= 2:
            second_previous_event = post_merged_result[-2]
            time_gap_to_second_prev = (
                current_candidate_event.start_sec - second_previous_event.end_sec
            )

            if (
                immediate_previous_event.duration_sec <= 0.25
                and time_gap_to_second_prev <= configured_maximum_gap * 1.5
            ):
                normalized_prev2_text = normalize_cjk_punctuation(
                    second_previous_event.text
                )
                normalized_curr_text = normalize_cjk_punctuation(
                    current_candidate_event.text
                )
                calculated_sim2_score = calculate_effective_similarity(
                    normalized_prev2_text,
                    normalized_curr_text,
                    float(second_previous_event.confidence),
                    float(current_candidate_event.confidence),
                    time_gap_to_second_prev,
                )

                if calculated_sim2_score >= configured_similarity_threshold:
                    post_merged_result.pop()
                    _execute_merge_onto_last(
                        post_merged_result,
                        second_previous_event,
                        current_candidate_event,
                    )
                    has_been_merged = True

        if not has_been_merged:
            post_merged_result.append(current_candidate_event)

    return post_merged_result


# ---------------------------------------------------------------------------
# Pass 7: Echo trail filter ở cấp event
# ---------------------------------------------------------------------------


def filter_echo_trail_events(
    source_events: list[SubtitleEvent],
    config: SubtitleBuilderConfig,
) -> list[SubtitleEvent]:
    """Filter echo trail ở cấp event — lớp filter cuối bắt rác còn sót.

    Có 2 loại candidate:
        1. **Echo dư âm**: duration < 0.20s, conf < 0.65, similar với
           neighbor conf >= 0.80, raw fuzz >= 0.20.
        2. **Single-frame outlier**: frame_count=1, duration < 0.12s,
           conf 0.65-0.90, similar với neighbor conf >= 0.90, raw fuzz >= 0.65.

    Args:
        source_events: Events sau toàn bộ pipeline.
        config: Cấu hình builder.

    Returns:
        Events đã loại bỏ echo còn sót.
    """
    if len(source_events) <= 1:
        return source_events

    max_neighbor_gap_seconds = config.merge_gap_sec * 1.5
    minimum_echo_ratio = 0.20
    single_frame_outlier_min_fuzz = 0.65
    kept_events: list[SubtitleEvent] = []
    dropped_count = 0

    for event_index, current_event in enumerate(source_events):
        current_confidence_value = float(current_event.confidence)

        is_echo_candidate = (
            current_event.duration_sec < 0.20 and current_confidence_value < 0.65
        )
        is_single_frame_outlier_candidate = (
            current_event.frame_count == 1
            and current_event.duration_sec < 0.12
            and 0.65 <= current_confidence_value < 0.90
        )

        if not is_echo_candidate and not is_single_frame_outlier_candidate:
            kept_events.append(current_event)
            continue

        current_normalized_text = normalize_cjk_punctuation(current_event.text)
        if not current_normalized_text:
            kept_events.append(current_event)
            continue

        required_minimum_ratio = (
            single_frame_outlier_min_fuzz
            if is_single_frame_outlier_candidate and not is_echo_candidate
            else minimum_echo_ratio
        )
        required_neighbor_confidence = (
            0.90
            if is_single_frame_outlier_candidate and not is_echo_candidate
            else 0.80
        )

        absorbed_by_neighbor = False

        if kept_events:
            prev_event = kept_events[-1]
            gap_to_prev = current_event.start_sec - prev_event.end_sec
            prev_confidence_value = float(prev_event.confidence)
            if (
                0.0 <= gap_to_prev <= max_neighbor_gap_seconds
                and prev_confidence_value >= required_neighbor_confidence
            ):
                prev_normalized_text = normalize_cjk_punctuation(prev_event.text)
                if prev_normalized_text:
                    echo_ratio = (
                        fuzz.ratio(prev_normalized_text, current_normalized_text)
                        / 100.0
                    )
                    if echo_ratio >= required_minimum_ratio:
                        absorbed_by_neighbor = True

        if not absorbed_by_neighbor and event_index + 1 < len(source_events):
            next_event = source_events[event_index + 1]
            gap_to_next = next_event.start_sec - current_event.end_sec
            next_confidence_value = float(next_event.confidence)
            if (
                0.0 <= gap_to_next <= max_neighbor_gap_seconds
                and next_confidence_value >= required_neighbor_confidence
            ):
                next_normalized_text = normalize_cjk_punctuation(next_event.text)
                if next_normalized_text:
                    echo_ratio = (
                        fuzz.ratio(next_normalized_text, current_normalized_text)
                        / 100.0
                    )
                    if echo_ratio >= required_minimum_ratio:
                        absorbed_by_neighbor = True

        if absorbed_by_neighbor:
            dropped_count += 1
            continue

        kept_events.append(current_event)

    if dropped_count:
        logger.debug("Event-level echo filter đã loại {} event.", dropped_count)

    return kept_events


def reindex_events(unindexed_events_list: list[SubtitleEvent]) -> list[SubtitleEvent]:
    """Cập nhật trường ``index`` của events thành thứ tự 1-based.

    Args:
        unindexed_events_list: Events đã sort theo thời gian.

    Returns:
        Cùng list đã được mutate (index 1, 2, 3, ...).
    """
    for new_index, event_object in enumerate(unindexed_events_list, start=1):
        event_object.index = new_index
    return unindexed_events_list


# ---------------------------------------------------------------------------
# Pass 8 (optional): Strip persistent watermark lines từ multi-line events
# ---------------------------------------------------------------------------


def _is_watermark_latin_line(line: str) -> bool:
    """Kiểm tra line có phải dòng watermark Latin trong event CJK.

    Watermark thường là: 1 từ viết hoa đơn (vd 'ARBOR', 'LOGO', 'HOME'),
    hoặc code/số (vd 'HD1080', 'EP01'). Không có CJK, không có dấu câu CJK.

    Args:
        line: Dòng văn bản cần kiểm tra.

    Returns:
        ``True`` nếu là watermark Latin đơn thuần.
    """
    stripped = line.strip()
    if not stripped:
        return False
    # Phải là ASCII thuần — không lẫn CJK.
    if any(ord(c) > 0x3000 for c in stripped):
        return False
    # Ít nhất 2 ký tự.
    if len(stripped) < 2:
        return False
    # Viết hoa hoàn toàn hoặc mixed alphanumeric ngắn.
    upper_clean = stripped.upper().replace("-", "").replace("_", "")
    if not upper_clean.replace(" ", "").isalnum():
        return False
    words = stripped.split()
    # Tối đa 3 từ để tránh nhầm subtitle tiếng Anh thật.
    if len(words) > 3:
        return False
    # Ít nhất 50% là chữ in hoa (loại câu tiếng Anh bình thường viết thường).
    upper_count = sum(1 for c in stripped if c.isupper())
    alpha_count = sum(1 for c in stripped if c.isalpha())
    if alpha_count == 0:
        return False
    return upper_count / alpha_count >= 0.80


def strip_latin_watermark_prefix_from_cjk_events(
    subtitle_events: list[SubtitleEvent],
) -> list[SubtitleEvent]:
    """Strip dòng watermark Latin khỏi event CJK multi-line.

    Nhận biết và loại bỏ dòng prefix/suffix Latin thuần (vd 'ARBOR',
    'HOME', 'LOGO') trong event có phần còn lại là CJK. Pattern:

    * Event có >= 2 dòng.
    * Dòng đầu hoặc cuối là Latin watermark (``_is_watermark_latin_line``).
    * Các dòng còn lại có chứa CJK.

    Không drop event — chỉ strip dòng Latin. Nếu sau strip event chỉ còn
    1 dòng CJK vẫn giữ nguyên.

    Args:
        subtitle_events: Danh sách events.

    Returns:
        Events đã strip watermark Latin prefix/suffix.
    """
    stripped_count = 0
    result: list[SubtitleEvent] = []
    import dataclasses as _dc

    for event in subtitle_events:
        lines = [ln.strip() for ln in event.text.splitlines() if ln.strip()]
        if len(lines) < 2:
            result.append(event)
            continue

        # Kiểm tra dòng đầu / dòng cuối là watermark Latin.
        head_is_watermark = _is_watermark_latin_line(lines[0])
        tail_is_watermark = _is_watermark_latin_line(lines[-1])

        if not head_is_watermark and not tail_is_watermark:
            result.append(event)
            continue

        # Chỉ strip nếu phần còn lại có CJK để tránh mất subtitle tiếng Anh thật.
        remaining_after_strip = lines[1:] if head_is_watermark else lines
        if tail_is_watermark and remaining_after_strip:
            remaining_after_strip = remaining_after_strip[:-1] if tail_is_watermark else remaining_after_strip

        # Kiểm tra lại: strip cả đầu lẫn cuối nếu cần.
        clean_lines = list(lines)
        if head_is_watermark and contains_cjk("".join(clean_lines[1:])):
            clean_lines = clean_lines[1:]
            stripped_count += 1
        if (
            clean_lines
            and tail_is_watermark
            and contains_cjk("".join(clean_lines[:-1]))
        ):
            clean_lines = clean_lines[:-1]
            stripped_count += 1

        if not clean_lines:
            # Không còn gì sau strip → drop.
            continue

        if len(clean_lines) == len(lines):
            result.append(event)
        else:
            result.append(_dc.replace(event, text="\n".join(clean_lines)))

    if stripped_count:
        logger.debug(
            "Latin-watermark-prefix stripper: {} line stripped từ multi-line events.",
            stripped_count,
        )
    return result



def strip_persistent_watermark_lines(
    subtitle_events: list[SubtitleEvent],
    min_occurrence_ratio: float = 0.08,
    min_absolute_count: int = 5,
) -> list[SubtitleEvent]:
    """Loại bỏ dòng watermark cứng nhắc xuất hiện trong nhiều event.

    Watermark logo/thương hiệu (vd 'ARBOR', 'HOME', '天之') hay xuất hiện
    như line đầu hoặc cuối của event multi-line, lặp lại liên tục qua hàng
    chục đến hàng trăm event. Hàm này:

    1. Đếm tần suất xuất hiện của từng line trong TẤT CẢ multi-line events.
    2. Nếu một line có:
       * tần suất >= ``min_occurrence_ratio`` × số lượng multi-line events
       * VÀ xuất hiện >= ``min_absolute_count`` lần tuyệt đối
    → đánh dấu là watermark line cần loại.
    3. Với mỗi multi-line event có chứa watermark line → strip, giữ lại
       các dòng còn lại.
    4. Nếu sau khi strip, event chỉ còn 1 dòng → giữ lại (không drop).
    5. Nếu event chỉ có watermark, bỏ hoàn toàn.

    Chỉ áp dụng cho **Latin-dominant** line để tránh strip nhầm CJK dialogue
    ngắn (vd '哦', '嗯'). CJK watermark thường dài hơn và bị bắt bởi các
    filter khác.

    Args:
        subtitle_events: Danh sách events sau toàn bộ pipeline chính.
        min_occurrence_ratio: Tỉ lệ tối thiểu (0.08 = 8% multi-line events).
        min_absolute_count: Số lần xuất hiện tuyệt đối tối thiểu.

    Returns:
        Danh sách events đã strip watermark lines.
    """
    if not subtitle_events:
        return subtitle_events

    # Bước 1: Thu thập tất cả multi-line events và đếm tần suất từng line.
    multiline_events_indices: list[int] = []
    line_occurrence_counter: dict[str, int] = {}

    for event_idx, event in enumerate(subtitle_events):
        raw_lines = [ln.strip() for ln in event.text.splitlines() if ln.strip()]
        if len(raw_lines) < 2:
            continue
        multiline_events_indices.append(event_idx)
        for line_text in raw_lines:
            line_occurrence_counter[line_text] = (
                line_occurrence_counter.get(line_text, 0) + 1
            )

    if not multiline_events_indices:
        return subtitle_events

    total_multiline_count = len(multiline_events_indices)
    absolute_threshold = max(min_absolute_count, 1)
    ratio_threshold = total_multiline_count * min_occurrence_ratio

    # Bước 2: Xác định watermark lines — chỉ Latin-dominant để tránh
    # false-positive với CJK dialogue ngắn.
    def _is_latin_dominant_line(line: str) -> bool:
        """Kiểm tra line có phải Latin-dominant (>50% Latin char)."""
        stripped = line.replace(" ", "")
        if not stripped:
            return False
        latin_chars = sum(1 for c in stripped if c.isascii() and c.isalpha())
        return latin_chars / len(stripped) > 0.50

    watermark_lines: frozenset[str] = frozenset(
        line
        for line, count in line_occurrence_counter.items()
        if (
            count >= absolute_threshold
            and count >= ratio_threshold
            and _is_latin_dominant_line(line)
        )
    )

    if not watermark_lines:
        return subtitle_events

    stripped_count = 0
    dropped_count = 0
    result_events: list[SubtitleEvent] = []

    for event in subtitle_events:
        raw_lines = [ln.strip() for ln in event.text.splitlines() if ln.strip()]

        if len(raw_lines) < 2:
            # [v3.23.150] Event MỘT DÒNG mà nội dung CHÍNH LÀ watermark đã xác định
            # (bằng chứng thống kê từ multi-line: >= min_absolute_count lần và >=
            # min_occurrence_ratio) -> đây là lúc logo hiện MỘT MÌNH trên màn hình
            # (không có thoại) -> drop. Trước đây nhánh này giữ nguyên mọi event
            # một dòng nên phụ đề đầu ra lẫn hàng loạt event watermark đơn độc.
            if len(raw_lines) == 1 and raw_lines[0] in watermark_lines:
                dropped_count += 1
                continue
            result_events.append(event)
            continue

        clean_lines = [ln for ln in raw_lines if ln not in watermark_lines]

        if not clean_lines:
            # Event chỉ toàn watermark → drop hoàn toàn.
            dropped_count += 1
            continue

        if len(clean_lines) == len(raw_lines):
            # Không có gì bị strip → giữ nguyên.
            result_events.append(event)
            continue

        # Tạo event mới với text đã strip watermark.
        stripped_count += 1
        new_text = "\n".join(clean_lines)
        import dataclasses as _dc

        result_events.append(
            _dc.replace(event, text=new_text)
        )

    if stripped_count or dropped_count:
        logger.debug(
            "Watermark-line stripper: {} event stripped, {} event dropped. "
            "Watermark lines: {}",
            stripped_count,
            dropped_count,
            list(watermark_lines)[:5],
        )

    return result_events


def extend_group_timing_from_soft_drops(
    groups: list[FrameGroup],
    soft_dropped_sorted: list[float],
    sample_step_sec: float,
) -> list[FrameGroup]:
    """Phục hồi timing biên của group bằng timestamp frame bị lọc sát biên.

    **Vấn đề cần giải quyết**: Stage 2 (spatial cleanup) và Stage 3
    (confidence filter) loại bỏ các frame có nội dung nhưng không đạt
    ngưỡng chất lượng. Các frame này thường là **frame đầu/cuối của một
    phụ đề** — khi text mới xuất hiện (animation fade-in, trượt vào) hoặc
    bắt đầu mờ dần (fade-out). Kết quả:

    * ``start_timestamp_sec`` của group **trễ hơn** thời điểm thực tế ≈ 1
      sample step (40 ms ở 25 fps lấy mẫu).
    * ``end_timestamp_sec`` **sớm hơn** thời điểm thực tế ≈ 1 sample step.

    **Cách sửa**: Thu thập timestamp của các frame "bị lọc nhưng có nội
    dung" (soft-drop) trong stages 2+3. Với mỗi group, kiểm tra xem có
    soft-drop nào nằm sát biên (trong cửa sổ 0.5–1.5 × sample_step) không.
    Nếu có, mở rộng timing biên tới soft-drop đó — nhưng không vượt qua
    group liền kề để tránh overlap.

    Args:
        groups:              FrameGroup đã merge (sort theo timestamp).
        soft_dropped_sorted: Timestamp sort tăng dần của frame bị lọc có nội dung.
        sample_step_sec:     Bước lấy mẫu (config.sample_step_sec).

    Returns:
        Danh sách group đã mở rộng timing (mutate in-place và return).
    """
    if not groups or not soft_dropped_sorted:
        return groups

    # Cửa sổ tìm kiếm: 0.5× – 1.5× step.
    # < 0.5×: quá gần, có thể là PTS float noise, không phải frame bị lọc.
    # > 1.5×: quá xa, không còn chắc là frame của cùng phụ đề.
    _WINDOW_LOW = sample_step_sec * 0.5
    _WINDOW_HIGH = sample_step_sec * 1.5

    n = len(groups)
    for i, group in enumerate(groups):
        # --- Mở rộng START về phía trước ---
        target_start = group.start_timestamp_sec
        min_allowed = groups[i - 1].end_timestamp_sec if i > 0 else -1e9

        lo = bisect.bisect_left(soft_dropped_sorted, target_start - _WINDOW_HIGH)
        hi = bisect.bisect_left(soft_dropped_sorted, target_start - _WINDOW_LOW)
        candidates_start = [
            t for t in soft_dropped_sorted[lo:hi] if t > min_allowed
        ]
        if candidates_start:
            new_start = candidates_start[0]  # frame bị lọc SỚM NHẤT liền kề
            group.start_timestamp_sec = max(new_start, min_allowed + 0.001)

        # --- Mở rộng END về phía sau ---
        target_end = group.end_timestamp_sec
        max_allowed = groups[i + 1].start_timestamp_sec if i + 1 < n else 1e9

        lo2 = bisect.bisect_right(soft_dropped_sorted, target_end + _WINDOW_LOW)
        hi2 = bisect.bisect_right(soft_dropped_sorted, target_end + _WINDOW_HIGH)
        candidates_end = [
            t for t in soft_dropped_sorted[lo2:hi2] if t < max_allowed
        ]
        if candidates_end:
            new_end = candidates_end[-1]  # frame bị lọc MUỘN NHẤT liền kề
            group.end_timestamp_sec = min(new_end, max_allowed - 0.001)

    return groups


__all__ = [
    "convert_groups_to_events",
    "extend_group_timing_from_soft_drops",
    "filter_echo_trail_events",
    "filter_echo_trail_groups",
    "filter_short_duration_events",
    "filter_short_text_events",
    "is_superset_within_one_char",
    "merge_adjacent_duplicates",
    "post_merge_duplicates",
    "reindex_events",
    "strip_latin_watermark_prefix_from_cjk_events",
    "strip_persistent_watermark_lines",
]
