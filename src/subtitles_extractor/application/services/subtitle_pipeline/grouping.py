"""Gộp các frame OCR liên tiếp thành các :class:`FrameGroup`.

Có 2 thuật toán:

1. **Greedy** (mặc định, O(n)): Duyệt tuần tự, mỗi frame mới hoặc gộp
   vào group hiện tại (nếu similarity >= ngưỡng động) hoặc mở group
   mới. Nhanh và đủ tốt cho 95% trường hợp.

2. **Viterbi** (O(n×W)): Dynamic programming tối ưu toàn cục — tốt hơn
   khi OCR nhiễu nhiều. Chậm hơn 5-10×, bật trong cài đặt khi cần.
"""

from __future__ import annotations

from collections.abc import Sequence

from subtitles_extractor.application.dtos.extract_subtitles_dto import (
    SubtitleBuilderConfig,
)
from subtitles_extractor.application.services.subtitle_pipeline.frame_group import (
    FrameGroup,
    extract_aggregated_bounding_box,
    extract_primary_spatial_position,
)
from subtitles_extractor.application.services.subtitle_pipeline.voting import (
    calculate_effective_similarity,
    extract_joined_text_from_frame,
    get_frame_confidence_cached,
    vote_best_text_rover,
)
from subtitles_extractor.application.services.viterbi_grouper import (
    ViterbiGrouper,
    ViterbiGrouperConfig,
)
from subtitles_extractor.domain.entities.ocr_frame_result import OcrFrameResult


def create_initial_frame_group(
    frame_result: OcrFrameResult,
    config: SubtitleBuilderConfig,
) -> FrameGroup:
    """Tạo group mới chứa duy nhất 1 frame.

    Args:
        frame_result: Frame đầu tiên của group.
        config: Cấu hình builder.

    Returns:
        :class:`FrameGroup` mới khởi tạo.
    """
    return FrameGroup(
        reconstructed_text=extract_joined_text_from_frame(
            frame_result,
            config.y_clustering_tolerance_ratio,
            config.y_clustering_tolerance_min_px,
        ),
        start_timestamp_sec=frame_result.timestamp_sec,
        end_timestamp_sec=frame_result.timestamp_sec,
        accumulated_confidence=float(frame_result.mean_confidence),
        total_frames_count=1,
        primary_center_position=extract_primary_spatial_position(frame_result),
        aggregated_bounding_box=extract_aggregated_bounding_box([frame_result]),
    )


#: [Repeat Utterance Guard] Khoảng trống tối thiểu (s) để coi phụ đề đã "biến mất
#: thật" giữa 2 lần xuất hiện cùng text (OCR flicker chỉ trống < 0.2s).
_REPEAT_GAP_SPLIT_SEC: float = 0.35
#: Group cũ phải ổn định tối thiểu (s) trước khi cho phép tách câu lặp.
_REPEAT_MIN_STABLE_SEC: float = 0.30


def group_using_greedy(
    frames_list: Sequence[OcrFrameResult],
    config: SubtitleBuilderConfig,
) -> list[FrameGroup]:
    """Gộp frame bằng greedy O(n) — nhanh, đủ tốt cho 95% trường hợp.

    Algorithm:
        1. Bắt đầu với group chứa frame[0].
        2. Với mỗi frame kế tiếp:
           - Nếu gap > merge_gap_sec → mở group mới.
           - Tính similarity với text hiện tại của group.
           - Threshold dynamic theo gap thời gian (gần thì lỏng hơn).
           - Nếu sim >= threshold → gộp vào, vote lại text canonical.
           - Ngược lại → mở group mới.

    Args:
        frames_list: Chuỗi frame OCR đã sort theo thời gian.
        config: Cấu hình builder.

    Returns:
        Danh sách :class:`FrameGroup` đã gộp.
    """
    if not frames_list:
        return []

    frames_seq = list(frames_list)
    final_assembled_groups: list[FrameGroup] = [
        create_initial_frame_group(frames_seq[0], config)
    ]
    raw_grouped_frames: list[list[OcrFrameResult]] = [[frames_seq[0]]]

    base_similarity_required = config.similarity_threshold
    maximum_allowable_gap_sec = config.merge_gap_sec
    configured_sample_step_sec = config.sample_step_sec
    y_ratio = config.y_clustering_tolerance_ratio
    y_min_px = config.y_clustering_tolerance_min_px

    for current_frame in frames_seq[1:]:
        active_frame_group = final_assembled_groups[-1]
        time_gap_duration_seconds = (
            current_frame.timestamp_sec - active_frame_group.end_timestamp_sec
        )

        if time_gap_duration_seconds > maximum_allowable_gap_sec:
            final_assembled_groups.append(
                create_initial_frame_group(current_frame, config)
            )
            raw_grouped_frames.append([current_frame])
            continue

        # [Repeat Utterance Guard] Câu LẶP: text mới GIỐNG HỆT group hiện tại nhưng
        # phụ đề đã biến mất một khoảng dài (>= 0.35s frame trống liên tục) và group
        # cũ đã ổn định (>= 0.30s). OCR flicker thật chỉ trống 1-4 frame (< 0.2s);
        # khoảng trống dài + hai khối ổn định = nhân vật nói lặp → tách 2 câu.
        if time_gap_duration_seconds >= _REPEAT_GAP_SPLIT_SEC:
            _active_duration = (
                active_frame_group.end_timestamp_sec
                - active_frame_group.start_timestamp_sec
            )
            _current_text_probe = extract_joined_text_from_frame(
                current_frame, y_ratio, y_min_px
            )
            if (
                _active_duration >= _REPEAT_MIN_STABLE_SEC
                and _current_text_probe
                and _current_text_probe == active_frame_group.reconstructed_text
            ):
                final_assembled_groups.append(
                    create_initial_frame_group(current_frame, config)
                )
                raw_grouped_frames.append([current_frame])
                continue

        previous_group_text = active_frame_group.reconstructed_text
        current_frame_text = extract_joined_text_from_frame(
            current_frame, y_ratio, y_min_px
        )
        previous_mean_confidence = active_frame_group.calculate_mean_confidence()
        current_frame_confidence = get_frame_confidence_cached(current_frame)

        measured_similarity_score = calculate_effective_similarity(
            previous_group_text,
            current_frame_text,
            previous_mean_confidence,
            current_frame_confidence,
            time_gap_duration_seconds,
        )

        is_temporally_continuous = time_gap_duration_seconds <= (
            configured_sample_step_sec * 1.5
        )
        dynamic_similarity_threshold = base_similarity_required

        if is_temporally_continuous:
            if current_frame_confidence < previous_mean_confidence:
                dynamic_similarity_threshold = max(
                    0.60, base_similarity_required - 0.30
                )
            else:
                dynamic_similarity_threshold = max(
                    0.50, base_similarity_required - 0.10
                )

        if measured_similarity_score >= dynamic_similarity_threshold:
            active_frame_group.end_timestamp_sec = current_frame.timestamp_sec
            active_frame_group.accumulated_confidence += current_frame_confidence
            active_frame_group.total_frames_count += 1
            raw_grouped_frames[-1].append(current_frame)
            active_frame_group.reconstructed_text = vote_best_text_rover(
                raw_grouped_frames[-1], y_ratio, y_min_px
            )
        else:
            final_assembled_groups.append(
                create_initial_frame_group(current_frame, config)
            )
            raw_grouped_frames.append([current_frame])

    # Cập nhật aggregated_bounding_box + member_texts cho các group.
    for assembled_group, member_frames in zip(
        final_assembled_groups, raw_grouped_frames, strict=True
    ):
        if len(member_frames) > 1:
            assembled_group.aggregated_bounding_box = extract_aggregated_bounding_box(
                member_frames
            )
        # [Stable Variant Split] Lưu text từng frame để pass hậu kỳ tách group
        # chứa nhiều khối text ổn định (2 câu thật bị merge nhầm).
        assembled_group.member_texts = [
            (mf.timestamp_sec, extract_joined_text_from_frame(mf, y_ratio, y_min_px))
            for mf in member_frames
        ]

    return final_assembled_groups


def group_using_viterbi(
    frames_list: Sequence[OcrFrameResult],
    config: SubtitleBuilderConfig,
) -> list[FrameGroup]:
    """Gộp frame bằng Viterbi DP — tối ưu toàn cục O(n×W).

    Dùng khi OCR nhiễu, các text variants nhiều và greedy có thể chọn
    sai. Chậm hơn 5-10× so với greedy.

    Args:
        frames_list: Chuỗi frame OCR đã sort theo thời gian.
        config: Cấu hình builder.

    Returns:
        Danh sách :class:`FrameGroup` đã gộp.
    """
    frames_seq = list(frames_list)
    viterbi_dp_engine = ViterbiGrouper(
        ViterbiGrouperConfig(
            open_penalty=config.viterbi_open_penalty,
            max_gap_sec=config.merge_gap_sec,
            min_similarity_to_join=max(0.50, config.similarity_threshold - 0.15),
            sample_step_sec=config.sample_step_sec,
            y_clustering_tolerance_ratio=config.y_clustering_tolerance_ratio,
            y_clustering_tolerance_min_px=config.y_clustering_tolerance_min_px,
        )
    )
    assigned_cluster_labels = viterbi_dp_engine.group(frames_seq)
    if not assigned_cluster_labels:
        return []

    clustered_frame_buckets: dict[int, list[OcrFrameResult]] = {}
    for cluster_label_id, current_frame in zip(
        assigned_cluster_labels, frames_seq, strict=True
    ):
        clustered_frame_buckets.setdefault(cluster_label_id, []).append(current_frame)

    final_constructed_groups: list[FrameGroup] = []
    y_ratio = config.y_clustering_tolerance_ratio
    y_min_px = config.y_clustering_tolerance_min_px

    for cluster_label_id in sorted(clustered_frame_buckets.keys()):
        frames_in_current_bucket = clustered_frame_buckets[cluster_label_id]
        optimal_voted_text = vote_best_text_rover(
            frames_in_current_bucket, y_ratio, y_min_px
        )
        representative_primary_frame = frames_in_current_bucket[0]

        for frame_candidate in frames_in_current_bucket:
            if (
                extract_joined_text_from_frame(frame_candidate, y_ratio, y_min_px)
                == optimal_voted_text
            ):
                representative_primary_frame = frame_candidate
                break

        final_constructed_groups.append(
            FrameGroup(
                reconstructed_text=optimal_voted_text,
                start_timestamp_sec=frames_in_current_bucket[0].timestamp_sec,
                end_timestamp_sec=frames_in_current_bucket[-1].timestamp_sec,
                accumulated_confidence=sum(
                    float(f.mean_confidence) for f in frames_in_current_bucket
                ),
                total_frames_count=len(frames_in_current_bucket),
                primary_center_position=extract_primary_spatial_position(
                    representative_primary_frame
                ),
                aggregated_bounding_box=extract_aggregated_bounding_box(
                    frames_in_current_bucket
                ),
            )
        )
    return final_constructed_groups


__all__ = ["create_initial_frame_group", "group_using_greedy", "group_using_viterbi"]
