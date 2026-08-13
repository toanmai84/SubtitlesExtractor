"""Lọc box theo **vị trí không gian** (ROI alignment, Y-axis outlier).

Có 2 lớp filter không gian:

1. **Per-frame** (:func:`clean_spatial_outliers`): Trong từng frame, lọc
   những line text lệch khỏi ROI alignment (CENTER/LEFT/RIGHT) — chống
   rác background như "土", "士" lệch tâm. Dùng dynamic threshold tuỳ
   theo độ dài text và confidence.

2. **Cross-frame** (:func:`filter_cross_frame_spatial_outliers`): Toàn
   bộ video, lọc box có Y-center là outlier so với median của tất cả
   box trong video — chống các box rớt out-of-line (vd subtitle xuất
   hiện ở vị trí khác bình thường).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

from subtitles_extractor.application.dtos.extract_subtitles_dto import (
    SubtitleBuilderConfig,
)
from subtitles_extractor.application.services.outlier_detection import (
    filter_y_position_outliers,
)
from subtitles_extractor.domain.entities.ocr_frame_result import (
    OcrFrameResult,
    OcrTextBox,
)
from subtitles_extractor.domain.value_objects.roi import Roi, TextAlignment


def clean_spatial_outliers(
    frame_result: OcrFrameResult,
    roi: Roi | None,
    config: SubtitleBuilderConfig,
) -> OcrFrameResult:
    """Lọc các line text lệch khỏi căn lề của ROI trong một frame.

    Algorithm:
        1. Gom các box thành các line dựa trên overlap Y >= 40%.
        2. Với mỗi line, tính x_center và so với center của ROI.
        3. Tính ngưỡng dung sai dynamic dựa trên độ dài text + confidence:
            * Text >= 5 chars: full tolerance.
            * Text 3-4 chars: 75-95% tolerance theo conf.
            * Text 1-2 chars conf >= 0.85: 90% tolerance.
            * Text 1-2 chars conf 0.65-0.85: 65% tolerance.
            * Text 1-2 chars conf < 0.65: 40% tolerance, max 35px.
        4. Drop line nào lệch quá ngưỡng.

    Args:
        frame_result: Frame OCR cần lọc.
        roi: ROI với alignment đã được xác định. Nếu ``None`` thì không lọc.
        config: Cấu hình builder (cho các ngưỡng tolerance).

    Returns:
        Frame mới với các box hợp lệ. Nếu không có ROI hoặc không có
        box nào có polygon thì trả nguyên bản frame.
    """
    if roi is None:
        return frame_result

    # Ép kiểu Alignment về CENTER nếu ROI cũ từ Database không có Alignment.
    active_alignment = (
        TextAlignment.CENTER if roi.alignment == TextAlignment.UNKNOWN else roi.alignment
    )

    polygon_equipped_boxes = [box for box in frame_result.text_boxes if box.polygon]
    if not polygon_equipped_boxes:
        return frame_result

    spatial_metrics: list[tuple[float, float, float, float, OcrTextBox]] = []
    for text_box in polygon_equipped_boxes:
        y_coordinates = [point[1] for point in text_box.polygon]
        x_coordinates = [point[0] for point in text_box.polygon]
        y_minimum, y_maximum = min(y_coordinates), max(y_coordinates)
        x_minimum, x_maximum = min(x_coordinates), max(x_coordinates)
        spatial_metrics.append((y_minimum, y_maximum, x_minimum, x_maximum, text_box))

    spatial_metrics.sort(key=lambda spatial_tuple: spatial_tuple[0])

    clustered_text_lines: list[
        list[tuple[float, float, float, float, OcrTextBox]]
    ] = [[spatial_metrics[0]]]

    for current_metric in spatial_metrics[1:]:
        previous_line_cluster = clustered_text_lines[-1]
        current_y_minimum, current_y_maximum = current_metric[0], current_metric[1]
        cluster_y_minimum = min(
            spatial_tuple[0] for spatial_tuple in previous_line_cluster
        )
        cluster_y_maximum = max(
            spatial_tuple[1] for spatial_tuple in previous_line_cluster
        )

        current_height = current_y_maximum - current_y_minimum
        cluster_height = cluster_y_maximum - cluster_y_minimum

        if current_height <= 0 or cluster_height <= 0:
            clustered_text_lines.append([current_metric])
            continue

        y_overlap_amount = max(
            0.0,
            min(current_y_maximum, cluster_y_maximum)
            - max(current_y_minimum, cluster_y_minimum),
        )
        if (y_overlap_amount / min(current_height, cluster_height)) > 0.40:
            clustered_text_lines[-1].append(current_metric)
        else:
            clustered_text_lines.append([current_metric])

    verified_text_boxes: list[OcrTextBox] = []
    roi_width_float_value = float(roi.width)

    for text_line_cluster in clustered_text_lines:
        line_x_minimum = min(spatial_tuple[2] for spatial_tuple in text_line_cluster)
        line_x_maximum = max(spatial_tuple[3] for spatial_tuple in text_line_cluster)
        line_x_center_point = (line_x_minimum + line_x_maximum) / 2.0
        total_line_width = line_x_maximum - line_x_minimum

        joined_cluster_text = "".join(
            spatial_tuple[4].text for spatial_tuple in text_line_cluster
        ).strip()
        text_character_count = len(joined_cluster_text)

        cluster_confidence_values = [
            float(spatial_tuple[4].confidence) for spatial_tuple in text_line_cluster
        ]
        mean_cluster_confidence = (
            sum(cluster_confidence_values) / len(cluster_confidence_values)
            if cluster_confidence_values
            else 0.0
        )

        is_text_line_valid = True
        if total_line_width > roi_width_float_value * 1.02:
            is_text_line_valid = False
        else:
            if active_alignment == TextAlignment.CENTER:
                base_allowed_deviation = max(
                    roi_width_float_value * config.alignment_center_tolerance_ratio,
                    config.alignment_tolerance_min_px,
                )

                if text_character_count <= 2:
                    if mean_cluster_confidence >= 0.85:
                        dynamic_allowed_deviation = base_allowed_deviation * 0.90
                    elif mean_cluster_confidence >= 0.65:
                        dynamic_allowed_deviation = base_allowed_deviation * 0.65
                    else:
                        dynamic_allowed_deviation = min(
                            base_allowed_deviation * 0.40, 35.0
                        )
                elif text_character_count <= 4:
                    if mean_cluster_confidence >= 0.85:
                        dynamic_allowed_deviation = base_allowed_deviation * 0.95
                    else:
                        dynamic_allowed_deviation = base_allowed_deviation * 0.75
                else:
                    dynamic_allowed_deviation = base_allowed_deviation

                if (
                    abs(line_x_center_point - (roi_width_float_value / 2.0))
                    > dynamic_allowed_deviation
                ):
                    is_text_line_valid = False

            elif active_alignment == TextAlignment.LEFT:
                allowed_left_margin_deviation = max(
                    roi_width_float_value * config.alignment_margin_tolerance_ratio,
                    config.alignment_tolerance_min_px,
                )
                if line_x_minimum > allowed_left_margin_deviation:
                    is_text_line_valid = False

            elif active_alignment == TextAlignment.RIGHT:
                allowed_right_margin_deviation = max(
                    roi_width_float_value * config.alignment_margin_tolerance_ratio,
                    config.alignment_tolerance_min_px,
                )
                if line_x_maximum < (
                    roi_width_float_value - allowed_right_margin_deviation
                ):
                    is_text_line_valid = False

        if is_text_line_valid:
            verified_text_boxes.extend(
                spatial_tuple[4] for spatial_tuple in text_line_cluster
            )

    return dataclasses.replace(frame_result, text_boxes=verified_text_boxes)


def _rescue_wrapped_multiline_boxes(
    frames_list: Sequence[OcrFrameResult],
    kept_box_memory_ids: set[int],
    kept_y_centers: list[float],
    roi_height: float,
    vertical_gap_ratio: float = 0.8,
) -> None:
    """Cứu các box của phụ đề NHIỀU DÒNG bị bộ lọc density-Y xoá nhầm.

    Phụ đề hai dòng có dòng-1 nằm TRÊN và dòng-2 nằm DƯỚI tâm băng đơn-dòng, nên
    khi xét từng box riêng lẻ thì không dòng nào rơi vào băng dày đặc → cả hai bị
    xoá. Hàm này gom các box trong cùng frame thành **chồng dọc** (x chồng nhau, y
    kề sát) rồi khôi phục cả chồng nếu **tâm dọc tổ hợp** của chồng nằm trong băng
    Y hợp lệ (gần tâm của các box đã được giữ). Nhiễu rời rạc ở rìa khung có tâm
    tổ hợp xa băng chính nên không bị cứu.

    Args:
        frames_list: Các frame OCR (đã qua per-frame cleanup).
        kept_box_memory_ids: Tập ``id`` box được giữ — mở rộng tại chỗ.
        kept_y_centers: Y-center của các box đã giữ (để xác định băng hợp lệ).
        roi_height: Chiều cao ROI (px) để định dung sai băng; 0 nếu không có ROI.
        vertical_gap_ratio: Khe dọc tối đa giữa hai box trong một chồng (× chiều cao).
    """
    sorted_centers = sorted(kept_y_centers)
    band_center = sorted_centers[len(sorted_centers) // 2]  # median
    # Dung sai băng: nửa chiều cao ROI nếu có, ngược lại từ độ trải của box giữ.
    if roi_height > 0:
        band_tolerance = roi_height / 2.0
    else:
        spread = sorted_centers[-1] - sorted_centers[0]
        band_tolerance = max(20.0, spread / 2.0)

    newly_rescued_ids: set[int] = set()
    for current_frame in frames_list:
        boxes_with_geometry = [
            box for box in current_frame.text_boxes if box.bounding_box
        ]
        if len(boxes_with_geometry) < 2:
            continue

        # [Cổng chính xác] Chỉ can thiệp khi TOÀN BỘ box của frame bị drop — tức
        # cue sắp biến mất hoàn toàn (đặc trưng phụ đề nhiều dòng nằm lệch băng
        # đơn-dòng). Frame đã có dòng chính được giữ thì KHÔNG đụng tới, tránh cứu
        # nhầm text phụ/nhiễu kề bên dòng chính (giữ nguyên hành vi, không hồi quy).
        if any(id(box) in kept_box_memory_ids for box in boxes_with_geometry):
            continue

        # Gom chồng dọc: sắp theo Y rồi nối các box x-overlap & y-kề-sát.
        ordered_boxes = sorted(
            boxes_with_geometry, key=lambda b: b.bounding_box[1]
        )
        current_stack: list[object] = [ordered_boxes[0]]
        stacks: list[list[object]] = []
        for box in ordered_boxes[1:]:
            top_box = current_stack[-1]
            t_x_min, _t_y_min, t_x_max, t_y_max = top_box.bounding_box
            b_x_min, b_y_min, b_x_max, _b_y_max = box.bounding_box
            horizontal_overlap = min(t_x_max, b_x_max) - max(t_x_min, b_x_min)
            top_height = max(1.0, float(t_y_max - _t_y_min))
            vertical_gap = b_y_min - t_y_max
            if horizontal_overlap > 0 and vertical_gap <= vertical_gap_ratio * top_height:
                current_stack.append(box)
            else:
                stacks.append(current_stack)
                current_stack = [box]
        stacks.append(current_stack)

        for stack in stacks:
            if len(stack) < 2:
                continue
            if all(id(box) in kept_box_memory_ids for box in stack):
                continue
            stack_top = min(box.bounding_box[1] for box in stack)
            stack_bottom = max(box.bounding_box[3] for box in stack)
            stack_center = (stack_top + stack_bottom) / 2.0
            if abs(stack_center - band_center) <= band_tolerance:
                for box in stack:
                    if id(box) not in kept_box_memory_ids:
                        newly_rescued_ids.add(id(box))

    kept_box_memory_ids.update(newly_rescued_ids)


def filter_cross_frame_spatial_outliers(
    frames_sequence: Sequence[OcrFrameResult],
    roi: Roi | None = None,
) -> list[OcrFrameResult]:
    """Lọc box có Y-center là cross-frame outlier so với median toàn cục.

    Vấn đề: Khi video có **nhiều loại phụ đề ở các Y khác nhau** (90% Y=74
    cho phụ đề thường, 10% Y=110 cho phụ đề 2-3 ký tự render thấp hơn),
    MAD cực nhỏ có thể lọc nhầm cụm phụ đề thiểu số. Giải pháp:
    nếu có ``roi``, đặt ``threshold_distance`` tối thiểu = ROI height / 2.

    Args:
        frames_sequence: Danh sách frame OCR đã qua per-frame spatial cleanup.
        roi: ROI dùng để định biên trên. Nếu None, dùng hành vi MAD-only.

    Returns:
        Danh sách frame đã purified. Frame nào hết text sau filter
        sẽ bị loại.
    """
    frames_list = list(frames_sequence)
    if len(frames_list) < 4:
        return frames_list

    aggregated_spatial_boxes: list[tuple[OcrFrameResult, OcrTextBox, float]] = []
    for current_frame in frames_list:
        for text_box in current_frame.text_boxes:
            if text_box.bounding_box:
                calculated_y_center = (
                    text_box.bounding_box[1] + text_box.bounding_box[3]
                ) / 2.0
                aggregated_spatial_boxes.append(
                    (current_frame, text_box, calculated_y_center)
                )

    if len(aggregated_spatial_boxes) < 6:
        return frames_list

    extracted_y_centers = [
        spatial_tuple[2] for spatial_tuple in aggregated_spatial_boxes
    ]
    roi_height_floor = float(roi.height) / 2.0 if roi is not None else 0.0
    keep_spatial_mask = filter_y_position_outliers(
        extracted_y_centers,
        k=4.0,
        minimum_threshold_distance=roi_height_floor,
    )

    kept_box_memory_ids: set[int] = set()
    kept_y_centers: list[float] = []
    for index, should_keep in enumerate(keep_spatial_mask):
        if should_keep:
            kept_box_memory_ids.add(id(aggregated_spatial_boxes[index][1]))
            kept_y_centers.append(aggregated_spatial_boxes[index][2])

    # [v3.17] Rescue phụ đề nhiều dòng bị density-Y xoá nhầm.
    # Bộ lọc density-Y dựa trên SỐ LƯỢNG box từng dòng: phụ đề HAI DÒNG có dòng-1
    # NẰM TRÊN và dòng-2 NẰM DƯỚI tâm băng đơn-dòng, nên KHÔNG dòng nào rơi vào
    # băng dày đặc → cả hai bị xoá → mất cả cue. Ta khôi phục một *chồng dọc*
    # (các box chồng trục X, kề sát trục Y trong cùng frame) nếu **tâm dọc tổ hợp**
    # của chồng rơi vào băng Y dày đặc (nơi text phụ đề thực sự nằm). Nhiễu rời rạc
    # ở rìa khung có tâm tổ hợp xa băng chính nên KHÔNG bị cứu nhầm.
    if kept_y_centers:
        _rescue_wrapped_multiline_boxes(
            frames_list, kept_box_memory_ids, kept_y_centers, float(roi.height) if roi else 0.0
        )

    purified_frames_sequence: list[OcrFrameResult] = []
    for current_frame in frames_list:
        surviving_boxes = [
            box
            for box in current_frame.text_boxes
            if not box.polygon or id(box) in kept_box_memory_ids
        ]
        if any(box.text.strip() for box in surviving_boxes):
            purified_frames_sequence.append(
                dataclasses.replace(current_frame, text_boxes=surviving_boxes)
            )

    return purified_frames_sequence if purified_frames_sequence else frames_list


def filter_noise_by_confidence(
    frames_list: Sequence[OcrFrameResult],
    minimum_confidence_threshold: float,
) -> list[OcrFrameResult]:
    """Lọc các frame có confidence trung bình thấp.

    Bỏ qua các frame rỗng hoặc có mean_confidence dưới ngưỡng. Tuy nhiên
    nếu box đầu tiên có confidence >= 0.75 × ngưỡng thì vẫn giữ (case
    khi 1 box conf cao + 1 box conf thấp kéo trung bình xuống).

    Args:
        frames_list: Chuỗi frame OCR.
        minimum_confidence_threshold: Ngưỡng confidence tối thiểu.

    Returns:
        Danh sách frame đã sort theo timestamp và lọc theo confidence.
    """
    highly_confident_results: list[OcrFrameResult] = []

    for current_frame in frames_list:
        if current_frame.is_empty:
            continue

        calculated_mean_conf = float(current_frame.mean_confidence)
        if calculated_mean_conf >= minimum_confidence_threshold or (
            len(current_frame.text_boxes) > 0
            and float(current_frame.text_boxes[0].confidence)
            >= (minimum_confidence_threshold * 0.75)
        ):
            highly_confident_results.append(current_frame)

    highly_confident_results.sort(key=lambda frame_result: frame_result.timestamp_sec)
    return highly_confident_results


__all__ = [
    "clean_spatial_outliers",
    "filter_cross_frame_spatial_outliers",
    "filter_noise_by_confidence",
]
