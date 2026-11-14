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
import itertools
import re
from collections.abc import Sequence

from subtitles_extractor.application.dtos.extract_subtitles_dto import (
    SubtitleBuilderConfig,
)
from subtitles_extractor.application.services.subtitle_pipeline.box_geometry import typical_char_pitch
from subtitles_extractor.application.services.outlier_detection import (
    filter_y_position_outliers,
)
from subtitles_extractor.domain.entities.ocr_frame_result import (
    OcrFrameResult,
    OcrTextBox,
)
from subtitles_extractor.domain.value_objects.roi import Roi, TextAlignment, TextOrientation


#: [v3.23.710] Đo tâm phụ đề của phim: cần chừng này hộp ≥ 3 chữ tin cậy cao, và 90% trong số đó nằm sát tâm đo được.
CENTER_MIN_BOXES: int = 200
CENTER_MIN_CONFIDENCE: float = 0.90
CENTER_MAX_SPREAD_HEIGHTS: float = 0.50


def measure_subtitle_center(frames: Sequence[OcrFrameResult], roi: Roi | None) -> float | None:
    """[v3.23.710] Tâm x THẬT của phụ đề căn giữa, đo từ chính các hộp chữ của phim.

    Bộ lọc căn giữa từng so với tâm VÙNG CHỮ (``roi.width / 2``). Vùng chữ do người dùng vẽ tay hiếm khi đối xứng
    quanh phụ đề, và dữ liệu OCR thô của các bản trước 3.23.710 có toạ độ hộp bị co (xem ``PaddleOcrAdapter``) —
    trên phim mẫu tâm thật là 290 trong khi tâm vùng chữ là 322: câu một chữ tin cậy thấp (dung sai 26 px) bị loại
    oan dù nằm đúng giữa. Trả ``None`` (dùng tâm vùng chữ như cũ) khi phụ đề không căn giữa hoặc chưa đủ dữ liệu.
    """
    if (
        roi is None or roi.alignment not in (TextAlignment.CENTER, TextAlignment.UNKNOWN)
        or roi.orientation != TextOrientation.HORIZONTAL      # phụ đề dọc: tâm x là vị trí cột chữ, giữ luật cũ
    ):
        return None
    centers: list[float] = []
    heights: list[float] = []
    for frame in frames:
        boxes = [box for box in frame.text_boxes if box.polygon]
        if len(boxes) != 1:
            continue                       # khung nhiều hộp: hộp lẻ là mảnh của câu, tâm không đại diện
        box = boxes[0]
        if len(box.text.strip()) < 3 or float(box.confidence) < CENTER_MIN_CONFIDENCE:
            continue
        xs = [point[0] for point in box.polygon]
        ys = [point[1] for point in box.polygon]
        centers.append((min(xs) + max(xs)) / 2.0)
        heights.append(float(max(ys) - min(ys)))
    if len(centers) < CENTER_MIN_BOXES:
        return None
    centers.sort()
    heights.sort()
    center, height = centers[len(centers) // 2], heights[len(heights) // 2]
    deviations = sorted(abs(value - center) for value in centers)
    if height <= 0 or deviations[int(0.9 * (len(deviations) - 1))] > CENTER_MAX_SPREAD_HEIGHTS * height:
        return None
    return center


#: [v3.23.710] Hộp rìa tin cậy thấp kéo lệch tâm một dòng vốn đúng giữa → bỏ hộp rìa đó.
WEAK_EDGE_MAX_CONFIDENCE: float = 0.60
WEAK_EDGE_CENTERED_HEIGHTS: float = 0.25
WEAK_EDGE_OFF_CENTER_HEIGHTS: float = 0.50

_SpatialTuple = tuple[float, float, float, float, OcrTextBox]


def _strip_weak_edge_boxes(line: list[_SpatialTuple], center_x: float) -> list[_SpatialTuple]:
    """[v3.23.710] Bỏ hộp RÌA tin cậy thấp khi phần còn lại nằm đúng tâm phụ đề còn cả dòng thì lệch.

    Phim mẫu 3122 s: "元婴大圆满" (0,97, đúng tâm) + hộp vật trong cảnh "小川" (0,49) bên phải → cả dòng lệch 50 px
    nhưng vẫn trong dung sai câu dài nên ra "元婴大圆满小川". Chỉ xét hộp rìa trái/phải, mỗi lần một hộp.
    """
    ordered = sorted(line, key=lambda spatial_tuple: spatial_tuple[2])

    def offset(boxes: list[_SpatialTuple]) -> float:
        return abs((min(b[2] for b in boxes) + max(b[3] for b in boxes)) / 2.0 - center_x)

    while len(ordered) >= 2:
        height = max(1.0, max(b[1] for b in ordered) - min(b[0] for b in ordered))
        if offset(ordered) <= WEAK_EDGE_OFF_CENTER_HEIGHTS * height:
            break
        for edge, rest in ((ordered[0], ordered[1:]), (ordered[-1], ordered[:-1])):
            if (
                float(edge[4].confidence) < WEAK_EDGE_MAX_CONFIDENCE
                and min(float(b[4].confidence) for b in rest) >= WEAK_EDGE_MAX_CONFIDENCE
                and offset(rest) <= WEAK_EDGE_CENTERED_HEIGHTS * height
            ):
                ordered = rest
                break
        else:
            break
    return ordered


#: [v3.23.718] Hộp chữ y hệt (cùng chữ, cùng chỗ) hiện LIÊN TỤC lâu hơn chừng này giây là lớp phủ tĩnh (logo kênh, tên
#: tài khoản), không phải phụ đề. Đo trên 4 bộ dữ liệu thô thật (≈ 208 000 khung mỗi bộ): lượt liên tục dài nhất của
#: một câu thật là 4,9 s ("圣子选拔战现在开赛"); thời lượng tối đa một câu của bộ dựng là 7 s.
STATIC_OVERLAY_MIN_RUN_SEC: float = 10.0
#: Lượt "liên tục" cho phép hở tối đa chừng này lần bước lấy mẫu (OCR hụt / khung mất).
STATIC_OVERLAY_MAX_GAP_STEPS: float = 3.0
#: Hộp cùng chỗ (IoU ≥ ngưỡng) và chữ giống (≥ ngưỡng) với lớp phủ cũng bị gỡ — lần đọc lệch của chính lớp phủ.
STATIC_OVERLAY_IOU: float = 0.7
STATIC_OVERLAY_TEXT_RATIO: float = 0.6


def _box_iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    width = min(left[2], right[2]) - max(left[0], right[0])
    height = min(left[3], right[3]) - max(left[1], right[1])
    if width <= 0 or height <= 0:
        return 0.0
    inter = width * height
    union = (left[2] - left[0]) * (left[3] - left[1]) + (right[2] - right[0]) * (right[3] - right[1]) - inter
    return inter / union if union > 0 else 0.0


def strip_static_overlays(
    frames: Sequence[OcrFrameResult], sample_step_sec: float, max_line_sec: float = 7.0,
) -> tuple[list[OcrFrameResult], list[str]]:
    """[v3.23.718] Gỡ lớp phủ tĩnh (logo kênh / tài khoản cố định trong vùng chữ) khỏi MỌI khung trước khi dựng.

    Thử sức trên OCR thật (thêm "@DuanJu88" cố định ở góc vùng chữ vào mọi khung): mất 6 câu một chữ ("是", "打", "好",
    "你", "那") và 12 câu lệch mốc — các luật hộp phình / dải phụ đề / gộp lần đọc chỉ xét khung có MỘT hộp nên hộp logo
    làm chúng tắt hết. Bộ lọc câu watermark có sẵn chạy ở cuối khâu dựng, quá muộn.

    Trả ``(khung, chữ của các lớp phủ đã gỡ)``. Không gỡ gì khi không biết bước lấy mẫu.
    """
    if sample_step_sec <= 0 or len(frames) < 2:
        return list(frames), []
    min_run = max(STATIC_OVERLAY_MIN_RUN_SEC, 1.5 * max_line_sec)
    max_gap = STATIC_OVERLAY_MAX_GAP_STEPS * sample_step_sec
    quantum = 8.0
    run_start: dict[tuple[str, int, int, int, int], float] = {}
    last_seen: dict[tuple[str, int, int, int, int], float] = {}
    overlays: dict[tuple[str, int, int, int, int], tuple[float, float, float, float]] = {}
    for frame in frames:
        for box in frame.text_boxes:
            bbox = box.bounding_box
            if bbox is None or not box.text.strip():
                continue
            x0, y0, x1, y1 = bbox
            key = (
                box.text.strip(), round(x0 / quantum), round(y0 / quantum), round(x1 / quantum), round(y1 / quantum),
            )
            if key in last_seen and frame.timestamp_sec - last_seen[key] <= max_gap:
                if frame.timestamp_sec - run_start[key] >= min_run:
                    overlays[key] = (float(x0), float(y0), float(x1), float(y1))
            else:
                run_start[key] = frame.timestamp_sec
            last_seen[key] = frame.timestamp_sec
    if not overlays:
        return list(frames), []
    from rapidfuzz import fuzz

    targets = [(key[0], bbox) for key, bbox in overlays.items()]

    def is_overlay(box: OcrTextBox) -> bool:
        bbox = box.bounding_box
        if bbox is None:
            return False
        text = box.text.strip()
        box_xyxy = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        return any(
            _box_iou(box_xyxy, target_box) >= STATIC_OVERLAY_IOU
            and (text == target_text or fuzz.ratio(text, target_text) >= 100 * STATIC_OVERLAY_TEXT_RATIO)
            for target_text, target_box in targets
        )

    result: list[OcrFrameResult] = []
    for frame in frames:
        kept = [box for box in frame.text_boxes if not is_overlay(box)]
        result.append(frame if len(kept) == len(frame.text_boxes) else dataclasses.replace(frame, text_boxes=kept))
    return result, sorted({key[0] for key in overlays})


#: [v3.23.717] Hộp CHÍNH (nhiều chữ nhất) tự nó đã đúng tâm phụ đề trong chừng này lần chiều cao hộp → nó là cả dòng.
#: Đo trên hai bộ dữ liệu thô (≈ 106 500 hộp câu thật mỗi bộ): lệch tâm trung vị 1 px, 99 % ≤ 5 px, 99,9 % ≤ 9,5 px
#: (cỡ chữ ≈ 67 px → 0,14).
MAIN_BOX_CENTERED_HEIGHTS: float = 0.12
MAIN_BOX_MIN_CONFIDENCE: float = 0.9
CONTAINED_BOX_OVERLAP: float = 0.6


def _keep_centered_main_box(line: list[_SpatialTuple], center_x: float) -> list[_SpatialTuple]:
    """[v3.23.717] Dòng nhiều hộp mà hộp chính tự nó đã đúng tâm, còn cả dòng thì lệch → chỉ giữ hộp chính.

    Phụ đề căn giữa bị tách hai hộp (có dấu cách) thì KHÔNG hộp nào tự đúng tâm (mỗi hộp lệch ít nhất nửa hộp kia), nên
    luật không đụng tới. Hộp vật trong cảnh nằm cạnh câu (kể cả tin cậy cao — ``_strip_weak_edge_boxes`` chỉ bỏ hộp
    tin cậy < 0,6) thì kéo tâm cả dòng lệch đi: thử sức trên OCR thật (thêm 1–2 hộp nhiễu vào 5 % lần đọc) ra 9 câu
    dính chữ thừa ("三台上还剩一人") kèm mốc lệch 0,5 s vì lần đọc dính nhiễu bị tách thành câu riêng.
    """
    if len(line) < 2:
        return line

    def chars(item: _SpatialTuple) -> int:
        return sum(1 for char in item[4].text if char.isalnum())

    main = max(line, key=lambda item: (chars(item), float(item[4].confidence)))
    if float(main[4].confidence) < MAIN_BOX_MIN_CONFIDENCE or chars(main) < 1:
        return line
    # Hộp ít chữ hơn nằm CHỒNG trong bề ngang hộp chính (≥ 60 % bề rộng của nó): một dòng ngang không thể có hai hộp
    # chồng nhau — ba bộ dữ liệu thô thật (≈ 208 000 khung mỗi bộ) không có ca nào, nên đây chỉ có thể là vật trong cảnh.
    line = [
        item for item in line
        if item is main or chars(item) >= chars(main)
        or min(main[3], item[3]) - max(main[2], item[2]) < CONTAINED_BOX_OVERLAP * max(1.0, item[3] - item[2])
    ]
    if len(line) < 2:
        return line
    if sum(1 for item in line if chars(item) == chars(main)) > 1:
        return line                                  # hai hộp ngang cỡ: không biết hộp nào là câu
    tolerance = MAIN_BOX_CENTERED_HEIGHTS * max(1.0, main[1] - main[0])
    main_offset = abs((main[2] + main[3]) / 2.0 - center_x)
    line_offset = abs((min(item[2] for item in line) + max(item[3] for item in line)) / 2.0 - center_x)
    if main_offset <= tolerance and line_offset > main_offset + tolerance:
        return [main]
    return line


#: [v3.23.716] Khe trống tối đa giữa hai hộp kề nhau của một dòng căn giữa, tính theo chiều cao hộp cao nhất của dòng.
MAX_INNER_GAP_HEIGHTS = 3.0


def clean_spatial_outliers(
    frame_result: OcrFrameResult,
    roi: Roi | None,
    config: SubtitleBuilderConfig,
    center_x: float | None = None,
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
        center_x: [v3.23.710] Tâm phụ đề đo từ phim; ``None`` = tâm vùng chữ.

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
    # [v3.23.710] Tâm phụ đề đo từ phim (``measure_subtitle_center``); không có thì tâm vùng chữ như cũ.
    subtitle_center_x = center_x if center_x is not None else roi_width_float_value / 2.0

    def _line_is_valid(text_line_cluster: list[tuple[float, float, float, float, OcrTextBox]]) -> bool:
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
        # [v3.23.716] Một dòng phụ đề không có khe trống rộng giữa các hộp (đo thật: khe lớn nhất giữa hai hộp của cùng
        # một câu là 0,15 lần cỡ chữ). Hộp ở hai mép cảnh cách nhau > 3 lần chiều cao hộp ("C气" mép trái + "下" mép
        # phải, khe 473 px) từng tạo "dòng" có tâm đúng giữa → qua kiểm căn lề.
        ordered_by_x = sorted(text_line_cluster, key=lambda spatial_tuple: spatial_tuple[2])
        widest_inner_gap = max(
            (right[2] - left[3] for left, right in itertools.pairwise(ordered_by_x)), default=0.0,
        )
        tallest_box = max(spatial_tuple[1] - spatial_tuple[0] for spatial_tuple in text_line_cluster)
        if total_line_width > roi_width_float_value * 1.02 or (
            active_alignment == TextAlignment.CENTER and widest_inner_gap > MAX_INNER_GAP_HEIGHTS * tallest_box
        ):
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

                if abs(line_x_center_point - subtitle_center_x) > dynamic_allowed_deviation:
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

        return is_text_line_valid

    for text_line_cluster in clustered_text_lines:
        if _line_is_valid(text_line_cluster):
            if center_x is not None and len(text_line_cluster) >= 2:
                text_line_cluster = _strip_weak_edge_boxes(text_line_cluster, center_x)
                if active_alignment == TextAlignment.CENTER:
                    text_line_cluster = _keep_centered_main_box(text_line_cluster, center_x)
            verified_text_boxes.extend(spatial_tuple[4] for spatial_tuple in text_line_cluster)
            continue
        # [v3.23.701] Dòng gộp nhiều hộp mà lệch tâm: hộp vật thể trong cảnh chồng dọc với phụ đề bị gom chung một
        # dòng (vd "二" của hoa văn + "叶然" đúng giữa → "二叶然" lệch 83 px → mất câu). Thử các dãy hộp liền nhau
        # theo trục x, giữ dãy hợp lệ có nhiều chữ nhất — chỉ CỨU thêm, không loại thêm hộp nào.
        if len(text_line_cluster) < 2:
            continue
        ordered = sorted(text_line_cluster, key=lambda spatial_tuple: spatial_tuple[2])
        best_subset: list[tuple[float, float, float, float, OcrTextBox]] = []
        best_key = (0, 0.0)
        for first in range(len(ordered)):
            for last in range(first + 1, len(ordered) + 1):
                subset = ordered[first:last]
                if len(subset) == len(ordered):
                    continue
                chars = len("".join(spatial_tuple[4].text for spatial_tuple in subset).strip())
                conf = sum(float(spatial_tuple[4].confidence) for spatial_tuple in subset) / len(subset)
                if chars and (chars, conf) > best_key and _line_is_valid(subset):
                    best_subset, best_key = subset, (chars, conf)
        verified_text_boxes.extend(spatial_tuple[4] for spatial_tuple in best_subset)

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


#: [v3.23.719] Cỡ chữ mà các hằng số pixel của bộ lọc mật độ Y (bin 5 px…) được tinh chỉnh, đo bằng BỀ RỘNG MỘT CHỮ
#: (:func:`typical_char_pitch`): bộ 86 tập 720×1280 ≈ 52–58 px (chữ cao 60–68 px), 豪门Z外 1080 px ≈ 73 px (cao 86 px).
#: Đo bằng bề rộng chứ không bằng chiều cao: vùng chữ vẽ hụt cắt chân chữ thì chiều cao co lại (67 → 46 px, thử sức
#: ``clip:25``) mà bề rộng không đổi (58,2 px) — đo theo chiều cao làm dải giữ hẹp lại và mất lần đọc "哼".
DENSITY_REFERENCE_CHAR_PITCH_PX = 58.3
#: Chiều cao hộp / bề rộng một chữ trên dữ liệu không bị cắt (đo 7 bộ thật: 1,16–1,18) — dùng khi không đo được bề rộng.
_HEIGHT_PER_PITCH = 1.17


def _density_pixel_scale(frames: Sequence[OcrFrameResult], box_heights: Sequence[float]) -> float:
    """[v3.23.719] Hệ số đổi hằng số pixel của bộ lọc mật độ Y theo cỡ chữ phim (1.0 = cỡ chữ tinh chỉnh)."""
    pitch = typical_char_pitch(frames)
    if pitch is None or pitch <= 0:
        ordered = sorted(box_heights)
        median_height = ordered[len(ordered) // 2] if ordered else 0.0
        pitch = median_height / _HEIGHT_PER_PITCH
    return pitch / DENSITY_REFERENCE_CHAR_PITCH_PX if pitch > 0 else 1.0


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
    # [v3.23.719] Bộ lọc mật độ Y tính theo CỠ CHỮ, không theo pixel: cùng một phim ở độ phân giải khác cho cùng kết quả.
    box_heights = [
        float(bbox[3] - bbox[1])
        for spatial_tuple in aggregated_spatial_boxes
        if (bbox := spatial_tuple[1].bounding_box) is not None
    ]
    keep_spatial_mask = filter_y_position_outliers(
        extracted_y_centers,
        k=4.0,
        minimum_threshold_distance=roi_height_floor,
        pixel_scale=_density_pixel_scale(frames_list, box_heights),
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

    _rescue_stable_offband_lines(frames_list, aggregated_spatial_boxes, kept_box_memory_ids)

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


#: [v3.23.701] Dòng phụ đề đặt lệch vị trí thường lệ (vd câu cuối phim nằm cao hơn) được cứu khi đủ mọi điều kiện:
OFFBAND_MIN_FRAMES = 8
OFFBAND_MIN_SPAN_SEC = 0.5
OFFBAND_MIN_CONFIDENCE = 0.9
OFFBAND_MAX_FRAME_GAP_SEC = 0.3
OFFBAND_HEIGHT_RATIO = (0.7, 1.4)
_CJK_PATTERN = re.compile(r"[\u3400-\u9fff]")


def _rescue_stable_offband_lines(
    frames_list: Sequence[OcrFrameResult],
    aggregated_spatial_boxes: Sequence[tuple[OcrFrameResult, OcrTextBox, float]],
    kept_box_memory_ids: set[int],
) -> None:
    """[v3.23.701] Cứu dòng phụ đề THẬT nằm lệch vị trí thường lệ mà bộ lọc Y toàn phim loại nhầm.

    Ca thật (phim nối 86 tập): câu cuối "我真的是冤枉的" đặt cao hơn mọi câu khác (tâm y ≈ 806 so với ≈ 868); OCR
    đọc đúng ở 65 khung (tin cậy 1,0) nhưng bộ lọc Y toàn phim coi là ngoại lai → mất câu. Chỉ cứu khi: cùng một chữ
    hiện liên tục ≥ 0,5 s, tin cậy cao, ≥ 2 chữ Hán, hộp ngang cao xấp xỉ cỡ chữ phụ đề, và khung đó KHÔNG còn phụ đề
    nào khác (chữ trong cảnh đi kèm phụ đề thường lệ không được cứu).
    """
    kept_heights = sorted(
        box.bounding_box[3] - box.bounding_box[1]
        for _frame, box, _y in aggregated_spatial_boxes
        if id(box) in kept_box_memory_ids and box.bounding_box
    )
    if not kept_heights:
        return
    typical_height = kept_heights[len(kept_heights) // 2]
    frames_with_kept = {id(frame) for frame, box, _y in aggregated_spatial_boxes if id(box) in kept_box_memory_ids}
    runs: dict[str, list[list[tuple[OcrFrameResult, OcrTextBox]]]] = {}
    for frame, box, _y in aggregated_spatial_boxes:
        if id(box) in kept_box_memory_ids or id(frame) in frames_with_kept or not box.bounding_box:
            continue
        text = box.text.strip()
        x0, y0, x1, y1 = box.bounding_box
        height = y1 - y0
        if (
            len(_CJK_PATTERN.findall(text)) < 2
            or float(box.confidence) < OFFBAND_MIN_CONFIDENCE
            or not (OFFBAND_HEIGHT_RATIO[0] * typical_height <= height <= OFFBAND_HEIGHT_RATIO[1] * typical_height)
            or (x1 - x0) < 1.5 * height
        ):
            continue
        text_runs = runs.setdefault(text, [])
        if text_runs and frame.timestamp_sec - text_runs[-1][-1][0].timestamp_sec <= OFFBAND_MAX_FRAME_GAP_SEC:
            text_runs[-1].append((frame, box))
        else:
            text_runs.append([(frame, box)])
    for text_runs in runs.values():
        for run in text_runs:
            span = run[-1][0].timestamp_sec - run[0][0].timestamp_sec
            if len(run) >= OFFBAND_MIN_FRAMES and span >= OFFBAND_MIN_SPAN_SEC:
                kept_box_memory_ids.update(id(box) for _frame, box in run)


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
    "strip_static_overlays",
]
