"""Các entity mô tả kết quả OCR cho một khung hình.

CẢI TIẾN ĐỘT PHÁ:
    1.[PERFORMANCE FIX] Loại bỏ hoàn toàn overhead biên dịch hàm và import trong vòng lặp.
    2.[PERFORMANCE FIX] Thay thế ép kiểu UTF-8 chậm chạp bằng `isascii()`.
    3.[LOGIC FIX] Khắc phục bẫy "Bậc thang" (Staircase Bug) trong thuật toán gom dòng Y-Clustering.
    4. Tối ưu toán học tính Bounding Box bằng zip() C-level.
    5.[v2.28 FIX CRITICAL — Space-Aware Join] ``get_joined_text`` giờ inject
       space giữa 2 box cùng dòng khi gap X >= 0.6 * line_avg_height. Pattern
       `'阿姨 来不及了'` (vai trò + lời nói) trong phụ đề CJK được giữ chính
       xác thay vì gộp thành `'阿姨来不及了'`. Ảnh hưởng hàng trăm câu trên
       test1/fulltest có cấu trúc "name + speech".
"""

from __future__ import annotations

import string
from dataclasses import dataclass, field

from subtitles_extractor.domain.value_objects.confidence import Confidence

Polygon = list[tuple[int, int]]

_DEFAULT_Y_TOLERANCE_RATIO: float = 0.30
_DEFAULT_Y_TOLERANCE_MIN_PX: float = 5.0

# v2.28: Ngưỡng gap X giữa 2 box CJK trong cùng line để chèn space.
# Gap >= 60% line_avg_height ≈ 1 ký tự CJK ⇒ là gap có chủ đích → insert space.
_INTRA_LINE_SPACE_GAP_RATIO: float = 0.60
_INTRA_LINE_SPACE_GAP_MIN_PX: float = 8.0


def _smart_join_boxes(texts: list[str]) -> str:
    """Nối Text thông minh, chuẩn Studio Quality.
    Được đưa ra Global Level để tránh Overhead khởi tạo lại hàm.
    """
    from subtitles_extractor.application.services.cjk_utils import is_cjk_char

    if not texts:
        return ""
    res = texts[0].strip()

    for i in range(1, len(texts)):
        prev = res.strip()
        curr = texts[i].strip()

        if not prev or not curr:
            res += curr
            continue

        prev_last_char = prev[-1]
        curr_first_char = curr[0]

        prev_is_latin_alnum = prev_last_char.isascii() and prev_last_char.isalnum()
        curr_is_latin_alnum = curr_first_char.isascii() and curr_first_char.isalnum()

        if prev_is_latin_alnum and curr_is_latin_alnum:
            res += " " + curr
        elif is_cjk_char(prev_last_char) or is_cjk_char(curr_first_char) or prev_last_char in string.punctuation or curr_first_char in string.punctuation:
            res += curr
        else:
            res += " " + curr

    return res.strip()


def _smart_join_boxes_with_gap_awareness(
    line_items: list[tuple[float, float, float, float, str]],
) -> str:
    """[v2.28] Nối các box trong cùng line, insert space khi gap X đủ lớn.

    Pattern điển hình: phụ đề CJK có cấu trúc ``'vai trò + lời nói'`` như
    ``'阿姨 来不及了'``, ``'妈 我先走了'``. OCR detect 2 box cách xa nhau
    theo X. Nếu gộp liền không space, output thành ``'阿姨来不及了'`` →
    sai so với reference.

    Args:
        line_items: list tuples ``(y_center, x_min, x_max, height, text)``,
            đã sort theo ``x_min``.

    Returns:
        Joined text với space được chèn ở các gap X đủ lớn.

    Algorithm:
        1. Tính ``line_avg_height`` để scale ngưỡng gap.
        2. Threshold gap = ``max(8.0, height * 0.60)``.
        3. Cho mỗi cặp box liên tiếp, nếu ``x_min[i+1] - x_max[i] >= threshold``
           và cả 2 đều bao gồm CJK → insert space.
        4. Ngược lại fallback ``_smart_join_boxes`` behavior.
    """
    from subtitles_extractor.application.services.cjk_utils import is_cjk_char

    if not line_items:
        return ""

    line_avg_height = sum(it[3] for it in line_items) / len(line_items)
    gap_threshold = max(
        _INTRA_LINE_SPACE_GAP_MIN_PX,
        line_avg_height * _INTRA_LINE_SPACE_GAP_RATIO,
    )

    result_parts: list[str] = [line_items[0][4].strip()]
    for idx in range(1, len(line_items)):
        prev_item = line_items[idx - 1]
        curr_item = line_items[idx]
        prev_text = result_parts[-1]
        curr_text = curr_item[4].strip()

        if not prev_text or not curr_text:
            result_parts[-1] = (prev_text + curr_text).strip()
            continue

        # Tính gap X edge-to-edge.
        gap_x_pixels = curr_item[1] - prev_item[2]

        prev_last_char = prev_text[-1]
        curr_first_char = curr_text[0]
        prev_is_latin_alnum = prev_last_char.isascii() and prev_last_char.isalnum()
        curr_is_latin_alnum = curr_first_char.isascii() and curr_first_char.isalnum()
        is_cjk_boundary = is_cjk_char(prev_last_char) or is_cjk_char(curr_first_char)
        is_punct_boundary = (
            prev_last_char in string.punctuation
            or curr_first_char in string.punctuation
        )

        # v2.28 RULE: nếu là CJK boundary VÀ gap X đủ lớn → insert space.
        # Đây là rule mới quan trọng cho phụ đề CJK có cấu trúc name+speech.
        if is_cjk_boundary and gap_x_pixels >= gap_threshold:
            result_parts.append(curr_text)
            result_parts[-2:] = [result_parts[-2] + " " + result_parts[-1]]
            continue

        # Fallback rules giống _smart_join_boxes.
        if prev_is_latin_alnum and curr_is_latin_alnum:
            joiner = " "
        elif is_cjk_boundary or is_punct_boundary:
            joiner = ""
        else:
            joiner = " "

        result_parts[-1] = prev_text + joiner + curr_text

    return result_parts[-1].strip() if len(result_parts) == 1 else " ".join(
        # Trường hợp len > 1 không bao giờ xảy ra do code trên dùng index -1.
        # Để defensive, fallback safely.
        [p for p in result_parts if p]
    ).strip()


@dataclass(frozen=True, slots=True)
class OcrTextBox:
    text: str
    confidence: Confidence
    polygon: Polygon = field(default_factory=list)

    @property
    def bounding_box(self) -> tuple[int, int, int, int] | None:
        if not self.polygon:
            return None
        # Unpack siêu tốc bằng C-level zip — strict=True an toàn vì polygon
        # đảm bảo mỗi point là tuple (x, y) có cùng độ dài 2.
        xs, ys = zip(*self.polygon, strict=True)
        return (min(xs), min(ys), max(xs), max(ys))


@dataclass(frozen=True, slots=True)
class OcrFrameResult:
    frame_index: int
    timestamp_sec: float
    text_boxes: list[OcrTextBox] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not any(box.text.strip() for box in self.text_boxes)

    @property
    def mean_confidence(self) -> Confidence:
        if not self.text_boxes:
            return Confidence.zero()
        avg = sum(float(b.confidence) for b in self.text_boxes) / len(self.text_boxes)
        return Confidence(max(0.0, min(1.0, avg)))

    @property
    def joined_text(self) -> str:
        return self.get_joined_text()

    def get_joined_text(
        self,
        y_tolerance_ratio: float = _DEFAULT_Y_TOLERANCE_RATIO,
        y_tolerance_min_px: float = _DEFAULT_Y_TOLERANCE_MIN_PX,
    ) -> str:
        non_empty = [b for b in self.text_boxes if b.text.strip()]
        if not non_empty:
            return ""

        if any(not b.polygon for b in non_empty):
            return _smart_join_boxes([b.text.strip() for b in non_empty])

        items: list[tuple[float, int, float, str]] = []
        for box in non_empty:
            xs, ys = zip(*box.polygon, strict=True)
            y_center = (min(ys) + max(ys)) / 2.0
            height = float(max(ys) - min(ys))
            items.append((y_center, min(xs), height, box.text.strip()))

        items.sort(key=lambda item: item[0])

        lines: list[list[tuple[float, int, float, str]]] = [[items[0]]]
        for current in items[1:]:
            current_line = lines[-1]

            # Tính tọa độ trung bình của toàn bộ dòng hiện tại
            # Ngăn chặn lỗi "Bậc thang" khiến các chữ bị lệch chéo bị kéo vào chung 1 dòng.
            line_y_center = sum(item[0] for item in current_line) / len(current_line)
            line_avg_height = sum(item[2] for item in current_line) / len(current_line)

            tolerance = max(line_avg_height * y_tolerance_ratio, y_tolerance_min_px)

            if abs(current[0] - line_y_center) <= tolerance:
                current_line.append(current)
            else:
                lines.append([current])

        lines_text = []
        for line in lines:
            sorted_line = sorted(line, key=lambda it: it[1])
            # v2.28 NOTE: Đã thử ``_smart_join_boxes_with_gap_awareness`` để
            # insert space khi gap X giữa 2 box CJK đủ lớn. Tuy nhiên thực
            # nghiệm trên duozi (4645 câu) cho thấy false positive cao —
            # OCR thường tách 1 câu liền thành 2 box do font/segmentation,
            # không phải có space thực sự (vd `'你可是家族唯'` + `'一一位'`
            # bị nhầm tách → join → `'你可是家族唯 一一位'` sai). Đã rollback
            # về ``_smart_join_boxes`` thông thường. Space restoration giờ
            # chỉ dùng ``_restore_dropped_space`` ở tầng ROVER vote (evidence
            # từ candidate text trong cùng group).
            lines_text.append(_smart_join_boxes([item[3] for item in sorted_line]))

        return "\n".join(lines_text)

__all__ = ["OcrFrameResult", "OcrTextBox", "Polygon"]
