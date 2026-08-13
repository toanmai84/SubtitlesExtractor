"""Lập kế hoạch trích xuất HÀNG LOẠT nhiều tập (thuần, không phụ thuộc giao diện).

VÌ SAO cần
==========
Ứng dụng phục vụ phim bộ CJK — mỗi bộ hàng chục tập, vị trí phụ đề gần như giống hệt
nhau. Nhưng trang Trích xuất chỉ mở được **một video mỗi lần**
(``QFileDialog.getOpenFileName`` số ít), nên người dùng phải lặp lại toàn bộ thao tác
(chọn tệp → vẽ ROI → đặt tham số → chạy) cho từng tập.

Module này lập kế hoạch chạy nhiều tập với **cùng một ROI và cùng bộ tham số**.

CÁI BẪY QUAN TRỌNG: ROI là toạ độ TUYỆT ĐỐI
-------------------------------------------
:class:`Roi` lưu ``x/y/width/height`` bằng điểm ảnh. Nếu các tập khác độ phân giải
(hay gặp khi trộn nguồn tải về), dùng lại ROI nguyên xi sẽ **cắt sai vùng** — OCR ra
rác mà không báo lỗi. Module này tự **co giãn ROI theo tỉ lệ** khi độ phân giải khác,
và đánh dấu rõ những mục đã phải co giãn để người dùng kiểm lại.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from subtitles_extractor.domain.value_objects.output_naming import (
    SubtitleFormat,
    extracted_subtitle_path,
)
from subtitles_extractor.domain.value_objects.roi import Roi

logger = logging.getLogger(__name__)


class BatchItemStatus(StrEnum):
    """Trạng thái của một tập trong hàng đợi."""

    READY = "ready"
    """Sẵn sàng chạy."""

    ROI_SCALED = "roi_scaled"
    """Sẵn sàng, nhưng ROI đã bị co giãn vì độ phân giải khác — nên kiểm lại."""

    SKIPPED_EXISTS = "skipped_exists"
    """Bỏ qua vì tệp phụ đề đích đã tồn tại."""

    INVALID = "invalid"
    """Không dùng được (tệp không tồn tại, ROI vượt khung hình…)."""


@dataclass(frozen=True, slots=True)
class BatchItem:
    """Một mục trong kế hoạch trích xuất hàng loạt.

    Attributes:
        video_path: Video sẽ xử lý.
        output_path: Tệp phụ đề sẽ ghi ra.
        roi: ROI áp dụng cho video này (đã co giãn nếu cần).
        status: Trạng thái mục.
        note: Ghi chú hiển thị cho người dùng (``None`` nếu không có).
    """

    video_path: Path
    output_path: Path
    roi: Roi | None
    status: BatchItemStatus
    note: str | None = None

    @property
    def will_run(self) -> bool:
        """``True`` nếu mục này sẽ được chạy."""
        return self.status in (BatchItemStatus.READY, BatchItemStatus.ROI_SCALED)


def scale_roi(
    roi: Roi,
    *,
    from_size: tuple[int, int],
    to_size: tuple[int, int],
) -> Roi | None:
    """Co giãn ROI từ độ phân giải này sang độ phân giải khác.

    Args:
        roi: ROI gốc (toạ độ tuyệt đối theo ``from_size``).
        from_size: ``(rộng, cao)`` của video đã vẽ ROI.
        to_size: ``(rộng, cao)`` của video đích.

    Returns:
        ROI đã co giãn và kẹp trong khung hình đích; ``None`` nếu tham số không hợp lệ
        hoặc kết quả suy biến (rộng/cao < 1 điểm ảnh).
    """
    from_width, from_height = from_size
    to_width, to_height = to_size
    if min(from_width, from_height, to_width, to_height) <= 0:
        return None

    scale_x = to_width / from_width
    scale_y = to_height / from_height

    x = int(round(roi.x * scale_x))
    y = int(round(roi.y * scale_y))
    width = int(round(roi.width * scale_x))
    height = int(round(roi.height * scale_y))

    # Kẹp vào khung hình đích, giữ tối thiểu 1 điểm ảnh mỗi chiều.
    x = max(0, min(x, to_width - 1))
    y = max(0, min(y, to_height - 1))
    width = max(1, min(width, to_width - x))
    height = max(1, min(height, to_height - y))

    try:
        return Roi(
            x=x, y=y, width=width, height=height,
            alignment=roi.alignment, orientation=roi.orientation,
        )
    except Exception as exc:  # noqa: BLE001 — Roi tự kiểm tra, hỏng thì bỏ mục này
        logger.debug("Không co giãn được ROI sang %s: %s.", to_size, exc)
        return None


def roi_fits_in(roi: Roi, size: tuple[int, int]) -> bool:
    """``True`` nếu ROI nằm trọn trong khung hình ``size``."""
    width, height = size
    return (
        roi.x >= 0
        and roi.y >= 0
        and roi.x + roi.width <= width
        and roi.y + roi.height <= height
    )


def build_batch_plan(
    videos: list[Path],
    *,
    reference_roi: Roi | None,
    reference_size: tuple[int, int] | None,
    video_sizes: dict[Path, tuple[int, int]] | None = None,
    output_format: SubtitleFormat = SubtitleFormat.SRT,
    skip_existing: bool = True,
) -> list[BatchItem]:
    """Lập kế hoạch chạy trích xuất cho danh sách video.

    Args:
        videos: Danh sách video cần xử lý (theo thứ tự người dùng chọn).
        reference_roi: ROI đã vẽ trên video mẫu; ``None`` = để hệ thống tự dò từng tập.
        reference_size: ``(rộng, cao)`` của video mẫu — bắt buộc khi có ``reference_roi``.
        video_sizes: Kích thước từng video (nếu biết trước). Video không có trong dict
            sẽ được coi là CÙNG kích thước với video mẫu.
        output_format: Định dạng phụ đề đầu ra.
        skip_existing: Bỏ qua video đã có tệp phụ đề đích (tránh chạy lại tốn thời gian).

    Returns:
        Danh sách :class:`BatchItem` theo đúng thứ tự đầu vào.
    """
    sizes = video_sizes or {}
    plan: list[BatchItem] = []

    for video in videos:
        output = extracted_subtitle_path(video, output_format)

        if not video.is_file():
            plan.append(
                BatchItem(video, output, None, BatchItemStatus.INVALID,
                          "Không tìm thấy tệp video.")
            )
            continue

        if skip_existing and output.is_file():
            plan.append(
                BatchItem(video, output, reference_roi, BatchItemStatus.SKIPPED_EXISTS,
                          f"Đã có {output.name} — bỏ qua.")
            )
            continue

        # Không có ROI mẫu -> để pipeline tự dò ROI cho từng tập.
        if reference_roi is None or reference_size is None:
            plan.append(BatchItem(video, output, None, BatchItemStatus.READY,
                                  "Tự dò ROI."))
            continue

        target_size = sizes.get(video)
        if target_size is None or target_size == reference_size:
            plan.append(BatchItem(video, output, reference_roi, BatchItemStatus.READY))
            continue

        scaled = scale_roi(reference_roi, from_size=reference_size, to_size=target_size)
        if scaled is None:
            plan.append(
                BatchItem(video, output, None, BatchItemStatus.INVALID,
                          f"Không co giãn được ROI sang {target_size[0]}×{target_size[1]}.")
            )
            continue

        plan.append(
            BatchItem(
                video, output, scaled, BatchItemStatus.ROI_SCALED,
                f"ROI đã co giãn {reference_size[0]}×{reference_size[1]} → "
                f"{target_size[0]}×{target_size[1]} — nên kiểm lại.",
            )
        )

    return plan


def summarise_plan(plan: list[BatchItem]) -> str:
    """Tóm tắt kế hoạch thành một dòng để hiển thị.

    Args:
        plan: Kế hoạch đã lập.

    Returns:
        Chuỗi dạng ``"12 tập sẽ chạy · 2 bỏ qua (đã có) · 1 lỗi"``.
    """
    running = sum(1 for item in plan if item.will_run)
    skipped = sum(1 for item in plan if item.status is BatchItemStatus.SKIPPED_EXISTS)
    invalid = sum(1 for item in plan if item.status is BatchItemStatus.INVALID)
    scaled = sum(1 for item in plan if item.status is BatchItemStatus.ROI_SCALED)

    parts = [f"{running} tập sẽ chạy"]
    if scaled:
        parts.append(f"{scaled} đã co giãn ROI")
    if skipped:
        parts.append(f"{skipped} bỏ qua (đã có)")
    if invalid:
        parts.append(f"{invalid} lỗi")
    return " · ".join(parts)


__all__ = [
    "BatchItem",
    "BatchItemStatus",
    "build_batch_plan",
    "roi_fits_in",
    "scale_roi",
    "summarise_plan",
]
