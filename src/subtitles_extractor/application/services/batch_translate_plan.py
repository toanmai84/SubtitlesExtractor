"""Lập kế hoạch DỊCH HÀNG LOẠT cho phim bộ (thuần, không phụ thuộc Qt).

VÌ SAO KHÂU NÀY KHÁC HẲN Trích xuất / TTS / Xuất bản
====================================================
Ba khâu kia chạy cục bộ nên cứ chạy tới hết. Khâu Dịch gọi **dịch vụ ngoài có hạn
mức**: ``gemini_quota_manager`` cho thấy giới hạn theo NGÀY (RPD) chỉ từ 20 đến 500
request tuỳ mô hình. Với bộ 84 tập, gần như chắc chắn **hết hạn mức giữa chừng**.

Nên kế hoạch dịch hàng loạt phải:

* **Ước lượng số request** trước khi chạy, cảnh báo nếu vượt hạn mức ngày.
* Khi hết hạn mức → **dừng êm** và nói rõ đã xong tới tập nào (không đốt hàng loạt lỗi).
* Chạy lại hôm sau phải **tự bỏ qua** các tập đã dịch — không tốn hạn mức lần nữa.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from subtitles_extractor.application.services.batch_publish_plan import (
    find_episode_videos,
)
from subtitles_extractor.domain.value_objects.output_naming import (
    extracted_subtitle_path,
    translated_subtitle_path,
)

logger = logging.getLogger(__name__)

#: Số câu ước lượng mỗi tập khi chưa đọc được tệp (mini-drama CJK thường 30–60 câu).
_ASSUMED_LINES_PER_EPISODE = 45


class TranslateItemStatus(StrEnum):
    """Trạng thái một tập trong kế hoạch dịch."""

    READY = "ready"
    """Có phụ đề gốc, sẵn sàng dịch."""

    MISSING_SOURCE = "missing_source"
    """Chưa có phụ đề gốc — cần chạy khâu Trích xuất trước."""

    ALREADY_DONE = "already_done"
    """Đã có bản dịch — bỏ qua để khỏi tốn hạn mức."""


@dataclass(frozen=True, slots=True)
class TranslateItem:
    """Một tập trong kế hoạch dịch hàng loạt.

    Attributes:
        video_path: Video của tập.
        source_path: Phụ đề gốc (``None`` nếu chưa có).
        output_path: Tệp bản dịch sẽ ghi ra.
        line_count: Số câu đọc được từ phụ đề gốc (``0`` nếu không đọc được).
        status: Trạng thái.
        note: Ghi chú cho người dùng.
    """

    video_path: Path
    source_path: Path | None
    output_path: Path
    line_count: int
    status: TranslateItemStatus
    note: str = ""

    @property
    def will_run(self) -> bool:
        """``True`` nếu tập này sẽ được dịch."""
        return self.status is TranslateItemStatus.READY


def count_subtitle_lines(path: Path) -> int:
    """Đếm số câu trong tệp phụ đề (SRT hoặc ASS).

    Dùng để ước lượng số request — đếm thô nhưng đủ chính xác cho việc cảnh báo.

    Args:
        path: Tệp phụ đề.

    Returns:
        Số câu; ``0`` nếu không đọc được.
    """
    try:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError as exc:
        logger.debug("Không đọc được %s: %s", path, exc)
        return 0

    if path.suffix.lower() in (".ass", ".ssa"):
        return sum(
            1 for line in text.splitlines() if line.strip().startswith("Dialogue:")
        )
    # SRT: mỗi câu có một dòng mốc thời gian chứa "-->".
    return sum(1 for line in text.splitlines() if "-->" in line)


def find_source_subtitle(video_path: Path) -> Path | None:
    """Tìm phụ đề GỐC (bản trích xuất/biên tập) của một tập.

    Args:
        video_path: Video của tập.

    Returns:
        Đường dẫn phụ đề gốc, hoặc ``None``.
    """
    preferred = extracted_subtitle_path(video_path)
    try:
        if preferred.is_file():
            return preferred
    except OSError:
        pass
    for suffix in (".ass", ".ssa"):
        candidate = preferred.with_suffix(suffix)
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            continue
    return None


def find_existing_translation(
    video_path: Path, target_language: str = "vi"
) -> Path | None:
    """Tìm bản dịch đã có của một tập.

    Quét mẫu ``*.translate.*.srt`` để không phụ thuộc việc biết trước mã ngôn ngữ —
    cùng bài học với v3.23.323.

    Args:
        video_path: Video của tập.
        target_language: Mã ngôn ngữ đích.

    Returns:
        Đường dẫn bản dịch, hoặc ``None``.
    """
    preferred = translated_subtitle_path(video_path, target_language or "vi")
    try:
        if preferred.is_file():
            return preferred
    except OSError:
        pass
    try:
        matches = sorted(preferred.parent.glob(f"{video_path.stem}.translate.*.srt"))
    except OSError:
        return None
    return matches[0] if matches else None


def build_translate_plan(
    videos: list[Path],
    *,
    target_language: str = "vi",
    skip_existing: bool = True,
) -> list[TranslateItem]:
    """Lập kế hoạch dịch cho danh sách tập.

    Args:
        videos: Các video nguồn.
        target_language: Mã ngôn ngữ đích.
        skip_existing: Bỏ qua tập đã có bản dịch (tránh tốn hạn mức lần nữa).

    Returns:
        Danh sách :class:`TranslateItem` theo đúng thứ tự đầu vào.
    """
    plan: list[TranslateItem] = []

    for video in videos:
        output = translated_subtitle_path(video, target_language or "vi")
        source = find_source_subtitle(video)

        if skip_existing:
            existing = find_existing_translation(video, target_language)
            if existing is not None:
                plan.append(
                    TranslateItem(video, source, existing, 0,
                                  TranslateItemStatus.ALREADY_DONE,
                                  f"Đã có {existing.name}")
                )
                continue

        if source is None:
            plan.append(
                TranslateItem(video, None, output, 0,
                              TranslateItemStatus.MISSING_SOURCE,
                              "Chưa có phụ đề gốc — chạy khâu Trích xuất trước.")
            )
            continue

        plan.append(
            TranslateItem(video, source, output, count_subtitle_lines(source),
                          TranslateItemStatus.READY)
        )

    return plan


def estimate_requests(plan: list[TranslateItem], batch_size: int = 40) -> int:
    """Ước lượng tổng số request API cần cho cả lô.

    Dịch chia câu thành lô; mỗi lô là một request. Tập không đọc được số câu thì dùng
    ước lượng mặc định để cảnh báo vẫn có ý nghĩa.

    Args:
        plan: Kế hoạch đã lập.
        batch_size: Số câu mỗi request (khớp ô "Kích thước lô" của trang Dịch).

    Returns:
        Tổng số request ước tính.
    """
    size = max(1, batch_size)
    total = 0
    for item in plan:
        if not item.will_run:
            continue
        lines = item.line_count or _ASSUMED_LINES_PER_EPISODE
        total += math.ceil(lines / size)
    return total


def quota_warning(
    estimated_requests: int, daily_limit: int | None
) -> str | None:
    """Cảnh báo nếu lô dự kiến vượt hạn mức ngày.

    Args:
        estimated_requests: Số request ước tính.
        daily_limit: Giới hạn RPD của mô hình đang chọn (``None`` = không rõ).

    Returns:
        Thông điệp cảnh báo, hoặc ``None`` nếu trong hạn mức.
    """
    if daily_limit is None or daily_limit <= 0:
        return None
    if estimated_requests <= daily_limit:
        return None
    return (
        f"Ước tính cần {estimated_requests} request nhưng hạn mức ngày của mô hình "
        f"đang chọn chỉ {daily_limit}. Lô sẽ dừng giữa chừng khi hết hạn mức — "
        "các tập đã dịch xong vẫn được giữ, chạy lại hôm sau sẽ tự bỏ qua chúng."
    )


def summarise_translate_plan(plan: list[TranslateItem]) -> str:
    """Tóm tắt kế hoạch thành một dòng để hiển thị.

    Args:
        plan: Kế hoạch đã lập.

    Returns:
        Chuỗi dạng ``"12 tập sẽ dịch (540 câu) · 3 chưa trích xuất · 2 đã dịch"``.
    """
    ready = [item for item in plan if item.will_run]
    missing = sum(
        1 for item in plan if item.status is TranslateItemStatus.MISSING_SOURCE
    )
    done = sum(1 for item in plan if item.status is TranslateItemStatus.ALREADY_DONE)

    lines = sum(item.line_count for item in ready)
    head = f"{len(ready)} tập sẽ dịch"
    if lines:
        head += f" ({lines} câu)"
    parts = [head]
    if missing:
        parts.append(f"{missing} chưa trích xuất")
    if done:
        parts.append(f"{done} đã dịch")
    return " · ".join(parts)


__all__ = [
    "TranslateItem",
    "TranslateItemStatus",
    "build_translate_plan",
    "count_subtitle_lines",
    "estimate_requests",
    "find_episode_videos",
    "find_existing_translation",
    "find_source_subtitle",
    "quota_warning",
    "summarise_translate_plan",
]
