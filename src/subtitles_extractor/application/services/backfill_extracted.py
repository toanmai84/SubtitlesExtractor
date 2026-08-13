"""[v3.23.365] Bù tệp ``.original.srt`` từ CƠ SỞ DỮ LIỆU cho các tập ĐÃ trích xuất.

VÌ SAO CẦN
==========
Trước v3.23.364, khâu trích xuất (đặc biệt là HÀNG LOẠT) chỉ lưu phụ đề gốc vào cơ sở dữ
liệu, KHÔNG ghi tệp ``<tên>.original.srt`` cạnh video. Trong khi khâu Dịch/TTS/Xuất bản
hàng loạt lại quét ĐĨA tìm tệp đó ⇒ báo "chưa trích xuất" dù đã trích rồi.

Module này quét cơ sở dữ liệu và GHI BÙ tệp ``.original.srt`` từ nội dung đã lưu — để
người dùng KHỎI PHẢI trích xuất lại. Tách phần quyết định (thuần, test được) khỏi phần
I/O (tiêm hàm ``file_exists`` / ``write_text``) theo đúng Clean Architecture.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from subtitles_extractor.domain.entities.project_record import WorkflowStage
from subtitles_extractor.domain.value_objects.output_naming import (
    extracted_subtitle_path,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BackfillResult:
    """Kết quả bù tệp phụ đề gốc.

    Attributes:
        written: Danh sách tệp ``.original.srt`` đã ghi bù.
        skipped_existing: Số tập đã có sẵn tệp (không cần bù).
        skipped_no_content: Số bản ghi không có nội dung phụ đề gốc.
        skipped_no_video: Số bản ghi trỏ tới video không còn tồn tại.
        failed: Số tệp ghi thất bại.
    """

    written: list[Path] = field(default_factory=list)
    skipped_existing: int = 0
    skipped_no_content: int = 0
    skipped_no_video: int = 0
    failed: int = 0

    def summary_vi(self) -> str:
        """Chuỗi tóm tắt tiếng Việt để hiển thị cho người dùng."""
        return (
            f"Đã bù {len(self.written)} tệp phụ đề gốc từ cơ sở dữ liệu"
            f" (bỏ qua {self.skipped_existing} tập đã có tệp)."
        )


def backfill_original_subtitles(
    records: Iterable[object],
    *,
    file_exists: Callable[[Path], bool],
    write_text: Callable[[Path, str], None],
    only_video_names: set[str] | None = None,
) -> BackfillResult:
    """Ghi bù ``.original.srt`` từ cơ sở dữ liệu cho các tập đã trích còn THIẾU tệp.

    Args:
        records: Danh sách ``ProjectRecord`` (từ ``repo.list_all()``).
        file_exists: Hàm kiểm tra tệp tồn tại (tiêm để test; thực tế ``Path.is_file``).
        write_text: Hàm ghi văn bản ra tệp (tiêm; thực tế ``atomic_write_text`` có BOM).
        only_video_names: Nếu khác ``None``, CHỈ bù cho các video có tên trong tập này
            (lọc theo phim bộ hiện tại). ``None`` = bù mọi bản ghi trong cơ sở dữ liệu.

    Returns:
        :class:`BackfillResult` thống kê số tệp đã ghi bù và các trường hợp bỏ qua.
    """
    result = BackfillResult()
    for record in records:
        stage = getattr(record, "stage", WorkflowStage.NEW)
        subtitle_text = (getattr(record, "original_subtitle", "") or "").strip()
        video_path_str = str(getattr(record, "video_path", "") or "")

        if stage < WorkflowStage.EXTRACTED or not subtitle_text:
            result = _bump(result, "skipped_no_content")
            continue
        if not video_path_str:
            result = _bump(result, "skipped_no_video")
            continue

        video_path = Path(video_path_str)
        if only_video_names is not None and video_path.name not in only_video_names:
            continue
        if not file_exists(video_path):
            result = _bump(result, "skipped_no_video")
            continue

        target = extracted_subtitle_path(video_path)
        if file_exists(target):
            result = _bump(result, "skipped_existing")
            continue

        try:
            write_text(target, subtitle_text)
        except OSError as exc:
            logger.warning("Bù .original.srt thất bại cho %s: %s", video_path.name, exc)
            result = _bump(result, "failed")
            continue
        result.written.append(target)
        logger.info("Đã bù phụ đề gốc từ CSDL → %s", target.name)

    return result


def _bump(result: BackfillResult, field_name: str) -> BackfillResult:
    """Trả bản sao BackfillResult với một trường đếm tăng 1 (giữ frozen)."""
    counts = {
        "skipped_existing": result.skipped_existing,
        "skipped_no_content": result.skipped_no_content,
        "skipped_no_video": result.skipped_no_video,
        "failed": result.failed,
    }
    counts[field_name] += 1
    return BackfillResult(written=result.written, **counts)
