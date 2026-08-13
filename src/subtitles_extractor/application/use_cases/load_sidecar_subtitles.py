"""[v3.23.168] Use case: dò & nạp file phụ đề rời cùng tên cạnh video.

Điều phối tầng ứng dụng: quét thư mục video, gọi hàm thuần
:func:`find_sidecar_subtitles` để lọc file phụ đề cùng tên, rồi ủy việc PARSE cho
:class:`ImportSubtitlesUseCase` (tái dùng importer SRT/ASS sẵn có — DRY). Không tự
parse để tránh trùng logic đọc mã hóa/định dạng.
"""

from __future__ import annotations

import logging
from pathlib import Path

from subtitles_extractor.application.use_cases.import_subtitles import (
    ImportSubtitlesUseCase,
)
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.infrastructure.subtitle.subtitle_sidecar_finder import (
    SidecarSubtitle,
    find_sidecar_subtitles,
)

logger = logging.getLogger(__name__)


class SidecarSubtitleNotFoundError(FileNotFoundError):
    """Không tìm thấy file phụ đề rời nào cạnh video."""


class LoadSidecarSubtitlesUseCase:
    """Dò và nạp phụ đề rời cùng tên với video (bổ trợ luồng OCR/track nhúng)."""

    def __init__(self, import_use_case: ImportSubtitlesUseCase) -> None:
        """Khởi tạo use case.

        Args:
            import_use_case: Use case parse tệp phụ đề thành event (tiêm vào để test).
        """
        self._import_use_case = import_use_case

    def find(self, video_path: Path) -> list[SidecarSubtitle]:
        """Liệt kê các file phụ đề rời cùng tên cạnh video (không nạp nội dung).

        Args:
            video_path: Đường dẫn video đang mở.

        Returns:
            Danh sách phụ đề rời theo thứ tự ưu tiên; rỗng nếu thư mục không tồn tại
            hoặc không có file phù hợp.
        """
        parent = video_path.parent
        if not parent.is_dir():
            return []
        try:
            siblings = [item for item in parent.iterdir() if item.is_file()]
        except OSError as exc:
            logger.warning("Không đọc được thư mục '%s' để dò phụ đề: %s", parent, exc)
            return []
        return find_sidecar_subtitles(video_path, siblings)

    def load(self, subtitle_path: Path) -> list[SubtitleEvent]:
        """Nạp một file phụ đề rời thành danh sách event.

        Args:
            subtitle_path: Đường dẫn file phụ đề (thường từ :meth:`find`).

        Returns:
            Danh sách :class:`SubtitleEvent`.

        Raises:
            KeyError: Định dạng không có importer tương ứng.
            FileNotFoundError: Tệp không tồn tại.
        """
        return self._import_use_case.execute(subtitle_path)

    def find_and_load_best(self, video_path: Path) -> tuple[Path, list[SubtitleEvent]]:
        """Dò rồi nạp file phụ đề rời ƯU TIÊN CAO NHẤT cạnh video.

        Args:
            video_path: Đường dẫn video đang mở.

        Returns:
            Cặp ``(đường_dẫn_phụ_đề, danh_sách_event)`` của file tốt nhất.

        Raises:
            SidecarSubtitleNotFoundError: Không có file phụ đề rời nào phù hợp.
        """
        candidates = self.find(video_path)
        if not candidates:
            raise SidecarSubtitleNotFoundError(
                f"Không tìm thấy phụ đề rời cùng tên cạnh '{video_path.name}'."
            )
        best = candidates[0]
        events = self.load(best.path)
        logger.info(
            "Đã nạp %d câu từ phụ đề rời '%s' (ngôn ngữ='%s').",
            len(events), best.path.name, best.language_tag or "không rõ",
        )
        return (best.path, events)


__all__ = [
    "LoadSidecarSubtitlesUseCase",
    "SidecarSubtitleNotFoundError",
]
