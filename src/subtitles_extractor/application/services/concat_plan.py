"""[v3.23.366] Lập kế hoạch NỐI các tập phim thành MỘT video trọn bộ.

Tách phần quyết định (thuần, test được) khỏi phần chạy ffmpeg (adapter). Nhiệm vụ:
tìm các tệp video trong thư mục, SẮP THỨ TỰ TỰ NHIÊN theo số tập (để "第2集" đứng trước
"第10集", không bị sắp theo bảng chữ cái), và dựng nội dung danh sách concat cho ffmpeg.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

#: Đuôi tệp video hợp lệ để nối.
_VIDEO_SUFFIXES: frozenset[str] = frozenset(
    {".mp4", ".mkv", ".avi", ".mov", ".ts", ".webm", ".m4v", ".flv"}
)

_NUMBER_RE = re.compile(r"\d+")


@dataclass(frozen=True, slots=True)
class ConcatPlan:
    """Kế hoạch nối video.

    Attributes:
        videos: Danh sách tệp đã sắp thứ tự tự nhiên.
        output_path: Tệp video trọn bộ sẽ ghi ra.
    """

    videos: list[Path]
    output_path: Path

    @property
    def is_valid(self) -> bool:
        """Cần tối thiểu 2 tệp mới có ý nghĩa nối."""
        return len(self.videos) >= 2


def natural_sort_key(path: Path) -> tuple:
    """Khoá sắp xếp TỰ NHIÊN: tách xen kẽ chữ và SỐ (số so sánh theo giá trị).

    Ví dụ thứ tự đúng: ``第1集 < 第2集 < 第10集 < 第84集`` (thay vì 1,10,2,84...).
    """
    parts = _NUMBER_RE.split(path.stem)
    numbers = _NUMBER_RE.findall(path.stem)
    # Xen kẽ: text, number, text, number… — số ép kiểu int để so theo giá trị.
    key: list[object] = []
    for index, text in enumerate(parts):
        key.append(text.lower())
        if index < len(numbers):
            key.append(int(numbers[index]))
    return tuple(key)


def find_concat_videos(
    folder: Path,
    *,
    name_filter: str | None = None,
    exclude_names: set[str] | None = None,
) -> list[Path]:
    """Tìm và SẮP THỨ TỰ TỰ NHIÊN các video trong thư mục để nối.

    Args:
        folder: Thư mục chứa các tập.
        name_filter: Nếu khác ``None``, chỉ lấy tệp có chứa chuỗi này trong tên (vd
            ``"_phude_thuyetminh"`` để chỉ nối các bản ĐÃ xuất bản).
        exclude_names: Tập tên tệp cần loại (vd chính tệp trọn bộ đầu ra nếu đã tồn tại).

    Returns:
        Danh sách ``Path`` đã sắp thứ tự tự nhiên (rỗng nếu không có).
    """
    if not folder.is_dir():
        return []
    exclude = exclude_names or set()
    candidates = [
        entry
        for entry in folder.iterdir()
        if entry.is_file()
        and entry.suffix.lower() in _VIDEO_SUFFIXES
        and entry.name not in exclude
        and (name_filter is None or name_filter in entry.name)
    ]
    return sorted(candidates, key=natural_sort_key)


def build_concat_list_content(videos: list[Path]) -> str:
    """Dựng nội dung tệp danh sách cho ffmpeg concat demuxer.

    Định dạng mỗi dòng: ``file '<đường dẫn tuyệt đối>'``. Dấu nháy đơn trong đường dẫn
    được escape theo quy tắc của ffmpeg (``'`` → ``'\\''``).
    """
    lines = []
    for video in videos:
        safe = str(video.resolve()).replace("'", "'\\''")
        lines.append(f"file '{safe}'")
    return "\n".join(lines) + "\n"


def default_concat_output(folder: Path, videos: list[Path]) -> Path:
    """Tên tệp trọn bộ mặc định: ``<tên thư mục>_trọn bộ.mkv`` trong cùng thư mục.

    Dùng MKV để bao được mọi codec khi sao chép luồng (``-c copy``) mà không phải nén lại.
    """
    base = folder.name or (videos[0].parent.name if videos else "video")
    return folder / f"{base}_tron_bo.mkv"
