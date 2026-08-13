"""Tìm video "anh em" cùng tên nằm cạnh tệp phụ đề (logic thuần, testable).

Phục vụ tính năng Smart Context Auto-Detect: khi nạp phụ đề SRT/ASS, tự dò video
cùng tên trong cùng thư mục để nạp làm Ngữ cảnh Video, đỡ thao tác tay 2 lần.
"""

from __future__ import annotations

from pathlib import Path

#: Phần mở rộng video thường gặp (ưu tiên theo thứ tự phổ biến).
VIDEO_EXTENSIONS: tuple[str, ...] = (
    ".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".ts", ".m4v", ".wmv", ".mpg", ".mpeg",
)

#: Hậu tố do ứng dụng tự sinh khi đặt tên phụ đề (cần bóc để tìm tên phim gốc).
_KNOWN_SUBTITLE_SUFFIXES: tuple[str, ...] = (".original", ".translate", ".tts")


def _candidate_stems(subtitle_path: Path) -> list[str]:
    """Sinh các "tên gốc" ứng viên từ tên phụ đề, từ cụ thể đến tổng quát.

    ``phim.translate.vi.srt`` → ``["phim.translate.vi", "phim.translate", "phim"]``
    để vẫn dò được ``phim.mp4`` dù phụ đề mang hậu tố ``.translate.vi``.
    """
    name = subtitle_path.name
    for sub_ext in (".srt", ".ass", ".vtt"):
        if name.lower().endswith(sub_ext):
            name = name[: -len(sub_ext)]
            break

    stems = [name]
    parts = name.split(".")
    # Cắt dần các phần đuôi (mã ngôn ngữ, hậu tố .original/.translate/.tts…).
    for cut in range(len(parts) - 1, 0, -1):
        candidate = ".".join(parts[:cut])
        if candidate and candidate not in stems:
            stems.append(candidate)
    return stems


def find_sibling_video(subtitle_path: str | Path) -> Path | None:
    """Tìm tệp video cùng tên nằm cùng thư mục với phụ đề.

    Args:
        subtitle_path: Đường dẫn tệp phụ đề (.srt/.ass).

    Returns:
        Đường dẫn video tìm được (ưu tiên tên khớp cụ thể nhất + đuôi phổ biến nhất),
        hoặc ``None`` nếu không có.
    """
    sub_path = Path(subtitle_path)
    folder = sub_path.parent
    if not folder.is_dir():
        return None

    for stem in _candidate_stems(sub_path):
        for extension in VIDEO_EXTENSIONS:
            candidate = folder / f"{stem}{extension}"
            if candidate.is_file():
                return candidate
    return None
