"""[v3.23.168] Dò file phụ đề "sidecar" cùng tên cùng thư mục với video.

Nhiều bộ phim tải về kèm sẵn file phụ đề rời đặt CẠNH video, cùng tên gốc:
``Movie.mkv`` + ``Movie.srt`` / ``Movie.vi.srt`` / ``Movie.en.ass``. Module này dò các
file đó để trang trích xuất nạp trực tiếp — nhanh hơn nhiều so với OCR và không cần
track nhúng. Logic dò tách thành hàm THUẦN (chỉ nhận đường dẫn + danh sách file có
sẵn) để kiểm thử độc lập, không chạm đĩa.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Đuôi phụ đề được hỗ trợ, thứ tự = ĐỘ ƯU TIÊN (SRT trước vì phổ biến + parse ổn định).
# Chỉ liệt kê đuôi mà ứng dụng THỰC SỰ nạp được (có importer): .srt và .ass/.ssa
# (SSA dùng chung parser với ASS). Không thêm .vtt khi chưa có importer tương ứng để
# tránh dò ra file rồi báo lỗi lúc nạp.
_SUPPORTED_EXTENSIONS: tuple[str, ...] = (".srt", ".ass", ".ssa")


@dataclass(frozen=True, slots=True)
class SidecarSubtitle:
    """Một file phụ đề rời tìm thấy cạnh video.

    Attributes:
        path: Đường dẫn tới file phụ đề.
        language_tag: Thẻ ngôn ngữ suy ra từ tên file (vd 'vi', 'en'); rỗng nếu không có.
        extension: Đuôi file chuẩn hoá chữ thường (vd '.srt').
    """

    path: Path
    language_tag: str
    extension: str


def _language_tag_from_stem(subtitle_stem: str, video_stem: str) -> str:
    """Suy thẻ ngôn ngữ từ phần tên phụ đề dôi ra so với tên video.

    Ví dụ video ``Movie`` + phụ đề ``Movie.vi`` -> thẻ ``vi``; ``Movie.forced.en`` ->
    ``en`` (lấy đoạn cuối cùng ngăn bằng dấu chấm). Trả rỗng nếu không có phần dôi.

    Args:
        subtitle_stem: Tên phụ đề đã bỏ đuôi (vd 'Movie.vi').
        video_stem: Tên video đã bỏ đuôi (vd 'Movie').

    Returns:
        Thẻ ngôn ngữ chữ thường, hoặc chuỗi rỗng.
    """
    if subtitle_stem == video_stem:
        return ""
    if subtitle_stem.lower().startswith(video_stem.lower()):
        suffix = subtitle_stem[len(video_stem):].lstrip(". ")
        if suffix:
            return suffix.split(".")[-1].lower()
    return ""


def find_sidecar_subtitles(
    video_path: Path,
    sibling_files: list[Path],
    supported_extensions: tuple[str, ...] = _SUPPORTED_EXTENSIONS,
) -> list[SidecarSubtitle]:
    """Tìm các file phụ đề rời cùng tên gốc với video trong danh sách file cùng thư mục.

    Hàm THUẦN: không quét đĩa (bên gọi truyền sẵn ``sibling_files``) -> dễ test. Nhận
    diện cả tên khớp CHÍNH XÁC (``Movie.srt``) lẫn tên có hậu tố ngôn ngữ/biến thể
    (``Movie.vi.srt``, ``Movie.forced.srt``). So khớp KHÔNG phân biệt hoa thường.

    Args:
        video_path: Đường dẫn video gốc.
        sibling_files: Danh sách file nằm cùng thư mục với video.
        supported_extensions: Bộ đuôi phụ đề chấp nhận, theo thứ tự ưu tiên.

    Returns:
        Danh sách :class:`SidecarSubtitle` đã lọc + SẮP theo (ưu tiên đuôi, rồi tên),
        khớp chính xác đứng trước biến thể có hậu tố. Rỗng nếu không tìm thấy.
    """
    video_stem = video_path.stem
    video_stem_lower = video_stem.lower()
    normalized_exts = tuple(ext.lower() for ext in supported_extensions)
    ext_priority = {ext: index for index, ext in enumerate(normalized_exts)}

    matches: list[SidecarSubtitle] = []
    for candidate in sibling_files:
        extension = candidate.suffix.lower()
        if extension not in ext_priority:
            continue
        candidate_stem = candidate.stem
        candidate_stem_lower = candidate_stem.lower()
        is_exact = candidate_stem_lower == video_stem_lower
        is_variant = candidate_stem_lower.startswith(video_stem_lower + ".")
        if not (is_exact or is_variant):
            continue
        matches.append(
            SidecarSubtitle(
                path=candidate,
                language_tag=_language_tag_from_stem(candidate_stem, video_stem),
                extension=extension,
            )
        )

    def _sort_key(item: SidecarSubtitle) -> tuple[int, int, str]:
        # Khớp chính xác (không có thẻ ngôn ngữ) ưu tiên hơn biến thể; rồi theo thứ tự
        # đuôi; cuối cùng theo tên để ổn định.
        exact_first = 0 if not item.language_tag else 1
        return (exact_first, ext_priority[item.extension], item.path.name.lower())

    return sorted(matches, key=_sort_key)
