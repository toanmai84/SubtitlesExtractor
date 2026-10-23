"""[v3.23.625] Thêm / xoá dòng khi lồng tiếng — các hàm THUẦN tính lại vị trí.

Mọi trạng thái của trang TTS (người nói chọn tay, lời sửa tay, dòng cần TTS lại, kết quả
tổng hợp, audio từng dòng) đánh theo VỊ TRÍ dòng. Thêm hoặc xoá một dòng làm dịch vị trí
các dòng sau nó, nên mô-đun này tính bảng ánh xạ ``vị trí cũ → vị trí mới`` và áp nó lên
từng loại trạng thái. Phần I/O (đổi tên tệp audio, ghép lại master) nằm ở nơi khác.
"""

from __future__ import annotations

import bisect
from collections.abc import Collection, Mapping, Sequence
from typing import TypeVar

from subtitles_extractor.application.services.dubbing_line_edits import LineSnapshot

T = TypeVar("T")

#: Thời lượng mặc định của dòng mới khi khoảng trống đủ rộng (giây).
DEFAULT_NEW_LINE_SEC = 2.0
#: Dòng mới ngắn hơn mức này thì không đủ đọc; dùng :data:`FALLBACK_NEW_LINE_SEC`.
MIN_NEW_LINE_SEC = 0.4
FALLBACK_NEW_LINE_SEC = 1.5


def delete_mapping(count: int, positions: Collection[int]) -> dict[int, int]:
    """``{vị trí cũ: vị trí mới}`` của các dòng CÒN LẠI sau khi xoá ``positions``."""
    removed = {p for p in positions if 0 <= p < count}
    mapping: dict[int, int] = {}
    new = 0
    for old in range(count):
        if old not in removed:
            mapping[old] = new
            new += 1
    return mapping


def insert_mapping(count: int, at: int) -> dict[int, int]:
    """``{vị trí cũ: vị trí mới}`` khi chèn một dòng mới vào vị trí ``at``."""
    at = max(0, min(at, count))
    return {old: old + 1 if old >= at else old for old in range(count)}


def remap_keys(values: Mapping[int, T], mapping: Mapping[int, int]) -> dict[int, T]:
    """Đổi khoá vị trí theo ``mapping``; khoá không còn (dòng đã xoá) bị bỏ."""
    return {mapping[key]: value for key, value in values.items() if key in mapping}


def remap_set(values: Collection[int], mapping: Mapping[int, int]) -> set[int]:
    """Đổi tập vị trí theo ``mapping``; vị trí không còn bị bỏ."""
    return {mapping[value] for value in values if value in mapping}


def insertion_position(starts: Sequence[float], start_sec: float) -> int:
    """Vị trí chèn dòng bắt đầu lúc ``start_sec`` để giữ thứ tự thời gian.

    Dòng mới đứng SAU mọi dòng bắt đầu cùng lúc hoặc sớm hơn.
    """
    return bisect.bisect_right(list(starts), start_sec)


def default_line_timing(
    starts: Sequence[float], ends: Sequence[float], after_position: int | None,
    media_duration_s: float | None = None,
) -> tuple[float, float]:
    """Mốc gợi ý cho dòng mới đặt NGAY SAU dòng ``after_position``.

    Dòng mới bắt đầu lúc dòng đó dứt và kéo dài tới
    :data:`DEFAULT_NEW_LINE_SEC` nhưng không lấn quá dòng kế; khoảng trống quá hẹp thì lấy
    :data:`FALLBACK_NEW_LINE_SEC` (người dùng chỉnh lại trong hộp thoại).

    Args:
        starts: Mốc bắt đầu các dòng hiện có.
        ends: Mốc kết thúc các dòng hiện có.
        after_position: Dòng đứng trước dòng mới; ``None`` = đầu phụ đề.
        media_duration_s: Thời lượng video (chặn mốc kết thúc), nếu biết.

    Returns:
        ``(bắt đầu, kết thúc)``.
    """
    if after_position is None or not starts:
        start = 0.0
        following = 0
    else:
        after = max(0, min(after_position, len(starts) - 1))
        start = float(ends[after])
        following = after + 1
    end = start + DEFAULT_NEW_LINE_SEC
    if following < len(starts):
        room = float(starts[following]) - start
        end = start + (min(DEFAULT_NEW_LINE_SEC, room) if room >= MIN_NEW_LINE_SEC
                       else FALLBACK_NEW_LINE_SEC)
    if media_duration_s is not None and media_duration_s > 0:
        end = min(end, max(start, float(media_duration_s)))
    return start, max(end, start)


_SPLIT_MARKS = "。！？，；：、,.;:!?…"  # noqa: RUF001 — dấu câu CJK/Latin cần nhận


def suggest_split(text: str, start_sec: float, end_sec: float) -> tuple[str, str, float]:
    """Gợi ý cách tách một dòng thành hai: ``(nửa đầu, nửa sau, mốc cắt)``.

    Cắt tại dấu câu (rồi tới khoảng trắng) gần GIỮA lời nhất; mốc cắt chia thời lượng theo tỉ
    lệ số ký tự. Lời không có chỗ cắt thì cắt đôi theo ký tự.
    """
    text = (text or "").strip()
    if len(text) < 2:
        return text, "", (start_sec + end_sec) / 2
    middle = len(text) / 2
    marks = [i + 1 for i, c in enumerate(text[:-1]) if c in _SPLIT_MARKS]
    spaces = [i + 1 for i, c in enumerate(text[:-1]) if c.isspace()]
    candidates = marks or spaces or [round(middle)]
    cut = min(candidates, key=lambda i: abs(i - middle))
    first, second = text[:cut].strip(), text[cut:].strip()
    if not first or not second:
        cut = max(1, round(middle))
        first, second = text[:cut].strip(), text[cut:].strip()
    share = len(first) / max(1, len(first) + len(second))
    split = start_sec + (end_sec - start_sec) * share
    return first, second, min(max(split, start_sec), end_sec)


def snapshot_is_structural(origins: Sequence[int], original_count: int) -> bool:
    """Danh sách dòng khác lúc nạp (có dòng thêm / xoá / đổi chỗ) không."""
    return list(origins) != list(range(original_count))


def build_snapshot(
    origins: Sequence[int], starts: Sequence[float], ends: Sequence[float],
    texts: Sequence[str],
) -> tuple[LineSnapshot, ...]:
    """Ảnh chụp danh sách dòng hiện tại để lưu theo phụ đề."""
    return tuple(
        LineSnapshot(int(o), float(s), float(e), str(t))
        for o, s, e, t in zip(origins, starts, ends, texts, strict=True)
    )


def valid_snapshot(snapshot: Sequence[LineSnapshot], original_count: int) -> bool:
    """Ảnh chụp có áp được lên phụ đề gốc ``original_count`` dòng không.

    Mỗi ``origin`` hợp lệ phải trỏ vào dòng gốc và KHÔNG lặp; mốc phải theo thứ tự thời gian.
    """
    seen: set[int] = set()
    last_start = -1.0
    for line in snapshot:
        if line.origin >= original_count or line.origin < -1:
            return False
        if line.origin >= 0:
            if line.origin in seen:
                return False
            seen.add(line.origin)
        if line.start_sec < last_start:
            return False
        last_start = line.start_sec
    return True


__all__ = [
    "DEFAULT_NEW_LINE_SEC",
    "build_snapshot",
    "default_line_timing",
    "delete_mapping",
    "insert_mapping",
    "insertion_position",
    "remap_keys",
    "remap_set",
    "snapshot_is_structural",
    "suggest_split",
    "valid_snapshot",
]
