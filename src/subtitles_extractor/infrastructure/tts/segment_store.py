"""[v3.23.450] Quản lý audio RIÊNG từng dòng TTS (để nghe lại + TTS lại dòng chọn).

Khi ``TTSRequest.keep_line_audio`` bật, mỗi dòng được ghi ra một WAV riêng trong thư mục segment
(cạnh file output). Module này chỉ tính TOÁN đường dẫn (hàm thuần) — việc ghi/đọc audio do adapter
thực hiện, để dễ test và tách biệt I/O.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

from subtitles_extractor.domain.ports.subtitle_tts_port import TTSSegmentResult

logger = logging.getLogger(__name__)


def segments_dir_for(output_path: Path, override: str = "") -> Path:
    """Thư mục chứa các segment audio của một lần TTS.

    Args:
        output_path: Đường dẫn WAV output chính (vd ``.../ep01.tts.wav``).
        override: Thư mục do người gọi chỉ định; rỗng → suy ra cạnh output.

    Returns:
        Đường dẫn thư mục segment (chưa đảm bảo tồn tại — người gọi tự tạo).
    """
    if override:
        return Path(override)
    stem = output_path.stem  # bỏ đuôi .wav (và .tts nếu có, giữ nguyên phần còn lại)
    return output_path.parent / f"{stem}_segments"


def segment_path(segments_dir: Path, event_index: int) -> Path:
    """Đường dẫn WAV của MỘT dòng theo chỉ số sự kiện (ổn định, để tra cứu/ghi đè khi TTS lại).

    Args:
        segments_dir: Thư mục segment (từ :func:`segments_dir_for`).
        event_index: Chỉ số dòng phụ đề (``TTSSegmentResult.event_index``).

    Returns:
        Đường dẫn ``<segments_dir>/seg_<index 6 chữ số>.wav``.
    """
    return segments_dir / f"seg_{int(event_index):06d}.wav"


def renumber_results(
    results: Sequence[TTSSegmentResult], mapping: Mapping[int, int],
    delete_removed: bool = True,
) -> list[TTSSegmentResult]:
    """[v3.23.625] Kết quả TTS sau khi thêm/xoá dòng: bỏ dòng đã xoá, đánh số lại vị trí.

    Đổi tên cả tệp audio từng dòng (``seg_<vị trí>.wav``) theo vị trí mới — nếu không, lần
    TTS lại kế tiếp ghi ``seg_<vị trí mới>.wav`` đè lên audio của một dòng KHÁC. Đổi tên hai
    bước (qua tên tạm) nên các vị trí dịch chồng lên nhau không đè nhau.

    Args:
        results: Kết quả hiện có.
        mapping: ``{vị trí cũ: vị trí mới}`` của dòng còn lại (xem ``dubbing_line_structure``).
        delete_removed: [v3.23.630] ``False`` = giữ nguyên file audio của dòng không có trong
            ``mapping`` (dùng khi chỉ dời các dòng dùng lại từ cache, dòng còn lại để nguyên).

    Returns:
        Kết quả mới, theo thứ tự vị trí mới.
    """
    survivors = [r for r in results if int(r.event_index) in mapping]
    for gone in results if delete_removed else ():
        if int(gone.event_index) not in mapping and gone.segment_audio_path:
            try:
                Path(gone.segment_audio_path).unlink(missing_ok=True)
            except OSError as exc:
                logger.debug("Không xoá được segment của dòng đã xoá: %s", exc)
    staged: list[tuple[Path, Path, int]] = []
    for result in survivors:
        old_index, new_index = int(result.event_index), mapping[int(result.event_index)]
        src = Path(result.segment_audio_path) if result.segment_audio_path else None
        if src is None or old_index == new_index or not src.exists():
            continue
        staged.append((src, segment_path(src.parent, new_index), new_index))
    temps: list[tuple[Path, Path]] = []
    for src, dest, _ in staged:
        tmp = src.with_name(src.name + ".renumber")
        try:
            os.replace(src, tmp)
            temps.append((tmp, dest))
        except OSError as exc:
            logger.warning("Không đổi tên segment %s: %s", src, exc)
    moved: dict[Path, Path] = {}
    for tmp, dest in temps:
        try:
            os.replace(tmp, dest)
            moved[tmp.with_name(tmp.name.removesuffix(".renumber"))] = dest
        except OSError as exc:
            logger.warning("Không đặt segment về %s: %s", dest, exc)
    renumbered: list[TTSSegmentResult] = []
    for result in survivors:
        path = result.segment_audio_path
        if path:
            path = str(moved.get(Path(path), path))
        renumbered.append(replace(
            result, event_index=mapping[int(result.event_index)], segment_audio_path=path
        ))
    renumbered.sort(key=lambda r: r.event_index)
    return renumbered


__all__ = ["renumber_results", "segment_path", "segments_dir_for"]
