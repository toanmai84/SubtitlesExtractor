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


def line_segments_dir(request: object, output_path: Path) -> Path | None:
    """[v3.23.655] Thư mục segment (đã tạo) nếu yêu cầu bật ``keep_line_audio``; ngược lại ``None``."""
    if not getattr(request, "keep_line_audio", False):
        return None
    directory = segments_dir_for(output_path, getattr(request, "segments_dir", ""))
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning("Không tạo được thư mục audio từng dòng %s: %s", directory, exc)
        return None
    return directory


def write_line_segment(segments_dir: Path | None, event_index: int, audio: object, sample_rate: int) -> str:
    """[v3.23.655] Ghi audio MỘT dòng (đúng đoạn đã đặt vào master) ra ``seg_<vị trí>.wav``.

    Dùng chung cho Edge/Gemini (VieNeu có bản riêng cùng định dạng). PCM_16 vì Media Foundation
    trên Windows phát WAV float chập chờn. Lỗi ghi chỉ cảnh báo — không chặn TTS chính.

    Returns:
        Đường dẫn tệp, hoặc ``""`` nếu không giữ audio / audio rỗng / ghi lỗi.
    """
    if segments_dir is None or audio is None or len(audio) == 0:  # type: ignore[arg-type]
        return ""
    try:
        import numpy as np
        import soundfile as sf

        path = segment_path(segments_dir, event_index)
        sf.write(str(path), np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0), int(sample_rate),
                 subtype="PCM_16")
        return str(path)
    except (OSError, RuntimeError, ValueError) as exc:
        logger.warning("Không ghi được audio dòng %d: %s", event_index, exc)
        return ""


def renumber_results(
    results: Sequence[TTSSegmentResult], mapping: Mapping[int, int],
    delete_removed: bool = True,
    trash_keys: Mapping[int, int] | None = None,
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
            # [v3.23.654] Dời vào thùng rác thay vì xoá: khôi phục dòng thì dùng lại audio, khỏi TTS lại.
            move_to_trash(gone, (trash_keys or {}).get(int(gone.event_index), -1))
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



# ── [v3.23.654] Thùng rác audio dòng đã xoá ─────────────────────────────────

TRASH_DIR_NAME = ".trash"
TRASH_INDEX = "index.json"
MAX_TRASH_ENTRIES = 400


def _trash_index(trash: Path) -> list[dict]:
    import json

    try:
        data = json.loads((trash / TRASH_INDEX).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [entry for entry in data if isinstance(entry, dict)] if isinstance(data, list) else []


def _write_trash_index(trash: Path, entries: list[dict]) -> None:
    import json

    keep = entries[-MAX_TRASH_ENTRIES:]
    for stale in entries[:-MAX_TRASH_ENTRIES] if len(entries) > MAX_TRASH_ENTRIES else ():
        try:
            (trash / str(stale.get("file", ""))).unlink(missing_ok=True)
        except OSError:
            pass
    try:
        tmp = trash / (TRASH_INDEX + ".tmp")
        tmp.write_text(json.dumps(keep, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, trash / TRASH_INDEX)
    except OSError as exc:
        logger.debug("Không ghi được chỉ mục thùng rác audio: %s", exc)


def clear_trash(segments_dir: Path) -> None:
    """Xoá thùng rác audio (lượt TTS mới: audio cũ có thể sai giọng/tốc độ)."""
    import shutil

    shutil.rmtree(Path(segments_dir) / TRASH_DIR_NAME, ignore_errors=True)


def move_to_trash(result: TTSSegmentResult, origin: int = -1) -> bool:
    """Dời audio của một dòng vừa xoá vào ``<thư mục segment>/.trash`` (thay vì xoá hẳn).

    Args:
        result: Kết quả của dòng bị xoá.
        origin: Vị trí dòng trong phụ đề GỐC (``-1`` = dòng thêm tay — không khôi phục được).
    """
    import uuid

    source = Path(result.segment_audio_path)
    if not source.is_file():
        return False
    trash = source.parent / TRASH_DIR_NAME
    name = f"{uuid.uuid4().hex}.wav"
    try:
        trash.mkdir(parents=True, exist_ok=True)
        os.replace(source, trash / name)
    except OSError as exc:
        logger.debug("Không dời được segment của dòng đã xoá vào thùng rác (%s) — xoá hẳn.", exc)
        try:
            source.unlink(missing_ok=True)
        except OSError:
            pass
        return False
    entries = _trash_index(trash)
    _drop_orphans(trash, [*entries, {"file": name}])
    entries.append({
        "file": name, "origin": int(origin), "text": result.text or result.subtitle_text or "",
        "duration": float(result.audio_duration_s or 0.0),
    })
    _write_trash_index(trash, entries)
    return True


def _drop_orphans(trash: Path, entries: list[dict]) -> None:
    """Tệp trong thùng rác không có trong chỉ mục (chỉ mục hỏng/ghi dở) → xoá, không để rác tích luỹ."""
    known = {str(entry.get("file", "")) for entry in entries}
    try:
        for path in trash.glob("*.wav"):
            if path.name not in known:
                path.unlink(missing_ok=True)
    except OSError:
        pass


def recover_from_trash(
    segments_dir: Path, text: str, event_index: int, origin: int,
) -> tuple[Path, float] | None:
    """Lấy lại audio của dòng GỐC ``origin`` đã xoá (đúng lời ``text``), đặt về vị trí ``event_index``.

    Returns:
        ``(đường dẫn segment mới, thời lượng)`` hoặc ``None`` nếu thùng rác không có.
    """
    trash = Path(segments_dir) / TRASH_DIR_NAME
    entries = _trash_index(trash)
    wanted = (text or "").strip()
    for position in range(len(entries) - 1, -1, -1):
        entry = entries[position]
        if int(entry.get("origin", -1)) != int(origin) or str(entry.get("text", "")).strip() != wanted:
            continue
        source = trash / str(entry.get("file", ""))
        if not source.is_file():
            continue
        target = segment_path(Path(segments_dir), event_index)
        if target.exists():
            return None  # vị trí đang có audio của dòng khác — không đè
        try:
            os.replace(source, target)
        except OSError as exc:
            logger.debug("Không lấy lại được audio từ thùng rác: %s", exc)
            return None
        del entries[position]
        _write_trash_index(trash, entries)
        return target, float(entry.get("duration", 0.0) or 0.0)
    return None


# ── [v3.23.657] Dung lượng + dọn audio từng dòng (người dùng bấm — KHÔNG tự dọn) ──


def _owned_line_audio_files(directory: Path) -> list[Path]:
    """Tệp do ứng dụng tạo trong thư mục segment: ``seg_*``, cache dòng, mọi tệp trong ``.trash``."""
    from subtitles_extractor.infrastructure.tts.line_cache_store import CACHE_FILENAME

    files = [*directory.glob("seg_*"), directory / CACHE_FILENAME]
    # Thùng rác + thư mục tạm của "TTS lại dòng chọn" sót lại sau khi sập (``resynth_seg_*``).
    for owned_dir in [directory / TRASH_DIR_NAME, *directory.glob("resynth_seg_*")]:
        if owned_dir.is_dir():
            files.extend(owned_dir.rglob("*"))
    return files


def line_audio_usage(segments_dir: Path) -> tuple[int, int]:
    """Dung lượng audio từng dòng (kể cả thùng rác + cache dòng) trong thư mục segment.

    Chỉ đếm tệp do ứng dụng tạo — thư mục segment có thể là thư mục video (``segments_dir``
    chỉ định riêng), không được tính lẫn video/master.

    Returns:
        ``(số byte, số tệp)``; thư mục không tồn tại → ``(0, 0)``.
    """
    directory = Path(segments_dir)
    if not directory.is_dir():
        return 0, 0
    total, count = 0, 0
    for path in _owned_line_audio_files(directory):
        try:
            if path.is_file():
                total += path.stat().st_size
                count += 1
        except OSError:
            continue
    return total, count


def clean_line_audio(segments_dir: Path) -> int:
    """Xoá audio từng dòng (``seg_*.wav``), thùng rác và cache dòng; master + SRT (ngoài thư mục) giữ nguyên.

    Chỉ xoá tệp do ứng dụng tạo (``seg_*``, ``.trash``, ``line_cache.json``); tệp lạ trong thư
    mục được để lại, thư mục chỉ bị xoá khi đã rỗng.

    Returns:
        Số byte đã giải phóng.
    """
    import shutil

    directory = Path(segments_dir)
    if not directory.is_dir():
        return 0
    before, _ = line_audio_usage(directory)
    tree_dirs = [directory / TRASH_DIR_NAME, *(p for p in directory.glob("resynth_seg_*") if p.is_dir())]
    for path in _owned_line_audio_files(directory):
        if any(tree in path.parents for tree in tree_dirs):
            continue  # cả cây bên dưới xoá bằng rmtree
        try:
            if path.is_file():
                path.unlink()
        except OSError as exc:
            logger.warning("Không xoá được %s: %s", path, exc)
    for tree in tree_dirs:
        shutil.rmtree(tree, ignore_errors=True)
    try:
        directory.rmdir()  # chỉ thành công khi rỗng
    except OSError:
        pass
    after, _ = line_audio_usage(directory)
    return max(0, before - after)


__all__ = [
    "clean_line_audio", "line_audio_usage", "line_segments_dir", "renumber_results", "segment_path", "segments_dir_for", "write_line_segment",
]
