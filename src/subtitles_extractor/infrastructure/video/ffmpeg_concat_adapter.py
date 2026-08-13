"""[v3.23.366] Adapter chạy ffmpeg để NỐI nhiều video thành một tệp trọn bộ.

Ưu tiên concat demuxer với ``-c copy`` (SAO CHÉP luồng — cực nhanh, không giảm chất
lượng, không nén lại). Chỉ hoạt động khi các tập cùng codec/thông số — điều luôn đúng
với video do chính khâu Xuất bản của ứng dụng tạo ra. Nếu ``-c copy`` thất bại (thông số
lệch), cho phép NÉN LẠI (re-encode) làm phương án dự phòng chắc chắn (chậm hơn).
"""

from __future__ import annotations

import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

from subtitles_extractor.infrastructure.media import find_ffmpeg, missing_ffmpeg_message
from subtitles_extractor.infrastructure.process.hidden_process import no_window_kwargs

ProgressCallback = Callable[[float], None]
CancelCheck = Callable[[], bool]


class VideoConcatError(RuntimeError):
    """Nối video thất bại."""


def _parse_progress_seconds(line: str) -> float | None:
    """Đọc mốc ``time=HH:MM:SS.ms`` từ một dòng stderr của ffmpeg."""
    marker = "time="
    index = line.find(marker)
    if index < 0:
        return None
    token = line[index + len(marker):].split(" ", 1)[0].strip()
    parts = token.split(":")
    if len(parts) != 3:
        return None
    try:
        hours, minutes, seconds = float(parts[0]), float(parts[1]), float(parts[2])
    except ValueError:
        return None
    return hours * 3600 + minutes * 60 + seconds


def concatenate_videos(
    videos: list[Path],
    output_path: Path,
    *,
    reencode: bool = False,
    total_duration_sec: float = 0.0,
    on_progress: ProgressCallback | None = None,
    cancel_check: CancelCheck | None = None,
    timeout_sec: float = 6 * 3600,
) -> Path:
    """Nối ``videos`` (theo đúng thứ tự) thành ``output_path``.

    Args:
        videos: Danh sách video ĐÃ sắp thứ tự.
        output_path: Tệp trọn bộ sẽ ghi.
        reencode: ``False`` = sao chép luồng (nhanh); ``True`` = nén lại (chắc chắn hơn).
        total_duration_sec: Tổng thời lượng (để tính % tiến độ). 0 = không tính %.
        on_progress: Callback nhận tiến độ 0.0–1.0.
        cancel_check: Hàm trả ``True`` khi người dùng yêu cầu huỷ.
        timeout_sec: Giới hạn thời gian.

    Returns:
        ``output_path`` khi thành công.

    Raises:
        VideoConcatError: Thiếu ffmpeg, danh sách rỗng, bị huỷ, hoặc ffmpeg lỗi.
    """
    if len(videos) < 2:
        raise VideoConcatError("Cần ít nhất 2 tệp để nối.")
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        raise VideoConcatError(missing_ffmpeg_message(feature="Nối video cả bộ"))

    from subtitles_extractor.application.services.concat_plan import (
        build_concat_list_content,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    list_content = build_concat_list_content(videos)

    # Ghi danh sách ra tệp tạm (concat demuxer đọc từ tệp).
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", prefix="concat_", delete=False, encoding="utf-8"
    ) as handle:
        handle.write(list_content)
        list_path = Path(handle.name)

    try:
        command = [
            ffmpeg, "-hide_banner", "-y",
            "-f", "concat", "-safe", "0", "-i", str(list_path),
        ]
        if reencode:
            # Nén lại: chuẩn hoá về H.264 + AAC để mọi tập ghép được dù lệch thông số.
            command += ["-c:v", "libx264", "-preset", "medium", "-crf", "20",
                        "-c:a", "aac", "-b:a", "192k"]
        else:
            command += ["-c", "copy"]
        command += ["-movflags", "+faststart", str(output_path)]

        return _run_ffmpeg_concat(
            command, output_path, total_duration_sec,
            on_progress, cancel_check, timeout_sec,
        )
    finally:
        list_path.unlink(missing_ok=True)


def _run_ffmpeg_concat(
    command: list[str],
    output_path: Path,
    total_duration_sec: float,
    on_progress: ProgressCallback | None,
    cancel_check: CancelCheck | None,
    timeout_sec: float,
) -> Path:
    """Chạy lệnh ffmpeg concat, đọc tiến độ từ stderr, hỗ trợ huỷ."""
    process = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        **no_window_kwargs(),
    )
    assert process.stderr is not None
    tail: list[str] = []
    try:
        for line in process.stderr:
            tail.append(line)
            if len(tail) > 40:
                tail.pop(0)
            if cancel_check is not None and cancel_check():
                process.kill()
                raise VideoConcatError("Đã huỷ nối video.")
            if on_progress is not None and total_duration_sec > 0:
                processed = _parse_progress_seconds(line)
                if processed is not None:
                    on_progress(min(1.0, processed / total_duration_sec))
        return_code = process.wait(timeout=timeout_sec)
    except subprocess.TimeoutExpired as exc:
        process.kill()
        raise VideoConcatError("Nối video quá thời gian cho phép.") from exc

    if return_code != 0:
        detail = "".join(tail[-6:]).strip()
        raise VideoConcatError(f"ffmpeg lỗi (mã {return_code}): {detail}")
    if not output_path.is_file():
        raise VideoConcatError("ffmpeg báo xong nhưng không thấy tệp trọn bộ.")
    if on_progress is not None:
        on_progress(1.0)
    return output_path


__all__ = ["VideoConcatError", "concatenate_videos"]
