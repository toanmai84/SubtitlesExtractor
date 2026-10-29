"""[v3.23.366] Adapter chạy ffmpeg để NỐI nhiều video thành một tệp trọn bộ.

Ưu tiên concat demuxer với ``-c copy`` (SAO CHÉP luồng — cực nhanh, không giảm chất
lượng, không nén lại). Chỉ hoạt động khi các tập cùng codec/thông số — điều luôn đúng
với video do chính khâu Xuất bản của ứng dụng tạo ra. Nếu ``-c copy`` thất bại (thông số
lệch), cho phép NÉN LẠI (re-encode) làm phương án dự phòng chắc chắn (chậm hơn).

[v3.23.617] SỬA LỖI THIẾU KHUNG HÌNH khi nối các tập lệch thông số. ``-c copy`` với các
tập khác tốc độ khung / time base KHÔNG làm ffmpeg báo lỗi nhưng tệp ra hỏng (khoảng trống
không có khung hình, hình giãn, tiếng lệch). Phương án "nén lại" cũ vẫn đi qua concat
demuxer nên hỏng y hệt. Nay:
    1. Đọc thông số mọi tập bằng ffprobe trước khi nối.
    2. Chỉ sao chép luồng khi mọi tập giống hệt nhau; lệch → CHUẨN HOÁ từng tập (tốc độ
       khung, kích thước, âm thanh) rồi nối bằng concat FILTER (nhãn thời gian liên tục).
    3. Sau khi sao chép luồng, đối chiếu thời lượng tệp ra với tổng các tập; lệch bất
       thường → tự nối lại bằng cách chuẩn hoá.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from itertools import pairwise
from pathlib import Path

from subtitles_extractor.application.services.concat_plan import (
    ConcatInputInfo,
    build_normalize_filter,
    choose_concat_target,
    copy_mismatch_reasons,
    duration_mismatch,
)
from subtitles_extractor.infrastructure.media import (
    find_ffmpeg,
    find_ffprobe,
    missing_ffmpeg_message,
)
from subtitles_extractor.infrastructure.process.hidden_process import no_window_kwargs
from subtitles_extractor.infrastructure.video.video_encoder_selector import (
    encoder_quality_args,
    is_hardware_encoder,
    mark_encoder_broken,
    pick_video_encoder,
    software_fallback,
)

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[float], None]
CancelCheck = Callable[[], bool]

#: Khoảng trống giữa hai khung hình liên tiếp (giây) coi là tệp nối hỏng.
_MAX_FRAME_GAP_SEC = 5.0

#: Dòng lệnh Windows tối đa 32767 ký tự — vượt ngưỡng này thì đưa filter vào tệp.
_INLINE_FILTER_MAX_CHARS = 24000


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
    """Nối ``videos`` (theo đúng thứ tự) thành ``output_path`` — ghi tệp tạm rồi mới thay tệp đích.

    [v3.23.659] Trước đây ffmpeg ghi thẳng ``-y`` lên tệp trọn bộ: lần nối lại bị huỷ / lỗi / tắt app phá luôn bản
    trọn bộ tốt của lần trước. Nay ghi ``<tên>.partial<đuôi>`` và chỉ thay tệp đích khi xong; phụ đề rời của các
    tập được nối kèm (nhánh chuẩn hoá trước đây bỏ mất mọi phụ đề).
    """
    if len(videos) < 2:
        raise VideoConcatError("Cần ít nhất 2 tệp để nối.")
    target = output_path.resolve()
    clash = next((v for v in videos if v.resolve() == target), None)
    if clash is not None:
        raise VideoConcatError(
            f"Tệp trọn bộ trùng với một tập đầu vào ({clash.name}). "
            "Hãy đổi tên hoặc chuyển tệp trọn bộ cũ ra khỏi thư mục rồi nối lại."
        )
    partial = output_path.with_name(f"{output_path.stem}.partial{output_path.suffix}")
    try:
        _concatenate_to(
            videos, partial, reencode=reencode, total_duration_sec=total_duration_sec,
            on_progress=on_progress, cancel_check=cancel_check, timeout_sec=timeout_sec,
        )
        _attach_episode_subtitles(videos, partial, cancel_check=cancel_check)
        _replace_with_retry(partial, output_path)
    except BaseException:
        with contextlib.suppress(OSError):
            partial.unlink(missing_ok=True)
        raise
    return output_path


def _replace_with_retry(source: Path, target: Path, attempts: int = 5) -> None:
    """``os.replace`` có thử lại (Windows: trình phát / chống virus giữ tệp đích trong chốc lát)."""
    for attempt in range(attempts):
        try:
            os.replace(source, target)
            return
        except PermissionError as exc:
            if attempt == attempts - 1:
                raise VideoConcatError(
                    f"Không thay được {target.name} (tệp đang mở ở ứng dụng khác?): {exc}"
                ) from exc
            time.sleep(0.5 * (attempt + 1))


def _concatenate_to(
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
        reencode: ``True`` = luôn chuẩn hoá + nén lại. ``False`` = tự chọn: sao chép
            luồng (nhanh) khi mọi tập cùng thông số, ngược lại chuẩn hoá (v3.23.617).
        total_duration_sec: Tổng thời lượng (để tính % tiến độ). 0 = tự tính từ ffprobe.
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
    # [v3.23.619] ffmpeg không thể vừa đọc vừa ghi cùng một tệp (lỗi mã 234 khó hiểu).
    target = output_path.resolve()
    clash = next((v for v in videos if v.resolve() == target), None)
    if clash is not None:
        raise VideoConcatError(
            f"Tệp trọn bộ trùng với một tập đầu vào ({clash.name}). "
            "Hãy đổi tên hoặc chuyển tệp trọn bộ cũ ra khỏi thư mục rồi nối lại."
        )
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        raise VideoConcatError(missing_ffmpeg_message(feature="Nối video cả bộ"))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    infos = probe_concat_inputs(videos)
    expected_sec = sum(info.duration_sec for info in infos) if infos else 0.0
    if total_duration_sec <= 0 and expected_sec > 0:
        total_duration_sec = expected_sec

    if infos is None:
        # Không có ffprobe → không biết thông số; giữ hành vi cũ (sao chép luồng / nén lại
        # qua concat demuxer). Bản đóng gói luôn kèm ffprobe nên nhánh này hiếm.
        logger.warning("Không đọc được thông số các tập (thiếu ffprobe) — nối theo cách cũ.")
        return _concat_with_demuxer(
            ffmpeg, videos, output_path, reencode, total_duration_sec,
            on_progress, cancel_check, timeout_sec,
        )

    reasons = copy_mismatch_reasons(infos)
    if reencode or reasons:
        if reasons:
            logger.info(
                "Các tập lệch thông số — chuẩn hoá rồi nối (không sao chép luồng): %s",
                "; ".join(reasons),
            )
        return _concat_normalized(
            ffmpeg, infos, output_path, total_duration_sec,
            on_progress, cancel_check, timeout_sec,
        )

    _concat_with_demuxer(
        ffmpeg, videos, output_path, False, total_duration_sec,
        on_progress, cancel_check, timeout_sec,
    )
    # Kiểm tra lại tệp ra: tổng thời lượng KHÔNG đủ — trên bộ 86 tập lỗi, tổng thời lượng
    # vẫn khớp trong khi 19 tập giữa phim không còn khung hình nào ở đúng vị trí. Quét
    # nhãn thời gian các gói hình (không giải mã, ~1s cho 2 giờ phim) để bắt khoảng trống.
    actual_sec = _probe_duration(output_path)
    max_gap_sec = _max_video_packet_gap(output_path)
    if duration_mismatch(expected_sec, actual_sec) or max_gap_sec > _MAX_FRAME_GAP_SEC:
        logger.warning(
            "Tệp nối sao chép luồng hỏng (dài %.1fs / tổng các tập %.1fs, khoảng trống "
            "hình lớn nhất %.1fs) — nối lại bằng chuẩn hoá.",
            actual_sec, expected_sec, max_gap_sec,
        )
        return _concat_normalized(
            ffmpeg, infos, output_path, total_duration_sec,
            on_progress, cancel_check, timeout_sec,
        )
    return output_path


def _concat_with_demuxer(
    ffmpeg: str,
    videos: list[Path],
    output_path: Path,
    reencode: bool,
    total_duration_sec: float,
    on_progress: ProgressCallback | None,
    cancel_check: CancelCheck | None,
    timeout_sec: float,
) -> Path:
    """Nối bằng concat demuxer (chỉ đúng khi mọi tập cùng thông số)."""
    from subtitles_extractor.application.services.concat_plan import (
        build_concat_list_content,
    )

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
        # [v3.23.659] Chọn luồng TƯỜNG MINH: mặc định ffmpeg chỉ giữ MỘT track tiếng (track nhiều kênh nhất = tiếng
        # gốc stereo) → bản "_tiengviet" nối xong mất giọng Việt; phụ đề rời chỉ giữ được khi ra .mkv.
        keep_subs = output_path.suffix.lower() == ".mkv"
        command += ["-map", "0:v:0", "-map", "0:a?"] + (["-map", "0:s?"] if keep_subs else [])
        if reencode:
            # Nén lại: chuẩn hoá về H.264 + AAC để mọi tập ghép được dù lệch thông số.
            command += ["-c:v", "libx264", "-preset", "medium", "-crf", "20",
                        "-c:a", "aac", "-b:a", "192k"] + (["-c:s", "copy"] if keep_subs else [])
        else:
            command += ["-c", "copy"]
        command += ["-movflags", "+faststart", str(output_path)]

        return _run_ffmpeg_concat(
            command, output_path, total_duration_sec,
            on_progress, cancel_check, timeout_sec,
        )
    finally:
        list_path.unlink(missing_ok=True)


def _concat_normalized(
    ffmpeg: str,
    infos: list[ConcatInputInfo],
    output_path: Path,
    total_duration_sec: float,
    on_progress: ProgressCallback | None,
    cancel_check: CancelCheck | None,
    timeout_sec: float,
) -> Path:
    """[v3.23.617] Chuẩn hoá các tập lệch thông số rồi nối thành một tệp (luôn nén lại).

    [v3.23.632] Chuẩn hoá TỪNG TẬP ra tệp riêng (có thể tiếp tục khi gián đoạn, song song nhiều
    tập) rồi nối sao chép luồng — xem :func:`_concat_by_episode`. Log thật 16/10 (86 tập, GPU):
    một lệnh ffmpeg mở 86 đầu vào cùng lúc, ``hevc_nvenc`` hỏng sau 4 phút (lý do không được
    ghi) rồi chạy lại TOÀN BỘ bằng ``libx265`` mất 39 phút. Cách cũ (một lệnh, concat filter)
    còn lại làm phương án cuối cùng.
    """
    target = choose_concat_target(infos)
    encoder = pick_video_encoder(ffmpeg, "hevc")
    attempts = [encoder]
    if is_hardware_encoder(encoder):
        attempts.append(software_fallback(encoder))
    last_error: VideoConcatError | None = None
    for attempt in attempts:
        try:
            return _concat_by_episode(
                ffmpeg, infos, target, output_path, attempt, total_duration_sec,
                on_progress, cancel_check, timeout_sec,
            )
        except VideoConcatError as exc:
            if cancel_check is not None and cancel_check():
                raise
            last_error = exc
            if is_hardware_encoder(attempt):
                logger.warning(
                    "Bộ mã hoá %s lỗi khi chuẩn hoá từng tập — chuyển sang %s. Lý do: %s",
                    attempt, software_fallback(attempt), exc,
                )
                mark_encoder_broken(ffmpeg, attempt)
            else:
                logger.warning("Chuẩn hoá từng tập thất bại (%s): %s", attempt, exc)
    logger.warning(
        "Chuẩn hoá từng tập không thành (%s) — nối bằng MỘT lệnh ffmpeg như bản cũ.", last_error
    )
    return _concat_single_graph(
        ffmpeg, infos, output_path, total_duration_sec, on_progress, cancel_check, timeout_sec
    )


#: Số tập chuẩn hoá song song: GPU 2 phiên mã hoá; CPU theo số nhân (libx265 mỗi tiến trình
#: đã đa luồng nên chỉ cần vài tập chạy cùng lúc).
_GPU_PARALLEL_PARTS = 2
_CPU_PARALLEL_CAP = 4
#: Phần tiến độ dành cho các tập chuẩn hoá (phần còn lại là bước nối sao chép luồng).
_PARTS_PROGRESS_SHARE = 0.95
#: Sai lệch thời lượng tối đa (giây) của một tập đã chuẩn hoá so với tập gốc.
_PART_DURATION_TOLERANCE_S = 1.0


def _parallel_parts(encoder: str) -> int:
    if is_hardware_encoder(encoder):
        return _GPU_PARALLEL_PARTS
    return max(1, min(_CPU_PARALLEL_CAP, (os.cpu_count() or 4) // 4))


def _part_signature(target: object, encoder: str, quality_args: list[str]) -> str:
    """Dấu của "cấu hình chuẩn hoá": đổi đích/bộ mã hoá/chất lượng thì tập cũ không dùng lại."""
    blob = f"v1|{target!r}|{encoder}|{' '.join(quality_args)}"
    return hashlib.sha1(blob.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]


def _source_stamp(info: ConcatInputInfo) -> dict[str, object]:
    try:
        stat = info.path.stat()
        return {"path": str(info.path), "size": stat.st_size, "mtime": int(stat.st_mtime)}
    except OSError:
        return {"path": str(info.path), "size": -1, "mtime": -1}


def _valid_part(part: Path, meta: Path, info: ConcatInputInfo, signature: str) -> bool:
    """Tập đã chuẩn hoá ở lần chạy trước còn dùng được (cùng nguồn, cùng cấu hình, đủ dài)."""
    try:
        saved = json.loads(meta.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if saved.get("signature") != signature or saved.get("source") != _source_stamp(info):
        return False
    if not part.is_file():
        return False
    actual = _probe_duration(part)
    return actual > 0 and abs(actual - info.duration_sec) <= _PART_DURATION_TOLERANCE_S + (
        0.005 * info.duration_sec
    )


def _concat_by_episode(
    ffmpeg: str,
    infos: list[ConcatInputInfo],
    target: object,
    output_path: Path,
    encoder: str,
    total_duration_sec: float,
    on_progress: ProgressCallback | None,
    cancel_check: CancelCheck | None,
    timeout_sec: float,
) -> Path:
    """[v3.23.632] Chuẩn hoá từng tập ra tệp riêng (tiếp tục được) rồi nối sao chép luồng.

    * Mỗi tập một lệnh ffmpeg một đầu vào (``concat=n=1`` của bộ lọc chuẩn hoá) → một tập lỗi
      không kéo cả bộ; tập đã xong ở lần chạy trước (cùng nguồn + cấu hình, đủ thời lượng) được
      BỎ QUA — bị huỷ/mất điện giữa chừng thì lần sau chạy tiếp.
    * Cùng một bộ mã hoá, cùng tham số cho mọi tập → thông số mã hoá (SPS/PPS) giống nhau nên
      nối ``-c copy`` hợp lệ; sau nối vẫn đối chiếu thời lượng + khoảng trống hình như trước.
    * Tập lỗi (kể cả bộ mã hoá GPU) → ném :class:`VideoConcatError` kèm LÝ DO ffmpeg; người gọi
      chuyển bộ mã hoá CPU cho TOÀN BỘ (trộn hai bộ mã hoá sẽ lệch SPS/PPS).
    """
    quality = encoder_quality_args(encoder, output_path=output_path)
    signature = _part_signature(target, encoder, quality)
    work = output_path.parent / f"{output_path.stem}.parts" / encoder
    work.mkdir(parents=True, exist_ok=True)
    total = total_duration_sec or sum(info.duration_sec for info in infos) or 1.0
    parts = [work / f"{index:03d}.mkv" for index in range(len(infos))]
    metas = [work / f"{index:03d}.json" for index in range(len(infos))]
    pending = [
        index for index, info in enumerate(infos)
        if not _valid_part(parts[index], metas[index], info, signature)
    ]
    reused = len(infos) - len(pending)
    logger.info(
        "Nối %d tập bằng chuẩn hoá từng tập (%s): %d tập đã có từ lần trước, %d tập cần làm, "
        "%d tập chạy song song.", len(infos), encoder, reused, len(pending), _parallel_parts(encoder),
    )
    lock = threading.Lock()
    finished = {index: infos[index].duration_sec for index in range(len(infos)) if index not in pending}
    running: dict[int, float] = {}
    stop = threading.Event()

    def report() -> None:
        if on_progress is None:
            return
        with lock:
            done = sum(finished.values()) + sum(running.values())
        on_progress(min(1.0, done / total) * _PARTS_PROGRESS_SHARE)

    def cancelled() -> bool:
        return stop.is_set() or (cancel_check is not None and cancel_check())

    def encode(index: int) -> None:
        info = infos[index]
        temp = work / f"{index:03d}.tmp.mkv"
        graph = build_normalize_filter([info], target)  # type: ignore[arg-type]
        command = _normalized_command(ffmpeg, [info], graph, None, encoder, temp)

        def part_progress(ratio: float) -> None:
            with lock:
                running[index] = ratio * info.duration_sec
            report()

        try:
            _run_ffmpeg_concat(
                command, temp, info.duration_sec, part_progress, cancelled, timeout_sec
            )
        except VideoConcatError as exc:
            temp.unlink(missing_ok=True)
            stop.set()
            raise VideoConcatError(f"Tập {info.path.name}: {exc}") from exc
        os.replace(temp, parts[index])
        metas[index].write_text(
            json.dumps({"signature": signature, "source": _source_stamp(info)}),
            encoding="utf-8",
        )
        with lock:
            running.pop(index, None)
            finished[index] = info.duration_sec
        report()

    if pending:
        workers = _parallel_parts(encoder)
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="concat-part") as pool:
            futures = [pool.submit(encode, index) for index in pending]
            errors: list[BaseException] = []
            for future in futures:
                try:
                    future.result()
                except BaseException as exc:  # noqa: BLE001 — gom rồi ném lỗi đầu tiên
                    errors.append(exc)
                    stop.set()
        if errors:
            if cancel_check is not None and cancel_check():
                raise VideoConcatError("Đã huỷ nối video.")
            # Ưu tiên lỗi có LÝ DO từ ffmpeg (các lỗi còn lại chỉ là hệ quả của việc dừng chung).
            first = next((e for e in errors if "mã" in str(e)), errors[0])
            raise first if isinstance(first, VideoConcatError) else VideoConcatError(str(first))

    def merge_progress(ratio: float) -> None:
        if on_progress is not None:
            on_progress(_PARTS_PROGRESS_SHARE + (1.0 - _PARTS_PROGRESS_SHARE) * ratio)

    _concat_with_demuxer(
        ffmpeg, parts, output_path, False, total, merge_progress, cancel_check, timeout_sec
    )
    expected = sum(info.duration_sec for info in infos)
    actual = _probe_duration(output_path)
    gap = _max_video_packet_gap(output_path)
    if duration_mismatch(expected, actual) or gap > _MAX_FRAME_GAP_SEC:
        output_path.unlink(missing_ok=True)
        raise VideoConcatError(
            f"Tệp nối từ các tập đã chuẩn hoá không hợp lệ (dài {actual:.1f}s / {expected:.1f}s, "
            f"khoảng trống hình lớn nhất {gap:.1f}s)."
        )
    shutil.rmtree(output_path.parent / f"{output_path.stem}.parts", ignore_errors=True)
    if on_progress is not None:
        on_progress(1.0)
    return output_path


def _concat_single_graph(
    ffmpeg: str,
    infos: list[ConcatInputInfo],
    output_path: Path,
    total_duration_sec: float,
    on_progress: ProgressCallback | None,
    cancel_check: CancelCheck | None,
    timeout_sec: float,
) -> Path:
    """[v3.23.617] Phương án cuối: MỘT lệnh ffmpeg, concat filter trên mọi tập (luôn nén lại).

    [v3.23.618] Nén bằng HEVC, ưu tiên GPU; GPU lỗi giữa chừng → chạy lại bằng CPU.
    """
    target = choose_concat_target(infos)
    filter_graph = build_normalize_filter(infos, target)
    logger.info(
        "Nối %d tập bằng chuẩn hoá về %dx%d @ %s fps, tiếng %d Hz %s.",
        len(infos), target.width, target.height, target.frame_rate,
        target.sample_rate, target.channel_layout,
    )

    # [v3.23.618] HEVC, ưu tiên GPU (NVENC/QSV/AMF/VideoToolbox) nếu máy có; không có thì
    # libx265. Trước đây luôn libx264 crf 20: bộ 86 tập 736 MB thành 2.75 GB.
    encoder = pick_video_encoder(ffmpeg, "hevc")
    logger.info("Bộ mã hoá cho bản nối chuẩn hoá: %s", encoder)

    script_path: Path | None = None
    if len(filter_graph) > _INLINE_FILTER_MAX_CHARS:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", prefix="concat_filter_", delete=False,
            encoding="utf-8",
        ) as handle:
            handle.write(filter_graph)
            script_path = Path(handle.name)
    try:
        try:
            return _run_ffmpeg_concat(
                _normalized_command(ffmpeg, infos, filter_graph, script_path, encoder,
                                    output_path),
                output_path, total_duration_sec, on_progress, cancel_check, timeout_sec,
            )
        except VideoConcatError as exc:
            cancelled = cancel_check is not None and cancel_check()
            if cancelled or not is_hardware_encoder(encoder):
                raise
            # Bộ mã hoá GPU thử được nhưng lỗi khi chạy thật (hết phiên, thiếu VRAM…).
            fallback = software_fallback(encoder)
            logger.warning(
                "Bộ mã hoá %s lỗi khi nối — chạy lại bằng %s. Lý do: %s", encoder, fallback, exc,
            )
            mark_encoder_broken(ffmpeg, encoder)
            return _run_ffmpeg_concat(
                _normalized_command(ffmpeg, infos, filter_graph, script_path, fallback,
                                    output_path),
                output_path, total_duration_sec, on_progress, cancel_check, timeout_sec,
            )
    finally:
        if script_path is not None:
            script_path.unlink(missing_ok=True)


def _normalized_command(
    ffmpeg: str,
    infos: list[ConcatInputInfo],
    filter_graph: str,
    script_path: Path | None,
    encoder: str,
    output_path: Path,
) -> list[str]:
    command = [ffmpeg, "-hide_banner", "-y"]
    for info in infos:
        command += ["-i", str(info.path)]
    if script_path is not None:
        command += _filter_script_args(ffmpeg, script_path)
    else:
        command += ["-filter_complex", filter_graph]
    command += [
        "-map", "[vout]", "-map", "[aout]",
        "-c:v", encoder, *encoder_quality_args(encoder, output_path=output_path),
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart", str(output_path),
    ]
    return command


def _ffmpeg_major_version(ffmpeg: str) -> int:
    """Số phiên bản chính của ffmpeg; bản dựng master ("N-xxxxx") coi như mới nhất."""
    try:
        result = subprocess.run(
            [ffmpeg, "-hide_banner", "-version"], capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=20, **no_window_kwargs(),
        )
    except (OSError, subprocess.SubprocessError):
        return 0
    first_line = (result.stdout or "").splitlines()[0] if result.stdout else ""
    match = re.search(r"version\s+n?(\d+)\.", first_line)
    if match:
        return int(match.group(1))
    if re.search(r"version\s+N-", first_line):
        return 99
    return 0


def _filter_script_args(ffmpeg: str, script_path: Path) -> list[str]:
    """Tham số đọc filter_complex từ tệp: ``-/filter_complex`` (ffmpeg ≥ 7) hoặc
    ``-filter_complex_script`` (bản cũ; đã bị bỏ ở các bản mới)."""
    if _ffmpeg_major_version(ffmpeg) >= 7:
        return ["-/filter_complex", str(script_path)]
    return ["-filter_complex_script", str(script_path)]


def probe_concat_inputs(videos: list[Path]) -> list[ConcatInputInfo] | None:
    """[v3.23.617] Đọc thông số từng tập bằng ffprobe; ``None`` nếu thiếu ffprobe.

    Raises:
        VideoConcatError: Một tệp không đọc được hoặc không có luồng hình.
    """
    ffprobe = find_ffprobe()
    if ffprobe is None:
        return None
    infos: list[ConcatInputInfo] = []
    for video in videos:
        try:
            result = subprocess.run(
                [ffprobe, "-v", "error", "-show_format", "-show_streams", "-show_data",
                 "-of", "json", str(video)],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=120, **no_window_kwargs(),
            )
            payload = json.loads(result.stdout or "{}")
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
            raise VideoConcatError(f"Không đọc được tệp {video.name}: {exc}") from exc
        infos.append(_info_from_probe(video, payload))
    return infos


def _info_from_probe(video: Path, payload: dict) -> ConcatInputInfo:
    streams = payload.get("streams") or []
    v_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    a_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if v_stream is None:
        raise VideoConcatError(f"Tệp {video.name} không có luồng hình.")
    try:
        duration = float((payload.get("format") or {}).get("duration") or 0.0)
    except (TypeError, ValueError):
        duration = 0.0
    extradata = v_stream.get("extradata") or ""
    extradata_hash = hashlib.md5(extradata.encode("utf-8")).hexdigest() if extradata else ""
    sar = v_stream.get("sample_aspect_ratio") or "1:1"
    if sar in ("0:1", "N/A"):
        sar = "1:1"
    return ConcatInputInfo(
        path=video,
        duration_sec=duration,
        video_codec=v_stream.get("codec_name", ""),
        width=int(v_stream.get("width") or 0),
        height=int(v_stream.get("height") or 0),
        pix_fmt=v_stream.get("pix_fmt", ""),
        frame_rate=v_stream.get("r_frame_rate", "0/0"),
        time_base=v_stream.get("time_base", ""),
        extradata_hash=extradata_hash,
        sar=sar,
        audio_codec=(a_stream or {}).get("codec_name", ""),
        sample_rate=int((a_stream or {}).get("sample_rate") or 0),
        channels=int((a_stream or {}).get("channels") or 0),
    )


def _max_video_packet_gap(path: Path) -> float:
    """Khoảng trống lớn nhất (giây) giữa các nhãn thời gian gói hình; 0.0 nếu không đọc được."""
    ffprobe = find_ffprobe()
    if ffprobe is None:
        return 0.0
    try:
        result = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "packet=pts_time", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=600, **no_window_kwargs(),
        )
    except (OSError, subprocess.SubprocessError):
        return 0.0
    stamps: list[float] = []
    for token in (result.stdout or "").split():
        token = token.strip().strip(",")
        try:
            stamps.append(float(token))
        except ValueError:
            continue
    if len(stamps) < 2:
        return 0.0
    stamps.sort()
    return max(later - earlier for earlier, later in pairwise(stamps))


def _stream_count(path: Path, kind: str) -> int:
    """Số luồng loại ``kind`` (``s`` / ``a`` / ``v``) trong tệp; 0 nếu không đọc được."""
    ffprobe = find_ffprobe()
    if ffprobe is None:
        return 0
    try:
        result = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", kind, "-show_entries", "stream=index", "-of", "csv=p=0",
             str(path)],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60, **no_window_kwargs(),
        )
    except (OSError, subprocess.SubprocessError):
        return 0
    return sum(1 for line in (result.stdout or "").splitlines() if line.strip())


def _attach_episode_subtitles(videos: list[Path], output_path: Path, cancel_check: CancelCheck | None = None) -> None:
    """[v3.23.659] Tệp nối chưa có phụ đề mà MỌI tập đều có → nối riêng phụ đề (lệch đúng thời lượng từng tập) rồi gắn.

    Nhánh chuẩn hoá (tập lệch fps — bộ mẫu có tập 30 fps xen 25 fps) chỉ nối hình + tiếng, nên bản "_phude" trọn
    bộ trước đây không còn phụ đề. Lỗi ở bước này chỉ cảnh báo: tệp hình + tiếng vẫn dùng được.
    """
    ffmpeg = find_ffmpeg()
    if (
        ffmpeg is None or output_path.suffix.lower() != ".mkv" or _stream_count(output_path, "s") > 0
        or not all(_stream_count(video, "s") > 0 for video in videos)
    ):
        return
    from subtitles_extractor.application.services.concat_plan import build_concat_list_content

    def cancelled() -> bool:
        return cancel_check is not None and bool(cancel_check())

    durations = []
    for video in videos:
        if cancelled():
            raise VideoConcatError("Đã huỷ nối video.")
        durations.append(_probe_duration(video))
    if any(d <= 0 for d in durations):
        logger.warning("Không đọc được thời lượng tập — không gắn được phụ đề vào tệp nối.")
        return
    # Mỗi tập kèm "duration" để phụ đề tập sau lệch đúng bằng thời lượng tập trước (không theo gói phụ đề cuối).
    lines = build_concat_list_content(videos).splitlines()
    listing = "\n".join(f"{line}\nduration {d:.6f}" for line, d in zip(lines, durations, strict=False)) + "\n"
    work = output_path.with_name(output_path.stem + ".subs")
    work.mkdir(exist_ok=True)
    list_path = work / "list.txt"
    subs_path = work / "subs.mkv"
    muxed = work / ("muxed" + output_path.suffix)
    try:
        list_path.write_text(listing, encoding="utf-8")
        for command in (
            [ffmpeg, "-hide_banner", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(list_path),
             "-map", "0:s", "-c", "copy", str(subs_path)],
            [ffmpeg, "-hide_banner", "-v", "error", "-y", "-i", str(output_path), "-i", str(subs_path),
             "-map", "0", "-map", "1:s", "-c", "copy", str(muxed)],
        ):
            # [v3.23.659] Chạy có thể huỷ (bước gắn chép lại cả tệp trọn bộ — vài phút với 86 tập).
            returncode, stderr = _run_cancellable(command, cancelled, timeout_sec=1800)
            if returncode != 0:
                logger.warning("Không gắn được phụ đề vào tệp nối: %s", stderr[-400:])
                return
        os.replace(muxed, output_path)
        logger.info("Đã gắn phụ đề của %d tập vào tệp nối.", len(videos))
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("Không gắn được phụ đề vào tệp nối: %s", exc)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _run_cancellable(command: list[str], cancelled, timeout_sec: float) -> tuple[int, str]:
    """Chạy ffmpeg, dừng (kill) khi người dùng huỷ → ``VideoConcatError``."""
    process = subprocess.Popen(
        command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
        **no_window_kwargs(),
    )
    deadline = time.monotonic() + timeout_sec
    try:
        while True:
            try:
                _out, err = process.communicate(timeout=0.5)
                return process.returncode, err or ""
            except subprocess.TimeoutExpired:
                if cancelled():
                    raise VideoConcatError("Đã huỷ nối video.") from None
                if time.monotonic() > deadline:
                    raise
    finally:
        if process.poll() is None:
            process.kill()
            with contextlib.suppress(Exception):
                process.communicate(timeout=10)


def _probe_duration(path: Path) -> float:
    """Thời lượng tệp (giây); 0.0 nếu không đọc được."""
    ffprobe = find_ffprobe()
    if ffprobe is None:
        return 0.0
    try:
        result = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(path)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=120, **no_window_kwargs(),
        )
        return float((result.stdout or "0").strip() or 0.0)
    except (OSError, subprocess.SubprocessError, ValueError):
        return 0.0


#: [v3.23.635] ffmpeg im lặng (không in dòng tiến độ nào) quá chừng này giây thì coi là TREO. Bình thường
#: ffmpeg in dòng ``time=`` vài lần mỗi giây; thật sự treo (driver NVENC kẹt, x265 kẹt luồng) thì trước
#: đây vòng đọc stderr chặn mãi nên cả ``timeout_sec`` lẫn nút huỷ đều không có tác dụng.
FFMPEG_STALL_TIMEOUT_SEC = 180.0
_POLL_INTERVAL_SEC = 0.25


def _run_ffmpeg_concat(
    command: list[str],
    output_path: Path,
    total_duration_sec: float,
    on_progress: ProgressCallback | None,
    cancel_check: CancelCheck | None,
    timeout_sec: float,
    stall_timeout_sec: float = FFMPEG_STALL_TIMEOUT_SEC,
) -> Path:
    """Chạy lệnh ffmpeg concat, đọc tiến độ từ stderr, hỗ trợ huỷ, hạn giờ và phát hiện treo.

    [v3.23.635] Đọc stderr ở LUỒNG RIÊNG còn luồng chính hỏi định kỳ: huỷ, hết ``timeout_sec``, hoặc
    ffmpeg im lặng quá ``stall_timeout_sec`` đều giết tiến trình và báo :class:`VideoConcatError` (để
    chuỗi dự phòng phần cứng → phần mềm chạy tiếp). Trước đây ``for line in process.stderr`` chặn tới
    khi ffmpeg đóng stderr nên ffmpeg treo làm treo cả ứng dụng. stdin = DEVNULL để ffmpeg không đọc
    bàn phím/tty của tiến trình cha.
    """
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        **no_window_kwargs(),
    )
    assert process.stderr is not None
    lines: queue.Queue[str | None] = queue.Queue()

    def _pump() -> None:
        try:
            for line in process.stderr:  # type: ignore[union-attr]
                lines.put(line)
        finally:
            lines.put(None)

    reader = threading.Thread(target=_pump, name="ffmpeg-stderr", daemon=True)
    reader.start()
    tail: list[str] = []
    started = last_output = time.monotonic()

    def _abort(message: str) -> VideoConcatError:
        process.kill()
        with contextlib.suppress(subprocess.TimeoutExpired, OSError):
            process.wait(timeout=10)
        return VideoConcatError(message)

    try:
        while True:
            try:
                line = lines.get(timeout=_POLL_INTERVAL_SEC)
            except queue.Empty:
                line = ""
                now = time.monotonic()
                if cancel_check is not None and cancel_check():
                    raise _abort("Đã huỷ nối video.") from None
                if now - started > timeout_sec:
                    raise _abort("Nối video quá thời gian cho phép.") from None
                if now - last_output > stall_timeout_sec:
                    raise _abort(
                        f"ffmpeg treo: không có dòng tiến độ nào trong {stall_timeout_sec:.0f} s "
                        f"(đã chạy {now - started:.0f} s)."
                    ) from None
                continue
            if line is None:
                break
            last_output = time.monotonic()
            tail.append(line)
            if len(tail) > 40:
                tail.pop(0)
            if cancel_check is not None and cancel_check():
                raise _abort("Đã huỷ nối video.")
            if on_progress is not None and total_duration_sec > 0:
                processed = _parse_progress_seconds(line)
                if processed is not None:
                    on_progress(min(1.0, processed / total_duration_sec))
        try:
            return_code = process.wait(timeout=max(1.0, timeout_sec - (time.monotonic() - started)))
        except subprocess.TimeoutExpired as exc:
            raise _abort("Nối video quá thời gian cho phép.") from exc
    except BaseException:
        if process.poll() is None:
            process.kill()
        raise

    if return_code != 0:
        detail = "".join(tail[-6:]).strip()
        raise VideoConcatError(f"ffmpeg lỗi (mã {return_code}): {detail}")
    if not output_path.is_file():
        raise VideoConcatError("ffmpeg báo xong nhưng không thấy tệp trọn bộ.")
    if on_progress is not None:
        on_progress(1.0)
    return output_path


__all__ = ["VideoConcatError", "concatenate_videos", "probe_concat_inputs"]
