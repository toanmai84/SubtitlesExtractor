"""[v3.23.728] Xuất video VÙNG ROI (đủ mọi khung) kèm dữ liệu OCR thô — thay cho ảnh JPG từng khung.

Thang chạy (dừng ở cách đầu tiên chạy được):

1. ``gpu_zero_copy`` — NVDEC giải mã VÀ cắt ROI ngay trong bộ giải mã (``<codec>_cuvid -crop``), khung nằm nguyên trên
   VRAM (``-hwaccel_output_format cuda``) và đi thẳng vào NVENC: không chép khung nào về RAM.
2. ``gpu_hybrid`` — NVDEC giải mã, cắt trên CPU, NVENC mã hoá (codec nguồn không có ``*_cuvid`` / driver từ chối cắt).
3. ``cpu`` — giải mã + mã hoá CPU (``libx265``; bản ffmpeg không có x265 thì ``libx264``).

Chất lượng: thang CRF 26 cho HEVC (đo tập 24 phim mẫu, ROI 720×136: 1,4 MB / 64 s, PSNR độ chói 39,5 dB, chữ rõ nét —
đủ để OCR lại và đo lại điểm sáng khi gỡ lỗi). Khung khoá mỗi 2 s để trang Gỡ lỗi OCR nhảy khung nhanh.

[v3.23.735] Video đổi kích thước khung giữa chừng (phim nối từ nhiều tập có tập độ phân giải khác — đo thật: tập
480×854 / 30 fps giữa phim 720×1280 / 25 fps): dò bằng tách gói (bộ tham số SPS mới trong luồng, ≈ 2 s cho phim 3 giờ).
Có đổi → bỏ ``gpu_zero_copy`` (``-crop`` trong bộ giải mã cố định theo kích thước đầu), mọi khung được ``scale`` về
kích thước gốc rồi mới ``crop`` (bộ lọc ``scale`` tự bỏ qua khi khung đã đúng kích thước). ffmpeg im lặng quá
``STALL_TIMEOUT_SEC`` → huỷ, thử cách kế; stderr luôn được đọc ở luồng riêng (trước đây không đọc: ống đầy → treo).

Mốc thời gian: bộ giải mã của ứng dụng ghi mốc theo PTS thật của luồng; ffmpeg dời mốc đầu tệp ra về 0 — tệp mô tả
``.roi.json`` ghi ``source_start_sec`` để quy đổi (mốc trong clip = mốc seraw − ``source_start_sec``).
"""

from __future__ import annotations

import json
import logging
import queue
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from subtitles_extractor.domain.ports.roi_clip_exporter_port import (
    RoiClipExporterPort,
    RoiClipResult,
)
from subtitles_extractor.domain.value_objects.roi import Roi

logger = logging.getLogger(__name__)

#: Mức chất lượng trên thang CRF của x264 (HEVC tự cộng 2 → CRF 26) — xem docstring module.
CLIP_QUALITY: Final[int] = 24
#: Khoảng khung khoá (khung): 2 s ở 25 fps.
KEYFRAME_INTERVAL: Final[int] = 50
#: [v3.23.735] ffmpeg không tiến thêm (mốc ``out_time`` đứng yên) quá chừng này giây → coi là treo: huỷ, thử cách kế.
STALL_TIMEOUT_SEC: float = 90.0
#: Bộ giải mã NVDEC của ffmpeg theo codec nguồn.
CUVID_DECODERS: Final[dict[str, str]] = {
    "h264": "h264_cuvid", "hevc": "hevc_cuvid", "av1": "av1_cuvid", "vp9": "vp9_cuvid", "vp8": "vp8_cuvid",
    "mpeg2video": "mpeg2_cuvid", "mpeg4": "mpeg4_cuvid", "vc1": "vc1_cuvid",
}

__all__ = [
    "CLIP_QUALITY", "CUVID_DECODERS", "STALL_TIMEOUT_SEC", "ClipAttempt", "FfmpegRoiClipExporter", "build_attempts",
    "clip_paths_for", "detect_size_changes", "even_crop",
]


def clip_paths_for(raw_ocr_path: Path) -> tuple[Path, Path]:
    """``(video, mô tả)`` đi kèm một tệp OCR thô: ``phim.seraw.json.gz`` → ``phim.roi.mp4`` + ``phim.roi.json``."""
    name = raw_ocr_path.name
    for suffix in (".seraw.json.gz", ".seraw.json", ".json.gz", ".json", ".gz"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return raw_ocr_path.with_name(f"{name}.roi.mp4"), raw_ocr_path.with_name(f"{name}.roi.json")


def even_crop(roi: Roi | None, width: int, height: int) -> tuple[int, int, int, int]:
    """Vùng cắt ``(x, y, w, h)`` chẵn (NVDEC / yuv420 cần kích thước chẵn), PHỦ trọn ROI, nằm trong khung."""
    if roi is None:
        return 0, 0, width - width % 2, height - height % 2
    x0, y0 = max(0, roi.x) // 2 * 2, max(0, roi.y) // 2 * 2
    x1 = min(width, roi.x + roi.width)
    y1 = min(height, roi.y + roi.height)
    w, h = x1 - x0, y1 - y0
    if w % 2:
        w = w + 1 if x0 + w + 1 <= width else w - 1
    if h % 2:
        h = h + 1 if y0 + h + 1 <= height else h - 1
    return x0, y0, max(2, w), max(2, h)


@dataclass(frozen=True, slots=True)
class ClipAttempt:
    """Một cách chạy ffmpeg trong thang."""

    pipeline: str
    encoder: str
    command: list[str]


def _base(ffmpeg: str) -> list[str]:
    return [ffmpeg, "-hide_banner", "-nostdin", "-y", "-loglevel", "error", "-progress", "pipe:1", "-nostats"]


def _output(encoder: str, output_path: Path, quality_args: Sequence[str]) -> list[str]:
    return [
        "-map", "0:v:0", "-an", "-sn", "-dn", "-fps_mode", "passthrough",
        "-c:v", encoder, *quality_args, "-g", str(KEYFRAME_INTERVAL),
        "-movflags", "+faststart", str(output_path),
    ]


def build_attempts(
    ffmpeg: str, video_path: Path, crop: tuple[int, int, int, int], frame_size: tuple[int, int], source_codec: str,
    output_path: Path, *, hardware_encoder: str | None, software_encoder: str,
    quality_args: Callable[[str], list[str]], size_changes: bool = False,
) -> list[ClipAttempt]:
    """Thang lệnh ffmpeg theo thứ tự ưu tiên (hàm thuần — test được không cần GPU).

    ``size_changes``: [v3.23.735] video đổi kích thước khung giữa chừng → không cắt trong bộ giải mã (toạ độ cố định
    theo kích thước đầu). Đường lọc luôn ``scale`` về ``frame_size`` trước ``crop`` (không tốn gì khi đúng cỡ).
    """
    x, y, w, h = crop
    width, height = frame_size
    crop_filter = f"scale={width}:{height},crop={w}:{h}:{x}:{y}"
    attempts: list[ClipAttempt] = []
    if hardware_encoder and hardware_encoder.endswith("_nvenc"):
        cuvid = None if size_changes else CUVID_DECODERS.get(source_codec.lower())
        if cuvid:
            top, bottom, left, right = y, height - y - h, x, width - x - w
            attempts.append(ClipAttempt("gpu_zero_copy", hardware_encoder, [
                *_base(ffmpeg), "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
                "-c:v", cuvid, "-crop", f"{top}x{bottom}x{left}x{right}", "-i", str(video_path),
                *_output(hardware_encoder, output_path, quality_args(hardware_encoder)),
            ]))
        attempts.append(ClipAttempt("gpu_hybrid", hardware_encoder, [
            *_base(ffmpeg), "-hwaccel", "cuda", "-i", str(video_path), "-vf", crop_filter,
            *_output(hardware_encoder, output_path, [*quality_args(hardware_encoder)]),
        ]))
    elif hardware_encoder:
        attempts.append(ClipAttempt("gpu_hybrid", hardware_encoder, [
            *_base(ffmpeg), "-i", str(video_path), "-vf", f"{crop_filter},format=nv12",
            *_output(hardware_encoder, output_path, quality_args(hardware_encoder)),
        ]))
    attempts.append(ClipAttempt("cpu", software_encoder, [
        *_base(ffmpeg), "-i", str(video_path), "-vf", crop_filter, "-pix_fmt", "yuv420p",
        *_output(software_encoder, output_path, quality_args(software_encoder)),
    ]))
    return attempts


def _probe(video_path: Path) -> tuple[int, int, str, float, float]:
    """``(rộng, cao, codec, thời lượng, mốc đầu tệp)`` bằng PyAV."""
    import av

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        width = int(stream.codec_context.width)
        height = int(stream.codec_context.height)
        codec = str(stream.codec_context.name or "")
        duration = float(container.duration or 0) / 1e6
        start = float(container.start_time or 0) / 1e6
    return width, height, codec, duration, start


#: NAL bộ tham số chuỗi (SPS) theo codec: H.264 loại 7 (5 bit thấp), HEVC loại 33 (6 bit sau bit cấm).
_SPS_NAL: Final[dict[str, Callable[[int], bool]]] = {
    "h264": lambda b0: (b0 & 0x1F) == 7,
    "hevc": lambda b0: ((b0 >> 1) & 0x3F) == 33,
}


def _nal_units(data: bytes, length_size: int) -> list[bytes]:
    """Tách NAL của một gói: dạng độ dài đứng trước (mp4/mkv, ``length_size`` byte) hoặc Annex-B (mã bắt đầu)."""
    units: list[bytes] = []
    if length_size:
        pos = 0
        while pos + length_size <= len(data):
            size = int.from_bytes(data[pos:pos + length_size], "big")
            pos += length_size
            if size <= 0 or pos + size > len(data):
                break
            units.append(data[pos:pos + size])
            pos += size
        return units
    for chunk in data.replace(b"\x00\x00\x00\x01", b"\x00\x00\x01").split(b"\x00\x00\x01"):
        if chunk:
            units.append(chunk)
    return units


def _length_size(codec: str, extradata: bytes) -> int:
    """Số byte độ dài NAL theo ``avcC``/``hvcC``; 0 = luồng Annex-B (không có cấu hình dạng hộp)."""
    if not extradata or extradata[0] != 1:
        return 0
    if codec == "h264" and len(extradata) > 4:
        return (extradata[4] & 0x03) + 1
    if codec == "hevc" and len(extradata) > 21:
        return (extradata[21] & 0x03) + 1
    return 0


def detect_size_changes(video_path: Path) -> list[tuple[float, int, int]]:
    """[v3.23.735] Các mốc video ĐỔI kích thước khung: ``[(giây, rộng, cao), …]`` (rỗng = cỡ khung không đổi).

    Chỉ tách gói (không giải mã cả phim — phim 2 giờ 49 phút ≈ 2 s): gói khung khoá mang SPS khác SPS trước → giải mã
    đúng một khung ở đó để đọc kích thước thật (SPS đổi mà cỡ không đổi thì bỏ qua). Codec ngoài H.264/HEVC → rỗng.
    """
    import av

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        codec = str(stream.codec_context.name or "")
        is_sps = _SPS_NAL.get(codec)
        if is_sps is None:
            return []
        base = (int(stream.codec_context.width), int(stream.codec_context.height))
        length_size = _length_size(codec, bytes(stream.codec_context.extradata or b""))
        time_base = float(stream.time_base or 0)
        candidates: list[float] = []
        last_sps: bytes | None = None
        for packet in container.demux(stream):
            if not packet.is_keyframe or packet.size == 0 or packet.pts is None:
                continue
            sps = next((nal for nal in _nal_units(bytes(packet), length_size) if is_sps(nal[0])), None)
            if sps is None:
                continue
            if last_sps is not None and sps != last_sps:
                candidates.append(float(packet.pts) * time_base)
            last_sps = sps
    changes: list[tuple[float, int, int]] = []
    current = base
    for moment in candidates:
        size = _frame_size_at(video_path, moment)
        if size is not None and size != current:
            changes.append((round(moment, 3), size[0], size[1]))
            current = size
    return changes


def _frame_size_at(video_path: Path, moment: float) -> tuple[int, int] | None:
    """Kích thước khung đầu tiên từ mốc ``moment`` (giây) — giải mã đúng một khung."""
    import av

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        time_base = float(stream.time_base or 0)
        if time_base <= 0:
            return None
        container.seek(int(moment / time_base), stream=stream, backward=True, any_frame=False)
        for frame in container.decode(stream):
            if frame.pts is not None and float(frame.pts) * time_base >= moment - 1e-3:
                return int(frame.width), int(frame.height)
    return None


class FfmpegRoiClipExporter(RoiClipExporterPort):
    """Cắt vùng ROI thành video nhỏ bằng ffmpeg — GPU zero-copy khi được chọn chạy GPU."""

    def __init__(self, ffmpeg_finder: Callable[[], str | None] | None = None) -> None:
        if ffmpeg_finder is None:
            from subtitles_extractor.infrastructure.media.ffmpeg_locator import (
                find_ffmpeg,
            )

            ffmpeg_finder = find_ffmpeg
        self._find_ffmpeg = ffmpeg_finder

    def export(
        self, video_path: Path, roi: Roi | None, output_path: Path, *, prefer_gpu: bool,
        progress: Callable[[float], bool] | None = None, quality: int | None = None,
    ) -> RoiClipResult | None:
        from subtitles_extractor.infrastructure.video.video_encoder_selector import (
            encoder_quality_args,
            is_hardware_encoder,
            mark_encoder_broken,
            pick_video_encoder,
        )

        ffmpeg = self._find_ffmpeg()
        if not ffmpeg:
            logger.warning("Không tìm thấy ffmpeg — bỏ qua xuất video vùng ROI.")
            return None
        try:
            width, height, codec, duration, start = _probe(video_path)
        except Exception as exc:  # noqa: BLE001 — video hỏng / không đọc được: chỉ mất phần phụ
            logger.warning("Không đọc được video để xuất vùng ROI: %s", exc)
            return None
        crop = even_crop(roi, width, height)
        try:
            size_changes = detect_size_changes(video_path)
        except Exception as exc:  # noqa: BLE001 — chỉ là dò trước: lỗi thì coi như cỡ không đổi (còn canh treo)
            logger.debug("Không dò được đổi kích thước khung: %s", exc)
            size_changes = []
        if size_changes:
            logger.info(
                "Video đổi kích thước khung giữa chừng (%s) — mọi khung được đưa về %dx%d trước khi cắt ROI; bỏ cắt "
                "trong bộ giải mã (GPU zero-copy) vì toạ độ cắt cố định theo kích thước đầu.",
                ", ".join(f"{t:.2f} s → {w}x{h}" for t, w, h in size_changes[:6]), width, height,
            )
        chosen = pick_video_encoder(ffmpeg, "hevc", allow_hardware=prefer_gpu)
        software = pick_video_encoder(ffmpeg, "hevc", allow_hardware=False)
        hardware = chosen if is_hardware_encoder(chosen) else None
        attempts = build_attempts(
            ffmpeg, video_path, crop, (width, height), codec, output_path,
            hardware_encoder=hardware, software_encoder=software,
            quality_args=lambda encoder: encoder_quality_args(
                encoder, int(quality) if quality else CLIP_QUALITY, output_path=output_path,
            ),
            size_changes=bool(size_changes),
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        for attempt in attempts:
            status = self._run(attempt.command, duration, progress)
            if status == "cancelled":
                output_path.unlink(missing_ok=True)
                return None
            if status == "ok" and output_path.is_file() and output_path.stat().st_size > 0:
                elapsed = time.perf_counter() - started
                sidecar = output_path.with_suffix(".json")       # phim.roi.mp4 → phim.roi.json (như clip_paths_for)
                sidecar.write_text(json.dumps({
                    "schema": 1, "source": video_path.name, "crop_xywh": list(crop),
                    "roi_xywh": [roi.x, roi.y, roi.width, roi.height] if roi else None,
                    "source_start_sec": start, "source_size": [width, height],
                    "encoder": attempt.encoder, "pipeline": attempt.pipeline, "elapsed_sec": round(elapsed, 2),
                    "size_changes": [[t, w, h] for t, w, h in size_changes],
                }, ensure_ascii=False, indent=1), encoding="utf-8")
                logger.info(
                    "Đã xuất video vùng ROI %s (%dx%d, %s, %s) trong %.1f s: %.1f MB.", output_path.name, crop[2], crop[3],
                    attempt.encoder, attempt.pipeline, elapsed, output_path.stat().st_size / 1e6,
                )
                return RoiClipResult(output_path, sidecar, crop, attempt.encoder, attempt.pipeline, elapsed)
            output_path.unlink(missing_ok=True)
            logger.info(
                "Xuất video vùng ROI bằng %s (%s) %s — thử cách kế.", attempt.encoder, attempt.pipeline,
                f"bị treo (không tiến thêm {STALL_TIMEOUT_SEC:.0f} s), đã huỷ" if status == "stalled" else "không được",
            )
            if attempt.pipeline == "gpu_hybrid" and attempt.encoder:
                mark_encoder_broken(ffmpeg, attempt.encoder)
        logger.warning("Không xuất được video vùng ROI bằng mọi cách.")
        return None

    @staticmethod
    def _run(command: list[str], duration: float, progress: Callable[[float], bool] | None) -> str:
        """Chạy ffmpeg, đọc ``-progress`` để báo tiến độ; ``"ok"`` / ``"failed"`` / ``"stalled"`` / ``"cancelled"``.

        [v3.23.735] stderr được đọc liên tục ở luồng riêng (giữ 40 dòng cuối cho log): trước đây không đọc → ffmpeg in
        lỗi đầy ống (Windows ≈ 4 KB) rồi đứng chờ, ứng dụng đứng chờ dòng tiến độ → treo vĩnh viễn ở 39 %. Mốc
        ``out_time`` đứng yên quá ``STALL_TIMEOUT_SEC`` → huỷ (``"stalled"``). Huỷ của người dùng được kiểm mỗi giây kể
        cả khi ffmpeg im lặng.
        """
        from subtitles_extractor.infrastructure.process.hidden_process import popen_hidden

        try:
            proc = popen_hidden(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL,
                text=True, encoding="utf-8", errors="replace",
            )
        except OSError as exc:
            logger.debug("Không chạy được ffmpeg: %s", exc)
            return "failed"
        assert proc.stdout is not None and proc.stderr is not None
        stdout, stderr = proc.stdout, proc.stderr
        tail: deque[str] = deque(maxlen=40)
        lines: queue.Queue[str | None] = queue.Queue()

        def _drain_stderr() -> None:
            for text in stderr:
                tail.append(text)

        def _pump_stdout() -> None:
            try:
                for text in stdout:
                    lines.put(text)
            finally:
                lines.put(None)

        readers = [threading.Thread(target=_drain_stderr, name="ffmpeg-stderr", daemon=True),
                   threading.Thread(target=_pump_stdout, name="ffmpeg-progress", daemon=True)]
        for reader in readers:
            reader.start()
        status = ""
        fraction = 0.0
        last_done = -1.0
        last_advance = time.monotonic()
        try:
            while True:
                try:
                    line = lines.get(timeout=1.0)
                except queue.Empty:
                    line = ""
                if line is None:
                    break
                if line.startswith("out_time_us="):
                    try:
                        done = int(line.split("=", 1)[1]) / 1e6
                    except ValueError:
                        done = last_done
                    if done > last_done:
                        last_done, last_advance = done, time.monotonic()
                        fraction = min(1.0, done / duration) if duration > 0 else 0.0
                elif line.startswith("progress=end"):
                    last_advance = time.monotonic()                 # đang ghi chỉ mục cuối tệp: không phải treo
                if progress is not None and (line == "" or line.startswith("out_time_us=")):
                    if progress(fraction) is False:
                        status = "cancelled"
                        break
                if time.monotonic() - last_advance > STALL_TIMEOUT_SEC:
                    status = "stalled"
                    logger.warning(
                        "ffmpeg không tiến thêm %.0f s (dừng ở %.1f s / %.1f s) — huỷ. stderr cuối: %s",
                        STALL_TIMEOUT_SEC, max(last_done, 0.0), duration, "".join(tail).strip()[-400:],
                    )
                    break
        finally:
            if status:
                proc.kill()
            proc.wait()
            for reader in readers:
                reader.join(timeout=5)
        if status:
            return status
        if proc.returncode != 0:
            logger.debug("ffmpeg lỗi (%s): %s", proc.returncode, "".join(tail).strip()[-400:])
            return "failed"
        return "ok"
