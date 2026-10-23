"""Cung cấp ngữ cảnh video cho khâu dịch bằng Gemini Files API.

Hai năng lực chính:

1. **Tự cắt video** thành các đoạn vừa NGÂN SÁCH TOKEN của mô hình (Gemini tính
   xấp xỉ ``_TOKENS_PER_SECOND`` token cho mỗi giây video) và vừa giới hạn dung
   lượng tải lên. Phim dài hàng giờ sẽ được chia đều thành nhiều đoạn thay vì tải
   nguyên khối (gây lỗi quá token/quá dung lượng như ở các công cụ chỉ tải cả file).

2. **Tải lên + cache theo nội dung**: mỗi đoạn được băm (hash) để nếu đã tải lên
   trước đó và còn hiệu lực trên đám mây thì tái sử dụng, khỏi tải lại.

Phần cắt (ffmpeg) là logic thuần, kiểm thử được offline; phần tải lên cần SDK
google-genai nên được tách riêng và phòng thủ khi SDK/khoá API vắng mặt.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
from subtitles_extractor.infrastructure.process.hidden_process import (
    no_window_kwargs,
    run_ffmpeg_hardened,
)
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from subtitles_extractor.infrastructure.media.ffmpeg_locator import (
    find_ffmpeg,
    missing_ffmpeg_message,
)
from subtitles_extractor.infrastructure.translation.context_audio import (
    AudioStreamInfo,
    context_audio_args,
    select_context_audio,
)
from subtitles_extractor.domain.ports.video_context_port import (
    RemoteVideoRef,
    VideoChunk,
    VideoContextPlan,
)
from subtitles_extractor.infrastructure.translation.chunk_cache_cleanup import (
    ChunkFileInfo,
    plan_chunk_cache_cleanup,
)

if TYPE_CHECKING:
    from datetime import datetime

logger = logging.getLogger(__name__)

# [v3.23.12] Vì ta NÉN video ngữ cảnh về độ phân giải thấp (360p, 1 fps) và gửi với
# media_resolution=LOW, Gemini tính ~100 token/giây video (66 token/frame @1fps +
# 32 token/s audio) thay vì ~263 ở độ phân giải mặc định. Dùng 100 (cận trên an toàn)
# để ước lượng — nhờ đó video dài 3 giờ vẫn vừa context 1M, KHÔNG cần cắt bớt đoạn.
_TOKENS_PER_SECOND = 100

# [v3.23.141] Số token video/giây Gemini tính THEO media_resolution (nguồn: tài liệu
# Gemini "Video understanding"): low ~100 token/s (66/frame @1fps + 32 audio), default/
# medium ~300 token/s (258/frame), high nhiều hơn -> lấy 400 (cận trên an toàn để ước
# lượng quota KHÔNG bị thấp gây 429, và để chia đoạn nhỏ lại cho vừa TPM).
_VIDEO_TOKENS_PER_SEC_BY_RES: dict[str, int] = {"low": 100, "medium": 300, "high": 400}


def _vf_gpu_full(height: int, fps_str: str, width: int = -2) -> str:
    """[v3.23.159] Chuỗi filter tầng GPU HOÀN TOÀN (frame ở CUDA suốt pipeline).

    ``fps`` đặt ĐẦU chuỗi: vứt khung thừa NGAY sau giải mã (fps đích thường 1fps,
    nguồn ~24fps) -> scale_cuda chỉ xử lý ~1/24 số khung. KHÔNG chèn hwupload_cuda ở
    đây: log thực tế cho thấy hwupload_cuda (bản legacy) KHÔNG cho frame CUDA đi
    xuyên qua (danh sách format đầu vào của nó không có 'cuda') -> chèn vào đường
    thuần-GPU làm gãy đàm phán format (-40) ngay cả với file decode-GPU tốt. Trường
    hợp decoder trả frame software được cứu bởi tầng 1.25 (``_vf_gpu_upload``).

    [v3.23.175] ``scale_cuda=...:format=nv12,hwdownload,format=nv12``: với video 10-bit
    HDR/DV (như Mario Galaxy 2160p DV), frame CUDA sau giải mã ở định dạng ``p010le``.
    Nếu ``hwdownload`` cố tải thẳng thành nv12 mà VRAM đang giữ p010 -> "Invalid output
    format nv12 for hwframe download" (-22). Phải để ``scale_cuda`` CHUYỂN p010->nv12
    NGAY TRÊN GPU (tham số ``:format=nv12`` của scale_cuda) rồi ``hwdownload`` mới lấy
    đúng nv12 về RAM. Đây là sửa lỗi hồi quy của v167 (v167 bỏ ``:format`` trong
    scale_cuda nên hwdownload gặp p010 -> gãy trên nội dung 10-bit).

    Args:
        height: Chiều cao đích (px).
        fps_str: Số khung/giây đích dạng chuỗi (vd "1").
        width: Chiều rộng đích (px); ``-2`` = tự tính giữ tỉ lệ và ép chẵn. [v3.23.622]
            Video DỌC phải truyền cả hai (xem :func:`target_frame_size`).

    Returns:
        Chuỗi cho tham số ``-vf`` của ffmpeg.
    """
    return (
        f"fps={fps_str},scale_cuda=w={width}:h={height}:format=nv12,hwdownload,format=nv12"
    )


def _vf_gpu_upload(height: int, fps_str: str, width: int = -2) -> str:
    """[v3.23.159] Chuỗi filter tầng GPU-QUA-UPLOAD (frame software -> GPU scale -> RAM).

    Dùng khi đường thuần-GPU gãy (một số stream h264 khiến giữ frame trên CUDA thất
    bại — log The Hot Spot/Three Against the World): decode NVDEC trả frame về RAM
    (KHÔNG ``-hwaccel_output_format cuda``), ``fps`` vứt khung thừa trước, ``hwupload_cuda``
    đẩy frame software lên GPU để ``scale_cuda`` xử lý, rồi ``hwdownload,format=nv12``
    KÉO FRAME VỀ RAM cho encoder — scale vẫn 100%% GPU, chỉ thêm vòng copy RAM.

    [v3.23.167] Bổ sung ``hwdownload,format=nv12`` sau ``scale_cuda``: log mới cho thấy
    lỗi thật là "Impossible to convert ... src: cuda dst: yuv420p" — frame còn trong
    VRAM khi encoder cần RAM. hwdownload là mắt xích bắc cầu VRAM->RAM còn thiếu.

    Args:
        height: Chiều cao đích (px).
        fps_str: Số khung/giây đích dạng chuỗi.
        width: Chiều rộng đích (px); ``-2`` = tự tính (xem :func:`_vf_gpu_full`).

    Returns:
        Chuỗi cho tham số ``-vf`` của ffmpeg.
    """
    return (
        f"fps={fps_str},hwupload_cuda,scale_cuda=w={width}:h={height}:format=nv12,"
        f"hwdownload,format=nv12"
    )


def _is_nvenc_driver_too_old(ffmpeg_stderr: str) -> bool:
    """Nhận diện lỗi NVENC do DRIVER NVIDIA quá cũ so với ffmpeg đóng gói (hàm thuần).

    Đây là điều kiện VĨNH VIỄN trong phiên (driver không đổi giữa chừng): mọi cấp mã
    hoá GPU (mọi biến thể NVENC) đều sẽ thất bại cùng lý do. Nhận diện sớm để tắt GPU
    ngay, bỏ các cấp GPU còn lại và tránh đổ toàn bộ stderr nhiều lần.

    Args:
        ffmpeg_stderr: Chuỗi stderr của ffmpeg.

    Returns:
        True nếu stderr chứa chữ ký "driver quá cũ cho NVENC".
    """
    low = ffmpeg_stderr.lower()
    return (
        "does not support the required nvenc api version" in low
        or "minimum required nvidia driver for nvenc" in low
    )


def _vf_cpu_scale(height: int, fps_str: str, width: int = -2) -> str:
    """[v3.23.157] Chuỗi filter scale CPU (các tầng fallback) — ``fps`` đặt ĐẦU.

    Trước đây ``fps`` nằm CUỐI chuỗi nên CPU scale toàn bộ ~24fps rồi mới vứt còn
    fps đích -> lãng phí ~24x công scale (The Hot Spot: 25-46s/đoạn). Đặt fps đầu
    chuỗi cho kết quả GIỐNG HỆT nhưng chỉ scale số khung thật sự cần.

    Args:
        height: Chiều cao đích (px).
        fps_str: Số khung/giây đích dạng chuỗi.
        width: Chiều rộng đích (px); ``-2`` = tự tính giữ tỉ lệ và ép chẵn.

    Returns:
        Chuỗi cho tham số ``-vf`` của ffmpeg.
    """
    return f"fps={fps_str},scale={width}:{height},format=yuv420p"


#: [v3.23.622] Cạnh NGẮN tối thiểu khi nén video DỌC (9:16). Thăm dò thật trên tập 720x1280:
#: nén 540x960 đọc đúng từng dòng phụ đề cứng kèm mốc MM:SS, còn 202x360 (360 là chiều CAO)
#: trả dòng bị gộp/thiếu ngay cả ở media_resolution HIGH. Token KHÔNG phụ thuộc số điểm ảnh
#: (HIGH = 51.151 token cho cả 202x360, 540x960 lẫn 720x1280), nên độ phân giải cao hơn chỉ
#: tốn thêm dung lượng tải lên, không tốn quota.
PORTRAIT_MIN_SHORT_SIDE = 540


def _even(value: float) -> int:
    return max(2, int(round(value / 2.0)) * 2)


def target_frame_size(src_width: int, src_height: int, setting: int) -> tuple[int, int]:
    """Kích thước ``(rộng, cao)`` khi nén video ngữ cảnh (hàm thuần, luôn chẵn).

    Cài đặt ``resolution_height`` (mặc định 360) là chiều cao của video NGANG, tức cạnh ngắn.
    Trước đây áp thẳng ``scale=-2:360`` cho mọi video nên 720x1280 thành 202x360 — chữ phụ
    đề cứng bé tí. Nay: video ngang giữ nguyên hành vi (cạnh ngắn = cài đặt); video dọc lấy
    cạnh ngắn = max(cài đặt, :data:`PORTRAIT_MIN_SHORT_SIDE`); không bao giờ phóng to quá nguồn.

    Args:
        src_width: Rộng nguồn (px), ``<= 0`` nếu không đọc được.
        src_height: Cao nguồn (px), ``<= 0`` nếu không đọc được.
        setting: Cài đặt ``resolution_height``.

    Returns:
        ``(-2, setting)`` khi không biết kích thước nguồn (hành vi cũ); ngược lại kích thước
        cụ thể.
    """
    if src_width <= 0 or src_height <= 0:
        return -2, setting
    if src_height > src_width:  # dọc
        short = min(max(setting, PORTRAIT_MIN_SHORT_SIDE), src_width)
        return _even(short), _even(src_height * short / src_width)
    short = min(setting, src_height)
    return _even(src_width * short / src_height), _even(short)


def video_tokens_per_sec(media_resolution: str) -> int:
    """Trả về số token/giây video theo mức media_resolution (mặc định medium=300)."""
    return _VIDEO_TOKENS_PER_SEC_BY_RES.get(
        (media_resolution or "medium").lower(), 300
    )


CLIENT_TIMEOUT_SEC = 180.0
"""[v3.23.535] Thời hạn cho mọi lời gọi HTTP của bộ cung cấp video.

Đủ rộng cho việc TẢI LÊN một đoạn video vài chục MB trên mạng chậm, nhưng hữu hạn — không lời gọi
nào được phép treo vô hạn, vì worker treo là giao diện khoá vĩnh viễn."""


class VideoContextError(Exception):
    """Lỗi khi chuẩn bị/tải ngữ cảnh video."""


GEMINI_FILE_LIFETIME_HOURS = 47.0
"""Files API tự xoá file sau 48 giờ; chừa 1 giờ an toàn để khỏi dùng file sắp bị xoá."""


def _upload_error_types() -> tuple[type[BaseException], ...]:
    """[v3.23.615] Các loại lỗi mạng/máy chủ có thể nổ khi tải file lên Gemini.

    Nạp trễ ``httpx``/``google.genai`` để module vẫn import được khi chưa cài SDK.
    """
    types: list[type[BaseException]] = [OSError]
    try:
        import httpx

        types.append(httpx.HTTPError)
    except ImportError:
        pass
    try:
        from google.genai import errors as genai_errors

        types.append(genai_errors.APIError)
    except ImportError:
        pass
    return tuple(types)


def is_upload_expired(
    created_at_iso: str | None,
    now: datetime | None = None,
    lifetime_hours: float = GEMINI_FILE_LIFETIME_HOURS,
) -> bool:
    """[v3.23.550] True nếu mục cache upload đã quá hạn sống của file Gemini.

    Mục không có (hoặc hỏng) dấu thời gian được coi là CÒN HẠN — giữ hành vi cũ (thử rồi tự
    chữa khi lỗi), không tải lại oan các mục ghi từ bản cũ.

    Args:
        created_at_iso: Thời điểm tải lên, định dạng ISO 8601.
        now: Thời điểm hiện tại (để kiểm thử); mặc định lấy giờ UTC hiện tại.
        lifetime_hours: Hạn sống tính bằng giờ.

    Returns:
        True nếu đã quá hạn.
    """
    from datetime import datetime, timezone

    if not created_at_iso:
        return False
    try:
        created = datetime.fromisoformat(str(created_at_iso))
    except ValueError:
        return False
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    current = now or datetime.now(timezone.utc)
    return (current - created).total_seconds() > lifetime_hours * 3600.0


class GeminiVideoContextProvider:
    """Chuẩn bị và tải các đoạn video lên Gemini để làm ngữ cảnh dịch.

    Args:
        api_key: khoá Gemini (chỉ cần khi tải lên).
        cache_db_path: file SQLite lưu ánh xạ hash→tên file đám mây để tái dùng.
        max_tokens_per_chunk: ngân sách token tối đa cho MỖI đoạn (mặc định để lại
            biên an toàn dưới 1 triệu token của Gemini, dành chỗ cho phụ đề + prompt).
        max_chunk_minutes: trần thời lượng mỗi đoạn (giới hạn cứng để tránh đoạn quá
            dài kể cả khi token cho phép).
        work_dir: thư mục chứa file đoạn tạm (mặc định cạnh video gốc).
    """

    @staticmethod
    def _first_api_key(raw: str) -> str:
        """[v3.23.130] Lấy API key ĐẦU TIÊN từ chuỗi có thể chứa nhiều key.

        Người dùng có thể nhập nhiều key (mỗi dòng/dấu phẩy một key) ở ô 'nhiều key'.
        Tải video chỉ dùng một key nên ta tách & lấy key đầu tiên không rỗng. Tránh
        đưa nguyên chuỗi (có '\\n') vào header HTTP → LocalProtocolError.
        """
        keys = GeminiVideoContextProvider._split_api_keys(raw)
        return keys[0] if keys else (raw or "").strip()

    @staticmethod
    def _split_api_keys(raw: str) -> list[str]:
        """[v3.23.609] Tách chuỗi nhiều key (mỗi dòng/dấu phẩy một key), bỏ trùng."""
        keys: list[str] = []
        for piece in (raw or "").replace(",", "\n").splitlines():
            key = piece.strip()
            if key and key not in keys:
                keys.append(key)
        return keys

    def __init__(
        self,
        api_key: str = "",
        cache_db_path: Path | None = None,
        max_tokens_per_chunk: int = 300_000,
        max_chunk_minutes: float = 18.0,
        work_dir: Path | None = None,
        max_total_tokens: int = 600_000,
        resolution_height: int = 360,
        fps: float = 1.0,
        nvenc_cq: int = 32,
        cpu_crf: int = 30,
        tokens_per_second: int = _TOKENS_PER_SECOND,
        chunk_cache_max_total_mb: int = 4096,
        chunk_cache_max_age_hours: int = 72,
        ignore_cloud_cache: bool = False,
    ) -> None:
        # [v3.23.130] Hỗ trợ chuỗi NHIỀU key (mỗi dòng/phẩy một key) từ giao diện: tải
        # video chỉ cần MỘT key → lấy key ĐẦU TIÊN hợp lệ. Nếu để nguyên chuỗi gộp, nó
        # lọt vào header 'x-goog-api-key' kèm '\n' → LocalProtocolError (crash khi tải).
        self._api_key = self._first_api_key(api_key)
        # [v3.23.609] Mọi key người dùng nhập (và mọi key từng dùng để tải): file Gemini
        # thuộc dự án của key đã tải nó, nên xoá phải thử đúng key đó.
        self._known_api_keys: list[str] = self._split_api_keys(api_key)
        self._cache_db_path = cache_db_path
        self._max_tokens_per_chunk = max_tokens_per_chunk
        self._max_chunk_seconds = max_chunk_minutes * 60.0
        self._work_dir = work_dir
        # [v3.23.39] Tham số nén video ngữ cảnh (chỉnh được trong Cài đặt).
        self._resolution_height = max(144, int(resolution_height))
        self._fps = max(0.2, float(fps))
        self._nvenc_cq = int(nvenc_cq)
        self._cpu_crf = int(cpu_crf)
        self._tokens_per_second = max(1, int(tokens_per_second))
        # [Video Token Guardian] Trần CỨNG tổng token ngữ cảnh gửi đi một lần,
        # tránh lỗi 400 Bad Request khi phim quá dài (vd 80 tập). Vượt trần →
        # chỉ giữ các đoạn đại diện Đầu/Giữa/Cuối (Smart Video Truncation).
        self._max_total_tokens = max(max_tokens_per_chunk, max_total_tokens)
        # [v3.23.479] Buộc cắt+tải lại (bỏ mọi cache cloud/DB) — dùng khi người dùng ĐÃ XOÁ file
        # trên cloud và muốn ép chuẩn bị mới, thay vì tái dùng ref cache.
        self._ignore_cloud_cache = bool(ignore_cloud_cache)
        self._client: Any = None
        # [v3.23.14] Cache refs trong phiên: tránh cắt+upload lặp khi analyze rồi
        # translate cùng một video. Khoá = chữ ký file (đường dẫn+mtime+size).
        self._inmem_refs: dict[str, list[RemoteVideoRef]] = {}
        # [v3.23.21] Cache kết quả dò NVENC (None=chưa dò, True/False=đã biết). Dùng
        # để ưu tiên GPU khi nén video ngữ cảnh; tự tắt nếu GPU lỗi giữa chừng.
        self._nvenc_available: bool | None = None
        # [v3.23.460] NHỚ NVENC treo giữa các phiên: nén GPU treo (driver NVIDIA lỗi) là tình
        # trạng CỦA MÁY, không đổi giữa các lần chạy. Trước đây ``_nvenc_available`` reset mỗi
        # phiên → NGƯỜI DÙNG PHẢI CHỜ 90s cho GPU treo MỖI LẦN phân tích video (app_log: treo
        # đoạn 0 lặp lại). Nay ghi marker khi phát hiện treo; phiên sau đọc marker → BỎ GPU luôn.
        if self._nvenc_marker_path_exists():
            self._nvenc_available = False
        # [v3.23.468/471] Nhớ các hwaccel GIẢI MÃ hỏng giữa các phiên (khỏi chờ treo lại). Theo
        # DÕI RIÊNG từng loại (d3d11va, cuda…) — d3d11va hỏng không có nghĩa cuda hỏng và ngược lại.
        self._hwdec_broken: set[str] = self._load_broken_hwdecs()
        # [v3.23.473] Mặc định KÈM audio; bật True khi file kẹt xen kẽ audio/video (tự học sau khi
        # thử có audio thất bại) → các đoạn sau đọc thẳng video.
        self._drop_audio = False
        # [v3.23.155] Đếm số lần tầng scale_cuda (GPU hoàn toàn) lỗi: lần ĐẦU coi là
        # NHẤT THỜI (NVDEC surface/session bận do trình phát mpv đang hwdec) — vẫn thử
        # lại ở đoạn kế; chỉ khi lỗi lần 2 mới bỏ hẳn tầng này cho các đoạn còn lại.
        self._scale_cuda_failures: int = 0
        # [v3.23.160] Video mà tầng GPU-THUẦN (giữ frame trên CUDA) đã fail: đặc
        # tính stream giống nhau cho MỌI đoạn của cùng video -> các đoạn sau vào
        # thẳng tầng GPU-qua-upload, khỏi tốn ~1.5s thử-fail lặp lại mỗi đoạn.
        self._gpu_full_failed_sources: set[str] = set()
        self._gpu_upload_failed_sources: set[str] = set()
        # [v3.23.128] Khi tầng scale_cuda (GPU hoàn toàn) lỗi một lần thì BỎ QUA nó cho
        # các đoạn sau (tránh fail lặp + log nhiễu) — vẫn dùng tầng GPU giải mã + NVENC.
        self._scale_cuda_disabled: bool = False
        # [v3.23.166] Ngân sách dọn cache file đoạn nén cục bộ (byte + giây).
        self._chunk_cache_max_total_bytes = max(0, int(chunk_cache_max_total_mb)) * 1024 * 1024
        self._chunk_cache_max_age_seconds = max(0, int(chunk_cache_max_age_hours)) * 3600.0

    # ── Lập kế hoạch cắt (thuần, không phụ thuộc mạng) ───────────────────────
    def _read_frame_size(self, video_path: Path) -> tuple[int, int]:
        """[v3.23.622] Rộng x cao khung hình của video (PyAV); ``(0, 0)`` nếu không đọc được."""
        cache = getattr(self, "_frame_size_cache", None)
        if cache is None:
            cache = {}
            self._frame_size_cache = cache
        key = str(video_path)
        if key in cache:
            return cache[key]
        size = (0, 0)
        try:
            import av

            with av.open(str(video_path)) as container:
                if container.streams.video:
                    stream = container.streams.video[0]
                    # Xoay hiển thị 90/270 độ (điện thoại quay dọc) làm đổi chiều rộng/cao.
                    width = int(stream.codec_context.width or 0)
                    height = int(stream.codec_context.height or 0)
                    rotation = 0
                    try:
                        rotation = abs(int(float(stream.metadata.get("rotate", 0)))) % 360
                    except (TypeError, ValueError):
                        rotation = 0
                    size = (height, width) if rotation in (90, 270) else (width, height)
        except Exception as exc:  # noqa: BLE001 — PyAV ném nhiều loại lỗi; không chặn nén
            logger.debug("Không đọc được kích thước khung hình %s: %s", video_path.name, exc)
        cache[key] = size
        return size

    @staticmethod
    def _read_duration_sec(video_path: Path) -> float:
        """Đọc thời lượng video (giây) bằng PyAV; raise nếu không xác định được.

        [v3.23.298] Thay ``ffprobe`` bằng PyAV (libav LGPL đã nhúng) — bỏ phụ thuộc
        binary ffprobe ngoài cho khâu này, dùng đúng mẫu ``pyav_metadata_reader``
        (container.duration / av.time_base, fallback stream.duration * time_base).

        Args:
            video_path: Đường dẫn video.

        Returns:
            Thời lượng (giây) > 0.

        Raises:
            VideoContextError: Khi không mở/đọc được thời lượng.
        """
        import av

        # PyAV đổi tên lớp lỗi giữa các phiên bản (AVError cũ → error.FFmpegError mới).
        # Xây tuple phòng thủ, KHÔNG tham chiếu trực tiếp av.AVError (có thể không tồn tại).
        av_errors: tuple[type[BaseException], ...] = (OSError, ValueError, IndexError)
        _ffmpeg_error = getattr(getattr(av, "error", None), "FFmpegError", None)
        if _ffmpeg_error is not None:
            av_errors = (*av_errors, _ffmpeg_error)

        try:
            with av.open(str(video_path)) as container:
                duration = 0.0
                if container.duration is not None:
                    duration = float(container.duration) / av.time_base
                if duration <= 0 and container.streams.video:
                    stream = container.streams.video[0]
                    if stream.duration is not None and stream.time_base is not None:
                        duration = float(stream.duration * stream.time_base)
        except av_errors as exc:
            raise VideoContextError(
                f"Không đọc được thời lượng video: {exc}"
            ) from exc

        if duration <= 0:
            raise VideoContextError(
                f"Không xác định được thời lượng video: {video_path.name}."
            )
        return duration

    def set_active_key(self, api_key: str) -> None:
        """[v3.23.132] Đổi API key đang dùng để TẢI LÊN (chống 403 khi adapter xoay key).

        File Gemini cô lập theo key; khi adapter chuyển sang key khác, cần tải lại đoạn
        bằng chính key đó. Đổi key ở đây → cache theo key đổi theo (lần tải lại sẽ upload
        bằng key mới) và buộc dựng lại client.
        """
        new_key = self._first_api_key(api_key)
        if new_key and new_key not in self._known_api_keys:
            self._known_api_keys.append(new_key)
        if new_key and new_key != self._api_key:
            self._api_key = new_key
            self._client = None
            self._inmem_refs.clear()  # refs trong phiên gắn theo key cũ → xoá

    def _chunk_seconds_budget(self) -> float:
        """Số giây tối đa cho mỗi đoạn theo cả ngân sách token lẫn trần thời lượng."""
        by_token = self._max_tokens_per_chunk / self._tokens_per_second
        return max(30.0, min(by_token, self._max_chunk_seconds))

    def estimate_chunk_count(self, video_path: Path) -> int:
        """[v3.23.157] Số đoạn DỰ KIẾN cho video (rẻ — chỉ đọc thời lượng).

        Cho tầng worker DỰ TRÙ số request phân tích TRƯỚC khi chọn API key: phân tích
        tuần tự tốn ~1 request/đoạn, nên key phải còn >= số đoạn mới đi trọn phiên
        (log The Hot Spot: chọn key còn 10 cho phim 11 đoạn -> cạn giữa chừng ->
        xoay key + 403 + upload lại).

        Args:
            video_path: Đường dẫn video.

        Returns:
            Số đoạn dự kiến (>= 1); trả 1 nếu không đọc được thời lượng.
        """
        try:
            return max(1, len(self.plan_chunks(Path(video_path)).chunks))
        except VideoContextError:
            return 1

    def plan_chunks(
        self, video_path: Path, allow_truncation: bool = False
    ) -> VideoContextPlan:
        """Tính số đoạn cần cắt; nếu video đủ nhỏ thì giữ nguyên một đoạn.

        Args:
            video_path: Đường dẫn video.
            allow_truncation: Nếu True, khi video CỰC dài sẽ chỉ giữ các đoạn đại diện
                (đầu/giữa/cuối) để giảm chi phí. MẶC ĐỊNH False — vì phân tích ngữ cảnh
                chạy TUẦN TỰ TÍCH LUỸ (mỗi đoạn một request riêng), nên giữ TẤT CẢ đoạn
                để phủ kín toàn phim, không thiếu sót; mỗi đoạn vẫn ≤ ngân sách token
                của một request nên không vượt quota.

        Lưu ý: chưa cắt file ở bước này — chỉ lập kế hoạch ranh giới thời gian. Việc
        cắt thực hiện ở :meth:`materialize_chunks` để tách phần I/O nặng.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise VideoContextError(f"Không tìm thấy video: {video_path}")
        duration = self._read_duration_sec(video_path)
        budget = self._chunk_seconds_budget()
        est_tokens = int(duration * self._tokens_per_second)

        if duration <= budget:
            chunk = VideoChunk(0, video_path, 0.0, duration, is_full_video=True)
            return VideoContextPlan(video_path, duration, est_tokens, [chunk])

        # Chia đều: số đoạn tối thiểu để mỗi đoạn ≤ budget; cân bằng độ dài các đoạn.
        import math
        n_chunks = math.ceil(duration / budget)
        seg = duration / n_chunks
        out_dir = self._work_dir or video_path.parent

        tokens_per_chunk = int(seg * self._tokens_per_second)
        if allow_truncation:
            # [Tuỳ chọn] Smart Video Truncation: chỉ giữ đoạn đại diện khi cần GIẢM chi phí.
            max_keep = max(1, self._max_total_tokens // max(1, tokens_per_chunk))
            kept_indices = self._select_representative_indices(n_chunks, max_keep)
        else:
            # [v3.23.38] MẶC ĐỊNH: giữ TẤT CẢ đoạn — phân tích tuần tự tích luỹ phủ kín
            # toàn phim. Mỗi đoạn ≤ budget (≤ token/request) nên không vượt quota.
            kept_indices = list(range(n_chunks))
        is_truncated = len(kept_indices) < n_chunks

        chunks: list[VideoChunk] = []
        for new_index, original_index in enumerate(kept_indices):
            start = original_index * seg
            end = duration if original_index == n_chunks - 1 else (original_index + 1) * seg
            # [v3.23.157] Nhúng cấu hình nén (h/fps) vào TÊN file -> file nén là hàm
            # thuần của (nguồn, đoạn, cấu hình): tái dùng an toàn giữa các lần chạy;
            # đổi cấu hình thì tên đổi -> không tái dùng nhầm bản cũ.
            part = out_dir / (
                f"{video_path.stem}.ctxpart{original_index:02d}"
                f".h{self._resolution_height}f{self._fps:g}.v2.mp4"
            )
            chunks.append(VideoChunk(new_index, part, start, end, is_full_video=False))

        kept_tokens = int(sum(chunk.duration_sec for chunk in chunks) * self._tokens_per_second)
        if is_truncated:
            logger.info(
                "Video dài (~%d token) vượt trần %d → giữ %d/%d đoạn đại diện "
                "(~%d token).", est_tokens, self._max_total_tokens,
                len(chunks), n_chunks, kept_tokens,
            )
        else:
            logger.info(
                "Video dài ~%d token → cắt thành %d đoạn (mỗi đoạn ~%ds, ~%d token) "
                "để phân tích TUẦN TỰ phủ kín toàn phim.",
                est_tokens, n_chunks, int(seg), tokens_per_chunk,
            )
        return VideoContextPlan(
            video_path, duration, est_tokens, chunks, is_truncated=is_truncated
        )

    @staticmethod
    def _select_representative_indices(n_chunks: int, max_keep: int) -> list[int]:
        """Chọn tối đa ``max_keep`` chỉ số đoạn trải đều, luôn gồm đầu và cuối.

        Args:
            n_chunks: Tổng số đoạn nếu phủ kín.
            max_keep: Số đoạn tối đa được giữ theo ngân sách token.

        Returns:
            Danh sách chỉ số đoạn (tăng dần, không trùng) đại diện Đầu/Giữa/Cuối.
        """
        if max_keep >= n_chunks:
            return list(range(n_chunks))
        if max_keep <= 1:
            return [0]
        # Rải đều trên [0, n_chunks-1] gồm cả hai đầu mút (đầu phim & cuối phim).
        step = (n_chunks - 1) / (max_keep - 1)
        selected = sorted({int(round(i * step)) for i in range(max_keep)})
        return selected

    def materialize_chunks(self, plan: VideoContextPlan) -> None:
        """Cắt + NÉN các đoạn video làm ngữ cảnh cho AI.

        [v3.23.12] Re-encode về ĐỘ PHÂN GIẢI THẤP (360p, 1 fps, mono 16kHz) thay vì
        ``-c copy``. Lý do:
          * Sửa lỗi cắt MKV→MP4 thất bại (``-c copy`` không tương thích timestamp/codec).
          * Khớp cách Gemini xử lý video (lấy mẫu 1 fps): 360p là đủ để model "xem"
            cảnh/đọc chữ to, mà giảm MẠNH dung lượng upload + token.
          * Output ghi ra tên TẠM ASCII rồi đổi về tên đích — tránh lỗi ffmpeg trên
            Windows khi đường dẫn chứa ký tự CJK (vd "中文字幕") làm file không sinh ra.
        """
        if not plan.needs_split:
            return
        ffmpeg = find_ffmpeg()
        if not ffmpeg:
            raise VideoContextError(
                missing_ffmpeg_message(
                    feature="Ngữ cảnh video khi dịch (cắt đoạn gửi Gemini)"
                )
            )
        # [v3.23.472] CHỊU LỖI TỪNG ĐOẠN: một đoạn nén lỗi (vd file HỎNG ở ~6s như app_log —
        # mọi decoder kẹt cùng chỗ) KHÔNG được làm hỏng cả ngữ cảnh. Bỏ đoạn lỗi, DÙNG các đoạn
        # còn lại (đoạn 1-15 thường vẫn tốt vì hỏng chỉ ở đầu phim). Chỉ thất bại HẲN nếu MỌI
        # đoạn đều lỗi.
        failed = self._encode_missing_chunks(ffmpeg, plan.source_path, plan.chunks)

        if failed:
            good = [c for c in plan.chunks if c.is_full_video or c.path.exists()]
            if not good:
                raise VideoContextError(
                    f"TẤT CẢ {len(plan.chunks)} đoạn video đều nén thất bại — file gần như CHẮC "
                    "CHẮN HỎNG (mọi decoder kẹt cùng chỗ). Hãy TẢI LẠI hoặc REMUX video "
                    "(ffmpeg -i input.mp4 -c copy clean.mp4), hoặc bỏ đính video."
                )
            logger.warning(
                "Nén ngữ cảnh: %d/%d đoạn lỗi (bỏ qua), DÙNG %d đoạn còn lại. Đoạn lỗi: %s",
                len(failed), len(plan.chunks), len(good), failed,
            )
            plan.chunks[:] = good  # lọc TẠI CHỖ (list mutable dù dataclass frozen)

    def _encode_missing_chunks(
        self, ffmpeg: str, source_path: Path, chunks: list[VideoChunk]
    ) -> list[int]:
        """Nén các đoạn CHƯA có file cục bộ; trả chỉ số các đoạn nén lỗi (không ném lỗi).

        Tách khỏi :meth:`materialize_chunks` để :meth:`upload_chunks_for_ranges` nén được đúng
        MỘT đoạn — ``materialize_chunks`` bỏ qua hẳn khi kế hoạch chỉ có một đoạn.
        """
        failed: list[int] = []
        for chunk in list(chunks):
            if chunk.is_full_video or chunk.path.exists():
                continue
            try:
                self._encode_low_res_chunk(ffmpeg, source_path, chunk)
            except VideoContextError as exc:
                failed.append(chunk.index)
                logger.warning("Bỏ qua đoạn %d (nén lỗi, dùng các đoạn còn lại): %s",
                               chunk.index, exc)
        return failed

    @staticmethod
    def _output_has_video_frames(ffmpeg: str, path: Path) -> bool:
        """[v3.23.462] True nếu file output có ÍT NHẤT 1 KHUNG VIDEO thật (không chỉ audio).

        Dùng để phát hiện tier CPU-nhanh (-skip_frame nokey) ra file chỉ-có-audio trên video
        keyframe thưa. Đếm khung bằng ffprobe (đi kèm ffmpeg). Nếu KHÔNG kiểm được (thiếu ffprobe/
        lỗi) → trả True (chấp nhận, không chặn nhầm — đường full-decode vẫn là lưới cuối).
        """
        import os

        ffprobe = ffmpeg
        base = os.path.basename(ffmpeg).lower()
        if "ffmpeg" in base:
            ffprobe = os.path.join(
                os.path.dirname(ffmpeg), os.path.basename(ffmpeg).replace("ffmpeg", "ffprobe")
            )
        try:
            result = run_ffmpeg_hardened(
                [
                    ffprobe, "-v", "error", "-select_streams", "v:0",
                    "-count_packets", "-show_entries", "stream=nb_read_packets",
                    "-of", "csv=p=0", str(path),
                ],
                timeout=30.0,
                check=False,
            )
            text = (result.stdout or b"").decode("utf-8", "ignore").strip()
            return text.isdigit() and int(text) > 0
        except (OSError, subprocess.SubprocessError, ValueError):
            return True  # không kiểm được → không chặn

    @staticmethod
    def _ffmpeg_progress_tail(exc: "subprocess.SubprocessError") -> str:
        """[v3.23.470] Trích dòng tiến độ CUỐI của ffmpeg (frame=/time=) từ stderr của exception
        timeout → để phân biệt KẸT (frame=0/không có) vs CHẬM (frame tăng dần)."""
        raw = getattr(exc, "stderr", None) or b""
        text = raw.decode("utf-8", "ignore") if isinstance(raw, bytes) else str(raw)
        # ffmpeg -stats dùng '\r' để cập nhật cùng dòng → tách theo cả \r lẫn \n, lấy dòng
        # cuối có 'frame='.
        parts = text.replace("\r", "\n").split("\n")
        for line in reversed(parts):
            if "frame=" in line:
                return " ".join(line.split())[:120]
        return "không có tiến độ (kẹt ngay từ đầu — nghi đọc/seek file)"

    def _encode_low_res_chunk(self, ffmpeg: str, source_path: Path, chunk: VideoChunk) -> None:
        """Cắt + nén MỘT đoạn về 360p/1fps, ghi qua file tạm ASCII rồi đổi tên.

        Raises:
            VideoContextError: nếu ffmpeg không tạo được file hợp lệ.
        """
        # File tạm ASCII trong thư mục tạm hệ thống (tránh CJK path lỗi trên Windows).
        tmp_out = Path(tempfile.gettempdir()) / f"sectx_{os.getpid()}_{chunk.index:03d}.mp4"
        if tmp_out.exists():
            tmp_out.unlink(missing_ok=True)
        # [v3.23.21] Ưu tiên GPU (NVENC) để cắt+nén NHANH hơn nhiều so với CPU
        # (libx264). Tự dò NVENC một lần; nếu không khả dụng → fallback libx264.
        # 360p (cao 360, rộng tự co, ép chẵn), 1 fps, audio mono 16kHz — đủ cho Gemini.
        encode_cmds = self._build_encode_commands(ffmpeg, source_path, chunk, tmp_out)
        last_exc: Exception | None = None
        # [v3.23.29] Timeout ĐỘNG chống treo: lệnh GPU (NVDEC+NVENC) cực nhanh — thực
        # tế ~18s cho đoạn 1075s. Nếu treo (NVENC hết phiên/driver lỗi) thì cắt SỚM để
        # fallback ngay, KHÔNG chờ tới 30 phút như trước. CPU (libx264) chậm hơn nhiều
        # nên cho hạn rộng theo thời lượng (đoạn 1075s/1fps ~ vài phút).
        # [v3.23.359] Hạ GPU timeout 180→90s: encode 360p/1fps chạy được chỉ mất vài
        # giây; 90s vẫn quá dư nhưng phát hiện TREO (driver NVENC lỗi) nhanh gấp đôi →
        # tắt NVENC + rơi sang CPU sớm hơn, đỡ bắt người dùng chờ vô ích.
        gpu_timeout = 45.0
        # [v3.23.462] Có tier CPU-nhanh (skip_frame, chỉ I-frame) → nén thường xong trong
        # vài–vài chục giây. Hạ timeout 300→120s: đủ dư mà nếu VẪN kẹt thì bỏ SỚM, rơi sang
        # phân tích văn bản, không bắt người dùng chờ 5 phút.
        cpu_timeout = max(120.0, chunk.duration_sec * 0.25)
        succeeded_with_gpu: bool | None = None  # None = chưa lệnh nào thành công
        scale_cuda_failed_this_chunk = False
        succeeded_cmd_used_scale_cuda = False
        for use_gpu, cmd in encode_cmds:
            # Bỏ qua các lệnh GPU nếu phiên này đã xác định GPU không dùng được.
            if use_gpu and self._nvenc_available is False:
                continue
            # [v3.23.467] Tier HYBRID NVDEC-giải-mã (use_gpu=False nhưng có -hwaccel cuda).
            is_nvdec = (not use_gpu) and ("-hwaccel" in cmd)
            # NVDEC giải mã cả chunk (hardware) mất ~vài chục giây → timeout riêng rộng hơn
            # gpu_timeout (vốn để bắt NVENC treo NHANH), vẫn ngắn hơn CPU để bắt treo NVDEC sớm.
            nvdec_timeout = max(90.0, chunk.duration_sec * 0.15)
            timeout_s = (
                nvdec_timeout if is_nvdec
                else (gpu_timeout if use_gpu else cpu_timeout)
            )
            try:
                # [v3.23.461] Runner chống treo: stdin=DEVNULL + kill CỨNG (taskkill /F /T) khi
                # quá hạn → ffmpeg kẹt trong driver GPU CHẮC CHẮN bị kết liễu, app không treo,
                # người dùng KHÔNG phải tự kill ffmpeg.exe.
                run_ffmpeg_hardened(cmd, timeout=timeout_s, check=True)
                # [v3.23.462] Không chỉ kiểm SIZE: tier CPU-nhanh (-skip_frame nokey) trên video
                # keyframe THƯA có thể ra file CHỈ CÓ AUDIO (0 khung video) — size vẫn ≥1KB nhưng
                # VÔ DỤNG cho ngữ cảnh. Phải xác nhận có KHUNG VIDEO thật; nếu không → rơi tier kế
                # (CPU-full sẽ giải mã đủ khung).
                if (
                    tmp_out.exists()
                    and tmp_out.stat().st_size >= 1024
                    and self._output_has_video_frames(ffmpeg, tmp_out)
                ):
                    mode = self._encode_mode_label(cmd)
                    logger.info("Nén đoạn %d bằng %s.", chunk.index, mode)
                    succeeded_with_gpu = use_gpu
                    succeeded_cmd_used_scale_cuda = any(
                        "scale_cuda" in str(a) for a in cmd
                    )
                    break
                tmp_out.unlink(missing_ok=True)
            except subprocess.TimeoutExpired as timeout_exc:
                last_exc = None
                tmp_out.unlink(missing_ok=True)
                # [v3.23.470] Trích tiến độ ffmpeg (frame=/time=) từ stderr để biết KẸT hay CHẬM.
                progress = self._ffmpeg_progress_tail(timeout_exc)
                if is_nvdec:
                    # [v3.23.471] hwdec treo → nhớ RIÊNG hwaccel đó cho phiên sau, thử hwaccel/CPU kế.
                    hw = self._hwaccel_of(cmd)
                    self._persist_broken_hwdec(hw)
                    logger.warning(
                        "Giải mã GPU '%s' đoạn %d treo quá %.0fs [%s] — bỏ '%s' (nhớ cho phiên "
                        "sau), thử cách kế.",
                        hw, chunk.index, timeout_s, progress, hw,
                    )
                elif use_gpu:
                    # [v3.23.358] Tier GPU TREO tới timeout là tín hiệu MẠNH cho thấy đường
                    # mã hoá GPU hỏng trên máy này (driver/hw). Tắt NVENC cho CẢ PHIÊN để
                    # BỎ các cấp GPU còn lại (nhảy thẳng CPU), tránh treo thêm 180s/tier và
                    # tránh làm hỏng cả tác vụ phân tích ngữ cảnh.
                    if self._nvenc_available is not False:
                        self._nvenc_available = False
                        self._persist_nvenc_broken()  # [v3.23.460] nhớ cho phiên sau
                        logger.warning(
                            "Nén GPU đoạn %d (%s) TREO quá %.0fs — TẮT mã hoá GPU cho cả "
                            "phiên, chuyển hẳn sang CPU (libx264). (Kiểm tra driver NVIDIA "
                            "nếu muốn dùng lại NVENC.)",
                            chunk.index, self._encode_mode_label(cmd), gpu_timeout,
                        )
                    else:
                        logger.warning(
                            "Nén GPU đoạn %d (%s) TREO quá %.0fs — huỷ, thử cấp kế.",
                            chunk.index, self._encode_mode_label(cmd), gpu_timeout,
                        )
                else:
                    logger.warning(
                        "Nén CPU đoạn %d treo quá %.0fs [%s] — huỷ.",
                        chunk.index, cpu_timeout, progress,
                    )
            except subprocess.SubprocessError as exc:
                last_exc = exc
                tmp_out.unlink(missing_ok=True)
                if is_nvdec:
                    # [v3.23.471] hwdec lỗi (không hỗ trợ codec / driver) → nhớ RIÊNG hwaccel đó.
                    hw = self._hwaccel_of(cmd)
                    self._persist_broken_hwdec(hw)
                    logger.warning(
                        "Giải mã GPU '%s' đoạn %d lỗi → bỏ '%s' (nhớ cho phiên sau), thử cách kế.",
                        hw, chunk.index, hw,
                    )
                    continue
                if use_gpu:
                    # [v3.23.128] In KÈM stderr của ffmpeg để biết LÝ DO thật (trước đây
                    # chỉ thấy 'exit status -40' vô nghĩa). Giúp chẩn đoán scale_cuda /
                    # thiếu surface / build ffmpeg thiếu filter.
                    err = getattr(exc, "stderr", b"")
                    err_txt = (
                        err.decode("utf-8", "ignore").strip()
                        if isinstance(err, bytes) else str(err or "")
                    )
                    # [v3.23.355] DRIVER NVIDIA QUÁ CŨ cho NVENC của ffmpeg đóng gói là
                    # điều kiện VĨNH VIỄN: mọi cấp GPU đều hỏng cùng lý do. Tắt GPU NGAY
                    # (bỏ các cấp GPU còn lại của đoạn này qua chốt đầu vòng), log GỌN 1
                    # lần kèm cách khắc phục — thay vì đổ nguyên stderr khổng lồ 4 lần.
                    if _is_nvenc_driver_too_old(err_txt):
                        if self._nvenc_available is not False:
                            self._nvenc_available = False
                            self._persist_nvenc_broken()  # [v3.23.460] driver cũ = hỏng vĩnh viễn
                            logger.warning(
                                "NVENC (mã hoá video GPU) KHÔNG dùng được: driver NVIDIA "
                                "thiếu NVENC API 13.1 mà ffmpeg đóng gói yêu cầu (máy "
                                "đang có NVENC API 13.0; cần driver ≥ 610.00). LƯU Ý: đây "
                                "là NVENC API (Video Codec SDK), KHÁC với 'CUDA Driver "
                                "API' — nên OCR GPU (CUDA) vẫn chạy bình thường. Bỏ mọi "
                                "cấp GPU, dùng CPU (libx264) để nén video ngữ cảnh cho "
                                "phiên này. Cập nhật driver NVIDIA để bật lại NVENC."
                            )
                        continue
                    logger.warning(
                        "Nén GPU đoạn %d (%s) thất bại, thử cấp kế. ffmpeg: %s",
                        chunk.index, self._encode_mode_label(cmd),
                        (
                            err_txt if len(err_txt) <= 1200
                            else err_txt[:400] + "\n[... cắt giữa ...]\n" + err_txt[-800:]
                        ) or "(không có stderr)",
                    )
                    # [v3.23.159] Ghi nhận biến thể scale_cuda lỗi bằng CỜ cục bộ;
                    # việc ĐẾM (mỗi đoạn tối đa 1 nhịp, chỉ khi KHÔNG biến thể
                    # scale_cuda nào thành công) thực hiện SAU vòng — tránh một đoạn
                    # tăng 2 nhịp (cấp 1 + cấp 1.25 cùng fail) rồi tắt oan sau 1 đoạn.
                    if any("scale_cuda" in str(a) for a in cmd):
                        scale_cuda_failed_this_chunk = True
                        # [v3.23.160] Lệnh GPU-THUẦN (giữ frame trên CUDA) fail: nhớ
                        # theo VIDEO -> các đoạn sau của cùng video bỏ qua tầng này
                        # (đặc tính stream không đổi giữa các đoạn), video khác vẫn
                        # được thử bình thường.
                        if "-hwaccel_output_format" in cmd:
                            self._gpu_full_failed_sources.add(str(source_path))
                        else:
                            # [v3.23.161] Cấp GPU-qua-upload cũng fail trên video này
                            # -> nhớ để đoạn sau bỏ cả cặp tầng CUDA (đặc tính stream
                            # không đổi giữa các đoạn của cùng video).
                            self._gpu_upload_failed_sources.add(str(source_path))

        # [v3.23.159] Mọi biến thể scale_cuda của ĐOẠN này đều hỏng (đoạn kết thúc
        # bằng lệnh không-scale_cuda hoặc thất bại hoàn toàn) -> đếm 1 nhịp; đủ 2
        # đoạn liên tiếp như vậy mới bỏ hẳn scale_cuda (lần đầu coi là nhất thời).
        if scale_cuda_failed_this_chunk and not succeeded_cmd_used_scale_cuda:
            self._scale_cuda_failures += 1
            if self._scale_cuda_failures >= 2 and not self._scale_cuda_disabled:
                self._scale_cuda_disabled = True
                logger.info(
                    "Tầng scale_cuda lỗi ở %d đoạn — bỏ hẳn cho các đoạn sau.",
                    self._scale_cuda_failures,
                )

        # [v3.23.33] CHỈ tắt GPU hẳn khi lệnh thành công là CPU (mọi cấp GPU đều hỏng).
        # Trước đây tắt GPU ngay khi cấp 1 (GPU full) lỗi, kể cả khi cấp 2 (NVENC-only)
        # vẫn chạy được → lãng phí NVENC cho các đoạn sau. Nay nếu NVENC-only còn chạy
        # được thì GIỮ GPU (chỉ cấp 1 scale_cuda là không dùng được trên máy này).
        if succeeded_with_gpu is False and self._nvenc_available is not False:
            self._nvenc_available = False
            logger.info(
                "GPU không nén được (mọi cấp GPU đều lỗi) — chuyển hẳn sang CPU cho "
                "các đoạn còn lại."
            )

        if not tmp_out.exists() or tmp_out.stat().st_size < 1024:
            detail = ""
            if last_exc is not None:
                stderr = getattr(last_exc, "stderr", b"")
                detail = stderr.decode("utf-8", "ignore")[:200] if isinstance(stderr, bytes) else ""
            # [v3.23.473] TỰ ĐỘNG BỎ AUDIO rồi thử lại: nếu đang KÈM audio mà mọi tier thất bại,
            # nghi kẹt xen kẽ audio/video (app_log: file này). Chuyển video-only (-an) + thử lại
            # NGAY đoạn này; nhớ ``_drop_audio`` cho các đoạn sau (đặc tính file không đổi). Nhờ vậy
            # video THƯỜNG giữ được tiếng, chỉ file lỗi mới rơi video-only.
            if not getattr(self, "_drop_audio", False):
                self._drop_audio = True
                logger.warning(
                    "Nén đoạn %d thất bại khi KÈM audio → thử lại KHÔNG audio (video-only) cho "
                    "cả phim (nghi kẹt xen kẽ audio/video).",
                    chunk.index,
                )
                self._encode_low_res_chunk(ffmpeg, source_path, chunk)
                return
            if last_exc is None:  # last_exc None ở nhánh timeout
                logger.warning(
                    "Nén đoạn %d thất bại do TIMEOUT ở mọi tier (kể cả video-only). Máy này chậm "
                    "bất thường HOẶC file hỏng. Thử: chép video sang SSD/ổ C:, tải lại/remux "
                    "video, hoặc bỏ đính video (phân tích văn bản vẫn tốt).",
                    chunk.index,
                )
            raise VideoContextError(
                f"Nén đoạn {chunk.index} thất bại (cả GPU lẫn CPU). {detail}"
            )

        # [v3.23.474] File kẹt xen kẽ a/v (đã rơi video-only) → THỬ ghép audio bằng cách trích
        # RIÊNG luồng audio (không xen kẽ với video thưa nên thường KHÔNG kẹt). Nếu được → chunk
        # có tiếng; nếu không → giữ video-only.
        if getattr(self, "_drop_audio", False):
            tmp_out = self._try_attach_audio(ffmpeg, source_path, chunk, tmp_out)

        # Đưa file về vị trí đích (tên có thể chứa CJK — dùng Python, KHÔNG qua ffmpeg).
        try:
            chunk.path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(tmp_out), str(chunk.path))
        except OSError as exc:
            tmp_out.unlink(missing_ok=True)
            raise VideoContextError(
                f"Không di chuyển đoạn {chunk.index} về đích: {exc}"
            ) from exc

    def _context_audio(self, source_path: Path) -> AudioStreamInfo | None:
        """[v3.23.610] Luồng âm thanh dùng làm ngữ cảnh (đọc một lần cho mỗi video)."""
        cache = getattr(self, "_audio_probe_cache", None)
        if cache is None:
            cache = {}
            self._audio_probe_cache = cache
        key = str(source_path)
        if key not in cache:
            cache[key] = select_context_audio(self._probe_audio_streams(source_path))
        return cache[key]

    @staticmethod
    def _probe_audio_streams(source_path: Path) -> list[AudioStreamInfo]:
        """Các luồng âm thanh của video (PyAV); lỗi đọc thì trả rỗng."""
        try:
            import av
        except ImportError:
            return []
        av_errors: tuple[type[BaseException], ...] = (OSError, ValueError, IndexError)
        ffmpeg_error = getattr(getattr(av, "error", None), "FFmpegError", None)
        if ffmpeg_error is not None:
            av_errors = (*av_errors, ffmpeg_error)
        default_flag = getattr(getattr(av, "stream", None), "Disposition", None)
        default_flag = getattr(default_flag, "default", 1)
        try:
            with av.open(str(source_path)) as container:
                streams: list[AudioStreamInfo] = []
                for position, stream in enumerate(container.streams.audio):
                    layout = getattr(stream, "layout", None)
                    names = tuple(
                        str(getattr(channel, "name", ""))
                        for channel in getattr(layout, "channels", ()) or ()
                    )
                    channels = int(getattr(stream, "channels", 0) or len(names))
                    disposition = getattr(stream, "disposition", 0) or 0
                    streams.append(
                        AudioStreamInfo(
                            position=position,
                            channels=channels,
                            channel_names=names,
                            is_default=bool(int(disposition) & int(default_flag)),
                        )
                    )
                return streams
        except av_errors as exc:
            logger.debug("Không đọc được luồng âm thanh của %s: %s", source_path.name, exc)
            return []

    def _try_attach_audio(
        self, ffmpeg: str, source_path: Path, chunk: VideoChunk, video_only: Path
    ) -> Path:
        """[v3.23.474] Trích RIÊNG luồng audio của đoạn (không kèm video → tránh kẹt xen kẽ) rồi
        GHÉP vào file video-only. Trả về file có tiếng nếu được; nếu trích/ghép lỗi → trả file
        video-only cũ (giữ nguyên hành vi an toàn)."""
        tmp_audio = video_only.with_name(video_only.stem + ".aud.m4a")
        muxed = video_only.with_name(video_only.stem + ".av.mp4")
        # Trích audio-only: -vn (bỏ video) → demuxer không phải xen kẽ với video 1fps.
        audio_cmd = [
            ffmpeg, "-nostdin", "-y", "-hide_banner", "-loglevel", "error",
            "-err_detect", "ignore_err",
            "-ss", f"{chunk.start_sec:.3f}", "-t", f"{chunk.duration_sec:.3f}",
            "-i", str(source_path),
            "-vn", *context_audio_args(self._context_audio(source_path)),
            "-avoid_negative_ts", "make_zero",
            str(tmp_audio),
        ]
        try:
            run_ffmpeg_hardened(audio_cmd, timeout=90.0, check=True)
        except (subprocess.SubprocessError, OSError) as exc:
            logger.info("Đoạn %d: trích audio riêng KHÔNG được (%s) → giữ video-only.",
                        chunk.index, type(exc).__name__)
            tmp_audio.unlink(missing_ok=True)
            return video_only
        if not tmp_audio.exists() or tmp_audio.stat().st_size < 256:
            tmp_audio.unlink(missing_ok=True)
            return video_only
        # Ghép video (video_only) + audio (tmp_audio), copy stream (nhanh, không encode lại).
        mux_cmd = [
            ffmpeg, "-nostdin", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(video_only), "-i", str(tmp_audio),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c", "copy", "-shortest",
            str(muxed),
        ]
        try:
            run_ffmpeg_hardened(mux_cmd, timeout=60.0, check=True)
        except (subprocess.SubprocessError, OSError) as exc:
            logger.info("Đoạn %d: ghép audio lỗi (%s) → giữ video-only.",
                        chunk.index, type(exc).__name__)
            tmp_audio.unlink(missing_ok=True)
            muxed.unlink(missing_ok=True)
            return video_only
        tmp_audio.unlink(missing_ok=True)
        if muxed.exists() and muxed.stat().st_size >= 1024:
            video_only.unlink(missing_ok=True)
            logger.info("Đoạn %d: đã GHÉP audio riêng vào video-only (có tiếng).", chunk.index)
            return muxed
        muxed.unlink(missing_ok=True)
        return video_only

    def _build_encode_commands(
        self, ffmpeg: str, source_path: Path, chunk: VideoChunk, tmp_out: Path
    ) -> list[tuple[bool, list[str]]]:
        """Tạo danh sách lệnh ffmpeg theo thứ tự ưu tiên: (GPU nếu có) → CPU.

        Mỗi phần tử là (dùng_gpu, lệnh). Bên gọi thử lần lượt tới khi tạo được file.
        """
        # [v3.23.29] Tối ưu theo đề xuất người dùng + tài liệu NVIDIA/ffmpeg:
        # - -threads 0: tối đa luồng CPU khi giải mã HEVC/VP9 nặng (cấp CPU/NVENC-only).
        # - [v3.23.472] BỎ AUDIO (-an): file lỗi làm demuxer kẹt khi đọc xen kẽ audio/video
        #   (app_log: audio tới 6.13s rồi mọi decoder đứng frame=0). Đọc thẳng video tránh kẹt +
        #   nhanh hơn (khỏi decode/resample audio). Lời thoại đã có trong phụ đề nên mất tiếng
        #   ngữ cảnh không đáng kể.
        #   fps mạnh (60→1) trên đoạn dài — đây là nguyên nhân chính gây TREO.
        # - -ss/-t TRƯỚC -i: fast seek (nhanh hơn nhiều với đoạn xa).
        # - -fflags +genpts: sinh lại PTS cho luồng giải mã phần cứng (NVDEC) vốn có thể
        #   thiếu timestamp, tránh ffmpeg treo chờ frame ở đoạn cuối khi kèm -r 1.
        #   (KHÔNG dùng -vsync 0: đã deprecated, thay bằng -fps_mode nhưng option này
        #   không có trên FFmpeg cũ → gỡ hẳn để tương thích ngược, genpts +
        #   max_muxing_queue_size đã đủ chống treo.)
        # - format=nv12/yuv420p ép pixel format rõ ràng (chống lỗi 10-bit).
        # - colorspace/primaries/trc=1 (BT.709) + profile main: chuẩn hoá SDR.
        # [v3.23.461] -nostdin: CHỐNG TREO quan trọng nhất. Không có nó, ffmpeg đọc stdin thừa
        # hưởng từ app GUI (không console) → có thể KẸT CHỜ stdin vô hạn (vd tưởng có lệnh nhập,
        # hoặc handle stdin lạ). App GUI phải luôn chạy ffmpeg với -nostdin + stdin=DEVNULL.
        # [v3.23.470] -stats: ffmpeg in tiến độ (frame=/time=) ra stderr → khi bị kill vì timeout
        # ta biết nó ĐANG kẹt (frame=0) hay chỉ CHẬM (frame tăng) → chẩn đoán đúng bệnh.
        base_flags = [
            ffmpeg, "-nostdin", "-y", "-hide_banner",
            "-loglevel", "error", "-stats", "-threads", "0",
        ]
        # [v3.23.469] -ignore_editlist 1: file MP4/MOV từ YouTube/Google THƯỜNG có EDIT LIST
        # (elst) → -ss/-t đọc lệch/treo. Bỏ qua edit list để đọc thẳng. CHỈ áp cho mov/mp4
        # (option riêng của demuxer mov — thêm cho webm/mkv sẽ LỖI "Option not found").
        mov_input_flags: list[str] = []
        if source_path.suffix.lower() in {".mp4", ".mov", ".m4v", ".m4a", ".mp4v"}:
            mov_input_flags = ["-ignore_editlist", "1"]
        input_args = [
            *mov_input_flags,
            # [v3.23.472] -err_detect ignore_err: bỏ qua lỗi giải mã, cố đọc tiếp thay vì KẸT khi
            # gặp khung hỏng (file tải dở/muxing lỗi — app_log: mọi decoder kẹt ở ~6s). Kèm tolerance
            # từng đoạn ở materialize_chunks (bỏ đoạn hỏng, dùng đoạn tốt).
            "-err_detect", "ignore_err",
            # [v3.23.471] -skip_loop_filter all: bỏ deblocking khi giải mã → nhanh hơn đáng kể
            # trên CPU (thumbnail 360p cho ngữ cảnh không cần deblocking). Giúp giảm tải cho các
            # file H264 giải mã chậm bất thường.
            "-skip_loop_filter", "all",
            "-ss", f"{chunk.start_sec:.3f}",
            "-t", f"{chunk.duration_sec:.3f}",
            "-i", str(source_path),
        ]
        # [v3.23.462] Biến thể input NHANH: chỉ giải mã KHUNG KHÓA (I-frame) → nhanh gấp 10-30
        # lần khi lấy ~1fps cho ngữ cảnh. RỦI RO: video keyframe THƯA (GOP dài) ra quá ít/0 khung
        # → dùng làm tier THỬ TRƯỚC; output rỗng (size <1KB) thì rơi xuống tier full-decode bên
        # dưới. Vừa nhanh vừa AN TOÀN mọi định dạng/mã hoá.
        input_args_fast = [
            *mov_input_flags,
            "-err_detect", "ignore_err",
            "-skip_loop_filter", "all",
            "-ss", f"{chunk.start_sec:.3f}",
            "-t", f"{chunk.duration_sec:.3f}",
            "-skip_frame", "nokey",
            "-i", str(source_path),
        ]
        # [v3.23.473] Mặc định KÈM AUDIO (Gemini nghe được ngữ điệu/nhạc nền). Chỉ bỏ audio khi
        # file bị KẸT xen kẽ audio/video (app_log: audio tới 6.13s, video đứng frame=0) — khi đó
        # ``_drop_audio`` bật (tự động sau khi thử có audio thất bại) → đọc thẳng video.
        if getattr(self, "_drop_audio", False):
            av_map_and_standard_args = [
                "-map", "0:v:0", "-an",
                "-avoid_negative_ts", "make_zero",
                "-colorspace", "1", "-color_primaries", "1", "-color_trc", "1",
            ]
        else:
            # [v3.23.610] Luồng tiếng mặc định, đồng bộ theo mốc thời gian, ưu tiên kênh
            # thoại khi nguồn nhiều kênh — xem context_audio.
            av_map_and_standard_args = [
                "-map", "0:v:0",
                *context_audio_args(self._context_audio(source_path)),
                "-avoid_negative_ts", "make_zero",
                "-colorspace", "1", "-color_primaries", "1", "-color_trc", "1",
                # Chống treo muxing khi hạ fps mạnh (video thưa so audio): flush ngay + queue lớn.
                "-max_muxing_queue_size", "9999",
                "-max_interleave_delta", "0",
            ]

        commands: list[tuple[bool, list[str]]] = []

        # [v3.23.39] Tham số nén từ Cài đặt (thay cho hardcode 360p/1fps/cq32/crf30).
        src_w, src_h = self._read_frame_size(source_path)
        w, h = target_frame_size(src_w, src_h, self._resolution_height)
        r = self._fps
        r_str = f"{r:g}"
        cq_str = str(self._nvenc_cq)
        crf_str = str(self._cpu_crf)

        if self._nvenc_available is not False and self._detect_nvenc(ffmpeg):
            # CẤP 1 — GPU HOÀN TOÀN (NVDEC giải mã + scale_cuda + NVENC).
            # [v3.23.33] -extra_hw_frames 8: cấp thêm surface giải mã cho NVDEC, tránh
            # lỗi "No decoder surfaces left" (crash exit -40) khi decode đoạn dài liên
            # tục — giải pháp chuẩn của Jellyfin. Cú pháp scale_cuda=w=-2:h=H (tường
            # minh) ổn định hơn scale_cuda=-2:H:format=nv12; format đặt riêng sau scale.
            # [v3.23.161] -init_hw_device + -filter_hw_device: cấp CUDA device TƯỜNG
            # MINH cho filter (đúng tài liệu ffmpeg HWAccelIntro). Log thực tế cho thấy
            # filter CUDA chết ENOSYS (-40) vì KHÔNG có device khi hwaccel decoder init
            # thất bại với một số stream — device tường minh độc lập với hwaccel.
            gpu_full_cmd = (
                base_flags
                + ["-init_hw_device", "cuda=cu", "-filter_hw_device", "cu",
                   "-fflags", "+genpts",
                   "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
                   "-extra_hw_frames", "8"]
                + input_args
                + av_map_and_standard_args
                + [
                    "-vf", _vf_gpu_full(h, r_str, w),
                    "-c:v", "h264_nvenc", "-preset", "p4", "-cq", cq_str,
                    "-profile:v", "main",
                    "-pix_fmt", "nv12",  # [v3.23.167] nhận nv12 từ RAM sau hwdownload
                    str(tmp_out),
                ]
            )
            commands.append((True, gpu_full_cmd))
            if (
                self._scale_cuda_disabled
                or str(source_path) in self._gpu_full_failed_sources
            ):
                commands.pop()  # scale_cuda lỗi trên máy / GPU-thuần lỗi với video này

            # CẤP 1.25 — GPU QUA UPLOAD (NVDEC -> RAM -> hwupload_cuda -> scale_cuda
            # -> NVENC). [v3.23.159] Cứu các file mà đường thuần-GPU (cấp 1) gãy vì
            # không giữ được frame trên CUDA: scale + mã hoá VẪN 100% GPU, chỉ thêm
            # một vòng copy RAM cho ~1 khung/giây (fps đứng đầu chuỗi) — nhanh hơn
            # hẳn scale CPU (cấp 1.5) trên các đoạn dài.
            gpu_upload_cmd = (
                base_flags
                + ["-init_hw_device", "cuda=cu", "-filter_hw_device", "cu",
                   "-fflags", "+genpts", "-hwaccel", "cuda"]
                + input_args
                + av_map_and_standard_args
                + [
                    "-vf", _vf_gpu_upload(h, r_str, w),
                    "-c:v", "h264_nvenc", "-preset", "p4", "-cq", cq_str,
                    "-profile:v", "main",
                    "-pix_fmt", "nv12",  # [v3.23.167] nhận nv12 từ RAM sau hwdownload
                    str(tmp_out),
                ]
            )
            commands.append((True, gpu_upload_cmd))
            if (
                self._scale_cuda_disabled
                or str(source_path) in self._gpu_upload_failed_sources
            ):
                commands.pop()

            # CẤP 1.5 — GPU GIẢI MÃ (NVDEC) + scale CPU + NVENC mã hóa.
            # [v3.23.128] Tầng trung gian QUAN TRỌNG: tránh hẳn scale_cuda (thủ phạm phổ
            # biến gây exit -40 do hết decoder surface hoặc build ffmpeg thiếu filter
            # CUDA), nhưng VẪN giải mã bằng NVDEC nên nhanh hơn NHIỀU so với giải mã CPU
            # thuần (cấp 2). Không dùng -hwaccel_output_format cuda → frame tải về RAM
            # ngay, không giữ surface lâu nên không bị -40.
            gpu_decode_cmd = (
                base_flags
                + ["-fflags", "+genpts", "-hwaccel", "cuda"]
                + input_args
                + av_map_and_standard_args
                + [
                    "-vf", _vf_cpu_scale(h, r_str, w),
                    "-c:v", "h264_nvenc", "-preset", "p4", "-cq", cq_str,
                    "-profile:v", "main",
                    str(tmp_out),
                ]
            )
            commands.append((True, gpu_decode_cmd))

            # CẤP 2 — NVENC-only (giải mã+scale CPU → mã hóa GPU).
            gpu_encode_cmd = (
                base_flags
                + ["-fflags", "+genpts"]
                + input_args
                + av_map_and_standard_args
                + [
                    "-vf", _vf_cpu_scale(h, r_str, w),
                    "-c:v", "h264_nvenc", "-preset", "p4", "-cq", cq_str,
                    "-profile:v", "main",
                    str(tmp_out),
                ]
            )
            commands.append((True, gpu_encode_cmd))

        # [v3.23.467] CẤP 2.4 — NVDEC GIẢI MÃ + CPU libx264 encode (HYBRID né NVENC hỏng).
        # Video YouTube (VP9/AV1) giải mã CPU RẤT chậm (VP9 ~7×, AV1 ~10× so H264) → timeout.
        # NVDEC (giải mã phần cứng) nhanh gấp bội cho VP9/AV1, và ĐỘC LẬP với NVENC (encode) —
        # máy có thể NVENC hỏng nhưng NVDEC vẫn tốt. Tier này KHÔNG bị nvenc-marker chặn (dùng
        # libx264 để encode, không đụng NVENC). Nếu NVDEC cũng treo/lỗi → đánh dấu bỏ trong phiên.
        # [v3.23.467/471] CẤP 2.4 — GIẢI MÃ PHẦN CỨNG (hwaccel) + CPU libx264 encode (HYBRID né
        # NVENC hỏng). File H264/VP9/AV1 giải mã CPU có thể CỰC chậm (app_log: 0.04× realtime với
        # một file YouTube → 4 giờ/đoạn!). Giải mã phần cứng nhanh gấp bội. Thử NHIỀU hwaccel:
        #   - d3d11va (Windows): CHÍNH LÀ hwdec mpv dùng (auto-safe) — tương thích rộng nhất, chạy
        #     được cả khi cuda/NVDEC hỏng.
        #   - cuda (NVDEC): NVIDIA.
        # Mỗi hwaccel theo dõi hỏng RIÊNG. KHÔNG bị nvenc-marker chặn (encode libx264, không NVENC).
        hwdec_candidates: list[str] = []
        if os.name == "nt":
            hwdec_candidates.append("d3d11va")
        hwdec_candidates.append("cuda")
        for hw in hwdec_candidates:
            if hw in self._hwdec_broken:
                continue
            hwdec_cmd = (
                base_flags
                + ["-fflags", "+genpts", "-hwaccel", hw]
                + input_args
                + av_map_and_standard_args
                + [
                    "-vf", _vf_cpu_scale(h, r_str, w),
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", crf_str,
                    "-profile:v", "main", "-pix_fmt", "yuv420p",
                    str(tmp_out),
                ]
            )
            commands.append((False, hwdec_cmd))

        # [v3.23.462] CẤP 2.5 — CPU NHANH (chỉ giải mã I-frame): thử TRƯỚC CPU thường. Output
        # rỗng (video keyframe thưa) → tự rơi xuống CPU-đầy-đủ (vòng thử kiểm size ≥1KB). Đa số
        # video (keyframe ~2-4s) → nén 1 đoạn vài–vài chục giây thay vì >5 phút.
        cpu_fast_cmd = (
            base_flags
            + input_args_fast
            + av_map_and_standard_args
            + [
                "-vf", _vf_cpu_scale(h, r_str, w),
                "-c:v", "libx264", "-preset", "veryfast", "-crf", crf_str,
                # [v3.23.466] ÉP 4:2:0 8-bit: video YouTube (VP9/AV1) hay 4:4:4/4:2:2/10-bit mà
                # libx264 profile 'main' KHÔNG hỗ trợ → "main profile doesn't support 4:4:4",
                # encode THẤT BẠI, không cắt được. -pix_fmt yuv420p quy đổi mọi chroma/độ sâu về
                # 4:2:0 8-bit (main chấp nhận) — decode+cắt được MỌI codec/định dạng.
                "-profile:v", "main", "-pix_fmt", "yuv420p",
                str(tmp_out),
            ]
        )
        commands.append((False, cpu_fast_cmd))

        # CẤP 3 — CPU THUẦN (LUÔN có, kể cả khi không NVENC — đây là fallback cuối).
        cpu_cmd = (
            base_flags
            + input_args
            + av_map_and_standard_args
            + [
                "-vf", _vf_cpu_scale(h, r_str, w),
                "-c:v", "libx264", "-preset", "veryfast", "-crf", crf_str,
                # [v3.23.466] ÉP 4:2:0 8-bit: video YouTube (VP9/AV1) hay 4:4:4/4:2:2/10-bit mà
                # libx264 profile 'main' KHÔNG hỗ trợ → "main profile doesn't support 4:4:4",
                # encode THẤT BẠI, không cắt được. -pix_fmt yuv420p quy đổi mọi chroma/độ sâu về
                # 4:2:0 8-bit (main chấp nhận) — decode+cắt được MỌI codec/định dạng.
                "-profile:v", "main", "-pix_fmt", "yuv420p",
                str(tmp_out),
            ]
        )
        commands.append((False, cpu_cmd))
        return commands

    @staticmethod
    def _encode_mode_label(cmd: list[str]) -> str:
        """Nhãn mô tả chế độ nén dựa trên lệnh ffmpeg (để ghi log rõ ràng)."""
        has_hwaccel = "cuda" in cmd and "-hwaccel" in cmd
        has_nvenc = "h264_nvenc" in cmd
        has_scale_cuda = any("scale_cuda" in str(a) for a in cmd)
        has_hwupload = any("hwupload_cuda" in str(a) for a in cmd)
        if has_hwaccel and has_nvenc and has_scale_cuda and has_hwupload:
            # [v3.23.159] Cấp 1.25: frame về RAM rồi upload lên GPU scale + NVENC.
            return "GPU qua upload (NVDEC -> RAM -> hwupload -> scale_cuda + NVENC)"
        if has_hwaccel and has_nvenc and has_scale_cuda:
            return "GPU hoàn toàn (NVDEC giải mã + scale_cuda + NVENC)"
        if has_hwaccel and has_nvenc:
            return "GPU (NVDEC giải mã + scale CPU + NVENC)"
        if has_nvenc:
            return "GPU một phần (chỉ mã hóa NVENC, giải mã+scale trên CPU)"
        return "CPU (libx264)"

    def _gpu_marker_path(self, kind: str) -> "Path | None":
        """[v3.23.468] Đường dẫn marker 'GPU <kind> hỏng trên máy này' (cạnh cache DB)."""
        if not self._cache_db_path:
            return None
        return Path(self._cache_db_path).parent / f"{kind}_runtime_broken.marker"

    def _gpu_marker_valid(self, kind: str) -> bool:
        """True nếu marker còn hiệu lực (< 7 ngày); cũ hơn → xoá + thử lại (phòng cập nhật driver)."""
        marker = self._gpu_marker_path(kind)
        if not marker or not marker.exists():
            return False
        try:
            import time

            if (time.time() - marker.stat().st_mtime) / 86400.0 > 7.0:
                marker.unlink(missing_ok=True)
                return False
        except OSError:
            return False
        return True

    def _persist_gpu_broken(self, kind: str, note: str) -> None:
        """Ghi marker để phiên sau bỏ GPU <kind> luôn (không chờ treo lại). Lỗi ghi → bỏ qua."""
        marker = self._gpu_marker_path(kind)
        if marker is None:
            return
        try:
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(note, encoding="utf-8")
        except OSError:
            pass

    def _nvenc_marker_path(self) -> "Path | None":
        return self._gpu_marker_path("nvenc")

    def _nvenc_marker_path_exists(self) -> bool:
        return self._gpu_marker_valid("nvenc")

    def _persist_nvenc_broken(self) -> None:
        self._persist_gpu_broken(
            "nvenc",
            "NVENC treo khi nén video ngữ cảnh — bỏ GPU. Xoá file này để thử lại NVENC "
            "sau khi cập nhật driver NVIDIA.",
        )

    def _load_broken_hwdecs(self) -> set[str]:
        """[v3.23.471] Đọc tập hwaccel giải mã đã hỏng (marker, hết hạn 7 ngày qua _gpu_marker_valid)."""
        marker = self._gpu_marker_path("hwdec")
        if marker is None or not self._gpu_marker_valid("hwdec"):
            return set()
        try:
            return {x.strip() for x in marker.read_text(encoding="utf-8").split(",") if x.strip()}
        except OSError:
            return set()

    def _persist_broken_hwdec(self, hwaccel: str) -> None:
        """[v3.23.471] Ghi nhớ một hwaccel giải mã hỏng → phiên sau bỏ luôn (khỏi chờ treo lại)."""
        self._hwdec_broken.add(hwaccel)
        marker = self._gpu_marker_path("hwdec")
        if marker is None:
            return
        try:
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(",".join(sorted(self._hwdec_broken)), encoding="utf-8")
        except OSError:
            pass

    @staticmethod
    def _hwaccel_of(cmd: list[str]) -> str:
        """Lấy tên hwaccel trong lệnh (sau '-hwaccel'), rỗng nếu không có."""
        if "-hwaccel" in cmd:
            i = cmd.index("-hwaccel")
            if i + 1 < len(cmd):
                return cmd[i + 1]
        return ""

    def _persist_nvdec_broken(self) -> None:
        """[v3.23.468] Giữ tương thích: nhớ 'cuda' hỏng."""
        self._persist_broken_hwdec("cuda")

    def _detect_nvenc(self, ffmpeg: str) -> bool:
        """Dò xem ffmpeg có hỗ trợ encoder h264_nvenc không (cache kết quả)."""
        if self._nvenc_available is not None:
            return self._nvenc_available
        try:
            out = subprocess.run(
                [ffmpeg, "-hide_banner", "-encoders"],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30,
                **no_window_kwargs(),
            )
            self._nvenc_available = "h264_nvenc" in (out.stdout or "")
        except (subprocess.SubprocessError, OSError):
            self._nvenc_available = False
        if self._nvenc_available:
            logger.info("Phát hiện NVENC — sẽ dùng GPU để nén video ngữ cảnh.")
        return self._nvenc_available

    # ── Tải lên Gemini + cache theo hash ─────────────────────────────────────
    @staticmethod
    def _hash_file(path: Path, sample_bytes: int = 8 * 1024 * 1024) -> str:
        """Băm nhanh theo kích thước + mẫu đầu/cuối file (đủ để định danh đoạn)."""
        if not path.exists():
            raise VideoContextError(
                f"Đoạn video không tồn tại để băm: {path.name} "
                "(ffmpeg có thể đã cắt thất bại)."
            )
        h = hashlib.sha256()
        size = path.stat().st_size
        h.update(str(size).encode())
        with open(path, "rb") as f:
            h.update(f.read(sample_bytes))
            if size > sample_bytes * 2:
                f.seek(-sample_bytes, 2)
                h.update(f.read(sample_bytes))
        return h.hexdigest()

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self._api_key:
            raise VideoContextError("Chưa cấu hình khoá API Gemini để tải video.")
        self._client = self._make_client(self._api_key)
        return self._client

    @staticmethod
    def _make_client(api_key: str) -> Any:
        """Tạo client Gemini cho ``api_key`` (có thời hạn cho mọi lời gọi HTTP).

        Raises:
            VideoContextError: Chưa cài google-genai.
        """
        try:
            from google import genai  # type: ignore
        except ImportError as exc:
            raise VideoContextError(
                "Chưa cài google-genai. Cài: pip install google-genai"
            ) from exc
        # [v3.23.535] ĐẶT THỜI HẠN cho mọi lời gọi HTTP. Trước đây client này KHÔNG có timeout
        # (khác với client của adapter dịch, vốn đã có), nên một lời gọi Files API treo là treo
        # VĨNH VIỄN. Hậu quả dây chuyền: ``run()`` của worker không bao giờ trả về → tín hiệu
        # ``finished`` không phát → giao diện không được mở khoá → người dùng không bấm được gì,
        # phải End Task. Nhật ký thật dừng đột ngột giữa chuỗi kiểm file trên cloud, đúng với
        # kịch bản này.
        timeout_ms = int(CLIENT_TIMEOUT_SEC * 1000)
        try:
            return genai.Client(api_key=api_key, http_options={"timeout": timeout_ms})
        except TypeError:
            # SDK cũ không nhận http_options — vẫn dùng được, chỉ là không có thời hạn.
            logger.warning(
                "SDK google-genai không hỗ trợ http_options.timeout — lời gọi tải video sẽ "
                "không có thời hạn. Nên cập nhật google-genai để tránh treo ứng dụng."
            )
            return genai.Client(api_key=api_key)

    def _cache_lookup(self, file_hash: str) -> str | None:
        """Tra tên file cloud đã tải cho một đoạn; bỏ qua mục đã quá hạn sống của Gemini.

        [v3.23.550] Files API tự XOÁ file sau 48 giờ, và với file không còn tồn tại Gemini trả
        **403** (không phải 404). Trước đây cột ``created_at`` có lưu nhưng không bao giờ được
        đọc, nên mọi mục cũ đều bị đem ra thử: app_log thật có 22 lượt "403 PERMISSION_DENIED"
        liên tiếp trước khi tải lại từng đoạn. Mục quá hạn nay bị coi là không có và xoá luôn.
        """
        if not self._cache_db_path:
            return None
        import sqlite3
        try:
            with sqlite3.connect(str(self._cache_db_path)) as conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS video_context_uploads ("
                    "file_hash TEXT PRIMARY KEY, remote_name TEXT, created_at TEXT)"
                )
                row = conn.execute(
                    "SELECT remote_name, created_at FROM video_context_uploads "
                    "WHERE file_hash=?",
                    (file_hash,),
                ).fetchone()
                if not row:
                    return None
                if is_upload_expired(row[1]):
                    conn.execute(
                        "DELETE FROM video_context_uploads WHERE file_hash=?", (file_hash,)
                    )
                    conn.commit()
                    logger.debug("Bỏ mục cache upload đã quá hạn sống: %s", row[0])
                    return None
                return str(row[0])
        except sqlite3.Error as exc:
            logger.warning("Lỗi tra cache upload video: %s", exc)
            return None

    def _cache_store(self, file_hash: str, remote_name: str) -> None:
        """Ghi ánh xạ khoá → file cloud; GIỮ mốc tải lên cũ khi vẫn là file đó.

        [v3.23.615] Mục theo khoá NGUỒN được ghi lại sau MỖI lần chuẩn bị video, kể cả khi
        đoạn chỉ được tái dùng. Ghi đè ``created_at`` bằng giờ hiện tại làm mục sống quá
        48 giờ thật của file: lần chuẩn bị sau đem tên file đã bị Gemini xoá ra dò (403)
        rồi mới làm lại. Mốc chỉ đổi khi tên file đổi (đã tải lên bản mới).
        """
        if not self._cache_db_path:
            return
        import sqlite3
        from datetime import datetime, timezone
        try:
            with sqlite3.connect(str(self._cache_db_path)) as conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS video_context_uploads ("
                    "file_hash TEXT PRIMARY KEY, remote_name TEXT, created_at TEXT)"
                )
                conn.execute(
                    "INSERT INTO video_context_uploads VALUES (?,?,?) "
                    "ON CONFLICT(file_hash) DO UPDATE SET "
                    "created_at = CASE WHEN video_context_uploads.remote_name = "
                    "excluded.remote_name THEN COALESCE("
                    "video_context_uploads.created_at, excluded.created_at) "
                    "ELSE excluded.created_at END, "
                    "remote_name = excluded.remote_name",
                    (file_hash, remote_name, datetime.now(timezone.utc).isoformat()),
                )
                conn.commit()
        except sqlite3.Error as exc:
            logger.warning("Lỗi lưu cache upload video: %s", exc)

    def _cache_key(self, file_hash: str) -> str:
        """[API Key Isolation] Trộn khoá API vào hash nội dung để cô lập cache.

        Đổi API Key ⇒ cache key đổi ⇒ cache miss ⇒ tự động tải lên và lưu đệm mới
        độc lập. Khoá API được băm cùng (không lưu trực tiếp) nên không lộ ra DB.

        Args:
            file_hash: Hash nội dung file đoạn.

        Returns:
            Khoá cache hex đã trộn khoá API.
        """
        mixer = hashlib.sha256()
        mixer.update((self._api_key or "no-key").encode("utf-8"))
        mixer.update(b"::")
        mixer.update(file_hash.encode("ascii"))
        return mixer.hexdigest()

    def _chunk_source_key(self, video_path: Path, chunk: VideoChunk) -> str:
        """[v3.23.131] Khoá cache theo NGUỒN (không cần file đã nén).

        Khác ``_cache_key`` (băm nội dung file ĐÃ NÉN → buộc phải nén trước mới biết),
        khoá này tính từ chữ ký video gốc + vị trí đoạn + tham số nén → xác định được
        NGAY khi mới lập kế hoạch. Nhờ vậy ta tra được cloud TRƯỚC khi cắt+nén, tránh
        nén lại tốn ~18s/đoạn rồi mới phát hiện đã có sẵn trên cloud.
        """
        sig = self._video_signature(video_path)
        mixer = hashlib.sha256()
        mixer.update((self._api_key or "no-key").encode("utf-8"))
        mixer.update(b"::src::")
        params = (
            f"{sig}|{chunk.index}|{chunk.start_sec:.3f}|{chunk.end_sec:.3f}"
            f"|h{self._resolution_height}|f{self._fps:.3f}"
            f"|cq{self._nvenc_cq}|crf{self._cpu_crf}|v2portrait"
        )
        mixer.update(params.encode("utf-8"))
        return mixer.hexdigest()

    def _resolve_chunks_from_source_cache(
        self, plan: VideoContextPlan, video_path: Path
    ) -> list[RemoteVideoRef] | None:
        """Tra cloud theo khoá NGUỒN cho MỌI đoạn — trả refs nếu tất cả còn ACTIVE.

        Trả None nếu thiếu bất kỳ đoạn nào (hoặc cloud đã xoá) → cần cắt+nén+upload.
        """
        if plan.needs_split is False or not plan.chunks:
            return None
        try:
            client = self._ensure_client()
        except VideoContextError:
            return None
        refs: list[RemoteVideoRef] = []
        for chunk in plan.chunks:
            cached_name = self._cache_lookup(self._chunk_source_key(video_path, chunk))
            if not cached_name:
                return None
            try:
                remote = client.files.get(name=cached_name)
            except Exception:  # noqa: BLE001 — bất kỳ lỗi cloud → coi như cần làm lại
                return None
            if getattr(remote.state, "name", "") != "ACTIVE":
                return None
            refs.append(
                RemoteVideoRef(
                    chunk.index, cached_name, chunk.start_sec, chunk.end_sec, "ACTIVE"
                )
            )
        return refs

    @staticmethod
    def _make_ascii_safe_source(real_path: Path, cache_key: str, chunk_index: int) -> tuple[Path, bool]:
        """[Unicode Bypass] Tạo bản tham chiếu tên thuần ASCII cho file đoạn.

        Tên file gốc chứa ký tự CJK/Việt khiến HTTPX dựng header thất bại
        (UnicodeEncodeError ASCII) và làm sập app khi tải lên. Ta tạo một tên ảo an
        toàn dạng ``video_<key8>_partNN.<ext>`` rồi **hardlink** (cùng inode, không
        tốn thêm dung lượng) tới file thật. Nếu hardlink bất khả thi (khác phân
        vùng/hệ thống không hỗ trợ), fallback sang copy.

        Args:
            real_path: Đường dẫn file đoạn thật (có thể chứa ký tự non-ASCII).
            cache_key: Khoá cache (đã trộn API key) để đặt tên ổn định.
            chunk_index: Thứ tự đoạn.

        Returns:
            Cặp ``(safe_path, is_temporary)``; ``is_temporary=True`` nghĩa là cần
            xoá ``safe_path`` sau khi tải lên.
        """
        suffix = real_path.suffix if real_path.suffix.isascii() else ".mp4"
        safe_name = f"video_{cache_key[:8]}_part{chunk_index:02d}{suffix}"
        safe_path = Path(tempfile.gettempdir()) / safe_name
        if safe_path.exists():
            try:
                safe_path.unlink()
            except OSError:
                return real_path, False
        try:
            os.link(real_path, safe_path)  # hardlink: 0 byte phụ
            return safe_path, True
        except OSError:
            try:
                shutil.copy2(real_path, safe_path)
                return safe_path, True
            except OSError as copy_error:
                logger.warning("Không tạo được tên ASCII an toàn: %s", copy_error)
                return real_path, False

    @staticmethod
    def _is_recoverable_remote_error(error: BaseException) -> bool:
        """[Cloud Cache Healing] Nhận diện lỗi cloud có thể tự chữa bằng tải lại.

        Bao gồm 404 NOT_FOUND (Google xoá video sau ~48h), 403 PERMISSION_DENIED
        (đổi khoá API), và các lỗi tham chiếu file hết hạn.
        """
        message = str(error).lower()
        recoverable_markers = (
            "404", "not_found", "not found", "403", "permission_denied",
            "permission denied", "was not found", "expired",
        )
        return any(marker in message for marker in recoverable_markers)

    def upload_chunk(self, chunk: VideoChunk, poll_interval_s: float = 8.0,
                     timeout_s: float = 600.0) -> RemoteVideoRef:
        """Tải một đoạn lên Gemini (tái dùng nếu cache còn hiệu lực), chờ ACTIVE."""
        client = self._ensure_client()
        file_hash = self._hash_file(chunk.path)
        cache_key = self._cache_key(file_hash)  # [1.2] cô lập theo API key

        cached_name = self._cache_lookup(cache_key)
        if cached_name:
            try:
                remote = client.files.get(name=cached_name)
                if getattr(remote.state, "name", "") == "ACTIVE":
                    logger.info("Tái dùng đoạn video đã tải lên: %s", cached_name)
                    return RemoteVideoRef(
                        chunk.index, cached_name, chunk.start_sec, chunk.end_sec, "ACTIVE"
                    )
            except Exception as exc:  # noqa: BLE001 — healing cần bắt mọi lỗi cloud
                # [1.3] Google đã xoá video (404) hoặc đổi quyền (403) → tải lại êm.
                if self._is_recoverable_remote_error(exc):
                    logger.info("Đoạn cache không còn trên cloud (%s), tải lại.", exc)
                else:
                    logger.warning("Lỗi không lường khi tra cloud (%s), thử tải lại.", exc)

        # [v3.23.596] Đánh số từ 1 như nhật ký phân tích ("đoạn 2/17"): bản cũ ghi
        # "đoạn 1" cho đúng đoạn mà khâu phân tích gọi là "đoạn 2".
        logger.info("Đang tải đoạn %d (%.0f–%.0fs) lên Gemini…",
                    chunk.index + 1, chunk.start_sec, chunk.end_sec)
        # [1.1] Upload qua tên ASCII an toàn để tránh sập do header Unicode.
        safe_path, is_temporary = self._make_ascii_safe_source(
            chunk.path, cache_key, chunk.index
        )
        try:
            remote = client.files.upload(file=str(safe_path))
            deadline = time.monotonic() + timeout_s
            while getattr(remote.state, "name", "") == "PROCESSING":
                if time.monotonic() > deadline:
                    raise VideoContextError(f"Quá hạn xử lý đoạn {chunk.index} trên cloud.")
                time.sleep(poll_interval_s)
                remote = client.files.get(name=remote.name)
        except _upload_error_types() as exc:
            # [v3.23.615] Lỗi mạng/máy chủ khi tải phải là VideoContextError — loại lỗi mà
            # khâu phân tích biết lùi về "phân tích từ phụ đề" và khâu dịch báo rõ ràng.
            # Trước đây httpx.ConnectError/ProxyError lọt thẳng ra ngoài: khâu dịch ghi
            # "Lỗi không lường trước khi dịch phụ đề" kèm traceback (bấm thử toàn ứng dụng
            # với video thật, mạng bị chặn), khâu phân tích hỏng hẳn thay vì bỏ video.
            raise VideoContextError(
                f"Không tải được đoạn {chunk.index + 1} lên Gemini: {exc}"
            ) from exc
        finally:
            if is_temporary and safe_path != chunk.path:
                try:
                    safe_path.unlink()
                except OSError:
                    pass
        state = getattr(remote.state, "name", "")
        if state != "ACTIVE":
            raise VideoContextError(f"Tải đoạn {chunk.index} thất bại, trạng thái {state}.")
        self._cache_store(cache_key, remote.name)
        return RemoteVideoRef(chunk.index, remote.name, chunk.start_sec, chunk.end_sec, "ACTIVE")

    def reupload_window(
        self, source_path: Path, start_sec: float, end_sec: float, index: int = 0
    ) -> RemoteVideoRef:
        """[Cloud Cache Healing] Cắt lại một khoảng video từ nguồn và tải lên mới.

        Dùng khi tham chiếu cloud cũ bị Google xoá (404) ngay lúc dịch: tầng adapter
        bắt lỗi và gọi hàm này để tái tạo đoạn từ video gốc, lấy tham chiếu mới.

        Args:
            source_path: Video gốc trên đĩa.
            start_sec: Mốc bắt đầu đoạn cần dựng lại.
            end_sec: Mốc kết thúc đoạn.
            index: Thứ tự đoạn (cho đặt tên/log).

        Returns:
            :class:`RemoteVideoRef` mới đã ACTIVE.

        Raises:
            VideoContextError: Khi không tìm thấy ffmpeg/nguồn hoặc cắt thất bại.
        """
        source_path = Path(source_path)
        if not source_path.exists():
            raise VideoContextError(f"Không tìm thấy video gốc để tải lại: {source_path}")
        ffmpeg = find_ffmpeg()
        if not ffmpeg:
            raise VideoContextError("Không tìm thấy ffmpeg để cắt video tải lại.")
        out_dir = self._work_dir or source_path.parent
        part_path = out_dir / f"{source_path.stem}.heal{index:02d}.mp4"
        duration = max(0.1, end_sec - start_sec)
        command = [
            ffmpeg, "-nostdin", "-y", "-hide_banner", "-loglevel", "error",
            "-ss", f"{start_sec:.3f}", "-i", str(source_path),
            "-t", f"{duration:.3f}", "-c", "copy",
            "-avoid_negative_ts", "make_zero", str(part_path),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, timeout=600, stdin=subprocess.DEVNULL, **no_window_kwargs())
        except subprocess.SubprocessError as exc:
            raise VideoContextError(f"Cắt đoạn tải lại thất bại: {exc}") from exc
        chunk = VideoChunk(index, part_path, start_sec, end_sec, is_full_video=False)
        try:
            return self.upload_chunk(chunk)
        finally:
            try:
                if part_path.exists():
                    part_path.unlink()
            except OSError:
                pass

    def cleanup_local_chunks(self, plan: VideoContextPlan) -> None:
        """Xoá các file đoạn tạm (không xoá video gốc)."""
        for chunk in plan.chunks:
            if chunk.is_full_video:
                continue
            try:
                if chunk.path.exists():
                    chunk.path.unlink()
            except OSError as exc:
                logger.warning("Không xoá được đoạn tạm %s: %s", chunk.path, exc)

    def sweep_chunk_cache(
        self, protected_paths: frozenset[str] = frozenset()
    ) -> tuple[int, int]:
        """[v3.23.166] Dọn cache file đoạn nén cục bộ theo ngân sách (LRU + tuổi).

        Quét thư mục làm việc tìm các file đoạn (``*.ctxpart*.mp4`` và ``*.heal*.mp4``),
        rồi xoá theo luật ngân sách: giữ file phiên hiện tại, xoá file quá hạn tuổi
        trước, sau đó xoá LRU tới khi tổng dung lượng về dưới ngân sách. An toàn khi
        chưa cấu hình work_dir (chunk nằm cạnh video gốc — không quét để tránh dọn
        nhầm file người dùng).

        Args:
            protected_paths: Đường dẫn (dạng chuỗi) KHÔNG được xoá — thường là các
                file đoạn của phiên vừa chuẩn bị (đang/sắp dùng).

        Returns:
            ``(số_file_đã_xoá, tổng_byte_giải_phóng)``.
        """
        if self._work_dir is None:
            return (0, 0)
        if self._chunk_cache_max_total_bytes <= 0 and self._chunk_cache_max_age_seconds <= 0:
            return (0, 0)

        work_dir = Path(self._work_dir)
        files: list[ChunkFileInfo] = []
        size_by_path: dict[str, int] = {}
        for pattern in ("*.ctxpart*.mp4", "*.heal*.mp4"):
            for path in work_dir.glob(pattern):
                try:
                    stat = path.stat()
                except OSError:
                    continue
                key = str(path)
                files.append(ChunkFileInfo(key, stat.st_size, stat.st_mtime))
                size_by_path[key] = stat.st_size

        to_delete = plan_chunk_cache_cleanup(
            files,
            max_total_bytes=self._chunk_cache_max_total_bytes,
            max_age_seconds=self._chunk_cache_max_age_seconds,
            now_epoch=time.time(),
            protected_paths=protected_paths,
        )

        deleted = 0
        freed = 0
        for path_str in to_delete:
            try:
                Path(path_str).unlink()
                deleted += 1
                freed += size_by_path.get(path_str, 0)
            except OSError as exc:
                logger.warning("Không dọn được cache đoạn %s: %s", path_str, exc)
        if deleted:
            logger.info(
                "Dọn cache đoạn video: xoá %d file, giải phóng %.1f MB "
                "(giữ %d file phiên hiện tại).",
                deleted, freed / (1024 * 1024), len(protected_paths),
            )
        return (deleted, freed)

    # ── Vòng đời file cloud (Bước 3) ─────────────────────────────────────────
    def delete_remote_files(self, remote_names: list[str]) -> dict[str, bool]:
        """[v3.23.17] Xoá các file đã upload trên Gemini cloud (giải phóng dung lượng).

        Dùng khi đã dịch xong hoặc người dùng yêu cầu dọn. Xoá cả bản ghi cache nội
        bộ để lần sau không "tái dùng" file đã biến mất. Không ném lỗi nếu file đã
        không còn (coi như đã xoá).

        [v3.23.609] Thử LẦN LƯỢT mọi key đã biết (key đang dùng trước). File Gemini thuộc
        dự án của key đã tải nó; key khác nhận 403 — mà 403 cũng là mã Gemini trả cho file
        đã hết hạn, nên trước đây file tải bằng key thứ hai bị báo "đã xoá" dù vẫn còn.

        Args:
            remote_names: Danh sách tên file cloud (vd 'files/abc123').

        Returns:
            Ánh xạ ``{remote_name: đã_xoá_hoặc_không_còn}``; ``False`` = còn lỗi khác
            (mạng, máy chủ) với mọi key, nên thử lại sau.
        """
        if not remote_names:
            return {}
        clients = self._deletion_clients()
        if not clients:
            return {name: False for name in remote_names}
        results = {
            name: self._delete_with_any_key(name, clients) for name in remote_names
        }
        for name, gone in results.items():
            # File còn trên cloud (xoá lỗi) vẫn dùng lại được → giữ bản ghi cache.
            if gone:
                self._cache_forget_remote(name)
        gone_names = {name for name, gone in results.items() if gone}
        for sig, refs in list(self._inmem_refs.items()):
            if any(r.remote_name in gone_names for r in refs):
                del self._inmem_refs[sig]
        return results

    def _deletion_clients(self) -> list[tuple[str, Any]]:
        """``(key, client)`` cho mọi key đã biết, key đang dùng đứng đầu."""
        clients: list[tuple[str, Any]] = []
        try:
            clients.append((self._api_key, self._ensure_client()))
        except VideoContextError as exc:
            logger.warning("Không khởi tạo được client để xoá file cloud: %s", exc)
        for key in self._known_api_keys:
            if key == self._api_key:
                continue
            try:
                clients.append((key, self._make_client(key)))
            except VideoContextError as exc:
                logger.warning("Không khởi tạo được client cho key phụ: %s", exc)
        return clients

    def _delete_with_any_key(self, name: str, clients: list[tuple[str, Any]]) -> bool:
        """Xoá ``name`` bằng key đầu tiên có quyền; ``True`` nếu đã xoá hoặc không còn."""
        failed_otherwise = False
        for position, (_key, client) in enumerate(clients, start=1):
            try:
                client.files.delete(name=name)
            except Exception as exc:  # noqa: BLE001 — SDK ném nhiều loại lỗi HTTP khác nhau
                if not self._is_recoverable_remote_error(exc):
                    failed_otherwise = True
                    logger.warning(
                        "Không xoá được file cloud %s (key %d): %s", name, position, exc
                    )
                continue
            logger.info("Đã xoá file cloud: %s (key %d).", name, position)
            return True
        if failed_otherwise:
            return False
        logger.info("File cloud %s đã không còn (xem như đã xoá).", name)
        return True

    def _cache_forget_remote(self, remote_name: str) -> None:
        """Xoá bản ghi cache theo remote_name (sau khi file cloud đã bị xoá)."""
        if not self._cache_db_path:
            return
        import sqlite3
        try:
            with sqlite3.connect(str(self._cache_db_path)) as conn:
                conn.execute(
                    "DELETE FROM video_context_uploads WHERE remote_name = ?",
                    (remote_name,),
                )
                conn.commit()
        except sqlite3.Error as exc:
            logger.warning("Không xoá được bản ghi cache cho %s: %s", remote_name, exc)

    # ── Điều phối toàn trình: lập kế hoạch → cắt → tải → dọn ─────────────────
    def prepare_and_upload(
        self,
        video_path: Path,
        progress_cb: "Callable[[str], None] | None" = None,
        cancel_cb: "Callable[[], bool] | None" = None,
    ) -> list[RemoteVideoRef]:
        """Chuẩn bị đầy đủ ngữ cảnh video: cắt (nếu cần) → tải từng đoạn → trả refs.

        Tự dọn file đoạn tạm sau khi tải xong. Báo tiến trình qua ``progress_cb`` và
        cho phép huỷ giữa chừng qua ``cancel_cb``.

        Returns:
            Danh sách :class:`RemoteVideoRef` (đã ACTIVE) theo thứ tự thời gian.
        """
        # [v3.23.14] CHỐNG LẶP trong RAM (cùng phiên app): nếu vừa chuẩn bị xong cho
        # đúng file này, trả lại refs đã có — bỏ qua cả materialize lẫn upload.
        cache_signature = self._video_signature(video_path)
        # [v3.23.479] Người dùng bật "bỏ cache cloud" (vd đã xoá file trên cloud) → BỎ QUA mọi
        # tái dùng (RAM/DB/khoá nguồn), buộc cắt+nén+tải lại từ đầu.
        skip_cache = self._ignore_cloud_cache
        if skip_cache and progress_cb:
            progress_cb("Bỏ cache — chuẩn bị lại video ngữ cảnh từ đầu…")
        cached_refs = None if skip_cache else self._inmem_refs.get(cache_signature)
        if cached_refs and self._refs_still_active(cached_refs):
            if progress_cb:
                progress_cb("Tái dùng ngữ cảnh video đã chuẩn bị (không cắt/tải lại).")
            logger.info(
                "Tái dùng %d đoạn video đã chuẩn bị cho %s (bỏ qua cắt+upload).",
                len(cached_refs), video_path.name,
            )
            return list(cached_refs)

        plan = self.plan_chunks(video_path)

        # [v3.23.21] CHỐNG LẶP BỀN VỮNG (qua DB, sống giữa các lần mở app): trước khi
        # CẮT+NÉN (tốn CPU/thời gian), tra cache plan trong DB theo chữ ký video. Nếu
        # đủ refs cho mọi đoạn VÀ cloud còn ACTIVE → bỏ qua HOÀN TOÀN cắt+nén+upload.
        # Đây là sửa lỗi: trước đây cache cloud chỉ tra SAU khi đã cắt (materialize),
        # nên dù không upload lại thì vẫn cắt+nén lại mỗi lần — rất chậm.
        persisted = None if skip_cache else self._plan_cache_lookup(
            cache_signature, len(plan.chunks)
        )
        if persisted and self._refs_still_active(persisted):
            if progress_cb:
                progress_cb("Tái dùng ngữ cảnh video đã lưu (không cắt/tải lại).")
            logger.info(
                "Tái dùng %d đoạn video từ cache DB cho %s (bỏ qua cắt+nén+upload).",
                len(persisted), video_path.name,
            )
            self._inmem_refs[cache_signature] = list(persisted)
            return list(persisted)

        # [v3.23.131] TRƯỚC khi cắt+nén (tốn ~18s/đoạn): tra cloud theo khoá NGUỒN. Bắt
        # đúng trường hợp đoạn đã từng nén+upload ở phiên trước (cloud còn sống) nhưng
        # plan-cache theo chữ ký lại miss (vd phiên trước upload xong nhưng chưa kịp lưu
        # plan-cache do thoát giữa chừng). Tránh nén lại rồi mới phát hiện "đã có sẵn".
        from_source = None if skip_cache else self._resolve_chunks_from_source_cache(
            plan, video_path
        )
        if from_source:
            if progress_cb:
                progress_cb("Tái dùng đoạn video đã có trên cloud (bỏ qua cắt+nén).")
            logger.info(
                "Tái dùng %d đoạn video trên cloud theo khoá nguồn cho %s "
                "(bỏ qua cắt+nén+upload).",
                len(from_source), video_path.name,
            )
            self._inmem_refs[cache_signature] = list(from_source)
            self._plan_cache_store(cache_signature, from_source)
            return list(from_source)

        if plan.needs_split:
            if progress_cb:
                progress_cb(
                    f"Cắt video thành {len(plan.chunks)} đoạn "
                    f"(~{plan.estimated_tokens:,} token)…"
                )
            self.materialize_chunks(plan)
        refs: list[RemoteVideoRef] = []
        try:
            for chunk in plan.chunks:
                if cancel_cb is not None and cancel_cb():
                    raise VideoContextError("Người dùng đã huỷ khi đang tải video.")
                if progress_cb:
                    progress_cb(
                        f"Tải đoạn {chunk.index + 1}/{len(plan.chunks)} lên Gemini…"
                    )
                refs.append(self.upload_chunk(chunk))
                # [v3.23.131] Lưu thêm theo khoá NGUỒN để lần sau tra được TRƯỚC khi nén.
                try:
                    self._cache_store(
                        self._chunk_source_key(video_path, chunk), refs[-1].remote_name
                    )
                except (OSError, ValueError):
                    pass
        finally:
            # [v3.23.157] GIỮ file nén cục bộ (KHÔNG cleanup ở đây): file nén không phụ
            # thuộc API key — chỉ bước UPLOAD gắn key. Trước đây xoá ngay sau upload nên
            # mỗi lần xoay key giữa phiên (quota cạn) phải CẮT+NÉN LẠI toàn bộ (~5 phút
            # x 11 đoạn x nhiều lần như log The Hot Spot); nay xoay key chỉ tốn upload
            # lại, materialize tự bỏ qua nén khi file đã tồn tại. Người dùng có thể dọn
            # thủ công qua cleanup_local_chunks (vẫn giữ trong port cho UI/tiện ích).
            pass
        self._inmem_refs[cache_signature] = list(refs)
        self._plan_cache_store(cache_signature, refs)  # lưu DB để lần sau khỏi cắt lại
        # [v3.23.166] Dọn cache đoạn theo ngân sách SAU mỗi lần chuẩn bị — GIỮ nguyên
        # các đoạn của phiên hiện tại (protected), chỉ dọn file cũ/thừa của phiên trước
        # để cache không phình mãi. Lỗi dọn không được ảnh hưởng kết quả chuẩn bị.
        try:
            protected = frozenset(
                str(chunk.path) for chunk in plan.chunks if not chunk.is_full_video
            )
            self.sweep_chunk_cache(protected_paths=protected)
        except OSError as exc:
            logger.warning("Dọn cache đoạn sau chuẩn bị gặp lỗi (bỏ qua): %s", exc)
        return refs

    def upload_chunks_for_ranges(
        self,
        video_path: Path,
        ranges: list[tuple[float, float]],
        cancel_cb: "Callable[[], bool] | None" = None,
    ) -> list[RemoteVideoRef]:
        """[v3.23.550] Tải lên (bằng key HIỆN TẠI) CHỈ các đoạn có khoảng thời gian cần dùng.

        Vì sao cần: khi xoay key giữa chừng, file của key cũ không truy cập được. Trước đây
        cách chữa là gọi :meth:`prepare_and_upload` — tải lại TOÀN BỘ phim. App_log thật: key #1
        hết quota đúng ở yêu cầu CUỐI (đoạn 22/22), ứng dụng tải lại cả 22 đoạn mất 5 phút 45
        giây chỉ để gửi đúng một yêu cầu cần đúng một đoạn. Nay tải lười từng đoạn: đoạn nào
        được dùng tới mới tải, đoạn không bao giờ dùng lại thì không tốn gì.

        Không ghi đè cache kế hoạch trọn bộ (kết quả ở đây chỉ là một phần của phim).

        Args:
            video_path: Video gốc.
            ranges: Các khoảng ``(start_sec, end_sec)`` đúng như trong ``RemoteVideoRef`` cũ.
            cancel_cb: Callback huỷ hợp tác.

        Returns:
            Refs mới cho các đoạn khớp khoảng, theo thứ tự thời gian. Rỗng nếu không khớp.
        """
        plan = self.plan_chunks(video_path)
        wanted = [
            chunk for chunk in plan.chunks
            if any(
                abs(chunk.start_sec - start) < 0.5 and abs(chunk.end_sec - end) < 0.5
                for start, end in ranges
            )
        ]
        if not wanted:
            return []
        if plan.needs_split:
            ffmpeg = find_ffmpeg()
            if not ffmpeg:
                raise VideoContextError(
                    missing_ffmpeg_message(
                        feature="Ngữ cảnh video khi dịch (cắt đoạn gửi Gemini)"
                    )
                )
            self._encode_missing_chunks(ffmpeg, plan.source_path, wanted)
        refs: list[RemoteVideoRef] = []
        for chunk in wanted:
            if cancel_cb is not None and cancel_cb():
                raise VideoContextError("Người dùng đã huỷ khi đang tải video.")
            if not (chunk.is_full_video or chunk.path.exists()):
                continue
            refs.append(self.upload_chunk(chunk))
            try:
                self._cache_store(
                    self._chunk_source_key(video_path, chunk), refs[-1].remote_name
                )
            except (OSError, ValueError):
                pass
        logger.info(
            "Tải lại %d/%d đoạn video CẦN DÙNG bằng key hiện tại (bỏ qua %d đoạn còn lại).",
            len(refs), len(plan.chunks), len(plan.chunks) - len(refs),
        )
        return refs

    def _plan_cache_lookup(
        self, video_signature: str, expected_chunks: int
    ) -> list[RemoteVideoRef] | None:
        """Tra cache plan trong DB: trả refs nếu có ĐỦ đoạn cho chữ ký video này."""
        if not self._cache_db_path:
            return None
        import sqlite3
        try:
            with sqlite3.connect(str(self._cache_db_path)) as conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS video_plan_cache ("
                    "video_sig TEXT, chunk_index INTEGER, remote_name TEXT, "
                    "start_sec REAL, end_sec REAL, "
                    "PRIMARY KEY (video_sig, chunk_index))"
                )
                rows = conn.execute(
                    "SELECT chunk_index, remote_name, start_sec, end_sec "
                    "FROM video_plan_cache WHERE video_sig=? ORDER BY chunk_index",
                    (video_signature,),
                ).fetchall()
        except sqlite3.Error as exc:
            logger.warning("Lỗi tra cache plan video: %s", exc)
            return None
        if len(rows) != expected_chunks or expected_chunks == 0:
            return None
        return [
            RemoteVideoRef(int(r[0]), str(r[1]), float(r[2]), float(r[3]), "ACTIVE")
            for r in rows
        ]

    def _plan_cache_store(self, video_signature: str, refs: list[RemoteVideoRef]) -> None:
        """Lưu mapping (chữ ký video → các đoạn cloud) vào DB để tái dùng giữa phiên."""
        if not self._cache_db_path or not refs:
            return
        import sqlite3
        try:
            with sqlite3.connect(str(self._cache_db_path)) as conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS video_plan_cache ("
                    "video_sig TEXT, chunk_index INTEGER, remote_name TEXT, "
                    "start_sec REAL, end_sec REAL, "
                    "PRIMARY KEY (video_sig, chunk_index))"
                )
                conn.execute("DELETE FROM video_plan_cache WHERE video_sig=?",
                             (video_signature,))
                conn.executemany(
                    "INSERT OR REPLACE INTO video_plan_cache VALUES (?,?,?,?,?)",
                    [(video_signature, r.chunk_index, r.remote_name, r.start_sec, r.end_sec)
                     for r in refs],
                )
                conn.commit()
        except sqlite3.Error as exc:
            logger.warning("Không lưu được cache plan video: %s", exc)

    @staticmethod
    def _video_signature(video_path: Path) -> str:
        """Chữ ký nhận dạng file theo đường dẫn + mtime + kích thước (rẻ, đủ chính xác)."""
        try:
            st = video_path.stat()
            return f"{video_path.resolve()}::{int(st.st_mtime)}::{st.st_size}"
        except OSError:
            return str(video_path)

    def _refs_still_active(self, refs: list[RemoteVideoRef]) -> bool:
        """Kiểm các đoạn cloud còn ACTIVE (chưa bị Google xoá sau ~48h)."""
        if not refs:
            return False
        try:
            client = self._ensure_client()
        except VideoContextError:
            return False
        for ref in refs:
            try:
                remote = client.files.get(name=ref.remote_name)
                if getattr(remote.state, "name", "") != "ACTIVE":
                    return False
            except Exception:  # noqa: BLE001 — bất kỳ lỗi cloud → coi như cần làm lại
                return False
        return True
