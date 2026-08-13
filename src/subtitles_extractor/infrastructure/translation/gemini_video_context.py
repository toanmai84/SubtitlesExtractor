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
from subtitles_extractor.infrastructure.process.hidden_process import no_window_kwargs
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

from subtitles_extractor.infrastructure.media.ffmpeg_locator import (
    find_ffmpeg,
    missing_ffmpeg_message,
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


def _vf_gpu_full(height: int, fps_str: str) -> str:
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
        height: Chiều cao đích (px), width tự tính giữ tỉ lệ và ép chẵn (-2).
        fps_str: Số khung/giây đích dạng chuỗi (vd "1").

    Returns:
        Chuỗi cho tham số ``-vf`` của ffmpeg.
    """
    return f"fps={fps_str},scale_cuda=w=-2:h={height}:format=nv12,hwdownload,format=nv12"


def _vf_gpu_upload(height: int, fps_str: str) -> str:
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
        height: Chiều cao đích (px), width tự tính giữ tỉ lệ và ép chẵn (-2).
        fps_str: Số khung/giây đích dạng chuỗi.

    Returns:
        Chuỗi cho tham số ``-vf`` của ffmpeg.
    """
    return (
        f"fps={fps_str},hwupload_cuda,scale_cuda=w=-2:h={height}:format=nv12,"
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


def _vf_cpu_scale(height: int, fps_str: str) -> str:
    """[v3.23.157] Chuỗi filter scale CPU (các tầng fallback) — ``fps`` đặt ĐẦU.

    Trước đây ``fps`` nằm CUỐI chuỗi nên CPU scale toàn bộ ~24fps rồi mới vứt còn
    fps đích -> lãng phí ~24x công scale (The Hot Spot: 25-46s/đoạn). Đặt fps đầu
    chuỗi cho kết quả GIỐNG HỆT nhưng chỉ scale số khung thật sự cần.

    Args:
        height: Chiều cao đích (px), width tự tính giữ tỉ lệ và ép chẵn (-2).
        fps_str: Số khung/giây đích dạng chuỗi.

    Returns:
        Chuỗi cho tham số ``-vf`` của ffmpeg.
    """
    return f"fps={fps_str},scale=-2:{height},format=yuv420p"


def video_tokens_per_sec(media_resolution: str) -> int:
    """Trả về số token/giây video theo mức media_resolution (mặc định medium=300)."""
    return _VIDEO_TOKENS_PER_SEC_BY_RES.get(
        (media_resolution or "medium").lower(), 300
    )


class VideoContextError(Exception):
    """Lỗi khi chuẩn bị/tải ngữ cảnh video."""


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
        for piece in (raw or "").replace(",", "\n").splitlines():
            key = piece.strip()
            if key:
                return key
        return (raw or "").strip()

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
    ) -> None:
        # [v3.23.130] Hỗ trợ chuỗi NHIỀU key (mỗi dòng/phẩy một key) từ giao diện: tải
        # video chỉ cần MỘT key → lấy key ĐẦU TIÊN hợp lệ. Nếu để nguyên chuỗi gộp, nó
        # lọt vào header 'x-goog-api-key' kèm '\n' → LocalProtocolError (crash khi tải).
        self._api_key = self._first_api_key(api_key)
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
        self._client: Any = None
        # [v3.23.14] Cache refs trong phiên: tránh cắt+upload lặp khi analyze rồi
        # translate cùng một video. Khoá = chữ ký file (đường dẫn+mtime+size).
        self._inmem_refs: dict[str, list[RemoteVideoRef]] = {}
        # [v3.23.21] Cache kết quả dò NVENC (None=chưa dò, True/False=đã biết). Dùng
        # để ưu tiên GPU khi nén video ngữ cảnh; tự tắt nếu GPU lỗi giữa chừng.
        self._nvenc_available: bool | None = None
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
                f".h{self._resolution_height}f{self._fps:g}.mp4"
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
        for chunk in plan.chunks:
            if chunk.is_full_video or chunk.path.exists():
                continue
            self._encode_low_res_chunk(ffmpeg, plan.source_path, chunk)

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
        gpu_timeout = 90.0
        cpu_timeout = max(300.0, chunk.duration_sec * 0.5)
        succeeded_with_gpu: bool | None = None  # None = chưa lệnh nào thành công
        scale_cuda_failed_this_chunk = False
        succeeded_cmd_used_scale_cuda = False
        for use_gpu, cmd in encode_cmds:
            # Bỏ qua các lệnh GPU nếu phiên này đã xác định GPU không dùng được.
            if use_gpu and self._nvenc_available is False:
                continue
            timeout_s = gpu_timeout if use_gpu else cpu_timeout
            try:
                subprocess.run(cmd, check=True, capture_output=True, timeout=timeout_s, **no_window_kwargs())
                if tmp_out.exists() and tmp_out.stat().st_size >= 1024:
                    mode = self._encode_mode_label(cmd)
                    logger.info("Nén đoạn %d bằng %s.", chunk.index, mode)
                    succeeded_with_gpu = use_gpu
                    succeeded_cmd_used_scale_cuda = any(
                        "scale_cuda" in str(a) for a in cmd
                    )
                    break
                tmp_out.unlink(missing_ok=True)
            except subprocess.TimeoutExpired:
                last_exc = None
                tmp_out.unlink(missing_ok=True)
                if use_gpu:
                    # [v3.23.358] Tier GPU TREO tới timeout là tín hiệu MẠNH cho thấy đường
                    # mã hoá GPU hỏng trên máy này (driver/hw). Tắt NVENC cho CẢ PHIÊN để
                    # BỎ các cấp GPU còn lại (nhảy thẳng CPU), tránh treo thêm 180s/tier và
                    # tránh làm hỏng cả tác vụ phân tích ngữ cảnh.
                    if self._nvenc_available is not False:
                        self._nvenc_available = False
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
                        "Nén CPU đoạn %d treo quá %.0fs — huỷ.", chunk.index, cpu_timeout,
                    )
            except subprocess.SubprocessError as exc:
                last_exc = exc
                tmp_out.unlink(missing_ok=True)
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
            raise VideoContextError(
                f"Nén đoạn {chunk.index} thất bại (cả GPU lẫn CPU). {detail}"
            )

        # Đưa file về vị trí đích (tên có thể chứa CJK — dùng Python, KHÔNG qua ffmpeg).
        try:
            chunk.path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(tmp_out), str(chunk.path))
        except OSError as exc:
            tmp_out.unlink(missing_ok=True)
            raise VideoContextError(
                f"Không di chuyển đoạn {chunk.index} về đích: {exc}"
            ) from exc

    def _build_encode_commands(
        self, ffmpeg: str, source_path: Path, chunk: VideoChunk, tmp_out: Path
    ) -> list[tuple[bool, list[str]]]:
        """Tạo danh sách lệnh ffmpeg theo thứ tự ưu tiên: (GPU nếu có) → CPU.

        Mỗi phần tử là (dùng_gpu, lệnh). Bên gọi thử lần lượt tới khi tạo được file.
        """
        # [v3.23.29] Tối ưu theo đề xuất người dùng + tài liệu NVIDIA/ffmpeg:
        # - -threads 0: tối đa luồng CPU khi giải mã HEVC/VP9 nặng (cấp CPU/NVENC-only).
        # - -max_muxing_queue_size 1024: chống lỗi "Too many packets buffered" khi hạ
        #   fps mạnh (60→1) trên đoạn dài — đây là nguyên nhân chính gây TREO.
        # - -ss/-t TRƯỚC -i: fast seek (nhanh hơn nhiều với đoạn xa).
        # - -fflags +genpts: sinh lại PTS cho luồng giải mã phần cứng (NVDEC) vốn có thể
        #   thiếu timestamp, tránh ffmpeg treo chờ frame ở đoạn cuối khi kèm -r 1.
        #   (KHÔNG dùng -vsync 0: đã deprecated, thay bằng -fps_mode nhưng option này
        #   không có trên FFmpeg cũ → gỡ hẳn để tương thích ngược, genpts +
        #   max_muxing_queue_size đã đủ chống treo.)
        # - format=nv12/yuv420p ép pixel format rõ ràng (chống lỗi 10-bit).
        # - colorspace/primaries/trc=1 (BT.709) + profile main: chuẩn hoá SDR.
        base_flags = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-threads", "0"]
        input_args = [
            "-ss", f"{chunk.start_sec:.3f}",
            "-t", f"{chunk.duration_sec:.3f}",
            "-i", str(source_path),
        ]
        av_map_and_standard_args = [
            "-map", "0:v:0", "-map", "0:a:0?",
            "-c:a", "aac", "-ac", "1", "-ar", "16000", "-b:a", "64k",
            "-avoid_negative_ts", "make_zero",
            "-colorspace", "1", "-color_primaries", "1", "-color_trc", "1",
            "-max_muxing_queue_size", "1024",
        ]

        commands: list[tuple[bool, list[str]]] = []

        # [v3.23.39] Tham số nén từ Cài đặt (thay cho hardcode 360p/1fps/cq32/crf30).
        h = self._resolution_height
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
                    "-vf", _vf_gpu_full(h, r_str),
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
                    "-vf", _vf_gpu_upload(h, r_str),
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
                    "-vf", _vf_cpu_scale(h, r_str),
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
                    "-vf", _vf_cpu_scale(h, r_str),
                    "-c:v", "h264_nvenc", "-preset", "p4", "-cq", cq_str,
                    "-profile:v", "main",
                    str(tmp_out),
                ]
            )
            commands.append((True, gpu_encode_cmd))

        # CẤP 3 — CPU THUẦN (LUÔN có, kể cả khi không NVENC — đây là fallback cuối).
        cpu_cmd = (
            base_flags
            + input_args
            + av_map_and_standard_args
            + [
                "-vf", _vf_cpu_scale(h, r_str),
                "-c:v", "libx264", "-preset", "veryfast", "-crf", crf_str,
                "-profile:v", "main",
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
        try:
            from google import genai  # type: ignore
        except ImportError as exc:
            raise VideoContextError(
                "Chưa cài google-genai. Cài: pip install google-genai"
            ) from exc
        self._client = genai.Client(api_key=self._api_key)
        return self._client

    def _cache_lookup(self, file_hash: str) -> str | None:
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
                    "SELECT remote_name FROM video_context_uploads WHERE file_hash=?",
                    (file_hash,),
                ).fetchone()
                return row[0] if row else None
        except sqlite3.Error as exc:
            logger.warning("Lỗi tra cache upload video: %s", exc)
            return None

    def _cache_store(self, file_hash: str, remote_name: str) -> None:
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
                    "INSERT OR REPLACE INTO video_context_uploads VALUES (?,?,?)",
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
            f"|cq{self._nvenc_cq}|crf{self._cpu_crf}"
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

        logger.info("Đang tải đoạn %d (%.0f–%.0fs) lên Gemini…",
                    chunk.index, chunk.start_sec, chunk.end_sec)
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
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-ss", f"{start_sec:.3f}", "-i", str(source_path),
            "-t", f"{duration:.3f}", "-c", "copy",
            "-avoid_negative_ts", "make_zero", str(part_path),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, timeout=600, **no_window_kwargs())
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

        Args:
            remote_names: Danh sách tên file cloud (vd 'files/abc123').

        Returns:
            Ánh xạ ``{remote_name: đã_xoá_thành_công}``.
        """
        if not remote_names:
            return {}
        results: dict[str, bool] = {}
        client: Any
        try:
            client = self._ensure_client()
        except VideoContextError as exc:
            logger.warning("Không khởi tạo được client để xoá file cloud: %s", exc)
            return {name: False for name in remote_names}

        for name in remote_names:
            try:
                client.files.delete(name=name)
                results[name] = True
                logger.info("Đã xoá file cloud: %s", name)
            except Exception as exc:  # noqa: BLE001 — coi 404/đã xoá là thành công
                if self._is_recoverable_remote_error(exc):
                    results[name] = True  # đã không còn → xem như đã xoá
                    logger.info("File cloud %s đã không còn (xem như đã xoá).", name)
                else:
                    results[name] = False
                    logger.warning("Không xoá được file cloud %s: %s", name, exc)
            self._cache_forget_remote(name)
        # Xoá khỏi cache refs trong-bộ-nhớ nếu có.
        for sig, refs in list(self._inmem_refs.items()):
            if any(r.remote_name in remote_names for r in refs):
                del self._inmem_refs[sig]
        return results

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
        cached_refs = self._inmem_refs.get(cache_signature)
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
        persisted = self._plan_cache_lookup(cache_signature, len(plan.chunks))
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
        from_source = self._resolve_chunks_from_source_cache(plan, video_path)
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
