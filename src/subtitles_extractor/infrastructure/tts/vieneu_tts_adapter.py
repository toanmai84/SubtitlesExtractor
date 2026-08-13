"""Adapter VieNeu-TTS — TTS tiếng Việt on-device chất lượng cao, chạy offline.

VieNeu-TTS là mô hình TTS tiếng Việt tiên tiến, tổng hợp giọng nói tự nhiên 24-48kHz,
hỗ trợ giọng dựng sẵn (preset), voice cloning tức thì từ 3-5 giây audio tham chiếu, và
code-switching Việt-Anh. Chạy hoàn toàn offline: trên CPU dùng ONNX Runtime (torch-free),
trên máy có CUDA tự chuyển sang PyTorch.

Cài đặt::

    pip install vieneu soundfile

Model được tải tự động về ``~/.cache/huggingface`` lần đầu.

.. note::
    [v3.23.253] VieNeu-TTS dùng license **Apache 2.0** (miễn phí, kể cả mục đích thương
    mại) — theo README chính thức của tác giả (github.com/pnnbao97/VieNeu-TTS). Ghi chú
    licence phi-thương-mại trong phiên bản cũ là SAI, đã sửa. Người dùng vẫn nên kiểm tra
    license bản model cụ thể mình tải nếu dùng model fine-tune riêng.
"""

from __future__ import annotations

import contextlib
import inspect
import logging
import os
import subprocess
from subtitles_extractor.infrastructure.process.hidden_process import (
    no_window_kwargs,
)
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Final

import numpy as np

from subtitles_extractor.domain.ports.subtitle_tts_port import (
    SubtitleTTSPort,
    TTSCancellationCallback,
    TTSCancelledError,
    TTSGenerationError,
    TTSProgressCallback,
    TTSRequest,
    TTSSegmentResult,
    TTSUnavailableError,
)
from subtitles_extractor.infrastructure.tts.audio_utils import (
    is_effectively_silent,
    resample_audio,
    shorter_take,
    trim_edge_silence,
)
from subtitles_extractor.infrastructure.tts.text_prep import (
    dem_am_tiet,
)
from subtitles_extractor.infrastructure.tts.text_prep import (
    has_speakable_content,
)
from subtitles_extractor.infrastructure.tts.text_prep import (
    preprocess_tts_text as _preprocess_tts_text,
)
from subtitles_extractor.infrastructure.tts.text_prep import (
    skip_from_request as _skip_from_request,
)
from subtitles_extractor.infrastructure.tts.timing_math import (
    MAX_LEAD_IN_S as _MAX_LEAD_IN_S,
)
from subtitles_extractor.infrastructure.tts.timing_math import (
    MASTER_TAIL_PAD_S as _MASTER_TAIL_PAD_S,  # noqa: F401  (re-export cho test)
)
from subtitles_extractor.infrastructure.tts.timing_math import (
    QUALITY_STRETCH_CAP as _QUALITY_STRETCH_CAP,  # noqa: F401  (re-export cho test)
)
from subtitles_extractor.infrastructure.tts.timing_math import (
    VIENEU_MIN_BASE_S,
    VIENEU_MIN_PER_SYLLABLE_S,
    compute_fit_stretch_ratio,
    effective_available_seconds,
    fit_limit_samples,
    generation_time_cap_seconds,
    is_abnormally_long,  # noqa: F401  (re-export cho test)
    is_abnormally_long_vs_floor,
    lead_in_seconds,
    master_length_samples,
    stretch_ratio_cap,
    total_speed_ratio,
)

logger = logging.getLogger(__name__)

# [v3.23.220] Các tên trên là RE-EXPORT từ những module thuần dùng chung
# (``timing_math`` / ``audio_utils`` / ``text_prep``). Trước đây chúng được ĐỊNH NGHĨA
# tại đây, nên Gemini phải import ngược từ adapter VieNeu (6 chỗ) và Edge phải lazy-import
# né vòng tròn — mọi sửa đổi cho VieNeu trở thành rủi ro thầm lặng cho hai engine kia.
# Giữ re-export để test/monkeypatch hiện có không phải sửa; code MỚI import thẳng module
# thuần.

# Sample rate nội bộ của pipeline (khớp Edge để master_finalize nhất quán). VieNeu
# xuất 24kHz (GGUF) hoặc 48kHz (v3 Turbo); luôn resample về mức này trước hậu xử lý.
_PIPELINE_SAMPLE_RATE = 24_000

# [v3.23.257] Frame rate v3 Turbo để quy đổi trần (giây -> max_new_frames). Đo từ
# source vieneu 3.2.3: base.py hop_length=480 @ sr 48000 -> 100 frame/s.
_V3TURBO_FRAMES_PER_SECOND = 100

# [v3.23.264] Chống ngân dài: phạt lặp token audio (mặc định SDK 1.2). Nâng nhẹ lên 1.3 để
# giảm "ngân" (lặp frame nguyên âm) mà không gây méo. Đây là cách chống ngân AN TOÀN nhất
# (không như temperature thấp có thể làm câu ngắn bị model từ chối đọc).
_VIENEU_REPETITION_PENALTY = 1.3

# Chế độ engine VieNeu hợp lệ (ánh xạ tới tham số ``mode`` của SDK).
_VALID_MODES = ("standard", "turbo", "v3turbo")

# [v3.23.254] Ánh xạ ``emotion`` (API cũ) -> ``style`` (API 3.x). [v3.23.255] ĐÃ XÁC MINH
# từ tài liệu VieNeu 3.2.3 (pypi.org/project/vieneu/3.2.3): ``style`` nhận "tu_nhien"
# (natural) | "tin_tuc" (news) | "doc_truyen" (storytelling), truyền per-call trong
# ``infer`` hoặc ở constructor. App ánh xạ 2 sắc thái đang expose sang tên chính thức.
_EMOTION_TO_STYLE = {
    "natural": "tu_nhien",
    "storytelling": "doc_truyen",
}


# Cache engine ở cấp MODULE theo khoá (mode, emotion). VieNeu nạp model rất nặng (~6s)
# và trang UI (``list_speakers``) với worker (``generate``) là hai INSTANCE adapter khác
# nhau (container tạo mới mỗi lần) — nếu cache theo instance, model bị nạp 2 lần. Cache
# theo module giúp mọi instance cùng cấu hình dùng chung một engine đã nạp.
_ENGINE_CACHE: dict[tuple[str, str], Any] = {}
# [v3.23.200] Khoá nạp engine: nạp giọng nền (UI) + Generate (worker) có thể trùng thời
# điểm -> không lock sẽ nạp model 2 lần (~9s + RAM x2). Double-checked locking.
_ENGINE_LOCK = threading.Lock()


def resolve_device_env(force_cpu: bool) -> dict[str, str]:
    """Xác định biến môi trường cần đặt để chọn thiết bị chạy VieNeu (hàm thuần).

    VieNeu SDK tự chuyển sang engine PyTorch khi phát hiện CUDA. Trên máy có bộ
    PyTorch/cuDNN lệch phiên bản, việc này gây lỗi tải DLL (``WinError 127`` với
    ``cudnn_engines_precompiled64_9.dll``). Ép chạy CPU (ONNX Runtime, torch-free) né
    hoàn toàn PyTorch/CUDA -> ổn định hơn. Cách chuẩn: ẩn GPU bằng
    ``CUDA_VISIBLE_DEVICES=""`` TRƯỚC khi SDK import torch.

    Args:
        force_cpu: True để ẩn GPU (ép CPU/ONNX); False để giữ nguyên môi trường (cho
            SDK tự chọn CPU/GPU như mặc định).

    Returns:
        Dict biến môi trường cần đặt. Rỗng nếu không ép CPU (không đụng môi trường).
    """
    if not force_cpu:
        return {}
    return {"CUDA_VISIBLE_DEVICES": ""}


@contextmanager
def temporary_env(overrides: dict[str, str]) -> Iterator[None]:
    """Đặt biến môi trường TẠM THỜI rồi KHÔI PHỤC nguyên trạng khi thoát (context mgr).

    [v3.23.190] Cực kỳ quan trọng: đặt ``os.environ`` toàn cục mà KHÔNG khôi phục sẽ ô
    nhiễm mọi tiến trình con sinh ra sau. Cụ thể: đặt ``CUDA_VISIBLE_DEVICES=""`` vĩnh
    viễn khiến subprocess WhisperX (copy ``dict(os.environ)``) KHÔNG THẤY GPU -> WhisperX
    thoát mã 1 dù trước đó chạy tốt. Context manager này đảm bảo env chỉ đổi trong phạm
    vi ``with`` rồi trả lại đúng giá trị cũ (kể cả khi khoá vốn không tồn tại -> xoá).

    Args:
        overrides: Cặp biến môi trường cần đặt tạm thời.

    Yields:
        None. Trong khối ``with`` env đã được áp; ra khỏi khối env khôi phục nguyên trạng.
    """
    saved: dict[str, str | None] = {}
    for key, value in overrides.items():
        saved[key] = os.environ.get(key)  # None nếu vốn không tồn tại
        os.environ[key] = value
    try:
        yield
    finally:
        for key, old_value in saved.items():
            if old_value is None:
                os.environ.pop(key, None)  # vốn không có -> xoá sạch
            else:
                os.environ[key] = old_value  # trả lại giá trị cũ


#: Số lần hỏng liên tiếp thì bỏ hẳn đường GPU. Đủ nhỏ để không lãng phí, đủ lớn để một
#: câu lỗi lẻ (vd văn bản lạ) không làm mất GPU cho cả phiên.
_GPU_FAILURE_LIMIT: Final[int] = 3


def _hidden_console_kwargs() -> dict[str, Any]:
    """Tham số ẩn cửa sổ console trên Windows (nền tảng khác trả rỗng)."""
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NO_WINDOW}
    return {}


def match_voice_name(requested: str, available: list[str]) -> str | None:
    """[v3.23.194] Khớp MỀM tên giọng với danh sách khả dụng (hàm thuần).

    Mỗi chế độ VieNeu (standard/v3turbo) có BỘ GIỌNG KHÁC NHAU (7 vs 10 giọng, tên khác).
    Khi người dùng đổi chế độ, tên giọng đã chọn có thể không tồn tại trong bộ mới ->
    trước đây raise cứng làm hỏng cả phiên TTS. Hàm này thử khớp theo thứ tự ưu tiên:
    1. Khớp CHÍNH XÁC.
    2. Khớp không phân biệt hoa/thường.
    3. Khớp không dấu (bỏ dấu tiếng Việt) — vd 'Ngoc' ~ 'Ngọc Linh' (tiền tố không dấu).
    4. Tiền tố/chứa (không dấu, không hoa/thường).

    Args:
        requested: Tên giọng người dùng yêu cầu.
        available: Danh sách tên giọng SDK chấp nhận.

    Returns:
        Tên giọng khớp trong ``available``, hoặc None nếu không tìm được.
    """
    import unicodedata

    def _fold(text: str) -> str:
        # Bỏ dấu + thường hoá: 'Ngọc Linh' -> 'ngoc linh'.
        normalized = unicodedata.normalize("NFD", text)
        stripped = "".join(c for c in normalized if unicodedata.category(c) != "Mn")
        return stripped.replace("đ", "d").replace("Đ", "d").lower().strip()

    if not requested:
        return None
    if requested in available:
        return requested
    requested_lower = requested.lower().strip()
    for name in available:
        if name.lower().strip() == requested_lower:
            return name
    requested_folded = _fold(requested)
    for name in available:
        if _fold(name) == requested_folded:
            return name
    for name in available:
        folded = _fold(name)
        if folded.startswith(requested_folded) or requested_folded in folded:
            return name
    return None


def describe_voice_error(error: Exception, force_cpu: bool) -> str:
    """Tạo thông báo lỗi tiếng Việt rõ ràng cho lỗi giải quyết giọng VieNeu (hàm thuần).

    Nhận diện lỗi tải DLL PyTorch/cuDNN (WinError 127 / "cudnn" / "dll") để hướng dẫn
    người dùng cách khắc phục thay vì hiện lỗi kỹ thuật khó hiểu.

    Args:
        error: Ngoại lệ gốc bắt được.
        force_cpu: Trạng thái ép CPU hiện tại (để gợi ý phù hợp).

    Returns:
        Chuỗi thông báo lỗi tiếng Việt, có hướng dẫn khắc phục nếu là lỗi DLL/CUDA.
    """
    message = str(error).lower()
    is_dll_error = (
        "winerror 127" in message
        or "cudnn" in message
        or "could not be found" in message
        or (".dll" in message and "load" in message)
    )
    if is_dll_error:
        if not force_cpu:
            return (
                "VieNeu-TTS lỗi tải thư viện CUDA/cuDNN (PyTorch). "
                "Hãy bật tuỳ chọn 'Ép chạy CPU/ONNX' trong cấu hình VieNeu để né lỗi này, "
                "hoặc cài lại đúng bộ PyTorch khớp CUDA của máy.\n"
                f"Chi tiết: {error}"
            )
        return (
            "VieNeu-TTS vẫn lỗi tải thư viện dù đã ép CPU. Có thể bản cài thiếu ONNX "
            "Runtime. Hãy cài lại: pip install vieneu soundfile onnxruntime.\n"
            f"Chi tiết: {error}"
        )
    return f"Không giải quyết được giọng VieNeu-TTS: {error}"


#: [v3.23.373] Ceiling transformers khớp torch mà whisperx_env cài (xem vieneu_gpu_plan).
_COMPATIBLE_TRANSFORMERS_SPEC: Final[str] = "transformers<4.56"


def _gpu_worker_failure_hint(reason: str) -> str:
    """Sinh gợi ý khắc phục CỤ THỂ cho lý do GPU worker chết (hàm thuần, test được).

    Trọng tâm là lỗi xung đột phiên bản torch↔transformers trong ``whisperx_env``: gói
    ``transformers`` (do ``vieneu[legacy]`` kéo vào) quá mới so với ``torch`` mà WhisperX
    ghim, nên import các symbol torch mới (``ScalingType``, ``RMSNorm``…) thất bại.

    Args:
        reason: Chuỗi lý do đã thu thập (thường chứa thông điệp ImportError).

    Returns:
        Chuỗi gợi ý (bắt đầu bằng khoảng trắng để nối vào log), hoặc rỗng nếu không nhận ra.
    """
    lowered = reason.lower()
    is_torch_transformers_mismatch = (
        "scalingtype" in lowered
        or "swizzletype" in lowered
        or ("cannot import name" in lowered and "torch" in lowered)
        or ("rmsnorm" in lowered and "import" in lowered)
    )
    if is_torch_transformers_mismatch:
        return (
            " NGUYÊN NHÂN: 'transformers' trong whisperx_env quá mới so với 'torch' đã "
            "cài (import symbol torch mới bị thiếu). KHẮC PHỤC: mở thư mục ứng dụng và chạy "
            f'"whisperx_env\\Scripts\\python.exe -m pip install \\"{_COMPATIBLE_TRANSFORMERS_SPEC}\\"" '
            "rồi thử lại — hoặc bấm lại nút 'Cài VieNeu cho GPU' sau khi cập nhật ứng dụng."
        )
    return ""


def normalize_mode(mode: str) -> str:
    """Chuẩn hoá tên chế độ engine về giá trị SDK hợp lệ.

    Args:
        mode: Tên chế độ do người dùng/cấu hình cung cấp (có thể hoa/thường lẫn lộn).

    Returns:
        Tên chế độ hợp lệ trong :data:`_VALID_MODES`; mặc định ``"standard"`` nếu không
        khớp (an toàn nhất, chất lượng cao, tương thích rộng).
    """
    normalized = (mode or "").strip().lower()
    return normalized if normalized in _VALID_MODES else "standard"


class VieNeuTtsAdapter(SubtitleTTSPort):
    """TTS tiếng Việt on-device dùng VieNeu-TTS (offline, chất lượng cao).

    Hỗ trợ hai cách chọn giọng:
    * **Giọng dựng sẵn** (preset): đặt ``request.speaker`` = ID giọng preset.
    * **Voice cloning**: đặt ``request.ref_audio_path`` = file WAV 3-5 giây; giọng sẽ
      được nhân bản từ đó (ưu tiên hơn preset nếu cả hai cùng có).

    Model được nạp một lần và cache trong bộ nhớ. Toàn bộ hậu xử lý (làm rõ giọng, chuẩn
    hoá LUFS, giới hạn đỉnh, ghi file) tái dùng module ``audio_mastering`` chung để chất
    lượng đồng nhất với các engine khác.

    Attributes:
        _mode: Chế độ engine VieNeu ("standard"/"turbo"/"v3turbo").
        _emotion: Sắc thái giọng ("natural" hoặc "storytelling").
        _engine: Instance SDK đã nạp (None cho tới khi dùng lần đầu).
    """

    def __init__(
        self, mode: str = "standard", emotion: str = "natural", force_cpu: bool = True
    ) -> None:
        """Khởi tạo adapter (chưa nạp model — lazy load khi generate).

        Args:
            mode: Chế độ engine ("standard" chất lượng cao nhất, "turbo"/"v3turbo" nhanh
                hơn). Giá trị không hợp lệ tự lùi về "standard".
            emotion: Sắc thái giọng — "natural" (hội thoại tự nhiên) hoặc "storytelling"
                (kể chuyện). Không hardcode: truyền vào để dễ cấu hình/test.
            force_cpu: True (mặc định) ép chạy CPU/ONNX, né lỗi PyTorch/cuDNN trên máy có
                CUDA lệch phiên bản (WinError 127). False để SDK tự chọn CPU/GPU.
        """
        self._mode: str = normalize_mode(mode)
        self._emotion: str = emotion if emotion in ("natural", "storytelling") else "natural"
        self._force_cpu: bool = force_cpu
        self._engine: Any | None = None
        self._voice_data_cache: Any | None = None
        # [v3.23.344] Ghi lại nguồn giọng để worker GPU tự giải quyết lại — không truyền
        # embedding qua ranh giới tiến trình (mảng lớn, tuần tự hoá vừa chậm vừa dễ sai).
        self._gpu_voice_preset: str = ""
        self._gpu_reference_wav: str = ""
        #: Đã xác định GPU không dùng được -> khỏi thử lại từng câu.
        self._gpu_worker_disabled: bool = False
        #: Số câu hỏng LIÊN TIẾP ở đường GPU (đặt lại khi có câu thành công).
        self._gpu_failure_streak: int = 0
        #: Tiến trình GPU thường trú (nạp model một lần cho cả phiên).
        self._gpu_server: Any | None = None
        # [v3.23.264] Nhiệt độ tuỳ chọn (None -> giữ mặc định SDK 0.8). Không ép
        # thấp mặc định vì câu ngắn có thể bị model từ chối đọc ở temperature thấp.
        self._temperature: float | None = None

    # ── Interface ─────────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """Kiểm tra SDK VieNeu và soundfile đã cài chưa.

        Returns:
            True nếu import được cả ``vieneu`` và ``soundfile``; False nếu thiếu.
        """
        try:
            import soundfile  # noqa: F401
            import vieneu  # noqa: F401

            return True
        except ImportError:
            return False

    def get_engine_name(self) -> str:
        """Trả tên engine hiển thị trên giao diện."""
        return "VieNeu-TTS (Offline)"

    def list_languages(self) -> list[str]:
        """Danh sách ngôn ngữ hỗ trợ (VieNeu chuyên tiếng Việt, có code-switching Anh)."""
        return ["vi-VN"]

    def list_speakers(self, language: str) -> list[str]:
        """Liệt kê ID giọng dựng sẵn (preset) của VieNeu.

        Nếu model chưa nạp hoặc SDK chưa cài, trả danh sách rỗng (UI sẽ hiển thị lựa
        chọn voice cloning thay thế). Không ném ngoại lệ để UI không sập.

        Args:
            language: Mã ngôn ngữ (bỏ qua — VieNeu chỉ tiếng Việt).

        Returns:
            Danh sách ID giọng preset, hoặc rỗng nếu không lấy được.
        """
        if not self.is_available():
            return []
        try:
            engine = self._get_or_load_engine("auto")
            return [voice_id for _label, voice_id in engine.list_preset_voices()]
        except TTSUnavailableError as exc:
            # [v3.23.256] VieNeu cài thiếu/lỗi dependency (vd sea_g2p ở bản 3.x)
            # -> KHÔNG để lỗi lọt ra thread UI (gây crash). Trả rỗng: UI dùng giọng mặc
            # định, và người dùng vẫn thấy được thông báo cài đặt khi bấm Tổng hợp.
            logger.warning("VieNeu không khả dụng (thiếu/lỗi phụ thuộc): %s", exc)
            return []
        except (RuntimeError, OSError, ValueError) as exc:
            logger.warning("Không lấy được danh sách giọng VieNeu: %s", exc)
            return []

    # ── Generate ──────────────────────────────────────────────────────────────

    def generate(
        self,
        request: TTSRequest,
        output_path: Path,
        progress_cb: TTSProgressCallback | None = None,
        cancel_cb: TTSCancellationCallback | None = None,
    ) -> list[TTSSegmentResult]:
        """Tổng hợp giọng cho toàn bộ phụ đề, master hoá và ghi ra file.

        Args:
            request: Yêu cầu TTS (events, giọng/ref audio, tham số master).
            output_path: Đường dẫn file đích (đuôi sẽ theo ``request.output_format``).
            progress_cb: Callback báo tiến độ (fraction, message).
            cancel_cb: Callback kiểm tra huỷ hợp tác (trả True để dừng).

        Returns:
            Danh sách kết quả từng câu (:class:`TTSSegmentResult`).

        Raises:
            TTSUnavailableError: SDK VieNeu chưa cài.
            TTSGenerationError: Cấu hình giọng không hợp lệ (không có preset lẫn ref).
            TTSCancelledError: Người dùng huỷ giữa chừng.
        """
        if not self.is_available():
            raise TTSUnavailableError(
                "VieNeu-TTS chưa cài. Chạy: pip install vieneu soundfile"
            )

        use_cloning = bool(request.ref_audio_path)
        if use_cloning and not Path(request.ref_audio_path).exists():
            raise TTSGenerationError(
                f"File audio tham chiếu không tồn tại: {request.ref_audio_path}"
            )

        if progress_cb:
            progress_cb(0.0, "Đang tải model VieNeu-TTS (có thể mất vài phút lần đầu)…")

        valid_events = [event for event in request.events if event.text.strip()]
        if not valid_events:
            return []

        last_end = max(event.end_sec for event in valid_events)
        results: list[TTSSegmentResult] = []
        total = len(request.events)

        # [v3.23.191] CÔ LẬP torch SUỐT quá trình nạp engine + inference (giống tinh thần
        # cô lập PaddleOCR, nhưng bao trùm cả inference vì backend VieNeu import torch
        # TRỄ lúc chạy). Nhờ đó VieNeu fallback ONNX/GGUF, KHÔNG chạm torch DLL hỏng —
        # mà vẫn GIỮ torch trong venv cho WhisperX align. Chỉ bật khi ép CPU.
        from contextlib import nullcontext

        from subtitles_extractor.infrastructure.torch_import_blocker import (
            torch_isolation,
        )

        isolation = torch_isolation() if self._force_cpu else nullcontext()
        with isolation:
            engine = self._get_or_load_engine(request.device)
            # [v3.23.260] GIỮ NGUYÊN sample rate GỐC của engine — v3 Turbo 48kHz lưu
            # 48kHz, standard/GGUF 24kHz lưu 24kHz. KHÔNG hạ 48kHz->24kHz (mất dải tần
            # 12-24kHz, giọng kém trong). Đọc SAU khi load engine để biết sr thật;
            # fallback _PIPELINE_SAMPLE_RATE nếu engine không khai báo.
            sample_rate = int(
                getattr(engine, "sample_rate", None) or _PIPELINE_SAMPLE_RATE
            )
            master = np.zeros(
                master_length_samples(
                    last_end, sample_rate, getattr(request, "media_duration_s", None)
                ),
                dtype=np.float32,
            )
            logger.info(
                "VieNeu-TTS: mode=%s emotion=%s events=%d sr=%dHz %s",
                self._mode, self._emotion, len(request.events), sample_rate,
                f"clone={Path(request.ref_audio_path).name}" if use_cloning
                else f"voice={request.speaker or 'mặc định'}",
            )

            # Giải quyết giọng MỘT LẦN (encode ref / lấy preset dict) rồi tái dùng cho
            # mọi câu — tránh encode lại 498 lần (điểm hiệu năng then chốt).
            voice_data = self._resolve_voice_data(engine, request, use_cloning)

            prev_audio_end = 0.0
            for idx, event in enumerate(request.events):
                if cancel_cb and cancel_cb():
                    raise TTSCancelledError("Người dùng đã huỷ TTS.")
                if progress_cb:
                    fraction = 0.05 + 0.90 * (idx / max(1, total))
                    progress_cb(fraction, f"Dòng {idx + 1}/{total}: {event.text[:30]}…")

                # [v3.23.194] Mốc câu KẾ TIẾP để tận dụng gap (giảm tỉ lệ nén -> giọng
                # ít gấp/méo). [v3.23.208] Câu CUỐI dùng thời lượng video làm biên ->
                # nén nhẹ để VỪA biên thay vì bị cắt tại biên video (ca 第1集: cắt 0.42s
                # -> sau fix nén 1.13x trọn vẹn); không biết thời lượng -> None như cũ.
                next_start = (
                    request.events[idx + 1].start_sec
                    if idx + 1 < len(request.events)
                    else getattr(request, "media_duration_s", None)
                )
                result = self._process_event(
                    event=event, idx=idx, master=master, engine=engine,
                    sr=sample_rate, request=request, voice_data=voice_data,
                    next_start_sec=next_start, cancel_cb=cancel_cb,
                    prev_audio_end_sec=prev_audio_end,
                )
                results.append(result)
                # [v3.23.218] Mốc audio câu này kết thúc -> câu sau biết còn bao nhiêu
                # khoảng lặng để "ăn gian đầu" (không bao giờ đè lên tiếng câu trước).
                if not result.was_skipped:
                    prev_audio_end = (
                        result.adjusted_start_sec + result.audio_duration_s
                    )

        # Master hoá + ghi file NGOÀI khối cô lập (scipy/DSP cần sys.modules sạch).
        self._finalize_and_write(master, sample_rate, request, output_path, progress_cb)

        n_ok = sum(1 for result in results if not result.was_skipped)
        logger.info("VieNeu-TTS xong: %d/%d OK.", n_ok, len(results))
        if progress_cb:
            progress_cb(1.0, "Hoàn tất!")
        return results

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_or_load_engine(self, device: str) -> Any:
        """Nạp engine VieNeu một lần rồi cache (lazy singleton, thread-safe).

        [v3.23.200] Bọc ``_ENGINE_LOCK`` (double-checked): nạp danh sách giọng chạy ở
        background thread (UI) có thể trùng thời điểm người dùng bấm Generate (worker
        thread) — không có lock, HAI thread cùng nạp model (~9s + RAM x2) và race trên
        dict cache. Lock đảm bảo model chỉ nạp đúng MỘT lần; torch blocker install/gỡ
        cũng nằm trọn trong lock (không đan xen giữa các thread).

        Args:
            device: "auto" để SDK tự chọn CPU(ONNX)/GPU(PyTorch), hoặc "cpu"/"cuda".

        Returns:
            Instance ``Vieneu`` đã sẵn sàng.

        Raises:
            TTSUnavailableError: Không import được SDK khi cần nạp.
        """
        if self._engine is not None:
            return self._engine
        cache_key = (self._mode, self._emotion)
        cached = _ENGINE_CACHE.get(cache_key)  # đọc nhanh không lock (dict atomic)
        if cached is not None:
            self._engine = cached
            return self._engine
        with _ENGINE_LOCK:
            # Double-check: thread khác có thể vừa nạp xong trong lúc mình chờ lock.
            cached = _ENGINE_CACHE.get(cache_key)
            if cached is not None:
                self._engine = cached
                return self._engine
            logger.info("Đang tải model VieNeu-TTS (mode=%s, lần đầu)…", self._mode)
            # [v3.23.190] Khi ép CPU: (1) chặn import torch (cơ chế v3.23.3) + (2) ẩn GPU
            # bằng CUDA_VISIBLE_DEVICES qua context manager KHÔI PHỤC env — TUYỆT ĐỐI
            # không đặt os.environ vĩnh viễn (sẽ ô nhiễm subprocess WhisperX -> thoát mã
            # 1). Chặn torch để vieneu fallback ONNX (torch-free) -> né WinError 127.
            blocker_installed = False
            if self._force_cpu:
                from subtitles_extractor.infrastructure.torch_import_blocker import (
                    install_torch_import_blocker,
                    is_torch_import_blocked,
                )
                if not is_torch_import_blocked():
                    install_torch_import_blocker()
                    blocker_installed = True
                logger.info("VieNeu-TTS: ép chạy CPU/ONNX (chặn torch, né PyTorch/cuDNN).")
            try:
                with temporary_env(resolve_device_env(self._force_cpu)):
                    from vieneu import Vieneu

                    engine = self._construct_engine(Vieneu)
            except ImportError as exc:
                # [v3.23.256] Phân biệt 2 loại ImportError:
                # (1) Chưa cài vieneu -> hướng dẫn cài.
                # (2) vieneu ĐÃ cài nhưng THIẾU dependency phụ (vd sea_g2p.normalizer của
                #     bản 3.x, hoặc onnxruntime) -> báo đúng module thiếu + cách sửa,
                #     tránh thông báo sai "chưa cài vieneu" gây hiểu lầm.
                missing = getattr(exc, "name", "") or ""
                if missing and not missing.startswith("vieneu"):
                    raise TTSUnavailableError(
                        f"VieNeu-TTS đã cài nhưng thiếu thư viện phụ thuộc "
                        f"'{missing}'. Bản VieNeu 3.x cần các gói mới (vd sea-g2p, "
                        f"onnxruntime). Hãy cập nhật đầy đủ:\n"
                        f"    pip install -U vieneu sea-g2p onnxruntime soundfile\n"
                        f"Nếu vẫn lỗi, gỡ rồi cài lại: "
                        f"pip uninstall -y vieneu sea-g2p && pip install vieneu"
                    ) from exc
                raise TTSUnavailableError(
                    "VieNeu-TTS chưa cài. Chạy: pip install vieneu soundfile"
                ) from exc
            except OSError as exc:
                # WinError 127 (torch/cuDNN DLL) lọt ra dù đã chặn -> báo rõ ràng.
                raise TTSUnavailableError(
                    describe_voice_error(exc, force_cpu=self._force_cpu)
                ) from exc
            finally:
                # Gỡ blocker NGAY sau khi khởi tạo xong (theo bài học v3.23.3): tránh
                # ``sys.modules['torch']=None`` gây AttributeError cho scipy/DSP về sau.
                if blocker_installed:
                    from subtitles_extractor.infrastructure.torch_import_blocker import (
                        uninstall_torch_import_blocker,
                    )
                    uninstall_torch_import_blocker()
            _ENGINE_CACHE[cache_key] = engine
            self._engine = engine
            logger.info("VieNeu-TTS model sẵn sàng.")
        return self._engine

    def _process_event(
        self, *, event: Any, idx: int, master: np.ndarray, engine: Any, sr: int,
        request: TTSRequest, voice_data: Any, next_start_sec: float | None = None,
        cancel_cb: TTSCancellationCallback | None,
        prev_audio_end_sec: float = 0.0,
    ) -> TTSSegmentResult:
        """Tổng hợp một câu, đặt vào master đúng mốc thời gian.

        Args:
            event: Sự kiện phụ đề (có start_sec/end_sec/text).
            idx: Chỉ số câu (0-based).
            master: Mảng master để cộng audio vào (sửa in-place).
            engine: Instance VieNeu đã nạp.
            sr: Tần số lấy mẫu pipeline.
            request: Yêu cầu TTS gốc.
            voice_data: Dict giọng đã giải quyết (dùng chung mọi câu).
            cancel_cb: Callback huỷ hợp tác.
            prev_audio_end_sec: Thời điểm audio câu TRƯỚC kết thúc thật (giây) — dùng
                để "ăn gian đầu" vào khoảng lặng còn trống mà không đè lên tiếng.

        Returns:
            Kết quả câu (:class:`TTSSegmentResult`), ``was_skipped=True`` nếu bỏ/lỗi.
        """
        text, is_dialog = _preprocess_tts_text(
            event.text, request.clean_tags, _skip_from_request(request),
            strip_speaker_tag=True,
        )
        start_sec: float = event.start_sec
        end_sec: float = event.end_sec
        available = end_sec - start_sec

        # [v3.23.202] Cửa sổ ngắn KHÔNG phải lý do bỏ khi "Cho phép chồng tiếng" bật:
        # audio tràn tự nhiên (fit_limit_samples=None) -> vẫn đọc trọn câu ngắn ("Đi!",
        # "Sao?"...) thay vì MẤT thoại. Chỉ bỏ khi text rỗng, hoặc khung quá ngắn VÀ
        # không cho chồng (cắt khít sẽ ra mẩu audio vô nghĩa).
        window_too_short = (
            available < request.gap_threshold_s
            and not getattr(request, "allow_audio_overlap", True)
        )
        # [v3.23.211] Văn bản KHÔNG có ký tự phát âm được (chỉ dấu câu / ký hiệu /
        # ô vuông "□" rác từ OCR — đo thực: 34/896 dòng phụ đề gốc) -> model sinh im
        # lặng -> retry 10 lần VÔ ÍCH (~30s + 10 lần gọi API tốn quota) rồi báo sai
        # nguyên nhân. Bỏ NGAY với thông báo đúng (Edge đã làm; đồng bộ parity).
        unspeakable = bool(text) and not has_speakable_content(text)
        if not text or unspeakable or window_too_short:
            return TTSSegmentResult(
                event_index=idx, start_sec=start_sec, end_sec=end_sec,
                text=text, was_skipped=True,
                error_msg=(
                    "Không có nội dung đọc được (chỉ dấu câu/ký hiệu)."
                    if unspeakable else "Cửa sổ quá ngắn / text rỗng."
                ),
                adjusted_start_sec=start_sec, adjusted_end_sec=end_sec,
            )

        audio = self._synthesize_with_retry(
            engine=engine, text=text, request=request, voice_data=voice_data,
            cancel_cb=cancel_cb,
        )
        if audio is None:
            return TTSSegmentResult(
                event_index=idx, start_sec=start_sec, end_sec=end_sec,
                text=text, was_skipped=True,
                error_msg=f"VieNeu-TTS thất bại sau {request.retry_count} lần.",
                adjusted_start_sec=start_sec, adjusted_end_sec=end_sec,
            )

        # [v3.23.204] Cắt im lặng biên NGAY sau tổng hợp (trước pause/stretch): VieNeu
        # sinh im lặng đầu câu dài (đo thực: max 1.42s) làm tiếng TRỄ so với phụ đề và
        # đẩy phần tiếng tràn sang câu sau. Pause hội thoại có kiểm soát chèn SAU trim.
        audio = trim_edge_silence(audio, sr, adaptive=True)  # [v3.23.241] ngưỡng tự dò

        # [v3.23.217] Khoảng nghỉ hội thoại KHÔNG bị nén và KHÔNG tính vào phần cần nén.
        # Bug cũ: pause được chèn TRƯỚC khi tính stretch -> (1) bị nén theo (300ms mặc
        # định còn 171-231ms — nhịp hội thoại bị bóp 23-43%%), (2) tính vào độ dài audio
        # nên giọng bị nén VÌ pause (sai bản chất — pause là khoảng LẶNG có chủ đích).
        # Nay: tính tỉ lệ trên riêng GIỌNG, khung dành cho giọng đã trừ pause; pause được
        # chèn SAU khi nén, giữ nguyên độ dài người dùng đặt.
        pause_s = (
            request.dialog_pause_ms / 1000.0
            if (is_dialog and request.dialog_pause_ms > 0)
            else 0.0
        )
        voice_duration = len(audio) / sr

        max_speed = float(getattr(request, "max_speed", 3.0)) or 3.0
        # [v3.23.218] "Ăn gian đầu": nếu câu trước đã đọc XONG sớm, bắt đầu câu này sớm
        # hơn một chút để có khung rộng hơn -> nén NHẸ hơn -> giọng rõ hơn. Chỉ ăn vào
        # khoảng lặng THẬT (không đè tiếng câu trước) và KHÔNG dời câu sau (không domino).
        # [v3.23.219] CHỈ ăn gian ĐÚNG MỨC CẦN: câu đã vừa khung ở tốc độ cơ bản thì giữ
        # NGUYÊN mốc phụ đề (bug v218: dời sớm 250ms cả 79/95 câu không cần -> toàn bộ
        # tiếng lệch trước khẩu hình -> nghe "không đồng bộ").
        base_available = effective_available_seconds(start_sec, end_sec, next_start_sec)
        needed_window = voice_duration / max(1.0, float(request.base_speed)) + pause_s
        lead_s = lead_in_seconds(
            start_sec,
            prev_audio_end_sec,
            max_lead_s=float(getattr(request, "lead_in_s", _MAX_LEAD_IN_S)),
            needed_lead_s=max(0.0, needed_window - base_available),
        )
        play_start_sec = start_sec - lead_s
        effective_available = base_available + lead_s
        available_for_voice = max(0.05, effective_available - pause_s)
        stretch_ratio = total_speed_ratio(
            voice_duration, available_for_voice, request.base_speed, max_speed
        )
        speed_used = stretch_ratio
        if stretch_ratio > 1.0:
            from subtitles_extractor.infrastructure.tts.time_stretch import (
                vocal_time_stretch,
            )

            audio = vocal_time_stretch(audio, sr, stretch_ratio)

        if pause_s > 0.0:
            pause_samples = int(pause_s * sr)
            audio = np.concatenate((np.zeros(pause_samples, dtype=np.float32), audio))

        # [v3.23.198] Đánh giá "Bỏ qua lấn" SAU khi đã nén và so với KHUNG HIỆU DỤNG.
        # Bug cũ: check TRƯỚC nén với khung gốc -> câu audio 4s/khung 2s bị BỎ HẲN (mất
        # thoại) dù sau nén 2x chỉ còn 2s vừa khít, không hề lấn. Chỉ bỏ khi ĐÃ vắt hết
        # cách (nén + gap) mà phần lấn còn lại vẫn vượt ngưỡng người dùng đặt.
        residual_overlap_s = (len(audio) / sr) - effective_available
        if (
            request.skip_overlap_ms > 0
            and residual_overlap_s > (request.skip_overlap_ms / 1000.0)
        ):
            return TTSSegmentResult(
                event_index=idx, start_sec=start_sec, end_sec=end_sec,
                text=text, was_skipped=True,
                error_msg=(
                    f"Lấn còn lại {residual_overlap_s * 1000:.0f}ms sau nén > "
                    f"ngưỡng bỏ qua {request.skip_overlap_ms}ms"
                ),
                adjusted_start_sec=start_sec, adjusted_end_sec=end_sec,
            )

        # [v3.23.197] Tôn trọng "Cho phép chồng tiếng": bật -> audio TRÀN tự nhiên vào
        # master (mảng cộng dồn hỗ trợ chồng sẵn), KHÔNG cắt -> giữ TRỌN nội dung thoại.
        # Bug cũ: chỉ đọc max_overlap_ms (=0) -> cắt đúng khít khung dù đã bật chồng.
        was_truncated = False
        max_samples = fit_limit_samples(
            effective_available,
            request.max_overlap_ms,
            getattr(request, "allow_audio_overlap", True),
            sr,
        )
        if max_samples is not None and len(audio) > max_samples:
            audio = audio[:max_samples]
            was_truncated = True

        # [v3.23.218] Đặt audio tại mốc ĐỌC SỚM (đã ăn gian vào khoảng lặng trống).
        start_sample = max(0, int(play_start_sec * sr))
        end_sample = min(start_sample + len(audio), len(master))
        if start_sample < len(master):
            placed = end_sample - start_sample
            master[start_sample:end_sample] += audio[:placed]
            if placed < len(audio):
                was_truncated = True  # chạm biên master (hiếm — đệm đuôi 5s)

        # [v3.23.203] Báo cáo CHỒNG THẬT (đồng bộ ngữ nghĩa Edge): phần audio cuối
        # cùng vượt khung hiệu dụng -> UI/debug đếm đúng "Có chồng tiếng (lấn)" (trước
        # đây VieNeu không set -> luôn báo 0 dù thực tế có, người dùng mất thông tin
        # để tự điều chỉnh max_speed / biên tập phụ đề).
        final_overlap_s = max(0.0, (len(audio) / sr) - effective_available)
        return TTSSegmentResult(
            event_index=idx, start_sec=start_sec, end_sec=end_sec,
            text=text, audio_duration_s=len(audio) / sr,
            speed_used=speed_used, was_truncated=was_truncated,
            overlap_s=final_overlap_s,
            # [v3.23.193] adjusted = mốc THẬT của tiếng -> use case nhận diện "có điều
            # chỉnh" -> XUẤT SRT đồng bộ TTS.
            # [v3.23.218] Mốc thật nay có thể sớm hơn phụ đề (ăn gian đầu) -> SRT đồng bộ
            # phải phản ánh đúng để phụ đề khớp tiếng.
            adjusted_start_sec=play_start_sec,
            adjusted_end_sec=max(end_sec, play_start_sec + len(audio) / sr),
        )

    def _synthesize_with_retry(
        self, *, engine: Any, text: str, request: TTSRequest, voice_data: Any,
        cancel_cb: TTSCancellationCallback | None,
    ) -> np.ndarray | None:
        """Gọi VieNeu ``infer`` có retry, trả audio mono 24kHz hoặc None nếu thất bại.

        Args:
            engine: Instance VieNeu đã nạp.
            text: Văn bản cần đọc (đã tiền xử lý).
            request: Yêu cầu TTS (chứa retry_count).
            voice_data: Dict giọng đã giải quyết (từ :meth:`_resolve_voice_data`).
            cancel_cb: Callback huỷ hợp tác.

        Returns:
            Mảng float32 mono ở 24kHz, hoặc None nếu mọi lần thử đều lỗi/rỗng.

        Raises:
            TTSCancelledError: Người dùng huỷ trong lúc thử/chờ.
        """
        retry_count = max(1, request.retry_count)
        # [v3.23.261] Đo "ngân dài" theo ÂM TIẾT (không phải ký tự). Ký tự đếm cả dấu câu
        # và ký tự không phát âm ("Chú..." = 6 ký tự, 1 âm tiết) -> lưới cũ theo ký tự
        # (R2=0.07) báo động sai. Biên dưới theo âm tiết R2=0.98 -> bội số chuẩn xác.
        syllable_count = dem_am_tiet(text.strip())
        # [v3.23.260] sr thật của engine (v3 Turbo 48kHz, standard 24kHz) để tính đúng
        # thời lượng — pipeline nay giữ nguyên sr engine thay vì ép 24kHz.
        engine_sr = int(getattr(engine, "sample_rate", None) or _PIPELINE_SAMPLE_RATE)
        # [v3.23.227] Phân biệt HAI loại retry (bug: trước đây gộp làm một):
        #   * Retry do LỖI HỆ THỐNG (exception): cần backoff tăng dần — hợp lý.
        #   * Retry do CHẤT LƯỢNG AUDIO (im lặng / ngân dài): VieNeu chạy OFFLINE, không
        #     có rate limit, không có lỗi mạng. Model lấy mẫu NGẪU NHIÊN, nên lần thử sau
        #     độc lập với lần trước -> CHỜ là lãng phí thuần tuý. Đo trên log thật: 9 lần
        #     retry chất lượng trong một phiên, mỗi lần chờ 1s/2s/3s… vô ích.
        wait_before_next = False
        # [v3.23.221] Ứng viên "hợp lệ nhưng DÀI BẤT THƯỜNG": không vứt đi (sẽ mất thoại),
        # giữ lại và thử tiếp; hết lượt thì trả bản NGẮN NHẤT đã thu được.
        overlong_best: np.ndarray | None = None
        for attempt in range(retry_count):
            if cancel_cb and cancel_cb():
                raise TTSCancelledError("Huỷ trong khi tổng hợp VieNeu-TTS.")
            try:
                raw_audio = self._infer_once(engine, text, voice_data)
                audio = self._to_mono_pipeline_rate(
                    raw_audio, getattr(engine, "sample_rate", None)
                )
                # [v3.23.205] Audio toàn IM LẶNG (model sinh nhưng không có tiếng) coi
                # như thất bại -> retry, giống audio rỗng (ca thực tế: câu #58 "Ý chú
                # là" 2.2s im lặng lọt lưới size>0 -> mất thoại mà báo OK).
                if audio.size > 0 and not is_effectively_silent(audio):
                    duration_s = audio.size / engine_sr
                    # [v3.23.221] Dạng lỗi NGƯỢC LẠI của im lặng: model "ngân dài" một âm
                    # tiết (ca thực: "Ừm." 3 ký tự -> 2.55s, +7.7σ) -> bị nén kịch trần
                    # 2.0x (tan formant) và đè lên câu sau. Lấy mẫu lại thường cho bản
                    # bình thường; giữ bản ngắn nhất nếu mọi lần đều dài.
                    # [v3.23.261] Đo theo BIÊN DƯỚI THEO ÂM TIẾT (R²=0.98) với hằng số
                    # engine VieNeu, thay lưới cũ theo ký tự (R²=0.07 -> báo động sai với
                    # câu nhiều dấu câu như "Chú...", "Ồ,"). "Ừm." 1 âm -> ngưỡng đúng.
                    if not is_abnormally_long_vs_floor(
                        duration_s,
                        syllable_count,
                        min_base_s=VIENEU_MIN_BASE_S,
                        min_per_syllable_s=VIENEU_MIN_PER_SYLLABLE_S,
                    ):
                        return audio
                    overlong_best = shorter_take(overlong_best, audio)
                    # Lấy mẫu lại NGAY: model offline + ngẫu nhiên, chờ vô ích.
                    wait_before_next = False
                    logger.warning(
                        "VieNeu-TTS lần %d/%d: audio DÀI BẤT THƯỜNG %.2fs cho %d âm tiết "
                        "('%s…') — nghi model ngân dài, lấy mẫu lại.",
                        attempt + 1, retry_count, duration_s, syllable_count, text[:25],
                    )
                else:
                    # Im lặng cũng là lỗi CHẤT LƯỢNG -> không chờ.
                    wait_before_next = False
                    logger.warning(
                        "VieNeu-TTS lần %d: audio %s cho '%s…'",
                        attempt + 1,
                        "rỗng" if audio.size == 0 else "toàn im lặng",
                        text[:25],
                    )
            except (RuntimeError, OSError, ValueError) as exc:
                # LỖI THẬT (OOM, DLL, I/O) -> backoff tăng dần có ý nghĩa.
                wait_before_next = True
                logger.warning(
                    "VieNeu-TTS lần %d/%d thất bại cho '%s…': %s",
                    attempt + 1, retry_count, text[:25], exc,
                )
            if wait_before_next and attempt < retry_count - 1:
                self._sleep_with_cancel(
                    request.retry_delay_s * (attempt + 1), cancel_cb
                )
        if overlong_best is not None:
            # Thà đọc lê thê còn hơn MẤT THOẠI: trả bản ngắn nhất, để pipeline nén/cắt.
            logger.warning(
                "VieNeu-TTS: mọi lần thử đều dài bất thường cho '%s…' — dùng bản ngắn "
                "nhất (%.2fs).",
                text[:25], overlong_best.size / engine_sr,
            )
            return overlong_best
        return None

    def _resolve_voice_data(
        self, engine: Any, request: TTSRequest, use_cloning: bool
    ) -> Any:
        """Giải quyết 'voice data' (dict) MỘT LẦN cho cả phiên tổng hợp.

        SDK VieNeu yêu cầu tham số ``voice=`` là một DICT giọng, không phải string ID:
        * Voice cloning: ``engine.encode_reference(path)`` -> dict mã hoá giọng tham chiếu.
        * Giọng preset: ``engine.get_preset_voice(voice_id)`` -> dict giọng dựng sẵn.
        * Mặc định (không chọn gì): lấy giọng preset ĐẦU TIÊN làm mặc định an toàn.

        Kết quả được cache để KHÔNG phải encode/lấy lại ở mỗi câu (498 câu × encode sẽ
        rất chậm) — đây là điểm hiệu năng then chốt.

        Args:
            engine: Instance VieNeu đã nạp.
            request: Yêu cầu TTS (chứa speaker/ref_audio_path).
            use_cloning: True nếu dùng voice cloning từ ref audio.

        Returns:
            Dict 'voice data' để truyền vào ``engine.infer(voice=...)``.

        Raises:
            TTSGenerationError: Không giải quyết được giọng (không encode được ref, hoặc
                không có preset nào khả dụng).
        """
        if self._voice_data_cache is not None:
            return self._voice_data_cache

        # Ghi nguồn giọng cho worker GPU (nó sẽ tự encode lại trong tiến trình của nó).
        self._gpu_reference_wav = (
            str(request.ref_audio_path) if use_cloning and request.ref_audio_path else ""
        )
        self._gpu_voice_preset = str(getattr(request, "voice_name", "") or "")

        try:
            if use_cloning:
                encoded = engine.encode_reference(request.ref_audio_path)
                # [v3.23.257] v3 Turbo ``encode_reference`` trả TUPLE (speaker_emb,
                # ref_codes), nhưng ``infer(voice=)`` chỉ nhận str|dict -> truyền tuple
                # trực tiếp sẽ bị BỎ QUA ÂM THẦM (rơi về giọng mặc định, mất cloning). Bọc
                # thành dict đúng khoá mà SDK mong đợi. Bản cũ trả sẵn dict thì giữ nguyên.
                if isinstance(encoded, tuple) and len(encoded) >= 2:
                    voice_data = {"speaker_emb": encoded[0], "codes": encoded[1]}
                else:
                    voice_data = encoded
            elif request.speaker:
                # [v3.23.194] Mỗi chế độ (standard/v3turbo) có BỘ GIỌNG KHÁC NHAU. Khớp
                # MỀM tên giọng (không dấu/hoa-thường/tiền tố); không thấy -> fallback
                # giọng ĐẦU TIÊN + cảnh báo (không raise phá cả phiên TTS như trước).
                voice_data = self._preset_voice_with_soft_match(
                    engine, request.speaker
                )
            else:
                presets = engine.list_preset_voices()
                if not presets:
                    raise TTSGenerationError(
                        "VieNeu-TTS không có giọng preset nào và không có audio tham "
                        "chiếu. Hãy chọn giọng hoặc cung cấp file nhân bản giọng."
                    )
                default_voice_id = presets[0][1]
                logger.info("VieNeu-TTS dùng giọng mặc định: %s", presets[0][0])
                voice_data = engine.get_preset_voice(default_voice_id)
        except (RuntimeError, OSError, ValueError, KeyError) as exc:
            raise TTSGenerationError(
                describe_voice_error(exc, force_cpu=self._force_cpu)
            ) from exc

        self._voice_data_cache = voice_data
        return voice_data

    def _preset_voice_with_soft_match(self, engine: Any, requested: str) -> Any:
        """Lấy voice dict theo tên có KHỚP MỀM; không thấy -> fallback giọng đầu tiên.

        Args:
            engine: Instance VieNeu đã nạp.
            requested: Tên giọng người dùng yêu cầu (có thể thuộc chế độ khác).

        Returns:
            Dict giọng từ ``get_preset_voice`` (giọng khớp, hoặc giọng đầu tiên nếu
            không tìm được — kèm cảnh báo log để người dùng biết).

        Raises:
            ValueError: Nếu engine không có giọng preset nào (lan lên resolver xử lý).
        """
        presets = engine.list_preset_voices()
        available_names = [voice_id for _label, voice_id in presets]
        matched = match_voice_name(requested, available_names)
        if matched is not None:
            if matched != requested:
                logger.info(
                    "VieNeu-TTS: giọng '%s' khớp mềm với '%s'.", requested, matched
                )
            return engine.get_preset_voice(matched)
        if not presets:
            raise ValueError("VieNeu-TTS không có giọng preset nào.")
        fallback_label, fallback_id = presets[0]
        logger.warning(
            "VieNeu-TTS: giọng '%s' không có trong chế độ %s (khả dụng: %s). "
            "Dùng giọng mặc định '%s'.",
            requested, self._mode, available_names, fallback_label,
        )
        return engine.get_preset_voice(fallback_id)

    def _construct_engine(self, vieneu_cls: Any) -> Any:
        """[v3.23.254] Khởi tạo ``Vieneu`` chỉ truyền tham số constructor THẬT SỰ nhận.

        API VieNeu thay đổi qua các bản. Bản app từng dùng nhận ``Vieneu(mode=...,
        emotion=...)``. Các bản mới hơn CÓ THỂ đổi hoặc bỏ tham số (vd thay ``emotion``
        bằng ``style``). Gọi cứng ``emotion=`` sẽ ném ``TypeError`` -> sập engine nếu bản
        cài trên máy người dùng không còn nhận tham số đó.

        Giải pháp KHÔNG phụ thuộc vào việc đoán đúng API bản nào: dò chữ ký constructor
        (như đã làm an toàn với ``infer`` ở :meth:`_supported_infer_params`) và chỉ truyền
        tham số mà chữ ký thật sự khai báo:

        * ``mode``   -> truyền nếu constructor nhận.
        * ``emotion``-> truyền nếu nhận (API cũ).
        * ``style``  -> nếu KHÔNG có ``emotion`` nhưng CÓ ``style``, ánh xạ giá trị
          emotion của app sang tên style tương ứng (xem :data:`_EMOTION_TO_STYLE`).

        Nếu không nội soi được chữ ký (vd C-extension), thử lần lượt ``{mode}`` rồi ``{}``
        (mặc định hoàn toàn) — luôn khởi tạo được engine thay vì sập.

        Args:
            vieneu_cls: Lớp ``Vieneu`` vừa import.

        Returns:
            Instance ``Vieneu`` đã khởi tạo.
        """
        try:
            params = frozenset(inspect.signature(vieneu_cls).parameters)
        except (AttributeError, TypeError, ValueError):
            params = frozenset()

        kwargs: dict[str, Any] = {}
        if "mode" in params:
            kwargs["mode"] = self._mode
        if "emotion" in params:
            kwargs["emotion"] = self._emotion
        elif "style" in params:
            # API mới có thể dùng ``style`` thay ``emotion``. Ánh xạ giá trị của app sang
            # tên style; nếu tên không khớp giá trị SDK chấp nhận, khối except bên dưới sẽ
            # lùi về khởi tạo mặc định thay vì sập.
            kwargs["style"] = _EMOTION_TO_STYLE.get(self._emotion, "tu_nhien")
        # [v3.23.255] VieNeu 3.x nhận ``precision`` ("int8" mặc định — nhanh ~1.6×,
        # nhỏ ~4×, "quality preserved"; "fp32" — chất lượng tối đa). Chỉ truyền khi app
        # có đặt (``self._precision``) VÀ constructor nhận; None -> SDK tự dùng int8.
        _precision = getattr(self, "_precision", None)
        if _precision and "precision" in params:
            kwargs["precision"] = _precision

        try:
            return vieneu_cls(**kwargs)
        except TypeError:
            # Chữ ký không nội soi được, hoặc giá trị style không hợp lệ -> thử tối giản
            # dần: chỉ mode, rồi mặc định hoàn toàn. Luôn có engine thay vì sập cả phiên.
            for fallback in ({"mode": self._mode} if self._mode else {}, {}):
                try:
                    return vieneu_cls(**fallback)
                except TypeError:
                    continue
            raise

    @staticmethod
    def _supported_infer_params(engine: Any) -> frozenset[str]:
        """[v3.23.227] Tên các tham số mà ``engine.infer`` của SDK thật sự nhận.

        Dò bằng introspection thay vì GIẢ ĐỊNH tên tham số: bản SDK khác nhau có thể đặt
        tên khác (hoặc không hỗ trợ). Nhờ vậy việc truyền thêm tuỳ chọn là AN TOÀN TUYỆT
        ĐỐI — SDK nào không có thì đơn giản là không truyền, hành vi không đổi.

        Args:
            engine: Instance VieNeu.

        Returns:
            Tập tên tham số; rỗng nếu không nội soi được chữ ký.
        """
        try:
            return frozenset(inspect.signature(engine.infer).parameters)
        except (AttributeError, TypeError, ValueError):
            # Built-in/C-extension không có chữ ký, hoặc SDK không phơi ``infer``.
            # Không được để lỗi thoát ra: cả phiên tổng hợp giọng sẽ sập chỉ vì một tính
            # năng PHỤ (truyền trần thời lượng).
            return frozenset()

    def _infer_once(self, engine: Any, text: str, voice_data: Any) -> Any:
        """Gọi một lần ``engine.infer`` với voice data dict đã giải quyết.

        [v3.23.227] Nếu SDK hỗ trợ trần thời lượng, truyền vào để CHẶN NGAY ca model ngân
        dài thảm hoạ — đo trên log thật: câu 12 ký tự khiến VieNeu sinh **32.10 giây**
        audio, đốt ~30s CPU trước khi lưới hậu kiểm phát hiện. Trần đặt rất rộng (6x kỳ
        vọng) nên câu hợp lệ (~1.0-1.2x) không bao giờ chạm tới.

        Args:
            engine: Instance VieNeu.
            text: Văn bản cần đọc.
            voice_data: Dict giọng (từ :meth:`_resolve_voice_data`).

        Returns:
            Kết quả thô từ SDK (numpy array hoặc tuple ``(audio, sr)``).
        """
        kwargs: dict[str, Any] = {"text": text, "voice": voice_data}
        supported = self._supported_infer_params(engine)
        # [v3.23.262] Trần theo ÂM TIẾT (nhất quán lưới hallucination) — câu cùng
        # số âm có cùng trần dù khác số ký tự (dấu câu).
        cap_s = generation_time_cap_seconds(
            len(text.strip()),
            syllable_count=dem_am_tiet(text.strip()),
            min_base_s=VIENEU_MIN_BASE_S,
            min_per_syllable_s=VIENEU_MIN_PER_SYLLABLE_S,
        )
        # Trần thời lượng theo GIÂY (API cũ, nếu có).
        for name in ("max_duration_s", "max_duration", "max_length_s", "max_seconds"):
            if name in supported:
                kwargs[name] = cap_s
                break
        else:
            # [v3.23.257] v3 Turbo KHÔNG có trần theo giây — trần thật là
            # ``max_new_frames`` (frame audio, 100 frame/s). Trước đây app chỉ tìm
            # 'max_duration_s'... đều KHÔNG khớp -> cơ chế chống "ngân dài" KHÔNG chạy
            # với v3 Turbo. Nay quy đổi cap giây -> frames. +50 frame đệm để câu hợp lệ
            # không chạm trần (chỉ chặn thảm hoạ).
            if "max_new_frames" in supported:
                kwargs["max_new_frames"] = int(cap_s * _V3TURBO_FRAMES_PER_SECOND) + 50
        # [v3.23.255] VieNeu 3.x nhận ``style`` per-call trong ``infer`` ("tu_nhien"/
        # "tin_tuc"/"doc_truyen"). Đây là nơi ánh xạ sắc thái đọc của app khi infer nhận
        # style. Introspection -> an toàn với bản cũ (không có 'style' thì không truyền).
        if "style" in supported:
            kwargs["style"] = _EMOTION_TO_STYLE.get(self._emotion, "tu_nhien")
        # [v3.23.264] CHỐNG NGÂN DÀI từ API. Đọc source VieNeu: model sinh audio
        # TỰ HỒI QUY từng frame, dừng khi (1) tự sinh EOS hoặc (2) chạm max_new_frames.
        # Câu NGẮN model hay "phân vân" không sinh EOS đúng lúc -> ngân (lặp frame
        # nguyên âm). ``repetition_penalty`` phạt token audio đã xuất hiện (logit/penalty)
        # -> chống lặp frame = chống ngân, mà KHÔNG làm câu ngắn bị từ chối (khác
        # temperature). Nâng nhẹ 1.2 -> 1.3.
        if "repetition_penalty" in supported:
            kwargs["repetition_penalty"] = _VIENEU_REPETITION_PENALTY
        # ``temperature`` tuỳ chọn: None -> GIỮ mặc định SDK (0.8). KHÔNG ép thấp vô
        # điều kiện — bài học v249: temperature quá thấp khiến câu ngắn ("Ơ.", "Hả?")
        # bị model từ chối đọc (trả rỗng). Chỉ truyền khi người dùng chủ động đặt.
        temperature = getattr(self, "_temperature", None)
        if temperature is not None and "temperature" in supported:
            kwargs["temperature"] = float(temperature)
        # [v3.23.344] MẢNH GHÉP CÒN THIẾU. v3.23.343 thêm nút cài `vieneu` vào môi
        # trường riêng, nhưng adapter vẫn nạp VieNeu TRONG tiến trình chính — nơi torch
        # cố ý không được đóng gói. Nên VieNeu gặp ImportError rồi ÂM THẦM lùi về
        # ONNX/CPU: cài đủ gói mà vẫn không dùng được GPU.
        # Nay khi bật GPU, đẩy lời gọi sang worker chạy bằng Python của môi trường riêng.
        if not self._force_cpu:
            result = self._infer_via_gpu_worker(text, kwargs)
            if result is not None:
                return result
        return engine.infer(**kwargs)

    def _infer_via_gpu_worker(
        self, text: str, kwargs: dict[str, Any]
    ) -> Any | None:
        """Tổng hợp một câu bằng GPU qua tiến trình THƯỜNG TRÚ ở môi trường riêng.

        [v3.23.349] SỬA LỖI "GPU chậm hơn CPU". Bản v3.23.344 mở một tiến trình MỚI cho
        MỖI CÂU: mỗi lần khởi động Python + nạp torch + nạp model VieNeu mất ~15 giây,
        chỉ để tổng hợp một câu 0,1 giây. Với 55 câu là ~14 phút — chậm hơn hẳn CPU (27
        giây) — và nháy 55 cửa sổ console.

        Nay giữ MỘT tiến trình sống suốt phiên: model nạp một lần, mỗi câu chỉ tốn đúng
        thời gian suy luận.

        Args:
            text: Câu cần đọc.
            kwargs: Tham số đã dựng cho ``infer``.

        Returns:
            Mảng mẫu, hoặc ``None`` để tầng trên lùi về ONNX/CPU.
        """
        import json
        import tempfile

        if self._gpu_worker_disabled:
            return None
        process = self._ensure_gpu_server()
        if process is None:
            return None

        with tempfile.TemporaryDirectory(prefix="subext_vieneu_gpu_") as temp_dir:
            output = Path(temp_dir) / "out.wav"
            request = {
                "text": text,
                "output": str(output),
                **{
                    name: kwargs[name]
                    for name in (
                        "max_new_frames", "style", "repetition_penalty", "temperature",
                    )
                    if name in kwargs
                },
            }
            try:
                assert process.stdin is not None and process.stdout is not None
                process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
                process.stdin.flush()
                reply_line = process.stdout.readline()
            except (OSError, ValueError, AssertionError) as exc:
                logger.warning("Mất liên lạc với tiến trình GPU: %s", exc)
                self._shutdown_gpu_server()
                return None

            if not reply_line:
                logger.warning("Tiến trình GPU đã thoát bất ngờ.")
                self._shutdown_gpu_server()
                return None
            try:
                reply = json.loads(reply_line)
            except ValueError:
                logger.warning("Tiến trình GPU trả dữ liệu không hợp lệ.")
                return None

            if not reply.get("ok"):
                reason = str(reply.get("error") or "không rõ")
                self._gpu_failure_streak += 1
                if self._gpu_failure_streak >= _GPU_FAILURE_LIMIT:
                    self._disable_gpu_worker(
                        f"{_GPU_FAILURE_LIMIT} câu liên tiếp đều hỏng — {reason}"
                    )
                else:
                    logger.warning(
                        "VieNeu GPU hỏng (%d/%d): %s",
                        self._gpu_failure_streak, _GPU_FAILURE_LIMIT, reason,
                    )
                return None

            if not output.is_file():
                logger.warning("Tiến trình GPU báo xong nhưng không có tệp âm thanh.")
                return None
            try:
                import soundfile as sf

                samples, _sr = sf.read(str(output), dtype="float32")
            except (OSError, RuntimeError, ImportError) as exc:
                logger.warning("Không đọc được âm thanh từ tiến trình GPU: %s", exc)
                return None
            self._gpu_failure_streak = 0
            return samples

    def _ensure_gpu_server(self) -> Any | None:
        """Khởi động tiến trình GPU thường trú nếu chưa có.

        Returns:
            Tiến trình đang sống, hoặc ``None`` nếu không dùng được GPU.
        """
        import json

        existing = self._gpu_server
        if existing is not None and existing.poll() is None:
            return existing

        from subtitles_extractor.infrastructure.tts.vieneu_gpu_plan import (
            build_gpu_tts_plan,
        )

        plan = build_gpu_tts_plan()
        if not plan.is_ready or plan.python_exe is None:
            self._disable_gpu_worker(
                "môi trường riêng chưa đủ gói — bấm “⬇️ Cài VieNeu cho GPU”"
            )
            return None
        worker = Path(__file__).with_name("vieneu_gpu_subprocess.py")
        if not worker.is_file():
            self._disable_gpu_worker(f"bản đóng gói thiếu {worker.name}")
            return None

        # [v3.23.364] GHI stderr worker ra file tạm (thay vì DEVNULL) để ĐỌC ĐƯỢC mã chẩn
        # đoán (VIENEU_GPU_NO_TORCH / NO_VIENEU / UNAVAILABLE / LOAD_CRASH…) khi GPU hỏng.
        # Dùng file (không phải PIPE) nên KHÔNG có nguy cơ bế tắc do không ai đọc pipe.
        import tempfile

        err_file = tempfile.NamedTemporaryFile(
            mode="w+", prefix="vieneu_gpu_err_", suffix=".log",
            encoding="utf-8", errors="replace", delete=False,
        )
        self._gpu_err_path = err_file.name

        try:
            process = subprocess.Popen(
                [plan.python_exe, str(worker), "--serve"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=err_file,
                text=True, encoding="utf-8", errors="replace",
                **no_window_kwargs(),
            )
        except OSError as exc:
            err_file.close()
            self._disable_gpu_worker(f"không khởi chạy được: {exc}")
            return None

        config = {
            "mode": self._mode,
            "sample_rate": _PIPELINE_SAMPLE_RATE,
            "voice_preset": self._gpu_voice_preset,
            "reference_wav": self._gpu_reference_wav,
        }
        try:
            assert process.stdin is not None and process.stdout is not None
            process.stdin.write(json.dumps(config, ensure_ascii=False) + "\n")
            process.stdin.flush()
            # Đợi worker báo đã nạp xong model — mất khoảng 15 giây, MỘT LẦN duy nhất.
            handshake = process.stdout.readline()
        except (OSError, ValueError, AssertionError) as exc:
            self._disable_gpu_worker(f"bắt tay thất bại: {exc}{self._read_gpu_error()}")
            process.kill()
            return None

        if not handshake or '"ready": true' not in handshake.lower():
            # Ưu tiên lý do có cấu trúc từ worker; nếu không có, lấy từ stderr đã ghi.
            reason = ""
            with contextlib.suppress(ValueError):
                reason = str(json.loads(handshake).get("error", "")) if handshake else ""
            if not reason:
                reason = self._read_gpu_error()
            self._disable_gpu_worker(
                f"tiến trình GPU không báo sẵn sàng — {reason or 'không rõ lý do'}"
            )
            process.kill()
            return None

        logger.info("VieNeu GPU: tiến trình thường trú đã sẵn sàng (model nạp 1 lần).")
        self._gpu_server = process
        return process

    def _read_gpu_error(self) -> str:
        """Đọc phần đuôi stderr của GPU worker (mã chẩn đoán VIENEU_GPU_*)."""
        path = getattr(self, "_gpu_err_path", None)
        if not path:
            return ""
        try:
            text = Path(path).read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            return ""
        if not text:
            return ""
        tail = " | ".join(text.splitlines()[-4:])
        return f" [{tail[:400]}]"

    def _shutdown_gpu_server(self) -> None:
        """Đóng tiến trình GPU thường trú (an toàn khi gọi nhiều lần)."""
        process = self._gpu_server
        self._gpu_server = None
        if process is None:
            return
        with contextlib.suppress(OSError, ValueError):
            if process.stdin is not None:
                process.stdin.write('{"quit": true}\n')
                process.stdin.flush()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()

    def _disable_gpu_worker(self, reason: str) -> None:
        """Tắt đường GPU cho phần còn lại của phiên, ghi log MỘT LẦN.

        Không ghi lặp: với 55 câu thì cùng một thông điệp 55 lần làm nhật ký vô dụng.
        """
        if self._gpu_worker_disabled:
            return
        self._gpu_worker_disabled = True
        self._shutdown_gpu_server()
        hint = _gpu_worker_failure_hint(reason)
        logger.warning(
            "VieNeu: KHÔNG dùng được GPU (%s) — chuyển sang ONNX/CPU cho phiên này.%s",
            reason,
            hint,
        )

    @staticmethod
    def _to_mono_pipeline_rate(
        raw_audio: Any, engine_sr: int | None = None
    ) -> np.ndarray:
        """Chuẩn hoá output SDK về mono float32, GIỮ NGUYÊN sample rate của engine.

        VieNeu trả về nhiều dạng tuỳ phiên bản:
        * ``np.ndarray`` THUẦN — sample rate của engine (vd v3 Turbo **48kHz**).
        * ``(audio, sr)`` tuple — sr nằm ngay trong kết quả (một số bản cũ).

        [v3.23.260] KHÔNG còn ép về 24kHz. Trước đây hàm resample MỌI output về 24kHz
        để master hoá nhất quán — nhưng với v3 Turbo (48kHz) điều đó HẠ chất lượng, cắt
        dải tần 12-24kHz (đo FLAC thật: năng lượng >10kHz = 0%). Nay pipeline chạy ở
        CHÍNH sr engine (xác định ở ``generate``), nên hàm chỉ ép mono + float32, giữ
        nguyên số mẫu. Engine 48kHz -> file 48kHz; engine 24kHz -> file 24kHz.

        [v3.23.256] Vẫn xử lý đúng dạng tuple ``(audio, sr)`` (không giả định sr).

        Args:
            raw_audio: Output thô từ ``engine.infer`` (array hoặc ``(array, sr)``).
            engine_sr: Sample rate engine (giữ tương thích chữ ký cũ; array thuần
                giữ nguyên, không resample).

        Returns:
            Mảng float32 mono ở sample rate GỐC của engine.
        """
        if isinstance(raw_audio, tuple) and len(raw_audio) >= 2:
            data = raw_audio[0]
        else:
            data = raw_audio
        audio = np.asarray(data, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio

    @staticmethod
    def _sleep_with_cancel(
        delay_s: float, cancel_cb: TTSCancellationCallback | None
    ) -> None:
        """Ngủ ``delay_s`` giây nhưng kiểm tra huỷ mỗi 100ms.

        Args:
            delay_s: Tổng thời gian chờ (giây).
            cancel_cb: Callback huỷ hợp tác.

        Raises:
            TTSCancelledError: Nếu người dùng huỷ trong lúc chờ.
        """
        elapsed = 0.0
        while elapsed < delay_s:
            if cancel_cb and cancel_cb():
                raise TTSCancelledError("Huỷ khi chờ retry VieNeu-TTS.")
            time.sleep(0.1)
            elapsed += 0.1

    def _finalize_and_write(
        self, master: np.ndarray, sr: int, request: TTSRequest,
        output_path: Path, progress_cb: TTSProgressCallback | None,
    ) -> Path:
        """Master hoá (nếu bật) và ghi file, tái dùng module ``audio_mastering`` chung.

        Args:
            master: Mảng master đã ghép toàn bộ câu.
            sr: Tần số lấy mẫu.
            request: Yêu cầu TTS (chứa target_lufs, voice_clarity, output_format…).
            output_path: Đường dẫn đích.
            progress_cb: Callback tiến độ.

        Returns:
            Đường dẫn file thực đã ghi.
        """
        from subtitles_extractor.infrastructure.tts import audio_mastering as mastering

        if request.normalize:
            if progress_cb:
                progress_cb(0.95, "Master âm thanh (LUFS + true-peak + làm rõ giọng)…")
            master = mastering.master_finalize(
                master, sr,
                target_lufs=getattr(request, "target_lufs", -16.0),
                apply_clarity=getattr(request, "voice_clarity", True),
            )
        else:
            # [v3.23.209] Tắt chuẩn hoá vẫn phải NÉN ÊM (soft-limit) thay vì clip cứng:
            # master cộng dồn chồng tiếng có đỉnh vượt 1.0 (2 giọng chồng ~1.5-1.8) ->
            # np.clip cắt phẳng gây MÉO nghe rõ ở chính đoạn chồng. soft_limit nén êm
            # vùng vượt, trong suốt với phần còn lại; write_audio vẫn clip lưới cuối.
            master = mastering.soft_limit(master)

        fmt = getattr(request, "output_format", "wav")
        if progress_cb:
            progress_cb(0.97, f"Đang ghi file {fmt.upper()}…")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        written = mastering.write_audio(
            master, sr, output_path, fmt=fmt,
            subtype=getattr(request, "wav_subtype", "PCM_16"),
            bitrate_kbps=getattr(request, "output_bitrate_kbps", 320),
        )
        logger.info("VieNeu-TTS ghi file → %s", written)
        return written


# [v3.23.220] Re-export các hàm thuần dùng chung (nay định nghĩa ở ``timing_math`` /
# ``audio_utils`` / ``text_prep``) để test và mã gọi hiện có không phải sửa import.
__all__ = [
    "VieNeuTtsAdapter",
    "compute_fit_stretch_ratio",
    "describe_voice_error",
    "effective_available_seconds",
    "fit_limit_samples",
    "has_speakable_content",
    "is_effectively_silent",
    "lead_in_seconds",
    "master_length_samples",
    "match_voice_name",
    "normalize_mode",
    "resample_audio",
    "resolve_device_env",
    "stretch_ratio_cap",
    "temporary_env",
    "total_speed_ratio",
    "trim_edge_silence",
]
