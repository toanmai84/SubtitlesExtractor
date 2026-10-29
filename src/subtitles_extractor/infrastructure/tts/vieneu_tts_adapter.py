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
import json
import logging
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
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
from subtitles_extractor.domain.value_objects.voice_profile import (
    VoiceProfile,
    gender_from_voice_label,
    is_featured_label,
    style_from_voice_label,
)
from subtitles_extractor.infrastructure.process.hidden_process import (
    no_window_kwargs,
)
from subtitles_extractor.infrastructure.torch_import_blocker import without_torch_block
from subtitles_extractor.infrastructure.tts.audio_utils import (
    is_effectively_silent,
    resample_audio,
    shorter_take,
    speech_seconds,
    trim_edge_silence,
    trim_repeated_reading,
)
from subtitles_extractor.infrastructure.tts.line_placement import (
    comfort_speed,
    line_context,
    place_line,
)
from subtitles_extractor.infrastructure.tts.text_prep import (
    count_internal_breaks,
    dem_am_tiet,
    has_repeated_word,
    has_speakable_content,
)
from subtitles_extractor.infrastructure.tts.text_prep import (
    preprocess_tts_text as _preprocess_tts_text,
)
from subtitles_extractor.infrastructure.tts.text_prep import (
    skip_from_request as _skip_from_request,
)
from subtitles_extractor.infrastructure.tts.timing_math import (
    MASTER_TAIL_PAD_S as _MASTER_TAIL_PAD_S,  # noqa: F401  (re-export cho test)
)
from subtitles_extractor.infrastructure.tts.timing_math import (
    MAX_LEAD_IN_S as _MAX_LEAD_IN_S,
)
from subtitles_extractor.infrastructure.tts.timing_math import (
    QUALITY_STRETCH_CAP as _QUALITY_STRETCH_CAP,  # noqa: F401  (re-export cho test)
)
from subtitles_extractor.infrastructure.tts.timing_math import (
    VIENEU_MIN_BASE_S,
    VIENEU_MIN_PER_SYLLABLE_S,
    audible_overlap_seconds,
    compute_fit_stretch_ratio,
    effective_available_seconds,
    fit_limit_samples,
    generation_time_cap_seconds,
    is_abnormally_long,  # noqa: F401  (re-export cho test)
    is_abnormally_long_vs_floor,
    is_short_line_overlong,
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

#: [v3.23.647] Điểm kiểm tra giữa lượt cả phim: mỗi ngần này dòng HOẶC ngần này giây (tuỳ cái nào tới trước).
_CHECKPOINT_EVERY_LINES = 25
_CHECKPOINT_EVERY_S = 30.0

#: [v3.23.415] Dừng sớm retry "dài bất thường" khi không rút ngắn thêm: cần thử tối thiểu
#: _OVERLONG_MIN_ATTEMPTS lần, và dừng nếu bản ngắn nhất chững lại _OVERLONG_STALL_LIMIT lần
#: LIÊN TIẾP. Với thán từ ngắn (model ngân dài nhất quán) → dừng sau ~3-4 lần thay vì 10, giữ
#: nguyên kết quả (bản ngắn nhất). Vẫn cho phép retry đủ khi model đang cải thiện dần.
_OVERLONG_MIN_ATTEMPTS = 3
_OVERLONG_STALL_LIMIT = 3

# [v3.23.257] Frame rate v3 Turbo để quy đổi trần (giây -> max_new_frames). Đo từ
# source vieneu 3.2.3: base.py hop_length=480 @ sr 48000 -> 100 frame/s.
_V3TURBO_FRAMES_PER_SECOND = 100

#: [v3.23.627] Frame codec THẬT của v3 Turbo (MOSS-Audio-Tokenizer-Nano): 3840 mẫu/frame @ 48 kHz
#: = 12,5 frame/giây (vieneu_utils/core_utils.py: ``CODEC_SAMPLES_PER_FRAME = 3840``;
#: ``max_new_frames`` đếm theo frame này). Hằng 100 ở trên là hop 480 của đường neucodec (chế độ
#: standard): áp cho v3 Turbo thì "trần 6× kỳ vọng" thành ~48× — không chặn được gì.
_V3TURBO_CODEC_FRAMES_PER_SECOND = 12.5


def frames_per_second_for_mode(mode: str) -> float:
    """Số frame sinh ra mỗi giây audio của chế độ VieNeu (để đổi trần giây → ``max_new_frames``)."""
    return _V3TURBO_CODEC_FRAMES_PER_SECOND if mode == "v3turbo" else _V3TURBO_FRAMES_PER_SECOND

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
#: [v3.23.595] Giá trị ``precision`` hợp lệ của VieNeu v3 Turbo (đường ONNX/CPU).
_VIENEU_PRECISIONS = frozenset({"int8", "fp32"})

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


def _strip_surrogates(value: Any) -> Any:
    """[v3.23.411] Bỏ ký tự surrogate lạc khỏi chuỗi (đệ quy cho dict/list).

    Text từ OCR/dịch đôi khi lẫn ký tự surrogate (U+DC80–U+DCFF — byte không giải mã được
    UTF-8, lọt vào qua ``surrogateescape``). ``json.dumps(..., ensure_ascii=False)`` rồi ghi
    vào stdin UTF-8 sẽ ném ``'utf-8' codec can't encode character '\\udcXX': surrogates not
    allowed`` → câu TTS hỏng, 3 câu liên tiếp thì tắt GPU. Làm sạch trước khi gửi.

    Args:
        value: Chuỗi, hoặc dict/list chứa chuỗi (payload gửi worker).

    Returns:
        Cùng cấu trúc nhưng chuỗi đã bỏ surrogate.
    """
    if isinstance(value, str):
        try:
            value.encode("utf-8")
            return value
        except UnicodeEncodeError:
            # 'ignore' loại BỎ surrogate (rác) thay vì thay bằng '?' (sẽ bị đọc thành tiếng).
            return value.encode("utf-8", "ignore").decode("utf-8")
    if isinstance(value, dict):
        return {key: _strip_surrogates(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_strip_surrogates(item) for item in value]
    return value


def _json_line_safe(payload: dict[str, Any]) -> str:
    """[v3.23.411] Tuần tự hoá payload thành 1 dòng JSON AN TOÀN cho stdin UTF-8.

    Làm sạch surrogate trước, rồi ``json.dumps(ensure_ascii=False)``. Kết quả luôn encode được
    UTF-8, không làm chết worker GPU.
    """
    return json.dumps(_strip_surrogates(payload), ensure_ascii=False) + "\n"


def normalize_voice_candidates(voice_name: Any, speaker: Any = "") -> list[str]:
    """[v3.23.407] Trả về danh sách ứng viên tên giọng đã làm sạch, ưu tiên giá trị.

    UI có thể truyền ``voice_name`` là chuỗi HOẶC tuple/list ``(nhãn, giá_trị)``. Hàm thuần này
    trải phẳng thành danh sách chuỗi ứng viên (giá trị đứng sau thường là ID đúng), kèm
    ``speaker`` làm phương án cuối. Không phụ thuộc engine → test dễ.

    Args:
        voice_name: Giá trị ``request.voice_name`` (chuỗi hoặc tuple/list).
        speaker: Giá trị ``request.speaker`` dự phòng.

    Returns:
        Danh sách chuỗi ứng viên (đã bỏ trống/trùng, giữ thứ tự).
    """
    raw_items: list[Any] = []
    if isinstance(voice_name, (tuple, list)):
        raw_items.extend(voice_name)
    elif voice_name:
        raw_items.append(voice_name)
    if speaker:
        raw_items.append(speaker)

    seen: set[str] = set()
    candidates: list[str] = []
    for item in raw_items:
        text = str(item).strip()
        if text and text not in seen:
            seen.add(text)
            candidates.append(text)
    return candidates


#: [v3.23.626] Tên giọng CŨ của v3 Turbo → tên hiện hành (VieNeu 3.8.1-3.8.2 đổi tên; bảng
#: `aliases` của voices_v3_turbo.json). Lựa chọn giọng đã lưu dưới tên cũ ("Minh Quân") trước
#: đây không khớp preset nào nữa: bảng phân vai âm thầm bỏ lựa chọn, dòng của vai đọc bằng
#: giọng chung. Dùng khi engine không tự có ``resolve_voice_name``.
VIENEU_VOICE_ALIASES: dict[str, str] = {
    "Minh Quân Pro": "Hải Đăng",
    "Minh Quân": "Hải Đăng",
    "Anh Khôi": "Thiện Minh",
    "Mạnh Dũng": "Quốc Tuấn",
}


#: [v3.23.626] Bản VieNeu tối thiểu nên dùng. Dưới 3.8.0 có lỗi bộ mã hoá nối thêm frame đệm
#: 455 NGHE ĐƯỢC vào clip giọng mẫu (nhân bản giọng và các giọng preset); 3.8.3 là bản mới nhất.
VIENEU_RECOMMENDED_VERSION: tuple[int, ...] = (3, 8, 3)
VIENEU_MIN_GOOD_VERSION: tuple[int, ...] = (3, 8, 0)


def parse_version(text: str | None) -> tuple[int, ...]:
    """``"3.8.3"``/``"3.8.3.post1"`` → ``(3, 8, 3)``; không đọc được → ``()``."""
    parts: list[int] = []
    for piece in (text or "").strip().split("."):
        digits = ""
        for char in piece:
            if not char.isdigit():
                break
            digits += char
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def vieneu_version_advice(installed: str | None) -> str:
    """Lời khuyên cập nhật VieNeu theo bản đang cài; rỗng khi đã đủ mới (hoặc không biết)."""
    version = parse_version(installed)
    if not version:
        return ""
    if version < VIENEU_MIN_GOOD_VERSION:
        return (
            f"VieNeu {installed} có lỗi âm thanh ở clip giọng mẫu (frame đệm nghe được) và "
            "chưa có 25 giọng preset mới. Hãy cập nhật: pip install -U vieneu sea-g2p"
        )
    if version < VIENEU_RECOMMENDED_VERSION:
        recommended = ".".join(map(str, VIENEU_RECOMMENDED_VERSION))
        return (
            f"VieNeu {installed} cũ hơn bản khuyên dùng {recommended} (đổi tên giọng, siết chặt "
            "phục vụ). Cập nhật: pip install -U vieneu sea-g2p"
        )
    return ""


def installed_vieneu_version() -> str | None:
    """Bản VieNeu đang cài (``None`` nếu không đọc được)."""
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover
        return None
    try:
        return version("vieneu")
    except PackageNotFoundError:
        return None


def describe_vieneu_import_error(exc: BaseException, mode: str, vieneu_installed: bool) -> str:
    """Thông báo cho ``ImportError`` khi dựng engine VieNeu (hàm thuần).

    [v3.23.628] SDK ném ``ImportError`` KHÔNG có ``.name`` khi chế độ standard/turbo thiếu
    ``llama-cpp-python`` ("Failed to import `llama_cpp`…"); bản trước coi mọi ImportError không
    tên là "VieNeu chưa cài" nên người dùng đã cài vieneu 3.8.3 vẫn bị bảo ``pip install vieneu``.
    """
    detail = str(exc).strip()
    if vieneu_installed:
        hint = (
            "Chế độ standard/turbo của VieNeu cần thêm llama-cpp-python (bản biên dịch lớn). "
            "Hãy dùng chế độ 'v3 Turbo' — chạy ONNX, không cần llama-cpp."
            if "llama" in detail.lower() and mode in ("standard", "turbo")
            else "Cập nhật đầy đủ: pip install -U vieneu sea-g2p onnxruntime soundfile"
        )
        return f"VieNeu-TTS đã cài nhưng không nạp được chế độ '{mode}': {detail}\n{hint}"
    return "VieNeu-TTS chưa cài. Chạy: pip install vieneu soundfile"


def resolve_voice_alias(requested: str, engine: object | None = None) -> str:
    """Tên giọng hiện hành của ``requested`` (nếu là tên cũ); không phải bí danh → giữ nguyên.

    Ưu tiên bảng alias CỦA ENGINE (``resolve_voice_name``, luôn đúng với bản SDK đang cài);
    không có thì dùng :data:`VIENEU_VOICE_ALIASES`.
    """
    resolver = getattr(engine, "resolve_voice_name", None)
    if callable(resolver):
        try:
            resolved = resolver(requested)
        except (RuntimeError, ValueError, TypeError):
            resolved = None
        if isinstance(resolved, str) and resolved:
            return resolved
    return VIENEU_VOICE_ALIASES.get((requested or "").strip(), requested)


def aliases_of(voice_id: str) -> tuple[str, ...]:
    """Các tên cũ trỏ về ``voice_id`` (theo bảng tĩnh)."""
    return tuple(old for old, new in VIENEU_VOICE_ALIASES.items() if new == voice_id)


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
_COMPATIBLE_TRANSFORMERS_SPEC: Final[str] = "transformers<4.53"


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


_LENGTH_CAP_KEYS = ("max_new_frames", "max_duration_s", "max_duration", "max_length_s", "max_seconds")


def _scale_length_caps(kwargs: dict, own_text: str, group_texts: list[str]) -> dict:  # type: ignore[type-arg]
    """[v3.23.655] Trần độ dài của lô theo câu DÀI NHẤT trong lô (trần vốn tính theo câu ``idx``).

    Gom lô toàn cục kéo câu dài ở xa vào cùng lô; giữ trần của câu ngắn thì audio câu dài bị cắt cụt.
    """
    own = max(1, dem_am_tiet((own_text or "").strip()))
    longest = max((dem_am_tiet((text or "").strip()) for text in group_texts), default=own)
    if longest <= own:
        return kwargs
    ratio = longest / own
    scaled = dict(kwargs)
    for key in _LENGTH_CAP_KEYS:
        value = scaled.get(key)
        if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
            continue
        scaled[key] = value * ratio if isinstance(value, float) else int(value * ratio) + 1
    return scaled


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
        self, mode: str = "v3turbo", emotion: str = "natural", force_cpu: bool = True,
        batch_size: int | None = None, emotion_cues: bool = False,
        precision: str | None = None,
    ) -> None:
        """Khởi tạo adapter (chưa nạp model — lazy load khi generate).

        Args:
            mode: Chế độ engine ("standard" chất lượng cao nhất, "turbo"/"v3turbo" nhanh
                hơn). Giá trị không hợp lệ tự lùi về "standard".
            emotion: Sắc thái giọng — "natural" (hội thoại tự nhiên) hoặc "storytelling"
                (kể chuyện). Không hardcode: truyền vào để dễ cấu hình/test.
            force_cpu: True (mặc định) ép chạy CPU/ONNX, né lỗi PyTorch/cuDNN trên máy có
                CUDA lệch phiên bản (WinError 127). False để SDK tự chọn CPU/GPU.
            precision: [v3.23.595] Độ chính xác backbone ONNX/CPU của v3 Turbo: ``"int8"``
                (nhanh hơn, cần CPU có VNNI) hoặc ``"fp32"``; ``None`` = mặc định SDK
                (``fp32`` ở VieNeu 3.8).
        """
        self._mode: str = normalize_mode(mode)
        self._emotion: str = emotion if emotion in ("natural", "storytelling") else "natural"
        # [v3.23.446] Emotion cues CHỈ v3 Turbo hỗ trợ → chỉ bật khi mode=v3turbo VÀ người dùng
        # đồng ý (thử nghiệm). Chuyển interjection cười/thở dài → tag cho giọng biểu cảm.
        self._emotion_cues: bool = bool(emotion_cues)
        self._force_cpu: bool = force_cpu
        self._precision: str | None = (
            precision if precision in _VIENEU_PRECISIONS else None
        )
        self._engine: Any | None = None
        self._voice_data_cache: Any | None = None
        #: [v3.23.647] Điểm kiểm tra giữa lượt cả phim: ``cb(kết quả các dòng đã xong, sample rate)``.
        self._checkpoint_cb: Callable[[list[TTSSegmentResult], int], object] | None = None
        self._checkpoint_every_lines: int = _CHECKPOINT_EVERY_LINES
        self._checkpoint_every_s: float = _CHECKPOINT_EVERY_S
        # [v3.23.593] Lồng tiếng: giọng riêng từng vai (ID yêu cầu -> (voice data, ID
        # khớp)) và tra ngược voice data -> ID preset để worker GPU tự lấy đúng giọng.
        self._line_voice_cache: dict[str, tuple[Any, str]] = {}
        self._line_voice_presets: dict[int, str] = {}
        self._batch_voices: list[str] = []
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
        # [v3.23.412/416] Batch TTS GPU. Ưu tiên tham số truyền vào (từ Cài đặt); nếu None thì
        # lùi về env SUBEXT_VIENEU_BATCH (tương thích ngược). 0/1 = TẮT (đường per-câu).
        if batch_size is not None:
            self._batch_size: int = max(0, int(batch_size))
        else:
            try:
                self._batch_size = max(0, int(os.environ.get("SUBEXT_VIENEU_BATCH", "0")))
            except (TypeError, ValueError):
                self._batch_size = 0
        #: Danh sách câu đã làm sạch (đặt đầu mỗi phiên tổng hợp) + cache audio theo idx.
        self._batch_texts: list[str] = []
        self._batch_cache: dict[int, Any] = {}
        # [v3.23.420] Số CHUNK/forward GPU cho infer_batch (đòn bẩy VRAM/tốc độ THẬT — mặc định
        # vieneu chỉ 4 → phí VRAM). Tự chọn theo VRAM (8GB→32), tính LƯỜI ở lần cần đầu tiên.
        self._gpu_chunk_batch: int | None = None
        # [v3.23.264] Nhiệt độ tuỳ chọn (None -> giữ mặc định SDK 0.8). Không ép
        # thấp mặc định vì câu ngắn có thể bị model từ chối đọc ở temperature thấp.
        self._temperature: float | None = None

    def set_checkpoint_callback(
        self, callback: "Callable[[list[TTSSegmentResult], int], object] | None"
    ) -> None:
        """[v3.23.647] Đặt hàm nhận điểm kiểm tra định kỳ cho LẦN :meth:`generate` KẾ TIẾP (``None`` = tắt).

        Dùng một lần: ``generate`` lấy hàm rồi xoá khỏi adapter, nên TTS lại vài dòng bằng cùng adapter không bao giờ ghi
        điểm kiểm tra của lượt cũ (kết quả tập con dòng sẽ làm hỏng cache). Gọi trên luồng đang tổng hợp, sau khi dòng
        đã ghi xong audio; lỗi của hàm không làm hỏng lượt TTS.
        """
        self._checkpoint_cb = callback

    def _maybe_checkpoint(
        self,
        callback: Callable[[list[TTSSegmentResult], int], object] | None,
        results: list[TTSSegmentResult],
        sample_rate: int,
        state: list[float],
    ) -> None:
        """Gửi điểm kiểm tra nếu đủ số dòng HOẶC đủ thời gian kể từ lần trước (``state = [số dòng, lúc]``)."""
        if callback is None:
            return
        now = time.monotonic()
        if (
            len(results) - state[0] < self._checkpoint_every_lines
            and now - state[1] < self._checkpoint_every_s
        ):
            return
        state[0], state[1] = float(len(results)), now
        try:
            callback(list(results), sample_rate)
        except Exception as exc:  # noqa: BLE001 - điểm kiểm tra là phụ: không được làm hỏng lượt TTS
            logger.warning("VieNeu-TTS: ghi điểm kiểm tra lỗi (%s) — tiếp tục.", exc)

    def _should_inject_cues(self) -> bool:
        """[v3.23.446] True nếu nên chèn tag cảm xúc: bật toggle VÀ engine là v3 Turbo (chỉ v3
        Turbo hỗ trợ). Các mode cũ (standard/turbo) không hiểu tag → giữ nguyên interjection."""
        return self._emotion_cues and self._mode == "v3turbo"

    #: [v3.23.630] Engine ghi audio RIÊNG từng dòng và tổng hợp lại được tập con dòng →
    #: dùng được bộ nhớ đệm từng dòng (``tts_line_cache``).
    supports_line_cache: bool = True

    def line_cache_key(self) -> str:
        """[v3.23.630] Dấu của engine + chế độ ảnh hưởng audio từng dòng (khoá cache).

        Không gồm ``force_cpu``/``batch_size``: chỉ đổi nơi chạy, không đổi giọng đọc.
        """
        return (
            f"vieneu|{self._mode}|{self._emotion}|cues={int(self._should_inject_cues())}"
            f"|prec={self._precision or ''}"
        )

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
        return [voice_id for _label, voice_id in self._preset_voice_pairs()]

    def list_voice_catalog(self, language: str) -> list[VoiceProfile]:
        """[v3.23.593] Giọng preset kèm giới tính đọc từ nhãn VieNeu (để phân vai).

        Args:
            language: Bỏ qua — VieNeu chỉ tiếng Việt.

        Returns:
            Danh sách :class:`VoiceProfile`; rỗng nếu không lấy được.
        """
        return [
            VoiceProfile(
                voice_id=voice_id, label=str(label),
                gender=gender_from_voice_label(str(label)),
                style=style_from_voice_label(str(label)),
                featured=is_featured_label(str(label)),
                aliases=aliases_of(voice_id),
            )
            for label, voice_id in self._preset_voice_pairs()
        ]

    def _preset_voice_pairs(self) -> list[tuple[str, str]]:
        """Cặp ``(nhãn, ID)`` giọng preset của chế độ hiện tại; rỗng nếu không lấy."""
        if not self.is_available():
            return []
        # [v3.23.647] Engine chưa nạp thì đọc danh mục từ đĩa (lần trước đã lưu) — khỏi nạp VieNeu ~36 s chỉ để liệt kê giọng.
        from subtitles_extractor.infrastructure.tts import voice_catalog_cache as catalog_cache

        cache_file = catalog_cache.default_cache_path()
        version = installed_vieneu_version()
        if self._engine is None:
            cached = catalog_cache.load_catalog(cache_file, self._mode, version)
            if cached is not None:
                return cached
        try:
            engine = self._get_or_load_engine("auto")
            pairs = [(label, voice_id) for label, voice_id in engine.list_preset_voices()]
            catalog_cache.save_catalog(cache_file, self._mode, version, pairs)
            return pairs
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
            # [v3.23.593] Lồng tiếng nhiều vai: giọng riêng của từng dòng (rỗng = một
            # giọng).
            line_voices = self._resolve_line_voices(engine, request, voice_data)
            self._batch_voices = [preset for _data, preset in line_voices]

            # [v3.23.412] Chuẩn bị lô nếu bật batch (env SUBEXT_VIENEU_BATCH≥2) và đang dùng
            # GPU. Làm sạch text y hệt _process_event để cache khớp; câu bỏ/rỗng để nguyên
            # (sẽ không được yêu cầu). TẮT batch khi ép CPU (infer_batch CPU chạy tuần tự,
            # không lợi).
            self._batch_cache = {}
            # [v3.23.450] Chuẩn bị thư mục segment nếu người dùng muốn giữ audio từng dòng
            # (để nghe lại / TTS lại dòng chọn). Tạo sẵn thư mục MỘT LẦN.
            self._segments_dir = None
            if getattr(request, "keep_line_audio", False):
                from subtitles_extractor.infrastructure.tts.segment_store import (
                    segments_dir_for,
                )
                self._segments_dir = segments_dir_for(
                    output_path, getattr(request, "segments_dir", "")
                )
                self._segments_dir.mkdir(parents=True, exist_ok=True)
            if self._batch_size >= 2 and not self._force_cpu:
                self._batch_texts = [
                    _preprocess_tts_text(
                        event.text, request.clean_tags, _skip_from_request(request),
                        strip_speaker_tag=True,
                        inject_emotion_cues=self._should_inject_cues(),
                    )[0]
                    for event in request.events
                ]
            else:
                self._batch_texts = []

            prev_audio_end = 0.0
            checkpoint_state = [0.0, time.monotonic()]
            checkpoint_cb, self._checkpoint_cb = self._checkpoint_cb, None      # dùng một lần (xem setter)
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
                # [v3.23.596] Biên chung với Gemini (``line_context``): ngân trễ elastic,
                # và mốc dòng kề THẬT khi TTS lại tập con.
                context = line_context(request, idx, prev_audio_end)
                result = self._process_event(
                    event=event, idx=idx, master=master, engine=engine,
                    sr=sample_rate, request=request,
                    voice_data=line_voices[idx][0] if line_voices else voice_data,
                    next_start_sec=context.next_start_sec, cancel_cb=cancel_cb,
                    prev_audio_end_sec=context.prev_audio_end_sec,
                    late_s=context.late_s,
                    defer_s=context.defer_s,
                )
                results.append(result)
                self._maybe_checkpoint(checkpoint_cb, results, sample_rate, checkpoint_state)
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
        # [v3.23.595] Engine int8 và fp32 là hai graph khác nhau: chọn int8 thì khoá có
        # thêm độ chính xác; mặc định giữ khoá cũ.
        precision = getattr(self, "_precision", None)
        cache_key: tuple[str, ...] = (
            (self._mode, self._emotion, precision) if precision
            else (self._mode, self._emotion)
        )
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
                # [v3.23.628] onnxruntime mới từ chối graph có external data trỏ ra ngoài thư mục
                # model — mà cache HF toàn symlink. Gỡ symlink ngay khi tải về.
                from subtitles_extractor.infrastructure.tts.hf_cache_fix import (
                    install_hf_materialize_patch,
                )

                install_hf_materialize_patch()
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
                import importlib.util

                raise TTSUnavailableError(
                    describe_vieneu_import_error(
                        exc, self._mode, importlib.util.find_spec("vieneu") is not None
                    )
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
            advice = vieneu_version_advice(installed_vieneu_version())
            if advice:
                logger.warning("%s", advice)
        return self._engine

    def _process_event(
        self, *, event: Any, idx: int, master: np.ndarray, engine: Any, sr: int,
        request: TTSRequest, voice_data: Any, next_start_sec: float | None = None,
        cancel_cb: TTSCancellationCallback | None,
        prev_audio_end_sec: float = 0.0,
        late_s: float = 0.0,
        defer_s: float = 0.0,
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
            late_s: [v3.23.596] Phần ngân trễ elastic cho phép (xem ``line_placement``).
            defer_s: [v3.23.596] Phần được đọc lùi khi câu trước ngân trễ sang.

        Returns:
            Kết quả câu (:class:`TTSSegmentResult`), ``was_skipped=True`` nếu bỏ/lỗi.
        """
        text, is_dialog = _preprocess_tts_text(
            event.text, request.clean_tags, _skip_from_request(request),
            strip_speaker_tag=True,
            inject_emotion_cues=self._should_inject_cues(),
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
        # [v3.23.547] Xét trên VĂN BẢN GỐC, không phải bản đã tiền xử lý: bước tiền xử lý
        # có thể đã quét sạch ký hiệu rác ("□□□") thành chuỗi rỗng, khiến nhánh này rơi vào
        # thông báo "text rỗng" — đúng kết quả (vẫn bỏ, vẫn không gọi model) nhưng SAI
        # nguyên nhân, làm người dùng đi tìm lỗi ở chỗ không có lỗi.
        raw_text = (event.text or "").strip()
        unspeakable = bool(raw_text) and not has_speakable_content(raw_text)
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
            cancel_cb=cancel_cb, idx=idx,
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
        # [v3.23.596] Một hàm đặt câu chung cho VieNeu/Gemini (``place_line``): giữ nguyên
        # cách tính cũ khi tắt elastic; bật thì câu chật nén tới mức êm rồi ngân trễ.
        placement = place_line(
            start_sec=start_sec,
            end_sec=end_sec,
            next_start_sec=next_start_sec,
            prev_audio_end_sec=prev_audio_end_sec,
            voice_duration_s=voice_duration,
            pause_s=pause_s,
            base_speed=float(request.base_speed),
            max_speed=max_speed,
            max_lead_s=float(getattr(request, "lead_in_s", _MAX_LEAD_IN_S)),
            late_s=late_s,
            comfort_speed=comfort_speed(request),
            defer_s=defer_s,
        )
        play_start_sec = placement.play_start_sec
        effective_available = placement.effective_available_s
        stretch_ratio = placement.speed
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
        final_overlap_s = audible_overlap_seconds(len(audio) / sr, effective_available)
        # [v3.23.450] Ghi audio RIÊNG của dòng này (nếu bật) để nghe lại / TTS lại dòng chọn.
        segment_audio_path = self._save_segment_audio(audio, sr, idx)
        # [v3.23.459] Đánh dấu dòng audio DÀI BẤT THƯỜNG (model ngân dài, mọi lần thử đều dài) để
        # bảng TTS hiện cảnh báo → người dùng lọc/nghe lại/TTS lại nhanh.
        # [v3.23.601] Đo trên TIẾNG đã cắt lặng, TRƯỚC khi nén — không phải audio cuối. Ca
        # thật dòng 991 "Đúng thế.": mọi lần thử ra 2.16s tiếng (3.4 lần mức đọc gọn), bị
        # nén 2.5x còn 0.86s nên lọt dưới ngưỡng 1.8s và bảng không báo dòng nào.
        abnormal_length = is_abnormally_long_vs_floor(
            voice_duration,
            dem_am_tiet(text.strip()),
            min_base_s=VIENEU_MIN_BASE_S,
            min_per_syllable_s=VIENEU_MIN_PER_SYLLABLE_S,
        ) or is_short_line_overlong(  # [v3.23.663] câu ngắn còn đọc lặp (không cắt được) cũng báo trên bảng
            voice_duration,
            dem_am_tiet(text.strip()),
            min_base_s=VIENEU_MIN_BASE_S,
            min_per_syllable_s=VIENEU_MIN_PER_SYLLABLE_S,
        )
        return TTSSegmentResult(
            event_index=idx, start_sec=start_sec, end_sec=end_sec,
            text=text, audio_duration_s=len(audio) / sr,
            speed_used=speed_used, was_truncated=was_truncated,
            overlap_s=final_overlap_s,
            segment_audio_path=segment_audio_path,
            abnormal_length=abnormal_length,
            # [v3.23.193] adjusted = mốc THẬT của tiếng -> use case nhận diện "có điều
            # chỉnh" -> XUẤT SRT đồng bộ TTS.
            # [v3.23.218] Mốc thật nay có thể sớm hơn phụ đề (ăn gian đầu) -> SRT đồng bộ
            # phải phản ánh đúng để phụ đề khớp tiếng.
            adjusted_start_sec=play_start_sec,
            adjusted_end_sec=max(end_sec, play_start_sec + len(audio) / sr),
        )

    def _save_segment_audio(self, audio: np.ndarray, sr: int, idx: int) -> str:
        """[v3.23.450] Ghi audio một dòng ra WAV riêng (nếu bật keep_line_audio). Trả đường dẫn
        (rỗng nếu không bật hoặc lỗi ghi — không chặn TTS chính)."""
        segments_dir = getattr(self, "_segments_dir", None)
        if segments_dir is None:
            return ""
        try:
            import soundfile as sf

            from subtitles_extractor.infrastructure.tts.segment_store import segment_path

            path = segment_path(segments_dir, idx)
            # [v3.23.452] Ghi PCM_16 (không phải float32): QMediaPlayer/Media Foundation trên
            # Windows hay phát chập chờn/không phát với WAV float 32-bit. PCM_16 tương thích rộng.
            # Re-merge đọc lại dtype=float32 (soundfile tự quy đổi) → không ảnh hưởng độ chính xác.
            clipped = np.clip(audio, -1.0, 1.0)
            sf.write(str(path), clipped, sr, subtype="PCM_16")
            return str(path)
        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning("Không ghi được segment audio dòng %d: %s", idx, exc)
            return ""

    def _synthesize_with_retry(
        self, *, engine: Any, text: str, request: TTSRequest, voice_data: Any,
        cancel_cb: TTSCancellationCallback | None, idx: int = -1,
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
        # [v3.23.415] Dừng sớm khi retry KHÔNG cải thiện: thán từ/từ ngắn ('Được.', 'Giá!')
        # model ngân dài NHẤT QUÁN — mọi lần thử đều ~bằng nhau, retry tới 10 lần vô ích (mỗi
        # lần 1 lần infer GPU). Nếu bản ngắn nhất không giảm thêm sau _OVERLONG_STALL_LIMIT lần
        # LIÊN TIẾP (và đã thử tối thiểu _OVERLONG_MIN_ATTEMPTS) → dừng, dùng bản ngắn nhất.
        # KHÔNG đổi kết quả (vẫn bản ngắn nhất), chỉ bỏ các lần thử thừa.
        _overlong_best_size: int | None = None
        _overlong_stall = 0
        # [v3.23.663] Câu ngắn đọc lặp mà không cắt được (hai lần đọc dính liền): các lần sau bỏ mã tham chiếu.
        drop_ref_codes = False
        for attempt in range(retry_count):
            if cancel_cb and cancel_cb():
                raise TTSCancelledError("Huỷ trong khi tổng hợp VieNeu-TTS.")
            try:
                raw_audio = self._infer_once(
                    engine, text, voice_data,
                    idx=idx if attempt == 0 else -1,
                    **({"use_ref_codes": False} if drop_ref_codes else {}),
                )
                audio = self._to_mono_pipeline_rate(
                    raw_audio, getattr(engine, "sample_rate", None)
                )
                # [v3.23.205] Audio toàn IM LẶNG (model sinh nhưng không có tiếng) coi
                # như thất bại -> retry, giống audio rỗng (ca thực tế: câu #58 "Ý chú
                # là" 2.2s im lặng lọt lưới size>0 -> mất thoại mà báo OK).
                if audio.size > 0 and not is_effectively_silent(audio):
                    # [v3.23.601] Đo PHẦN TIẾNG, không đo độ dài thô. Khâu đặt câu cắt
                    # lặng biên ngay sau bước này, nên im lặng đầu/cuối không bao giờ tới
                    # tai người nghe. Nhật ký thật (giọng Ngọc Huyền, 1-3 âm tiết): 27 lần
                    # lấy mẫu lại vì bản THÔ 2.16-2.88s vượt ngưỡng 1.8-2.1s, trong khi
                    # bản được giữ của chính các dòng đó có 0.57-2.08s tiếng (VieNeu sinh
                    # im lặng đầu câu tới 1.42s — đo ở v3.23.204).
                    duration_s = speech_seconds(audio, engine_sr)
                    # [v3.23.221] Dạng lỗi NGƯỢC LẠI của im lặng: model "ngân dài" một âm
                    # tiết (ca thực: "Ừm." 3 ký tự -> 2.55s, +7.7σ) -> bị nén kịch trần
                    # 2.0x (tan formant) và đè lên câu sau. Lấy mẫu lại thường cho bản
                    # bình thường; giữ bản ngắn nhất nếu mọi lần đều dài.
                    # [v3.23.261] Đo theo BIÊN DƯỚI THEO ÂM TIẾT (R²=0.98) với hằng số
                    # engine VieNeu, thay lưới cũ theo ký tự (R²=0.07 -> báo động sai với
                    # câu nhiều dấu câu như "Chú...", "Ồ,"). "Ừm." 1 âm -> ngưỡng đúng.
                    # [v3.23.663] Câu ngắn (≤3 tiếng) có lưới riêng: lỗi chính là ĐỌC LẠI cả câu, luôn ngắn hơn
                    # sàn 1,8 s của lưới chung nên trước đây lọt hết. Có khoảng ngừng giữa hai lần đọc thì cắt
                    # bỏ lần đọc lại (bản đầu đọc đủ lời) — sinh lại thường vô ích vì câu hỏng hay hỏng mọi lần.
                    short_overlong = is_short_line_overlong(
                        duration_s, syllable_count,
                        min_base_s=VIENEU_MIN_BASE_S,
                        min_per_syllable_s=VIENEU_MIN_PER_SYLLABLE_S,
                    )
                    general_overlong = is_abnormally_long_vs_floor(
                        duration_s,
                        syllable_count,
                        min_base_s=VIENEU_MIN_BASE_S,
                        min_per_syllable_s=VIENEU_MIN_PER_SYLLABLE_S,
                    )
                    if short_overlong:
                        # Lời tự lặp ("Hả? Hả?") thì khoảng ngừng là thật — không cắt.
                        trimmed = (
                            None if has_repeated_word(text)
                            else trim_repeated_reading(
                                audio, engine_sr, syllable_count, count_internal_breaks(text),
                            )
                        )
                        if trimmed is not None and not is_short_line_overlong(
                            speech_seconds(trimmed, engine_sr), syllable_count,
                            min_base_s=VIENEU_MIN_BASE_S,
                            min_per_syllable_s=VIENEU_MIN_PER_SYLLABLE_S,
                        ):
                            logger.info(
                                "VieNeu-TTS: '%s' bị đọc lặp (%.2fs tiếng cho %d âm tiết) — cắt bỏ lần đọc "
                                "lại, còn %.2fs.",
                                text[:25], duration_s, syllable_count,
                                speech_seconds(trimmed, engine_sr),
                            )
                            return trimmed
                        if not general_overlong:
                            # Không cắt được (hai lần đọc dính liền, hoặc giọng đọc chậm hợp lệ): chỉ sinh lại
                            # MỘT lần, bỏ mã tham chiếu, rồi lấy bản ít tiếng hơn — không đốt cả chuỗi retry.
                            overlong_best = shorter_take(overlong_best, audio, sr=engine_sr)
                            if drop_ref_codes:
                                return overlong_best
                            drop_ref_codes = True
                            logger.info(
                                "VieNeu-TTS: '%s' nghi đọc lặp (%.2fs tiếng cho %d âm tiết) — sinh lại một lần "
                                "không dùng mã tham chiếu.",
                                text[:25], duration_s, syllable_count,
                            )
                            continue
                        drop_ref_codes = True
                    if not short_overlong and not general_overlong:
                        return audio
                    # [v3.23.601] "Ngắn nhất" theo phần TIẾNG: bản thô dài vì im lặng biên
                    # vẫn là bản đọc gọn.
                    overlong_best = shorter_take(overlong_best, audio, sr=engine_sr)
                    # [v3.23.415] Theo dõi cải thiện; dừng sớm nếu chững lại.
                    best_speech_samples = round(
                        speech_seconds(overlong_best, engine_sr) * engine_sr
                    )
                    if (
                        _overlong_best_size is not None
                        and best_speech_samples >= _overlong_best_size
                    ):
                        _overlong_stall += 1
                    else:
                        _overlong_stall = 0
                    _overlong_best_size = best_speech_samples
                    # Lấy mẫu lại NGAY: model offline + ngẫu nhiên, chờ vô ích.
                    wait_before_next = False
                    logger.warning(
                        "VieNeu-TTS lần %d/%d: audio DÀI BẤT THƯỜNG %.2fs cho %d âm tiết "
                        "('%s…') — nghi model ngân dài, lấy mẫu lại.",
                        attempt + 1, retry_count, duration_s, syllable_count, text[:25],
                    )
                    if (
                        attempt + 1 >= _OVERLONG_MIN_ATTEMPTS
                        and _overlong_stall >= _OVERLONG_STALL_LIMIT
                    ):
                        logger.info(
                            "VieNeu-TTS: retry không rút ngắn thêm sau %d lần cho '%s…' — "
                            "dừng sớm, dùng bản ngắn nhất (tiết kiệm %d lần infer).",
                            attempt + 1, text[:25], retry_count - (attempt + 1),
                        )
                        break
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
                text[:25], speech_seconds(overlong_best, engine_sr),
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
        # [v3.23.407] Chuẩn hoá tên giọng cho worker GPU. Trước đây dùng thẳng
        # str(request.voice_name); nhưng UI có thể truyền TUPLE (nhãn, giá_trị) → str(tuple)
        # cho ra "('Minh Đức — …', 'Minh Đức')" → worker báo Voice not found (đường CPU không
        # dính vì dùng request.speaker + khớp mềm). Tách giá trị hợp lệ + khớp mềm y như CPU.
        self._gpu_voice_preset = self._normalize_gpu_voice_name(engine, request)

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

    def _resolve_line_voices(
        self, engine: Any, request: TTSRequest, default_voice_data: Any
    ) -> list[tuple[Any, str]]:
        """[v3.23.593] Giọng của TỪNG dòng khi lồng tiếng nhiều vai.

        Args:
            engine: Instance VieNeu đã nạp.
            request: Yêu cầu TTS (``line_voices`` theo vị trí dòng).
            default_voice_data: Giọng chung của phiên (preset hoặc nhân bản).

        Returns:
            ``(voice data, ID preset cho worker GPU)`` theo vị trí dòng; ID rỗng = giọng
            chung. Danh sách rỗng khi yêu cầu chỉ dùng một giọng.
        """
        if not request.line_voices:
            return []
        resolved: list[tuple[Any, str]] = []
        for position in range(len(request.events)):
            requested = request.voice_at(position, "")
            if not requested:
                resolved.append((default_voice_data, ""))
                continue
            if requested not in self._line_voice_cache:
                self._line_voice_cache[requested] = self._resolve_line_voice(
                    engine, requested
                )
            voice_data, preset = self._line_voice_cache[requested]
            resolved.append((voice_data, preset) if preset else (default_voice_data, ""))
        return resolved

    def _resolve_line_voice(self, engine: Any, requested: str) -> tuple[Any, str]:
        """Lấy dữ liệu một giọng preset của vai; không có thì báo để đọc bằng giọng chung.

        Khác :meth:`_preset_voice_with_soft_match` (lùi về giọng ĐẦU TIÊN): một vai
        thiếu giọng mà lấy giọng đầu tiên sẽ trùng giọng vai khác — lùi về giọng chung
        người dùng đã chọn thì dễ nhận ra hơn.

        Args:
            engine: Instance VieNeu đã nạp.
            requested: ID giọng yêu cầu cho vai.

        Returns:
            ``(voice data, ID khớp)``; ``(None, "")`` nếu không lấy được.
        """
        try:
            available = [voice_id for _label, voice_id in engine.list_preset_voices()]
        except (AttributeError, TypeError, RuntimeError, ValueError):
            available = []
        requested = resolve_voice_alias(requested, engine)
        matched = match_voice_name(requested, available) if available else None
        if matched is None:
            logger.warning(
                "VieNeu-TTS: chế độ %s không có giọng '%s' — các dòng của vai này đọc "
                "bằng giọng chung.", self._mode, requested,
            )
            return None, ""
        try:
            voice_data = engine.get_preset_voice(matched)
        except (RuntimeError, OSError, ValueError, KeyError) as exc:
            logger.warning(
                "VieNeu-TTS: không lấy được giọng '%s' (%s) — đọc bằng giọng chung.",
                matched, exc,
            )
            return None, ""
        self._line_voice_presets[id(voice_data)] = matched
        return voice_data, matched

    def _normalize_gpu_voice_name(self, engine: Any, request: Any) -> str:
        """[v3.23.407] Chọn tên giọng HỢP LỆ để gửi worker GPU (xử lý tuple + khớp mềm).

        Trải phẳng ``voice_name`` (có thể là tuple ``(nhãn, giá_trị)``) + ``speaker`` thành ứng
        viên, rồi khớp mềm với danh sách giọng của engine (y hệt đường CPU). Trả tên khớp; nếu
        không khớp được thì trả ứng viên cuối (để worker tự fallback) hoặc rỗng.

        Args:
            engine: Instance VieNeu đã nạp (để lấy danh sách giọng khả dụng).
            request: TTSRequest chứa ``voice_name``/``speaker``.

        Returns:
            Tên giọng hợp lệ (chuỗi), hoặc rỗng nếu không có ứng viên nào.
        """
        candidates = normalize_voice_candidates(
            getattr(request, "voice_name", ""), getattr(request, "speaker", "")
        )
        if not candidates:
            return ""
        try:
            available = [voice_id for _label, voice_id in engine.list_preset_voices()]
        except (AttributeError, TypeError, RuntimeError, ValueError):
            available = []
        if available:
            for candidate in candidates:
                matched = match_voice_name(resolve_voice_alias(candidate, engine), available)
                if matched is not None:
                    if matched != candidate:
                        logger.info(
                            "VieNeu-TTS (GPU): giọng '%s' khớp mềm với '%s'.",
                            candidate, matched,
                        )
                    return matched
            logger.warning(
                "VieNeu-TTS (GPU): không khớp giọng %s với %s — để worker tự fallback.",
                candidates, available,
            )
        return candidates[-1]

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
        requested = resolve_voice_alias(requested, engine)
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
        * ``precision`` -> truyền nếu chữ ký nhận, hoặc nếu là factory ``**kwargs`` của
          VieNeu 3.8 ở mode ``v3turbo`` (lớp duy nhất có tham số này).

        Nếu không nội soi được chữ ký (vd C-extension), thử lần lượt ``{mode}`` rồi ``{}``
        (mặc định hoàn toàn) — luôn khởi tạo được engine thay vì sập.

        Args:
            vieneu_cls: Lớp ``Vieneu`` vừa import.

        Returns:
            Instance ``Vieneu`` đã khởi tạo.
        """
        try:
            signature_params = inspect.signature(vieneu_cls).parameters
        except (AttributeError, TypeError, ValueError):
            signature_params = {}
        params = frozenset(signature_params)
        # [v3.23.595] ``vieneu.Vieneu`` 3.8 là hàm factory ``Vieneu(mode, **kwargs)`` chuyển
        # tiếp kwargs cho lớp của từng mode, nên chữ ký KHÔNG liệt kê ``precision`` — dò tên
        # tham số thì không bao giờ truyền được (ứng dụng Gradio chính thức gọi
        # ``Vieneu(mode="v3turbo", precision=...)``).
        forwards_kwargs = any(
            param.kind is inspect.Parameter.VAR_KEYWORD
            for param in signature_params.values()
        )

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
        # [v3.23.255] VieNeu 3.x nhận ``precision``. [v3.23.595] Ở VieNeu 3.8 mặc định là
        # "fp32" (chất lượng tối đa); "int8" nhanh ~3x mỗi frame trên CPU có VNNI (chú
        # thích SDK). Chỉ truyền khi app có đặt VÀ constructor nhận.
        _precision = getattr(self, "_precision", None)
        if _precision and (
            "precision" in params or (forwards_kwargs and self._mode == "v3turbo")
        ):
            kwargs["precision"] = _precision

        try:
            return vieneu_cls(**kwargs)
        except TypeError:
            # Chữ ký không nội soi được, hoặc giá trị style không hợp lệ -> thử tối giản
            # dần: chỉ mode, rồi mặc định hoàn toàn. Luôn có engine thay vì sập cả phiên.
            if "precision" in kwargs:
                logger.warning(
                    "VieNeu-TTS: bản SDK này không nhận precision=%s — chạy độ chính xác "
                    "mặc định.", kwargs["precision"],
                )
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

    def _infer_once(
        self, engine: Any, text: str, voice_data: Any, idx: int = -1,
        use_ref_codes: bool = True,
    ) -> Any:
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
                kwargs["max_new_frames"] = (
                    int(cap_s * frames_per_second_for_mode(self._mode)) + 50
                )
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
        # [v3.23.663] Bỏ mã tham chiếu trong ngữ cảnh (chỉ giữ speaker embedding) cho lần sinh lại câu ngắn bị đọc
        # lặp. Đo thật (71 câu 1–3 tiếng × 3 lần, giọng Ngọc Huyền): tỉ lệ đọc lặp 23% → 9,4%; độ giống giọng
        # (cosine speaker embedding) 0,684 → 0,687 — không đổi.
        if not use_ref_codes and "use_ref_codes" in supported:
            kwargs["use_ref_codes"] = False
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
            # [v3.23.412] Lần thử đầu (idx>=0) + bật batch: lấy audio từ lô infer_batch.
            if idx >= 0 and self._batch_size >= 2:
                batched = self._get_batched_audio(idx, kwargs)
                if batched is not None:
                    return batched
            # [v3.23.593] Worker GPU tự giải quyết giọng — gửi kèm ID preset của vai.
            gpu_preset = getattr(self, "_line_voice_presets", {}).get(id(voice_data), "")
            result = self._infer_via_gpu_worker(
                text, {**kwargs, "voice_preset": gpu_preset} if gpu_preset else kwargs
            )
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

        if getattr(self, "_gpu_worker_disabled", False):
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
                        "voice_preset", "use_ref_codes",
                    )
                    if name in kwargs
                },
            }
            try:
                assert process.stdin is not None and process.stdout is not None
                process.stdin.write(_json_line_safe(request))
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

    def _resolve_gpu_chunk_batch(self) -> int:
        """[v3.23.420] Số chunk/forward GPU cho ``infer_batch`` (tự chọn theo VRAM, cache).

        Mặc định vieneu chỉ gộp 4 chunk/forward → dùng rất ít VRAM. Ta nâng theo VRAM thực
        (8GB→32, 12GB→48, 16GB→64…) để tận dụng bộ nhớ dư → nhanh hơn rõ rệt. Dò LƯỜI một lần
        (nvidia-smi) rồi cache. Nếu lô lớn OOM, worker tự bắt lỗi và lùi per-câu.

        Returns:
            Số chunk mỗi forward (≥4).
        """
        if self._gpu_chunk_batch is None:
            from subtitles_extractor.infrastructure.tts.vieneu_batch_tuning import (
                suggest_gpu_chunk_batch,
            )

            self._gpu_chunk_batch = suggest_gpu_chunk_batch()
            logger.info(
                "VieNeu GPU: %d chunk/forward (tự chọn theo VRAM để tăng tốc batch).",
                self._gpu_chunk_batch,
            )
        return self._gpu_chunk_batch

    def _get_batched_audio(
        self, idx: int, kwargs: dict[str, Any]
    ) -> Any | None:
        """[v3.23.412] Trả audio thô cho câu ``idx`` từ CACHE batch (điền theo nhóm nếu trống).

        Chỉ dùng ở LẦN THỬ ĐẦU (attempt 0). Nếu ``idx`` chưa có trong cache, tổng hợp cả NHÓM
        ``[idx, idx+batch_size)`` bằng một lời gọi ``infer_batch`` qua worker (GPU gộp forward →
        nhanh) rồi lưu cache. Câu "dài bất thường" ở tầng trên vẫn được thử lại per-câu (attempt
        > 0 không gọi hàm này).

        Args:
            idx: Chỉ số câu hiện tại.
            kwargs: Tham số chung (temperature…) cho lô.

        Returns:
            Mảng mẫu thô, hoặc ``None`` nếu batch không khả dụng (tầng trên tự lùi per-câu).
        """
        if idx in self._batch_cache:
            return self._batch_cache.pop(idx)
        texts = self._batch_texts
        if not texts or idx >= len(texts):
            return None
        # [v3.23.593] Một lô = MỘT giọng (worker gọi infer_batch với một voice).
        # [v3.23.655] Gom TOÀN CỤC: lấy ``batch_size`` câu cùng giọng kế tiếp ở BẤT KỲ đâu phía sau (trước đây chỉ
        # trong cửa sổ ``batch_size`` câu kế tiếp → phim nhiều vai nói xen kẽ làm lô co còn 1–2 câu, GPU chạy gần
        # như từng câu). Câu sau lấy từ cache khi tới lượt.
        known_voices: list[str] = getattr(self, "_batch_voices", [])
        voices = known_voices + [""] * max(0, len(texts) - len(known_voices))
        voice = voices[idx]
        group_idx = [idx]
        for j in range(idx + 1, len(texts)):
            if len(group_idx) >= self._batch_size:
                break
            if j not in self._batch_cache and voices[j] == voice and texts[j]:
                group_idx.append(j)
        # [v3.23.655] Sắp theo độ dài chữ: GPU chia lô thành các chunk — câu dài ngắn trộn lẫn thì câu ngắn phải chờ
        # (đệm) theo câu dài nhất trong chunk.
        group_idx.sort(key=lambda j: len(texts[j]))
        group_texts = [texts[j] for j in group_idx]
        kwargs = _scale_length_caps(kwargs, texts[idx], group_texts)
        audios = self._infer_batch_via_gpu_worker(
            group_texts, {**kwargs, "voice_preset": voice} if voice else kwargs
        )
        if audios is None:
            return None
        # [v3.23.434] Cảnh báo nếu worker trả số audio LỆCH số câu gửi (contract violation).
        # zip cắt theo bên ngắn hơn — câu thiếu không được cache → tự rơi về tổng hợp per-câu
        # (an toàn, không mất ngầm), nhưng cần log để phát hiện lỗi worker.
        if len(audios) != len(group_idx):
            logger.warning(
                "Batch TTS lệch: nhận %d audio cho %d câu — câu thiếu sẽ tổng hợp lại per-câu.",
                len(audios), len(group_idx),
            )
        for j, audio in zip(group_idx, audios):  # noqa: B905 — thiếu→fallback per-câu (cố ý)
            if audio is not None:
                self._batch_cache[j] = audio
        return self._batch_cache.pop(idx, None)

    def _infer_batch_via_gpu_worker(
        self, texts: list[str], kwargs: dict[str, Any]
    ) -> list[Any] | None:
        """[v3.23.412] Tổng hợp một LÔ câu qua worker GPU (``infer_batch``), trả list audio.

        Worker ghi mỗi WAV ra một tệp tạm; đọc lại theo thứ tự. Item lỗi → ``None`` trong list
        (tầng trên lùi per-câu cho item đó). Cả lô lỗi/không dùng được → trả ``None``.

        Args:
            texts: Danh sách câu (đã làm sạch) — chung một giọng.
            kwargs: Tham số chung (temperature…).

        Returns:
            List audio (cùng thứ tự ``texts``; phần tử có thể ``None``), hoặc ``None``.
        """
        import json
        import tempfile

        if getattr(self, "_gpu_worker_disabled", False) or not texts:
            return None
        process = self._ensure_gpu_server()
        if process is None:
            return None

        with tempfile.TemporaryDirectory(prefix="subext_vieneu_batch_") as temp_dir:
            outputs = [str(Path(temp_dir) / f"b{i}.wav") for i in range(len(texts))]
            request = {
                "batch": [
                    {"text": text, "output": out}
                    for text, out in zip(texts, outputs)
                ],
                # [v3.23.420] Số chunk/forward GPU (đòn bẩy VRAM). Tự chọn theo VRAM lần đầu.
                "gpu_subbatch": self._resolve_gpu_chunk_batch(),
                **{
                    name: kwargs[name]
                    for name in (
                        "max_new_frames", "repetition_penalty", "temperature",
                        "voice_preset",
                    )
                    if name in kwargs
                },
            }
            try:
                assert process.stdin is not None and process.stdout is not None
                process.stdin.write(_json_line_safe(request))
                process.stdin.flush()
                reply_line = process.stdout.readline()
            except (OSError, ValueError, AssertionError) as exc:
                logger.warning("Mất liên lạc với worker GPU (batch): %s", exc)
                self._shutdown_gpu_server()
                return None
            if not reply_line:
                logger.warning("Worker GPU thoát bất ngờ (batch).")
                self._shutdown_gpu_server()
                return None
            try:
                reply = json.loads(reply_line)
            except ValueError:
                return None
            if not reply.get("ok"):
                logger.warning(
                    "Worker GPU không hỗ trợ/thất bại batch (%s) — lùi per-câu.",
                    str(reply.get("error") or "?"),
                )
                return None

            written = reply.get("batch") or []
            try:
                import soundfile as sf
            except ImportError:
                return None
            results: list[Any] = []
            for out in outputs:
                path = out if out in written and Path(out).is_file() else None
                if path is None:
                    results.append(None)
                    continue
                try:
                    samples, _sr = sf.read(path, dtype="float32")
                    results.append(samples)
                except (OSError, RuntimeError):
                    results.append(None)
            return results

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
            process.stdin.write(_json_line_safe(config))
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
        # [v3.23.401] Nếu là lỗi transformers-quá-mới (hint khác rỗng), TỰ SỬA 1 lần ở nền:
        # force-hạ transformers khớp torch trong whisperx_env → phiên SAU dùng được GPU mà
        # người dùng KHÔNG phải gõ lệnh tay hay xoá môi trường.
        if hint:
            self._maybe_auto_repair_transformers()

    def _maybe_auto_repair_transformers(self) -> None:
        """[v3.23.401] Tự hạ transformers khớp torch trong whisperx_env — CHẠY TỐI ĐA 1 LẦN.

        Chạy pip ở luồng nền (không chặn TTS đang chạy CPU). Dùng file mốc trong whisperx_env
        để KHÔNG lặp (tránh vòng lặp sửa nếu vẫn lỗi). Có hiệu lực ở lần khởi động app kế tiếp.
        """
        import threading

        try:
            from subtitles_extractor.infrastructure.tts.vieneu_gpu_plan import (
                VIENEU_TRANSFORMERS_SPEC,
                build_gpu_tts_plan,
                build_transformers_repair_command,
            )

            plan = build_gpu_tts_plan()
            python_exe = plan.python_exe
            if not python_exe:
                return
            marker = Path(python_exe).resolve().parent.parent / ".vieneu_tf_repaired"
            # [v3.23.402] Mốc chứa CHÍNH spec đã sửa. Nếu app cập nhật pin (vd hạ thấp hơn vì
            # 4.53 vẫn lỗi), spec MỚI khác nội dung mốc → auto-repair CHẠY LẠI với pin mới,
            # KHÔNG bị kẹt vĩnh viễn. Chỉ bỏ qua khi đã sửa với ĐÚNG spec hiện tại.
            if marker.exists():
                try:
                    if marker.read_text(encoding="utf-8").strip() == VIENEU_TRANSFORMERS_SPEC:
                        return
                except OSError:
                    pass
        except (ImportError, OSError):
            return

        command = build_transformers_repair_command(python_exe)

        def _run_repair() -> None:
            try:
                # [v3.23.403] Ghi phiên bản torch + transformers HIỆN TẠI để chẩn đoán chính
                # xác ngưỡng tương thích (nếu pin vẫn chưa đủ thấp, log này chỉ ra torch nào).
                try:
                    probe = subprocess.run(
                        [
                            python_exe, "-c",
                            "import torch,transformers;"
                            "print('torch',torch.__version__,'| transformers',"
                            "transformers.__version__)",
                        ],
                        capture_output=True, text=True, encoding="utf-8",
                        errors="replace", timeout=120, check=False,
                        **no_window_kwargs(),
                    )
                    logger.info(
                        "VieNeu chẩn đoán phiên bản (trước khi hạ): %s",
                        (probe.stdout or probe.stderr or "?").strip(),
                    )
                except (OSError, subprocess.TimeoutExpired):
                    pass

                logger.info("VieNeu: tự hạ transformers khớp torch (nền)…")
                result = subprocess.run(
                    command, capture_output=True, text=True,
                    encoding="utf-8", errors="replace", timeout=900,
                    check=False, **no_window_kwargs(),
                )
                if result.returncode == 0:
                    marker.write_text(VIENEU_TRANSFORMERS_SPEC, encoding="utf-8")
                    logger.info(
                        "VieNeu: đã hạ transformers xong — KHỞI ĐỘNG LẠI app để dùng GPU cho TTS."
                    )
                else:
                    logger.warning(
                        "VieNeu: tự hạ transformers thất bại: %s",
                        (result.stderr or result.stdout or "")[-300:],
                    )
            except (OSError, subprocess.TimeoutExpired) as exc:
                logger.warning("VieNeu: tự hạ transformers gặp lỗi: %s", exc)

        threading.Thread(target=_run_repair, name="vieneu-tf-repair", daemon=True).start()

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

    @without_torch_block
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
