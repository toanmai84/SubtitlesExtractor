"""Adapter :class:`OcrEnginePort` dựa trên PaddleOCR.

BẢN CẬP NHẬT v3.29 (HOTFIX):
    * [CRITICAL BUG FIX]: Vá lỗi `ValueError: not enough values to unpack` trong
      hàm `infer_batch` do thiếu tham số `prepared_images` truyền vào hàm `zip()`.
      Lỗi này gây sập hoàn toàn tiến trình Trích xuất phụ đề.
"""

from __future__ import annotations

import dataclasses
import gc
import logging
import os
import re
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Final

import cv2
import numpy as np

from subtitles_extractor.domain.entities.ocr_frame_result import (
    OcrFrameResult,
    OcrTextBox,
)
from subtitles_extractor.domain.exceptions import OcrInferenceError, OcrModelLoadError
from subtitles_extractor.domain.ports.ocr_engine_port import OcrEngineConfig
from subtitles_extractor.domain.value_objects.confidence import Confidence
from subtitles_extractor.domain.value_objects.device_kind import DeviceKind, PrecisionMode
from subtitles_extractor.domain.value_objects.ocr_model_names import (
    PADDLE_VL_DRAFT_RECOGNITION_MODEL,
    is_vl_recognizer,
)
from subtitles_extractor.infrastructure.ocr.device_probe import should_use_gpu
from subtitles_extractor.infrastructure.ocr.paddle_vl_process import make_vl_reader
from subtitles_extractor.infrastructure.ocr.paddle_vl_reader import PaddleVlLineReader
from subtitles_extractor.infrastructure.ocr.preprocessing.image_filters import (
    apply_cpu_preprocessing_pipeline,
)
from subtitles_extractor.infrastructure.ocr.result_parser import parse_paddle_result
from subtitles_extractor.infrastructure.ocr.script_group_versions import (
    resolve_script_group_version,
)
from subtitles_extractor.infrastructure.ocr.unreadable_rescue import (
    RESCUE_MAP,
    RESCUE_RECOGNITION_MODEL,
    apply_rescues,
    best_reading,
    crop_rect,
    detected_polygons,
    find_rescue_candidates,
)

logger = logging.getLogger(__name__)

#: [v3.23.733] PaddleOCR-VL không trả độ tin cậy: bản đọc VL khác bản nháp nhận mức này; trùng bản nháp thì ít nhất mức kia.
VL_CONFIDENCE = 0.90
VL_AGREE_CONFIDENCE = 0.97
#: Bản nháp tin cậy từ mức này mới dùng làm khoá bộ đệm bản đọc VL (cùng chữ, cùng chỗ → cùng hình).
VL_CACHE_MIN_CONFIDENCE = 0.90
VL_CACHE_SIZE = 2048


def _pipeline_rec_name(name: str | None) -> str | None:
    return PADDLE_VL_DRAFT_RECOGNITION_MODEL if is_vl_recognizer(name) else name


def _overlap_ratio(a: Any, b: Any) -> float:
    ax = [float(p[0]) for p in a]; ay = [float(p[1]) for p in a]
    bx = [float(p[0]) for p in b]; by = [float(p[1]) for p in b]
    inter = max(0.0, min(max(ax), max(bx)) - max(min(ax), min(bx))) * max(0.0, min(max(ay), max(by)) - max(min(ay), min(by)))
    smaller = min((max(ax) - min(ax)) * (max(ay) - min(ay)), (max(bx) - min(bx)) * (max(by) - min(by)))
    return inter / smaller if smaller > 0 else 0.0


def _same_letters(left: str | None, right: str | None) -> bool:
    return "".join(ch for ch in (left or "") if ch.isalnum()) == "".join(ch for ch in (right or "") if ch.isalnum())

#: [v3.23.388] Thông điệp khi thiếu lõi paddle — hướng người dùng tới nút tải trong Cài đặt.
_PADDLE_MISSING_MSG = (
    "Thiếu lõi OCR (paddlepaddle-gpu). Vào Cài đặt → bấm \"Tải lõi OCR (paddle)\" để tải "
    "(một lần, ~810MB), rồi khởi động lại ứng dụng."
)

_MAX_INIT_RETRY: Final[int] = 6
_FALLBACK_BLACK_IMAGE_SIZE: Final[int] = 64
_UNEXPECTED_KWARG_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"unexpected keyword argument '([^']+)'"),
    re.compile(r"got an unexpected keyword argument \"([^\"]+)\""),
    re.compile(r"Unknown argument:\s*([a-zA-Z0-9_]+)"),
)

# [v3.23.296] Các giá trị coi là "bật" cho cờ môi trường boolean.
_TRUTHY_ENV_VALUES: Final[frozenset[str]] = frozenset({"1", "true", "yes", "on"})


def _env_flag_enabled(name: str) -> bool:
    """True nếu biến môi trường ``name`` được đặt về một giá trị "bật".

    Args:
        name: Tên biến môi trường (vd ``"SUBEXT_FORCE_CPU"``).

    Returns:
        True nếu giá trị (không phân biệt hoa/thường, đã strip) nằm trong
        ``{"1", "true", "yes", "on"}``; ngược lại False (kể cả khi chưa đặt).
    """
    return os.environ.get(name, "").strip().lower() in _TRUTHY_ENV_VALUES


def _chan_doan_paddle_deps(error_text: str) -> str:
    """Liệt kê dependency paddlex thiếu (khi lỗi 'dependency error' lúc đóng gói).

    paddlex báo lỗi chung chung không cho biết dep nào thiếu. Hàm này gọi paddlex kiểm từng
    required dep + báo cái nào ``get_dep_version`` trả None (thường do PyInstaller thiếu
    metadata dist-info). Trả chuỗi mô tả để nối vào thông báo lỗi (rỗng nếu không chẩn được).
    """
    if "dependency" not in error_text.lower():
        return ""
    try:
        from paddlex.utils import deps
    except ImportError:
        return ""

    missing: list[str] = []
    try:
        # Kiểm base required deps
        deps_to_check = list(deps.REQUIRED_DEP_SPECS)
        # [v3.23.286] Kiểm THÊM dep nhóm mở rộng ocr-core/ocr (paddlex kiểm khi tạo pipeline
        # OCR). Chẩn đoán v285 cho thấy lỗi ở gói mở rộng, không phải base.
        try:
            extras = deps._get_extras()
            # paddleocr mặc định chỉ dùng paddlex[ocr-core] — chỉ kiểm nhóm này.
            for spec in extras.get("ocr-core", []):
                name = spec.split(">")[0].split("<")[0].split("=")[0].split("[")[0].strip()
                if name and name not in deps_to_check:
                    deps_to_check.append(name)
        except (AttributeError, KeyError, TypeError):
            pass

        for dep in deps_to_check:
            try:
                if not deps.is_dep_available(dep):
                    missing.append(f"{dep}(version={deps.get_dep_version(dep)})")
            except (ValueError, KeyError, ImportError):
                missing.append(f"{dep}(check_lỗi)")
    except (AttributeError, TypeError):
        return ""

    if not missing:
        return " [Chẩn đoán: required deps đều có — lỗi có thể ở gói mở rộng paddlex.]"
    return (
        " [Chẩn đoán: paddlex THIẾU dependency (thường do đóng gói thiếu metadata "
        f"dist-info): {', '.join(missing)}.]"
    )


@dataclasses.dataclass(frozen=True, slots=True)
class _PreparedImage:
    image_bgr: np.ndarray
    scale: float
    border_px: int

#: [v3.23.646] Thư viện khoa học mà paddle/paddlex (và các phụ thuộc của chúng) có thể kéo vào lúc import.
_SCIENTIFIC_PREWARM_MODULES: Final[tuple[str, ...]] = ("scipy.stats", "scipy.signal", "sklearn.cluster")

#: Module lõi native của paddle (đồng thời là tiền tố các module con do native tự đăng ký).
_NATIVE_PADDLE_PREFIX: Final[str] = "paddle.base.libpaddle"


def _prewarm_scientific_imports() -> None:
    """[v3.23.646] Nạp scipy/sklearn TRƯỚC khi nạp paddle, lúc ``sys.modules['torch']`` không còn là ``None``.

    Bộ chặn torch (``sys.modules["torch"] = None``) làm ``scipy.stats``/``scipy.signal``/``sklearn.cluster``
    chết ngay khi nạp lần đầu: ``AttributeError: 'NoneType' object has no attribute 'Tensor'``
    (``array_api_compat`` dò kiểu tensor bằng ``getattr(sys.modules["torch"], "Tensor")``). Trước v3.23.643
    ``bbox_analyzer`` import sklearn ngay lúc khởi động nên các thư viện này đã nạp xong từ trước khi bộ chặn
    bật; v3.23.643 chuyển sang nạp trễ ⇒ lần nạp đầu rơi vào giữa ``import paddle``. Lỗi gốc lại bị che bởi
    nhánh "dọn rồi thử lại" (xoá luôn module con native ``paddle.base.libpaddle.pir`` mà chỉ thư viện native
    mới đăng ký được) nên người dùng chỉ thấy "Thiếu lõi OCR … 'libpaddle' is not a package".
    Chỉ nạp thư viện nào CÓ cài; lỗi nạp không chặn OCR (paddle có thể không cần chúng).
    """
    import importlib
    import importlib.util

    from subtitles_extractor.infrastructure.torch_import_blocker import (
        torch_none_bypassed,
    )

    with torch_none_bypassed():
        for name in _SCIENTIFIC_PREWARM_MODULES:
            root = name.split(".", 1)[0]
            try:
                if importlib.util.find_spec(root) is None:
                    continue
                importlib.import_module(name)
            except Exception as exc:  # noqa: BLE001 - thư viện hỏng ABI có thể ném nhiều loại ngoại lệ
                logger.debug("Nạp trước %s không được: %s", name, exc)


def _native_paddle_loaded() -> bool:
    """Lõi native ``paddle.base.libpaddle`` đã nạp (dù ``import paddle`` sau đó lỗi) chưa?"""
    import sys

    return _NATIVE_PADDLE_PREFIX in sys.modules


def _purge_partial_paddle_modules() -> int:
    """Xoá module Python của paddle nửa-khởi-tạo để nạp lại. CHỈ dùng khi lõi native CHƯA nạp.

    Khi ``paddle.base.libpaddle`` đã nạp thì nạp lại KHÔNG cứu được: module con native
    (``.pir``, ``.eager``…) do chính thư viện tự đăng ký nên xoá đi là mất
    (``No module named 'paddle.base.libpaddle.pir'; … is not a package``), còn giữ lại thì chạy lại mã Python
    của paddle đập vào đăng ký toàn cục của native (``Pass '…' is registered more than once``) hoặc của protobuf
    (``duplicate file name``) — đã thử trên paddle 3.3.1. Trường hợp đó phải báo LỖI GỐC của lần nạp đầu.

    Returns:
        Số module đã xoá.
    """
    import sys

    removed = 0
    for name in list(sys.modules):
        if name == "paddle" or name.startswith("paddle."):
            del sys.modules[name]
            removed += 1
    return removed


def _rearm_fault_handler() -> None:
    """[v3.23.650] Paddle đè trình xử lý crash native của faulthandler khi nạp — cài lại để fault.log ghi được."""
    try:
        from subtitles_extractor.infrastructure.logging.crash_hooks import rearm

        rearm()
    except Exception:  # noqa: BLE001 - chẩn đoán không được làm hỏng việc nạp OCR
        pass


def pad_to_common_shape(images: list[np.ndarray]) -> list[np.ndarray]:
    """[v3.23.748] Ảnh trong một lô khác cỡ → đệm ĐEN ở cạnh phải / dưới cho bằng ảnh lớn nhất (hàm thuần).

    Bộ dò chữ chạy ``det_batch_size`` ảnh mỗi lần (máy chủ dự án: 64) ghép lô bằng ``np.stack`` → ảnh cắt sát từng câu
    của gói kiểm định (3.23.744, mỗi câu một cỡ) làm hỏng cả gói: "all input arrays must have the same shape" (log
    19/11 00:16). Gốc toạ độ giữ nguyên → hộp chữ trả về đúng chỗ; vùng đệm đen không có chữ. Cùng cỡ: trả nguyên.
    """
    if len({image.shape for image in images}) <= 1:
        return images
    height = max(image.shape[0] for image in images)
    width = max(image.shape[1] for image in images)
    padded = []
    for image in images:
        if image.shape[0] == height and image.shape[1] == width:
            padded.append(image)
            continue
        canvas = np.zeros((height, width, *image.shape[2:]), dtype=image.dtype)
        canvas[:image.shape[0], :image.shape[1]] = image
        padded.append(canvas)
    return padded


#: [v3.23.765] Lưới an toàn hình dạng TensorRT của PaddleX đã cài (tự chia lô) → không cần ép lô nhỏ ở bộ chuyển đổi.
_TRT_PROFILE_PATCHED = False

#: [v3.23.767] LUỒNG PADDLE cố định cho TensorRT. Paddle giữ trạng thái riêng từng luồng: kiểm bằng paddle thật — CHỈ luồng
#: đã nạp paddle chạy được ``paddle.static.io.load_inference_model`` (bước chuyển TensorRT); luồng khác, kể cả luồng chính
#: Python, lỗi "only supported in static graph mode" (đúng lỗi tự kiểm 3.23.766, log 19/11 16:32). Ứng dụng nạp paddle ở
#: luồng NẠP SẴN (kết thúc ngay sau đó) rồi chạy OCR TensorRT ở luồng TRÍCH XUẤT — bố cục của mọi lần "access violation"
#: (log 14:46, 15:46); GPU thường thì Paddle chịu được khác luồng. Khi bật TensorRT: nạp paddle + dựng + chạy + huỷ mô hình
#: TensorRT đều trên luồng này (sống suốt phiên).
_PADDLE_IMPORT_THREAD: int | None = None
_PADDLE_THREAD: Any = None
_PADDLE_THREAD_IDENT: int | None = None
_PADDLE_THREAD_GUARD = threading.Lock()


class _DaemonWorker:
    """Một luồng NỀN (daemon) nhận việc qua hàng đợi, trả ``Future`` — như ``ThreadPoolExecutor(1)`` nhưng không giữ
    tiến trình sống khi đóng ứng dụng (rà độc lập 3.23.767: luồng của ``ThreadPoolExecutor`` được ``join`` không hạn lúc
    thoát → đóng cửa sổ giữa lúc tự kiểm / dựng engine thì exe chạy ngầm tới vài chục phút)."""

    def __init__(self, name: str) -> None:
        import queue

        self._queue: Any = queue.SimpleQueue()
        self._thread = threading.Thread(target=self._loop, name=name, daemon=True)
        self._thread.start()

    @property
    def ident(self) -> int | None:
        return self._thread.ident

    def _loop(self) -> None:
        while True:
            future, work, args, kwargs = self._queue.get()
            if not future.set_running_or_notify_cancel():
                continue
            try:
                future.set_result(work(*args, **kwargs))
            except BaseException as exc:  # noqa: BLE001 — chuyển về luồng gọi
                future.set_exception(exc)
            finally:
                del work, args, kwargs              # không giữ tham chiếu (vd mô hình cần huỷ) sau khi xong việc

    def submit(self, work: Any, *args: Any, **kwargs: Any) -> Any:
        from concurrent.futures import Future

        future: Any = Future()
        self._queue.put((future, work, args, kwargs))
        return future


def _paddle_thread() -> Any:
    global _PADDLE_THREAD, _PADDLE_THREAD_IDENT
    with _PADDLE_THREAD_GUARD:
        if _PADDLE_THREAD is None:
            worker = _DaemonWorker("paddle-tensorrt")
            _PADDLE_THREAD_IDENT = worker.submit(threading.get_ident).result()
            _PADDLE_THREAD = worker
        return _PADDLE_THREAD


def _run_on_paddle_thread(work: Any, *args: Any, **kwargs: Any) -> Any:
    """Chạy ``work`` trên luồng Paddle (chặn tới khi xong; đang ở luồng đó thì gọi thẳng — không tự khoá chết)."""
    executor = _paddle_thread()
    if threading.get_ident() == _PADDLE_THREAD_IDENT:
        return work(*args, **kwargs)
    return executor.submit(work, *args, **kwargs).result()


def _legacy_threads() -> bool:
    """[v3.23.768] Bố cục luồng CŨ (nạp + dựng ở luồng gọi, OCR ở luồng gọi khác) — CHỈ tiến trình tự kiểm đặt
    ``SE_TRT_LEGACY_THREADS=1`` để chẩn đoán so sánh; ứng dụng luôn dùng luồng Paddle."""
    return os.environ.get("SE_TRT_LEGACY_THREADS") == "1"


def _tensorrt_thread_ok() -> bool:
    """TensorRT chỉ chạy khi paddle được nạp CHÍNH trên luồng Paddle và đang ở luồng đó (tách hàm để kiểm thử thay)."""
    if _legacy_threads():
        return _PADDLE_IMPORT_THREAD == threading.get_ident()
    return _PADDLE_IMPORT_THREAD is not None and _PADDLE_IMPORT_THREAD == _PADDLE_THREAD_IDENT == threading.get_ident()


def _prepare_paddle_tensorrt() -> Path | None:
    """[v3.23.756] Trước khi dựng mô hình TensorRT: vá chỗ Paddle gọi cứng tên thư viện Linux (xem
    ``tensorrt_runtime_plan.patch_paddle_tensorrt_plugin_loader``) nếu TensorRT đã tải về models/tensorrt_runtime."""
    try:
        from subtitles_extractor.infrastructure.model_store import (
            ensure_model_store_root,
            model_store_root,
        )
        from subtitles_extractor.infrastructure.ocr.tensorrt_runtime_plan import (
            TENSORRT_RUNTIME_DIRNAME,
            install_paddlex_trt_profile_patch,
            is_tensorrt_installed,
            patch_paddle_tensorrt_plugin_loader,
            patch_tensorrt_numpy_bridge,
        )

        trt_dir: Path | None = None
        for root in (model_store_root(), ensure_model_store_root()):
            if root is not None and is_tensorrt_installed(root / TENSORRT_RUNTIME_DIRNAME):
                trt_dir = root / TENSORRT_RUNTIME_DIRNAME
                patch_paddle_tensorrt_plugin_loader(trt_dir)
                break
        # [v3.23.758] Bindings TensorRT 10.x chưa hỗ trợ NumPy 2 ("NumPy data type: i0", ``Weights.numpy()`` sai dữ liệu).
        patch_tensorrt_numpy_bridge()
        # [v3.23.765] Engine bộ dò lô 8 + lưới an toàn hình dạng (chia lô / lỗi Python thay vì sập).
        global _TRT_PROFILE_PATCHED
        _TRT_PROFILE_PATCHED = install_paddlex_trt_profile_patch()
        if trt_dir is not None and not _TRT_PROFILE_PATCHED:
            logger.warning("Không cài được lưới an toàn hình dạng TensorRT cho PaddleX — không dùng TensorRT.")
            return None
        return trt_dir
    except Exception as exc:  # noqa: BLE001 — chuẩn bị phụ: lỗi thì bước dựng TensorRT tự lùi về GPU thường
        logger.debug("Chuẩn bị TensorRT cho paddle lỗi: %s", exc)
        return None


#: [v3.23.763] Dấu hiệu lỗi suy luận do TensorRT (thư viện / engine), không phải lỗi dữ liệu hay hết VRAM.
_TENSORRT_RUNTIME_MARKERS: tuple[str, ...] = ("tensorrt", "nvinfer", "trt engine", "tensorrt_engine")


def _is_tensorrt_runtime_error(error: BaseException) -> bool:
    text = " ".join(str(item) for item in (error, error.__cause__) if item is not None).lower()
    return any(marker in text for marker in _TENSORRT_RUNTIME_MARKERS)


def _tensorrt_selftest(trt_dir: Path, init_kwargs: dict[str, Any], det_batch: int, config: Any = None) -> Any:
    """[v3.23.766] Xem :mod:`~subtitles_extractor.infrastructure.ocr.tensorrt_selftest` (tách hàm để kiểm thử thay được)."""
    from subtitles_extractor import __version__
    from subtitles_extractor.infrastructure.ocr.tensorrt_selftest import (
        ensure_tensorrt_selftest,
    )

    return ensure_tensorrt_selftest(trt_dir, __version__, init_kwargs, det_batch, config=config)


def _det_profile_max_batch() -> int:
    from subtitles_extractor.infrastructure.ocr.tensorrt_runtime_plan import (
        TRT_DET_PROFILE,
    )

    return int(next(iter(TRT_DET_PROFILE.values()))[2][0])


def _tensorrt_crashed_last_time(trt_dir: Path) -> bool:
    from subtitles_extractor import __version__
    from subtitles_extractor.infrastructure.ocr.tensorrt_runtime_plan import (
        crashed_on_previous_first_inference,
    )

    return crashed_on_previous_first_inference(trt_dir, __version__)


def _begin_tensorrt_first_inference(trt_dir: Path | None) -> None:
    from subtitles_extractor import __version__
    from subtitles_extractor.infrastructure.ocr.tensorrt_runtime_plan import (
        begin_first_inference,
    )

    # [v3.23.768] Tiến trình tự kiểm không ghi dấu chống sập: bố cục chẩn đoán sập (dự kiến) không được tắt TensorRT oan.
    if trt_dir is not None and os.environ.get("SE_TRT_SELFTEST_CHILD") != "1":
        begin_first_inference(trt_dir, __version__)


def _end_tensorrt_first_inference(trt_dir: Path | None) -> None:
    from subtitles_extractor.infrastructure.ocr.tensorrt_runtime_plan import (
        end_first_inference,
    )

    if trt_dir is not None:
        end_first_inference(trt_dir)


def _log_tensorrt_failure_details(error: BaseException) -> None:
    """[v3.23.760] Ghi lỗi gốc của TensorRT + chỗ Paddle vấp (traceback rút gọn) để lần chạy sau sửa được đúng chỗ."""
    try:
        import traceback

        from subtitles_extractor.infrastructure.ocr.tensorrt_runtime_plan import (
            tensorrt_recent_errors,
        )

        errors = tensorrt_recent_errors()
        if errors:
            logger.warning("TensorRT báo lỗi trước đó: %s", " | ".join(errors[-8:]))
        cause = error.__cause__ or error
        frames = traceback.format_exception(type(cause), cause, cause.__traceback__)
        logger.info("Chi tiết lỗi dựng TensorRT:\n%s", "".join(frames[-16:]).rstrip())
    except Exception as exc:  # noqa: BLE001 — chỉ là chẩn đoán
        logger.debug("Không ghi được chi tiết lỗi TensorRT: %s", exc)


def vram_summary() -> str:
    """[v3.23.746] "VRAM trống X/Y GB" của GPU hiện tại (CuPy); rỗng nếu không đo được (máy không GPU)."""
    try:
        import cupy  # type: ignore[import-not-found,unused-ignore]

        free, total = cupy.cuda.runtime.memGetInfo()
    except Exception:  # noqa: BLE001 — chỉ để ghi nhật ký
        return ""
    return f"VRAM trống {free / 2**30:.1f}/{total / 2**30:.1f} GB"


class PaddleOcrAdapter:
    def __init__(self, config: OcrEngineConfig) -> None:
        self._config: OcrEngineConfig = config
        self._model: Any | None = None
        # [v3.23.295] Thiết bị THỰC TẾ đang chạy (cập nhật sau initialize; có thể khác
        # config nếu tự lùi GPU->CPU). None = chưa khởi tạo.
        self._active_device: DeviceKind | None = None
        self._init_lock: threading.RLock = threading.RLock()
        self._release_after_init: bool = False
        # [v3.23.656] B1/B2: cấu hình MẶC ĐỊNH (Cài đặt) tách khỏi cấu hình đã DỰNG model (``_config``). Mỗi lượt chạy
        # truyền cấu hình của nó vào ``infer_batch(config=…)``; trước đây lượt nào bắt đầu cũng ``apply_config`` lên
        # engine dùng chung → Re-OCR chạy giữa lúc trích xuất âm thầm đổi ngưỡng/tiền xử lý của lượt trích xuất (đo:
        # 17 → 4 ô chữ trên cùng 40 khung), đổi model thì giải phóng model giữa chừng.
        self._default_config: OcrEngineConfig = config
        #: Một lô suy luận tại một thời điểm; nạp lại model (khi lượt cần model khác) chỉ xảy ra trong khoá này.
        self._infer_lock: threading.Lock = threading.Lock()
        self._release_after_infer: bool = False
        # [v3.23.732] Bộ nhận dạng phụ đọc lại hộp không đọc được ("诶" — không có trong bảng chữ của mọi mô hình).
        self._rescue_model: Any | None = None
        self._rescue_device: str | None = None
        self._rescue_unavailable: bool = False
        # [v3.23.733] Bộ đọc dòng PaddleOCR-VL (chạy chính khi chọn phiên bản "PaddleOCR-VL-1.6", và lượt soát câu).
        # [v3.23.751] Windows: PaddleOCR-VL ở TIẾN TRÌNH RIÊNG (lệnh GPU của VL kẹt từng kéo cả tiến trình treo theo).
        self._vl_reader: PaddleVlLineReader = make_vl_reader()
        #: [v3.23.754] TensorRT đã thử và lỗi trong phiên này → các lần dựng mô hình sau chạy GPU không TensorRT.
        self._tensorrt_unavailable = False
        #: [v3.23.763] Mô hình đang nạp được dựng bằng TensorRT (để biết lỗi suy luận có phải của TensorRT không).
        self._model_uses_trt = False
        #: [v3.23.774] Mô hình GPU thường + runner TensorRT trực tiếp (ứng dụng chạy engine).
        self._model_uses_direct_trt = False
        #: [v3.23.764] Thư mục TensorRT (dấu chống sập) + lần suy luận TensorRT đầu tiên của phiên đã chạy xong.
        self._trt_dir: Path | None = None
        self._trt_first_inference_done = False
        self._vl_unavailable: bool = False
        self._vl_error: str | None = None
        self._vl_cache: OrderedDict[tuple[Any, ...], str] = OrderedDict()
        #: [v3.23.758] Bộ đọc dòng THAY PaddleOCR-VL trong bước soát câu (GLM-OCR theo "Bộ đọc kiểm định"); ``None`` = VL.
        self._line_reader_override: Any | None = None
        #: [v3.23.759] Luồng đã gắn bộ đọc thay thế: chỉ lượt soát câu ĐÓ dùng nó (Re-OCR chạy song song ở luồng khác vẫn VL).
        self._line_reader_thread: int | None = None
        self._line_reader_error: str | None = None

    def set_line_reader(self, reader: Any | None) -> None:
        """[v3.23.758] Đặt (``reader``) / bỏ (``None``) bộ đọc dòng thay PaddleOCR-VL — cùng giao diện ``read(ảnh BGR,
        device)``. Bộ đệm bản đọc VL bị xoá cả hai chiều (không trộn chữ của hai mô hình)."""
        with self._infer_lock:
            self._line_reader_override = reader
            self._line_reader_thread = threading.get_ident() if reader is not None else None
            self._line_reader_error = None
            self._vl_cache.clear()

    @property
    def line_reader_error(self) -> str | None:
        """[v3.23.758] Lý do bộ đọc dòng thay thế (GLM-OCR) bị bỏ giữa chừng (đã lùi về PaddleOCR-VL); ``None`` = ổn."""
        return self._line_reader_error

    @property
    def vl_error(self) -> str | None:
        """[v3.23.739] Lý do PaddleOCR-VL không chạy trong phiên (``None`` = chưa lỗi) — để dữ liệu không mang nhãn sai."""
        return self._vl_error

    @property
    def is_initialized(self) -> bool:
        # [v3.23.650] KHÔNG lấy ``_init_lock``: ``initialize()`` giữ khoá suốt lúc import paddle + nạp mô hình (7–19 s),
        # mà thuộc tính này được gọi từ luồng giao diện (showEvent của trang Trích xuất/Chỉnh sửa, preload) →
        # cửa sổ đơ tới khi mô hình nạp xong. Đọc một tham chiếu là nguyên tử trong CPython.
        return self._model is not None

    @property
    def acceleration_description(self) -> str:
        """[v3.23.777] Đường OCR THẬT SỰ chạy, ghi vào tệp mô tả lượt chạy (CLAUDE.md: ghi rõ đường đã chạy). Tệp của lượt
        cả bộ 20/11 (bản 3.23.776) không cho biết TensorRT có chạy hay đã lùi GPU thường."""
        if self._model is None or self._active_device is None:
            return "chưa khởi tạo"
        if self._active_device != DeviceKind.GPU:
            return "cpu"
        precision = "fp16" if self._config.precision == PrecisionMode.FP16 else "fp32"
        if getattr(self, "_model_uses_direct_trt", False):
            parts = []
            for name, label in (("text_det_model", "bộ dò"), ("text_rec_model", "bộ nhận dạng")):
                runner = self._predictor_runner(name)
                engine = getattr(runner, "_engine", None)
                limit = getattr(engine, "max_shape", [None])[0]
                parts.append(f"{label} ≤{limit}: {int(getattr(runner, 'calls', 0) or 0)} lô TensorRT, "
                             f"{int(getattr(runner, 'fallbacks', 0) or 0)} lô lùi Paddle GPU")
            return f"gpu+tensorrt_direct {precision} (" + "; ".join(parts) + ")"
        if self._model_uses_trt:
            return f"gpu+tensorrt_paddle {precision}"
        if self._config.use_tensorrt and getattr(self, "_tensorrt_unavailable", False):
            return "gpu (TensorRT đã bật nhưng không dùng được — xem nhật ký)"
        return "gpu"

    def _predictor_runner(self, name: str) -> Any:
        try:
            return getattr(getattr(self._model.paddlex_pipeline._pipeline, name, None), "runner", None)  # type: ignore[union-attr]
        except AttributeError:
            return None

    @property
    def active_device(self) -> DeviceKind | None:
        """Thiết bị THỰC TẾ OCR đang chạy sau khi khởi tạo.

        Returns:
            :class:`DeviceKind` (GPU/CPU) sau ``initialize``; ``None`` nếu chưa khởi
            tạo. Có thể là CPU dù config đặt GPU (khi máy không NVIDIA hoặc GPU init lỗi).
        """
        return self._active_device

    def initialize(self) -> None:
        with self._init_lock:
            self._release_after_init = False  # yêu cầu release() từ lần trước là cũ; chỉ giữ yêu cầu đến TRONG lúc này
            if self._model is None:
                if self._wants_tensorrt():
                    _run_on_paddle_thread(self._initialize_model)      # [v3.23.767] xem _PADDLE_THREAD
                else:
                    self._initialize_model()
        if self._release_after_init:
            # [v3.23.650] release() được gọi (đóng app) lúc khoá đang bận: dọn ngay khi nạp xong (kể cả đường thoát sớm).
            self._release_after_init = False
            self.release()

    def _wants_tensorrt(self) -> bool:
        return bool(self._config.use_tensorrt) and not getattr(self, "_tensorrt_unavailable", False) and (
            self._config.device != DeviceKind.CPU) and not _legacy_threads()

    def _initialize_model(self) -> None:
        paddle_module, paddle_ocr_cls = self._import_paddle()
        use_gpu: bool = self._resolve_use_gpu(paddle_module)
        self._model = self._create_model_with_device_fallback(paddle_ocr_cls, use_gpu=use_gpu)
        self._apply_det_batch_size(self._model)
        _rearm_fault_handler()

    def _apply_det_batch_size(self, model: Any) -> None:
        """[v3.23.644] Cho bộ DÒ CHỮ chạy ``det_batch_size`` ảnh mỗi lần GPU thay vì từng ảnh một.

        PaddleOCR 3.x nhận cả lô ``predict(input=[...])`` nhưng ``_OCRPipeline`` và bộ dò chữ đều dựng bộ chia lô
        với kích thước MẶC ĐỊNH 1 — ``text_recognition_batch_size`` / ``batch_size_ocr`` không ảnh hưởng tới bước dò.
        Đo (PaddleOCR + CPU, 10 ảnh 603×119): mặc định → 10 lượt det × 1 ảnh; đặt hai bộ chia lô = 10 → 1 lượt det ×
        10 ảnh, chữ nhận dạng GIỐNG HỆT. Trên CPU không nhanh hơn (tính toán, không phải độ trễ lần chạy); lợi ích
        trên GPU chưa đo được — dùng ``tools/bench_ocr.py``. Chạm vào thuộc tính nội bộ nên mọi lỗi cấu trúc chỉ
        cảnh báo và giữ mặc định.
        """
        size = int(getattr(self._config, "det_batch_size", 1) or 1)
        if model is not None and self._model_uses_trt and not _TRT_PROFILE_PATCHED:
            size = self._cap_batches_for_tensorrt(model, size)
        elif model is not None and self._model_uses_trt:
            logger.info("TensorRT: bộ dò %d ảnh/lô — lô vượt engine tự chia lô con (engine bộ dò tối đa %d ảnh).", size,
                        _det_profile_max_batch())
        if size <= 1 or model is None:
            return
        try:
            pipeline = model.paddlex_pipeline._pipeline
            pipeline.batch_sampler.batch_size = size
            pipeline.text_det_model.batch_sampler.batch_size = size
        except (AttributeError, TypeError, ValueError) as exc:
            logger.warning(
                "Không đặt được batch bộ dò chữ = %d (cấu trúc PaddleOCR khác dự kiến: %s) — dùng mặc định 1.",
                size, exc,
            )
            return
        logger.info("Bộ dò chữ PaddleOCR chạy %d ảnh mỗi lần.", size)

    def _attach_direct_tensorrt(self, model: Any, expected: Any = None) -> None:
        """[v3.23.774] Gắn runner TensorRT trực tiếp (``trt_direct_runner``) cho bộ dò + bộ nhận dạng; lỗi → GPU thường.
        ``expected``: dấu tệp engine lần tự kiểm đã kiểm — tệp đổi (vd dựng lại theo cấu hình khác) → không gắn, xoá kết
        quả tự kiểm để lần dựng sau kiểm lại."""
        try:
            from subtitles_extractor.infrastructure.ocr.trt_direct_runner import (
                attach_direct_tensorrt,
            )

            described = attach_direct_tensorrt(model, expected=expected if isinstance(expected, dict) else None)
        except Exception as exc:  # noqa: BLE001 — thiếu engine / thư viện: GPU thường, không chặn OCR
            logger.warning("TensorRT trực tiếp không gắn được (%s) — chạy GPU thường.", exc)
            self._tensorrt_unavailable = True
            if "khác lần tự kiểm" in str(exc) and getattr(self, "_trt_dir", None) is not None:
                self._forget_selftest()
            return
        self._model_uses_direct_trt = True
        self._trt_first_inference_done = False
        logger.info("PaddleOCR chạy GPU + TensorRT TRỰC TIẾP (ứng dụng chạy engine, %s): %s.",
                    "FP16" if self._config.precision == PrecisionMode.FP16 else "FP32", "; ".join(described))

    def _apply_trt_batches(self) -> None:
        """[v3.23.776] Cỡ lô engine TensorRT theo cài đặt (bộ nhận dạng tối ưu = lô ứng dụng đưa vào, không quá lô tối
        đa) — đặt TRƯỚC tự kiểm / dựng (tiến trình tự kiểm thừa hưởng biến môi trường)."""
        from subtitles_extractor.infrastructure.ocr.tensorrt_runtime_plan import (
            set_trt_batches,
        )

        rec = int(getattr(self._config, "trt_rec_batch", 32) or 32)
        det, rec, opt = set_trt_batches(int(getattr(self._config, "trt_det_batch", 16) or 16), rec,
                                        min(rec, int(getattr(self._config, "batch_size", 16) or 16)))
        logger.info("TensorRT: engine bộ dò tối đa %d ảnh/lô, bộ nhận dạng tối đa %d dòng/lô (tối ưu %d) — theo cài đặt.",
                    det, rec, opt)

    def _forget_selftest(self) -> None:
        try:
            from subtitles_extractor import __version__
            from subtitles_extractor.infrastructure.ocr.tensorrt_selftest import (
                forget_tensorrt_selftest,
            )

            assert self._trt_dir is not None
            forget_tensorrt_selftest(self._trt_dir, __version__, self._build_init_kwargs(use_gpu=True, use_trt=True),
                                     int(getattr(self._config, "det_batch_size", 1) or 1))
        except Exception as exc:  # noqa: BLE001 — chỉ là dọn kết quả cũ
            logger.debug("Không xoá được kết quả tự kiểm TensorRT: %s", exc)

    def _direct_trt_calls(self) -> int:
        """Số lô runner TensorRT trực tiếp đã chạy (0 = chưa chạy TensorRT lần nào)."""
        try:
            pipeline = self._model.paddlex_pipeline._pipeline  # type: ignore[union-attr]
        except AttributeError:
            return 0
        return sum(int(getattr(getattr(getattr(pipeline, name, None), "runner", None), "calls", 0) or 0)
                   for name in ("text_det_model", "text_rec_model"))

    def _cap_batches_for_tensorrt(self, model: Any, det_size: int) -> int:
        """[v3.23.764] Lô của bộ dò / bộ nhận dạng không vượt lô lớn nhất engine TensorRT được dựng (xem
        :func:`~.tensorrt_runtime_plan.trt_max_batch` — lô lớn hơn từng làm SẬP tiến trình). Trả cỡ lô bộ dò."""
        from subtitles_extractor.infrastructure.ocr.tensorrt_runtime_plan import (
            TRT_DEFAULT_DET_MAX_BATCH,
            TRT_DEFAULT_REC_MAX_BATCH,
            trt_max_batch,
        )

        try:
            pipeline = model.paddlex_pipeline._pipeline
        except AttributeError:
            pipeline = None
        det_model = getattr(pipeline, "text_det_model", None)
        rec_model = getattr(pipeline, "text_rec_model", None)
        det_max = trt_max_batch(getattr(det_model, "config", None), TRT_DEFAULT_DET_MAX_BATCH)
        rec_max = trt_max_batch(getattr(rec_model, "config", None), TRT_DEFAULT_REC_MAX_BATCH)
        sampler = getattr(rec_model, "batch_sampler", None)
        rec_size = int(getattr(sampler, "batch_size", 1) or 1)
        if sampler is not None and rec_size > rec_max:
            sampler.batch_size = rec_max
        capped = min(det_size, det_max)
        if det_model is not None and getattr(det_model, "batch_sampler", None) is not None:
            det_model.batch_sampler.batch_size = capped
        if pipeline is not None and getattr(pipeline, "batch_sampler", None) is not None:
            pipeline.batch_sampler.batch_size = capped
        logger.info("TensorRT: bộ dò %d ảnh/lô (engine tối đa %d), bộ nhận dạng %d dòng/lô (engine tối đa %d).",
                    capped, det_max, min(rec_size, rec_max), rec_max)
        return capped

    def _create_model_with_device_fallback(
        self, paddle_ocr_cls: Any, *, use_gpu: bool
    ) -> Any:
        """Tạo model PaddleOCR; nếu init GPU thất bại thì TỰ LÙI về CPU.

        [v3.23.295] Ngay cả khi ``device_count() > 0`` báo có GPU, việc khởi tạo vẫn
        có thể lỗi (driver quá cũ cho CUDA 12.9, VRAM cạn lúc nạp, xung đột thư viện…).
        Khi đó, thay vì để OCR chết, ta dựng lại kwargs cho CPU và thử lại -> bản
        standalone luôn OCR được, chỉ chậm hơn.

        Args:
            paddle_ocr_cls: Lớp ``PaddleOCR`` đã import.
            use_gpu: Kết quả dò thiết bị (đã tính tới GPU thật).

        Returns:
            Đối tượng pipeline PaddleOCR đã tạo.

        Raises:
            OcrModelLoadError: Khi cả GPU lẫn CPU đều không dựng được model.
        """
        use_trt: bool = self._config.use_tensorrt and use_gpu and not getattr(self, "_tensorrt_unavailable", False)
        direct_trt = False
        direct_expected: Any = None
        if use_trt:
            self._apply_trt_batches()
        if use_trt and not _tensorrt_thread_ok():
            # [v3.23.767] Paddle đã nạp ở luồng khác (vd bật TensorRT giữa phiên) — TensorRT ở luồng không nạp paddle là
            # bố cục từng SẬP; phiên sau (mở lại ứng dụng) nạp paddle đúng trên luồng Paddle.
            logger.warning("TensorRT: paddle đã được nạp ở luồng khác (%s) — phiên này chạy GPU thường, mở lại ứng dụng để "
                           "dùng TensorRT.", "không rõ luồng" if _PADDLE_IMPORT_THREAD is None else
                           "trước khi bật TensorRT")
            self._tensorrt_unavailable = True
            use_trt = False
        if use_trt:
            self._trt_dir = _prepare_paddle_tensorrt()
            if self._trt_dir is None:
                # [v3.23.766] Chưa tải TensorRT, hoặc bản vá / lưới an toàn hình dạng không cài được → không chạy
                # TensorRT "trần" (không tự kiểm, không dấu chống sập — đúng kịch bản sập "access violation").
                logger.warning("TensorRT chưa sẵn sàng (chưa tải trong Cài đặt, hoặc không chuẩn bị được) — chạy GPU "
                               "thường.")
                self._tensorrt_unavailable = True
                use_trt = False
            elif _tensorrt_crashed_last_time(self._trt_dir):
                # [v3.23.764] Lỗi sập trong thư viện C++ không bắt được bằng Python: lần mở trước của CÙNG bản ứng dụng
                # sập giữa lần suy luận TensorRT đầu tiên → không thử lại (lặp sập mỗi lần mở); nâng cấp thì thử lại.
                logger.warning("Lần chạy trước ứng dụng SẬP ở lần suy luận TensorRT đầu tiên — tắt TensorRT cho bản này, "
                               "chạy GPU thường. Bản ứng dụng mới sẽ tự thử lại TensorRT.")
                self._tensorrt_unavailable = True
                use_trt = False
            elif self._trt_dir is not None:
                # [v3.23.766] Chạy thử ĐÚNG cấu hình ở tiến trình riêng trước: lỗi sập C++ của TensorRT (log 19/11
                # 14:46, 15:46) chỉ giết tiến trình con; tiến trình chính ghi lý do rồi chạy GPU thường.
                try:
                    outcome = _tensorrt_selftest(
                        self._trt_dir, self._build_init_kwargs(use_gpu=True, use_trt=True),
                        int(getattr(self._config, "det_batch_size", 1) or 1), self._config,
                    )
                except Exception as exc:  # noqa: BLE001 — lỗi của chính bộ máy tự kiểm: không được chặn OCR
                    from subtitles_extractor.infrastructure.ocr.tensorrt_selftest import (
                        SelfTestOutcome,
                    )

                    outcome = SelfTestOutcome(False, f"không chạy được tự kiểm: {type(exc).__name__}: {exc}",
                                              definitive=False)
                if outcome.ok:
                    logger.info("TensorRT tự kiểm%s: %s", " (đã kiểm trước đó)" if outcome.cached else "", outcome.reason)
                    # [v3.23.774] Lớp chạy engine (C++) của Paddle SẬP với mọi engine (chẩn đoán 19/11) → KHÔNG dựng
                    # predictor TensorRT của Paddle; dựng mô hình GPU thường rồi gắn runner TensorRT trực tiếp.
                    direct_trt = True
                    direct_expected = (outcome.data or {}).get("engines")
                    use_trt = False
                else:
                    logger.warning("TensorRT tự kiểm KHÔNG đạt%s — chạy GPU thường. %s", " (đã kiểm trước đó)"
                                   if outcome.cached else "", outcome.reason)
                    self._tensorrt_unavailable = True
                    use_trt = False
        kwargs: dict[str, Any] = self._build_init_kwargs(use_gpu=use_gpu, use_trt=use_trt)
        logger.info(
            "Khởi tạo PaddleOCR: device=%s, det=%s, rec=%s",
            "gpu" if use_gpu else "cpu",
            self._config.detection_model_name,
            self._config.recognition_model_name,
        )
        self._model_uses_trt = False
        self._model_uses_direct_trt = False
        try:
            model = self._try_create_model_with_retry(paddle_ocr_cls, kwargs)
            self._model_uses_trt = use_trt
            if direct_trt:
                self._attach_direct_tensorrt(model, direct_expected)
            # [v3.23.766] Mỗi mô hình TensorRT dựng mới: lần suy luận đầu của NÓ lại được canh (rà độc lập: cờ của mô
            # hình trước làm mô hình dựng lại trong phiên không còn dấu chống sập).
            self._trt_first_inference_done = False
            if use_trt:
                logger.info("PaddleOCR chạy GPU + TensorRT (%s).",
                            "FP16" if self._config.precision == PrecisionMode.FP16 else "FP32")
        except OcrModelLoadError as gpu_error:
            if not use_gpu:
                # Đã ở CPU mà vẫn lỗi -> lỗi thật, không có gì để lùi thêm.
                raise
            if use_trt:
                # [v3.23.754] Chuỗi lùi GPU+TensorRT → GPU → CPU (CLAUDE.md). Trước: TensorRT lỗi (log 19/11 10:00 — dựng
                # lỗi sau 0,12 s, bản paddle không kèm thư viện TensorRT) nhảy THẲNG xuống CPU → cả phim OCR trên CPU
                # (lô 512, 16 luồng) — ứng dụng ì, chậm. Nhớ cho cả phiên: không thử TensorRT lại mỗi lần dựng mô hình.
                self._tensorrt_unavailable = True
                logger.warning(
                    "TensorRT không dùng được (%s) — chạy GPU không TensorRT. Tắt TensorRT trong Cài đặt để bỏ thông "
                    "báo này.", gpu_error,
                )
                _log_tensorrt_failure_details(gpu_error)
                try:
                    model = self._try_create_model_with_retry(
                        paddle_ocr_cls, self._build_init_kwargs(use_gpu=True, use_trt=False),
                    )
                    self._active_device = DeviceKind.GPU
                    return model
                except OcrModelLoadError as plain_gpu_error:
                    gpu_error = plain_gpu_error
            logger.warning(
                "Khởi tạo OCR trên GPU thất bại (%s) — tự động lùi về CPU (chậm hơn nhưng vẫn chạy). Kiểm tra driver "
                "NVIDIA nếu muốn tăng tốc GPU.", gpu_error,
            )
            cpu_kwargs = self._build_init_kwargs(use_gpu=False, use_trt=False)
            model = self._try_create_model_with_retry(paddle_ocr_cls, cpu_kwargs)
            self._active_device = DeviceKind.CPU
            return model
        self._active_device = DeviceKind.GPU if use_gpu else DeviceKind.CPU
        return model

    def apply_config(self, new_config: OcrEngineConfig) -> None:
        """Đổi cấu hình MẶC ĐỊNH (Cài đặt).

        [v3.23.656] Không giải phóng model (lượt khác có thể đang suy luận) và không chờ khoá nạp model (gọi từ luồng
        giao diện khi lưu Cài đặt — trước đây đứng tới hết lượt nạp model). Lượt nào cần model khác sẽ tự nạp lại giữa
        hai lô; đổi thông số không dựng vào model (ngưỡng, tiền xử lý) thì áp ngay cho lượt không truyền cấu hình.
        """
        self._default_config = new_config
        if self._model is None or not self._needs_model_reload(new_config):
            self._config = new_config

    def prepare(self, config: OcrEngineConfig | None = None) -> None:
        """[v3.23.656] Nạp sẵn model đúng cho cấu hình của lượt sắp chạy (tránh nạp hai lần lúc khởi động lạnh)."""
        with self._infer_lock:
            self._ensure_model_for(config if config is not None else self._default_config)

    def _ensure_model_for(self, config: OcrEngineConfig) -> None:
        """[v3.23.656] Model đang nạp phải khớp ``config`` (tên model, thiết bị…); lệch thì nạp lại. Gọi trong khoá suy luận."""
        with self._init_lock:
            if self._model is not None and self._needs_model_reload(config):
                logger.info("Lượt OCR cần model/cấu hình dựng khác → nạp lại model (không lượt nào đang suy luận).")
                self._release_internal()
            if self._model is None:
                self._config = config
        if self._model is None:
            self.initialize()

    def update_preprocess_config(self, new_config: OcrEngineConfig) -> None:
        self.apply_config(new_config)

    def release(self) -> None:
        # [v3.23.650] Không chặn luồng gọi (thường là giao diện lúc đóng app) khi ``initialize()`` đang giữ khoá:
        # đánh dấu để ``initialize()`` tự dọn sau khi nạp xong.
        # [v3.23.656] Một lô đang suy luận: không giải phóng model/VRAM dưới chân nó — hoãn tới khi lô xong.
        if not self._infer_lock.acquire(blocking=False):
            self._release_after_infer = True
            return
        try:
            self._release_locked_safely()
        finally:
            self._infer_lock.release()

    def release_auxiliary_models(self) -> None:
        """[v3.23.739] Trả mô hình PHỤ (PaddleOCR-VL, bộ cứu chữ PP-OCRv5) + bộ đệm VRAM ngay trong luồng gọi.

        Gọi cuối lượt trích xuất (luồng nền). Văng 18/11 05:48 xảy ra 2 s sau khi mpv trang Chỉnh sửa mở, ngay sau lượt
        kiểm định 152 s đọc 6393 ảnh trên GPU: đối tượng Paddle để lại cho bộ thu rác dọn sau trên luồng GIAO DIỆN / VRAM
        còn bị chiếm khi mpv cấp bộ giải mã. Mô hình chính giữ nguyên (lượt sau không phải nạp lại).
        """
        with self._infer_lock:
            self._rescue_model = None
            self._vl_reader.release()
            self._vl_cache.clear()
            gc.collect()
            try:
                self._empty_paddle_vram()
                self._empty_cupy_pool()
            except (RuntimeError, OSError, ImportError) as exc:
                logger.debug("Trả VRAM mô hình phụ không hoàn tất (bỏ qua): %s", exc)

    def trim_gpu_memory(self) -> str:
        """[v3.23.746] Trả bộ nhớ đệm VRAM (Paddle + CuPy) mà bước OCR để lại, GIỮ mô hình; trả chuỗi VRAM còn trống.

        Gọi trước bước soát câu bằng PaddleOCR-VL và trước gói kiểm định: log 18/11 21:34 treo ngay lần sinh chữ đầu
        tiên của VL khi VRAM 7,6/8,0 GB (bước OCR 228 675 khung để lại bộ nhớ đệm lớn). Ghi số VRAM vào nhật ký để lần
        sau có bằng chứng.
        """
        with self._infer_lock:
            gc.collect()
            try:
                self._empty_paddle_vram()
                self._empty_cupy_pool()
            except (RuntimeError, OSError, ImportError) as exc:
                logger.debug("Trả bộ nhớ đệm VRAM không hoàn tất (bỏ qua): %s", exc)
            return vram_summary()

    def _release_locked_safely(self) -> None:
        if not self._init_lock.acquire(blocking=False):
            self._release_after_init = True
            return
        try:
            self._release_locked()
        finally:
            self._init_lock.release()

    def _release_locked(self) -> None:
        self._rescue_model = None
        self._vl_reader.release()
        self._vl_cache.clear()
        if self._model is None: return
        if self._model_uses_trt:
            # [v3.23.767] Mô hình TensorRT: tham chiếu DUY NHẤT trao cho luồng Paddle (hộp), huỷ ở đó TRƯỚC khi trả VRAM.
            self._drop_tensorrt_model(wait=False)
            return
        model_ref = self._model
        self._model = None
        self._active_device = None
        def _async_cleanup(model: Any) -> None:
            try:
                del model
                gc.collect()
                self._empty_paddle_vram()
                self._empty_cupy_pool()
            except (RuntimeError, OSError, ImportError) as exc:
                # Cleanup best-effort — log để chẩn đoán rò rỉ VRAM nếu có
                logger.debug("Dọn dẹp VRAM không hoàn tất (bỏ qua): %s", exc)
        threading.Thread(target=_async_cleanup, args=(model_ref,), daemon=True).start()

    def _release_internal(self) -> None:
        self._rescue_model = None
        self._vl_reader.release()
        self._vl_cache.clear()
        if self._model is None: return
        if self._model_uses_trt:
            self._drop_tensorrt_model(wait=True)              # [v3.23.767] huỷ trên luồng Paddle
            return
        del self._model
        self._model = None
        self._active_device = None
        gc.collect()
        self._empty_paddle_vram()
        self._empty_cupy_pool()

    def _drop_tensorrt_model(self, *, wait: bool) -> None:
        """[v3.23.767] Huỷ mô hình TensorRT TRÊN LUỒNG PADDLE (luồng đã dựng nó) rồi mới trả VRAM.

        Rà độc lập 3.23.767 (đo được): trao ``self._model`` làm tham số thì luồng gọi vẫn giữ tham chiếu cuối (biến cục
        bộ, bộ tham số, việc trong hàng đợi) → mô hình thật ra huỷ ở LUỒNG GỌI, và ``empty_cache`` chạy khi nó còn sống.
        Nay tham chiếu duy nhất nằm trong một hộp; luồng Paddle làm rỗng hộp, ``gc.collect``, rồi mới trả VRAM.
        """
        box = [self._model]
        self._model = None
        self._model_uses_trt = False
        self._active_device = None

        def drop() -> None:
            box.clear()
            gc.collect()
            try:
                self._empty_paddle_vram()
                self._empty_cupy_pool()
            except (RuntimeError, OSError, ImportError) as exc:
                logger.debug("Trả VRAM sau khi huỷ mô hình TensorRT không hoàn tất (bỏ qua): %s", exc)

        if wait:
            _run_on_paddle_thread(drop)
        else:
            _paddle_thread().submit(drop)

    def infer(
        self, image_rgb: np.ndarray, frame_index: int, timestamp_sec: float, config: OcrEngineConfig | None = None,
    ) -> OcrFrameResult:
        results = self.infer_batch([image_rgb], [frame_index], [timestamp_sec], config=config)
        return results[0]

    def infer_batch(
        self, images_rgb: list[np.ndarray], frame_indices: list[int], timestamps_sec: list[float],
        config: OcrEngineConfig | None = None,
    ) -> list[OcrFrameResult]:
        """Nhận dạng một lô ảnh.

        Args:
            config: [v3.23.656] Cấu hình của LƯỢT gọi (ngưỡng, tiền xử lý, model); ``None`` = cấu hình mặc định (Cài
                đặt). Không đổi trạng thái của engine dùng chung.
        """
        if not images_rgb: return []
        with self._infer_lock:
            if config is not None:
                run_config = config
            elif self._model is not None:
                # [v3.23.656] Lượt không có cấu hình riêng (tự dò ROI, OCR phụ đề nhúng): dùng model đang nạp — trước
                # đây chúng dùng cấu hình áp gần nhất; ép về Cài đặt sẽ nạp lại model qua lại với lượt trích xuất.
                run_config = self._config
            else:
                run_config = self._default_config
            self._ensure_model_for(run_config)
            prepared_images = [self._prepare_image(img, run_config) for img in images_rgb]
            bgr_list = pad_to_common_shape([p.image_bgr for p in prepared_images])
            try:
                raw_results = self._predict_with_tensorrt_fallback(bgr_list, frame_indices, run_config)
                parsed = [
                    parse_paddle_result(raw, frame_idx, ts)
                    for raw, frame_idx, ts in zip(raw_results, frame_indices, timestamps_sec, strict=True)
                ]
                parsed = self._rescue_unreadable(parsed, raw_results, prepared_images, images_rgb, run_config)
                if is_vl_recognizer(run_config.recognition_model_name):
                    parsed = self._vl_reread(parsed, raw_results, prepared_images, images_rgb)
            finally:
                if self._release_after_infer:
                    self._release_after_infer = False
                    self._release_locked_safely()

        # [CRITICAL BUG FIX]: Đã thêm "prepared_images" vào trong zip()
        return [
            self._scale_back_result(result, prep.scale, prep.border_px)
            for result, prep in zip(parsed, prepared_images, strict=True)
        ]

    def _rescue_unreadable(
        self, parsed: list[OcrFrameResult], raw_results: list[Any], prepared: list[_PreparedImage],
        originals: list[np.ndarray], config: OcrEngineConfig,
    ) -> list[OcrFrameResult]:
        """[v3.23.732] Đọc lại hộp bộ dò thấy mà nhận dạng không đọc được / hộp một chữ đọc yếu bằng bộ nhận dạng phụ
        (xem :mod:`~subtitles_extractor.infrastructure.ocr.unreadable_rescue`). Chỉ cho chữ Hán; lỗi → bỏ qua."""
        if self._rescue_unavailable or (config.language or "ch") not in ("ch", "chinese", "zh"):
            return parsed
        jobs: list[tuple[int, list[Any], list[np.ndarray]]] = []
        for index, (result, raw, prep, original) in enumerate(zip(parsed, raw_results, prepared, originals, strict=True)):
            image = prep.image_bgr
            if raw is None or image is None or original is None or getattr(original, "ndim", 0) != 3:
                continue
            candidates = find_rescue_candidates(detected_polygons(raw), result.text_boxes)
            kept, crops = [], []
            scale = prep.scale if prep.scale > 0 else 1.0
            for candidate in candidates:
                rect = crop_rect(candidate.polygon, image.shape[1], image.shape[0])
                source_polygon = [((x - prep.border_px) / scale, (y - prep.border_px) / scale) for x, y in candidate.polygon]
                source_rect = crop_rect(source_polygon, original.shape[1], original.shape[0])
                if rect is None or source_rect is None:
                    continue
                x0, y0, x1, y1 = rect
                sx0, sy0, sx1, sy1 = source_rect
                kept.append(candidate)
                # Hai lần đọc mỗi hộp: ảnh OCR đã thấy (đã tiền xử lý) và ảnh gốc — mỗi cách cứu được 4/6 khung "诶" đo thật.
                crops.append(np.ascontiguousarray(image[y0:y1, x0:x1]))
                crops.append(np.ascontiguousarray(cv2.cvtColor(original[sy0:sy1, sx0:sx1], cv2.COLOR_RGB2BGR)))
            if kept:
                jobs.append((index, kept, crops))
        if not jobs:
            return parsed
        model = self._get_rescue_model()
        if model is None:
            return parsed
        crops = [crop for _index, _kept, job_crops in jobs for crop in job_crops]
        try:
            outputs = list(model.predict(input=crops, batch_size=min(len(crops), 32)))
        except Exception as exc:  # noqa: BLE001 — bước phụ: lỗi không được phá lượt OCR
            logger.warning("Đọc lại hộp không đọc được thất bại (bỏ qua bước cứu chữ): %s", exc)
            self._rescue_unavailable = True
            return parsed
        raw_readings = [self._rescue_reading(output) for output in outputs]
        if len(raw_readings) != len(crops):
            return parsed
        readings = [best_reading(raw_readings[i:i + 2]) for i in range(0, len(raw_readings), 2)]
        position = 0
        result_list = list(parsed)
        for index, kept, _crops in jobs:
            chunk = readings[position:position + len(kept)]
            position += len(kept)
            result_list[index], rescued = apply_rescues(result_list[index], kept, chunk)
            if rescued:
                logger.debug("Khung %.2f s: cứu %d hộp chữ bằng %s.", result_list[index].timestamp_sec, rescued,
                             RESCUE_RECOGNITION_MODEL)
        return result_list

    def _vl_reread(
        self, parsed: list[OcrFrameResult], raw_results: list[Any], prepared: list[_PreparedImage],
        originals: list[np.ndarray],
    ) -> list[OcrFrameResult]:
        """[v3.23.733] Đọc lại mọi hộp bộ dò thấy bằng PaddleOCR-VL (cắt từ ẢNH GỐC — đo: VL đọc ảnh gốc tốt nhất).

        * Bản đọc nháp PP-OCR tin cậy ≥ :data:`VL_CACHE_MIN_CONFIDENCE` cùng chữ cùng chỗ đã được VL đọc → dùng lại (khung
          đọc lại theo lịch / xác nhận của cùng một câu chiếm phần lớn số khung OCR);
        * chỉ đọc lại hộp PP-OCR đã nhận ra chữ: hộp đọc không ra thường là vật trong cảnh mà VL đọc thành chữ ("品",
          "福" — đo trên 148 khung thật); chữ thật ngoài bảng chữ ("诶") do bước cứu chữ lo;
        * hộp được bước cứu chữ ghi chữ trong :data:`RESCUE_MAP` (vd "诶") giữ nguyên — VL cũng không có chữ đó;
        * VL lỗi → giữ bản nháp, ghi log một lần, tắt VL cho phiên.
        """
        override = self._line_reader_override if self._line_reader_thread == threading.get_ident() else None
        if override is not None:
            self._vl_cache.clear()        # [v3.23.759] mỗi lô đọc lại MỌI hộp bằng bộ đọc thay thế (không lẫn bản đọc VL)
        elif self._vl_unavailable:
            return parsed
        rescued_texts = set(RESCUE_MAP.values())
        plans: list[list[tuple[OcrTextBox | None, Any, tuple[Any, ...] | None]]] = []
        crops: list[np.ndarray] = []
        crop_owner: list[tuple[int, int]] = []
        for index, (result, raw, prep, original) in enumerate(zip(parsed, raw_results, prepared, originals, strict=True)):
            entries: list[tuple[OcrTextBox | None, Any, tuple[Any, ...] | None]] = []
            plans.append(entries)
            if raw is None or original is None or getattr(original, "ndim", 0) != 3:
                continue
            boxes = list(result.text_boxes)
            used: set[int] = set()
            polygons = detected_polygons(raw) or [list(box.polygon) for box in boxes if box.polygon]
            scale = prep.scale if prep.scale > 0 else 1.0
            for polygon in polygons:
                draft = None
                for position, box in enumerate(boxes):
                    if position not in used and box.polygon and _overlap_ratio(polygon, box.polygon) >= 0.5:
                        draft = box
                        used.add(position)
                        break
                if draft is None:
                    # Hộp PP-OCR không đọc được (dưới ngưỡng): thường là vật trong cảnh — VL đọc chúng thành chữ ("品",
                    # "福"; đo trên 148 khung thật). Chữ thật không có trong bảng chữ ("诶") đã do bước cứu chữ xử lý.
                    continue
                if (draft.text or "").strip() in rescued_texts:
                    entries.append((draft, None, None))
                    continue
                source = [((x - prep.border_px) / scale, (y - prep.border_px) / scale) for x, y in polygon]
                rect = crop_rect(source, original.shape[1], original.shape[0])
                if rect is None:
                    if draft is not None:
                        entries.append((draft, None, None))
                    continue
                x0, y0, x1, y1 = rect
                key = None
                if draft is not None and float(draft.confidence) >= VL_CACHE_MIN_CONFIDENCE and draft.text:
                    key = (draft.text, x0 // 8, y0 // 8, x1 // 8, y1 // 8)
                    if key in self._vl_cache:
                        self._vl_cache.move_to_end(key)
                        entries.append((draft, polygon, key))
                        continue
                entries.append((draft, polygon, key))
                crop_owner.append((index, len(entries) - 1))
                crops.append(np.ascontiguousarray(cv2.cvtColor(original[y0:y1, x0:x1], cv2.COLOR_RGB2BGR)))
            for position, box in enumerate(boxes):            # hộp nhận dạng không khớp hộp dò nào (hiếm): giữ
                if position not in used and not any(entry[0] is box for entry in entries):
                    entries.append((box, None, None))
        readings: dict[tuple[int, int], str] = {}
        if crops:
            device = "gpu" if self._active_device == DeviceKind.GPU else "cpu"
            texts: list[str] | None = None
            if override is not None:
                try:
                    texts = list(override.read(crops, device=device))
                    if len(texts) != len(crops):
                        raise RuntimeError(f"trả {len(texts)} bản đọc cho {len(crops)} ảnh")
                except Exception as exc:  # noqa: BLE001 — [v3.23.758] GLM-OCR hỏng: bỏ nó, lùi về PaddleOCR-VL
                    # [v3.23.759] Lô này giữ bản nháp PP-OCR; KHÔNG nạp VL ngay (llama-server còn giữ GLM trong VRAM — VL
                    # nạp lỗi vì tranh VRAM sẽ tắt VL cả phiên). Lượt soát tắt server rồi các câu sau mới đọc bằng VL.
                    name = getattr(override, "model_name", "bộ đọc thay thế")
                    logger.warning("%s soát câu lỗi — phần còn lại soát bằng PaddleOCR-VL: %s", name, exc)
                    self._line_reader_override = None
                    self._line_reader_thread = None
                    self._line_reader_error = f"{type(exc).__name__}: {exc}"[:300]
                    return parsed
            try:
                if texts is None:
                    texts = self._vl_reader.read(crops, device=device)
            except Exception as exc:  # noqa: BLE001 — VL lỗi (thiếu mô hình / VRAM / thư viện): giữ bản nháp PP-OCR
                logger.warning("PaddleOCR-VL không chạy được (%s) — dùng bộ nhận dạng PP-OCR cho phiên này.", exc)
                self._vl_unavailable = True
                self._vl_error = f"{type(exc).__name__}: {exc}"[:300]
                return parsed
            readings = dict(zip(crop_owner, texts, strict=True))
        out: list[OcrFrameResult] = []
        for index, (result, entries) in enumerate(zip(parsed, plans, strict=True)):
            if not entries:
                out.append(result)
                continue
            boxes_out: list[OcrTextBox] = []
            for position, (draft, polygon, key) in enumerate(entries):
                if polygon is None:
                    if draft is not None:
                        boxes_out.append(draft)
                    continue
                if (index, position) in readings:
                    text = readings[(index, position)]
                    if key is not None and text:
                        self._vl_cache[key] = text
                        while len(self._vl_cache) > VL_CACHE_SIZE:
                            self._vl_cache.popitem(last=False)
                else:
                    text = self._vl_cache.get(key, "") if key is not None else ""
                if not text:
                    if draft is not None:
                        boxes_out.append(draft)
                    continue
                agree = draft is not None and _same_letters(text, draft.text)
                confidence = max(float(draft.confidence), VL_AGREE_CONFIDENCE) if agree and draft is not None else VL_CONFIDENCE
                shape = list(draft.polygon) if draft is not None and draft.polygon else [
                    (int(round(x)), int(round(y))) for x, y in polygon
                ]
                boxes_out.append(OcrTextBox(text=text, confidence=Confidence(confidence), polygon=shape))
            out.append(dataclasses.replace(result, text_boxes=boxes_out))
        return out

    @staticmethod
    def _rescue_reading(output: Any) -> tuple[str, float]:
        target = getattr(output, "res", output)
        if isinstance(target, dict) and isinstance(target.get("res"), dict):
            target = target["res"]
        if isinstance(target, dict):
            return str(target.get("rec_text", "") or ""), float(target.get("rec_score", 0.0) or 0.0)
        return str(getattr(target, "rec_text", "") or ""), float(getattr(target, "rec_score", 0.0) or 0.0)

    def _get_rescue_model(self) -> Any | None:
        """Nạp bộ nhận dạng phụ khi cần lần đầu, CÙNG thiết bị với OCR chính (GPU khi lượt chạy GPU)."""
        device = "gpu" if self._active_device == DeviceKind.GPU else "cpu"
        if self._rescue_model is not None and self._rescue_device == device:
            return self._rescue_model
        try:
            from paddleocr import TextRecognition

            self._rescue_model = TextRecognition(model_name=RESCUE_RECOGNITION_MODEL, device=device)
            self._rescue_device = device
            logger.info("Nạp bộ nhận dạng phụ %s (%s) để đọc lại chữ không có trong bảng chữ.",
                        RESCUE_RECOGNITION_MODEL, device)
        except Exception as exc:  # noqa: BLE001 — thiếu mô hình / không tải được: tắt bước cứu, OCR vẫn chạy
            logger.warning("Không nạp được bộ nhận dạng phụ %s — bỏ bước cứu chữ: %s", RESCUE_RECOGNITION_MODEL, exc)
            self._rescue_unavailable = True
            self._rescue_model = None
        return self._rescue_model

    def _predict_with_tensorrt_fallback(
        self, bgr_images: list[np.ndarray], frame_indices: list[int], config: OcrEngineConfig,
    ) -> list[Any]:
        """[v3.23.763] Chuỗi lùi GPU+TensorRT → GPU lúc CHẠY (CLAUDE.md), không chỉ lúc dựng mô hình.

        Log 19/11 14:21 (bản 3.23.761): engine TensorRT dựng xong nhưng MỌI lô OCR lỗi "(NotFound) TensorRT is needed,
        but TensorRT dynamic library 'nvinfer_plugin.dll' is not found" → từng khung "lỗi OCR riêng lẻ", cả phim không
        ra phụ đề. Nay: mô hình dựng bằng TensorRT mà lô lỗi do TensorRT → nhớ cho cả phiên, dựng lại mô hình GPU KHÔNG
        TensorRT, chạy lại đúng lô đó. Gọi trong khoá suy luận.
        """
        guard = ((self._model_uses_trt or getattr(self, "_model_uses_direct_trt", False))
                 and not self._trt_first_inference_done and self._trt_dir is not None)
        if guard:
            _begin_tensorrt_first_inference(self._trt_dir)
        try:
            if self._model_uses_trt and not _legacy_threads():
                results: list[Any] = _run_on_paddle_thread(self._predict_batch, bgr_images, frame_indices,
                                                           config)  # [v3.23.767] luồng đã dựng engine
            else:
                results = self._predict_batch(bgr_images, frame_indices, config)
            if guard:
                _end_tensorrt_first_inference(self._trt_dir)
                # [v3.23.774] Lô lùi Paddle (ngoài hồ sơ) chưa chạy TensorRT → lô sau vẫn được canh (rà độc lập).
                if not getattr(self, "_model_uses_direct_trt", False) or self._direct_trt_calls() > 0:
                    self._trt_first_inference_done = True
                    logger.info("TensorRT: lần suy luận đầu tiên chạy xong (%d ảnh).", len(bgr_images))
            return results
        except (OcrInferenceError, RuntimeError) as exc:
            if guard:
                _end_tensorrt_first_inference(self._trt_dir)          # lỗi Python (không sập): không phải lỗi sập
            if not self._model_uses_trt or not _is_tensorrt_runtime_error(exc):
                raise
            logger.warning("TensorRT lỗi khi chạy (%s) — dựng lại mô hình GPU không TensorRT cho phiên này rồi chạy lại "
                           "lô.", str(exc).splitlines()[0][:300])
            _log_tensorrt_failure_details(exc)
            self._tensorrt_unavailable = True
            # [v3.23.767] Traceback giữ khung ``predict`` của Paddle (tham chiếu tới mô hình TensorRT) → xoá trước khi huỷ,
            # để mô hình thật sự huỷ trên luồng Paddle (rà độc lập: từng huỷ ở luồng trích xuất).
            import traceback

            if exc.__traceback__ is not None:
                traceback.clear_frames(exc.__traceback__)
            exc.__traceback__ = None
            if exc.__cause__ is not None and exc.__cause__.__traceback__ is not None:
                traceback.clear_frames(exc.__cause__.__traceback__)
                exc.__cause__.__traceback__ = None
            with self._init_lock:
                self._release_internal()
            self._ensure_model_for(config)
            return self._predict_batch(bgr_images, frame_indices, config)

    def _predict_batch(
        self, bgr_images: list[np.ndarray], frame_indices: list[int], config: OcrEngineConfig | None = None,
    ) -> list[Any]:
        if self._model is None: raise OcrInferenceError("Model chưa khởi tạo.")
        cfg = config if config is not None else self._config
        try:
            raw_output = self._model.predict(
                input=bgr_images, use_doc_orientation_classify=False, use_doc_unwarping=False,
                use_textline_orientation=cfg.use_textline_orientation,
                **self._det_limit_kwargs(cfg),
                text_det_thresh=cfg.det_thresh, text_det_box_thresh=cfg.det_box_thresh,
                text_det_unclip_ratio=cfg.det_unclip_ratio, text_rec_score_thresh=cfg.score_threshold,
            )
        except RuntimeError as exc:
            # [CRITICAL BUG FIX v2.9+]: Phân biệt lỗi OOM GPU với lỗi thông thường.
            # TRƯỚC đây: mọi RuntimeError đều bị wrap thành OcrInferenceError → caller
            # bắt OcrInferenceError và trả kết quả trống, bypassing OOM failsafe.
            # SAU: OOM/CUDA error được re-raise nguyên bản để caller phát hiện đúng.
            _err_msg = str(exc).lower()
            _OOM_INDICATORS = ("out of memory", "cudamemoryerror", "oom", "memory exhausted",
                               "paddle out", "alloc failed", "insufficient memory")
            if any(indicator in _err_msg for indicator in _OOM_INDICATORS):
                logger.error("GPU Out-of-Memory: %s", exc)
                raise  # Re-raise RuntimeError → caller sẽ wrap thành SubtitlesExtractorError
            raise OcrInferenceError(f"PaddleOCR lỗi: {exc}") from exc

        if hasattr(raw_output, "__iter__") and not isinstance(raw_output, dict):
            materialized = list(raw_output)
        else:
            materialized = [raw_output]

        if len(materialized) < len(bgr_images):
            materialized.extend([None] * (len(bgr_images) - len(materialized)))
        elif len(materialized) > len(bgr_images):
            materialized = materialized[:len(bgr_images)]
        return materialized

    def _prepare_image(self, image_rgb: np.ndarray, config: OcrEngineConfig | None = None) -> _PreparedImage:
        if image_rgb is None or image_rgb.ndim < 2 or image_rgb.size == 0:
            black = np.zeros((_FALLBACK_BLACK_IMAGE_SIZE, _FALLBACK_BLACK_IMAGE_SIZE, 3), dtype=np.uint8)
            return _PreparedImage(image_bgr=black, scale=1.0, border_px=0)

        pre = (config if config is not None else self._config).preprocess
        orig_height = float(max(1, image_rgb.shape[0]))

        working_rgb = apply_cpu_preprocessing_pipeline(
            image_rgb,
            upscale_small_text=pre.upscale_small_text,
            upscale_target_height_px=pre.upscale_target_height_px,
            apply_clahe_flag=pre.apply_clahe,
            clahe_clip_limit=pre.clahe_clip_limit,
            clahe_tile_size=pre.clahe_tile_size,
            apply_sharpen_flag=pre.apply_sharpen,
            apply_contrast_boost_flag=pre.apply_contrast_boost,
            contrast_factor=pre.contrast_factor,
            add_white_border=pre.add_white_border,
            border_thickness_px=pre.border_thickness_px
        )

        # [v3.23.710] Tỉ lệ thu-phóng tính trên phần ẢNH, không gồm viền: trước đây lấy cả chiều cao đã cộng 2 viền
        # → mọi hộp chữ trả về bị co (vùng cao 136 px, viền 8 px: co 0,895 lần), tâm câu lệch khỏi tâm vùng chữ ~35 px
        # và bộ lọc căn giữa loại oan các câu một chữ tin cậy thấp. Đo trên phim mẫu: tâm thật 360 px, lưu thành 290+35.
        border_px = int(pre.border_thickness_px) if pre.add_white_border and pre.border_thickness_px > 0 else 0
        scale = max(1.0, float(working_rgb.shape[0] - 2 * border_px)) / orig_height
        image_bgr = np.ascontiguousarray(cv2.cvtColor(working_rgb, cv2.COLOR_RGB2BGR), dtype=np.uint8)
        return _PreparedImage(image_bgr=image_bgr, scale=scale, border_px=border_px)

    @staticmethod
    def _scale_back_result(result: OcrFrameResult, scale: float, border: int) -> OcrFrameResult:
        if abs(scale - 1.0) < 1e-4 and border == 0: return result
        safe_scale = max(1e-5, scale)
        scaled_boxes = []
        for box in result.text_boxes:
            if not box.polygon:
                scaled_boxes.append(box); continue
            scaled_polygon = [
                (int(round((x - border) / safe_scale)), int(round((y - border) / safe_scale)))
                for (x, y) in box.polygon
            ]
            scaled_boxes.append(OcrTextBox(text=box.text, confidence=box.confidence, polygon=scaled_polygon))
        return dataclasses.replace(result, text_boxes=scaled_boxes)

    def _try_create_model_with_retry(self, paddle_ocr_cls: Any, kwargs: dict[str, Any]) -> Any:
        working_kwargs = dict(kwargs)
        for _attempt in range(_MAX_INIT_RETRY):
            try: return paddle_ocr_cls(**working_kwargs)
            except (TypeError, ValueError) as exc:
                bad_param = None
                for pattern in _UNEXPECTED_KWARG_PATTERNS:
                    match = pattern.search(str(exc))
                    if match: bad_param = match.group(1); break
                if bad_param and bad_param in working_kwargs:
                    working_kwargs.pop(bad_param); continue
                raise OcrModelLoadError(f"Cấu hình PaddleOCR không hợp lệ: {exc}.") from exc
            except (RuntimeError, OSError, ImportError) as exc:
                # [v3.23.280] Neu la loi "dependency error" cua paddlex khi dong goi, liet ke
                # CHINH XAC dep nao thieu (paddlex bao chung chung). Giup chan doan build.
                diag = _chan_doan_paddle_deps(str(exc))
                raise OcrModelLoadError(
                    f"Không khởi tạo được mô hình PaddleOCR: {exc}.{diag}"
                ) from exc
        raise OcrModelLoadError(f"Không khởi tạo được PaddleOCR sau {_MAX_INIT_RETRY} lần thử.")

    def _det_limit_kwargs(self, config: OcrEngineConfig | None = None) -> dict[str, Any]:
        """[v3.23.619] Giới hạn cạnh ảnh khi dò chữ theo cài đặt; 0 = PaddleOCR tự quyết.

        Trước đây luôn truyền ``None`` → cài đặt "Giới hạn cạnh" ở trang Cài đặt/Biên
        tập/Debug không có tác dụng (đổi giá trị chỉ làm nạp lại model vô ích).
        """
        cfg = config if config is not None else self._config
        side = int(cfg.limit_side_len or 0)
        if side <= 0:
            return {"text_det_limit_side_len": None, "text_det_limit_type": None}
        kind = cfg.limit_type if cfg.limit_type in ("max", "min") else "max"
        return {"text_det_limit_side_len": side, "text_det_limit_type": kind}

    def _build_init_kwargs(self, *, use_gpu: bool, use_trt: bool) -> dict[str, Any]:
        precision_str = "fp16" if (use_trt and self._config.precision == PrecisionMode.FP16) else "fp32"
        kwargs: dict[str, Any] = {
            "device": "gpu" if use_gpu else "cpu",
            "lang": self._config.language,
            "text_recognition_batch_size": self._config.batch_size,
            **self._det_limit_kwargs(),
            "text_det_thresh": self._config.det_thresh,
            "text_det_box_thresh": self._config.det_box_thresh,
            "text_det_unclip_ratio": self._config.det_unclip_ratio,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": self._config.use_textline_orientation,
            "enable_mkldnn": self._config.enable_mkldnn,
            "mkldnn_cache_capacity": 16,
            "cpu_threads": self._config.parallel_workers,
            "text_rec_score_thresh": self._config.score_threshold,
            "enable_hpi": False,
        }
        if use_trt:
            kwargs["use_tensorrt"] = True; kwargs["precision"] = precision_str
        # [v3.23.102] Chỉ ép tên model khi có; rỗng -> để 'lang' tự chọn model đúng ngôn ngữ.
        if self._config.detection_model_name:
            kwargs["text_detection_model_name"] = self._config.detection_model_name
        if self._config.recognition_model_name:
            # [v3.23.733] PaddleOCR-VL không phải bộ nhận dạng của pipeline PP-OCR: pipeline chạy bộ đọc NHÁP, VL đọc lại hộp.
            kwargs["text_recognition_model_name"] = (
                PADDLE_VL_DRAFT_RECOGNITION_MODEL if is_vl_recognizer(self._config.recognition_model_name)
                else self._config.recognition_model_name
            )
        kwargs.update(self._config.extra_kwargs)
        # [v3.23.303] Nhóm HỆ CHỮ VIẾT (latin/cyrillic/arabic/devanagari) không có model
        # ở phiên bản mặc định -> PaddleOCR báo "No models are available for lang=...
        # and ocr_version=None". Chỉ định phiên bản tường minh để phân giải được model.
        # KHÔNG đụng tới các ngôn ngữ khác (chúng tự phân giải đúng). Người dùng vẫn có
        # thể ghi đè qua extra_kwargs vì dòng này đặt SAU update() ở trên -> nên chỉ
        # đặt khi extra_kwargs chưa cung cấp.
        if "ocr_version" not in kwargs:
            script_group_version = resolve_script_group_version(self._config.language)
            if script_group_version:
                kwargs["ocr_version"] = script_group_version
        return kwargs

    def _needs_model_reload(self, new_config: OcrEngineConfig) -> bool:
        old = self._config
        return (
            old.detection_model_name != new_config.detection_model_name
            # [v3.23.733] Đổi qua lại PaddleOCR-VL ↔ bộ nháp của nó không cần nạp lại pipeline (VL nạp riêng khi cần).
            or _pipeline_rec_name(old.recognition_model_name) != _pipeline_rec_name(new_config.recognition_model_name)
            or old.use_tensorrt != new_config.use_tensorrt or old.precision != new_config.precision
            # [v3.23.776] Cỡ lô engine TensorRT đổi → engine khác (chỉ khi đang bật TensorRT).
            or (bool(new_config.use_tensorrt)
                and (getattr(old, "trt_det_batch", 16), getattr(old, "trt_rec_batch", 32))
                != (getattr(new_config, "trt_det_batch", 16), getattr(new_config, "trt_rec_batch", 32)))
            # [v3.23.656] Ngưỡng dò chữ / giới hạn cạnh / ngưỡng nhận dạng truyền theo TỪNG lần predict → không cần
            # nạp lại (trước đây đổi chúng nạp lại model 7–19 s; nay lượt trích xuất và Re-OCR khác ngưỡng xen kẽ nhau).
            or old.use_textline_orientation != new_config.use_textline_orientation
            or old.enable_mkldnn != new_config.enable_mkldnn or old.batch_size != new_config.batch_size or old.language != new_config.language
            or old.det_batch_size != new_config.det_batch_size
            # [v3.23.656] Thiết bị / luồng CPU / tham số thêm cũng dựng vào model.
            or old.device != new_config.device or old.parallel_workers != new_config.parallel_workers
            or old.extra_kwargs != new_config.extra_kwargs
        )

    @staticmethod
    def _import_paddle() -> tuple[Any, Any]:
        """Nạp paddle + PaddleOCR; [v3.23.767] ghi luồng nạp paddle LẦN ĐẦU (xem ``_PADDLE_THREAD``)."""
        import sys

        from subtitles_extractor.infrastructure.heavy_import_lock import HEAVY_IMPORT_LOCK

        global _PADDLE_IMPORT_THREAD
        with HEAVY_IMPORT_LOCK:                   # RLock — ``_import_paddle_modules`` lấy lại được
            fresh = "paddle" not in sys.modules
            try:
                return PaddleOcrAdapter._import_paddle_modules()
            finally:
                # Ghi cả khi PaddleOCR nạp lỗi sau khi paddle đã nạp (paddle vẫn ở sys.modules — lần thử sau không
                # "fresh" nữa); paddle bị xoá khỏi sys.modules (nạp lại) thì lần nạp lại ghi lại.
                if fresh and "paddle" in sys.modules:
                    _PADDLE_IMPORT_THREAD = threading.get_ident()
                    logger.debug("Nạp paddle trên luồng %s.", threading.current_thread().name)

    @staticmethod
    def _import_paddle_modules() -> tuple[Any, Any]:
        from subtitles_extractor.infrastructure.heavy_import_lock import HEAVY_IMPORT_LOCK
        from subtitles_extractor.infrastructure.torch_import_blocker import (
            uninstall_torch_import_blocker,
        )

        try:
            # [v3.22.6] Serialize với import torch của WhisperX (lock chung) để tránh
            # "partially initialized module 'paddle' ... circular import".
            with HEAVY_IMPORT_LOCK:
                # [v3.23.646] Nạp TRƯỚC các thư viện khoa học mà paddle/paddlex kéo vào lúc import.
                _prewarm_scientific_imports()
                try:
                    import paddle
                    from paddleocr import PaddleOCR
                except AttributeError as first_error:
                    # [v3.22.8] paddle nửa-khởi-tạo trong sys.modules (vd shm.dll lỗi
                    # lần trước) → xoá sạch rồi import lại một lần.
                    logger.warning(
                        "Nạp paddle lần 1 lỗi (%s: %s).", type(first_error).__name__, first_error,
                        exc_info=True,
                    )
                    if _native_paddle_loaded():
                        # Không thể nạp lại một paddle đã khởi tạo dở: báo đúng lỗi GỐC (xem docstring).
                        raise OcrModelLoadError(
                            f"{_PADDLE_MISSING_MSG}\n[chi tiết: {type(first_error).__name__}: {first_error}]"
                        ) from first_error
                    _purge_partial_paddle_modules()
                    import paddle
                    from paddleocr import PaddleOCR
            # [v3.23.10] Paddle đã import XONG (transformers đã fallback "no torch").
            # GỠ blocker ngay: xoá entry torch=None khỏi sys.modules để các thư viện
            # về sau (scipy/TTS true-peak) không gặp getattr(None, ...) gây crash.
            uninstall_torch_import_blocker()
            return paddle, PaddleOCR
        except (ImportError, ModuleNotFoundError, OSError) as exc:
            # [v3.23.4] Nếu import THẤT BẠI vì torch bị chặn (paddleocr build này thật
            # sự cần torch ở top-level, không try/except), gỡ blocker & thử LẠI một
            # lần. Chấp nhận rủi ro cuDNN còn hơn không OCR được.
            if "torch" in str(exc).lower():
                logger.warning(
                    "Nạp paddle lần 1 lỗi liên quan torch (%s) → gỡ bộ chặn torch rồi thử lại.",
                    exc, exc_info=True,
                )
                uninstall_torch_import_blocker()
                if _native_paddle_loaded():
                    raise OcrModelLoadError(f"{_PADDLE_MISSING_MSG}\n[chi tiết: {exc}]") from exc
                try:
                    with HEAVY_IMPORT_LOCK:
                        _purge_partial_paddle_modules()
                        import paddle
                        from paddleocr import PaddleOCR
                    return paddle, PaddleOCR
                except (ImportError, ModuleNotFoundError, OSError, AttributeError) as exc2:
                    raise OcrModelLoadError(
                        f"{_PADDLE_MISSING_MSG}\n[chi tiết: {exc2}]\n[lỗi lần đầu: {exc}]"
                    ) from exc2
            # [v3.23.296] Phân biệt lỗi NẠP DLL (thường do thiếu VC++ Redistributable
            # hoặc DLL CUDA hỏng) với thiếu package — giúp chẩn đoán trên máy người dùng.
            # LƯU Ý: paddlepaddle-gpu VẪN import được trên máy KHÔNG có GPU NVIDIA (CUDA
            # nạp lazy) → lỗi ở đây thường KHÔNG phải do thiếu GPU.
            if isinstance(exc, OSError) and "dll load failed" in str(exc).lower():
                raise OcrModelLoadError(
                    "Không nạp được thư viện native của PaddleOCR (DLL load failed). "
                    "Thường do thiếu Microsoft Visual C++ Redistributable — hãy cài bản "
                    f"mới nhất. Chi tiết: {exc}"
                ) from exc
            raise OcrModelLoadError(f"{_PADDLE_MISSING_MSG}\n[chi tiết: {exc}]") from exc
        except AttributeError as exc:
            raise OcrModelLoadError(
                f"Paddle khởi tạo lỗi (circular import): {exc}"
            ) from exc

    def _resolve_use_gpu(self, paddle_module: Any) -> bool:
        """Quyết định dùng GPU hay không cho OCR.

        [v3.23.295] Không chỉ kiểm ``is_compiled_with_cuda()`` (luôn True ở bản GPU,
        kể cả máy không NVIDIA) mà còn kiểm GPU NVIDIA THẬT qua ``device_count() > 0``.
        [v3.23.296] Tôn trọng biến môi trường ``SUBEXT_FORCE_CPU`` (ép CPU) — dùng để
        kiểm thử đường CPU ngay trên máy có GPU, hoặc làm escape hatch khi GPU trục trặc.
        -> MỘT bản build chạy được cả GPU (khi có) lẫn CPU (khi không).
        """
        want_gpu = self._config.device == DeviceKind.GPU
        force_cpu = _env_flag_enabled("SUBEXT_FORCE_CPU")
        return should_use_gpu(
            paddle_module, want_gpu=want_gpu, force_cpu=force_cpu
        )

    @staticmethod
    def _empty_paddle_vram() -> None:
        # [v3.23.764] KHÔNG tự nạp paddle (hàm chạy cả ở luồng dọn dẹp nền ``_async_cleanup``): paddle chưa nạp thì không
        # có VRAM của nó để trả. Trước đây luồng nền ``import paddle`` đúng lúc nơi khác đang nạp lại paddle → mô-đun
        # nạp dở kẹt trong ``sys.modules`` ("partially initialized module 'paddle' … has no attribute 'tensor'").
        import sys

        paddle = sys.modules.get("paddle")
        if paddle is None or not hasattr(paddle, "__version__"):
            return
        try:
            if hasattr(paddle, "device") and hasattr(paddle.device, "cuda"):
                paddle.device.cuda.empty_cache()
        except RuntimeError: pass

    @staticmethod
    def _empty_cupy_pool() -> None:
        # [v3.23.724] Bộ nạp CuPy an toàn dùng chung (không import CuPy lần hai, không ném lỗi khởi tạo).
        try:
            from subtitles_extractor.infrastructure.ocr.preprocessing.gpu_image_filters import (
                cp,
            )
        except ImportError:
            return
        if cp is None:
            return
        try:
            pool = cp.get_default_memory_pool()
            if pool is not None: pool.free_all_blocks()
        except (RuntimeError, AttributeError): pass

__all__ = ["PaddleOcrAdapter"]
