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
from typing import Any, Final

import cv2
import numpy as np

from subtitles_extractor.domain.entities.ocr_frame_result import OcrFrameResult, OcrTextBox
from subtitles_extractor.domain.exceptions import OcrInferenceError, OcrModelLoadError
from subtitles_extractor.domain.ports.ocr_engine_port import OcrEngineConfig
from subtitles_extractor.domain.value_objects.device_kind import DeviceKind, PrecisionMode
from subtitles_extractor.infrastructure.ocr.device_probe import should_use_gpu
from subtitles_extractor.infrastructure.ocr.preprocessing.image_filters import apply_cpu_preprocessing_pipeline
from subtitles_extractor.infrastructure.ocr.result_parser import parse_paddle_result
from subtitles_extractor.infrastructure.ocr.script_group_versions import (
    resolve_script_group_version,
)

logger = logging.getLogger(__name__)

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

    from subtitles_extractor.infrastructure.torch_import_blocker import torch_none_bypassed

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

    @property
    def is_initialized(self) -> bool:
        # [v3.23.650] KHÔNG lấy ``_init_lock``: ``initialize()`` giữ khoá suốt lúc import paddle + nạp mô hình (7–19 s),
        # mà thuộc tính này được gọi từ luồng giao diện (showEvent của trang Trích xuất/Chỉnh sửa, preload) →
        # cửa sổ đơ tới khi mô hình nạp xong. Đọc một tham chiếu là nguyên tử trong CPython.
        return self._model is not None

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
                paddle_module, paddle_ocr_cls = self._import_paddle()
                use_gpu: bool = self._resolve_use_gpu(paddle_module)
                self._model = self._create_model_with_device_fallback(
                    paddle_ocr_cls, use_gpu=use_gpu
                )
                self._apply_det_batch_size(self._model)
                _rearm_fault_handler()
        if self._release_after_init:
            # [v3.23.650] release() được gọi (đóng app) lúc khoá đang bận: dọn ngay khi nạp xong (kể cả đường thoát sớm).
            self._release_after_init = False
            self.release()

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
        use_trt: bool = self._config.use_tensorrt and use_gpu
        kwargs: dict[str, Any] = self._build_init_kwargs(use_gpu=use_gpu, use_trt=use_trt)
        logger.info(
            "Khởi tạo PaddleOCR: device=%s, det=%s, rec=%s",
            "gpu" if use_gpu else "cpu",
            self._config.detection_model_name,
            self._config.recognition_model_name,
        )
        try:
            model = self._try_create_model_with_retry(paddle_ocr_cls, kwargs)
        except OcrModelLoadError:
            if not use_gpu:
                # Đã ở CPU mà vẫn lỗi -> lỗi thật, không có gì để lùi thêm.
                raise
            logger.warning(
                "Khởi tạo OCR trên GPU thất bại — tự động lùi về CPU (chậm hơn nhưng "
                "vẫn chạy). Kiểm tra driver NVIDIA nếu muốn tăng tốc GPU."
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

    def _release_locked_safely(self) -> None:
        if not self._init_lock.acquire(blocking=False):
            self._release_after_init = True
            return
        try:
            self._release_locked()
        finally:
            self._init_lock.release()

    def _release_locked(self) -> None:
        if self._model is None: return
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
        if self._model is None: return
        del self._model
        self._model = None
        self._active_device = None
        gc.collect()
        self._empty_paddle_vram()
        self._empty_cupy_pool()

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
            bgr_list = [p.image_bgr for p in prepared_images]
            try:
                raw_results = self._predict_batch(bgr_list, frame_indices, run_config)
            finally:
                if self._release_after_infer:
                    self._release_after_infer = False
                    self._release_locked_safely()

        # [CRITICAL BUG FIX]: Đã thêm "prepared_images" vào trong zip()
        return [
            self._scale_back_result(parse_paddle_result(raw, frame_idx, ts), prep.scale, prep.border_px)
            for raw, frame_idx, ts, prep in zip(raw_results, frame_indices, timestamps_sec, prepared_images, strict=True)
        ]

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

        scale = float(working_rgb.shape[0]) / orig_height
        border_px = int(pre.border_thickness_px) if pre.add_white_border else 0
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
            kwargs["text_recognition_model_name"] = self._config.recognition_model_name
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
            old.detection_model_name != new_config.detection_model_name or old.recognition_model_name != new_config.recognition_model_name
            or old.use_tensorrt != new_config.use_tensorrt or old.precision != new_config.precision
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
        try: import paddle
        except ImportError: return
        try:
            if hasattr(paddle, "device") and hasattr(paddle.device, "cuda"):
                paddle.device.cuda.empty_cache()
        except RuntimeError: pass

    @staticmethod
    def _empty_cupy_pool() -> None:
        try: import cupy as cp
        except ImportError: return
        try:
            pool = cp.get_default_memory_pool()
            if pool is not None: pool.free_all_blocks()
        except (RuntimeError, AttributeError): pass

__all__ = ["PaddleOcrAdapter"]
