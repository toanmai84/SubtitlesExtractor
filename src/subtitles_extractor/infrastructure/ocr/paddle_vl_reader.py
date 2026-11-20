"""[v3.23.733] Đọc dòng phụ đề bằng PaddleOCR-VL-1.6 (VLM 0,9B chạy trên Paddle, không cần torch).

Đo trên 148 dòng phụ đề thật của 3 phim (15 ca khó đã kiểm bằng mắt), CPU 4 nhân:

=============================  ==========  ======  =========  ============
Bộ đọc                         đúng dòng   CER     ca khó     giây / dòng
=============================  ==========  ======  =========  ============
PP-OCRv6 small                 143/148     0,66 %  10/15      0,02
PaddleOCR-VL-1.6 (từng dòng)   146/148     0,26 %  13/15      9,9
PaddleOCR-VL-1.6 (xếp 8 dòng)  145/148             12/15      3,6
=============================  ==========  ======  =========  ============

PaddleX chạy trực tiếp chỉ nhận lô MỘT ảnh cho mô hình này (lô nhiều ảnh cần máy chủ vLLM — không hợp ứng dụng chạy
độc lập). Vì VL vốn là mô hình đọc TÀI LIỆU, ta xếp nhiều dòng phụ đề chồng lên nhau thành một "trang" (cùng chiều cao,
cách nhau một khoảng trống) và đọc một lần: kết quả là các dòng cách nhau bằng xuống dòng — nhanh gấp 2,8 lần mà vẫn đúng
145/148. Số dòng đọc ra không khớp → đọc lại từng dòng (đo: 1/19 trang).
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import queue
import re
import sys
import threading
import traceback
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

import cv2
import numpy as np

from subtitles_extractor.domain.value_objects.ocr_model_names import (
    PADDLE_VL_RECOGNITION_MODEL,
)

__all__ = [
    "LINES_PER_PAGE", "PaddleVlLineReader", "VlReadTimeoutError", "clean_vl_text", "split_page_text", "stack_lines",
]

logger = logging.getLogger(__name__)

#: Số dòng xếp vào một "trang" (đo: 8 dòng → 3,6 s/dòng trên CPU, 145/148 đúng).
LINES_PER_PAGE = 8
#: [v3.23.747] Thời hạn MỘT lần đọc (một trang ≤ 8 dòng; bình thường < 5 s trên GPU). Log 18/11 23:11: luồng trích
#: xuất kẹt vĩnh viễn trong ``_C_ops.embedding`` ở lần sinh chữ ĐẦU TIÊN của VL (GPU 3 %, VRAM còn 5,9 GB) — quá hạn
#: thì bỏ VL cho phiên, giữ bản đọc PP-OCR, lượt chạy đi tiếp thay vì treo hàng giờ.
VL_READ_TIMEOUT_GPU_SEC = 120.0
VL_READ_TIMEOUT_CPU_SEC = 1800.0
#: Nạp mô hình (lần đầu còn TẢI ~2 GB) — thời hạn rộng.
VL_LOAD_TIMEOUT_SEC = 3600.0
_LINE_HEIGHT = 48
_LINE_GAP = 24
_QUERY = "OCR:"
_MAX_TOKENS_PER_LINE = 24


_LATEX_COMMAND = re.compile(r"\\[A-Za-z]+\*?\{([^{}]*)\}")
_LATEX_DELIMITERS = re.compile(r"\\[()\[\]]|\$+")
_MARKDOWN = re.compile(r"\*\*|__|`+|^#+\s*")


def clean_vl_text(text: str) -> str:
    """Gỡ định dạng LaTeX / markdown mà VL (mô hình đọc TÀI LIỆU) tự thêm, giữ nguyên chữ bên trong.

    Đo trên 148 dòng phụ đề xếp 8 dòng/trang: 5 dòng bị bọc ``\\(\\underline{\\text{灵力}}\\)`` (VL tưởng chữ có gạch chân)
    — chữ bên trong vẫn đúng.
    """
    out = str(text or "")
    for _ in range(4):                       # lệnh lồng nhau: \underline{\text{…}}
        unwrapped = _LATEX_COMMAND.sub(r"\1", out)
        if unwrapped == out:
            break
        out = unwrapped
    out = _LATEX_DELIMITERS.sub("", out)
    out = _MARKDOWN.sub("", out).replace("\\", "")
    return " ".join(out.split())


def stack_lines(crops: Sequence[np.ndarray], height: int = _LINE_HEIGHT, gap: int = _LINE_GAP) -> np.ndarray:
    """Xếp các ảnh dòng (BGR) thành một trang: cùng chiều cao ``height``, căn trái, cách nhau ``gap`` px, nền đen."""
    resized = []
    for crop in crops:
        h, w = crop.shape[:2]
        width = max(8, int(round(w * height / max(1, h))))
        resized.append(cv2.resize(crop, (width, height), interpolation=cv2.INTER_AREA if h > height else cv2.INTER_CUBIC))
    page_width = max(img.shape[1] for img in resized) + 2 * gap
    page = np.zeros((len(resized) * (height + gap) + gap, page_width, 3), np.uint8)
    for index, img in enumerate(resized):
        top = gap + index * (height + gap)
        page[top:top + height, gap:gap + img.shape[1]] = img[..., :3] if img.ndim == 3 else img[..., None]
    return page


def split_page_text(text: str, count: int) -> list[str] | None:
    """Tách kết quả đọc một trang thành ``count`` dòng; ``None`` nếu số dòng không khớp (đọc lại từng dòng)."""
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    return lines if len(lines) == count else None


def _result_text(output: Any) -> str:
    target = getattr(output, "res", output)
    if isinstance(target, dict):
        inner = target.get("res")
        if isinstance(inner, dict):
            target = inner
        return str(target.get("result", "") or "")
    return str(getattr(target, "result", "") or "")


def _bind_thread_device(device: str) -> None:
    """[v3.23.739/740] Đưa LUỒNG đang đọc VL về chế độ đồ thị động + gán thiết bị trước khi nạp / dự đoán.

    PaddleX gọi ``paddle.amp.is_bfloat16_supported()`` không truyền thiết bị → ``_current_expected_place()``, hàm này trả
    ``core.Place()`` (= ``Place(undefined:0)``) khi luồng ở chế độ TĨNH PIR. Chế độ động là tracer RIÊNG từng luồng, chép
    từ biến toàn cục lúc luồng được tạo: chỉ cần ở đâu đó trong tiến trình từng đặt tracer về ``None`` (như
    ``paddle.enable_static()``) là mọi luồng tạo sau đều ở chế độ tĩnh → VL không chạy ở MỌI phiên trên máy chủ dự án
    (log 17–18/11). 3.23.739 chỉ ``set_device`` (biến TOÀN CỤC) nên không đủ — log 18/11 07:05 vẫn lỗi. Tái hiện bằng
    Paddle thật: ``enable_static`` ở một luồng → luồng mới ``is_bfloat16_supported()`` lỗi đúng ``Place(undefined:0)``;
    ``disable_static`` trong luồng đó → hết lỗi, luồng khác giữ nguyên chế độ.
    """
    try:
        import paddle
    except ImportError:
        return
    in_dynamic = getattr(paddle, "in_dynamic_mode", None)
    disable_static = getattr(paddle, "disable_static", None)
    try:
        if callable(in_dynamic) and callable(disable_static) and not in_dynamic():
            disable_static()
            logger.info("Luồng đọc PaddleOCR-VL đang ở chế độ tĩnh (PIR) — chuyển về chế độ động cho luồng này.")
    except Exception as exc:  # noqa: BLE001 — để create_model / predict báo lỗi thật
        logger.debug("Không chuyển được luồng đọc VL về chế độ động: %s", exc)
    set_device = getattr(paddle, "set_device", None)
    if not callable(set_device):
        return
    try:
        set_device(device)
    except Exception as exc:  # noqa: BLE001 — thiết bị không có: để create_model / predict báo lỗi thật
        logger.debug("Không gán được thiết bị %s cho luồng đọc VL: %s", device, exc)


class VlReadTimeoutError(RuntimeError):
    """PaddleOCR-VL không trả kết quả trong thời hạn (lệnh GPU không bao giờ xong) — bỏ VL cho phiên."""


_T = TypeVar("_T")


class _VlThread:
    """[v3.23.747] Luồng RIÊNG sở hữu mô hình VL: mọi lần nạp / đọc / trả mô hình chạy trên luồng này (chế độ đồ thị
    động + thiết bị gán đúng một lần cho luồng). Luồng nền ``daemon`` → kẹt trong lệnh GPU cũng không chặn ứng dụng thoát."""

    def __init__(self) -> None:
        self._jobs: queue.Queue[tuple[Callable[[], Any], concurrent.futures.Future[Any]]] = queue.Queue()
        self.thread = threading.Thread(target=self._run, name="paddle-vl", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while True:
            job, future = self._jobs.get()
            if not future.set_running_or_notify_cancel():
                continue
            try:
                future.set_result(job())
            except BaseException as exc:  # noqa: BLE001 — chuyển nguyên lỗi về luồng gọi
                future.set_exception(exc)

    def call(self, job: Callable[[], _T], timeout: float) -> _T:
        future: concurrent.futures.Future[_T] = concurrent.futures.Future()
        self._jobs.put((job, future))
        return future.result(timeout=timeout)

    def stack(self) -> str:
        frame = sys._current_frames().get(self.thread.ident or -1)  # noqa: SLF001 — chẩn đoán treo
        return "".join(traceback.format_stack(frame)) if frame is not None else ""


class PaddleVlLineReader:
    """Đọc ảnh dòng chữ bằng PaddleOCR-VL; mô hình nạp khi cần lần đầu, cùng thiết bị với OCR chính.

    Lỗi nạp mô hình được ném ra (người gọi tự lùi về bộ nhận dạng PP-OCR và ghi log).
    """

    def __init__(self, model_name: str = PADDLE_VL_RECOGNITION_MODEL, lines_per_page: int = LINES_PER_PAGE) -> None:
        self._model_name = model_name
        self._lines_per_page = max(1, int(lines_per_page))
        self._model: Any | None = None
        self._device: str | None = None
        self._lock = threading.Lock()
        self._thread: _VlThread | None = None
        self._stuck: str = ""

    @property
    def model_name(self) -> str:
        return self._model_name

    def release(self) -> None:
        """Trả mô hình — trên luồng VL (nơi tạo ra nó); luồng VL đang kẹt thì chỉ bỏ tham chiếu."""
        thread = self._thread
        if thread is None or self._stuck:
            self._model = None
            self._device = None
            return

        def drop() -> None:
            with self._lock:
                self._model = None
                self._device = None

        try:
            thread.call(drop, timeout=30.0)
        except concurrent.futures.TimeoutError:
            logger.warning("Trả mô hình PaddleOCR-VL quá hạn — luồng VL không phản hồi.")

    def _ensure(self, device: str) -> Any:
        if self._model is not None and self._device == device:
            return self._model
        # PaddleOCR-VL cần sentencepiece (tokenizer) và einops (bộ mã hoá ảnh) — đã khai báo trong gói phát hành.
        from paddlex import create_model  # type: ignore[import-untyped]

        os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "huggingface")
        self._model = create_model(model_name=self._model_name, device=device)
        self._device = device
        logger.info("Nạp bộ đọc dòng %s (%s).", self._model_name, device)
        return self._model

    def _read_one(self, model: Any, crop: np.ndarray) -> str:
        outputs = list(model.predict({"image": crop, "query": _QUERY}, max_new_tokens=2 * _MAX_TOKENS_PER_LINE))
        return clean_vl_text(_result_text(outputs[0])) if outputs else ""

    def read(self, crops_bgr: Sequence[np.ndarray], device: str = "cpu") -> list[str]:
        """Đọc từng ảnh dòng (BGR); trả chuỗi cùng thứ tự (rỗng nếu không đọc được).

        [v3.23.747] Chạy trên luồng VL riêng, có thời hạn; quá hạn → :class:`VlReadTimeoutError` (người gọi tắt VL cho
        phiên, giữ bản đọc PP-OCR) kèm ngăn xếp luồng VL trong nhật ký.
        """
        if not crops_bgr:
            return []
        if self._stuck:
            raise VlReadTimeoutError(self._stuck)
        if self._thread is None:
            self._thread = _VlThread()
        thread = self._thread
        crops = list(crops_bgr)
        self._call(thread, lambda: self._load(device), VL_LOAD_TIMEOUT_SEC, "nạp mô hình")
        limit = VL_READ_TIMEOUT_GPU_SEC if device == "gpu" else VL_READ_TIMEOUT_CPU_SEC
        pages = max(1, -(-len(crops) // self._lines_per_page))
        return self._call(thread, lambda: self._read_on_thread(crops, device), limit * pages, "đọc")

    def _call(self, thread: _VlThread, job: Callable[[], _T], timeout: float, what: str) -> _T:
        try:
            return thread.call(job, timeout)
        except concurrent.futures.TimeoutError:
            self._stuck = (f"PaddleOCR-VL {what} quá {timeout:.0f} s không trả kết quả (lệnh GPU không xong) — "
                           "bỏ VL cho phiên này")
            logger.warning("%s. Ngăn xếp luồng VL:\n%s", self._stuck, thread.stack())
            raise VlReadTimeoutError(self._stuck) from None

    def _load(self, device: str) -> None:
        with self._lock:
            _bind_thread_device(device)
            self._ensure(device)

    def _read_on_thread(self, crops_bgr: Sequence[np.ndarray], device: str) -> list[str]:
        with self._lock:
            _bind_thread_device(device)
            model = self._ensure(device)
            texts: list[str] = []
            for start in range(0, len(crops_bgr), self._lines_per_page):
                group = list(crops_bgr[start:start + self._lines_per_page])
                if len(group) == 1:
                    texts.append(self._read_one(model, group[0]))
                    continue
                outputs = list(model.predict(
                    {"image": stack_lines(group), "query": _QUERY},
                    max_new_tokens=_MAX_TOKENS_PER_LINE * len(group),
                ))
                lines = split_page_text(_result_text(outputs[0]) if outputs else "", len(group))
                if lines is None:
                    lines = [self._read_one(model, crop) for crop in group]
                texts.extend(clean_vl_text(line) for line in lines)
            return texts
