"""[v3.23.751] PaddleOCR-VL chạy ở TIẾN TRÌNH RIÊNG — lệnh GPU kẹt không còn kéo cả ứng dụng treo theo.

Bằng chứng (phim 大乾悍刀行 84 tập, máy chủ dự án, RTX 3070 Ti): 5 lượt cùng phim cùng cấu hình — 3 lượt bước soát câu
bằng VL chạy trơn ~1 s/câu, 2 lượt (17:01 18/11 và 08:14 19/11, cả khi máy được giữ thức ở 3.23.750) lần sinh chữ ĐẦU
TIÊN của VL kẹt trong ``_C_ops.embedding`` (VRAM còn 5,9 GB, GPU 3 %). Lưới an toàn 3.23.747 (luồng VL có thời hạn) bỏ
VL sau 120 s, nhưng NGỮ CẢNH CUDA của tiến trình đã kẹt: lệnh GPU kế tiếp ở luồng chính (mở NVDEC) treo luôn — log
dừng ở 08:16:10. VL (đồ thị ĐỘNG) dùng chung tiến trình với Paddle Inference (đồ thị tĩnh PIR — phải ép luồng VL "rời
chế độ tĩnh"), NVDEC và CuPy.

Tiến trình riêng có ngữ cảnh CUDA riêng: kẹt / sập → giết tiến trình con (trả VRAM), bộ chuyển đổi tắt VL cho phiên và
giữ bản đọc PP-OCR; GPU của tiến trình chính không bị ảnh hưởng. Trả mô hình = tắt tiến trình con.
"""

from __future__ import annotations

import logging
import math
import multiprocessing
import os
import sys
import threading
import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from subtitles_extractor.domain.value_objects.ocr_model_names import (
    PADDLE_VL_RECOGNITION_MODEL,
)
from subtitles_extractor.infrastructure.ocr import paddle_vl_reader as _vl

logger = logging.getLogger(__name__)

#: Tiến trình con nạp xong Paddle (chưa nạp mô hình) trong thời hạn này.
WORKER_START_TIMEOUT_SEC = 600.0


def _vl_worker_main(conn: Any, prepare: bool) -> None:  # pragma: no cover — chạy ở tiến trình con
    """Vòng lặp tiến trình con: nhận ``("read", ảnh, thiết bị)`` → trả ``("ok", chữ)`` / ``("error", lý do)``.

    Mọi lỗi (kể cả lúc chuẩn bị / nạp) được GỬI về tiến trình chính rồi thoát êm — không để ngoại lệ thoát ra
    ``multiprocessing`` (exe cửa sổ: bật hộp thoại "Unhandled exception in script" trước mặt người dùng, log 19/11 09:14).
    """
    try:
        if prepare:
            from subtitles_extractor.composition.child_process_setup import (
                prepare_child_process,
            )

            prepare_child_process()
            # Đúng trình tự nạp của tiến trình chính (``PaddleOcrAdapter._import_paddle``): nạp trước scipy/sklearn, nạp
            # paddle + paddleocr khi torch còn bị chặn, rồi GỠ chặn. Tự làm khác đi là gặp cạm bẫy CLAUDE.md
            # ("'NoneType' object has no attribute 'Tensor'" khi sklearn nạp lần đầu lúc torch bị chặn).
            from subtitles_extractor.infrastructure.ocr.paddle_ocr_adapter import (
                PaddleOcrAdapter,
            )

            PaddleOcrAdapter._import_paddle()
        reader = _vl.PaddleVlLineReader()
        conn.send(("ready", os.getpid()))
    except BaseException as exc:  # noqa: BLE001 — báo về tiến trình chính thay vì văng hộp thoại
        _send_quietly(conn, ("error", f"{type(exc).__name__}: {exc}"[:800]))
        return
    try:
        while True:
            message = conn.recv()
            if message[0] == "quit":
                break
            if message[0] == "read":
                _kind, crops, device = message
                try:
                    conn.send(("ok", reader.read(crops, device=device)))
                except Exception as exc:  # noqa: BLE001 — chuyển lý do về tiến trình chính
                    conn.send(("error", f"{type(exc).__name__}: {exc}"[:800]))
    except BaseException:  # noqa: BLE001 — ống đứt / bị tắt: thoát êm
        return


def _send_quietly(conn: Any, message: tuple[str, Any]) -> None:  # pragma: no cover — tiến trình con
    try:
        conn.send(message)
    except BaseException:  # noqa: BLE001
        pass


class PaddleVlProcessReader:
    """Cùng giao diện :class:`~.paddle_vl_reader.PaddleVlLineReader` (``read`` / ``release`` / ``model_name``), mô hình
    nằm ở tiến trình con. Lỗi / quá hạn → ném lỗi (bộ chuyển đổi tắt VL cho phiên, giữ bản đọc PP-OCR)."""

    def __init__(self, model_name: str = PADDLE_VL_RECOGNITION_MODEL, *,
                 worker: Callable[[Any, bool], None] = _vl_worker_main, prepare: bool = True) -> None:
        self._model_name = model_name
        self._worker = worker
        self._prepare = prepare
        self._lock = threading.Lock()
        self._process: Any = None
        self._conn: Any = None
        self._loaded = False
        self._stuck = ""

    @property
    def model_name(self) -> str:
        return self._model_name

    def _start(self) -> None:
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe()
        process = context.Process(target=self._worker, args=(child, self._prepare), name="paddle-vl", daemon=True)
        process.start()
        child.close()
        self._process, self._conn, self._loaded = process, parent, False
        if not parent.poll(WORKER_START_TIMEOUT_SEC):
            self._terminate()
            raise _vl.VlReadTimeoutError(f"Tiến trình PaddleOCR-VL không khởi động sau {WORKER_START_TIMEOUT_SEC:.0f} s")
        kind, payload = self._receive()
        if kind != "ready":
            self._terminate()
            raise RuntimeError(f"Tiến trình PaddleOCR-VL lỗi khi khởi động: {payload}")
        logger.info("Đã bật tiến trình riêng cho PaddleOCR-VL (pid %s).", payload)

    def _receive(self) -> tuple[str, Any]:
        try:
            message = self._conn.recv()
        except (EOFError, OSError) as exc:
            code = self._process.exitcode if self._process is not None else "?"
            self._terminate()
            raise RuntimeError(f"Tiến trình PaddleOCR-VL kết thúc bất thường (mã {code})") from exc
        return message[0], message[1]

    def read(self, crops_bgr: Sequence[np.ndarray], device: str = "cpu") -> list[str]:
        if not crops_bgr:
            return []
        with self._lock:
            if self._stuck:
                raise _vl.VlReadTimeoutError(self._stuck)
            if self._process is None or not self._process.is_alive():
                self._start()
            pages = max(1, math.ceil(len(crops_bgr) / _vl.LINES_PER_PAGE))
            limit = (_vl.VL_READ_TIMEOUT_GPU_SEC if device == "gpu" else _vl.VL_READ_TIMEOUT_CPU_SEC) * pages
            if not self._loaded:
                limit += _vl.VL_LOAD_TIMEOUT_SEC                  # lần đầu: tiến trình con còn nạp (có thể tải) mô hình
            started = time.monotonic()
            self._conn.send(("read", [np.ascontiguousarray(crop) for crop in crops_bgr], device))
            if not self._conn.poll(limit):
                self._stuck = (f"PaddleOCR-VL (tiến trình riêng) quá {limit:.0f} s không trả kết quả — đã dừng tiến "
                               "trình đó, bỏ VL cho phiên này")
                logger.warning("%s.", self._stuck)
                self._terminate()
                raise _vl.VlReadTimeoutError(self._stuck)
            kind, payload = self._receive()
            if kind != "ok":
                self._terminate()                                 # lỗi GPU trong tiến trình con: trả VRAM ngay
                raise RuntimeError(str(payload))
            self._loaded = True
            logger.debug("PaddleOCR-VL (tiến trình riêng) đọc %d ảnh trong %.1f s.", len(crops_bgr),
                         time.monotonic() - started)
            return [str(text) for text in payload]

    def release(self) -> None:
        """Tắt tiến trình con (trả toàn bộ VRAM của VL); lần đọc sau bật lại."""
        with self._lock:
            if self._process is not None and self._process.is_alive() and self._conn is not None:
                try:
                    self._conn.send(("quit",))
                except (OSError, ValueError):
                    pass
                self._process.join(timeout=10)
            self._terminate()

    def _terminate(self) -> None:
        process, conn = self._process, self._conn
        self._process, self._conn, self._loaded = None, None, False
        if conn is not None:
            try:
                conn.close()
            except OSError:
                pass
        if process is not None and process.is_alive():
            process.terminate()
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join(timeout=10)


def make_vl_reader() -> Any:
    """Bộ đọc VL cho bộ chuyển đổi: tiến trình riêng trên Windows (máy chủ dự án), trong tiến trình ở nơi khác.

    ``SE_VL_MODE=process|thread`` để chọn tay (kiểm thử / chẩn đoán).
    """
    mode = os.environ.get("SE_VL_MODE", "process" if sys.platform == "win32" else "thread")
    return PaddleVlProcessReader() if mode == "process" else _vl.PaddleVlLineReader()


__all__ = ["PaddleVlProcessReader", "make_vl_reader"]
