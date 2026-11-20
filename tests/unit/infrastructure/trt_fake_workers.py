"""Tiến trình con giả cho ``tensorrt_selftest`` (mức mô-đun: ``spawn`` cần nhập được theo tên)."""

from __future__ import annotations

import time
from typing import Any


def crash_worker(conn: Any, variant: str, payload: dict, det_batch: int, fault_path: str, prepare: bool) -> None:
    import ctypes
    import faulthandler

    handle = open(fault_path, "w", encoding="utf-8")  # noqa: SIM115
    faulthandler.enable(handle, all_threads=True)
    ctypes.string_at(0)                                       # truy cập bộ nhớ sai THẬT — như access violation của TRT


def crash_only_cross_thread(conn: Any, variant: str, payload: dict, det_batch: int, fault_path: str,
                            prepare: bool) -> None:
    if variant == "khác luồng":                              # bố cục CŨ sập, luồng Paddle riêng chạy được
        crash_worker(conn, variant, payload, det_batch, fault_path, prepare)
    conn.send(("ok", f"{det_batch} ảnh"))


def ok_worker(conn: Any, variant: str, payload: dict, det_batch: int, fault_path: str, prepare: bool) -> None:
    conn.send(("ok", f"{det_batch} ảnh, {payload['init_kwargs'].get('precision')}"))


def error_worker(conn: Any, variant: str, payload: dict, det_batch: int, fault_path: str, prepare: bool) -> None:
    conn.send(("error", "OcrModelLoadError: dựng engine lỗi"))


def hang_worker(conn: Any, variant: str, payload: dict, det_batch: int, fault_path: str, prepare: bool) -> None:
    time.sleep(600)


def oom_worker(conn: Any, variant: str, payload: dict, det_batch: int, fault_path: str, prepare: bool) -> None:
    conn.send(("error", "predict: RuntimeError: CUDA error: out of memory"))


def terminated_worker(conn: Any, variant: str, payload: dict, det_batch: int, fault_path: str,
                      prepare: bool) -> None:
    import os
    import signal

    os.kill(os.getpid(), signal.SIGTERM)                       # bị dừng từ ngoài (không phải sập)
    time.sleep(30)


def preload_paddle_then_real(conn: Any, variant: str, payload: dict, det_batch: int, fault_path: str,
                             prepare: bool) -> None:
    """Như tiến trình con trên Windows (log 19/11 18:47): paddle đã nạp trên luồng CHÍNH trước mã tự kiểm."""
    from subtitles_extractor.infrastructure.ocr import tensorrt_selftest
    from subtitles_extractor.infrastructure.ocr.paddle_ocr_adapter import PaddleOcrAdapter

    PaddleOcrAdapter._import_paddle_modules()                 # nạp paddle KHÔNG ghi luồng (như mã lạ nạp sớm)
    tensorrt_selftest._selftest_worker(conn, variant, payload, det_batch, fault_path, prepare)


def import_paddle_in_prepare(conn: Any, variant: str, payload: dict, det_batch: int, fault_path: str,
                             prepare: bool) -> None:
    """Mã "lạ" nạp paddle trong lúc chuẩn bị tiến trình — tự kiểm phải ghi được ai nạp (luồng + ngăn xếp)."""
    import os

    from subtitles_extractor.infrastructure.ocr import tensorrt_selftest

    os.environ["SE_TEST_IMPORT_PADDLE_IN_PREPARE"] = "1"
    tensorrt_selftest._selftest_worker(conn, variant, payload, det_batch, fault_path, prepare)


def crash_on_build_ok_direct(conn: Any, variant: str, payload: dict, det_batch: int, fault_path: str,
                             prepare: bool) -> None:
    if variant.startswith("dựng engine"):
        crash_worker(conn, variant, payload, det_batch, fault_path, prepare)
    conn.send(("ok", "kết quả GIỐNG Paddle GPU"))


def ok_with_data_worker(conn: Any, variant: str, payload: dict, det_batch: int, fault_path: str,
                        prepare: bool) -> None:
    """[v3.23.774] Tin nhắn ba phần (kèm dấu tệp engine)."""
    conn.send(("ok", "kết quả GIỐNG Paddle GPU", {"engines": {"x": 1}}))
