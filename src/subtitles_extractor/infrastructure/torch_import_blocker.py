"""Chặn ``import torch`` TẠM THỜI ở tiến trình chính để bảo vệ PaddleOCR.

VẤN ĐỀ (xác nhận qua PaddleOCR issue #14475, pytorch #147274): PaddleOCR 3.x dùng
PaddleX/transformers, vốn TỰ DÒ torch bằng ``importlib.util.find_spec("torch")``; nếu
thấy có, chúng import torch → torch chiếm cuDNN/cuBLAS DLL → paddle lỗi WinError 127.

CÁCH HOẠT ĐỘNG: gán ``sys.modules["torch"] = None`` (cùng torchvision/torchaudio).
Khi đó ``find_spec("torch")`` trả None → transformers/paddlex coi như KHÔNG có torch
→ fallback (log "None of PyTorch... found") → paddle dùng cuDNN riêng, chạy ngon.
Đây là cách ĐÃ ĐƯỢC KIỂM CHỨNG chạy với paddle trong thực tế.

NHƯNG ``sys.modules["torch"] = None`` gây tác dụng phụ: thư viện khác (vd
scipy.array_api_compat) làm ``getattr(sys.modules["torch"], "Tensor")`` → AttributeError
trên None. VÌ VẬY blocker này chỉ nên TẠM THỜI: gọi ``install`` trước khi import
paddle, rồi ``uninstall`` NGAY SAU KHI paddle import xong (xoá entry None). Sau đó
tiến trình chính không còn None trong sys.modules → scipy/TTS an toàn. Tiến trình
chính KHÔNG bao giờ tự import torch (WhisperX chạy ở tiến trình con riêng) nên việc
gỡ blocker không khiến torch bị nạp nhầm.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager

_BLOCKED_ROOTS: tuple[str, ...] = ("torch", "torchvision", "torchaudio")

_INSTALLED = False


def install_torch_import_blocker() -> None:
    """Chặn torch bằng ``sys.modules[...] = None`` (find_spec → None → fallback).

    Idempotent. Chỉ gán None nếu module CHƯA được nạp thật (tránh đè module thật).
    """
    global _INSTALLED
    if _INSTALLED:
        return
    for root in _BLOCKED_ROOTS:
        existing = sys.modules.get(root, "absent")
        if existing == "absent" or existing is None:
            sys.modules[root] = None  # type: ignore[assignment]
    _INSTALLED = True


def uninstall_torch_import_blocker() -> None:
    """Gỡ chặn: XOÁ các entry None khỏi sys.modules.

    Sau khi gỡ, ``sys.modules`` không còn ``torch=None`` → mọi thư viện về sau (scipy,
    TTS true-peak…) gọi ``sys.modules["torch"]`` sẽ gặp KeyError và xử lý an toàn,
    thay vì ``getattr(None, ...)`` gây AttributeError.
    """
    global _INSTALLED
    for root in _BLOCKED_ROOTS:
        if sys.modules.get(root, "absent") is None:
            del sys.modules[root]
    _INSTALLED = False


def is_torch_import_blocked() -> bool:
    """Trả True nếu blocker đang được cài (phục vụ kiểm thử/chẩn đoán)."""
    return _INSTALLED


@contextmanager
def torch_isolation() -> Iterator[None]:
    """[v3.23.191] Cô lập torch TRONG SUỐT khối ``with`` (cho engine thuần ONNX/GGUF).

    Khác PaddleOCR (chặn torch chỉ lúc IMPORT rồi gỡ ngay), một số engine như VieNeu-TTS
    còn import torch TRỄ (lazy) lúc INFERENCE qua backend — nên phải giữ chặn suốt cả
    quá trình chạy, không chỉ lúc nạp. Context manager này cài blocker khi vào khối và
    GỠ khi ra (kể cả có ngoại lệ), khôi phục ``sys.modules`` sạch cho scipy/DSP về sau.

    Idempotent: chỉ cài nếu blocker CHƯA được cài trước đó (tôn trọng blocker bên ngoài);
    trong trường hợp đã có sẵn thì KHÔNG gỡ (để bên cài ngoài chịu trách nhiệm gỡ).

    Yields:
        None. Trong khối ``with`` mọi ``import torch`` bị chặn -> engine fallback ONNX.
    """
    already_blocked = is_torch_import_blocked()
    if not already_blocked:
        install_torch_import_blocker()
    try:
        yield
    finally:
        if not already_blocked:
            uninstall_torch_import_blocker()


def is_torch_dll_load_error(error: BaseException) -> bool:
    """Nhận diện lỗi tải DLL của torch/cuDNN (hàm thuần, không side-effect).

    Lỗi điển hình trên Windows khi bản torch CUDA không khớp driver/cuDNN:
    ``[WinError 127] ... torch\\lib\\cudnn_engines_precompiled64_9.dll ...``. Dùng để
    các adapter bắt đúng loại lỗi và đưa thông báo hướng dẫn thay vì traceback thô.

    Args:
        error: Ngoại lệ bắt được (thường là OSError/ImportError khi import torch).

    Returns:
        True nếu thông điệp lỗi khớp mẫu lỗi tải DLL torch/cuDNN.
    """
    message = str(error).lower()
    mentions_torch = "torch" in message or "cudnn" in message
    mentions_dll_fail = (
        "winerror 127" in message
        or "could not be found" in message
        or (".dll" in message and ("load" in message or "loading" in message))
    )
    return mentions_torch and mentions_dll_fail


def torch_dll_error_message(error: BaseException, engine_name: str) -> str:
    """Tạo thông báo tiếng Việt hướng dẫn khắc phục lỗi tải DLL torch (hàm thuần).

    Args:
        error: Ngoại lệ gốc.
        engine_name: Tên engine để chèn vào thông báo (vd "WhisperX").

    Returns:
        Chuỗi thông báo tiếng Việt có hướng dẫn cài lại torch khớp CUDA.
    """
    return (
        f"{engine_name} không nạp được vì thư viện PyTorch/cuDNN trên máy bị lỗi "
        f"(bản torch không khớp driver CUDA). Đây là engine BẮT BUỘC dùng PyTorch nên "
        f"không thể né bằng CPU. Hãy cài lại torch khớp CUDA của máy, ví dụ:\n"
        f"  pip uninstall torch torchaudio -y\n"
        f"  pip install torch torchaudio --index-url "
        f"https://download.pytorch.org/whl/cu124\n"
        f"Hoặc dùng engine khác (VieNeu-TTS chạy được trên CPU/ONNX không cần torch).\n"
        f"Chi tiết: {error}"
    )


__all__ = [
    "install_torch_import_blocker",
    "uninstall_torch_import_blocker",
    "is_torch_import_blocked",
    "torch_isolation",
    "is_torch_dll_load_error",
    "torch_dll_error_message",
]
