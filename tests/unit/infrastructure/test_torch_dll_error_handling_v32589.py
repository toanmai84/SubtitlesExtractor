"""[v3.23.189] Test xử lý lỗi tải DLL torch/cuDNN cho VieNeu (chặn torch) + F5 (báo lỗi).

Lỗi thực tế từ log máy người dùng: cả VieNeu và F5 chết tại ``import torch`` -> nạp
``cudnn_engines_precompiled64_9.dll`` hỏng (WinError 127). Bản torch CUDA không khớp
driver 13.2 quá mới. Áp KINH NGHIỆM từ v3.23.3 (torch_import_blocker đã kiểm chứng cho
PaddleOCR):
- VieNeu: chặn import torch -> fallback ONNX (torch-free) -> né lỗi tận gốc.
- F5-TTS: BẮT BUỘC torch -> không né được -> bắt lỗi sớm + thông báo hướng dẫn rõ ràng.
"""

from __future__ import annotations

from subtitles_extractor.infrastructure.torch_import_blocker import (
    is_torch_dll_load_error,
    is_torch_import_blocked,
    torch_dll_error_message,
)

# ── is_torch_dll_load_error ──────────────────────────────────────────────


def test_detects_winerror_127_cudnn() -> None:
    error = OSError(
        "[WinError 127] The specified procedure could not be found. "
        "Error loading torch\\lib\\cudnn_engines_precompiled64_9.dll"
    )
    assert is_torch_dll_load_error(error) is True


def test_detects_cudnn_load_failure() -> None:
    assert is_torch_dll_load_error(OSError("Failed loading cudnn .dll")) is True


def test_ignores_non_torch_error() -> None:
    assert is_torch_dll_load_error(ValueError("voice not found")) is False


def test_ignores_torch_error_without_dll() -> None:
    # Lỗi có "torch" nhưng không phải lỗi tải DLL -> không nhận nhầm.
    assert is_torch_dll_load_error(RuntimeError("torch tensor shape mismatch")) is False


# ── torch_dll_error_message ──────────────────────────────────────────────


def test_message_includes_engine_name() -> None:
    msg = torch_dll_error_message(OSError("WinError 127"), "WhisperX")
    assert "WhisperX" in msg


def test_message_suggests_reinstall_and_alternative() -> None:
    msg = torch_dll_error_message(OSError("WinError 127"), "WhisperX")
    assert "pip install torch" in msg
    assert "VieNeu" in msg  # gợi ý engine thay thế chạy CPU


# ── Blocker vòng đời (không rò rỉ sys.modules) ───────────────────────────


def test_blocker_lifecycle_clean() -> None:
    import sys

    from subtitles_extractor.infrastructure.torch_import_blocker import (
        install_torch_import_blocker,
        uninstall_torch_import_blocker,
    )

    # Chỉ chạy nếu torch chưa nạp thật (môi trường test không có torch).
    if sys.modules.get("torch", "absent") not in ("absent", None):
        return
    install_torch_import_blocker()
    assert is_torch_import_blocked() is True
    uninstall_torch_import_blocker()
    assert is_torch_import_blocked() is False
    # Sau gỡ: KHÔNG còn torch=None trong sys.modules (tránh AttributeError scipy/DSP).
    assert "torch" not in sys.modules
