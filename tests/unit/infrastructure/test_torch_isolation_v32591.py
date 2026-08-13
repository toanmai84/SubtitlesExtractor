"""[v3.23.191] Test cô lập torch suốt inference VieNeu (giống cô lập PaddleOCR).

Yêu cầu người dùng: cô lập VieNeu giống PaddleOCR để chạy được (thuần ONNX, né torch DLL
hỏng) mà VẪN GIỮ torch trong venv cho WhisperX align + F5. Khác PaddleOCR (chặn torch chỉ
lúc import), VieNeu import torch TRỄ lúc inference -> phải giữ chặn SUỐT quá trình chạy.
``torch_isolation`` context manager làm điều đó và khôi phục sys.modules sạch sau khi xong.
"""

from __future__ import annotations

import sys

from subtitles_extractor.infrastructure.torch_import_blocker import (
    install_torch_import_blocker,
    is_torch_import_blocked,
    torch_isolation,
    uninstall_torch_import_blocker,
)


def _torch_absent() -> bool:
    """True nếu torch chưa nạp thật (môi trường test không có torch)."""
    return sys.modules.get("torch", "absent") in ("absent", None)


def test_blocks_inside_restores_outside() -> None:
    if not _torch_absent():
        return
    assert not is_torch_import_blocked()
    with torch_isolation():
        assert is_torch_import_blocked()
        assert sys.modules.get("torch") is None
    assert not is_torch_import_blocked()
    assert "torch" not in sys.modules  # không rò rỉ torch=None


def test_restores_even_on_exception() -> None:
    if not _torch_absent():
        return
    import pytest

    with pytest.raises(ValueError), torch_isolation():
        assert is_torch_import_blocked()
        raise ValueError("lỗi giả lập")
    assert not is_torch_import_blocked()
    assert "torch" not in sys.modules


def test_idempotent_respects_outer_blocker() -> None:
    if not _torch_absent():
        return
    # Nếu blocker đã cài bên ngoài -> torch_isolation KHÔNG gỡ (bên ngoài chịu trách nhiệm).
    install_torch_import_blocker()
    try:
        with torch_isolation():
            assert is_torch_import_blocked()
        assert is_torch_import_blocked()  # vẫn giữ blocker ngoài
    finally:
        uninstall_torch_import_blocker()
    assert not is_torch_import_blocked()


def test_nested_isolation_safe() -> None:
    if not _torch_absent():
        return
    with torch_isolation():
        assert is_torch_import_blocked()
        with torch_isolation():  # lồng nhau -> inner tôn trọng outer
            assert is_torch_import_blocked()
        assert is_torch_import_blocked()  # outer vẫn giữ sau khi inner thoát
    assert not is_torch_import_blocked()
    assert "torch" not in sys.modules
