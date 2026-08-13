"""[v3.23.190] Test khôi phục môi trường — SỬA GỐC lỗi WhisperX thoát mã 1.

Nguyên nhân gốc (từ chuỗi hành vi log): v188 đặt ``os.environ['CUDA_VISIBLE_DEVICES']=''``
VĨNH VIỄN khi chạy VieNeu. Subprocess WhisperX sinh SAU đó copy ``dict(os.environ)`` ->
kế thừa biến này -> KHÔNG THẤY GPU -> WhisperX thoát mã 1 (dù lần chạy trước OK). Fix:
context manager ``temporary_env`` đặt env trong phạm vi ``with`` rồi KHÔI PHỤC nguyên
trạng -> không ô nhiễm subprocess.
"""

from __future__ import annotations

import os

import pytest

from subtitles_extractor.infrastructure.tts.vieneu_tts_adapter import (
    resolve_device_env,
    temporary_env,
)

_KEY = "CUDA_VISIBLE_DEVICES"


@pytest.fixture(autouse=True)
def _clean_env():
    """Đảm bảo biến test không tồn tại trước/sau mỗi test."""
    saved = os.environ.get(_KEY)
    os.environ.pop(_KEY, None)
    yield
    if saved is None:
        os.environ.pop(_KEY, None)
    else:
        os.environ[_KEY] = saved


def test_absent_var_removed_after_context() -> None:
    # Biến vốn KHÔNG tồn tại -> trong with có, ra khỏi with bị XÓA sạch (không rò rỉ).
    assert _KEY not in os.environ
    with temporary_env({_KEY: ""}):
        assert os.environ[_KEY] == ""
    assert _KEY not in os.environ


def test_existing_var_restored_after_context() -> None:
    # Biến vốn CÓ giá trị -> ra khỏi with trả lại ĐÚNG giá trị cũ.
    os.environ[_KEY] = "0"
    with temporary_env({_KEY: ""}):
        assert os.environ[_KEY] == ""
    assert os.environ[_KEY] == "0"


def test_env_restored_even_on_exception() -> None:
    # Ngay cả khi trong with ném lỗi, env vẫn phải khôi phục (finally).
    assert _KEY not in os.environ
    with pytest.raises(ValueError), temporary_env({_KEY: ""}):
        raise ValueError("lỗi giả lập")
    assert _KEY not in os.environ  # không rò rỉ dù có lỗi


def test_empty_overrides_noop() -> None:
    # force_cpu=False -> dict rỗng -> không đụng env.
    with temporary_env(resolve_device_env(False)):
        assert _KEY not in os.environ
    assert _KEY not in os.environ


def test_multiple_vars_restored() -> None:
    os.environ["VAR_A"] = "old_a"
    # VAR_B vốn không tồn tại.
    try:
        with temporary_env({"VAR_A": "new_a", "VAR_B": "new_b"}):
            assert os.environ["VAR_A"] == "new_a"
            assert os.environ["VAR_B"] == "new_b"
        assert os.environ["VAR_A"] == "old_a"  # trả lại cũ
        assert "VAR_B" not in os.environ       # xóa sạch
    finally:
        os.environ.pop("VAR_A", None)
        os.environ.pop("VAR_B", None)
