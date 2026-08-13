"""Test cho :mod:`infrastructure.ocr.device_probe` (v3.23.295).

Kiểm chứng dò GPU NVIDIA THẬT: bản build GPU (``is_compiled_with_cuda()==True``)
chỉ nên chọn GPU khi ``device_count() > 0``. Trên máy không NVIDIA (device_count=0
hoặc ném lỗi) phải trả False -> tầng adapter lùi về CPU.

Dùng stub ``paddle`` giả để không cần cài paddle thật.
"""

from __future__ import annotations

from typing import Any

from subtitles_extractor.infrastructure.ocr.device_probe import (
    cuda_runtime_available,
    should_use_gpu,
)


class _FakeCuda:
    def __init__(self, count: Any, *, raises: Exception | None = None) -> None:
        self._count = count
        self._raises = raises

    def device_count(self) -> int:
        if self._raises is not None:
            raise self._raises
        return self._count


class _FakeDevice:
    def __init__(self, cuda: _FakeCuda) -> None:
        self.cuda = cuda


class _FakePaddle:
    """Stub tối thiểu tương thích PaddleLike."""

    def __init__(
        self,
        *,
        compiled: bool,
        device_count: Any = 0,
        count_raises: Exception | None = None,
        compiled_raises: Exception | None = None,
    ) -> None:
        self._compiled = compiled
        self._compiled_raises = compiled_raises
        self.device = _FakeDevice(_FakeCuda(device_count, raises=count_raises))

    def is_compiled_with_cuda(self) -> bool:
        if self._compiled_raises is not None:
            raise self._compiled_raises
        return self._compiled


def test_not_compiled_returns_false() -> None:
    """Bản build không CUDA -> False (dù có GPU)."""
    paddle = _FakePaddle(compiled=False, device_count=4)
    assert cuda_runtime_available(paddle) is False


def test_compiled_but_no_gpu_returns_false() -> None:
    """SỬA LỖI CHÍNH: build GPU nhưng máy không NVIDIA (device_count=0) -> False."""
    paddle = _FakePaddle(compiled=True, device_count=0)
    assert cuda_runtime_available(paddle) is False


def test_compiled_with_gpu_returns_true() -> None:
    """Build GPU + có GPU thật -> True."""
    paddle = _FakePaddle(compiled=True, device_count=2)
    assert cuda_runtime_available(paddle) is True


def test_device_count_raises_returns_false() -> None:
    """device_count() ném lỗi (thiếu driver) -> coi như không GPU."""
    paddle = _FakePaddle(
        compiled=True, count_raises=RuntimeError("no CUDA driver")
    )
    assert cuda_runtime_available(paddle) is False


def test_device_count_oserror_returns_false() -> None:
    """device_count() ném OSError (DLL lỗi) -> False."""
    paddle = _FakePaddle(compiled=True, count_raises=OSError("nvcuda.dll missing"))
    assert cuda_runtime_available(paddle) is False


def test_is_compiled_raises_returns_false() -> None:
    """is_compiled_with_cuda() ném lỗi -> False (an toàn)."""
    paddle = _FakePaddle(
        compiled=True, device_count=1, compiled_raises=RuntimeError("boom")
    )
    assert cuda_runtime_available(paddle) is False


def test_device_count_non_int_returns_false() -> None:
    """device_count() trả kiểu lạ -> 0 -> False (không crash)."""
    paddle = _FakePaddle(compiled=True, device_count=None)
    assert cuda_runtime_available(paddle) is False


def test_single_gpu_boundary() -> None:
    """Đúng 1 GPU -> True (biên)."""
    paddle = _FakePaddle(compiled=True, device_count=1)
    assert cuda_runtime_available(paddle) is True


class TestShouldUseGpu:
    """Quyết định cuối cùng: gộp force_cpu + want_gpu + probe."""

    def test_force_cpu_beats_everything(self) -> None:
        """force_cpu=True -> CPU dù có GPU và want_gpu."""
        paddle = _FakePaddle(compiled=True, device_count=4)
        assert (
            should_use_gpu(paddle, want_gpu=True, force_cpu=True) is False
        )

    def test_want_cpu_returns_false(self) -> None:
        """Cấu hình chọn CPU -> CPU dù có GPU."""
        paddle = _FakePaddle(compiled=True, device_count=4)
        assert (
            should_use_gpu(paddle, want_gpu=False, force_cpu=False) is False
        )

    def test_want_gpu_with_gpu_returns_true(self) -> None:
        """Muốn GPU + có GPU + không ép CPU -> GPU."""
        paddle = _FakePaddle(compiled=True, device_count=1)
        assert (
            should_use_gpu(paddle, want_gpu=True, force_cpu=False) is True
        )

    def test_want_gpu_no_gpu_returns_false(self) -> None:
        """Muốn GPU nhưng máy không NVIDIA -> CPU (một bản build, tự lùi)."""
        paddle = _FakePaddle(compiled=True, device_count=0)
        assert (
            should_use_gpu(paddle, want_gpu=True, force_cpu=False) is False
        )
