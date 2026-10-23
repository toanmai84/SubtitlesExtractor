"""[v3.23.405] Test shim ScalingType — cho phép import vieneu khi torch thiếu symbol nightly."""

from __future__ import annotations

import enum
import sys
import types

import pytest


def _install_fake_torch(monkeypatch: pytest.MonkeyPatch, *, with_c_symbol: bool) -> types.ModuleType:
    """Dựng torch giả với ``torch.nn.functional`` KHÔNG có ScalingType."""
    torch = types.ModuleType("torch")
    torch_nn = types.ModuleType("torch.nn")
    functional = types.ModuleType("torch.nn.functional")
    torch_c = types.ModuleType("torch._C")
    if with_c_symbol:
        torch_c._ScalingType = enum.Enum("_ScalingType", {"TensorWise": 0})
        torch_c._SwizzleType = enum.Enum("_SwizzleType", {"No": 0})
    torch.nn = torch_nn
    torch_nn.functional = functional
    for name, mod in {
        "torch": torch,
        "torch.nn": torch_nn,
        "torch.nn.functional": functional,
        "torch._C": torch_c,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    return functional


def _load_shim():
    from subtitles_extractor.infrastructure.tts.vieneu_gpu_subprocess import (
        _shim_torch_scaling_type,
    )

    return _shim_torch_scaling_type


def test_shim_adds_placeholder_when_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    functional = _install_fake_torch(monkeypatch, with_c_symbol=False)
    _load_shim()()
    # Đủ CẢ 5 symbol nightly (2 enum + 3 hàm).
    for name in ("ScalingType", "SwizzleType", "grouped_mm", "scaled_mm", "scaled_grouped_mm"):
        assert hasattr(functional, name), name
    from torch.nn.functional import ScalingType, scaled_grouped_mm  # noqa: F401


def test_shim_enum_usable_in_union(monkeypatch: pytest.MonkeyPatch) -> None:
    """Enum placeholder phải là LỚP (dùng được trong kiểu ``X | list[X]``)."""
    _install_fake_torch(monkeypatch, with_c_symbol=False)
    _load_shim()()
    from torch.nn.functional import ScalingType  # type: ignore

    assert (ScalingType | list) is not None  # không ném lỗi


def test_shim_fallback_covers_new_fp8_symbol(monkeypatch: pytest.MonkeyPatch) -> None:
    """Symbol FP8 nightly MỚI (chưa liệt kê) vẫn qua nhờ fallback theo mẫu."""
    _install_fake_torch(monkeypatch, with_c_symbol=False)
    _load_shim()()
    from torch.nn.functional import some_scaled_mm_v3  # type: ignore  # noqa: F401


def test_shim_does_not_mask_unrelated_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Symbol KHÔNG phải FP8 vẫn ném AttributeError (giữ feature-detection đúng)."""
    functional = _install_fake_torch(monkeypatch, with_c_symbol=False)
    _load_shim()()
    with pytest.raises(AttributeError):
        _ = functional.totally_unrelated_symbol_xyz


def test_shim_uses_real_symbol_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    functional = _install_fake_torch(monkeypatch, with_c_symbol=True)
    _load_shim()()
    # Dùng bản THẬT từ torch._C, không đè bằng placeholder.
    assert functional.ScalingType.__name__ == "_ScalingType"


def test_shim_idempotent_and_preserves_existing(monkeypatch: pytest.MonkeyPatch) -> None:
    functional = _install_fake_torch(monkeypatch, with_c_symbol=False)
    sentinel = object()
    functional.ScalingType = sentinel  # đã có sẵn → không đè
    _load_shim()()
    assert functional.ScalingType is sentinel
