"""[v3.23.257] resample_audio kháng ``sys.modules["torch"]=None`` của torch blocker.

**Lỗi runtime (Toan gặp):** ``'NoneType' object has no attribute 'Tensor'`` trong
``resample_audio`` -> ``resample_poly``, xảy ra TRONG khối ``torch_isolation``.

**Điều tra (cài scipy 1.17 thật + tái hiện):** scipy >=1.15 dùng array-API compat.
``is_torch_array`` gọi ``getattr(sys.modules["torch"], "Tensor")``. Blocker đặt
``torch=None`` -> ``getattr(None, "Tensor")`` -> ``AttributeError``.

**Điểm mấu chốt (điều tra sâu):** lỗi xảy ra ngay lúc **IMPORT** ``scipy.signal``
(scipy khởi tạo stats dựng docstring ví dụ -> chạm is_torch_array), KHÔNG chỉ lúc gọi
``resample_poly``. Nên phải bọc CẢ import lẫn gọi.

**Sửa:** ``_torch_none_bypassed`` tạm xoá entry ``torch=None`` (chỉ entry None, không đụng
torch thật) quanh cả import scipy lẫn ``resample_poly``, rồi khôi phục -> blocker vẫn chặn
import torch thật cho phần còn lại của pipeline.
"""

from __future__ import annotations

import sys

import numpy as np

from subtitles_extractor.infrastructure.tts.audio_utils import (
    _torch_none_bypassed,
    resample_audio,
)


def test_resample_với_torch_none() -> None:
    # Mô phỏng blocker: torch=None. resample 48k->24k phải chạy, không AttributeError.
    saved = sys.modules.get("torch", "absent")
    sys.modules["torch"] = None  # type: ignore[assignment]
    try:
        out = resample_audio(np.ones(48_000, dtype=np.float32), 48_000, 24_000)
        assert out.size == 24_000
    finally:
        if saved == "absent":
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = saved


def test_torch_none_khôi_phục_sau_resample() -> None:
    # Sau khi resample, torch=None phải còn nguyên (blocker không bị phá).
    saved = sys.modules.get("torch", "absent")
    sys.modules["torch"] = None  # type: ignore[assignment]
    try:
        resample_audio(np.ones(48_000, dtype=np.float32), 48_000, 24_000)
        assert sys.modules.get("torch", "absent") is None
    finally:
        if saved == "absent":
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = saved


def test_bypass_context_khôi_phục_đúng() -> None:
    saved = sys.modules.get("torch", "absent")
    sys.modules["torch"] = None  # type: ignore[assignment]
    try:
        with _torch_none_bypassed():
            # Trong khối, torch=None tạm biến mất (scipy thấy "vắng mặt").
            assert sys.modules.get("torch", "absent") == "absent"
        # Ra khối, torch=None khôi phục.
        assert sys.modules.get("torch", "absent") is None
    finally:
        if saved == "absent":
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = saved


def test_bypass_không_đụng_torch_thật() -> None:
    # Nếu torch là module THẬT (không phải None), bypass KHÔNG được xoá.
    saved = sys.modules.get("torch", "absent")
    sentinel = object()
    sys.modules["torch"] = sentinel  # type: ignore[assignment]
    try:
        with _torch_none_bypassed():
            # torch thật vẫn còn (chỉ xoá entry None).
            assert sys.modules.get("torch") is sentinel
    finally:
        if saved == "absent":
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = saved


def test_resample_bằng_nhau_không_đổi() -> None:
    # orig == target -> trả nguyên bản (không chạm scipy).
    audio = np.ones(1000, dtype=np.float32)
    out = resample_audio(audio, 24_000, 24_000)
    assert out.size == 1000
