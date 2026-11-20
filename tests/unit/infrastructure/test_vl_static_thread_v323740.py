"""[v3.23.740] PaddleOCR-VL vẫn ``Place(undefined:0)`` sau 3.23.739 (log 18/11 07:05).

Nguyên nhân thật: chế độ TĨNH PIR. ``_current_expected_place()`` trả ``core.Place()`` (undefined) khi luồng ở chế độ
tĩnh PIR. Chế độ động = tracer riêng từng luồng, chép từ biến toàn cục lúc luồng được tạo: một lần đặt tracer về
``None`` ở đâu đó (như ``paddle.enable_static()``) → mọi luồng tạo sau ở chế độ tĩnh. ``set_device`` (3.23.739) chỉ
đổi biến toàn cục → không đủ.

Chạy Paddle THẬT trong TIẾN TRÌNH CON sạch: trong bộ test đầy đủ, test khác gỡ mô-đun paddle khỏi ``sys.modules``
(giả lập chưa cài) → import lại paddle cùng tiến trình hỏng ("libpaddle is not a package").
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[3] / "src"

SCRIPT = textwrap.dedent(
    """
    import threading
    import paddle
    from paddle.base import framework
    from subtitles_extractor.infrastructure.ocr.paddle_vl_reader import _bind_thread_device

    def in_thread(fn):
        out = {}
        def run():
            try:
                out["value"] = fn()
            except Exception as exc:
                out["error"] = str(exc)
        t = threading.Thread(target=run); t.start(); t.join(120)
        return out

    def drop_tracer():
        # Tác dụng cuối của enable_static() / guard tĩnh ở luồng chính: tracer TOÀN CỤC = None (luồng mới chép theo).
        framework.global_var._dygraph_tracer_ = None

    in_thread(drop_tracer)
    broken = in_thread(paddle.amp.is_bfloat16_supported)
    print("BROKEN", "undefined" in broken.get("error", ""))

    def fixed():
        _bind_thread_device("cpu")
        paddle.amp.is_bfloat16_supported()
        return True

    print("FIXED", in_thread(fixed).get("value") is True)
    """
)


@pytest.mark.skipif(importlib.util.find_spec("paddle") is None, reason="chưa cài paddle")
def test_vl_thread_recovers_from_static_mode_left_by_another_thread() -> None:
    result = subprocess.run([sys.executable, "-c", SCRIPT], capture_output=True, text=True, timeout=300,
                            env=dict(os.environ, PYTHONPATH=str(SRC)), check=False)
    lines = {
        line.split()[0]: line.split()[1] for line in result.stdout.splitlines()
        if line.split()[:1] in (["BROKEN"], ["FIXED"])
    }
    assert lines == {"BROKEN": "True", "FIXED": "True"}, result.stderr[-1500:]
