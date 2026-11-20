"""[v3.23.751] PaddleOCR-VL ở tiến trình riêng: lần sinh chữ đầu tiên của VL kẹt ngẫu nhiên (2/5 lượt phim 大乾悍刀行, cả
khi máy giữ thức) và kéo NGỮ CẢNH CUDA của cả tiến trình kẹt theo — lưới an toàn 3.23.747 bỏ VL nhưng lệnh GPU kế tiếp ở
luồng chính treo (log 19/11 08:16:10). Tiến trình riêng: kẹt / sập → giết tiến trình con, tiến trình chính đi tiếp."""

from __future__ import annotations

import time

import numpy as np
import pytest

from subtitles_extractor.infrastructure.ocr import paddle_vl_process as proc
from subtitles_extractor.infrastructure.ocr import paddle_vl_reader as vl
from tests.unit.infrastructure import vl_fake_workers as fake

CROP = np.zeros((40, 200, 3), np.uint8)


def test_reads_through_child_process_and_release_stops_it() -> None:
    reader = proc.PaddleVlProcessReader(worker=fake.echo_worker, prepare=False)
    try:
        assert reader.read([CROP, np.zeros((30, 90, 3), np.uint8)], device="gpu") == ["200x40", "90x30"]
        process = reader._process
        assert process is not None and process.is_alive()
        assert reader.read([CROP], device="gpu") == ["200x40"]          # cùng tiến trình, mô hình giữ nguyên
        assert reader._process is process
    finally:
        reader.release()
    assert reader._process is None and not process.is_alive()          # trả VRAM: tiến trình con đã tắt


def test_hung_child_is_killed_and_vl_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vl, "VL_READ_TIMEOUT_GPU_SEC", 1.0)
    monkeypatch.setattr(vl, "VL_LOAD_TIMEOUT_SEC", 1.0)
    reader = proc.PaddleVlProcessReader(worker=fake.hang_worker, prepare=False)
    started = time.monotonic()
    with pytest.raises(vl.VlReadTimeoutError, match="đã dừng tiến trình"):
        reader.read([CROP], device="gpu")
    assert time.monotonic() - started < 30 and reader._process is None
    with pytest.raises(vl.VlReadTimeoutError):                          # lần sau báo ngay, không bật lại
        reader.read([CROP], device="gpu")


def test_crashed_child_reported_and_restartable() -> None:
    reader = proc.PaddleVlProcessReader(worker=fake.crash_worker, prepare=False)
    with pytest.raises(RuntimeError, match="kết thúc bất thường"):
        reader.read([CROP], device="gpu")
    assert reader._process is None and not reader._stuck


def test_child_error_terminates_child() -> None:
    reader = proc.PaddleVlProcessReader(worker=fake.error_worker, prepare=False)
    with pytest.raises(RuntimeError, match="CUDA error 700"):
        reader.read([CROP], device="gpu")
    assert reader._process is None


def test_adapter_uses_process_reader_on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SE_VL_MODE", raising=False)
    monkeypatch.setattr(proc.sys, "platform", "win32")
    assert isinstance(proc.make_vl_reader(), proc.PaddleVlProcessReader)
    monkeypatch.setattr(proc.sys, "platform", "linux")
    assert isinstance(proc.make_vl_reader(), vl.PaddleVlLineReader)
    monkeypatch.setenv("SE_VL_MODE", "process")
    assert isinstance(proc.make_vl_reader(), proc.PaddleVlProcessReader)


# ── [v3.23.752] log 19/11 09:14: tiến trình con của exe thiếu shim chardet + stderr None → hộp thoại lỗi ──────────────
def test_child_setup_matches_exe_main(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SE_VL_SKIP", "1")
    reader = proc.PaddleVlProcessReader(worker=fake.exe_like_setup_worker, prepare=False)
    try:
        stream, encoding = reader.read([CROP], device="gpu")
    finally:
        reader.release()
    assert stream == "_NullStream" and encoding.lower().replace("-", "") == "utf8"


def test_startup_failure_reported_not_raised_in_child() -> None:
    reader = proc.PaddleVlProcessReader(worker=fake.failing_start_worker, prepare=False)
    with pytest.raises(RuntimeError, match="chardet"):
        reader.read([CROP], device="gpu")
    assert reader._process is None


def test_real_worker_never_lets_exceptions_escape() -> None:
    import inspect

    source = inspect.getsource(proc._vl_worker_main)
    assert "prepare_child_process()" in source and "except BaseException" in source
