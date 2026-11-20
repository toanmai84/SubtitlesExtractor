"""[v3.23.747] Nguyên nhân gốc treo PaddleOCR-VL: Windows tắt màn hình vì không dùng → GPU hạ trạng thái nguồn → lệnh CUDA
đang chờ không bao giờ xong (Event Viewer máy chủ dự án 18/11: Kernel-Power 566 lúc 23:04:17, VL treo 23:04:31,
nvlddmkm 153 lúc 23:12:26). Trong lúc chạy việc dài, ứng dụng giữ máy thức + màn hình sáng (yêu cầu nguồn Windows)."""

from __future__ import annotations

import re
import sys
import threading
from pathlib import Path

import pytest

from subtitles_extractor.infrastructure.process import keep_awake as ka


@pytest.fixture
def fake_windows(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(ka, "_win_acquire", lambda reason: calls.append(f"acquire:{reason}") or 1234)
    monkeypatch.setattr(ka, "_win_release", lambda handle: calls.append(f"release:{handle}"))
    return calls


def test_request_held_during_job_and_released_after(fake_windows: list[str]) -> None:
    with ka.keep_awake("Trích xuất phụ đề"):
        assert fake_windows == ["acquire:Trích xuất phụ đề"]
    assert fake_windows == ["acquire:Trích xuất phụ đề", "release:1234"]


def test_overlapping_jobs_share_one_request(fake_windows: list[str]) -> None:
    started, finish = threading.Event(), threading.Event()

    def other_job() -> None:
        with ka.keep_awake("Lồng tiếng"):
            started.set()
            finish.wait(5)

    worker = threading.Thread(target=other_job)
    with ka.keep_awake("Trích xuất phụ đề"):
        worker.start()
        started.wait(5)
    assert fake_windows == ["acquire:Trích xuất phụ đề"]          # việc kia còn chạy → chưa trả
    finish.set()
    worker.join(5)
    assert fake_windows[-1] == "release:1234" and len(fake_windows) == 2


def test_released_even_when_job_fails(fake_windows: list[str]) -> None:
    @ka.keeps_awake("Xuất video")
    def run() -> None:
        raise RuntimeError("lỗi giữa chừng")

    with pytest.raises(RuntimeError):
        run()
    assert fake_windows == ["acquire:Xuất video", "release:1234"]


def test_acquire_failure_does_not_break_job(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(ka, "_win_acquire", lambda reason: (_ for _ in ()).throw(OSError("API")))
    with ka.keep_awake():
        pass


def test_noop_elsewhere(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(ka, "_win_acquire", lambda reason: pytest.fail("không gọi API Windows"))
    with ka.keep_awake():
        pass


def test_long_running_workers_keep_machine_awake() -> None:
    root = Path(__file__).resolve().parents[3] / "src/subtitles_extractor/presentation/workers"
    for name in ("extract_subtitles_worker", "reocr_worker", "ocr_benchmark_worker", "transcribe_worker", "tts_worker",
                 "resynthesize_worker", "render_video_worker", "concat_video_worker", "translate_worker",
                 "embedded_extract_worker"):
        source = (root / f"{name}.py").read_text(encoding="utf-8")
        runs = re.findall(r"^ {4}def run\(self\)", source, flags=re.M)
        decorated = re.findall(r"^ {4}@keeps_awake\(.*\n {4}def run\(self\)", source, flags=re.M)
        assert runs and len(decorated) == len(runs), name
