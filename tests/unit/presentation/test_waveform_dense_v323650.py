"""[v3.23.650] Waveform vẽ gộp theo cột điểm ảnh khi có quá nhiều dòng trong khung nhìn (3649 dòng / 2,5 giờ: 512 ms → ~75 ms)."""

from __future__ import annotations

import os
import time

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent  # noqa: E402
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval  # noqa: E402
from subtitles_extractor.presentation.widgets import audio_waveform_widget as aww  # noqa: E402

_app = QApplication.instance() or QApplication([])


def _events(count: int, step: float = 1.0, length: float = 0.5, text: str = "chữ") -> list[SubtitleEvent]:
    return [
        SubtitleEvent(index=i + 1, text=text, interval=TimeInterval(i * step, i * step + length),
                      confidence=0.9, frame_count=3)
        for i in range(count)
    ]


def _widget(events: list[SubtitleEvent], view_sec: float) -> aww.AudioWaveformWidget:
    widget = aww.AudioWaveformWidget()
    widget.resize(1000, 128)
    widget.show()
    _app.processEvents()
    widget.set_events(events)
    widget._view_timeline_start_sec = 0.0
    widget._view_timeline_duration_sec = view_sec
    return widget


def test_dense_path_is_used_and_colours_event_columns_v323650(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    original = aww.AudioWaveformWidget._paint_dense_blocks

    def spy(self, *args, **kwargs):  # noqa: ANN001
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(aww.AudioWaveformWidget, "_paint_dense_blocks", spy)
    widget = _widget(_events(300), view_sec=300.0)          # 300 dòng trong 1000 px
    image = widget.grab().toImage()
    assert calls, "phải dùng nhánh vẽ gộp khi > ngưỡng dòng trong khung nhìn"
    mid = image.height() // 2
    # Dòng 0 chiếm 0–0,5 s → cột 0..1; khoảng trống 0,5–1,0 s → cột 2..3 (3,33 px/giây).
    assert image.pixelColor(1, mid) != image.pixelColor(2, mid) or image.pixelColor(1, mid) != image.pixelColor(3, mid)


def test_dense_arrays_rebuilt_after_set_events_v323650() -> None:
    widget = _widget(_events(300), view_sec=300.0)
    first = widget._dense_event_arrays()
    assert len(first[0]) == 300
    widget.set_events(_events(120))
    second = widget._dense_event_arrays()
    assert len(second[0]) == 120


def test_dense_severity_levels_v323650() -> None:
    fast = _events(100, text="x" * 40)                      # 40 ký tự / 0,5 s = 80 cps → mức 2
    widget = _widget(fast, view_sec=100.0)
    _, _, severity = widget._dense_event_arrays()
    assert set(severity.tolist()) == {2}


def test_dense_paint_far_faster_than_per_event_v323650() -> None:
    events = _events(3000, step=3.0, length=2.0)
    widget = _widget(events, view_sec=9000.0)
    widget.repaint()                                         # khởi tạo cache
    start = time.perf_counter()
    for _ in range(3):
        widget.repaint()
    per_paint = (time.perf_counter() - start) / 3
    assert per_paint < 0.25, f"vẽ gộp phải nhanh (đo {per_paint * 1000:.0f} ms/khung)"


def _run_isolated(body: str):  # noqa: ANN202
    import subprocess
    import sys
    import textwrap
    from pathlib import Path

    script = "import os\nos.environ.setdefault('QT_QPA_PLATFORM','offscreen')\n" + textwrap.dedent(body)
    src = str(Path(__file__).resolve().parents[3] / "src")
    env = dict(os.environ, PYTHONPATH=src, QT_QPA_PLATFORM="offscreen")
    return subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60, env=env, check=False)


def test_parentless_waveform_garbage_collected_while_threads_run_does_not_abort_v323650() -> None:
    """Widget sóng âm không parent bị GC khi luồng render còn chạy: trước đây abort (exit 134)."""
    result = _run_isolated(
        """
        import gc, time
        from PySide6.QtWidgets import QApplication
        app = QApplication([])
        from subtitles_extractor.presentation.widgets.audio_waveform_widget import AudioWaveformWidget
        w = AudioWaveformWidget()
        w.resize(400, 100); w.show(); app.processEvents(); time.sleep(0.2); app.processEvents()
        del w
        gc.collect()
        print("OK")
        """
    )
    assert result.returncode == 0, f"exit={result.returncode}\n{result.stderr[-1500:]}"
    assert "OK" in result.stdout


def test_preview_seek_service_garbage_collected_while_running_does_not_abort_v323650() -> None:
    result = _run_isolated(
        """
        import gc, time
        from pathlib import Path
        from PySide6.QtWidgets import QApplication
        app = QApplication([])
        from subtitles_extractor.presentation.widgets.preview_seek_service import PreviewSeekService
        s = PreviewSeekService(Path("/nonexistent.mp4"))
        s.start()
        t0 = time.time()
        while time.time() - t0 < 0.3:
            app.processEvents()
        del s
        gc.collect()
        print("OK")
        """
    )
    assert result.returncode == 0, f"exit={result.returncode}\n{result.stderr[-1500:]}"
    assert "OK" in result.stdout


def test_corrupt_waveform_cache_is_discarded_instead_of_killing_thread_v323650(tmp_path, monkeypatch) -> None:
    """Tệp cache .npy bị cắt cụt (app sập giữa lúc ghi): luồng phải bỏ cache, không ném ngoại lệ ra khỏi run()."""
    import hashlib
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    video = str(tmp_path / "phim.mp4")
    digest = hashlib.md5(os.path.abspath(video).encode("utf-8")).hexdigest()[:12]
    cache = tmp_path / f"phim.mp4_{digest}.waveform.npy"
    cache.write_bytes(b"\x93NUMPY\x01\x00\x76\x00{'descr': '<f4', 'fortran_order': False, 'shape': (1000,), }")  # cụt
    monkeypatch.setattr("subtitles_extractor.infrastructure.media.find_ffmpeg", lambda *a, **k: None)

    thread = aww._AudioExtractThread(video)
    thread.run()                       # trước đây: ValueError/EOFError thoát khỏi run() → luồng chết, sóng âm "đang phân tích" mãi
    assert not cache.exists()
