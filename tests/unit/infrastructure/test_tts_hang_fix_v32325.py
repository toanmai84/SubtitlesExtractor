"""Test [v3.23.25] EdgeTTS + audio_mastering có timeout chống treo vô hạn."""

from __future__ import annotations

import inspect

from subtitles_extractor.infrastructure.tts import audio_mastering
from subtitles_extractor.infrastructure.tts.edge_tts_adapter import EdgeTTSAdapter


def test_async_generate_wraps_save_in_timeout() -> None:
    # [v3.23.268] edge-tts cách ly GPL vào subprocess. Timeout chống treo nay nằm ở:
    # (1) worker _synthesize (wait_for 30s quanh communicate.save), (2)
    # _run_edge_subprocess (wait_for 35s quanh proc.communicate). Kiểm cả hai.
    from subtitles_extractor.infrastructure.tts import edge_tts_subprocess

    worker_src = inspect.getsource(edge_tts_subprocess)
    assert "wait_for" in worker_src
    assert "timeout=30.0" in worker_src
    assert "TimeoutError" in worker_src

    run_src = inspect.getsource(EdgeTTSAdapter._run_edge_subprocess)
    assert "wait_for" in run_src
    assert "timeout=35.0" in run_src


def test_edge_tts_cach_ly_gpl_subprocess() -> None:
    # [v3.23.268] Tiến trình CHÍNH không được import edge_tts (cách ly GPL). Chỉ worker
    # subprocess mới import. Kiểm adapter không có 'import edge_tts' trong code thực thi.
    src = inspect.getsource(EdgeTTSAdapter)
    # Chỉ được nhắc trong comment, KHÔNG có lệnh import thực (dòng bắt đầu bằng 'import').
    for line in src.splitlines():
        stripped = line.strip()
        assert not stripped.startswith("import edge_tts"), (
            "Tiến trình chính KHÔNG được import edge_tts"
        )


def test_audio_mastering_encode_has_timeout() -> None:
    # ffmpeg encode output phải có timeout + bắt TimeoutExpired.
    src = inspect.getsource(audio_mastering)
    assert "timeout=600" in src
    assert "TimeoutExpired" in src
