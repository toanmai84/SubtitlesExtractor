"""[v3.23.618] Không dùng ``preexec_fn`` khi khởi chạy tiến trình con.

``preexec_fn`` chạy mã Python trong tiến trình con ngay sau fork. Ứng dụng nhiều luồng (Qt,
loguru, worker) → nếu lúc fork luồng khác đang giữ khoá, tiến trình con kẹt vĩnh viễn và
lời gọi ``Popen`` treo. Đã gặp thật ở widget sóng âm: cả bộ test đứng ở
``audio_waveform_widget.py`` với 0% CPU. Dùng ``start_new_session=True`` thay thế.
"""
from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[3] / "src" / "subtitles_extractor"


def test_no_preexec_fn_anywhere_in_source() -> None:
    offenders = []
    for path in SRC.rglob("*.py"):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("#", 1)[0]
            if re.search(r"preexec_fn\s*[=\]'\"]", code):
                offenders.append(f"{path.relative_to(SRC)}:{number}")
    assert offenders == [], f"Dùng start_new_session=True thay preexec_fn: {offenders}"


def test_waveform_uses_start_new_session_on_posix() -> None:
    source = (SRC / "presentation" / "widgets" / "audio_waveform_widget.py").read_text(
        encoding="utf-8"
    )
    assert "no_window_kwargs_with_group['start_new_session'] = True" in source
