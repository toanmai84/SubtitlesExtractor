"""[v3.23.656] B4: huỷ xuất bản phải dừng ffmpeg kể cả khi nó im lặng (không in dòng nào); ffmpeg treo thì tự dừng.

Trước đây lệnh huỷ chỉ được kiểm khi ffmpeg in một dòng stderr (``for line in stderr``) → ffmpeg treo im lặng thì nút
Huỷ không có tác dụng, luồng xuất kẹt mãi.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest


from subtitles_extractor.infrastructure.video import ffmpeg_video_renderer as module
from subtitles_extractor.infrastructure.video.video_render_command import RenderMode, RenderRequest, VideoRenderError

SILENT = [sys.executable, "-c", "import time; time.sleep(30)"]


def _request(tmp_path: Path) -> RenderRequest:
    return RenderRequest(video_path=tmp_path / "in.mp4", output_path=tmp_path / "out.partial.mp4", mode=RenderMode.SOFT_SUB)


def test_cancel_works_while_ffmpeg_is_silent(tmp_path) -> None:
    started = time.monotonic()
    with pytest.raises(VideoRenderError, match="huỷ"):
        module._run_render(SILENT, _request(tmp_path), 10.0, None, lambda: time.monotonic() - started > 0.5)
    assert time.monotonic() - started < 5.0


def test_silent_hang_is_stopped_by_stall_timeout(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(module, "_STALL_TIMEOUT_SEC", 1.0)
    started = time.monotonic()
    with pytest.raises(VideoRenderError, match="không phản hồi"):
        module._run_render(SILENT, _request(tmp_path), 10.0, None, None)
    assert time.monotonic() - started < 6.0


def test_mp4_output_gets_faststart(tmp_path) -> None:
    from subtitles_extractor.infrastructure.video.video_render_command import build_render_command

    src = tmp_path / "in.mp4"
    src.write_bytes(b"x")
    sub = tmp_path / "a.srt"
    sub.write_text("1\n00:00:00,000 --> 00:00:01,000\nx\n", encoding="utf-8")
    for out, expected in (("o.mp4", True), ("o.partial.mp4", True), ("o.mkv", False)):
        command = build_render_command("ffmpeg", RenderRequest(video_path=src, output_path=tmp_path / out,
                                                                mode=RenderMode.SOFT_SUB, subtitle_path=sub))
        assert ("+faststart" in command) is expected, out
        if expected:
            assert command.index("+faststart") < len(command) - 1    # trước tên tệp đích
