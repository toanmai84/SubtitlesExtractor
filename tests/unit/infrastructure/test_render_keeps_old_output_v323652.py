"""[v3.23.652] Xuất video lỗi/huỷ không được xoá bản xuất cũ của người dùng; thành công thì thay thế nguyên khối."""

from __future__ import annotations

from pathlib import Path

import pytest

from subtitles_extractor.infrastructure.video import ffmpeg_video_renderer as renderer
from subtitles_extractor.infrastructure.video.video_render_command import RenderMode, RenderRequest, VideoRenderError


def _request(tmp_path: Path) -> RenderRequest:
    (tmp_path / "in.mp4").write_bytes(b"video")
    subtitle = tmp_path / "a.srt"
    subtitle.write_text("1\n00:00:00,000 --> 00:00:01,000\nxin chào\n", encoding="utf-8")
    return RenderRequest(tmp_path / "in.mp4", tmp_path / "out.mp4", RenderMode.HARD_SUB,
                         subtitle_path=subtitle, video_encoder="libx264")


def _fake_run(fail: bool):
    def run(command, request, *_args):
        assert "out.mp4" not in str(command[-1]) or ".partial" in str(command[-1])
        request.output_path.write_bytes(b"half")  # ffmpeg ghi dở
        if fail:
            request.output_path.unlink()  # _run_render thật xoá tệp nó đang ghi
            raise VideoRenderError("ffmpeg lỗi (mã 1): hết đĩa")
        request.output_path.write_bytes(b"new")
        return request.output_path

    return run


def test_failure_keeps_previous_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(renderer, "find_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(renderer, "_run_render", _fake_run(fail=True))
    request = _request(tmp_path)
    request.output_path.write_bytes(b"old export")
    with pytest.raises(VideoRenderError, match="hết đĩa"):
        renderer.render_video(request)
    assert request.output_path.read_bytes() == b"old export"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.srt", "in.mp4", "out.mp4"]


def test_success_replaces_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(renderer, "find_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(renderer, "_run_render", _fake_run(fail=False))
    request = _request(tmp_path)
    request.output_path.write_bytes(b"old export")
    assert renderer.render_video(request) == request.output_path
    assert request.output_path.read_bytes() == b"new"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.srt", "in.mp4", "out.mp4"]


def test_output_equal_to_input_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from dataclasses import replace

    monkeypatch.setattr(renderer, "find_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(renderer, "_run_render", _fake_run(fail=False))
    request = _request(tmp_path)
    request = replace(request, output_path=request.video_path)
    with pytest.raises(VideoRenderError, match="trùng"):
        renderer.render_video(request)
    assert request.video_path.read_bytes() == b"video"


def test_locked_target_keeps_finished_render(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(renderer, "find_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(renderer, "_run_render", _fake_run(fail=False))
    monkeypatch.setattr(renderer.time, "sleep", lambda _s: None)

    def locked(_src, _dst):
        raise PermissionError(32, "The process cannot access the file")

    monkeypatch.setattr(renderer.os, "replace", locked)
    request = _request(tmp_path)
    request.output_path.write_bytes(b"old export")
    with pytest.raises(VideoRenderError, match="partial"):
        renderer.render_video(request)
    assert (tmp_path / "out.partial.mp4").read_bytes() == b"new"   # bản vừa xuất không bị xoá
    assert request.output_path.read_bytes() == b"old export"
