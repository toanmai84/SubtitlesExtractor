"""Canh giữ: mọi tiến trình con phải chạy KHÔNG hiện cửa sổ — v3.23.349.

Người dùng báo: *"Ứng dụng chạy hay hiện cmd windows."*

Rà soát tự động tìm được **11 lời gọi** ``subprocess`` thiếu ``CREATE_NO_WINDOW``. Trên
Windows, mỗi lời gọi thiếu cờ sẽ bật một cửa sổ cmd đen nhấp nháy. Với TTS 55 câu (bản
cũ mở một tiến trình mỗi câu) là 55 lần nhấp nháy.

Cách rải rác mỗi tệp một kiểu (``_subprocess_flags``, ``_hidden_console_kwargs``, hoặc
không có gì) khiến rất dễ sót khi thêm lời gọi mới — nên gom về một module dùng chung.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from subtitles_extractor.infrastructure.process.hidden_process import no_window_kwargs

#: Cách hợp lệ để ẩn cửa sổ (giữ lại các tên cũ để không bắt nhầm).
_ACCEPTED = (
    "no_window_kwargs", "creationflags", "_subprocess_flags",
    "_hidden_console_kwargs", "_no_window_kwargs", "startupinfo",
)

_SPAWN_CALLS = (
    "subprocess.run", "subprocess.Popen", "subprocess.check_output",
    "subprocess.call", "asyncio.create_subprocess_exec",
)


def _source_root() -> Path:
    import subtitles_extractor.domain.entities.project_record as anchor

    return Path(anchor.__file__).resolve().parents[2]


def test_every_subprocess_call_hides_window() -> None:
    """BẤT BIẾN CHÍNH: không lời gọi nào được bỏ sót cờ ẩn cửa sổ."""
    offenders: list[str] = []
    for path in sorted(_source_root().rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if "subprocess" not in source:
            continue
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Call):
                continue
            if ast.unparse(node.func) not in _SPAWN_CALLS:
                continue
            if not any(marker in ast.unparse(node) for marker in _ACCEPTED):
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, f"Thiếu cờ ẩn cửa sổ tại: {offenders}"


def test_helper_is_empty_off_windows() -> None:
    """Nền tảng khác không có cờ này — trả rỗng thay vì lỗi."""
    result = no_window_kwargs()
    if sys.platform == "win32":
        assert "creationflags" in result
    else:
        assert result == {}


def test_helper_result_is_usable_as_kwargs() -> None:
    """Phải truyền được thẳng vào ``subprocess.run`` bằng ``**``."""
    completed = subprocess.run(
        [sys.executable, "-c", "print('ok')"],
        capture_output=True, text=True, **no_window_kwargs(),
    )
    assert completed.returncode == 0
    assert completed.stdout.strip() == "ok"


def test_module_exports_expected_helpers() -> None:
    from subtitles_extractor.infrastructure.process import hidden_process

    for name in ("no_window_kwargs", "run_hidden", "popen_hidden"):
        assert hasattr(hidden_process, name)


# ── Tiến trình TTS GPU thường trú ────────────────────────────────────────────
def _tts_source() -> str:
    import subtitles_extractor.infrastructure.tts.vieneu_tts_adapter as module

    return Path(module.__file__).read_text(encoding="utf-8")


def test_gpu_worker_is_persistent() -> None:
    """Mở tiến trình MỚI mỗi câu khiến GPU chậm hơn CPU (nạp model ~15s/câu)."""
    source = _tts_source()
    assert "def _ensure_gpu_server" in source
    assert '"--serve"' in source


def test_gpu_server_reused_while_alive() -> None:
    """Đang sống thì dùng lại, không mở thêm."""
    source = _tts_source()
    index = source.index("def _ensure_gpu_server")
    window = source[index : index + 500]
    assert "poll() is None" in window


def test_gpu_server_is_shut_down_cleanly() -> None:
    source = _tts_source()
    assert "def _shutdown_gpu_server" in source
    assert '"quit": true' in source


def test_gpu_server_stderr_is_devnull() -> None:
    """Không ai đọc stderr của tiến trình này -> phải DEVNULL, tránh bế tắc."""
    source = _tts_source()
    index = source.index("def _ensure_gpu_server")
    window = source[index : index + 1500]
    assert "stderr=subprocess.DEVNULL" in window


def test_worker_supports_serve_mode() -> None:
    import subtitles_extractor.domain.entities.project_record as anchor

    root = Path(anchor.__file__).resolve().parents[2]
    worker = root / "infrastructure" / "tts" / "vieneu_gpu_subprocess.py"
    source = worker.read_text(encoding="utf-8")
    assert "def serve()" in source
    assert '"ready"' in source


def test_serve_mode_rejects_missing_config() -> None:
    """Chạy thật: thiếu dòng cấu hình -> mã 2, không treo."""
    import subtitles_extractor.domain.entities.project_record as anchor

    root = Path(anchor.__file__).resolve().parents[2]
    worker = root / "infrastructure" / "tts" / "vieneu_gpu_subprocess.py"
    result = subprocess.run(
        [sys.executable, str(worker), "--serve"],
        input="\n", capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 2


def test_worker_requires_a_mode() -> None:
    import subtitles_extractor.domain.entities.project_record as anchor

    root = Path(anchor.__file__).resolve().parents[2]
    worker = root / "infrastructure" / "tts" / "vieneu_gpu_subprocess.py"
    result = subprocess.run(
        [sys.executable, str(worker)], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 2


# ── Trộn nhiều kênh xuống stereo ─────────────────────────────────────────────
def test_downmix_keeps_centre_channel_gain() -> None:
    """Kênh giữa (thứ 3 ở bố cục 5.1) chứa gần hết thoại — không được hạ nó."""
    from subtitles_extractor.infrastructure.video.video_render_command import (
        DOWNMIX_FILTER,
    )

    assert "FC" in DOWNMIX_FILTER
    assert "pan=stereo" in DOWNMIX_FILTER


def test_downmix_enabled_by_default() -> None:
    """Nguồn 5.1 giữ nguyên có thể KHÔNG NGHE THẤY GÌ trên máy người xem."""
    from subtitles_extractor.infrastructure.video.video_render_command import (
        RenderMode,
        RenderRequest,
    )

    request = RenderRequest(
        video_path=Path("a.mp4"), output_path=Path("o.mkv"), mode=RenderMode.SOFT_SUB
    )
    assert request.downmix_multichannel is True


def test_downmix_can_be_disabled(tmp_path: Path) -> None:
    """Vẫn cho phép sao chép nguyên luồng khi người dùng biết mình cần gì."""
    from subtitles_extractor.infrastructure.video.video_render_command import (
        RenderMode,
        RenderRequest,
        SubtitleMode,
        build_render_command,
    )

    video = tmp_path / "a.mp4"
    video.write_bytes(b"\x00")
    subtitle = tmp_path / "s.srt"
    subtitle.write_text("1\n", encoding="utf-8")

    command = build_render_command(
        "ffmpeg",
        RenderRequest(
            video_path=video, output_path=tmp_path / "o.mkv",
            mode=RenderMode.SOFT_SUB, subtitle_path=subtitle,
            subtitle_mode=SubtitleMode.SOFT, downmix_multichannel=False,
        ),
    )
    assert command[command.index("-c:a") + 1] == "copy"


def test_video_still_copied_when_downmixing(tmp_path: Path) -> None:
    """Trộn TIẾNG không được kéo theo mã hoá lại HÌNH — đó mới là phần đắt."""
    from subtitles_extractor.infrastructure.video.video_render_command import (
        RenderMode,
        RenderRequest,
        SubtitleMode,
        build_render_command,
    )

    video = tmp_path / "a.mp4"
    video.write_bytes(b"\x00")
    subtitle = tmp_path / "s.srt"
    subtitle.write_text("1\n", encoding="utf-8")

    command = build_render_command(
        "ffmpeg",
        RenderRequest(
            video_path=video, output_path=tmp_path / "o.mkv",
            mode=RenderMode.SOFT_SUB, subtitle_path=subtitle,
            subtitle_mode=SubtitleMode.SOFT,
        ),
    )
    assert command[command.index("-c:v") + 1] == "copy"
    assert command[command.index("-ac") + 1] == "2"
