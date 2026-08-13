"""Canh giữ bế tắc tiến trình con WhisperX — v3.23.345.

LỖI THẬT: adapter mở ``stdout=subprocess.PIPE`` nhưng **chỉ đọc stderr**. WhisperX và
faster-whisper in rất nhiều ra stdout (nhận diện ngôn ngữ, thanh tiến độ tqdm, cảnh
báo). Khi bộ đệm ống đầy (~4–64 KB), tiến trình con KẸT ở lệnh ghi stdout còn tiến
trình cha KẸT ở vòng đọc stderr — **bế tắc vĩnh viễn**.

Đã tái hiện bằng tiến trình giả ghi ~1 MB ra stdout: treo hẳn, phải giết. Không timeout
nào cứu được vì ``for line in process.stderr`` chặn ngay ở lệnh đọc.

Giao thức chỉ dùng stderr (``PROGRESS``/``ERROR``/``WARN``) nên stdout hoàn toàn là
nhiễu — chuyển sang ``DEVNULL``.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


def _adapter_source() -> str:
    import subtitles_extractor.infrastructure.stt.whisperx_adapter as module

    return Path(module.__file__).read_text(encoding="utf-8")


# ── Bất biến chính ───────────────────────────────────────────────────────────
def test_no_unread_stdout_pipe() -> None:
    """Không được mở ống stdout mà không đọc — đó chính là nguyên nhân bế tắc."""
    assert "stdout=subprocess.PIPE" not in _adapter_source()


def test_both_branches_use_devnull() -> None:
    """CẢ HAI nhánh (phiên âm và căn chỉnh) đều phải sửa — dễ sót một."""
    assert _adapter_source().count("stdout=subprocess.DEVNULL") == 2


def test_stderr_still_piped() -> None:
    """Giao thức tiến độ và lỗi đi qua stderr — không được đóng nó."""
    assert _adapter_source().count("stderr=subprocess.PIPE") == 2


# ── Hạn giờ chờ tiến trình ───────────────────────────────────────────────────
def test_no_wait_without_timeout() -> None:
    """``wait()`` không hạn giờ treo ứng dụng nếu tiến trình con kẹt trong CUDA."""
    source = _adapter_source()
    # Bỏ dòng chú thích rồi mới đếm.
    code_lines = [
        line for line in source.splitlines() if not line.strip().startswith("#")
    ]
    assert not any(".wait()" in line for line in code_lines)


def test_timeout_path_kills_process() -> None:
    """Quá hạn thì phải giết, không được bỏ mặc tiến trình mồ côi."""
    source = _adapter_source()
    index = source.index("except subprocess.TimeoutExpired:")
    window = source[index : index + 300]
    assert "process.kill()" in window


# ── Chứng minh hành vi bằng tiến trình thật ──────────────────────────────────
def _noisy_child(temp: Path) -> Path:
    """Tiến trình con in nhiều ra stdout rồi mới ghi stderr — như WhisperX."""
    script = temp / "noisy.py"
    script.write_text(
        "import sys\n"
        "for i in range(20000):\n"
        "    print('Detected language... ' + 'x' * 40)\n"
        "print('ERROR xong', file=sys.stderr)\n",
        encoding="utf-8",
    )
    return script


def test_devnull_avoids_hang_and_keeps_stderr() -> None:
    """Cách sửa phải vừa không treo, vừa nhận đủ stderr."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        process = subprocess.Popen(
            [sys.executable, str(_noisy_child(temp))],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
        )
        assert process.stderr is not None
        lines = [line.strip() for line in process.stderr]
        assert process.wait(timeout=30) == 0
        assert lines == ["ERROR xong"]


@pytest.mark.parametrize("prefix", ["PROGRESS ", "ERROR ", "WARN "])
def test_protocol_prefixes_are_parsed(prefix: str) -> None:
    """Ba tiền tố của giao thức đều phải được xử lý ở phía adapter."""
    assert f'"{prefix}"' in _adapter_source() or f"'{prefix}'" in _adapter_source()


# ── Chú thích không được nói sai ─────────────────────────────────────────────
def test_align_device_choice_is_respected() -> None:
    """Giao diện có tuỳ chọn "GPU (nhanh hơn)" — code phải tôn trọng nó."""
    source = _adapter_source()
    assert 'if config.align_device == "cpu":' in source
    index = source.index('if config.align_device == "cpu":')
    window = source[index : index + 200]
    assert 'CUDA_VISIBLE_DEVICES' in window


def test_stale_comment_removed() -> None:
    """Chú thích cũ bảo "ép CPU thuần" vô điều kiện — trái với code."""
    assert "Ép tiến trình align dùng CPU thuần:" not in _adapter_source()


def test_adapter_is_valid_python() -> None:
    ast.parse(_adapter_source())
