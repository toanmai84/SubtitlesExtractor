"""Test phân giải môi trường WhisperX — v3.23.333.

WhisperX phải cài vào môi trường RIÊNG, không chung với ứng dụng. Ba lý do (đã tra
metadata thật của whisperx 3.8.6):

1. **Xung đột phụ thuộc** — whisperx ghim ``huggingface-hub<1.0.0`` trong khi ứng dụng
   đang chạy ``1.24.0``. Cài chung sẽ HẠ CẤP gói này, có thể làm hỏng VieNeu-TTS và
   PaddleOCR (cả hai đều dùng huggingface-hub để tải mô hình).
2. **Xung đột DLL** — torch nạp CUDA riêng, đụng paddle. Đây vốn đã là lý do adapter
   chạy qua subprocess.
3. **Dung lượng** — gom torch vào bundle thêm ~3GB.

Trước đây adapter dùng thẳng ``sys.executable``, tức buộc cài chung → dính cả ba.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from subtitles_extractor.infrastructure.stt.whisperx_adapter import (
    WHISPERX_ENV_DIRNAME,
    WHISPERX_PYTHON_ENV_VAR,
    _MISSING_ENV_MESSAGE,
    resolve_whisperx_python,
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Đảm bảo mỗi test bắt đầu với biến môi trường sạch."""
    monkeypatch.delenv(WHISPERX_PYTHON_ENV_VAR, raising=False)


def test_env_var_takes_priority(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Biến môi trường được ưu tiên cao nhất — cho phép trỏ tới môi trường bất kỳ."""
    fake = tmp_path / "python.exe"
    fake.write_text("", encoding="utf-8")
    monkeypatch.setenv(WHISPERX_PYTHON_ENV_VAR, str(fake))
    assert resolve_whisperx_python() == str(fake)


def test_env_var_pointing_to_missing_file_is_ignored(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Biến trỏ tệp không tồn tại KHÔNG được dùng — nếu không sẽ lỗi khó hiểu lúc chạy."""
    monkeypatch.setenv(WHISPERX_PYTHON_ENV_VAR, str(tmp_path / "khong-co.exe"))
    result = resolve_whisperx_python()
    assert result != str(tmp_path / "khong-co.exe")


def test_empty_env_var_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(WHISPERX_PYTHON_ENV_VAR, "   ")
    # Không được coi chuỗi rỗng là đường dẫn hợp lệ.
    assert resolve_whisperx_python() != "   "


def test_returns_none_when_nothing_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Không có môi trường nào -> ``None`` để tầng trên hiện hướng dẫn cài."""
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert resolve_whisperx_python() is None


def test_frozen_build_never_uses_sys_executable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ở bản đóng gói, ``sys.executable`` là chính tệp .exe — chạy nó sẽ mở lại app."""
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert resolve_whisperx_python() != sys.executable


def test_env_dir_name_is_stable() -> None:
    """Tên thư mục phải khớp giữa adapter và build script."""
    assert WHISPERX_ENV_DIRNAME == "whisperx_env"


# ── Thông điệp hướng dẫn ─────────────────────────────────────────────────────
def test_message_explains_why_separate_env_is_needed() -> None:
    """Không chỉ báo 'chưa cài' — phải nói RÕ vì sao phải cài riêng."""
    assert "huggingface-hub" in _MISSING_ENV_MESSAGE
    assert "CUDA" in _MISSING_ENV_MESSAGE or "DLL" in _MISSING_ENV_MESSAGE


def test_message_contains_exact_install_commands() -> None:
    """Lệnh cài phải copy-paste chạy được ngay, kèm ĐÚNG index CUDA."""
    assert "python -m venv whisperx_env" in _MISSING_ENV_MESSAGE
    assert "download.pytorch.org/whl/cu129" in _MISSING_ENV_MESSAGE
    assert "torch torchaudio torchvision" in _MISSING_ENV_MESSAGE
    assert "pip install whisperx" in _MISSING_ENV_MESSAGE


def test_message_mentions_env_var_alternative() -> None:
    assert WHISPERX_PYTHON_ENV_VAR in _MISSING_ENV_MESSAGE


# ── Hợp đồng với build script và spec ────────────────────────────────────────
def _project_root() -> Path:
    import subtitles_extractor.domain.entities.project_record as anchor

    return Path(anchor.__file__).resolve().parents[4]


def test_build_script_creates_matching_env_dir() -> None:
    """Build script phải tạo đúng thư mục mà adapter đi tìm."""
    script = _project_root() / "build_windows.bat"
    if not script.is_file():
        pytest.skip("Không tìm thấy build_windows.bat")
    text = script.read_text(encoding="utf-8", errors="replace")
    assert f"venv {WHISPERX_ENV_DIRNAME}" in text


def test_build_script_uses_cuda_index_url() -> None:
    """torch phải cài từ index CUDA 12.9 — mặc định PyPI cho bản CPU."""
    script = _project_root() / "build_windows.bat"
    if not script.is_file():
        pytest.skip("Không tìm thấy build_windows.bat")
    text = script.read_text(encoding="utf-8", errors="replace")
    assert "download.pytorch.org/whl/cu129" in text
    assert "torch torchaudio torchvision" in text


def test_spec_always_excludes_torch_from_bundle() -> None:
    """torch/whisperx KHÔNG được gom vào bundle — chúng ở môi trường riêng."""
    spec = _project_root() / "SubtitlesExtractor.spec"
    if not spec.is_file():
        pytest.skip("Không tìm thấy SubtitlesExtractor.spec")
    text = spec.read_text(encoding="utf-8")
    assert '"whisperx", "torch", "torchvision", "torchaudio"' in text
    # Không còn khối gom torch vào bundle.
    assert '_add_all("torch")' not in text
