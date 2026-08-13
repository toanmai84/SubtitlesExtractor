"""Test chẩn đoán WhisperX hiển thị trên giao diện — v3.23.335.

LỖI ĐƯỢC SỬA: v3.23.333 đổi WhisperX sang môi trường riêng và cập nhật thông điệp
trong *adapter*, nhưng BỎ SÓT ``dependency_doctor`` — nơi giao diện thực sự lấy chữ để
hiển thị. Kết quả: người dùng vẫn thấy hướng dẫn cũ ``pip install whisperx``, tức là
được khuyên làm đúng cái sẽ **làm hỏng ứng dụng**.

Bộ test này canh giữ ba điều:
    * Hai nguồn thông điệp phải NHẤT QUÁN.
    * Tuyệt đối KHÔNG gợi ý cài vào môi trường chính.
    * Tạo môi trường riêng xong phải được nhận là sẵn sàng ngay.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from subtitles_extractor.infrastructure.diagnostics.dependency_doctor import (
    DependencyStatus,
    check_whisperx,
    install_package,
)
from subtitles_extractor.infrastructure.stt.whisperx_adapter import (
    WHISPERX_ENV_DIRNAME,
    WHISPERX_PYTHON_ENV_VAR,
    _MISSING_ENV_MESSAGE,
)


@pytest.fixture
def no_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(WHISPERX_PYTHON_ENV_VAR, raising=False)


@pytest.fixture
def fake_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Giả lập đã tạo môi trường riêng."""
    python_exe = tmp_path / "python.exe"
    python_exe.write_text("", encoding="utf-8")
    monkeypatch.setenv(WHISPERX_PYTHON_ENV_VAR, str(python_exe))
    return python_exe


# ── Trạng thái ───────────────────────────────────────────────────────────────
def test_reports_missing_when_no_environment(no_env: None) -> None:
    assert check_whisperx().status is DependencyStatus.MISSING_PACKAGE


def test_reports_ok_once_environment_exists(fake_env: Path) -> None:
    """Tạo môi trường riêng xong PHẢI được nhận ngay — lỗi cũ vẫn báo chưa cài."""
    report = check_whisperx()
    assert report.status is DependencyStatus.OK
    assert str(fake_env) in report.detail


# ── KHÔNG được gợi ý cài vào môi trường chính ────────────────────────────────
def test_never_offers_auto_install(no_env: None) -> None:
    """``pip_args`` rỗng = không có nút tự cài.

    :func:`install_package` chạy pip bằng ``sys.executable``, tức môi trường CHÍNH —
    bấm nút đó sẽ hạ cấp huggingface-hub và làm hỏng VieNeu-TTS/PaddleOCR.
    """
    assert check_whisperx().pip_args == ()


def test_hint_does_not_suggest_bare_pip_install(no_env: None) -> None:
    """Hướng dẫn KHÔNG được chứa lệnh cài thẳng vào môi trường chính."""
    hint = check_whisperx().install_hint
    # Mọi lệnh pip trong hướng dẫn đều phải đi qua môi trường riêng.
    for line in hint.splitlines():
        if "pip install" in line and "torch" not in line and "whisperx" in line:
            assert WHISPERX_ENV_DIRNAME in line


@pytest.mark.parametrize("package", ["whisperx", "torch", "torchaudio", "torchvision"])
def test_install_package_refuses_dangerous_packages(package: str) -> None:
    """Chặn cứng: dù ai đó gọi thẳng cũng không được cài vào môi trường chính."""
    ok, message = install_package((package,))
    assert ok is False
    assert package in message.lower()


def test_install_package_message_explains_alternative() -> None:
    _ok, message = install_package(("whisperx",))
    assert WHISPERX_ENV_DIRNAME in message


# ── Hai nguồn thông điệp phải nhất quán ──────────────────────────────────────
def test_both_messages_mention_separate_environment(no_env: None) -> None:
    """Adapter và dependency_doctor phải nói CÙNG một điều — lỗi cũ là lệch nhau."""
    doctor = check_whisperx()
    combined = f"{doctor.detail} {doctor.install_hint}"
    assert WHISPERX_ENV_DIRNAME in combined
    assert WHISPERX_ENV_DIRNAME in _MISSING_ENV_MESSAGE


def test_both_messages_explain_the_reason(no_env: None) -> None:
    """Phải nói VÌ SAO cài riêng, không chỉ bảo làm."""
    doctor = check_whisperx()
    combined = f"{doctor.detail} {doctor.install_hint}"
    assert "huggingface-hub" in combined
    assert "huggingface-hub" in _MISSING_ENV_MESSAGE


def test_both_messages_use_cuda_index(no_env: None) -> None:
    """torch phải cài từ index CUDA — mặc định PyPI cho bản CPU."""
    assert "download.pytorch.org/whl/cu129" in check_whisperx().install_hint
    assert "download.pytorch.org/whl/cu129" in _MISSING_ENV_MESSAGE


def test_no_stale_message_remains_in_sources() -> None:
    """Chuỗi cũ 'pip install whisperx (cần torch + CUDA' KHÔNG được còn ở đâu.

    Đây chính là chuỗi người dùng nhìn thấy trên màn hình dù adapter đã sửa.
    """
    import subtitles_extractor.domain.entities.project_record as anchor

    source_root = Path(anchor.__file__).resolve().parents[2]
    offenders = [
        path.name
        for path in source_root.rglob("*.py")
        if "pip install whisperx (cần torch" in path.read_text(
            encoding="utf-8", errors="replace"
        )
    ]
    assert not offenders, f"Còn thông điệp cũ tại: {offenders}"
