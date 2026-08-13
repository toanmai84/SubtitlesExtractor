"""Test cho ``ocr.bundled_models`` — lớp tương thích ngược (cập nhật ở v3.23.301).

THAY ĐỔI HÀNH VI CÓ CHỦ ĐÍCH (v3.23.301):
    Bản v3.23.293 chỉ trỏ cache khi chạy ĐÓNG GÓI (frozen); chạy nguồn là no-op.
    Từ v3.23.301, kho model tập trung ``models/`` hoạt động CẢ khi chạy nguồn — nên
    dev cũng OCR offline được sau khi prefetch. Các test dưới phản ánh hành vi mới.

Logic thực tế nằm ở :mod:`infrastructure.model_store` (xem test_model_store_v323301).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from subtitles_extractor.infrastructure.ocr.bundled_models import (
    configure_bundled_paddle_models,
)

_ENV = "PADDLE_PDX_CACHE_HOME"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_ENV, raising=False)
    monkeypatch.delenv("SUBEXT_MODELS_DIR", raising=False)


def test_no_model_store_is_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Kho model rỗng (chưa prefetch) -> no-op, PaddleX tải như cũ."""
    monkeypatch.setenv("SUBEXT_MODELS_DIR", str(tmp_path))

    assert configure_bundled_paddle_models() is None
    assert _ENV not in os.environ


def test_paddle_dir_without_official_models_is_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Có models/paddle nhưng chưa prefetch -> vẫn no-op (an toàn)."""
    (tmp_path / "paddle").mkdir()
    monkeypatch.setenv("SUBEXT_MODELS_DIR", str(tmp_path))

    assert configure_bundled_paddle_models() is None
    assert _ENV not in os.environ


def test_prefetched_models_set_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Đã prefetch (có official_models/) -> set env trỏ đúng gốc cache."""
    (tmp_path / "paddle" / "official_models" / "PP-OCRv6_medium_rec").mkdir(parents=True)
    monkeypatch.setenv("SUBEXT_MODELS_DIR", str(tmp_path))

    result = configure_bundled_paddle_models()

    assert result == tmp_path / "paddle"
    assert os.environ[_ENV] == str(tmp_path / "paddle")
    assert (Path(os.environ[_ENV]) / "official_models").is_dir()


def test_respects_preexisting_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Env đã đặt sẵn -> tôn trọng, không ghi đè."""
    (tmp_path / "paddle" / "official_models").mkdir(parents=True)
    monkeypatch.setenv("SUBEXT_MODELS_DIR", str(tmp_path))
    monkeypatch.setenv(_ENV, "/custom/cache")

    assert configure_bundled_paddle_models() is None
    assert os.environ[_ENV] == "/custom/cache"


def test_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Gọi 2 lần: lần 2 tôn trọng env lần 1 đã set, giá trị giữ nguyên."""
    (tmp_path / "paddle" / "official_models").mkdir(parents=True)
    monkeypatch.setenv("SUBEXT_MODELS_DIR", str(tmp_path))

    first = configure_bundled_paddle_models()
    env_after_first = os.environ[_ENV]
    second = configure_bundled_paddle_models()

    assert first == tmp_path / "paddle"
    assert second is None
    assert os.environ[_ENV] == env_after_first
