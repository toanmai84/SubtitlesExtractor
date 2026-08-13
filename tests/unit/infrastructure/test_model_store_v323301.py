"""Test cho :mod:`infrastructure.model_store` (kho model tập trung, v3.23.301).

Kiểm chứng phân giải gốc ``models`` (env -> bundle -> nguồn) và cấu hình biến môi
trường cache cho PaddleOCR / HuggingFace theo nguyên tắc an toàn:
    * Chỉ set khi model THỰC SỰ có sẵn.
    * Tôn trọng biến môi trường người dùng đã đặt.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from subtitles_extractor.infrastructure import model_store


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mỗi test bắt đầu với môi trường sạch."""
    for name in ("SUBEXT_MODELS_DIR", "PADDLE_PDX_CACHE_HOME", "HF_HOME"):
        monkeypatch.delenv(name, raising=False)


def test_env_override_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SUBEXT_MODELS_DIR", str(tmp_path))
    assert model_store.model_store_root() == tmp_path


def test_env_override_nonexistent_falls_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Env trỏ thư mục không tồn tại -> bỏ qua, không crash."""
    monkeypatch.setenv("SUBEXT_MODELS_DIR", "/khong/ton/tai/models")
    result = model_store.model_store_root()
    assert result is None or isinstance(result, Path)


def test_subdir_exists(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "paddle").mkdir()
    monkeypatch.setenv("SUBEXT_MODELS_DIR", str(tmp_path))
    assert model_store.model_store_subdir("paddle") == tmp_path / "paddle"


def test_subdir_missing_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SUBEXT_MODELS_DIR", str(tmp_path))
    assert model_store.model_store_subdir("khong-ton-tai") is None


def test_paddle_without_official_models_does_not_set_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AN TOÀN: models/paddle có nhưng chưa prefetch -> KHÔNG set env."""
    (tmp_path / "paddle").mkdir()
    monkeypatch.setenv("SUBEXT_MODELS_DIR", str(tmp_path))
    assert model_store.configure_paddle_model_store() is None
    import os

    assert "PADDLE_PDX_CACHE_HOME" not in os.environ


def test_paddle_with_official_models_sets_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "paddle" / "official_models" / "PP-OCRv6_medium_rec").mkdir(parents=True)
    monkeypatch.setenv("SUBEXT_MODELS_DIR", str(tmp_path))
    import os

    assert model_store.configure_paddle_model_store() == tmp_path / "paddle"
    assert os.environ["PADDLE_PDX_CACHE_HOME"] == str(tmp_path / "paddle")


def test_paddle_respects_preexisting_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "paddle" / "official_models").mkdir(parents=True)
    monkeypatch.setenv("SUBEXT_MODELS_DIR", str(tmp_path))
    monkeypatch.setenv("PADDLE_PDX_CACHE_HOME", "/custom/paddle")
    import os

    assert model_store.configure_paddle_model_store() is None
    assert os.environ["PADDLE_PDX_CACHE_HOME"] == "/custom/paddle"


def test_huggingface_sets_hf_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "huggingface").mkdir()
    monkeypatch.setenv("SUBEXT_MODELS_DIR", str(tmp_path))
    import os

    assert model_store.configure_huggingface_model_store() == tmp_path / "huggingface"
    assert os.environ["HF_HOME"] == str(tmp_path / "huggingface")


def test_huggingface_respects_preexisting_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "huggingface").mkdir()
    monkeypatch.setenv("SUBEXT_MODELS_DIR", str(tmp_path))
    monkeypatch.setenv("HF_HOME", "/custom/hf")
    import os

    assert model_store.configure_huggingface_model_store() is None
    assert os.environ["HF_HOME"] == "/custom/hf"


def test_configure_all_returns_applied_map(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "paddle" / "official_models").mkdir(parents=True)
    (tmp_path / "huggingface").mkdir()
    monkeypatch.setenv("SUBEXT_MODELS_DIR", str(tmp_path))

    applied = model_store.configure_all_model_stores()

    assert applied[model_store.PADDLE_SUBDIR] == tmp_path / "paddle"
    assert applied[model_store.HUGGINGFACE_SUBDIR] == tmp_path / "huggingface"


def test_configure_all_empty_when_nothing_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SUBEXT_MODELS_DIR", str(tmp_path))
    assert model_store.configure_all_model_stores() == {}


def test_backward_compat_delegate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hàm cũ ``configure_bundled_paddle_models`` vẫn hoạt động (uỷ quyền)."""
    from subtitles_extractor.infrastructure.ocr.bundled_models import (
        configure_bundled_paddle_models,
    )

    (tmp_path / "paddle" / "official_models").mkdir(parents=True)
    monkeypatch.setenv("SUBEXT_MODELS_DIR", str(tmp_path))
    assert configure_bundled_paddle_models() == tmp_path / "paddle"


def test_hf_fallback_creates_stable_dir_when_frozen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """[v3.23.393] Frozen + chưa có models/huggingface → tạo kho cố định cạnh exe."""
    exe_dir = tmp_path / "app"
    exe_dir.mkdir()
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(exe_dir / "App.exe"))
    monkeypatch.delenv("HF_HOME", raising=False)
    # Giả lập app đóng gói CHƯA có models/huggingface nào tồn tại.
    monkeypatch.setattr(model_store, "model_store_subdir", lambda name: None)

    result = model_store.configure_huggingface_model_store()
    assert result is not None
    assert result == exe_dir / "models" / "huggingface"
    assert result.is_dir()  # đã được tạo


def test_hf_returns_none_from_source_when_absent(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """Chạy từ nguồn (không frozen) + chưa có → None (để dev dùng cache mặc định)."""
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.setattr(model_store, "model_store_subdir", lambda name: None)
    assert model_store.configure_huggingface_model_store() is None


def test_paddle_cache_fallback_when_frozen_not_prefetched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """[v3.23.396] Frozen + chưa prefetch → tạo <exe_dir>/models/paddle + set cache env."""
    exe_dir = tmp_path / "app"
    exe_dir.mkdir()
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(exe_dir / "App.exe"))
    monkeypatch.delenv("PADDLE_PDX_CACHE_HOME", raising=False)
    monkeypatch.setattr(model_store, "model_store_subdir", lambda name: None)

    result = model_store.configure_paddle_model_store()
    assert result == exe_dir / "models" / "paddle"
    assert result.is_dir()
    import os

    assert os.environ["PADDLE_PDX_CACHE_HOME"] == str(exe_dir / "models" / "paddle")


def test_paddle_cache_none_from_source_when_not_prefetched(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """Chạy nguồn + chưa prefetch → None (PaddleX dùng cache mặc định của dev)."""
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    monkeypatch.delenv("PADDLE_PDX_CACHE_HOME", raising=False)
    monkeypatch.setattr(model_store, "model_store_subdir", lambda name: None)
    assert model_store.configure_paddle_model_store() is None
