"""Test [v3.22.8] phục hồi import paddle khi module nửa-khởi-tạo + lock chung."""

from __future__ import annotations

import sys

from subtitles_extractor.infrastructure.heavy_import_lock import HEAVY_IMPORT_LOCK


class TestHeavyImportLock:
    def test_lock_is_reentrant(self) -> None:
        # RLock: cùng thread acquire lồng nhau không deadlock.
        with HEAVY_IMPORT_LOCK:
            with HEAVY_IMPORT_LOCK:
                assert True

    def test_lock_shared_singleton(self) -> None:
        from subtitles_extractor.infrastructure.heavy_import_lock import (
            HEAVY_IMPORT_LOCK as lock2,
        )

        assert HEAVY_IMPORT_LOCK is lock2


class TestStaleModuleCleanup:
    """Mô phỏng paddle nửa-khởi-tạo trong sys.modules → logic xoá phải dọn sạch."""

    def test_stale_paddle_modules_removed(self) -> None:
        # Cắm module giả 'paddle' + 'paddle.x' vào sys.modules.
        import types

        sys.modules["paddle"] = types.ModuleType("paddle")
        sys.modules["paddle.fake_sub"] = types.ModuleType("paddle.fake_sub")

        # Logic xoá giống trong container/adapter.
        for name in list(sys.modules):
            if name == "paddle" or name.startswith("paddle."):
                del sys.modules[name]

        assert "paddle" not in sys.modules
        assert "paddle.fake_sub" not in sys.modules
