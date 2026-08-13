"""Test [v3.23.10] torch blocker: sys.modules=None khi cài (find_spec→None để
transformers fallback), XOÁ entry khi gỡ (scipy an toàn)."""

from __future__ import annotations

import importlib.util
import sys

from subtitles_extractor.infrastructure.torch_import_blocker import (
    install_torch_import_blocker,
    is_torch_import_blocked,
    uninstall_torch_import_blocker,
)


class TestTorchImportBlocker:
    def teardown_method(self) -> None:
        uninstall_torch_import_blocker()

    def test_find_spec_none_when_installed(self) -> None:
        # MẤU CHỐT cho paddle: find_spec('torch') phải trả None → transformers/paddlex
        # coi như KHÔNG có torch → fallback → paddle dùng cuDNN riêng.
        install_torch_import_blocker()
        assert importlib.util.find_spec("torch") is None

    def test_import_torch_fails_when_installed(self) -> None:
        install_torch_import_blocker()
        try:
            import torch  # noqa: F401

            raise AssertionError("torch không bị chặn")
        except ImportError:
            pass

    def test_uninstall_removes_none_entry(self) -> None:
        # MẤU CHỐT cho scipy/TTS: sau khi gỡ, torch KHÔNG còn trong sys.modules (kể cả
        # None) → sys.modules['torch'] gặp KeyError → scipy.array_api_compat an toàn.
        install_torch_import_blocker()
        uninstall_torch_import_blocker()
        try:
            _ = sys.modules["torch"]
            raised = False
        except KeyError:
            raised = True
        assert raised

    def test_scipy_resample_poly_after_uninstall(self) -> None:
        import numpy as np
        from scipy.signal import resample_poly

        install_torch_import_blocker()
        uninstall_torch_import_blocker()
        # Không crash AttributeError 'NoneType'.Tensor.
        out = resample_poly(np.zeros(200, dtype=np.float32), 4, 1)
        assert len(out) == 800

    def test_blocks_torchaudio_torchvision(self) -> None:
        install_torch_import_blocker()
        for name in ("torchaudio", "torchvision"):
            assert importlib.util.find_spec(name) is None

    def test_allows_numpy(self) -> None:
        install_torch_import_blocker()
        import numpy  # noqa: F401

    def test_idempotent(self) -> None:
        install_torch_import_blocker()
        install_torch_import_blocker()
        assert is_torch_import_blocked()

    def test_state_flag(self) -> None:
        assert not is_torch_import_blocked()
        install_torch_import_blocker()
        assert is_torch_import_blocked()
        uninstall_torch_import_blocker()
        assert not is_torch_import_blocked()
