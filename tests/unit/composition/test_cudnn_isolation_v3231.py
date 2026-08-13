"""Test [v3.23.1] cô lập cuDNN paddle/torch — process chính sạch torch."""

from __future__ import annotations


class TestPaddleCudnnPriority:
    def test_prioritize_cudnn_no_crash(self) -> None:
        # Hàm phải an toàn trên mọi OS (no-op nếu không phải Windows / không có paddle).
        from subtitles_extractor.composition.bootstrap import _prioritize_paddle_cudnn

        _prioritize_paddle_cudnn()  # không được ném

    def test_main_process_bootstrap_has_no_torch_import(self) -> None:
        # bootstrap.py KHÔNG được chứa lệnh import torch (giữ process chính sạch).
        import inspect

        from subtitles_extractor.composition import bootstrap

        source = inspect.getsource(bootstrap)
        offending = [
            ln.strip() for ln in source.splitlines()
            if ln.strip().startswith("import torch") or ln.strip().startswith("from torch")
        ]
        assert offending == [], f"bootstrap không được import torch: {offending}"

    def test_container_has_no_torch_import(self) -> None:
        import inspect

        from subtitles_extractor.composition import container

        source = inspect.getsource(container)
        offending = [
            ln.strip() for ln in source.splitlines()
            if ln.strip().startswith("import torch") or ln.strip().startswith("from torch")
        ]
        assert offending == [], f"container không được import torch: {offending}"
