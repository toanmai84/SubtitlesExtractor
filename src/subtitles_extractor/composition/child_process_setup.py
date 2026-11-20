"""[v3.23.752] Chuẩn bị TIẾN TRÌNH CON (multiprocessing spawn) giống tiến trình chính trước khi nạp Paddle.

Tiến trình con của exe KHÔNG chạy ``main()`` (``freeze_support`` tiếp quản ngay đầu ``main.py``) nên thiếu các bước
``main()`` làm trước khi nạp thư viện. Log 19/11 09:14 (bản 3.23.751, tiến trình riêng của PaddleOCR-VL):
``No module named 'chardet'`` (shim chardet → charset-normalizer chỉ cài ở ``main()``), rồi
``'NoneType' object has no attribute 'write'`` (exe ``console=False`` → ``sys.stderr is None``) bật hộp thoại
"Unhandled exception in script" trên màn hình người dùng.
"""

from __future__ import annotations

import io
import sys


class _NullStream(io.TextIOBase):
    def write(self, text: str) -> int:
        return len(text)

    def flush(self) -> None:
        return


def prepare_child_process() -> None:
    """Đúng thứ tự của ``main()`` + ``bootstrap._prepare_process_for_paddle`` (không tạo giao diện)."""
    if sys.stdout is None:
        sys.stdout = _NullStream()
    if sys.stderr is None:
        sys.stderr = _NullStream()
    from subtitles_extractor.infrastructure.process.hidden_process import (
        install_hidden_console_default,
    )

    install_hidden_console_default()              # paddle chạy `where nvcc` lúc nạp: không chớp cửa sổ cmd
    from subtitles_extractor.infrastructure.model_store import (
        configure_all_model_stores,
    )

    configure_all_model_stores()                  # mô hình ở models/ của ứng dụng — không tải lại vào cache khác
    from subtitles_extractor.infrastructure.compat import install_chardet_shim

    install_chardet_shim()                        # paddlex `import chardet` → charset-normalizer (bản đóng gói không có chardet)
    from subtitles_extractor.composition.bootstrap import (
        _prepare_process_for_paddle,
    )

    _prepare_process_for_paddle()                 # lõi paddle tải-lúc-chạy, cuDNN của paddle, chặn torch


__all__ = ["prepare_child_process"]
