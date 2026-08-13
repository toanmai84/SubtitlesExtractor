"""Cấu hình logging tập trung — Module re-export cho backward compatibility.

[v2.29 UPGRADE]: Module này đã được **rewrite từ Python ``logging`` mặc định
sang Loguru**. Toàn bộ logic Loguru nằm trong ``loguru_config.py`` để tách
biệt rõ ràng — module này chỉ re-export hàm ``setup_logging`` để bootstrap
cũ tiếp tục hoạt động không cần thay đổi.

Cách dùng (như cũ):
    >>> from subtitles_extractor.infrastructure.logging.setup import setup_logging
    >>> setup_logging(level=logging.INFO, log_dir=Path("./logs"))

Hoặc dùng API Loguru mới (preferred):
    >>> from subtitles_extractor.infrastructure.logging.loguru_config import setup_loguru
    >>> setup_loguru(level="INFO", log_dir=Path("./logs"))
"""

from __future__ import annotations

from subtitles_extractor.infrastructure.logging.loguru_config import (
    setup_logging,
    setup_loguru,
)

__all__ = ["setup_logging", "setup_loguru"]
