"""[v3.23.527] Hết hạn mức NGÀY phải báo tử tế, không hiện như crash.

Nhật ký thật: ``QuotaExhaustedError`` rơi vào nhánh bắt lỗi chung nên giao diện hiện "Lỗi không
lường trước khi phân tích ngữ cảnh" — người dùng tưởng ứng dụng hỏng, trong khi đây là tình huống
BÌNH THƯỜNG của free-tier (20 request/ngày cho các model Flash mới).
"""
from __future__ import annotations

from pathlib import Path

import pytest

WORKERS = Path("src/subtitles_extractor/presentation/workers")


@pytest.mark.parametrize(
    "filename", ["analyze_context_worker.py", "translate_worker.py"]
)
class TestWorkersHandleQuotaError:
    def _source(self, filename: str) -> str:
        return (WORKERS / filename).read_text(encoding="utf-8")

    def test_catches_quota_error_explicitly(self, filename: str) -> None:
        assert "except QuotaExhaustedError" in self._source(filename)

    def test_caught_before_generic_handler(self, filename: str) -> None:
        """Phải đứng TRƯỚC nhánh chung, nếu không sẽ không bao giờ chạy tới."""
        source = self._source(filename)
        assert source.index("except QuotaExhaustedError") < source.index("except Exception")

    def test_message_offers_next_steps(self, filename: str) -> None:
        source = self._source(filename)
        assert "flash-lite" in source
        assert "DỰ ÁN KHÁC" in source

    def test_logged_as_warning_not_exception(self, filename: str) -> None:
        """Không phải lỗi lập trình nên không cần traceback đầy đủ."""
        source = self._source(filename)
        index = source.index("except QuotaExhaustedError")
        block = source[index : index + 400]
        assert "logger.warning" in block
