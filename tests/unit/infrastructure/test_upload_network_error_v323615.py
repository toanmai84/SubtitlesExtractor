"""[v3.23.615] Lỗi mạng khi tải video ngữ cảnh lên Gemini là ``VideoContextError``.

Bấm thử toàn ứng dụng với video thật (mạng tới Gemini bị chặn): ``httpx.ProxyError`` nổ
ngay trong ``files.upload`` và lọt thẳng ra ngoài — khâu dịch ghi "Lỗi không lường trước
khi dịch phụ đề" kèm traceback, còn khâu phân tích (vốn biết lùi về phân tích từ phụ đề
khi gặp ``VideoContextError``) hỏng hẳn. Mạng chập chờn ở máy người dùng đi đúng đường
này.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from subtitles_extractor.domain.ports.video_context_port import VideoChunk
from subtitles_extractor.infrastructure.translation.gemini_video_context import (
    GeminiVideoContextProvider,
    VideoContextError,
)

httpx = pytest.importorskip("httpx")


def _provider(tmp_path: Path, error: Exception) -> GeminiVideoContextProvider:
    provider = GeminiVideoContextProvider(api_key="k", cache_db_path=tmp_path / "c.db")

    def upload(**_kwargs: object) -> None:
        raise error

    provider._client = SimpleNamespace(files=SimpleNamespace(upload=upload, get=upload))
    return provider


@pytest.mark.parametrize(
    "error",
    [
        httpx.ProxyError("403 Forbidden"),
        httpx.ConnectError("[Errno 11001] getaddrinfo failed"),
        httpx.ReadTimeout("The read operation timed out"),
    ],
)
def test_network_error_becomes_video_context_error(
    tmp_path: Path, error: Exception
) -> None:
    chunk_path = tmp_path / "doan.mp4"
    chunk_path.write_bytes(b"\x00" * 64)
    provider = _provider(tmp_path, error)
    with pytest.raises(VideoContextError, match="Không tải được đoạn 1"):
        provider.upload_chunk(VideoChunk(0, chunk_path, 0.0, 10.0, is_full_video=True))
