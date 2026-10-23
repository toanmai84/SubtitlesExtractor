"""[v3.23.650] PreviewSeekService không được kẹt vĩnh viễn khi một lần giải mã ném ngoại lệ lạ / mốc không hợp lệ."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")

from subtitles_extractor.presentation.widgets.preview_seek_service import PreviewSeekService  # noqa: E402


class _Reader:
    def __init__(self) -> None:
        self.calls: list[float] = []
        self.explode_on: set[float] = set()

    def seek(self, ts: float):
        self.calls.append(ts)
        if ts in self.explode_on:
            raise OverflowError("cannot convert float infinity to integer")
        return np.zeros((4, 4, 3), dtype=np.uint8)

    def close(self) -> None:  # pragma: no cover - không dùng
        pass


def _service() -> PreviewSeekService:
    return PreviewSeekService(Path("x.mp4"))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_request_seek_ignores_non_finite_timestamps_v323650(bad: float) -> None:
    service = _service()
    service.request_seek(bad)
    assert service._pending_timestamp_sec is None
    assert service._is_decoding is False


def test_request_seek_clamps_negative_to_zero_v323650() -> None:
    service = _service()
    service._seek_requested.disconnect() if False else None
    service.request_seek(-5.0)
    assert service._pending_timestamp_sec == 0.0


def test_unexpected_decode_exception_does_not_wedge_service_v323650() -> None:
    service = _service()
    reader = _Reader()
    reader.explode_on = {3.0}
    service._reader = reader
    service._pending_timestamp_sec = 3.0
    service._is_decoding = True
    service._on_seek_requested()           # ném OverflowError bên trong → không được thoát khỏi slot
    assert service._is_decoding is False   # trước đây kẹt True → mọi request_seek sau đó bị bỏ qua
    service._pending_timestamp_sec = 4.0
    service._is_decoding = True
    service._on_seek_requested()
    assert reader.calls == [3.0, 4.0]
