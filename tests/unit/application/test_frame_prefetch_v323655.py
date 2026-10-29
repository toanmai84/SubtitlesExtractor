"""[v3.23.655] Giải mã khung ở luồng riêng: đúng thứ tự, lỗi không bị nuốt, dừng sớm thì đóng bộ giải mã."""

from __future__ import annotations

import threading
import time

import pytest

from subtitles_extractor.application.services.frame_prefetch import prefetched


def test_order_and_thread() -> None:
    seen_threads: set[str] = set()

    def source():
        for i in range(1000):
            seen_threads.add(threading.current_thread().name)
            yield i

    assert list(prefetched(source, maxsize=8)) == list(range(1000))
    assert seen_threads == {"frame-decoder"}


def test_error_is_reraised_after_items() -> None:
    def source():
        yield 1
        yield 2
        raise RuntimeError("NVDEC hỏng")

    got = []
    with pytest.raises(RuntimeError, match="NVDEC"):
        for item in prefetched(source, maxsize=4):
            got.append(item)
    assert got == [1, 2]


def test_early_stop_closes_decoder_and_thread_exits() -> None:
    closed = threading.Event()

    def source():
        try:
            for i in range(10_000):
                yield i
        finally:
            closed.set()

    for item in prefetched(source, maxsize=4):
        if item == 5:
            break
    assert closed.wait(5)
    time.sleep(0.05)
    assert not any(t.name == "frame-decoder" and t.is_alive() for t in threading.enumerate())


def test_disabled_runs_inline() -> None:
    names = []

    def source():
        names.append(threading.current_thread().name)
        yield 1

    assert list(prefetched(source, maxsize=0)) == [1]
    assert names == [threading.current_thread().name]


def test_decoding_overlaps_ocr() -> None:
    """Giải mã 20 ms + OCR 20 ms mỗi khung: nối tiếp ~0,8 s cho 20 khung, chạy song song ~0,4 s."""
    def source():
        for i in range(20):
            time.sleep(0.02)
            yield i

    started = time.monotonic()
    for _ in prefetched(source, maxsize=8):
        time.sleep(0.02)
    assert time.monotonic() - started < 0.65


def test_byte_budget_limits_queued_memory_v323655() -> None:
    """[v3.23.655] Rà soát: hàng đợi 256 khung không ROI trên 1080p ~1,5 GB — giới hạn thêm theo dung lượng."""
    import threading
    import time

    from subtitles_extractor.application.services.frame_prefetch import prefetched

    produced: list[int] = []

    def factory():
        for i in range(50):
            produced.append(i)
            yield i

    gen = prefetched(factory, maxsize=256, max_bytes=10, size_of=lambda _x: 4)
    assert next(gen) == 0
    time.sleep(0.3)
    # 1 đã lấy + tối đa 2 phần tử 4 byte chờ (8 ≤ 10) + 1 đang chờ chỗ → còn xa 50
    assert len(produced) <= 5
    assert list(gen) == list(range(1, 50))
    assert not any(t.name == "frame-decoder" and t.is_alive() for t in threading.enumerate())
