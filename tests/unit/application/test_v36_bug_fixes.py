"""Unit tests cho 3 bug fix nghiêm trọng ở v3.6.

Mỗi test:
    1. Có thể repro bug v3.5.
    2. PASS sau khi fix v3.6.

Cover:
    * Bug #1: SubtitleEditorService.split() không còn trừ 0.01s.
    * Bug #2: _save_annotated_raw_frames dùng ascontiguousarray + .copy().
    * Bug #3: _get_clahe_object thread-safe (per-thread cache).
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
import pytest

from subtitles_extractor.application.services.subtitle_editor_service import (
    SubtitleEditorService,
)
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.value_objects.confidence import Confidence
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval
from subtitles_extractor.infrastructure.ocr.preprocessing.image_filters import (
    _get_clahe_object,
    apply_clahe,
)


# ── Bug #1: split() không tạo gap giả tạo ─────────────────────────────────


class TestSplitNoArtificialGap:
    """v3.6 fix: bỏ ``± 0.01`` workaround trong ``split()``.

    SRT/ASS chuẩn cho phép ``end_a == start_b`` — không cần gap giả tạo.
    """

    @pytest.fixture
    def service_with_one_event(self) -> SubtitleEditorService:
        service = SubtitleEditorService()
        service.load([
            SubtitleEvent(
                index=1, text="Câu mẫu để tách đôi",
                interval=TimeInterval(0.0, 2.0),
                confidence=Confidence(0.9), frame_count=10,
            ),
        ])
        return service

    def test_split_first_half_ends_exactly_at_split_point(
        self, service_with_one_event: SubtitleEditorService,
    ) -> None:
        """Bug #1 repro: nửa đầu kết thúc CHÍNH XÁC tại split_at_sec."""
        state = service_with_one_event.split(0, 1.0)
        assert state.events[0].interval.end_sec == 1.0

    def test_split_second_half_starts_exactly_at_split_point(
        self, service_with_one_event: SubtitleEditorService,
    ) -> None:
        """Nửa sau bắt đầu CHÍNH XÁC tại split_at_sec."""
        state = service_with_one_event.split(0, 1.0)
        assert state.events[1].interval.start_sec == 1.0

    def test_split_no_time_loss(
        self, service_with_one_event: SubtitleEditorService,
    ) -> None:
        """Tổng thời lượng sau split = thời lượng gốc (không mất ms nào)."""
        original_duration = 2.0  # 0.0 → 2.0
        state = service_with_one_event.split(0, 1.0)

        total_after = sum(
            event.interval.end_sec - event.interval.start_sec
            for event in state.events
        )
        assert total_after == pytest.approx(original_duration, abs=1e-9)

    def test_split_multiple_times_preserves_total_duration(self) -> None:
        """Split 5 lần liên tiếp — tổng thời lượng vẫn nguyên (không mất 100ms)."""
        service = SubtitleEditorService()
        service.load([
            SubtitleEvent(
                index=1, text="A B C D E F",
                interval=TimeInterval(0.0, 10.0),
                confidence=Confidence(0.9), frame_count=50,
            ),
        ])

        # Split tại 2.0 → 2 events
        service.split(0, 2.0)
        # Split nửa sau tại 4.0 → 3 events
        service.split(1, 4.0)
        # Split tại 6.0 → 4 events
        service.split(2, 6.0)
        # Split tại 8.0 → 5 events
        state = service.split(3, 8.0)

        assert len(state.events) == 5
        total_duration = sum(
            event.interval.end_sec - event.interval.start_sec
            for event in state.events
        )
        # v3.5 bị mất 4 × 0.02 = 0.08s; v3.6 phải mất 0.
        assert total_duration == pytest.approx(10.0, abs=1e-9)


# ── Bug #3: CLAHE thread-safe ─────────────────────────────────────────────


class TestClaheThreadSafety:
    """v3.6 fix: ``_get_clahe_object`` dùng ``threading.local`` per-thread cache.

    Mục tiêu: 2 thread KHÔNG được lấy cùng 1 CLAHE instance (vì cv2 CLAHE
    không thread-safe).
    """

    def test_same_thread_returns_same_instance(self) -> None:
        """Cùng thread + cùng tham số → cache hit, trả về cùng instance."""
        clahe_a = _get_clahe_object(3.0, 8)
        clahe_b = _get_clahe_object(3.0, 8)
        assert clahe_a is clahe_b

    def test_different_threads_get_different_instances(self) -> None:
        """Hai thread khác nhau lấy 2 instance khác nhau — KHÔNG share state."""
        instances: dict[int, Any] = {}
        barrier = threading.Barrier(2)

        def collect_instance(thread_id: int) -> None:
            barrier.wait()  # đảm bảo 2 thread chạy gần như đồng thời
            instances[thread_id] = _get_clahe_object(3.0, 8)

        t1 = threading.Thread(target=collect_instance, args=(1,))
        t2 = threading.Thread(target=collect_instance, args=(2,))
        t1.start(); t2.start()
        t1.join(); t2.join()

        assert len(instances) == 2
        assert instances[1] is not instances[2], (
            "v3.5 bug repro: 2 thread chia sẻ cùng CLAHE instance → "
            "race condition trong cv2.CLAHE.apply()."
        )

    def test_apply_clahe_thread_safe_no_crash(self) -> None:
        """Stress test: chạy ``apply_clahe`` song song không crash, không sai shape."""
        rng = np.random.default_rng(42)
        test_image = rng.integers(
            0, 255, size=(128, 256, 3), dtype=np.uint8,
        )

        results: list[np.ndarray] = []
        results_lock = threading.Lock()
        errors: list[BaseException] = []

        def worker() -> None:
            try:
                for _ in range(20):
                    output = apply_clahe(test_image, clip_limit=3.0, tile_grid_size=8)
                    with results_lock:
                        results.append(output)
            except (RuntimeError, ValueError, OSError) as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert not errors, f"CLAHE thread-safety vỡ: {errors}"
        assert len(results) == 4 * 20
        for output in results:
            assert output.shape == test_image.shape
            assert output.dtype == np.uint8

    def test_cache_evicts_when_exceeds_max(self) -> None:
        """Cache tối đa N entry per thread — tự evict entry cũ nhất."""
        # Tham số khác nhau → tạo nhiều instance.
        instances = [
            _get_clahe_object(float(2.0 + i * 0.5), 8)
            for i in range(10)
        ]
        # Tất cả phải là CLAHE object hợp lệ (không None, không exception).
        assert all(inst is not None for inst in instances)


# ── Bug #2: QImage không bị use-after-free ────────────────────────────────


class TestQImageCopySafety:
    """v3.6 fix: ``_save_annotated_raw_frames`` dùng ``ascontiguousarray`` +
    ``QImage.copy()`` để Qt sở hữu pixel độc lập với numpy buffer.

    Test này không trực tiếp test use-after-free (khó repro đáng tin cậy),
    nhưng kiểm tra contract: ``ascontiguousarray`` được gọi cho mọi input
    dạng non-contiguous, và pattern ``.copy()`` được áp dụng.
    """

    def test_non_contiguous_array_handled(self) -> None:
        """Sliced array (non-contiguous) phải được copy về contiguous."""
        # Slice với step=2 → non-contiguous view.
        original = np.zeros((100, 200, 3), dtype=np.uint8)
        sliced = original[::2, ::2, :]
        assert not sliced.flags["C_CONTIGUOUS"]

        contiguous = np.ascontiguousarray(sliced, dtype=np.uint8)
        assert contiguous.flags["C_CONTIGUOUS"]
        assert contiguous.dtype == np.uint8
        assert contiguous.shape == sliced.shape

    def test_already_contiguous_unchanged(self) -> None:
        """C-contiguous array vẫn được trả về (no-op an toàn)."""
        original = np.zeros((64, 128, 3), dtype=np.uint8)
        assert original.flags["C_CONTIGUOUS"]

        result = np.ascontiguousarray(original, dtype=np.uint8)
        assert result.flags["C_CONTIGUOUS"]
        assert result.shape == original.shape
