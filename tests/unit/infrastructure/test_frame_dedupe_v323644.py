"""[v3.23.644] Bộ quyết định khung: chế độ cổ điển giữ nguyên hành vi, chế độ chữ trắng đúng như thiết kế."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from subtitles_extractor.domain.ports.frame_sampler_port import FrameSamplingConfig
from subtitles_extractor.infrastructure.video import frame_dedupe
from subtitles_extractor.infrastructure.video.frame_dedupe import FrameDeduper, FrameVerdict
from subtitles_extractor.infrastructure.video.perceptual_hash import (
    compute_phash,
    hamming_distance,
    pixel_diff_ratio,
)

H, W = 119, 603


def _background(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.integers(10, 90, (H, W, 3), dtype=np.uint8)           # nền tối nhiễu (chuyển động nền)
    return base


def _with_text(background: np.ndarray, text: str, x: int = 40) -> np.ndarray:
    image = background.copy()
    cv2.putText(image, text, (x, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 4, cv2.LINE_AA)
    return image


def _legacy_decision(config: FrameSamplingConfig, image, last_hash, last_image) -> bool:
    """Bản sao logic cũ (``_should_keep_*``): True = giữ (cần OCR)."""
    if last_hash is None or last_image is None:
        return True
    if hamming_distance(compute_phash(image), last_hash) > config.phash_distance_threshold:
        return True
    return pixel_diff_ratio(image, last_image) > config.pixel_diff_threshold


class TestClassicModeIsUnchanged:
    def test_same_decisions_as_the_old_sampler_logic(self) -> None:
        config = FrameSamplingConfig()
        deduper = FrameDeduper(config)
        frames = [_with_text(_background(0), "ABC")] * 3 + [_with_text(_background(1), "XYZ")] * 3
        frames += [_background(2)] * 2 + [_with_text(_background(2), "ABC")]
        last_hash = last_image = None
        for i, image in enumerate(frames):
            expected_keep = _legacy_decision(config, image, last_hash, last_image)
            verdict = deduper.verdict(image, i * 0.04)
            assert (verdict is FrameVerdict.OCR) == expected_keep, i
            assert verdict is not FrameVerdict.BLANK
            if expected_keep:
                deduper.accept(image, i * 0.04)
                last_hash, last_image = compute_phash(image), image


class TestWhiteTextGate:
    @staticmethod
    def _gate() -> FrameDeduper:
        return FrameDeduper(FrameSamplingConfig(white_text_gate=True))

    def test_background_motion_alone_is_not_a_change(self) -> None:
        deduper = self._gate()
        first = _with_text(_background(0), "你好世界")
        assert deduper.verdict(first, 0.0) is FrameVerdict.OCR
        deduper.accept(first, 0.0)
        for i in range(1, 20):                                       # nền đổi hoàn toàn mỗi khung, chữ giữ nguyên
            moving = _with_text(_background(100 + i), "你好世界")
            assert deduper.verdict(moving, i * 0.04) is FrameVerdict.DUPLICATE, i

    def test_different_text_triggers_ocr(self) -> None:
        deduper = self._gate()
        first = _with_text(_background(0), "ABCDEF")
        deduper.verdict(first, 0.0)
        deduper.accept(first, 0.0)
        second = _with_text(_background(0), "UVWXYZ")
        assert deduper.verdict(second, 0.04) is FrameVerdict.OCR

    def test_empty_roi_is_blank_and_next_text_is_always_ocr(self) -> None:
        deduper = self._gate()
        text = _with_text(_background(0), "ABC")
        deduper.verdict(text, 0.0)
        deduper.accept(text, 0.0)
        assert deduper.verdict(_background(1), 0.04) is FrameVerdict.BLANK
        assert deduper.verdict(_background(2), 0.08) is FrameVerdict.BLANK
        assert deduper.verdict(text, 0.12) is FrameVerdict.OCR        # cùng chữ như trước nhưng sau khoảng trống

    def test_forced_refresh_after_max_gap(self) -> None:
        deduper = self._gate()
        text = _with_text(_background(0), "ABC")
        deduper.verdict(text, 0.0)
        deduper.accept(text, 0.0)
        assert deduper.verdict(text, frame_dedupe.MAX_GAP_SEC - 0.1) is FrameVerdict.DUPLICATE
        assert deduper.verdict(text, frame_dedupe.MAX_GAP_SEC + 0.01) is FrameVerdict.OCR

    def test_small_jitter_is_tolerated_but_a_new_character_is_not(self) -> None:
        deduper = self._gate()
        text = _with_text(_background(0), "ABC")
        deduper.verdict(text, 0.0)
        deduper.accept(text, 0.0)
        shifted = np.roll(text, 1, axis=1)                           # rung ngang 1 px
        assert deduper.verdict(shifted, 0.04) is FrameVerdict.DUPLICATE
        longer = _with_text(_background(0), "ABCD")
        assert deduper.verdict(longer, 0.08) is FrameVerdict.OCR

    def test_gate_does_not_compute_perceptual_hashes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(frame_dedupe, "compute_phash", lambda _i: pytest.fail("không cần pHash"))
        deduper = self._gate()
        image = _with_text(_background(0), "ABC")
        deduper.verdict(image, 0.0)
        deduper.accept(image, 0.0)
        deduper.verdict(image, 0.04)

    def test_works_on_different_roi_sizes_and_tiny_inputs(self) -> None:
        deduper = self._gate()
        tiny = np.zeros((6, 10, 3), np.uint8)
        assert deduper.verdict(tiny, 0.0) is FrameVerdict.BLANK
        grey = np.full((30, 90), 255, np.uint8)                      # ảnh xám 2-D
        assert deduper.verdict(grey, 0.04) is FrameVerdict.OCR

    def test_shape_change_resets_the_reference(self) -> None:
        deduper = self._gate()
        a = _with_text(_background(0), "ABC")
        deduper.verdict(a, 0.0)
        deduper.accept(a, 0.0)
        b = cv2.resize(a, (W // 2, H // 2))
        assert deduper.verdict(b, 0.04) is FrameVerdict.OCR
