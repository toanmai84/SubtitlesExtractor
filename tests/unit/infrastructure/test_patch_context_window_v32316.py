"""Test [v3.23.16] vá dòng thiếu theo CỬA SỔ LIỀN KỀ kèm ngữ cảnh (chống dịch lệch/lặp)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
    _compute_patch_windows,
)
from subtitles_extractor.domain.ports.subtitle_translator_port import TranslationLine


class TestComputePatchWindows:
    def test_single_missing_expands_with_padding(self) -> None:
        idx = list(range(291, 311))  # 20 câu
        w = _compute_patch_windows(idx, {300}, 6)
        assert w == [(3, 16)]  # vị trí 294..306

    def test_adjacent_missing_merged(self) -> None:
        idx = list(range(291, 311))
        w = _compute_patch_windows(idx, {299, 300}, 6)
        assert len(w) == 1  # gộp một cửa sổ

    def test_far_missing_separate(self) -> None:
        idx = list(range(291, 311))
        w = _compute_patch_windows(idx, {292, 308}, 3)
        assert len(w) == 2

    def test_missing_at_start(self) -> None:
        idx = list(range(291, 311))
        w = _compute_patch_windows(idx, {291}, 6)
        assert w[0][0] == 0

    def test_empty_inputs(self) -> None:
        assert _compute_patch_windows([], {1}, 6) == []
        assert _compute_patch_windows([1, 2, 3], set(), 6) == []

    def test_missing_not_in_batch_ignored(self) -> None:
        assert _compute_patch_windows([1, 2, 3], {999}, 6) == []


class TestPatchMissingWithContext:
    def _adapter(self) -> GeminiSubtitleTranslator:
        a = GeminiSubtitleTranslator.__new__(GeminiSubtitleTranslator)
        a._retry_count = 3
        return a

    def test_patches_missing_and_neighbors(self) -> None:
        adapter = self._adapter()
        batch = [TranslationLine(index=i, start_ms=0, end_ms=1, text=f"s{i}",
                                 original_text=f"o{i}") for i in range(291, 311)]
        bres = [TranslationLine(index=i, start_ms=0, end_ms=1, text=f"d{i}")
                for i in range(291, 311)]

        windows_called = []

        def fake_tsb(**kw):
            wb = kw["batch"]
            windows_called.append((wb[0].index, wb[-1].index))
            return [TranslationLine(index=l.index, start_ms=0, end_ms=1,
                                    text=f"FIX_{l.index}") for l in wb]

        with patch.object(adapter, "_line_to_payload", lambda l: {"line_no": l.index, "text": l.text}), \
             patch.object(adapter, "_translate_single_batch", fake_tsb):
            out = adapter._patch_missing_with_context(
                batch=batch, batch_result=bres, missing_set={300},
                source_before=[], source_after=[], history_before=[],
                is_preprocess=False, is_literal=True, config=MagicMock(),
                model_name="m", cancel_cb=None, _depth=0, _ctx_size=20,
                video_files=None, dual_payload=False,
            )
        by = {o.index: o.text for o in out}
        assert by[300] == "FIX_300"          # dòng thiếu được vá
        # [v3.23.131] Hàng xóm GIỮ NGUYÊN bản dịch đúng — KHÔNG bị bản vá (có thể lệch)
        # ghi đè. Nhờ fix v128 các dòng kề không còn bị dồn lệch nên không cần dịch lại.
        assert by[297] == "d297"             # lân cận trong cửa sổ: giữ nguyên
        assert by[291] == "d291"             # xa cửa sổ: giữ nguyên
        assert len(out) == 20                # không mất dòng

    def test_window_provides_context(self) -> None:
        # MẤU CHỐT chống lỗi: cửa sổ vá phải KÈM ngữ cảnh (source_before/after khác rỗng).
        adapter = self._adapter()
        batch = [TranslationLine(index=i, start_ms=0, end_ms=1, text=f"s{i}",
                                 original_text=f"o{i}") for i in range(291, 311)]
        bres = [TranslationLine(index=i, start_ms=0, end_ms=1, text=f"d{i}")
                for i in range(291, 311)]
        captured = {}

        def fake_tsb(**kw):
            captured["before"] = kw["source_before"]
            captured["after"] = kw["source_after"]
            return [TranslationLine(index=l.index, start_ms=0, end_ms=1, text="x")
                    for l in kw["batch"]]

        with patch.object(adapter, "_line_to_payload", lambda l: {"line_no": l.index, "text": l.text}), \
             patch.object(adapter, "_translate_single_batch", fake_tsb):
            adapter._patch_missing_with_context(
                batch=batch, batch_result=bres, missing_set={300},
                source_before=[], source_after=[], history_before=[],
                is_preprocess=False, is_literal=True, config=MagicMock(),
                model_name="m", cancel_cb=None, _depth=0, _ctx_size=20,
                video_files=None, dual_payload=False,
            )
        # Cửa sổ 294..306 → có câu trước (291-293) và sau (307-310) làm ngữ cảnh.
        assert len(captured["before"]) > 0
        assert len(captured["after"]) > 0

    def test_full_batch_window_skips(self) -> None:
        # Nếu cửa sổ phủ cả batch → không vá riêng (để halving lo).
        adapter = self._adapter()
        batch = [TranslationLine(index=i, start_ms=0, end_ms=1, text=f"s{i}")
                 for i in range(1, 6)]  # 5 câu
        bres = [TranslationLine(index=i, start_ms=0, end_ms=1, text=f"d{i}")
                for i in range(1, 6)]
        called = []
        with patch.object(adapter, "_line_to_payload", lambda l: {}), \
             patch.object(adapter, "_translate_single_batch",
                          lambda **k: called.append(1) or []):
            out = adapter._patch_missing_with_context(
                batch=batch, batch_result=bres, missing_set={3},
                source_before=[], source_after=[], history_before=[],
                is_preprocess=False, is_literal=True, config=MagicMock(),
                model_name="m", cancel_cb=None, _depth=0, _ctx_size=20,
                video_files=None, dual_payload=False,
            )
        # padding 6 phủ cả batch 5 câu → bỏ qua vá riêng.
        assert called == []
        assert out == bres
