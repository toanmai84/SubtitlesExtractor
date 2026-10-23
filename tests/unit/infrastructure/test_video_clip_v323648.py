"""[v3.23.648] Dịch đính video: chỉ gửi khoảng video lô cần (± lề), không gửi cả đoạn ~6 phút.

Log thật 22/10: 111 request, 7,48 triệu token prompt (67 nghìn/request) so với 268 nghìn token đầu ra. Đo ``count_tokens`` trên đoạn
138 s: nguyên đoạn 14 218 token; cắt 30 s còn 2 384–3 094.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from subtitles_extractor.application.services.video_clip_window import (
    DEFAULT_MARGIN_S,
    MIN_CLIP_S,
    clip_window,
)
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    TranslationContext,
    TranslationLine,
    TranslationStageConfig,
    TranslationStageKind,
)
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    ClippedVideoHandle,
    GeminiSubtitleTranslator,
)
from subtitles_extractor.infrastructure.translation.gemini_video_context import RemoteVideoRef


class TestClipWindow:
    def test_window_is_batch_plus_margin_relative_to_chunk_start(self) -> None:
        assert clip_window(360.0, 720.0, 400.0, 460.0) == (400.0 - 8.0 - 360.0, 460.0 + 8.0 - 360.0)

    def test_clamped_inside_the_chunk(self) -> None:
        start, end = clip_window(0.0, 360.0, 2.0, 60.0) or (None, None)
        assert start == 0.0 and end == 60.0 + DEFAULT_MARGIN_S

    @pytest.mark.parametrize("args", [
        (0.0, 3.0, 0.0, 2.0),            # đoạn quá ngắn
        (0.0, 100.0, 0.0, 95.0),         # lô phủ gần hết đoạn: gửi nguyên cho đỡ rủi ro
        (0.0, 100.0, 50.0, 40.0),        # dữ liệu ngược
    ])
    def test_no_clip_when_it_does_not_save_enough_or_input_is_invalid(self, args: tuple[float, ...]) -> None:
        assert clip_window(*args) is None

    def test_tiny_batch_is_widened_to_the_minimum(self) -> None:
        start, end = clip_window(0.0, 360.0, 100.0, 100.5, margin_s=0.0) or (0, 0)
        assert end - start >= MIN_CLIP_S - 1e-6 and 0.0 <= start < end <= 360.0

    def test_batch_at_the_end_of_the_chunk_stays_inside(self) -> None:
        start, end = clip_window(0.0, 360.0, 358.0, 359.0, margin_s=0.0) or (0, 0)
        assert end <= 360.0 and end - start >= MIN_CLIP_S - 1e-6


class TestApplyVideoFps:
    def _adapter(self, fps: float = 1.0) -> GeminiSubtitleTranslator:
        adapter = GeminiSubtitleTranslator(api_key="K")
        adapter._analysis_video_fps = fps
        return adapter

    def test_unclipped_handles_are_passed_through_untouched(self) -> None:
        handle = SimpleNamespace(uri="u", mime_type="video/mp4")
        assert self._adapter()._apply_video_fps([handle]) == [handle]

    def test_clipped_handle_becomes_a_part_with_offsets(self) -> None:
        handle = SimpleNamespace(uri="https://x/files/a", mime_type="video/mp4", name="files/a")
        (part,) = self._adapter()._apply_video_fps([ClippedVideoHandle(handle, 12.5, 98.0)])
        assert part.file_data.file_uri == "https://x/files/a"
        assert part.video_metadata.start_offset == "12.50s" and part.video_metadata.end_offset == "98.00s"
        assert part.video_metadata.fps is None

    def test_offsets_combine_with_fps(self) -> None:
        handle = SimpleNamespace(uri="u", mime_type="video/mp4")
        (part,) = self._adapter(fps=4.0)._apply_video_fps([ClippedVideoHandle(handle, 1.0, 9.0)])
        assert part.video_metadata.fps == 4.0 and part.video_metadata.end_offset == "9.00s"

    def test_clipped_handle_without_uri_falls_back_to_the_raw_handle(self) -> None:
        handle = SimpleNamespace(name="files/a")
        assert self._adapter()._apply_video_fps([ClippedVideoHandle(handle, 1.0, 9.0)]) == [handle]

    def test_proxy_forwards_attributes_and_survives_copy(self) -> None:
        import copy

        handle = SimpleNamespace(name="files/a", uri="u")
        clipped = ClippedVideoHandle(handle, 1.0, 9.0)
        assert clipped.name == "files/a" and clipped.length_s == 8.0
        assert copy.copy(clipped).start_s == 1.0
        with pytest.raises(AttributeError):
            clipped.không_tồn_tại             # noqa: B018


CHUNK_S = 360.0
CHUNKS = 3


def _refs() -> list[RemoteVideoRef]:
    return [RemoteVideoRef(i, f"files/c{i}", i * CHUNK_S, (i + 1) * CHUNK_S, "ACTIVE") for i in range(CHUNKS)]


def _lines(count: int = 180) -> list[TranslationLine]:
    return [
        TranslationLine(
            index=i + 1, start_ms=int(i * 6000) + 100, end_ms=int(i * 6000) + 4000,
            text=f"句{i}", original_text=f"句{i}",
        )
        for i in range(count)
    ]


class _Harness:
    def __init__(self, clip: bool = True) -> None:
        self.adapter = GeminiSubtitleTranslator(api_key="K")
        self.adapter._clip_video_to_batch = clip
        self.calls: list[dict[str, Any]] = []
        self.adapter._resolve_video_handles = lambda refs: [   # type: ignore[assignment]
            SimpleNamespace(name=r.remote_name, uri=f"u/{r.remote_name}", mime_type="video/mp4")
            for r in refs or []
        ]
        self.adapter._ensure_available = lambda: None          # type: ignore[assignment]

        def single(*, batch, video_files, video_token_estimate, **_kw):
            self.calls.append({
                "batch": batch, "files": list(video_files or []), "tokens": video_token_estimate,
            })
            return [replace(line, text=f"dịch {line.index}") for line in batch]

        self.adapter._translate_single_batch = single         # type: ignore[assignment]

    def run(self) -> None:
        lines = _lines()
        self.adapter.translate_stage(
            stage=TranslationStageConfig(
                kind=TranslationStageKind.LITERAL, model_name="m", batch_size=20,
            ),
            context=TranslationContext(target_lang="Vietnamese"),
            source_lines=lines, input_lines=lines, has_prior_translation=False,
            video_refs=_refs(), attach_video=True,
        )


class TestTranslationStage:
    def test_each_batch_sends_only_its_window_and_far_fewer_tokens(self) -> None:
        harness = _Harness()
        harness.run()
        assert len(harness.calls) == 9
        sent_s = old_s = 0.0
        for call in harness.calls:
            start_s = min(l.start_ms for l in call["batch"]) / 1000.0
            end_s = max(l.end_ms for l in call["batch"]) / 1000.0
            old_s += CHUNK_S * sum(1 for r in _refs() if r.start_sec < end_s and r.end_sec > start_s)
            clipped = [h for h in call["files"] if isinstance(h, ClippedVideoHandle)]
            assert clipped, call
            for handle in clipped:
                chunk_index = int(handle.name.removeprefix("files/c"))
                base = chunk_index * CHUNK_S
                assert 0.0 <= handle.start_s < handle.end_s <= CHUNK_S
                assert base + handle.start_s <= start_s + 1e-6 and base + handle.end_s >= min(end_s, base + CHUNK_S) - 1e-6
                sent_s += handle.length_s
            assert call["tokens"] < 150 * 100            # ~2 phút + lề, không phải 6 phút
        assert sent_s < 0.6 * old_s                         # trước: cả đoạn phủ lô (360 s mỗi đoạn)

    def test_disabled_flag_keeps_the_old_whole_chunk_behaviour(self) -> None:
        harness = _Harness(clip=False)
        harness.run()
        assert all(not isinstance(h, ClippedVideoHandle) for c in harness.calls for h in c["files"])
        assert all(c["tokens"] >= 300 * 100 for c in harness.calls[:2])

    def test_missing_handles_never_guess_which_chunk_is_which(self) -> None:
        harness = _Harness()
        original = harness.adapter._resolve_video_handles
        harness.adapter._resolve_video_handles = lambda refs: original(refs)[:0]    # type: ignore[assignment]
        harness.run()
        assert all(not c["files"] for c in harness.calls)

    def test_batch_spanning_two_chunks_clips_each_part_separately(self) -> None:
        harness = _Harness()
        lines = _lines(180)
        # Lô thứ hai (dòng 51–100, 300–600 s) bắc qua mép đoạn 1/2 (360 s).
        harness.adapter.translate_stage(
            stage=TranslationStageConfig(kind=TranslationStageKind.LITERAL, model_name="m", batch_size=50),
            context=TranslationContext(target_lang="Vietnamese"),
            source_lines=lines, input_lines=lines, has_prior_translation=False,
            video_refs=_refs(), attach_video=True,
        )
        spanning = [c for c in harness.calls if len(c["files"]) == 2]
        assert spanning
        first, second = spanning[0]["files"]
        assert isinstance(first, ClippedVideoHandle) and isinstance(second, ClippedVideoHandle)
        assert first.end_s <= CHUNK_S and second.start_s >= 0.0
        assert first.start_s > 0.0                      # chỉ phần cuối đoạn 1, không phải cả 360 s


class TestSettingAndWiring:
    def test_default_is_on_and_container_passes_it_to_the_adapter(self, tmp_path) -> None:
        from pathlib import Path

        from subtitles_extractor.composition.container import ApplicationContainer
        from subtitles_extractor.infrastructure.settings.json_file_repository import JsonFileRepository
        from subtitles_extractor.presentation.pages import editor_page

        container = ApplicationContainer(
            settings_repository=JsonFileRepository(file_path=tmp_path / "c.json"),
            i18n_data_dir=Path(editor_page.__file__).resolve().parents[2] / "data",
            user_data_dir=tmp_path,
        )
        assert container.settings_service.current.translation.clip_video_to_batch is True
        assert container.make_subtitle_translator("K")._clip_video_to_batch is True
        container.settings_service.update(translation={"clip_video_to_batch": False})
        assert container.make_subtitle_translator("K")._clip_video_to_batch is False

    def test_translate_page_checkbox_roundtrips_to_settings(self, tmp_path) -> None:
        from pathlib import Path

        from PySide6.QtWidgets import QApplication

        from subtitles_extractor.composition.container import ApplicationContainer
        from subtitles_extractor.infrastructure.settings.json_file_repository import JsonFileRepository
        from subtitles_extractor.presentation.pages import editor_page
        from subtitles_extractor.presentation.pages.translate_page import TranslatePage

        QApplication.instance() or QApplication([])
        container = ApplicationContainer(
            settings_repository=JsonFileRepository(file_path=tmp_path / "c.json"),
            i18n_data_dir=Path(editor_page.__file__).resolve().parents[2] / "data",
            user_data_dir=tmp_path,
        )
        page = TranslatePage(container)
        assert page._clip_video_check.isChecked() and not page._clip_video_check.isEnabled()
        page._attach_video_check.setChecked(True)
        assert page._clip_video_check.isEnabled()
        page._clip_video_check.setChecked(False)
        assert container.settings_service.current.translation.clip_video_to_batch is False
        page.close()
