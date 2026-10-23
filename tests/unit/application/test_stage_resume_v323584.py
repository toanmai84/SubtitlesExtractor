"""[v3.23.584] Dịch đứt GIỮA giai đoạn → lần sau tiếp tục từ lô đã xong, không dịch lại.

Nhật ký thật v3.23.583 (phim "he thong sung the", 5540 dòng): lần 1 dừng vì hết hạn mức
ngày sau 43 lượt gọi, lần 2 dừng vì mạng chập chờn sau 25 lượt, lần 3 dịch lại giai đoạn
dịch thô từ đầu (96 lượt) — 68 lượt gọi của hạn mức ngày bị vứt bỏ.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from subtitles_extractor.application.services.batch_resume import BatchResume
from subtitles_extractor.application.use_cases.translate_subtitles import (
    TranslateSubtitlesUseCase,
    _compute_checkpoint_key,
    _StageCheckpoint,
    stage_input_fingerprint,
)
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    SubtitleTranslationError,
)
from tests.unit.application.stage_resume_harness import (
    LITERAL,
    STYLE,
    FlakyBatches,
    build_adapter,
    build_request,
    make_line,
)


class TestBatchResume:
    def test_whole_batch_is_reused(self) -> None:
        resume = BatchResume.build([make_line(1, "một", "一"), make_line(2, "hai", "二")])
        batch = [make_line(1, "一"), make_line(2, "二")]
        assert [line.text for line in resume.lines_for(batch)] == ["một", "hai"]

    def test_incomplete_batch_is_translated_again(self) -> None:
        resume = BatchResume.build([make_line(1, "một", "一")])
        assert resume.lines_for([make_line(1, "一"), make_line(2, "二")]) is None

    def test_different_source_is_not_reused(self) -> None:
        resume = BatchResume.build([make_line(1, "một", "一")])
        assert resume.lines_for([make_line(1, "壹")]) is None

    def test_reindexed_batch_maps_back_to_original_indices(self) -> None:
        # Dòng rác 2 bị bỏ: lô đánh số lại 1, 2 ↔ chỉ số gốc 1, 3.
        resume = BatchResume.build(
            [make_line(1, "một", "一"), make_line(3, "ba", "三")],
            reindex_map={1: 1, 2: 3},
        )
        restored = resume.lines_for([make_line(1, "一"), make_line(2, "三")])
        assert [(line.index, line.text) for line in restored] == [(1, "một"), (2, "ba")]
        assert [line.index for line in resume.to_original(restored)] == [1, 3]

    def test_nothing_to_resume(self) -> None:
        assert not BatchResume.build(None)
        assert BatchResume.build(None).lines_for([make_line(1, "一")]) is None


class TestCheckpointPartial:
    def test_partial_survives_reload_and_is_dropped_when_stage_completes(
        self, tmp_path: Path
    ) -> None:
        inputs = [make_line(1, "一"), make_line(2, "二"), make_line(3, "三")]
        fingerprint = stage_input_fingerprint(LITERAL, inputs)
        store = _StageCheckpoint(tmp_path, "k")
        store.add_partial(LITERAL, fingerprint, [make_line(1, "một", "一")])
        store.add_partial(LITERAL, fingerprint, [make_line(2, "hai", "二")])

        reloaded = _StageCheckpoint(tmp_path, "k")
        saved = reloaded.get_partial(LITERAL, fingerprint)
        assert [(line.index, line.text) for line in saved] == [(1, "một"), (2, "hai")]
        assert reloaded.get_partial(STYLE, fingerprint) is None
        assert reloaded.get_partial(LITERAL, "khác") is None

        reloaded.save_stage(LITERAL, [make_line(i, f"dịch {i}") for i in (1, 2, 3)])
        assert _StageCheckpoint(tmp_path, "k").get_partial(LITERAL, fingerprint) is None
        assert not (tmp_path / "k.partial.json").exists()

    def test_corrupt_progress_file_is_ignored(self, tmp_path: Path) -> None:
        (tmp_path / "k.partial.json").write_text("{hỏng", encoding="utf-8")
        store = _StageCheckpoint(tmp_path, "k")
        assert store.get_partial(LITERAL, "x") is None

    def test_progress_of_other_stage_survives_saving_a_stage(
        self, tmp_path: Path
    ) -> None:
        store = _StageCheckpoint(tmp_path, "k")
        store.add_partial(STYLE, "fp", [make_line(1, "a")])
        store.save_stage(LITERAL, [make_line(1, "x")])
        assert store.get_partial(STYLE, "fp") is not None

    def test_progress_of_another_input_is_replaced(self, tmp_path: Path) -> None:
        store = _StageCheckpoint(tmp_path, "k")
        store.add_partial(STYLE, "cũ", [make_line(1, "a")])
        store.add_partial(STYLE, "mới", [make_line(2, "b")])
        assert [line.index for line in store.get_partial(STYLE, "mới")] == [2]
        assert store.get_partial(STYLE, "cũ") is None

    def test_fingerprint_depends_on_stage_and_text(self) -> None:
        lines = [make_line(1, "一")]
        assert stage_input_fingerprint(LITERAL, lines) != stage_input_fingerprint(
            STYLE, lines
        )
        assert stage_input_fingerprint(LITERAL, lines) != stage_input_fingerprint(
            LITERAL, [make_line(1, "壹")]
        )


class TestInterruptedRunResumes:
    def test_second_run_only_translates_the_missing_batches(self, tmp_path: Path) -> None:
        batches = FlakyBatches(fail_at=3)
        use_case = TranslateSubtitlesUseCase(
            translator=build_adapter(batches), checkpoint_dir=tmp_path
        )
        with pytest.raises(SubtitleTranslationError):
            use_case.execute(build_request())
        assert batches.calls == [0, 2, 4]  # 3 lô xong trước khi đứt

        batches.calls.clear()
        response = use_case.execute(build_request())
        # Giai đoạn dịch thô: chỉ 2 lô còn thiếu (6-7, 8-9); tinh chỉnh: đủ 5 lô.
        assert batches.calls == [6, 8, 0, 2, 4, 6, 8]
        assert [event.text for event in response.events] == [
            f"dịch {i}" for i in range(1, 11)
        ]
        # Xong cả lượt → tiến độ dở bị xoá; [v3.23.608] kết quả các giai đoạn được GIỮ
        # để lần sau dùng lại / sửa tay (trước đây cũng bị xoá).
        key = _compute_checkpoint_key(build_request())
        assert (tmp_path / f"{key}.json").exists()
        assert not (tmp_path / f"{key}.partial.json").exists()
        batches.calls.clear()
        again = use_case.execute(build_request())
        assert batches.calls == []
        assert again.reused_stages == (LITERAL, STYLE)

    def test_interruption_in_the_second_stage_keeps_the_first(
        self, tmp_path: Path
    ) -> None:
        batches = FlakyBatches(fail_at=7)  # 5 lô dịch thô + 2 lô tinh chỉnh, rồi đứt
        use_case = TranslateSubtitlesUseCase(
            translator=build_adapter(batches), checkpoint_dir=tmp_path
        )
        with pytest.raises(SubtitleTranslationError):
            use_case.execute(build_request())
        batches.calls.clear()
        use_case.execute(build_request())
        assert batches.calls == [4, 6, 8]

    def test_parallel_batches_resume_too(self, tmp_path: Path) -> None:
        # Đợt 2 gồm lô 4-5 và 6-7: lô 6-7 hỏng, lô 4-5 xong vẫn được lưu.
        batches = FlakyBatches(fail_start=6)
        use_case = TranslateSubtitlesUseCase(
            translator=build_adapter(batches, parallel=2), checkpoint_dir=tmp_path
        )
        with pytest.raises(SubtitleTranslationError):
            use_case.execute(build_request())
        assert sorted(batches.calls) == [0, 2, 4]
        batches.calls.clear()
        use_case.execute(build_request())
        assert sorted(batches.calls[:2]) == [6, 8]
        assert sorted(batches.calls[2:]) == [0, 2, 4, 6, 8]

    def test_without_checkpoint_nothing_is_saved(self) -> None:
        batches = FlakyBatches(fail_at=None)
        use_case = TranslateSubtitlesUseCase(translator=build_adapter(batches))
        use_case.execute(build_request())
        assert len(batches.calls) == 10
