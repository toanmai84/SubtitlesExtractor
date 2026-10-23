"""[v3.23.608] Quản lý / sửa / dùng lại bộ nhớ đệm TỪNG giai đoạn dịch.

Câu hỏi của người dùng (13/10): "Khi dịch ta có thể quản lý cache của từng giai đoạn dịch
không? có thể chỉnh sửa nội dung của từng giai đoạn dịch không? có tận dụng lại nội dung
của từng giai đoạn dịch không?" — ở v3.23.607 câu trả lời là KHÔNG:

* bộ nhớ đệm bị XOÁ ngay khi lượt dịch xong → đổi một giai đoạn là dịch lại cả lượt;
* không có chỗ sửa nội dung một giai đoạn (chỉ sửa được bản cuối, và mất khi dịch lại);
* bật Visual Cues mà đổi model giai đoạn đầu (vì hết hạn mức) là đổi tên file → mất
  mọi giai đoạn đã xong (khoá chứa model quét Visual Cues = model giai đoạn đầu).

Chạy THẬT use case + adapter Gemini, chỉ thay bước gọi API một lô.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from subtitles_extractor.application.services.stage_cache import (
    StageAction,
    StageCacheService,
    StageReason,
)
from subtitles_extractor.application.services.stage_checkpoint import (
    CheckpointIdentity,
    StageCheckpoint,
    StageNotCachedError,
    StageRun,
    prune_stage_checkpoints,
)
from subtitles_extractor.application.use_cases.translate_subtitles import (
    TranslateSubtitlesRequest,
    TranslateSubtitlesUseCase,
    build_stage_cache_scope,
    checkpoint_identity,
)
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    TranslationContext,
    TranslationLine,
    TranslationStageConfig,
    TranslationStageKind,
)
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
)

LITERAL = TranslationStageKind.LITERAL
STYLE = TranslationStageKind.STYLE
PREPROCESS = TranslationStageKind.PREPROCESS
LOCALIZE = TranslationStageKind.LOCALIZE

STRONG = "gemini-3.8-flash"
LITE = "gemini-3.5-flash-lite"
LINES = 10
BATCH = 2


class _Batches:
    """Thay bước gọi API một lô: dịch thô ra "dịch i", tinh chỉnh giữ nguyên câu vào."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int]] = []

    def __call__(self, *, batch, start_idx, model_name, is_literal=False, **_kwargs):
        self.calls.append(("literal" if is_literal else "style", model_name, start_idx))
        if is_literal:
            return [replace(line, text=f"dịch {line.index}") for line in batch]
        return list(batch)

    def of(self, stage: str) -> list[int]:
        return [start for name, _model, start in self.calls if name == stage]


def _use_case(tmp_path: Path, batches: _Batches) -> TranslateSubtitlesUseCase:
    adapter = GeminiSubtitleTranslator(api_key="KEY")
    adapter._ensure_available = lambda: None  # type: ignore[method-assign]
    adapter._translate_single_batch = batches  # type: ignore[method-assign]
    return TranslateSubtitlesUseCase(translator=adapter, checkpoint_dir=tmp_path)


def _request(
    literal_model: str = STRONG,
    style_model: str = STRONG,
    *,
    characters: str = "",
    style_temperature: float = 0.3,
    visual_cues: bool = False,
    stages: tuple[TranslationStageKind, ...] = (LITERAL, STYLE),
) -> TranslateSubtitlesRequest:
    models = {LITERAL: literal_model, STYLE: style_model}
    configs = [
        TranslationStageConfig(
            kind=kind,
            model_name=models.get(kind, STRONG),
            batch_size=BATCH,
            temperature=style_temperature if kind is STYLE else 0.3,
        )
        for kind in stages
    ]
    return TranslateSubtitlesRequest(
        events=[
            SubtitleEvent(index=i, text=f"句{i}", interval=TimeInterval(i, i + 0.9))
            for i in range(1, LINES + 1)
        ],
        stages=configs,
        context=TranslationContext(
            target_lang="Vietnamese", source_lang="zh", characters=characters
        ),
        enable_visual_cues=visual_cues,
        visual_cues_model=configs[0].model_name,
    )


def _texts(events: list[SubtitleEvent]) -> list[str]:
    return [event.text for event in events]


class TestFinishedRunIsReused:
    def test_second_run_with_the_same_settings_calls_no_api(self, tmp_path: Path) -> None:
        batches = _Batches()
        use_case = _use_case(tmp_path, batches)
        first = use_case.execute(_request())
        assert len(batches.calls) == 2 * LINES // BATCH
        batches.calls.clear()
        second = use_case.execute(_request())
        assert batches.calls == []
        assert second.reused_stages == (LITERAL, STYLE)
        assert _texts(second.events) == _texts(first.events)

    def test_changing_only_the_style_stage_reuses_the_literal_stage(
        self, tmp_path: Path
    ) -> None:
        batches = _Batches()
        use_case = _use_case(tmp_path, batches)
        use_case.execute(_request())
        batches.calls.clear()
        response = use_case.execute(_request(style_temperature=0.9))
        assert batches.of("literal") == []
        assert batches.of("style") == [0, 2, 4, 6, 8]
        assert response.reused_stages == (LITERAL,)

    def test_adding_a_stage_reuses_the_ones_already_done(self, tmp_path: Path) -> None:
        batches = _Batches()
        use_case = _use_case(tmp_path, batches)
        use_case.execute(_request(stages=(LITERAL,)))
        batches.calls.clear()
        response = use_case.execute(_request())
        assert batches.of("literal") == []
        assert batches.of("style") == [0, 2, 4, 6, 8]
        assert response.reused_stages == (LITERAL,)


class TestEditingAStage:
    def test_edit_in_literal_retranslates_only_the_style_batch_of_that_line(
        self, tmp_path: Path
    ) -> None:
        batches = _Batches()
        use_case = _use_case(tmp_path, batches)
        use_case.execute(_request())
        scope = build_stage_cache_scope(_request(), tmp_path)
        outcome = StageCacheService().save_edits(scope, LITERAL, {5: "  sửa tay 5 "})
        assert outcome.result.changed == (5,)
        assert outcome.downstream == (STYLE,)

        batches.calls.clear()
        response = use_case.execute(_request())
        # Dịch thô dùng lại nguyên (đã gồm dòng sửa); tinh chỉnh chỉ dịch lô 5-6.
        assert batches.calls == [("style", STRONG, 4)]
        assert response.reused_stages == (LITERAL,)
        assert response.carried_lines == {STYLE: LINES - 1}
        assert response.stage_outputs[LITERAL][4].text == "sửa tay 5"
        assert response.events[4].text == "sửa tay 5"

    def test_prediction_matches_what_the_run_does(self, tmp_path: Path) -> None:
        batches = _Batches()
        use_case = _use_case(tmp_path, batches)
        use_case.execute(_request())
        service = StageCacheService()
        scope = build_stage_cache_scope(_request(), tmp_path)
        service.save_edits(scope, LITERAL, {5: "sửa tay 5"})
        overview = service.overview(scope)
        assert overview.entry(LITERAL).action is StageAction.REUSE
        assert overview.entry(LITERAL).edited_lines == 1
        style = overview.entry(STYLE)
        assert style.action is StageAction.PARTIAL
        assert style.reason is StageReason.INPUT_CHANGED
        response = use_case.execute(_request())
        assert response.carried_lines == {STYLE: style.reusable_lines}

    def test_manual_edit_survives_a_retranslation_with_a_stronger_model(
        self, tmp_path: Path
    ) -> None:
        batches = _Batches()
        use_case = _use_case(tmp_path, batches)
        use_case.execute(_request(LITE, LITE))
        scope = build_stage_cache_scope(_request(LITE, LITE), tmp_path)
        StageCacheService().save_edits(scope, STYLE, {3: "chốt 3"})

        batches.calls.clear()
        response = use_case.execute(_request(LITE, STRONG))
        # Bản tinh chỉnh của flash-lite không dùng thay 3.8-flash → dịch lại cả giai đoạn…
        assert batches.of("style") == [0, 2, 4, 6, 8]
        # …nhưng dòng người dùng đã chốt vẫn thắng bản model mới.
        assert response.events[2].text == "chốt 3"
        assert response.reapplied_edits == {STYLE: 1}
        edits = StageCheckpoint(tmp_path, scope.key).stage_edits(STYLE)
        assert edits[3].text == "chốt 3"
        assert edits[3].model_text == "dịch 3"

    def test_changed_roster_retranslates_but_keeps_manual_edits(
        self, tmp_path: Path
    ) -> None:
        batches = _Batches()
        use_case = _use_case(tmp_path, batches)
        use_case.execute(_request(characters="A: nam"))
        scope = build_stage_cache_scope(_request(characters="A: nam"), tmp_path)
        StageCacheService().save_edits(scope, LITERAL, {2: "chốt 2"})

        batches.calls.clear()
        response = use_case.execute(_request(characters="A: nữ"))
        assert batches.of("literal") == [0, 2, 4, 6, 8]
        assert response.stage_outputs[LITERAL][1].text == "chốt 2"
        assert response.events[1].text == "chốt 2"

    def test_writing_back_the_model_text_removes_the_edit(self, tmp_path: Path) -> None:
        use_case = _use_case(tmp_path, _Batches())
        use_case.execute(_request())
        service = StageCacheService()
        scope = build_stage_cache_scope(_request(), tmp_path)
        service.save_edits(scope, LITERAL, {1: "sửa 1", 2: "sửa 2"})
        outcome = service.save_edits(scope, LITERAL, {1: "dịch 1", 2: "sửa 2", 99: "x"})
        assert outcome.result.reverted == (1,)
        assert outcome.result.changed == ()
        assert outcome.result.unknown == (99,)
        rows = {row.index: row for row in service.editor_rows(scope, LITERAL)}
        assert not rows[1].edited and rows[1].text == "dịch 1"
        assert rows[2].edited and rows[2].model_text == "dịch 2"
        assert service.discard_edits(scope, LITERAL) == 1
        assert service.stage_lines(scope, LITERAL)[1].text == "dịch 2"

    def test_editor_shows_source_and_stage_input(self, tmp_path: Path) -> None:
        use_case = _use_case(tmp_path, _Batches())
        use_case.execute(_request())
        service = StageCacheService()
        scope = build_stage_cache_scope(_request(), tmp_path)
        service.save_edits(scope, LITERAL, {4: "sửa 4"})
        literal_rows = service.editor_rows(scope, LITERAL)
        assert literal_rows[3].source_text == "句4"
        assert literal_rows[3].input_text == "句4"
        style_rows = service.editor_rows(scope, STYLE)
        # Đầu vào hiển thị là bản thô hiện tại (khớp vân tay mọi dòng còn lại), gồm cả
        # dòng vừa sửa — đúng thứ giai đoạn tinh chỉnh sẽ nhận ở lần dịch tới.
        assert style_rows[0].input_text == "dịch 1"
        assert style_rows[3].input_text == "sửa 4"
        with pytest.raises(StageNotCachedError):
            service.editor_rows(scope, LOCALIZE)


class TestDeleting:
    def test_deleted_stage_is_translated_again_and_keeps_edits(
        self, tmp_path: Path
    ) -> None:
        batches = _Batches()
        use_case = _use_case(tmp_path, batches)
        use_case.execute(_request())
        service = StageCacheService()
        scope = build_stage_cache_scope(_request(), tmp_path)
        service.save_edits(scope, STYLE, {7: "chốt 7"})
        service.delete_stage(scope, STYLE, keep_edits=True)
        assert service.overview(scope).entry(STYLE).action is StageAction.TRANSLATE

        batches.calls.clear()
        response = use_case.execute(_request())
        assert batches.of("style") == [0, 2, 4, 6, 8]
        assert response.events[6].text == "chốt 7"

    def test_deleting_with_edits_drops_them(self, tmp_path: Path) -> None:
        use_case = _use_case(tmp_path, _Batches())
        use_case.execute(_request())
        service = StageCacheService()
        scope = build_stage_cache_scope(_request(), tmp_path)
        service.save_edits(scope, STYLE, {7: "chốt 7"})
        service.delete_stage(scope, STYLE, keep_edits=False)
        response = use_case.execute(_request())
        assert response.events[6].text == "dịch 7"

    def test_delete_all_removes_files_of_every_key(self, tmp_path: Path) -> None:
        request = _request(visual_cues=True)
        identity = checkpoint_identity(request)
        StageCheckpoint(tmp_path, identity.key).save_stage(LITERAL, _lines())
        StageCheckpoint(tmp_path, identity.previous_key).save_stage(LITERAL, _lines())
        scope = build_stage_cache_scope(request, tmp_path)
        assert StageCacheService().delete_all(scope) == 2
        assert list(tmp_path.glob("*.json")) == []


class TestVisualCuesKey:
    def test_switching_the_first_stage_model_keeps_finished_stages(
        self, tmp_path: Path
    ) -> None:
        """Bật Visual Cues: model quét = model giai đoạn đầu. v3.23.607 đưa nó vào khoá
        file nên đổi model (vì hết hạn mức) là mất mọi giai đoạn đã xong."""
        batches = _Batches()
        use_case = _use_case(tmp_path, batches)
        use_case.execute(_request(STRONG, STRONG, visual_cues=True))
        batches.calls.clear()
        response = use_case.execute(_request(LITE, LITE, visual_cues=True))
        assert batches.calls == []
        assert response.reused_stages == (LITERAL, STYLE)

    def test_file_written_by_v323607_is_adopted(self, tmp_path: Path) -> None:
        request = _request(visual_cues=True)
        identity = checkpoint_identity(request)
        assert identity.previous_key != identity.key
        sources = TranslateSubtitlesUseCase._events_to_lines(request.events)
        scope = build_stage_cache_scope(request, tmp_path)
        old = StageCheckpoint(tmp_path, identity.previous_key, scope.context_fingerprint)
        literal = [replace(line, text=f"cũ {line.index}") for line in sources]
        old.save_stage(
            LITERAL, literal, run=scope.run_for(request.stages[0], sources),
            input_lines=sources,
        )
        batches = _Batches()
        response = _use_case(tmp_path, batches).execute(request)
        assert batches.of("literal") == []
        assert response.stage_outputs[LITERAL][0].text == "cũ 1"
        assert not (tmp_path / f"{identity.previous_key}.json").exists()

    def test_visual_cues_still_separate_the_cache(self) -> None:
        on = checkpoint_identity(_request(visual_cues=True)).key
        off = checkpoint_identity(_request(visual_cues=False)).key
        assert on != off
        assert checkpoint_identity(_request(visual_cues=False)).previous_key == off


class TestOverview:
    def _prepared(self, tmp_path: Path) -> StageCacheService:
        _use_case(tmp_path, _Batches()).execute(_request(LITE, LITE))
        return StageCacheService()

    def test_same_settings_reuse_everything(self, tmp_path: Path) -> None:
        service = self._prepared(tmp_path)
        scope = build_stage_cache_scope(_request(LITE, LITE), tmp_path)
        overview = service.overview(scope)
        assert overview.exists
        assert overview.stored_kinds == [LITERAL, STYLE]
        assert [entry.action for entry in overview.entries] == [
            StageAction.ABSENT, StageAction.REUSE, StageAction.REUSE, StageAction.ABSENT,
        ]
        literal = overview.entry(LITERAL)
        assert literal.stored_model == LITE
        assert literal.line_count == LINES
        assert literal.completed_at

    def test_stronger_model_upstream_makes_the_next_stage_wait(
        self, tmp_path: Path
    ) -> None:
        service = self._prepared(tmp_path)
        overview = service.overview(
            build_stage_cache_scope(_request(STRONG, LITE), tmp_path)
        )
        literal = overview.entry(LITERAL)
        assert literal.action is StageAction.RETRANSLATE
        assert literal.reason is StageReason.WEAKER_MODEL
        style = overview.entry(STYLE)
        assert style.action is StageAction.PARTIAL
        assert style.reason is StageReason.UPSTREAM_PENDING
        assert style.reusable_lines == -1

    def test_config_change_disabled_stage_and_roster_change(self, tmp_path: Path) -> None:
        service = self._prepared(tmp_path)
        changed = service.overview(
            build_stage_cache_scope(_request(LITE, LITE, style_temperature=0.9), tmp_path)
        )
        assert changed.entry(STYLE).action is StageAction.RETRANSLATE
        assert changed.entry(STYLE).reason is StageReason.CONFIG_CHANGED
        only_literal = service.overview(
            build_stage_cache_scope(_request(LITE, LITE, stages=(LITERAL,)), tmp_path)
        )
        assert only_literal.entry(STYLE).action is StageAction.SKIPPED
        roster = service.overview(
            build_stage_cache_scope(_request(LITE, LITE, characters="B: nữ"), tmp_path)
        )
        assert roster.context_changed
        assert roster.entry(LITERAL).reason is StageReason.CONTEXT_CHANGED
        # Mở để xem không được xoá gì (khác lúc dịch thật).
        assert service.overview(
            build_stage_cache_scope(_request(LITE, LITE), tmp_path)
        ).entry(STYLE).action is StageAction.REUSE

    def test_partial_progress_is_reported(self, tmp_path: Path) -> None:
        request = _request()
        scope = build_stage_cache_scope(request, tmp_path)
        sources = list(scope.source_lines)
        run = scope.run_for(request.stages[0], sources)
        checkpoint = StageCheckpoint(tmp_path, scope.key, scope.context_fingerprint)
        checkpoint.add_partial(LITERAL, run.input_fingerprint, sources[:4], run)
        entry = StageCacheService().overview(scope).entry(LITERAL)
        assert entry.action is StageAction.PARTIAL
        assert entry.partial_lines == 4
        assert entry.reusable_lines == 4


def _lines(count: int = 3) -> list[TranslationLine]:
    return [
        TranslationLine(index=i, start_ms=i * 1000, end_ms=i * 1000 + 900, text=f"t{i}")
        for i in range(1, count + 1)
    ]


class TestCheckpointStorage:
    def test_opening_without_context_keeps_the_stored_context(
        self, tmp_path: Path
    ) -> None:
        StageCheckpoint(tmp_path, "k", "ngữ cảnh A").save_stage(LITERAL, _lines())
        StageCheckpoint(tmp_path, "k").edit_stage(LITERAL, {1: "sửa"})
        reopened = StageCheckpoint(tmp_path, "k", "ngữ cảnh B")
        assert reopened.get_stage(LITERAL) is None
        assert reopened.stage_edits(LITERAL)[1].text == "sửa"

    def test_edit_requires_a_stored_stage(self, tmp_path: Path) -> None:
        with pytest.raises(StageNotCachedError):
            StageCheckpoint(tmp_path, "k").edit_stage(STYLE, {1: "x"})

    def test_file_is_written_atomically_and_has_format_and_meta(
        self, tmp_path: Path
    ) -> None:
        moment = datetime(2026, 10, 13, 1, 2, 3, tzinfo=UTC)
        checkpoint = StageCheckpoint(
            tmp_path, "k", meta={"lines": 3}, clock=lambda: moment
        )
        checkpoint.save_stage(LITERAL, _lines(), input_lines=_lines())
        raw = json.loads((tmp_path / "k.json").read_text(encoding="utf-8"))
        assert raw["format"] == 2
        assert raw["meta"] == {"lines": 3, "updated_at": "2026-10-13T01:02:03+00:00"}
        assert raw["stage_times"] == {"literal": "2026-10-13T01:02:03+00:00"}
        assert set(raw["stage_inputs"]["literal"]) == {"1", "2", "3"}
        assert not list(tmp_path.glob("*.tmp"))

    def test_corrupted_file_starts_fresh(self, tmp_path: Path) -> None:
        (tmp_path / "k.json").write_text("[1, 2", encoding="utf-8")
        checkpoint = StageCheckpoint(tmp_path, "k")
        assert checkpoint.is_empty
        (tmp_path / "k.json").write_text("[1, 2]", encoding="utf-8")
        assert StageCheckpoint(tmp_path, "k").is_empty

    def test_run_info_round_trip(self, tmp_path: Path) -> None:
        run = StageRun("in", "cfg", STRONG)
        StageCheckpoint(tmp_path, "k").save_stage(LITERAL, _lines(), run=run)
        assert StageCheckpoint(tmp_path, "k").stored_run(LITERAL) == run

    def test_identity_key_matches_the_formula_of_v323606(self) -> None:
        """Tắt Visual Cues thì khoá không đổi so với v3.23.606-607 (file cũ vẫn khớp)."""
        import hashlib

        identity = CheckpointIdentity(texts=("a", "b"), target_lang="Vietnamese")
        payload = json.dumps(
            {"texts": ["a", "b"], "target": "Vietnamese", "source": "",
             "visual_cues": False},
            ensure_ascii=False, sort_keys=True,
        )
        assert identity.key == hashlib.sha256(payload.encode()).hexdigest()[:20]


class TestPrune:
    def _make(self, directory: Path, key: str, age_days: float, now: float) -> None:
        path = directory / f"{key}.json"
        path.write_text("{}", encoding="utf-8")
        stamp = now - age_days * 86_400
        os.utime(path, (stamp, stamp))

    def test_keeps_newest_files_within_age(self, tmp_path: Path) -> None:
        now = time.time()
        for position, key in enumerate(("a", "b", "c", "d")):
            self._make(tmp_path, key, position, now)
        self._make(tmp_path, "old", 400, now)
        (tmp_path / "c.partial.json").write_text("{}", encoding="utf-8")
        orphan = tmp_path / "x.partial.json"
        orphan.write_text("{}", encoding="utf-8")
        os.utime(orphan, (now - 500 * 86_400,) * 2)
        removed = prune_stage_checkpoints(
            tmp_path, max_files=2, max_age_days=180, keep={"d"},
            now=datetime.fromtimestamp(now, UTC),
        )
        assert sorted(removed) == ["c", "old", "x"]
        assert sorted(path.name for path in tmp_path.iterdir()) == [
            "a.json", "b.json", "d.json",
        ]

    def test_missing_directory_is_fine(self, tmp_path: Path) -> None:
        assert prune_stage_checkpoints(
            tmp_path / "không có", max_files=1, max_age_days=1
        ) == []

    def test_use_case_prunes_old_caches(self, tmp_path: Path) -> None:
        now = time.time()
        self._make(tmp_path, "rất-cũ", 1000, now)
        _use_case(tmp_path, _Batches()).execute(_request())
        assert not (tmp_path / "rất-cũ.json").exists()


def test_settings_expose_cache_limits() -> None:
    from subtitles_extractor.infrastructure.settings.application_settings import (
        TranslationSettings,
    )

    settings = TranslationSettings()
    assert settings.stage_cache_max_files >= 1
    assert settings.stage_cache_max_age_days >= 1
    assert timedelta(days=settings.stage_cache_max_age_days) >= timedelta(days=30)
