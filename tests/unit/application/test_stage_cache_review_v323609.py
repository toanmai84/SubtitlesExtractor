"""[v3.23.609] Rà lại bộ nhớ đệm giai đoạn (3.23.608).

* Bộ nhớ đệm được dùng lại nguyên (không ghi gì) thì giờ sửa file không đổi → bước dọn
  theo tuổi/số lượng coi phim vừa dịch lại hôm qua là "cũ" và xoá mất.
* Xoá trắng một ô trong trình sửa giai đoạn được lưu thành dòng sửa tay RỖNG → dòng đó
  mất khỏi phụ đề và lồng tiếng, và được "giữ" qua mọi lần dịch lại.
* Bật suy nghĩ, đổi mức suy nghĩ rồi tắt lại: yêu cầu gửi model y hệt trước, nhưng dấu
  vân tay cấu hình vẫn tính mức suy nghĩ → cả giai đoạn bị dịch lại.
"""

from __future__ import annotations

import os
import time
from dataclasses import replace
from pathlib import Path

from subtitles_extractor.application.services.stage_cache import (
    StageAction,
    StageCacheService,
)
from subtitles_extractor.application.services.stage_checkpoint import (
    StageCheckpoint,
    legacy_stage_config_signature,
    prune_stage_checkpoints,
    stage_config_signature,
)
from subtitles_extractor.application.use_cases.translate_subtitles import (
    TranslateSubtitlesRequest,
    TranslateSubtitlesUseCase,
    build_stage_cache_scope,
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
DAY = 86_400


class _Batches:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *, batch, **_kwargs):
        self.calls += 1
        return [replace(line, text=f"dịch {line.index}") for line in batch]


def _use_case(tmp_path: Path, batches: _Batches) -> TranslateSubtitlesUseCase:
    adapter = GeminiSubtitleTranslator(api_key="KEY")
    adapter._ensure_available = lambda: None  # type: ignore[method-assign]
    adapter._translate_single_batch = batches  # type: ignore[method-assign]
    return TranslateSubtitlesUseCase(translator=adapter, checkpoint_dir=tmp_path)


def _request(**stage_over) -> TranslateSubtitlesRequest:
    stage = {"kind": LITERAL, "model_name": "gemini-3.8-flash", "batch_size": 2}
    stage.update(stage_over)
    return TranslateSubtitlesRequest(
        events=[
            SubtitleEvent(index=i, text=f"句{i}", interval=TimeInterval(i, i + 0.9))
            for i in range(1, 5)
        ],
        stages=[TranslationStageConfig(**stage)],
        context=TranslationContext(target_lang="Vietnamese", source_lang="zh"),
    )


def test_cache_used_today_is_not_pruned_as_old(tmp_path: Path) -> None:
    batches = _Batches()
    use_case = _use_case(tmp_path, batches)
    use_case.execute(_request())
    key = build_stage_cache_scope(_request(), tmp_path).key
    old = time.time() - 179 * DAY
    os.utime(tmp_path / f"{key}.json", (old, old))
    use_case.execute(_request())  # dùng lại nguyên, không dịch gì
    assert batches.calls == 2
    removed = prune_stage_checkpoints(
        tmp_path, max_files=60, max_age_days=178, keep=()
    )
    assert removed == []


def test_blank_edit_is_rejected(tmp_path: Path) -> None:
    lines = [
        TranslationLine(index=1, start_ms=0, end_ms=900, text="Xin chào"),
        TranslationLine(index=2, start_ms=1000, end_ms=1900, text="Tạm biệt"),
    ]
    checkpoint = StageCheckpoint(tmp_path, "k")
    checkpoint.save_stage(LITERAL, lines)
    result = checkpoint.edit_stage(LITERAL, {1: "   ", 2: "Hẹn gặp lại"})
    assert result.blank == (1,)
    assert result.changed == (2,)
    assert StageCheckpoint(tmp_path, "k").get_stage(LITERAL)[0].text == "Xin chào"


def test_thinking_settings_do_not_matter_while_thinking_is_off() -> None:
    off = TranslationStageConfig(kind=LITERAL, model_name="m", enable_thinking=False,
                                 thinking_level="low", thinking_budget=-1)
    off_changed = replace(off, thinking_level="high", thinking_budget=2048)
    assert stage_config_signature(off, False) == stage_config_signature(
        off_changed, False
    )
    on = replace(off, enable_thinking=True)
    assert stage_config_signature(on, False) != stage_config_signature(
        replace(on, thinking_level="high"), False
    )
    zero_budget = replace(on, thinking_budget=0)
    assert stage_config_signature(zero_budget, False) == stage_config_signature(
        off, False
    )


def test_cache_saved_by_v323608_is_still_reused(tmp_path: Path) -> None:
    """Dấu vân tay cấu hình kiểu cũ (gồm cả mức suy nghĩ khi đang tắt) vẫn được nhận."""
    request = _request(thinking_level="medium")
    scope = build_stage_cache_scope(request, tmp_path)
    sources = list(scope.source_lines)
    run = scope.run_for(request.stages[0], sources)
    legacy_run = replace(
        run,
        config_signature=legacy_stage_config_signature(request.stages[0], False),
        accepted_configs=(),
    )
    lines = [replace(line, text=f"cũ {line.index}") for line in sources]
    StageCheckpoint(tmp_path, scope.key, scope.context_fingerprint).save_stage(
        LITERAL, lines, run=legacy_run, input_lines=sources
    )
    assert StageCacheService().overview(scope).entry(LITERAL).action is StageAction.REUSE
    batches = _Batches()
    response = _use_case(tmp_path, batches).execute(request)
    assert batches.calls == 0
    assert response.events[0].text == "cũ 1"
