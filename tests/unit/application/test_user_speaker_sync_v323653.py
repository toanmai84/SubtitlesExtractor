"""[v3.23.653] Người nói chọn đúng ở trang TTS → Visual Cues trang Dịch; lần dịch sau chỉ dịch lại lô có dòng đổi.

Trước đây sửa người nói ở trang TTS chỉ dùng cho giọng đọc; Visual Cues (đầu vào xưng hô của khâu dịch) giữ người nói
sai, và kể cả khi sửa cues bằng tay, bộ nhớ đệm giai đoạn chỉ băm chữ nên bấm Dịch vẫn dùng lại bản dịch cũ.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from subtitles_extractor.application.services.analysis_audit import apply_audit_patch
from subtitles_extractor.application.services.stage_checkpoint import (
    line_input_digest,
    stage_input_digests,
    stage_input_fingerprint,
)
from subtitles_extractor.application.services.visual_cue_serializer import (
    apply_user_speakers,
    deserialize_visual_cues,
    keep_user_cues_json,
    serialize_visual_cues,
    tag_user_cues,
)
from subtitles_extractor.application.use_cases.translate_subtitles import TranslateSubtitlesUseCase
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    CUE_ORIGIN_USER,
    SubtitleContextAnalysis,
    VisualCue,
)
from tests.unit.application.stage_resume_harness import (
    LITERAL,
    FlakyBatches,
    build_adapter,
    build_request,
    make_line,
)

AI_CUES = '[{"id":1,"spk":"A","to":"B","cue":""},{"id":3,"spk":"B","to":"A","cue":"giận"}]'


def test_cue_json_without_user_origin_is_unchanged() -> None:
    assert serialize_visual_cues(deserialize_visual_cues(AI_CUES)) == AI_CUES


def test_apply_user_speakers_marks_origin_and_keeps_scene() -> None:
    raw, changed = apply_user_speakers(AI_CUES, {3: "A", 5: "C", 7: ""})
    assert changed == [3, 5]
    cues = {cue.line_no: cue for cue in deserialize_visual_cues(raw)}
    # [v3.23.655] kèm nhãn AI gốc (B → A) để "chọn lại Tự động" trả về được.
    assert cues[3] == VisualCue(3, speaker="A", addressee="", scene="giận", origin=CUE_ORIGIN_USER, auto_label="B\x1fA")
    assert cues[5].speaker == "C" and cues[5].origin == CUE_ORIGIN_USER
    assert cues[1].origin == ""
    assert apply_user_speakers(raw, {3: "A"}) == (raw, [])  # đã đồng bộ: không đổi gì


def test_reanalysis_and_audit_keep_user_speakers() -> None:
    user_raw, _ = apply_user_speakers(AI_CUES, {3: "A"})
    fresh_ai = '[{"id":1,"spk":"A","to":"B"},{"id":3,"spk":"B","to":"A"}]'
    merged = {c.line_no: c for c in deserialize_visual_cues(keep_user_cues_json(fresh_ai, user_raw))}
    assert merged[3].speaker == "A" and merged[3].origin == CUE_ORIGIN_USER
    assert keep_user_cues_json(fresh_ai, AI_CUES) == fresh_ai

    analysis = SubtitleContextAnalysis(source_lang="zh", characters="", overview="", visual_cues=user_raw)
    patched, _changes = apply_audit_patch(
        analysis, {"cue_fixes": [{"id": 3, "spk": "B"}, {"id": 1, "spk": "C"}]}, valid_line_ids={1, 3},
    )
    cues = {c.line_no: c for c in deserialize_visual_cues(patched.visual_cues)}
    assert cues[3].speaker == "A"      # cue người dùng: AI không được vá đè
    assert cues[1].speaker == "C"      # cue AI: vẫn vá như trước


def test_fingerprint_unchanged_for_lines_without_user_cue() -> None:
    lines = [make_line(1, "句1"), make_line(2, "句2")]
    assert stage_input_digests(lines) == {1: line_input_digest("句1"), 2: line_input_digest("句2")}
    tagged = tag_user_cues(lines, apply_user_speakers("", {2: "A"})[0])
    assert stage_input_digests(tagged)[1] == line_input_digest("句1")
    assert stage_input_digests(tagged)[2] != line_input_digest("句2")
    assert stage_input_fingerprint(LITERAL, tagged) != stage_input_fingerprint(LITERAL, lines)


def test_rerun_after_speaker_sync_translates_only_that_batch(tmp_path: Path) -> None:
    batches = FlakyBatches()
    use_case = TranslateSubtitlesUseCase(translator=build_adapter(batches), checkpoint_dir=tmp_path)
    request = build_request(10)
    use_case.execute(request)
    first_calls = len(batches.calls)
    assert first_calls == 10  # 5 lô × 2 giai đoạn

    batches.calls.clear()
    use_case.execute(request)
    assert batches.calls == []  # không đổi gì: dùng lại nguyên bộ nhớ đệm

    cues, _ = apply_user_speakers("", {7: "Lâm Dật"})
    synced = replace(request, context=replace(request.context, visual_cues=cues))
    use_case.execute(synced)
    # Dòng 7 nằm ở lô [7, 8] (lô 2 dòng): chỉ lô đó được dịch lại ở mỗi giai đoạn.
    assert len(batches.calls) == 2, batches.calls


def test_stage_manager_predicts_speaker_changed_lines_v323656(tmp_path: Path) -> None:
    """[v3.23.656] B8: màn "Quản lý giai đoạn" phải dự đoán đúng lượt dịch tới sau khi đổi người nói (như lượt thật)."""
    from subtitles_extractor.application.services.stage_cache import StageCacheService
    from subtitles_extractor.application.use_cases.translate_subtitles import build_stage_cache_scope

    batches = FlakyBatches()
    use_case = TranslateSubtitlesUseCase(translator=build_adapter(batches), checkpoint_dir=tmp_path)
    request = build_request(10)
    use_case.execute(request)
    service = StageCacheService()
    before = service.overview(build_stage_cache_scope(request, tmp_path))
    assert all(e.reusable_lines == e.total_lines for e in before.entries if e.enabled)

    cues, _ = apply_user_speakers("", {7: "Lâm Dật"})
    synced = replace(request, context=replace(request.context, visual_cues=cues))
    after = service.overview(build_stage_cache_scope(synced, tmp_path))
    from subtitles_extractor.application.services.stage_cache import StageAction

    first = next(e for e in after.entries if e.enabled)
    # Dòng 7 đổi người nói: dự đoán "dịch lại một phần" (đếm theo dòng: 9/10 dùng lại; lượt thật dịch lại lô [7, 8]).
    assert first.action is StageAction.PARTIAL and first.reusable_lines == first.total_lines - 1, first
