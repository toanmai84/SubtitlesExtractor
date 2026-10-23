"""[v3.23.586] Lượt dịch dở KHÔNG được nối tiếp khi bảng nhân vật/thuật ngữ đã đổi.

Chuẩn đoán thật v3.23.585 (phim "灭门惨案II"): một lần dịch dở để lại 360/1246 dòng dịch
thô, dịch bằng bảng nhân vật do gemini-3.8-flash phân tích. Hôm sau phim được phân tích
lại bằng gemini-3.7-flash (bảng khác: ``當當`` "Đang Đang" → "Đương Đương", ``老表`` "Biểu
ca" → "anh em họ", quy tắc xưng hô khác) rồi dịch tiếp — 360 dòng cũ được dùng lại nên
bản dịch thô mang hai bộ tên và hai bộ quy tắc xưng hô.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from subtitles_extractor.application.services.translation_process_log import (
    get_process_log,
)
from subtitles_extractor.application.use_cases.translate_subtitles import (
    TranslateSubtitlesUseCase,
    _StageCheckpoint,
    context_fingerprint,
    stage_input_fingerprint,
)
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    SubtitleTranslationError,
    TranslationContext,
)
from tests.unit.application.stage_resume_harness import (
    LITERAL,
    STYLE,
    FlakyBatches,
    build_adapter,
    build_request,
    make_line,
)

# Hai dòng bảng nhân vật và hai mục thuật ngữ CHÉP NGUYÊN từ hai lần phân tích thật.
OLD_ROSTER = "余小蓮 / 當當 (Dư Tiểu Liên / Đang Đang) [nữ, ~22] — Nữ công an trẻ tuổi"
NEW_ROSTER = "余小蓮 / 當當 (Dư Tiểu Liên / Đương Đương) [nữ, ~22] — Nữ công an đồn"
OLD_GLOSSARY = "老表 = biểu ca (anh họ/người bà con)"
NEW_GLOSSARY = "老表 = anh em họ (biểu đệ)"

OLD = context_fingerprint(
    TranslationContext(target_lang="vi", characters=OLD_ROSTER, glossary=OLD_GLOSSARY)
)
NEW = context_fingerprint(
    TranslationContext(target_lang="vi", characters=NEW_ROSTER, glossary=NEW_GLOSSARY)
)


def _discard_events() -> list[dict]:
    return [
        event for event in get_process_log().snapshot()["events"]
        if event["event"] == "checkpoint_discarded"
    ]


class TestContextFingerprint:
    def test_depends_on_roster_and_glossary(self) -> None:
        base = TranslationContext(
            target_lang="vi", characters=OLD_ROSTER, glossary=OLD_GLOSSARY
        )
        other_roster = TranslationContext(
            target_lang="vi", characters=NEW_ROSTER, glossary=OLD_GLOSSARY
        )
        other_glossary = TranslationContext(
            target_lang="vi", characters=OLD_ROSTER, glossary=NEW_GLOSSARY
        )
        contexts = (base, other_roster, other_glossary)
        assert len({context_fingerprint(context) for context in contexts}) == 3

    def test_ignores_what_changes_between_resumed_runs(self) -> None:
        """Visual Cues được quét bổ sung giữa hai lần chạy — không làm mất tiến độ."""
        first = TranslationContext(
            target_lang="vi", characters=OLD_ROSTER, glossary=OLD_GLOSSARY,
            visual_cues='[{"id":1,"spk":"A","to":"B"}]', overview="tóm tắt",
        )
        second = TranslationContext(
            target_lang="vi", characters=f"{OLD_ROSTER}\n", glossary=OLD_GLOSSARY,
            visual_cues='[{"id":1,"spk":"A","to":"B"},{"id":2,"spk":"B","to":"A"}]',
            overview="tóm tắt đã sửa",
        )
        assert context_fingerprint(first) == context_fingerprint(second)


class TestCheckpointFiles:
    def test_progress_from_another_context_is_dropped(self, tmp_path: Path) -> None:
        inputs = [make_line(1, "一"), make_line(2, "二")]
        fingerprint = stage_input_fingerprint(LITERAL, inputs)
        _StageCheckpoint(tmp_path, "k", OLD).add_partial(
            LITERAL, fingerprint, [make_line(1, "Đang Đang", "一")]
        )

        reloaded = _StageCheckpoint(tmp_path, "k", NEW)
        assert reloaded.get_partial(LITERAL, fingerprint) is None
        assert not (tmp_path / "k.partial.json").exists()

    def test_progress_from_the_same_context_is_kept(self, tmp_path: Path) -> None:
        fingerprint = stage_input_fingerprint(LITERAL, [make_line(1, "一")])
        _StageCheckpoint(tmp_path, "k", OLD).add_partial(
            LITERAL, fingerprint, [make_line(1, "một", "一")]
        )
        saved = _StageCheckpoint(tmp_path, "k", OLD).get_partial(LITERAL, fingerprint)
        assert [line.text for line in saved] == ["một"]

    def test_finished_stage_from_another_context_is_dropped(self, tmp_path: Path) -> None:
        first = _StageCheckpoint(tmp_path, "k", OLD)
        first.save_stage(LITERAL, [make_line(1, "Biểu ca")])
        assert _StageCheckpoint(tmp_path, "k", OLD).get_stage(LITERAL) is not None

        reloaded = _StageCheckpoint(tmp_path, "k", NEW)
        assert reloaded.get_stage(LITERAL) is None
        assert not (tmp_path / "k.json").exists()

    def test_files_written_by_older_versions_are_still_resumed(
        self, tmp_path: Path
    ) -> None:
        """File của bản ≤ 3.23.585 không ghi dấu vân tay ngữ cảnh — vẫn dùng như trước."""
        fingerprint = stage_input_fingerprint(LITERAL, [make_line(1, "一")])
        legacy = _StageCheckpoint(tmp_path, "k")
        legacy.save_stage(STYLE, [make_line(1, "cũ")])
        legacy.add_partial(LITERAL, fingerprint, [make_line(1, "một", "一")])
        assert "context" not in json.loads((tmp_path / "k.json").read_text("utf-8"))

        reloaded = _StageCheckpoint(tmp_path, "k", NEW)
        assert reloaded.get_stage(STYLE) is not None
        assert reloaded.get_partial(LITERAL, fingerprint) is not None

    def test_reading_without_a_context_never_discards(self, tmp_path: Path) -> None:
        """Nút "Xoá bộ nhớ đệm dịch" mở checkpoint mà không biết ngữ cảnh."""
        _StageCheckpoint(tmp_path, "k", OLD).save_stage(LITERAL, [make_line(1, "x")])
        assert _StageCheckpoint(tmp_path, "k").get_stage(LITERAL) is not None


class TestInterruptedRunAfterReanalysis:
    def _interrupted_run(self, tmp_path: Path) -> TranslateSubtitlesUseCase:
        batches = FlakyBatches(fail_at=3)
        use_case = TranslateSubtitlesUseCase(
            translator=build_adapter(batches), checkpoint_dir=tmp_path
        )
        with pytest.raises(SubtitleTranslationError):
            use_case.execute(
                build_request(characters=OLD_ROSTER, glossary=OLD_GLOSSARY)
            )
        assert batches.calls == [0, 2, 4]  # 3 lô (6 dòng) xong trước khi đứt
        return use_case

    def test_new_roster_retranslates_everything(self, tmp_path: Path) -> None:
        self._interrupted_run(tmp_path)
        batches = FlakyBatches()
        use_case = TranslateSubtitlesUseCase(
            translator=build_adapter(batches), checkpoint_dir=tmp_path
        )
        use_case.execute(build_request(characters=NEW_ROSTER, glossary=NEW_GLOSSARY))
        # Dịch thô chạy lại đủ 5 lô (trước đây chỉ 2 lô còn thiếu), rồi 5 lô tinh chỉnh.
        assert batches.calls == [0, 2, 4, 6, 8, 0, 2, 4, 6, 8]
        [event] = _discard_events()
        assert event["reason"] == "context_changed"
        assert event["partial_lines"] == 6

    def test_same_roster_still_resumes(self, tmp_path: Path) -> None:
        self._interrupted_run(tmp_path)
        batches = FlakyBatches()
        use_case = TranslateSubtitlesUseCase(
            translator=build_adapter(batches), checkpoint_dir=tmp_path
        )
        use_case.execute(build_request(characters=OLD_ROSTER, glossary=OLD_GLOSSARY))
        assert batches.calls == [6, 8, 0, 2, 4, 6, 8]
        assert _discard_events() == []
