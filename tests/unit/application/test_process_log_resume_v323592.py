"""[v3.23.592] Hồi quy: chạy tiếp từ checkpoint làm MẤT nhật ký của các giai đoạn đã xong.

Lượt dịch thật v3.23.591: giai đoạn 4 hỏng vì 503 lúc 09:26, người dùng chạy tiếp lúc
10:20. Nhật ký ứng dụng ghi rõ hai giai đoạn đầu đã làm gì —

    09:01:27 Giai đoạn 2: … chuẩn hoá con số ở 1 dòng: 747 (sai giá trị: hai mươi vạn →
             hai vạn)
    09:10:44 Giai đoạn 3: … dải 68-77 bị VIẾT LẠI khác hẳn … → lùi về bản trước.

— nhưng file chuẩn đoán dịch xuất sau lần chạy tiếp chỉ có sự kiện của giai đoạn 4
(``stage_completed`` đúng một mốc, không ``numbers_normalised``, không ``stage_rewrite``):
nhật ký diễn biến bị xoá ở đầu mỗi lần chạy, mà giai đoạn được bỏ qua thì không sinh lại
sự kiện nào.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from subtitles_extractor.application.services.translation_process_log import (
    TranslationProcessLog,
    get_process_log,
)
from subtitles_extractor.application.use_cases.translate_subtitles import (
    TranslateSubtitlesUseCase,
    _compute_checkpoint_key,
    _StageCheckpoint,
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

#: Hai sự kiện thật của lượt chạy v3.23.591 (chép từ nhật ký ứng dụng, dạng ghi của bản
#: v3.23.590 cho cùng loại sự kiện).
NUMBERS_EVENT: dict[str, Any] = {
    "event": "numbers_normalised",
    "label": "Giai đoạn 2: Dịch thô sát nghĩa",
    "fixes": [
        {
            "index": 747,
            "reason": "sai giá trị",
            "before": "hai mươi vạn",
            "after": "hai vạn",
        }
    ],
}
REWRITE_EVENT: dict[str, Any] = {
    "event": "stage_rewrite",
    "label": "Giai đoạn 3: Tinh chỉnh phong cách",
    "from_index": 68,
    "to_index": 77,
    "offset": 0,
    "verdict": "previous",
}


def _one_line() -> list[Any]:
    return [make_line(1, "một", "一")]


class TestProcessLogMarks:
    def test_events_since_returns_only_the_newer_events(self) -> None:
        log = TranslationProcessLog()
        log.record("a", n=1)
        mark = log.count()
        log.record("b", n=2)
        log.record("c", n=3)
        assert mark == 1
        assert log.events_since(mark) == [{"event": "b", "n": 2}, {"event": "c", "n": 3}]

    def test_events_since_gives_copies(self) -> None:
        log = TranslationProcessLog()
        log.record("a", n=1)
        log.events_since(0)[0]["n"] = 99
        assert log.snapshot()["events"] == [{"event": "a", "n": 1}]

    def test_mark_beyond_the_end_gives_nothing(self) -> None:
        log = TranslationProcessLog()
        log.record("a")
        assert log.events_since(5) == []
        assert log.events_since(-3) == [{"event": "a"}]

    def test_replay_marks_where_the_events_came_from(self) -> None:
        log = TranslationProcessLog()
        log.replay([NUMBERS_EVENT, REWRITE_EVENT])
        assert log.snapshot()["events"] == [
            {**NUMBERS_EVENT, "from_checkpoint": True},
            {**REWRITE_EVENT, "from_checkpoint": True},
        ]
        assert log.snapshot()["counts"] == {"numbers_normalised": 1, "stage_rewrite": 1}

    def test_replay_skips_junk(self) -> None:
        log = TranslationProcessLog()
        log.replay([{"no_event": 1}, "chuỗi", {"event": ""}, {"event": "ok"}])  # type: ignore[list-item]
        assert log.snapshot()["events"] == [{"event": "ok", "from_checkpoint": True}]

    def test_replaying_twice_keeps_a_single_flag(self) -> None:
        log = TranslationProcessLog()
        log.replay([{**NUMBERS_EVENT, "from_checkpoint": True}])
        assert log.snapshot()["events"] == [{**NUMBERS_EVENT, "from_checkpoint": True}]

    def test_replay_respects_the_event_cap(self) -> None:
        log = TranslationProcessLog(max_events=1)
        log.replay([NUMBERS_EVENT, REWRITE_EVENT])
        snapshot = log.snapshot()
        assert len(snapshot["events"]) == 1
        assert snapshot["dropped"] == 1


class TestCheckpointKeepsStageEvents:
    def test_events_survive_a_reload(self, tmp_path: Path) -> None:
        checkpoint = _StageCheckpoint(tmp_path, "k")
        checkpoint.save_stage(LITERAL, _one_line(), events=[NUMBERS_EVENT])
        checkpoint.save_stage(STYLE, _one_line(), events=[REWRITE_EVENT])
        reloaded = _StageCheckpoint(tmp_path, "k")
        assert reloaded.get_stage_events(LITERAL) == [NUMBERS_EVENT]
        assert reloaded.get_stage_events(STYLE) == [REWRITE_EVENT]

    def test_saving_without_events_keeps_what_was_stored(self, tmp_path: Path) -> None:
        checkpoint = _StageCheckpoint(tmp_path, "k")
        checkpoint.save_stage(LITERAL, _one_line(), events=[NUMBERS_EVENT])
        checkpoint.save_stage(LITERAL, _one_line())
        reloaded = _StageCheckpoint(tmp_path, "k")
        assert reloaded.get_stage_events(LITERAL) == [NUMBERS_EVENT]

    def test_file_written_by_older_versions_has_no_events(self, tmp_path: Path) -> None:
        (tmp_path / "k.json").write_text(
            json.dumps({"key": "k", "stages": {}}), encoding="utf-8"
        )
        assert _StageCheckpoint(tmp_path, "k").get_stage_events(LITERAL) == []

    def test_malformed_events_are_ignored(self, tmp_path: Path) -> None:
        (tmp_path / "k.json").write_text(
            json.dumps({"key": "k", "stages": {}, "events": ["hỏng"]}), encoding="utf-8"
        )
        assert _StageCheckpoint(tmp_path, "k").get_stage_events(LITERAL) == []
        (tmp_path / "k.json").write_text(
            json.dumps(
                {"key": "k", "stages": {}, "events": {"literal": ["x", {"event": "a"}]}}
            ),
            encoding="utf-8",
        )
        reloaded = _StageCheckpoint(tmp_path, "k")
        assert reloaded.get_stage_events(LITERAL) == [{"event": "a"}]

    def test_changed_context_drops_the_events_too(self, tmp_path: Path) -> None:
        old = _StageCheckpoint(tmp_path, "k", "ngữ cảnh cũ")
        old.save_stage(LITERAL, _one_line(), events=[NUMBERS_EVENT])
        new = _StageCheckpoint(tmp_path, "k", "ngữ cảnh mới")
        assert new.get_stage(LITERAL) is None
        assert new.get_stage_events(LITERAL) == []


def _stage_events(name: str) -> list[dict[str, Any]]:
    return [e for e in get_process_log().snapshot()["events"] if e["event"] == name]


class TestResumedRunKeepsTheWholeStory:
    def test_real_case_second_run_still_shows_the_first_stage(
        self, tmp_path: Path
    ) -> None:
        """Đứt ở giai đoạn 2 (như lượt thật đứt ở giai đoạn 4) → chạy tiếp: file chuẩn
        đoán phải có mốc của CẢ hai giai đoạn. Trước đây chỉ còn giai đoạn chạy lại."""
        batches = FlakyBatches(fail_at=7)  # 5 lô dịch thô + 2 lô tinh chỉnh, rồi đứt
        use_case = TranslateSubtitlesUseCase(
            translator=build_adapter(batches), checkpoint_dir=tmp_path
        )
        with pytest.raises(SubtitleTranslationError):
            use_case.execute(build_request())
        use_case.execute(build_request())

        completed = _stage_events("stage_completed")
        assert [event["stage"] for event in completed] == ["literal", "style"]
        assert completed[0]["from_checkpoint"] is True
        assert completed[0]["lines"] == 10
        assert "from_checkpoint" not in completed[1]

    def test_events_recorded_during_the_first_stage_are_replayed(
        self, tmp_path: Path
    ) -> None:
        batches = FlakyBatches(fail_at=7)
        use_case = TranslateSubtitlesUseCase(
            translator=build_adapter(batches), checkpoint_dir=tmp_path
        )
        with pytest.raises(SubtitleTranslationError):
            use_case.execute(build_request())
        # Lần chạy đầu đã lưu nhật ký của giai đoạn dịch thô vào checkpoint.
        key = _compute_checkpoint_key(build_request())
        stored = json.loads((tmp_path / f"{key}.json").read_text(encoding="utf-8"))
        assert [e["event"] for e in stored["events"]["literal"]][-1] == "stage_completed"
        assert "style" not in stored["events"]

    def test_uninterrupted_run_has_no_replayed_events(self, tmp_path: Path) -> None:
        use_case = TranslateSubtitlesUseCase(
            translator=build_adapter(FlakyBatches()), checkpoint_dir=tmp_path
        )
        use_case.execute(build_request())
        events = get_process_log().snapshot()["events"]
        assert [e["stage"] for e in _stage_events("stage_completed")] == [
            "literal", "style"
        ]
        assert not any(event.get("from_checkpoint") for event in events)

    def test_stage_events_are_not_mixed_between_stages(self, tmp_path: Path) -> None:
        """Mỗi giai đoạn lưu ĐÚNG phần của mình: phát lại không nhân đôi sự kiện."""
        batches = FlakyBatches(fail_at=7)
        use_case = TranslateSubtitlesUseCase(
            translator=build_adapter(batches), checkpoint_dir=tmp_path
        )
        with pytest.raises(SubtitleTranslationError):
            use_case.execute(build_request())
        use_case.execute(build_request())
        assert len(_stage_events("stage_completed")) == 2

    def test_without_checkpoint_dir_nothing_changes(self) -> None:
        use_case = TranslateSubtitlesUseCase(translator=build_adapter(FlakyBatches()))
        use_case.execute(build_request())
        assert len(_stage_events("stage_completed")) == 2
