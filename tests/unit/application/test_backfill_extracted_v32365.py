"""[v3.23.365] Kiểm thử service bù tệp .original.srt từ CSDL."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from subtitles_extractor.application.services.backfill_extracted import (
    backfill_original_subtitles,
)
from subtitles_extractor.domain.entities.project_record import WorkflowStage


def _rec(stage: WorkflowStage, text: str, video: str) -> SimpleNamespace:
    return SimpleNamespace(stage=stage, original_subtitle=text, video_path=video)


def test_backfills_only_missing_extracted() -> None:
    existing = {"/v/ep1.mp4", "/v/ep2.mp4", "/v/ep2.original.srt"}
    written: dict[str, str] = {}
    records = [
        _rec(WorkflowStage.EXTRACTED, "srt-1", "/v/ep1.mp4"),   # bù
        _rec(WorkflowStage.TRANSLATED, "srt-2", "/v/ep2.mp4"),  # đã có tệp → bỏ
        _rec(WorkflowStage.NEW, "", "/v/ep3.mp4"),              # chưa trích → bỏ
        _rec(WorkflowStage.EXTRACTED, "srt-4", "/v/gone.mp4"),  # video mất → bỏ
    ]

    result = backfill_original_subtitles(
        records,
        file_exists=lambda p: str(p) in existing,
        write_text=lambda p, t: written.__setitem__(str(p), t),
    )

    assert list(written) == ["/v/ep1.original.srt"]
    assert written["/v/ep1.original.srt"] == "srt-1"
    assert result.skipped_existing == 1
    assert result.skipped_no_content == 1
    assert result.skipped_no_video == 1
    assert [p.name for p in result.written] == ["ep1.original.srt"]


def test_filter_by_video_name() -> None:
    existing = {"/v/ep1.mp4"}
    written: dict[str, str] = {}
    records = [_rec(WorkflowStage.EXTRACTED, "srt-1", "/v/ep1.mp4")]

    # Lọc đúng tên → bù.
    res_match = backfill_original_subtitles(
        records, file_exists=lambda p: str(p) in existing,
        write_text=lambda p, t: written.__setitem__(str(p), t),
        only_video_names={"ep1.mp4"},
    )
    assert len(res_match.written) == 1

    # Lọc tên khác → không bù.
    written.clear()
    res_miss = backfill_original_subtitles(
        records, file_exists=lambda p: str(p) in existing,
        write_text=lambda p, t: written.__setitem__(str(p), t),
        only_video_names={"khac.mp4"},
    )
    assert res_miss.written == []
    assert written == {}


def test_write_failure_counted(monkeypatch) -> None:
    existing = {"/v/ep1.mp4"}
    records = [_rec(WorkflowStage.EXTRACTED, "srt-1", "/v/ep1.mp4")]

    def _boom(_path: Path, _text: str) -> None:
        raise OSError("disk full")

    result = backfill_original_subtitles(
        records, file_exists=lambda p: str(p) in existing, write_text=_boom,
    )
    assert result.failed == 1
    assert result.written == []
