"""[v3.23.650] Block phụ đề có mốc kết thúc trước mốc bắt đầu không được làm cả tệp không nạp được."""

from __future__ import annotations

from subtitles_extractor.infrastructure.subtitle.importers.ass_importer import AssImporter  # noqa: F401
from subtitles_extractor.infrastructure.subtitle.importers.srt_importer import _parse_srt

SRT = """1
00:00:01,000 --> 00:00:02,000
đầu

2
00:00:05,000 --> 00:00:04,000
kết thúc trước bắt đầu

3
00:00:07,000 --> 00:00:08,000
cuối
"""


def test_srt_end_before_start_keeps_line_with_default_duration_v323650() -> None:
    events = list(_parse_srt(SRT))
    assert [e.text for e in events] == ["đầu", "kết thúc trước bắt đầu", "cuối"]
    assert events[1].start_sec == 5.0 and abs(events[1].end_sec - 5.5) < 1e-9


def test_ass_end_before_start_keeps_line_v323650(tmp_path) -> None:
    from subtitles_extractor.infrastructure.subtitle.importers import ass_importer

    ass = (
        "[Script Info]\nScriptType: v4.00+\n\n[V4+ Styles]\nFormat: Name, Fontname\nStyle: Default,Arial\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        "Dialogue: 0,0:00:05.00,0:00:04.00,Default,,0,0,0,,xin chào\n"
    )
    parse = getattr(ass_importer, "_parse_ass", None)
    assert parse is not None
    events = list(parse(ass))
    assert [e.text for e in events] == ["xin chào"]


def test_saved_project_with_bad_interval_loads_empty_instead_of_raising_v323650() -> None:
    import json

    from subtitles_extractor.infrastructure.database.sqlite_subtitle_repository import SqliteSubtitleRepository

    payload = json.dumps([{"index": 1, "text": "x", "start_sec": 5.0, "end_sec": 1.0, "confidence": 1.0}])
    repo = SqliteSubtitleRepository.__new__(SqliteSubtitleRepository)   # chỉ cần phương thức thuần, không mở DB
    assert repo._deserialize_events(payload) == []
