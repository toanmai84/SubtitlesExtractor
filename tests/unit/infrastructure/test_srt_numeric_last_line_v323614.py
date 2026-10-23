"""[v3.23.614] Phụ đề CUỐI tệp chỉ gồm chữ số không được biến mất khi đọc SRT.

Bộ đọc bỏ dòng số cuối mỗi khối vì coi nó là số thứ tự của khối kế. Ở khối CUỐI tệp
không còn khối kế nào, nên câu thoại "520" (hay "3" khi đếm ngược) bị bỏ và cả câu mất.
"""

from __future__ import annotations

from subtitles_extractor.infrastructure.subtitle.importers.srt_importer import _parse_srt

SRT = (
    "1\n00:00:01,000 --> 00:00:02,000\nĐếm đi!\n\n"
    "2\n00:00:03,000 --> 00:00:04,000\n3\n\n"
    "3\n00:00:05,000 --> 00:00:06,000\nGiá là\n520\n\n"
    "4\n00:00:07,000 --> 00:00:08,000\n520\n"
)


def test_numeric_lines_are_kept() -> None:
    assert [event.text for event in _parse_srt(SRT)] == [
        "Đếm đi!", "3", "Giá là\n520", "520",
    ]


def test_export_then_import_keeps_numeric_last_line() -> None:
    from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
    from subtitles_extractor.domain.value_objects.time_interval import TimeInterval
    from subtitles_extractor.infrastructure.subtitle.exporters.srt_exporter import (
        SrtExporter,
    )

    events = [
        SubtitleEvent(index=1, text="Một", interval=TimeInterval(1.0, 2.0)),
        SubtitleEvent(index=2, text="2", interval=TimeInterval(3.0, 4.0)),
    ]
    content = SrtExporter._build_content(events)
    assert [event.text for event in _parse_srt(content)] == ["Một", "2"]
