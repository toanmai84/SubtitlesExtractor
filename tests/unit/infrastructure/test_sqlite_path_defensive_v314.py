"""Unit test cho #5 — chống sập SQLite vì kiểu Path (v3.14.4)."""

from __future__ import annotations

from pathlib import Path

import pytest

from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.entities.video_state import VideoState
from subtitles_extractor.domain.value_objects.confidence import Confidence
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval
from subtitles_extractor.infrastructure.database.sqlite_subtitle_repository import (
    SqliteSubtitleRepository,
)
from subtitles_extractor.infrastructure.database.sqlite_video_state_repository import (
    SqliteVideoStateRepository,
)


class TestVideoStatePathDefensive:
    def test_save_get_with_windows_path_object(self, tmp_path: Path) -> None:
        repo = SqliteVideoStateRepository(tmp_path / "vs.db")
        video = tmp_path / "phim 玄幻.mp4"  # Path object + ký tự CJK
        repo.save(VideoState(video_path=video, roi=None))  # type: ignore[arg-type]
        # get bằng Path object cũng phải hoạt động (defensive str()).
        loaded = repo.get(video)  # type: ignore[arg-type]
        assert loaded is not None
        assert loaded.video_path == str(video)


class TestSubtitleRepoPathDefensive:
    def test_save_load_with_path_object(self, tmp_path: Path) -> None:
        repo = SqliteSubtitleRepository(tmp_path / "sub.db")
        video = tmp_path / "movie.mkv"
        events = [SubtitleEvent(1, "Xin chào", TimeInterval(0.0, 1.0), Confidence(1.0))]
        repo.save_events(video, events)  # type: ignore[arg-type]
        assert repo.has_events(video) is True  # type: ignore[arg-type]
        loaded = repo.load_events(video)  # type: ignore[arg-type]
        assert loaded is not None and loaded[0].text == "Xin chào"
