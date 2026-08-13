"""Unit test cho nền tảng liên thông dự án Auto-Dubbing."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from subtitles_extractor.domain.entities.project_record import (
    ProjectRecord,
    WorkflowStage,
)
from subtitles_extractor.domain.value_objects.output_naming import (
    SubtitleFormat,
    extracted_subtitle_path,
    translated_subtitle_path,
    tts_audio_path,
    tts_subtitle_path,
)
from subtitles_extractor.infrastructure.database.sqlite_project_repository import (
    SqliteProjectRepository,
)
from subtitles_extractor.infrastructure.video.video_hasher import compute_video_hash


def test_video_hash_independent_of_name_and_dir() -> None:
    with tempfile.TemporaryDirectory() as d:
        content = os.urandom(4_000_000)
        a = Path(d) / "movie_A.mp4"
        sub = Path(d) / "nested"
        sub.mkdir()
        b = sub / "renamed_B.mkv"
        a.write_bytes(content)
        b.write_bytes(content)
        assert compute_video_hash(a) == compute_video_hash(b)


def test_video_hash_differs_for_different_content() -> None:
    with tempfile.TemporaryDirectory() as d:
        a = Path(d) / "a.mp4"
        b = Path(d) / "b.mp4"
        a.write_bytes(os.urandom(4_000_000))
        b.write_bytes(os.urandom(4_000_000))
        assert compute_video_hash(a) != compute_video_hash(b)


def test_output_naming_conventions() -> None:
    video = "/movies/show (S01).mkv"
    assert extracted_subtitle_path(video).name == "show (S01).original.srt"
    assert extracted_subtitle_path(video, SubtitleFormat.ASS).name == "show (S01).original.ass"
    assert translated_subtitle_path(video, "vi-VN").name == "show (S01).translate.vi.srt"
    assert translated_subtitle_path(video, "en", SubtitleFormat.ASS).name == "show (S01).translate.en.ass"
    assert tts_subtitle_path(video, "vi-VN").name == "show (S01).tts.vi.srt"
    assert tts_subtitle_path(video).name == "show (S01).tts.srt"  # không lang → tương thích cũ
    assert tts_audio_path(video).name == "show (S01).wav"


def test_subtitle_format_from_str() -> None:
    assert SubtitleFormat.from_str("ass") is SubtitleFormat.ASS
    assert SubtitleFormat.from_str("ASS") is SubtitleFormat.ASS
    assert SubtitleFormat.from_str("srt") is SubtitleFormat.SRT
    assert SubtitleFormat.from_str(None) is SubtitleFormat.SRT


def test_project_repository_crud_and_upsert() -> None:
    with tempfile.TemporaryDirectory() as d:
        repo = SqliteProjectRepository(Path(d) / "app.db")
        try:
            rec = ProjectRecord(
                video_hash="hash1",
                video_name="phim.mkv",
                original_subtitle="sub-goc",
                stage=WorkflowStage.EXTRACTED,
            )
            repo.save(rec)
            got = repo.get("hash1")
            assert got is not None
            assert got.original_subtitle == "sub-goc"
            assert got.stage is WorkflowStage.EXTRACTED

            # Cập nhật khâu dịch — giữ phụ đề gốc, nâng stage.
            got.translated_subtitle = "ban-dich"
            got.target_lang = "vi"
            got.stage = WorkflowStage.TRANSLATED
            repo.save(got)
            after = repo.get("hash1")
            assert after.original_subtitle == "sub-goc"
            assert after.translated_subtitle == "ban-dich"
            assert after.stage is WorkflowStage.TRANSLATED

            assert len(repo.list_all()) == 1
            repo.delete("hash1")
            assert repo.get("hash1") is None
        finally:
            repo.close()


def test_project_stage_never_decreases_on_save() -> None:
    with tempfile.TemporaryDirectory() as d:
        repo = SqliteProjectRepository(Path(d) / "app.db")
        try:
            repo.save(ProjectRecord(video_hash="h", stage=WorkflowStage.TRANSLATED))
            # Lưu lại với stage thấp hơn → DB giữ MAX (không thụt lùi tiến độ).
            repo.save(ProjectRecord(video_hash="h", stage=WorkflowStage.EXTRACTED))
            assert repo.get("h").stage is WorkflowStage.TRANSLATED
        finally:
            repo.close()
