"""[v3.23.168] Test dò file phụ đề rời cùng tên cạnh video (hàm thuần + use case).

Tính năng trang trích xuất: tự nhận file phụ đề đặt cạnh video (Movie.srt, Movie.vi.srt,
Movie.ass) để nạp trực tiếp thay vì OCR. Kiểm luật dò (khớp chính xác/biến thể, ưu tiên
đuôi, suy thẻ ngôn ngữ) và use case điều phối (bỏ file không liên quan, ưu tiên đúng).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from subtitles_extractor.application.use_cases.load_sidecar_subtitles import (
    LoadSidecarSubtitlesUseCase,
    SidecarSubtitleNotFoundError,
)
from subtitles_extractor.infrastructure.subtitle.subtitle_sidecar_finder import (
    find_sidecar_subtitles,
)


def _paths(*names: str) -> list[Path]:
    return [Path("/movies") / name for name in names]


# ── Hàm thuần find_sidecar_subtitles ─────────────────────────────────────


def test_exact_name_match() -> None:
    video = Path("/movies/Movie.mkv")
    result = find_sidecar_subtitles(video, _paths("Movie.srt", "Other.srt"))
    assert [item.path.name for item in result] == ["Movie.srt"]
    assert result[0].language_tag == ""


def test_language_variant_match_and_tag() -> None:
    video = Path("/movies/Movie.mkv")
    result = find_sidecar_subtitles(video, _paths("Movie.vi.srt", "Movie.en.srt"))
    names = [item.path.name for item in result]
    tags = {item.path.name: item.language_tag for item in result}
    assert set(names) == {"Movie.vi.srt", "Movie.en.srt"}
    assert tags["Movie.vi.srt"] == "vi"
    assert tags["Movie.en.srt"] == "en"


def test_exact_prioritized_over_variant() -> None:
    video = Path("/movies/Movie.mkv")
    result = find_sidecar_subtitles(video, _paths("Movie.vi.srt", "Movie.srt"))
    assert result[0].path.name == "Movie.srt"  # khớp chính xác đứng đầu


def test_extension_priority_srt_before_ass() -> None:
    video = Path("/movies/Movie.mkv")
    result = find_sidecar_subtitles(video, _paths("Movie.ass", "Movie.srt"))
    assert result[0].path.name == "Movie.srt"


def test_ignores_unrelated_and_unsupported() -> None:
    video = Path("/movies/Movie.mkv")
    result = find_sidecar_subtitles(
        video, _paths("Movie.txt", "Movie.nfo", "OtherFilm.srt", "Movie.mkv")
    )
    assert result == []


def test_case_insensitive_match() -> None:
    video = Path("/movies/Movie.mkv")
    result = find_sidecar_subtitles(video, _paths("movie.SRT"))
    assert len(result) == 1


def test_ssa_supported() -> None:
    video = Path("/movies/Movie.mkv")
    result = find_sidecar_subtitles(video, _paths("Movie.ssa"))
    assert [item.extension for item in result] == [".ssa"]


# ── Use case điều phối (với importer giả) ────────────────────────────────


class _FakeImportUseCase:
    """Importer giả: trả số câu cố định theo tên file (đủ để kiểm điều phối)."""

    def __init__(self) -> None:
        self.loaded: list[Path] = []

    def execute(self, source_path: Path) -> list[str]:
        self.loaded.append(source_path)
        return ["câu 1", "câu 2", "câu 3"]


def test_use_case_find_lists_candidates(tmp_path) -> None:
    video = tmp_path / "Movie.mkv"
    video.write_bytes(b"x")
    (tmp_path / "Movie.srt").write_text("1\n", encoding="utf-8")
    (tmp_path / "Movie.vi.srt").write_text("1\n", encoding="utf-8")

    use_case = LoadSidecarSubtitlesUseCase(import_use_case=_FakeImportUseCase())
    found = use_case.find(video)
    assert [item.path.name for item in found] == ["Movie.srt", "Movie.vi.srt"]


def test_use_case_find_and_load_best(tmp_path) -> None:
    video = tmp_path / "Movie.mkv"
    video.write_bytes(b"x")
    (tmp_path / "Movie.srt").write_text("1\n", encoding="utf-8")
    fake = _FakeImportUseCase()

    use_case = LoadSidecarSubtitlesUseCase(import_use_case=fake)
    path, events = use_case.find_and_load_best(video)
    assert path.name == "Movie.srt"
    assert len(events) == 3
    assert fake.loaded == [tmp_path / "Movie.srt"]


def test_use_case_raises_when_no_sidecar(tmp_path) -> None:
    video = tmp_path / "Movie.mkv"
    video.write_bytes(b"x")
    use_case = LoadSidecarSubtitlesUseCase(import_use_case=_FakeImportUseCase())
    with pytest.raises(SidecarSubtitleNotFoundError):
        use_case.find_and_load_best(video)


def test_use_case_find_empty_when_dir_missing() -> None:
    use_case = LoadSidecarSubtitlesUseCase(import_use_case=_FakeImportUseCase())
    assert use_case.find(Path("/nonexistent/Movie.mkv")) == []
