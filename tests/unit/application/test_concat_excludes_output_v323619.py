"""[v3.23.619] Nối cả bộ lần 2 không được lấy tệp trọn bộ của lần trước làm đầu vào.

Bấm thử giao diện: trang Xuất bản nối "phim_tron_bo.mkv, tap01, tap02" — tệp trọn bộ cũ
vừa là đầu vào vừa là đầu ra (cùng tên) → ffmpeg lỗi mã 234; "Nén lại" cũng lỗi y hệt.
Trang Trích xuất có loại trừ, trang Xuất bản thì không.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from subtitles_extractor.application.services.concat_plan import (
    default_concat_output,
    find_concat_videos,
)


def _folder(tmp_path: Path) -> Path:
    folder = tmp_path / "phim"
    folder.mkdir()
    for name in ("tap01.mp4", "tap02.mp4", "phim_tron_bo.mkv", "khac_tron_bo.mp4"):
        (folder / name).write_bytes(b"x")
    return folder


def test_find_concat_videos_never_returns_full_series_outputs(tmp_path: Path) -> None:
    folder = _folder(tmp_path)
    assert [p.name for p in find_concat_videos(folder)] == ["tap01.mp4", "tap02.mp4"]
    assert default_concat_output(folder, []).name == "phim_tron_bo.mkv"


def test_adapter_refuses_output_that_is_also_an_input(tmp_path: Path) -> None:
    from subtitles_extractor.infrastructure.video.ffmpeg_concat_adapter import (
        VideoConcatError,
        concatenate_videos,
    )

    folder = _folder(tmp_path)
    same = folder / "tap01.mp4"
    with pytest.raises(VideoConcatError, match="trùng"):
        concatenate_videos([same, folder / "tap02.mp4"], same)


def test_publish_page_concat_excludes_previous_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

    from subtitles_extractor.presentation.pages.publish_page import PublishPage

    QApplication.instance() or QApplication([])
    folder = _folder(tmp_path)
    page = PublishPage(_container(tmp_path))
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *_a, **_k: str(folder))
    monkeypatch.setattr(
        QMessageBox, "question", lambda *_a, **_k: QMessageBox.StandardButton.Yes
    )
    started: list[Any] = []
    monkeypatch.setattr(page, "_start_concat", lambda videos, out, **_k: started.append((videos, out)))
    page._on_concat_series_clicked()
    videos, output = started[0]
    assert [v.name for v in videos] == ["tap01.mp4", "tap02.mp4"]
    assert output.name == "phim_tron_bo.mkv"
    page.close()


def _container(tmp: Path) -> Any:
    from subtitles_extractor.composition.container import ApplicationContainer
    from subtitles_extractor.infrastructure.settings.json_file_repository import (
        JsonFileRepository,
    )
    from subtitles_extractor.presentation.pages import editor_page

    return ApplicationContainer(
        settings_repository=JsonFileRepository(file_path=tmp / "c.json"),
        i18n_data_dir=Path(editor_page.__file__).resolve().parents[2] / "data",
        user_data_dir=tmp,
    )
