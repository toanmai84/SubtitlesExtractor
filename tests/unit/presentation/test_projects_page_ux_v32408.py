"""[v3.23.108] Test UX trang Thư viện: empty-state + bật/tắt nút theo lựa chọn."""

from __future__ import annotations

import pytest

from subtitles_extractor.domain.entities.project_record import ProjectRecord
from subtitles_extractor.presentation.pages.projects_page import ProjectsPage


class _FakeRepo:
    def __init__(self, records: list[ProjectRecord]) -> None:
        self._records = records
        self.deleted: list[str] = []

    def list_all(self) -> list[ProjectRecord]:
        return list(self._records)

    def delete(self, video_hash: str) -> None:
        self.deleted.append(video_hash)


def _rec(name: str, h: str) -> ProjectRecord:
    from subtitles_extractor.domain.entities.project_record import WorkflowStage

    return ProjectRecord(
        video_hash=h, video_path=f"/x/{name}", video_name=name,
        stage=WorkflowStage.EXTRACTED, target_lang="vi", updated_at="2026-06-24T10:00:00",
    )


@pytest.fixture
def _app():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    return app


def test_empty_state_shown_when_no_projects(_app) -> None:
    page = ProjectsPage(_FakeRepo([]))
    assert page._empty_label.isVisibleTo(page)
    assert not page._table.isVisibleTo(page)
    # Không có lựa chọn -> nút Mở/Xoá tắt
    assert not page._open_btn.isEnabled()
    assert not page._delete_btn.isEnabled()


def test_table_shown_and_buttons_follow_selection(_app) -> None:
    page = ProjectsPage(_FakeRepo([_rec("phim1", "h1"), _rec("phim2", "h2")]))
    assert page._table.isVisibleTo(page)
    assert not page._empty_label.isVisibleTo(page)
    # Chưa chọn -> nút tắt
    assert not page._open_btn.isEnabled()
    # Chọn dòng 0 -> nút bật
    page._table.selectRow(0)
    assert page._open_btn.isEnabled()
    assert page._delete_btn.isEnabled()
    assert page._selected_record().video_hash == "h1"
