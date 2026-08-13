"""Test [v3.23.18] nối UI phiên dịch: lưu cloud_files khi analyze, xoá cloud theo yêu cầu."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from subtitles_extractor.presentation.view_models.translate_page_view_model import (
    TranslatePageViewModel,
)
from subtitles_extractor.infrastructure.database.sqlite_translation_session_store import (
    CloudVideoFile,
)
from subtitles_extractor.infrastructure.translation.gemini_video_context import (
    RemoteVideoRef,
)


@pytest.fixture(scope="module")
def _app():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    return app


@pytest.fixture()
def vm(_app, tmp_path):
    from subtitles_extractor.composition.bootstrap import bootstrap_for_gui
    container = bootstrap_for_gui()
    container._user_data_dir = tmp_path
    container._translation_session_store = None
    return TranslatePageViewModel(container), container


class TestCloudFilesSession:
    def test_video_refs_ready_saves_cloud_files(self, vm) -> None:
        view_model, container = vm
        vp = "/fake/movie.mkv"
        view_model._pending_analysis_video_path = vp
        view_model._on_video_refs_ready([
            RemoteVideoRef(0, "files/a", 0, 100, "ACTIVE"),
            RemoteVideoRef(1, "files/b", 100, 200, "ACTIVE"),
        ])
        session = container.translation_session_store.get(vp)
        assert [cf.remote_name for cf in session.cloud_files] == ["files/a", "files/b"]

    def test_no_video_path_no_save(self, vm) -> None:
        view_model, container = vm
        view_model._pending_analysis_video_path = None
        view_model._on_video_refs_ready([RemoteVideoRef(0, "files/x", 0, 1, "ACTIVE")])
        # Không có video_path → không lưu (không lỗi).

    def test_delete_cloud_files(self, vm) -> None:
        view_model, container = vm
        vp = "/fake/movie2.mkv"
        container.translation_session_store.save_cloud_files(
            vp, [CloudVideoFile("files/a", 0, 100), CloudVideoFile("files/b", 100, 200)]
        )
        fake_provider = MagicMock()
        fake_provider.delete_remote_files.return_value = {"files/a": True, "files/b": True}
        with patch.object(container, "make_video_context_provider", lambda k: fake_provider):
            deleted = view_model.delete_cloud_files_for_video(vp, "key")
        assert deleted == 2
        # Phiên đã clear cloud files.
        assert len(container.translation_session_store.get(vp).cloud_files) == 0

    def test_delete_no_cloud_files(self, vm) -> None:
        view_model, container = vm
        deleted = view_model.delete_cloud_files_for_video("/fake/none.mkv", "key")
        assert deleted == 0

    def test_delete_empty_path(self, vm) -> None:
        view_model, container = vm
        assert view_model.delete_cloud_files_for_video("", "key") == 0
