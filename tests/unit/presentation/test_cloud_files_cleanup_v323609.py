"""[v3.23.609] Nút "Xoá file cloud": xoá được MỌI đoạn video đã tải, bằng đúng key đã tải.

Trước bản này:

* Đoạn video được tải bằng key nào thì chỉ key đó (cùng dự án) xoá được. Ứng dụng tải
  bằng key còn hạn mức (``set_active_key``) và tải lại khi xoay key, nhưng nút xoá chỉ
  dùng key ĐẦU TIÊN. Gemini trả 403 cho file của key khác, mà 403 lại được coi là "đã
  không còn" → báo "Đã xoá 17/17 file cloud", xoá luôn danh sách, file vẫn nằm trên cloud.
* Chỉ khâu phân tích ghi lại các đoạn đã tải. Đoạn tải khi DỊCH (đính video) hay tải lại
  khi xoay key không được ghi → nút báo "Không có file cloud nào để xoá".
* Mỗi lần phân tích ghi ĐÈ danh sách cũ, nên các đoạn của lần trước bị quên.
* Xoá lỗi mạng giữa chừng vẫn xoá cả danh sách → không thử lại được.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from subtitles_extractor.domain.ports.video_context_port import RemoteVideoRef
from subtitles_extractor.infrastructure.database.sqlite_translation_session_store import (
    CloudVideoFile,
)
from subtitles_extractor.infrastructure.translation.gemini_video_context import (
    GeminiVideoContextProvider,
)
from subtitles_extractor.presentation.workers.video_reupload import (
    make_video_reupload_callback,
)

_DENIED = (
    "403 PERMISSION_DENIED. You do not have permission to access the File {name} or "
    "it may not exist."
)


class _FakeFiles:
    def __init__(self, key: str, owned: set[tuple[str, str]], log: list) -> None:
        self._key = key
        self._owned = owned
        self._log = log

    def delete(self, *, name: str) -> None:
        if (self._key, name) not in self._owned:
            raise RuntimeError(_DENIED.format(name=name))
        self._owned.discard((self._key, name))
        self._log.append((self._key, name))


def _fake_genai(owned: set[tuple[str, str]], log: list) -> type:
    class _Client:
        def __init__(self, *, api_key: str, http_options: Any = None) -> None:
            self.files = _FakeFiles(api_key, owned, log)

    return _Client


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def vm(app: QApplication, tmp_path: Path):
    from subtitles_extractor.composition.bootstrap import bootstrap_for_gui
    from subtitles_extractor.presentation.view_models.translate_page_view_model import (
        TranslatePageViewModel,
    )

    container = bootstrap_for_gui()
    container._user_data_dir = tmp_path
    container._translation_session_store = None
    return TranslatePageViewModel(container), container


class TestDeleteWithTheKeyThatUploaded:
    def test_file_uploaded_with_another_key_is_deleted_with_that_key(
        self, tmp_path: Path
    ) -> None:
        owned = {("KEY_A", "files/a"), ("KEY_B", "files/b")}
        log: list = []
        provider = GeminiVideoContextProvider(
            api_key="KEY_A\nKEY_B", cache_db_path=tmp_path / "c.db", work_dir=tmp_path
        )
        with patch("google.genai.Client", _fake_genai(owned, log)):
            result = provider.delete_remote_files(["files/a", "files/b"])
        assert result == {"files/a": True, "files/b": True}
        assert sorted(log) == [("KEY_A", "files/a"), ("KEY_B", "files/b")]
        assert owned == set()

    def test_original_keys_still_tried_after_switching_the_upload_key(
        self, tmp_path: Path
    ) -> None:
        owned = {("KEY_A", "files/a"), ("KEY_C", "files/c")}
        log: list = []
        provider = GeminiVideoContextProvider(
            api_key="KEY_A, KEY_B", cache_db_path=tmp_path / "c.db", work_dir=tmp_path
        )
        provider.set_active_key("KEY_C")
        with patch("google.genai.Client", _fake_genai(owned, log)):
            provider.delete_remote_files(["files/a", "files/c"])
        assert owned == set()

    def test_file_no_key_can_reach_is_reported_gone(self, tmp_path: Path) -> None:
        """Hết hạn 48 giờ (hoặc của key đã bỏ khỏi danh sách): không gì để xoá thêm."""
        provider = GeminiVideoContextProvider(
            api_key="KEY_A\nKEY_B", cache_db_path=tmp_path / "c.db", work_dir=tmp_path
        )
        with patch("google.genai.Client", _fake_genai(set(), [])):
            assert provider.delete_remote_files(["files/old"]) == {"files/old": True}


class TestEveryUploadIsRecorded:
    def test_reupload_after_key_rotation_reports_the_new_files(self) -> None:
        uploaded = [RemoteVideoRef(0, "files/new", 0.0, 60.0, "ACTIVE")]
        provider = MagicMock()
        provider.upload_chunks_for_ranges.return_value = uploaded
        reported: list = []
        reupload = make_video_reupload_callback(
            provider, "phim.mp4", lambda: False, on_uploaded=reported.append
        )
        assert reupload("KEY_B", [(0.0, 60.0)]) == uploaded
        assert reported == [uploaded]

    def test_translation_upload_is_reported(
        self, app: QApplication, tmp_path: Path
    ) -> None:
        from subtitles_extractor.presentation.workers.translate_worker import (
            TranslateWorker,
        )
        from tests.unit.presentation.test_cached_rerun_skips_video_v323609 import (
            LITERAL,
            _request,
            _use_case,
        )

        refs = [RemoteVideoRef(0, "files/t0", 0.0, 60.0, "ACTIVE")]
        provider = MagicMock()
        provider.prepare_and_upload.return_value = refs
        worker = TranslateWorker(
            use_case=_use_case(tmp_path, []), request=_request(),
            video_provider=provider, video_path="phim.mp4",
            attach_video_stages=frozenset({LITERAL}),
        )
        reported: list = []
        worker.video_refs_ready.connect(reported.append)
        worker.run()
        # Lần đầu là đoạn tải trước khi dịch (các lần sau: tải lại khi key đổi).
        assert reported and reported[0] == refs

    def test_translation_uploads_are_saved_next_to_earlier_ones(self, vm) -> None:
        view_model, container = vm
        store = container.translation_session_store
        store.save_cloud_files("/phim/tap1.mkv", [CloudVideoFile("files/a", 0.0, 60.0)])
        view_model._last_translate_video_path = "/phim/tap1.mkv"
        view_model._on_translate_video_refs_ready(
            [RemoteVideoRef(0, "files/t0", 0.0, 60.0, "ACTIVE")]
        )
        names = [f.remote_name for f in store.get("/phim/tap1.mkv").cloud_files]
        assert names == ["files/a", "files/t0"]

    def test_new_analysis_does_not_forget_earlier_uploads(self, vm) -> None:
        view_model, container = vm
        store = container.translation_session_store
        view_model._pending_analysis_video_path = "/phim/tap2.mkv"
        view_model._on_video_refs_ready([RemoteVideoRef(0, "files/a", 0.0, 60.0)])
        view_model._on_video_refs_ready([RemoteVideoRef(0, "files/b", 0.0, 60.0)])
        names = [f.remote_name for f in store.get("/phim/tap2.mkv").cloud_files]
        assert names == ["files/a", "files/b"]

    def test_records_past_the_cloud_lifetime_are_dropped(self, vm) -> None:
        view_model, container = vm
        store = container.translation_session_store
        stale = (datetime.now(UTC) - timedelta(hours=49)).isoformat()
        store.save_cloud_files(
            "/phim/tap3.mkv", [CloudVideoFile("files/old", 0.0, 60.0, stale)]
        )
        view_model._pending_analysis_video_path = "/phim/tap3.mkv"
        view_model._on_video_refs_ready([RemoteVideoRef(0, "files/new", 0.0, 60.0)])
        files = store.get("/phim/tap3.mkv").cloud_files
        assert [f.remote_name for f in files] == ["files/new"]
        assert files[0].uploaded_at


def test_files_that_failed_to_delete_are_kept_for_a_retry(vm) -> None:
    view_model, container = vm
    store = container.translation_session_store
    store.save_cloud_files(
        "/phim/tap4.mkv",
        [CloudVideoFile("files/a", 0.0, 60.0), CloudVideoFile("files/b", 60.0, 120.0)],
    )
    provider = MagicMock()
    provider.delete_remote_files.return_value = {"files/a": True, "files/b": False}
    messages: list[str] = []
    view_model.status_message.connect(messages.append)
    with patch.object(container, "make_video_context_provider", lambda _key: provider):
        assert view_model.delete_cloud_files_for_video("/phim/tap4.mkv", "KEY") == 1
    assert [f.remote_name for f in store.get("/phim/tap4.mkv").cloud_files] == ["files/b"]
    assert "thử lại" in messages[-1]


def test_clear_cloud_button_uses_every_key(
    app: QApplication, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Video được tải bằng mọi key (chính + dự phòng); nút xoá chỉ gửi key chính."""
    from PySide6.QtWidgets import QMessageBox

    from tests.unit.presentation.test_translate_review2_v323609 import _page

    page = _page(tmp_path)
    page._api_key_edit.setText("KEY_A")
    page._extra_keys = ["KEY_B", "KEY_C"]
    page._session_video_path = lambda: "/phim/tap5.mkv"  # type: ignore[method-assign]
    monkeypatch.setattr(
        QMessageBox, "question", lambda *_a, **_k: QMessageBox.StandardButton.Yes
    )
    received: list[str] = []
    page._view_model.delete_cloud_files_for_video = (  # type: ignore[method-assign]
        lambda _path, keys: received.append(keys) or 0
    )
    page._on_clear_cloud_clicked()
    assert received and received[0].split() == ["KEY_A", "KEY_B", "KEY_C"]
    page.close()
