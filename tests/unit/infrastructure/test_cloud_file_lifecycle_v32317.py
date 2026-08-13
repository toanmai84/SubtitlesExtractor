"""Test [v3.23.17] vòng đời file cloud: xoá file Gemini + dọn cache (Bước 3)."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

from subtitles_extractor.infrastructure.translation.gemini_video_context import (
    GeminiVideoContextProvider,
)


class TestDeleteRemoteFiles:
    def _provider(self, tmp_path: Path) -> GeminiVideoContextProvider:
        return GeminiVideoContextProvider(
            cache_db_path=tmp_path / "cache.db", work_dir=tmp_path
        )

    def test_deletes_and_clears_cache(self, tmp_path: Path) -> None:
        prov = self._provider(tmp_path)
        prov._cache_store("h1", "files/abc")
        prov._cache_store("h2", "files/def")
        fake = MagicMock()
        with patch.object(prov, "_ensure_client", lambda: fake):
            res = prov.delete_remote_files(["files/abc", "files/def"])
        assert res == {"files/abc": True, "files/def": True}
        assert fake.files.delete.call_count == 2
        with sqlite3.connect(str(tmp_path / "cache.db")) as c:
            rows = c.execute("SELECT remote_name FROM video_context_uploads").fetchall()
        assert rows == []

    def test_404_treated_as_success(self, tmp_path: Path) -> None:
        prov = self._provider(tmp_path)
        fake = MagicMock()
        fake.files.delete.side_effect = Exception("404 not_found")
        with patch.object(prov, "_ensure_client", lambda: fake):
            res = prov.delete_remote_files(["files/ghost"])
        assert res == {"files/ghost": True}

    def test_real_error_returns_false(self, tmp_path: Path) -> None:
        prov = self._provider(tmp_path)
        fake = MagicMock()
        fake.files.delete.side_effect = Exception("500 internal error")
        with patch.object(prov, "_ensure_client", lambda: fake):
            res = prov.delete_remote_files(["files/x"])
        assert res == {"files/x": False}

    def test_empty_list_noop(self, tmp_path: Path) -> None:
        prov = self._provider(tmp_path)
        assert prov.delete_remote_files([]) == {}

    def test_clears_inmem_refs(self, tmp_path: Path) -> None:
        from subtitles_extractor.infrastructure.translation.gemini_video_context import (
            RemoteVideoRef,
        )
        prov = self._provider(tmp_path)
        prov._inmem_refs["sig1"] = [RemoteVideoRef(0, "files/abc", 0, 100, "ACTIVE")]
        fake = MagicMock()
        with patch.object(prov, "_ensure_client", lambda: fake):
            prov.delete_remote_files(["files/abc"])
        assert "sig1" not in prov._inmem_refs
