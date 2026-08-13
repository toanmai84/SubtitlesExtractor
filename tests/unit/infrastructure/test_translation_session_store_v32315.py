"""Test [v3.23.15] SqliteTranslationSessionStore + hàm hash phiên dịch (Bước 2)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from subtitles_extractor.infrastructure.database.sqlite_translation_session_store import (
    CloudVideoFile,
    SqliteTranslationSessionStore,
    StageResult,
)
from subtitles_extractor.application.services.translation_session_hashing import (
    hash_analysis_input,
    hash_stage_input,
    hash_text_lines,
)


@pytest.fixture()
def store(tmp_path: Path) -> SqliteTranslationSessionStore:
    return SqliteTranslationSessionStore(tmp_path / "sess.db")


class TestSessionStore:
    def test_save_and_get_analysis(self, store: SqliteTranslationSessionStore) -> None:
        store.save_analysis(
            "vk1", characters="Lâm Côn", overview="Tóm tắt",
            source_lang="zh", target_lang="vi", input_hash="ih1", source_hash="sh1",
        )
        s = store.get("vk1")
        assert s is not None
        assert s.analysis_characters == "Lâm Côn"
        assert s.analysis_source_lang == "zh"
        assert s.source_hash == "sh1"

    def test_has_valid_analysis(self, store: SqliteTranslationSessionStore) -> None:
        store.save_analysis(
            "vk1", characters="c", overview="o", source_lang="zh",
            target_lang="vi", input_hash="ih1",
        )
        s = store.get("vk1")
        assert s.has_valid_analysis("ih1") is True
        assert s.has_valid_analysis("ih_different") is False

    def test_save_stage_no_duplicate(self, store: SqliteTranslationSessionStore) -> None:
        store.save_stage("vk1", StageResult("literal", "h1", json.dumps([{"i": 1}])))
        store.save_stage("vk1", StageResult("style", "h2", json.dumps([{"i": 1}])))
        store.save_stage("vk1", StageResult("literal", "h1b", json.dumps([{"i": 99}])))
        s = store.get("vk1")
        assert len(s.stages) == 2  # literal ghi đè, không nhân đôi
        assert json.loads(s.stage("literal").lines_json) == [{"i": 99}]

    def test_cloud_files_roundtrip(self, store: SqliteTranslationSessionStore) -> None:
        files = [CloudVideoFile("files/a", 0.0, 100.0), CloudVideoFile("files/b", 100.0, 200.0)]
        store.save_cloud_files("vk1", files)
        s = store.get("vk1")
        assert [c.remote_name for c in s.cloud_files] == ["files/a", "files/b"]
        assert s.cloud_files[0].end_sec == 100.0

    def test_clear_stages_keeps_others(self, store: SqliteTranslationSessionStore) -> None:
        store.save_analysis("vk1", characters="c", overview="o", source_lang="zh",
                            target_lang="vi", input_hash="ih1")
        store.save_stage("vk1", StageResult("literal", "h1", "[]"))
        store.save_cloud_files("vk1", [CloudVideoFile("files/a", 0, 100)])
        store.clear_stages("vk1")
        s = store.get("vk1")
        assert len(s.stages) == 0
        assert len(s.cloud_files) == 1   # giữ
        assert s.analysis_overview == "o"  # giữ

    def test_clear_cloud_files(self, store: SqliteTranslationSessionStore) -> None:
        store.save_cloud_files("vk1", [CloudVideoFile("files/a", 0, 100)])
        store.clear_cloud_files("vk1")
        assert len(store.get("vk1").cloud_files) == 0

    def test_delete(self, store: SqliteTranslationSessionStore) -> None:
        store.save_analysis("vk1", characters="c", overview="o", source_lang="zh",
                            target_lang="vi", input_hash="ih1")
        store.delete("vk1")
        assert store.get("vk1") is None

    def test_get_missing_returns_none(self, store: SqliteTranslationSessionStore) -> None:
        assert store.get("never_seen") is None

    def test_empty_key_raises(self, store: SqliteTranslationSessionStore) -> None:
        with pytest.raises(ValueError):
            store.save_analysis("", characters="c", overview="o", source_lang="z",
                               target_lang="v", input_hash="h")


class TestSessionHashing:
    def test_text_lines_stable_and_sensitive(self) -> None:
        assert hash_text_lines(["a", "b"]) == hash_text_lines(["a", "b"])
        assert hash_text_lines(["a", "b"]) != hash_text_lines(["a", "c"])

    def test_analysis_sensitive_to_target_lang(self) -> None:
        assert hash_analysis_input(["x"], "vi") != hash_analysis_input(["x"], "en")

    def test_analysis_sensitive_to_video_signature(self) -> None:
        assert hash_analysis_input(["x"], "vi", "sig1") != hash_analysis_input(["x"], "vi", "sig2")

    def test_stage_chain_dependency(self) -> None:
        # Đổi hash giai đoạn trước → hash giai đoạn sau đổi (dây chuyền).
        assert hash_stage_input("A", "style", "vi") != hash_stage_input("B", "style", "vi")

    def test_all_hex_64(self) -> None:
        assert len(hash_text_lines(["x"])) == 64
        assert len(hash_analysis_input(["x"], "vi")) == 64
        assert len(hash_stage_input("a", "s", "vi")) == 64
