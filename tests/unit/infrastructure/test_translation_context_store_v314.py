"""Test store SQLite ngữ cảnh dịch per-project (chống context leak)."""

from __future__ import annotations

from pathlib import Path

import pytest

from subtitles_extractor.infrastructure.database.sqlite_translation_context_store import (
    SqliteTranslationContextStore,
    TranslationContext,
)


@pytest.fixture
def store(tmp_path: Path):
    s = SqliteTranslationContextStore(tmp_path / "ctx.db")
    yield s
    s.close()


class TestTranslationContextStore:
    def test_save_and_get(self, store) -> None:
        store.save("phimA.mp4", TranslationContext(
            characters="Lâm Côn: chính", overview="Tu tiên", target_lang="vi"))
        ctx = store.get("phimA.mp4")
        assert ctx is not None
        assert ctx.characters == "Lâm Côn: chính"
        assert ctx.overview == "Tu tiên"
        assert ctx.target_lang == "vi"

    def test_per_project_isolation(self, store) -> None:
        # CỐT LÕI chống leak: phim A và phim B độc lập hoàn toàn.
        store.save("phimA.mp4", TranslationContext(overview="Bối cảnh A"))
        store.save("phimB.mp4", TranslationContext(overview="Bối cảnh B"))
        assert store.get("phimA.mp4").overview == "Bối cảnh A"
        assert store.get("phimB.mp4").overview == "Bối cảnh B"

    def test_get_missing_returns_none(self, store) -> None:
        assert store.get("chua_co.mp4") is None

    def test_update_overwrites(self, store) -> None:
        store.save("p.mp4", TranslationContext(overview="cũ"))
        store.save("p.mp4", TranslationContext(overview="mới"))
        assert store.get("p.mp4").overview == "mới"

    def test_delete(self, store) -> None:
        store.save("p.mp4", TranslationContext(overview="x"))
        store.delete("p.mp4")
        assert store.get("p.mp4") is None

    def test_empty_key_raises(self, store) -> None:
        with pytest.raises(ValueError):
            store.save("", TranslationContext(overview="x"))

    def test_persists_across_reopen(self, tmp_path: Path) -> None:
        db = tmp_path / "ctx.db"
        s1 = SqliteTranslationContextStore(db)
        s1.save("p.mp4", TranslationContext(characters="A"))
        s1.close()
        s2 = SqliteTranslationContextStore(db)
        assert s2.get("p.mp4").characters == "A"
        s2.close()

    def test_is_empty(self) -> None:
        assert TranslationContext().is_empty()
        assert not TranslationContext(overview="x").is_empty()
