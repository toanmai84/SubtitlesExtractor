"""Test [v3.23.56] merge_glossary + ngữ cảnh chung phim bộ (glossary/roster tích luỹ)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from subtitles_extractor.application.services.translation_memory import (
    SeriesContext,
    merge_glossary,
)
from subtitles_extractor.infrastructure.database.sqlite_translation_memory_store import (
    SqliteTranslationMemoryStore,
)


class TestMergeGlossary:
    def test_keeps_existing_on_conflict(self) -> None:
        merged = merge_glossary("灵力 => linh lực", "灵力 => sức mạnh")
        assert "linh lực" in merged
        assert "sức mạnh" not in merged

    def test_adds_new_terms(self) -> None:
        merged = merge_glossary("林恒 => Lâm Hằng", "叶天 => Diệp Thiên")
        assert "Lâm Hằng" in merged and "Diệp Thiên" in merged

    def test_order_existing_first(self) -> None:
        merged = merge_glossary("a => A", "b => B")
        assert merged.index("a => A") < merged.index("b => B")

    def test_empty_blocks(self) -> None:
        assert merge_glossary("", "") == ""
        assert merge_glossary("x => y", "") == "x => y"


@pytest.fixture()
def store():
    path = Path(tempfile.mktemp(suffix=".db"))
    s = SqliteTranslationMemoryStore(path)
    yield s
    s.close()
    path.unlink(missing_ok=True)


class TestSeriesContextStore:
    def test_save_and_get(self, store) -> None:
        store.save_series_context("s1", "g", "c", "o")
        ctx = store.get_series_context("s1")
        assert isinstance(ctx, SeriesContext)
        assert ctx.glossary == "g" and ctx.characters == "c" and ctx.overview == "o"

    def test_upsert(self, store) -> None:
        store.save_series_context("s1", "g1", "c1", "o1")
        store.save_series_context("s1", "g2", "c2", "o2")
        ctx = store.get_series_context("s1")
        assert ctx.glossary == "g2"

    def test_missing_returns_none(self, store) -> None:
        assert store.get_series_context("never") is None

    def test_empty_key(self, store) -> None:
        assert store.get_series_context("") is None
        with pytest.raises(ValueError):
            store.save_series_context("", "g", "c", "o")
