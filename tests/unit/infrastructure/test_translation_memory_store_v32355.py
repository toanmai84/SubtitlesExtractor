"""Test [v3.23.55] SqliteTranslationMemoryStore: lưu/lấy/cô lập theo phim bộ."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from subtitles_extractor.application.services.translation_memory import (
    TranslationMemoryEntry as E,
)
from subtitles_extractor.infrastructure.database.sqlite_translation_memory_store import (
    SqliteTranslationMemoryStore,
)


@pytest.fixture()
def store():
    path = Path(tempfile.mktemp(suffix=".db"))
    s = SqliteTranslationMemoryStore(path)
    yield s
    s.close()
    path.unlink(missing_ok=True)


def test_add_and_count(store) -> None:
    n = store.add_entries("series_a", [E("林恒", "Lâm Hằng"), E("你好", "Xin chào")])
    assert n == 2
    assert store.count_entries("series_a") == 2


def test_skips_empty(store) -> None:
    n = store.add_entries("series_a", [E("", "x"), E("y", ""), E("a", "b")])
    assert n == 1


def test_upsert_updates(store) -> None:
    store.add_entries("series_a", [E("林恒", "Lin Heng")])
    store.add_entries("series_a", [E("林恒", "Lâm Hằng")])  # cập nhật
    assert store.count_entries("series_a") == 1
    entries = store.get_entries("series_a")
    assert entries[0].target_text == "Lâm Hằng"


def test_series_isolation(store) -> None:
    store.add_entries("series_a", [E("a", "A")])
    store.add_entries("series_b", [E("b", "B")])
    assert store.count_entries("series_a") == 1
    assert store.count_entries("series_b") == 1
    sources_a = {e.source_text for e in store.get_entries("series_a")}
    assert sources_a == {"a"}


def test_clear_series(store) -> None:
    store.add_entries("series_a", [E("a", "A")])
    store.clear_series("series_a")
    assert store.count_entries("series_a") == 0


def test_empty_key_raises(store) -> None:
    with pytest.raises(ValueError):
        store.add_entries("", [E("a", "b")])


def test_get_empty_key(store) -> None:
    assert store.get_entries("") == []
    assert store.count_entries("") == 0
