"""Test [v3.23.57] quản lý bộ nhớ dịch: list_series + clear_series (cả ngữ cảnh)."""

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


def test_list_series_with_counts(store) -> None:
    store.add_entries("alpha", [E("a", "A"), E("b", "B")])
    store.add_entries("beta", [E("c", "C")])
    listing = store.list_series()
    assert ("alpha", 2) in listing
    assert ("beta", 1) in listing


def test_list_series_sorted(store) -> None:
    store.add_entries("zeta", [E("z", "Z")])
    store.add_entries("alpha", [E("a", "A")])
    keys = [k for k, _ in store.list_series()]
    assert keys == sorted(keys)


def test_list_empty(store) -> None:
    assert store.list_series() == []


def test_clear_removes_memory_and_context(store) -> None:
    store.add_entries("alpha", [E("a", "A")])
    store.save_series_context("alpha", "g", "c", "o")
    store.clear_series("alpha")
    assert store.count_entries("alpha") == 0
    assert store.get_series_context("alpha") is None
    assert ("alpha", 1) not in store.list_series()
