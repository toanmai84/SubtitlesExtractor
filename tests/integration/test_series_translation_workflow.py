"""Test tích hợp: luồng dịch phim bộ nhiều tập với Translation Memory + ngữ cảnh chung.

Mô phỏng kịch bản thực: dịch tập 1 → bộ nhớ tích luỹ → dịch tập 2 (cùng thư mục) truy hồi
được tham chiếu + kế thừa glossary/roster. Dùng store SQLite thật (file tạm), service thật.
Đây là nơi các lỗi tích hợp (phình overview, lệch dòng, cô lập sai) từng ẩn náu.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from subtitles_extractor.application.services.translation_memory import (
    TranslationMemoryEntry,
    derive_series_key,
    format_reference_block,
    merge_glossary,
    retrieve_relevant,
)
from subtitles_extractor.infrastructure.database.sqlite_translation_memory_store import (
    SqliteTranslationMemoryStore,
)


@pytest.fixture()
def store() -> SqliteTranslationMemoryStore:
    path = Path(tempfile.mktemp(suffix=".db"))
    instance = SqliteTranslationMemoryStore(path)
    yield instance
    instance.close()
    path.unlink(missing_ok=True)


def test_full_series_workflow_two_episodes(store: SqliteTranslationMemoryStore) -> None:
    series_ep1 = "/phim/TienHiep/EP01.mp4"
    series_ep2 = "/phim/TienHiep/EP02.mp4"
    key1 = derive_series_key(series_ep1)
    key2 = derive_series_key(series_ep2)
    # Cùng thư mục → cùng khoá phim bộ.
    assert key1 == key2 == "TienHiep"

    # --- Tập 1: dịch xong, lưu câu + ngữ cảnh ---
    store.add_entries(key1, [
        TranslationMemoryEntry("林恒走进来", "Lâm Hằng đi vào"),
        TranslationMemoryEntry("我要修炼", "Ta phải tu luyện"),
    ])
    store.save_series_context(key1, "林恒 => Lâm Hằng", "林恒 (nam chính)", "Phim tu tiên")

    # --- Tập 2: truy hồi tham chiếu cho câu tương tự ---
    entries = store.get_entries(key2)
    refs = retrieve_relevant("林恒又走进来", entries, top_k=2)
    block = format_reference_block(refs)
    assert "Lâm Hằng" in block  # tái sử dụng cách dịch tên riêng

    # --- Tập 2: kế thừa + bổ sung glossary ---
    prior = store.get_series_context(key2)
    merged = merge_glossary(prior.glossary, "叶天 => Diệp Thiên")
    store.save_series_context(key2, merged, prior.characters, prior.overview)
    final_ctx = store.get_series_context(key2)
    assert "Lâm Hằng" in final_ctx.glossary  # giữ tập 1
    assert "Diệp Thiên" in final_ctx.glossary  # thêm tập 2
    assert final_ctx.characters == "林恒 (nam chính)"  # roster kế thừa


def test_series_isolation_no_crosstalk(store: SqliteTranslationMemoryStore) -> None:
    store.add_entries("PhimA", [TranslationMemoryEntry("句子A", "Câu A")])
    store.add_entries("PhimB", [TranslationMemoryEntry("句子B", "Câu B")])
    # Truy hồi cho PhimB không được lẫn dữ liệu PhimA.
    entries_b = store.get_entries("PhimB")
    sources_b = {entry.source_text for entry in entries_b}
    assert sources_b == {"句子B"}


def test_overview_stable_across_many_episodes(store: SqliteTranslationMemoryStore) -> None:
    # Lặp lưu ngữ cảnh nhiều lần (mô phỏng nhiều tập) — overview KHÔNG được phình to.
    key = "SeriesStable"
    overview = "Tóm tắt cố định của phim bộ."
    glossary = "X => Y"
    for _episode in range(5):
        prior = store.get_series_context(key)
        merged = merge_glossary(prior.glossary if prior else "", glossary)
        store.save_series_context(key, merged, "roster", overview)
    final_ctx = store.get_series_context(key)
    assert final_ctx.overview == overview  # không đổi
    # glossary chỉ có 1 mục (X) vì lặp cùng nội dung → không trùng lặp.
    assert final_ctx.glossary.count("X => Y") == 1


def test_clear_series_full_cleanup(store: SqliteTranslationMemoryStore) -> None:
    key = "ToClear"
    store.add_entries(key, [TranslationMemoryEntry("a", "A")])
    store.save_series_context(key, "g", "c", "o")
    store.clear_series(key)
    # Cả câu lẫn ngữ cảnh đều bị xoá sạch.
    assert store.count_entries(key) == 0
    assert store.get_series_context(key) is None
    assert key not in [name for name, _ in store.list_series()]
