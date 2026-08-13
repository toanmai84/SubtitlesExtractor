"""Test [v3.20 B1]: SqliteTranslationContextStore an toàn đa luồng.

Trước đây store mở connection không ``check_same_thread=False`` và không có khoá —
nếu worker dịch (QThread) save/get ngữ cảnh sẽ ``ProgrammingError`` hoặc đua dữ
liệu. Test mô phỏng nhiều thread save/get/delete đồng thời.
"""

from __future__ import annotations

import threading
from pathlib import Path

from subtitles_extractor.infrastructure.database.sqlite_translation_context_store import (
    SqliteTranslationContextStore,
    TranslationContext,
)


def test_concurrent_access_does_not_raise(tmp_path: Path) -> None:
    store = SqliteTranslationContextStore(tmp_path / "ctx.db")
    errors: list[BaseException] = []

    def worker(index: int) -> None:
        try:
            for i in range(40):
                key = f"project_{index}"
                store.save(
                    key,
                    TranslationContext(
                        characters=f"nhân vật {index}-{i}",
                        overview="tóm tắt",
                        source_lang="zh",
                        target_lang="vi",
                    ),
                )
                _ = store.get(key)
            store.delete(f"project_{index}")
        except BaseException as exc:  # noqa: BLE001 - thu lỗi từ thread phụ
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors, f"Truy cập đa luồng gây lỗi: {errors[:3]}"
    store.close()


def test_save_get_roundtrip(tmp_path: Path) -> None:
    store = SqliteTranslationContextStore(tmp_path / "ctx.db")
    ctx = TranslationContext(characters="A, B", overview="cốt truyện", source_lang="ja")
    store.save("phim_X", ctx)
    loaded = store.get("phim_X")
    assert loaded is not None
    assert loaded.characters == "A, B" and loaded.source_lang == "ja"
    store.delete("phim_X")
    assert store.get("phim_X") is None
    store.close()
