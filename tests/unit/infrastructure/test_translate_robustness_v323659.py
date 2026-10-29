"""[v3.23.659] Rà BUG khâu Dịch: lô bị chặn, JSON sai hình dạng, schema của lô con, dòng thiếu ở lô nhỏ, TM thiếu cột."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from subtitles_extractor.domain.ports.subtitle_translator_port import (
    TranslationContext,
    TranslationLine,
    TranslationStageConfig,
    TranslationStageKind,
)
from tests.unit.infrastructure.gemini_call_harness import build_translator, install_fake_clock


def _lines(n: int) -> list[TranslationLine]:
    return [TranslationLine(index=i, start_ms=i * 1000, end_ms=i * 1000 + 900, text=f"你好{i}") for i in range(1, n + 1)]


def _current(contents) -> list[dict]:
    prompt = contents[-1]
    return json.loads(prompt.split("<current_batch>")[1].split("</current_batch>")[0])


def _ok(items, skip=()) -> SimpleNamespace:
    out = [{"line_no": it["line_no"], "text": f"VI{it['line_no']}"} for it in items if it["line_no"] not in skip]
    return SimpleNamespace(text=json.dumps({"subtitles": out}), usage_metadata=None, candidates=None)


def _run(monkeypatch, server, n: int, batch: int):
    clock = install_fake_clock(monkeypatch)
    translator, _ = build_translator(monkeypatch, clock, None, server)
    stage = TranslationStageConfig(kind=TranslationStageKind.LITERAL, model_name="gemini-2.5-flash", batch_size=batch)
    context = TranslationContext(target_lang="vi", source_lang="zh")
    return translator.translate_stage(
        stage=stage, context=context, source_lines=_lines(n), input_lines=_lines(n), has_prior_translation=False,
    )


class _Blocked:
    """Lô chứa dòng 7 luôn bị chặn (PROHIBITED_CONTENT / RECITATION — lời bài hát trong phim)."""

    calls = 0

    def generate_content(self, model, contents, config):
        self.calls += 1
        items = _current(contents)
        if any(it["line_no"] == 7 for it in items):
            return SimpleNamespace(text=None, usage_metadata=None, prompt_feedback=None,
                                   candidates=[SimpleNamespace(finish_reason="PROHIBITED_CONTENT")])
        return _ok(items)


def test_blocked_batch_does_not_fail_the_episode(monkeypatch) -> None:
    """Một lô bị chặn từng làm hỏng CẢ tập (lỗi lặp lại ở mọi lần chạy, "Dịch cả bộ" luôn báo lỗi tập đó)."""
    out = _run(monkeypatch, _Blocked(), n=60, batch=20)
    texts = [line.text for line in out]
    assert texts[6] == "你好7"                                        # chỉ dòng bị chặn giữ bản gốc
    assert all(t == f"VI{i + 1}" for i, t in enumerate(texts) if i != 6)


@pytest.mark.parametrize("bad", [
    json.dumps({"subtitles": None}),
    json.dumps({"subtitles": ["a", "b", "c"]}),
    json.dumps({"subtitles": [{"line_no": 1, "text": "a"}, "b", {"line_no": 3, "text": "c"}]}),
    '"just a string"',
    json.dumps({"subtitles": {"line_no": 1, "text": "a"}}),
])
def test_malformed_shape_is_retried(monkeypatch, bad: str) -> None:
    """JSON sai hình dạng từng ném AttributeError/TypeError không được thử lại → hỏng cả tập ngay lần gọi đầu."""

    class Server:
        calls = 0

        def generate_content(self, model, contents, config):
            self.calls += 1
            if self.calls == 1:
                return SimpleNamespace(text=bad, usage_metadata=None, candidates=None)
            return _ok(_current(contents))

    server = Server()
    out = _run(monkeypatch, server, n=3, batch=10)
    assert [line.text for line in out] == ["VI1", "VI2", "VI3"] and server.calls >= 2


def test_top_level_array_is_accepted(monkeypatch) -> None:
    class Server:
        def generate_content(self, model, contents, config):
            items = _current(contents)
            out = [{"line_no": it["line_no"], "text": f"VI{it['line_no']}"} for it in items]
            return SimpleNamespace(text=json.dumps(out), usage_metadata=None, candidates=None)

    assert [line.text for line in _run(monkeypatch, Server(), n=3, batch=10)] == ["VI1", "VI2", "VI3"]


def test_missing_line_in_small_batch_is_patched(monkeypatch) -> None:
    """Lô ≤13 dòng thiếu một dòng: trước đây giữ NGUYÊN chữ Hán của dòng đó, không vá, không chia đôi."""

    class Server:
        calls = 0

        def generate_content(self, model, contents, config):
            self.calls += 1
            items = _current(contents)
            return _ok(items, skip=(3,) if self.calls == 1 else ())

    out = _run(monkeypatch, Server(), n=8, batch=10)
    assert [line.text for line in out] == [f"VI{i}" for i in range(1, 9)]


def _schema_limits(config) -> tuple:
    schema = getattr(config, "response_schema", None)
    if schema is None:
        return ()
    if not isinstance(schema, dict):
        schema = schema.model_dump(exclude_none=True) if hasattr(schema, "model_dump") else {}
    array = schema.get("properties", {}).get("subtitles", {})
    return (array.get("minItems") or array.get("min_items"), array.get("maxItems") or array.get("max_items"))


def test_patch_window_uses_its_own_schema(monkeypatch) -> None:
    """Cửa sổ vá 12 dòng từng dùng schema của lô 30 dòng (minItems=maxItems=30) → ép model bịa thêm 18 dòng."""

    class Server:
        def __init__(self) -> None:
            self.calls: list[tuple[int, tuple]] = []

        def generate_content(self, model, contents, config):
            items = _current(contents)
            self.calls.append((len(items), _schema_limits(config)))
            return _ok(items, skip=(25,) if len(self.calls) == 1 else ())

    server = Server()
    _run(monkeypatch, server, n=30, batch=30)
    for size, limits in server.calls:
        if limits and limits[0]:
            assert limits[0] == limits[1] == size, server.calls


def test_translation_memory_without_video_key_column(tmp_path: Path) -> None:
    """Thêm cột video_key thất bại (CSDL bị khoá) → mọi tra cứu TM lỗi "no such column" và TM tắt cả phiên."""
    from subtitles_extractor.application.services.translation_memory import TranslationMemoryEntry
    from subtitles_extractor.infrastructure.database.sqlite_translation_memory_store import (
        SqliteTranslationMemoryStore,
    )

    path = tmp_path / "tm.db"
    with sqlite3.connect(path) as conn:                               # CSDL bản cũ: chưa có cột video_key
        conn.execute("CREATE TABLE translation_memory (series_key TEXT NOT NULL, source_text TEXT NOT NULL, "
                     "target_text TEXT NOT NULL, updated_at TEXT, PRIMARY KEY (series_key, source_text))")
        conn.execute("INSERT INTO translation_memory VALUES ('bo', '你好', 'Chào', datetime('now'))")
    store = SqliteTranslationMemoryStore.__new__(SqliteTranslationMemoryStore)
    store._db_path = str(path)
    import threading

    store._lock = threading.RLock()
    store._conn = sqlite3.connect(str(path), check_same_thread=False)
    store._conn.row_factory = sqlite3.Row
    store._has_video_key = False                                      # ALTER TABLE đã thất bại
    assert [e.target_text for e in store.get_entries("bo", exclude_video="ep1")] == ["Chào"]
    assert store.add_entries("bo", [TranslationMemoryEntry(source_text="走", target_text="Đi")], video_key="ep1") == 1
    store._conn.close()


def test_parallel_batches_run_stage_safety_nets(monkeypatch) -> None:
    """Dịch song song từng trả thẳng kết quả, bỏ qua hoàn nguyên dòng rớt nội dung + cảnh báo dịch thiếu."""

    class Server:
        def generate_content(self, model, contents, config):
            return _ok(_current(contents))

    clock = install_fake_clock(monkeypatch)
    translator, _ = build_translator(monkeypatch, clock, None, Server())
    monkeypatch.setattr(translator, "_effective_parallel", lambda *_a, **_k: 2)
    monkeypatch.setattr(translator, "_run_batches_parallel", lambda **kw: list(kw["input_lines"]))
    calls: list[str] = []
    real_revert = translator._revert_stage_regressions
    monkeypatch.setattr(translator, "_revert_stage_regressions",
                        lambda *a, **k: calls.append("revert") or real_revert(*a, **k))
    monkeypatch.setattr(translator, "_warn_under_translated", lambda *a, **k: calls.append("warn"))
    stage = TranslationStageConfig(kind=TranslationStageKind.LITERAL, model_name="gemini-2.5-flash", batch_size=5)
    translator.translate_stage(
        stage=stage, context=TranslationContext(target_lang="vi", source_lang="zh"),
        source_lines=_lines(12), input_lines=_lines(12), has_prior_translation=False,
    )
    assert calls == ["revert", "warn"]


def test_fully_blocked_batch_has_bounded_cost(monkeypatch) -> None:
    """Rà độc lập: lô bị chặn HOÀN TOÀN (chặn chung cả tập) từng chia đôi tới từng dòng — 297 lượt gọi cho 50 dòng."""

    class AllBlocked:
        calls = 0

        def generate_content(self, model, contents, config):
            self.calls += 1
            return SimpleNamespace(text=None, usage_metadata=None, prompt_feedback=None,
                                   candidates=[SimpleNamespace(finish_reason="PROHIBITED_CONTENT")])

    server = AllBlocked()
    try:
        _run(monkeypatch, server, n=50, batch=50)
    except Exception:  # noqa: BLE001 — điều cần kiểm là số lượt gọi
        pass
    assert server.calls <= 30, server.calls  # trước: 297
