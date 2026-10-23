"""[v3.23.641] Hồi quy các lỗi sập / kẹt tìm được khi rà soát bản 3.23.636-640."""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from subtitles_extractor.domain.ports.subtitle_translator_port import TranslatorUnavailableError
from subtitles_extractor.infrastructure.translation import gemini_quota_manager as gqm
from subtitles_extractor.infrastructure.translation.gemini_quota_manager import GeminiQuotaManager
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
)
from subtitles_extractor.infrastructure.translation.model_health import ModelHealth

MODEL = "gemini-3.1-flash-lite"
PERIOD = GeminiQuotaManager._period_key()


def _cfg() -> SimpleNamespace:
    return SimpleNamespace(
        temperature=0.2, response_mime_type="application/json", response_schema={},
        system_instruction="x",
    )


class TestCorruptQuotaState:
    """Tệp trạng thái là JSON hợp lệ nhưng sai hình dạng / sai kiểu → KHÔNG được làm hỏng khởi động."""

    @pytest.mark.parametrize("payload", [
        "[]", "[1, 2]", '"x"', "5", "null", "{}", '{"__meta__": []}',
        '{"k|m": {"period_key": "%s", "count": "abc"}}' % PERIOD,
        '{"k|m": {"period_key": "%s", "count": null}}' % PERIOD,
        '{"k|m": {"period_key": "%s", "count": [1]}}' % PERIOD,
        '{"k|m": {"period_key": "%s", "count": Infinity}}' % PERIOD,
        '{"__meta__": {"learned_rpd": {"m": Infinity}}}',
        '{"__meta__": {"learned_minute": {"m": [Infinity, 5]}}}',
        '{"__meta__": {"invalid_keys": {"a": [1], "b": "xx", "c": [null, null], "d": 5}}}',
        '{"__meta__": {"server_daily_block": {"a": null, "b": [1]}}}',
    ])
    def test_constructor_survives(self, tmp_path: Path, payload: str) -> None:
        path = tmp_path / "q.json"
        path.write_text(payload, encoding="utf-8")
        manager = GeminiQuotaManager(state_path=path)
        manager.snapshot()
        assert manager.get_remaining(MODEL, key_id="k")["rpd_remaining"] >= 0

    def test_bad_count_only_drops_that_entry(self, tmp_path: Path) -> None:
        path = tmp_path / "q.json"
        path.write_text(json.dumps({
            "bad|m": {"period_key": PERIOD, "count": "abc"},
            "good|m": {"period_key": PERIOD, "count": 7},
        }), encoding="utf-8")
        manager = GeminiQuotaManager(state_path=path)
        assert manager._daily["good|m"]["count"] == 7
        assert "bad|m" not in manager._daily

    def test_save_is_atomic_and_leaves_no_temp_file(self, tmp_path: Path) -> None:
        path = tmp_path / "q.json"
        manager = GeminiQuotaManager(state_path=path)
        manager.mark_key_invalid("fp1", "lý do")
        assert json.loads(path.read_text(encoding="utf-8"))["__meta__"]["invalid_keys"]["fp1"][0] == "lý do"
        assert not list(tmp_path.glob("*.tmp"))

    def test_lone_surrogate_reason_does_not_raise(self, tmp_path: Path) -> None:
        path = tmp_path / "q.json"
        manager = GeminiQuotaManager(state_path=path)
        manager.mark_key_invalid("fp1", "lỗi \ud800 lạ" + "x" * 1000)
        assert manager.is_key_invalid("fp1")
        assert len(manager.invalid_key_reason("fp1")) <= 300
        assert manager._lock.acquire(timeout=1)  # khoá không bị kẹt sau ngoại lệ
        manager._lock.release()


class TestInvalidKeyExpires:
    def test_mark_expires_after_ttl(self, monkeypatch: pytest.MonkeyPatch) -> None:
        manager = GeminiQuotaManager()
        manager.mark_key_invalid("fp1", "hỏng")
        assert manager.is_key_invalid("fp1")
        manager._invalid_marked_at["fp1"] -= gqm.INVALID_KEY_TTL_S + 1
        assert not manager.is_key_invalid("fp1")
        assert manager.invalid_key_reason("fp1") == ""

    def test_mark_from_previous_session_survives_restart_until_ttl(self, tmp_path: Path) -> None:
        path = tmp_path / "q.json"
        GeminiQuotaManager(state_path=path).mark_key_invalid("fp1", "hỏng")
        assert GeminiQuotaManager(state_path=path).is_key_invalid("fp1")


class _Server:
    ok = False

    def __init__(self) -> None:
        self.models = self

    def generate_content(self, model, contents, config):  # noqa: ANN001
        if not _Server.ok:
            raise RuntimeError(
                "400 INVALID_ARGUMENT. API key not valid. reason: API_KEY_INVALID"
            )
        return SimpleNamespace(text='{"subtitles": []}', usage_metadata=None)


class TestAdapterFixes:
    @pytest.mark.parametrize("keys", ["", "  \n , "])
    @pytest.mark.parametrize("with_manager", [False, True])
    def test_zero_keys_raises_unavailable_not_index_error(self, keys: str, with_manager: bool) -> None:
        manager = GeminiQuotaManager() if with_manager else None
        adapter = GeminiSubtitleTranslator(api_key=keys, quota_manager=manager, retry_count=2)
        with pytest.raises(TranslatorUnavailableError):
            adapter._call_gemini(MODEL, "p", _cfg(), lambda d: None)

    def test_fixed_key_is_unmarked_after_a_successful_call(self, tmp_path: Path) -> None:
        manager = GeminiQuotaManager(state_path=tmp_path / "q.json")
        adapter = GeminiSubtitleTranslator(api_key="ONLY_KEY", quota_manager=manager, retry_count=2)
        adapter._client_for_index = lambda i: _Server()  # type: ignore[method-assign]
        adapter._interruptible_sleep = lambda s, c: False  # type: ignore[method-assign]
        key_id = GeminiQuotaManager.key_fingerprint("ONLY_KEY")
        manager.mark_key_invalid(key_id, "hỏng")
        _Server.ok = True
        try:
            adapter._call_gemini(MODEL, "p", _cfg(), lambda d: None)
        finally:
            _Server.ok = False
        assert not manager.is_key_invalid(key_id)
        assert adapter.has_any_daily_quota(MODEL)

    def test_client_for_index_never_returns_other_keys_client(self) -> None:
        class FakeGenai:
            class Client:
                def __init__(self, api_key=None, http_options=None):  # noqa: ANN001
                    self.api_key = api_key

        adapter = GeminiSubtitleTranslator(api_key="K0\nK1", retry_count=1)
        adapter._genai_module = FakeGenai
        old_interval = sys.getswitchinterval()
        sys.setswitchinterval(1e-6)
        stop = threading.Event()
        wrong: list[str] = []

        def switcher() -> None:
            i = 0
            while not stop.is_set():
                with adapter._state_lock:
                    adapter._switch_to_key(i % 2)
                i += 1

        def caller() -> None:
            for _ in range(20000):
                with adapter._state_lock:
                    idx = adapter._key_index
                client = adapter._client_for_index(idx)
                if client is None or client.api_key != adapter._api_keys[idx]:
                    wrong.append(str(getattr(client, "api_key", None)))

        threads = [threading.Thread(target=switcher)]
        threads += [threading.Thread(target=caller) for _ in range(3)]
        try:
            for t in threads:
                t.start()
            for t in threads[1:]:
                t.join()
        finally:
            stop.set()
            threads[0].join()
            sys.setswitchinterval(old_interval)
        assert wrong == []


def test_model_health_cooldown_does_not_overflow() -> None:
    now = [0.0]
    health = ModelHealth(clock=lambda: now[0])
    for _ in range(1500):
        health.record_overload("m")
        now[0] += 10_000
    assert health.cooling_remaining("m") >= 0
