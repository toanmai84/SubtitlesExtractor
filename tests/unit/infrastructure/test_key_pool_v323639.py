"""[v3.23.639] Quản lý quota + xoay key thông minh.

Dữ liệu thật dùng để thiết kế:

* Trạng thái quota của 3 key sau một buổi làm việc: key #1 chạm 500/500 request cho cả hai model
  flash-lite (và bị server từ chối), key #2 chỉ 81-134, key #3 chỉ 45-49 — cách "dính một key tới khi
  cạn" dồn tải thay vì dàn đều.
* Key sai đo bằng API thật: ``400 INVALID_ARGUMENT … reason: API_KEY_INVALID`` — trước đây là lỗi
  "không thể thử lại" làm hỏng cả lượt dịch dù các key khác còn tốt.
* 60 lời gọi dồn trên gemini-3.1-flash-lite: ``429 … GenerateRequestsPerMinutePerProjectPerModel-
  FreeTier, quotaValue: '15'`` → số RPM thật có trong lỗi.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from subtitles_extractor.domain.ports.subtitle_translator_port import SubtitleTranslationError
from subtitles_extractor.infrastructure.translation.gemini_quota_manager import (
    GeminiQuotaManager,
    RateLimit,
)
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
)
from subtitles_extractor.infrastructure.translation.key_selection import (
    KeyView,
    choose_key,
)
from tests.unit.infrastructure.gemini_call_harness import (
    FakeClock,
    build_translator,
    call_gemini,
    install_fake_clock,
)

MODEL = "gemini-3.1-flash-lite"
REAL_INVALID_KEY = (
    "400 INVALID_ARGUMENT. {'error': {'code': 400, 'message': 'API key not valid. Please pass a "
    "valid API key.', 'status': 'INVALID_ARGUMENT', 'details': [{'@type': 'type.googleapis.com/"
    "google.rpc.ErrorInfo', 'reason': 'API_KEY_INVALID', 'domain': 'googleapis.com', 'metadata': "
    "{'service': 'generativelanguage.googleapis.com'}}]}}"
)
REAL_RPM_429 = (
    "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota. "
    "* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_"
    "requests, limit: 15, model: gemini-3.1-flash-lite\\nPlease retry in 5.465032021s.', 'status': "
    "'RESOURCE_EXHAUSTED', 'details': [{'@type': 'type.googleapis.com/google.rpc.QuotaFailure', "
    "'violations': [{'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_"
    "requests', 'quotaId': 'GenerateRequestsPerMinutePerProjectPerModel-FreeTier', "
    "'quotaDimensions': {'location': 'global', 'model': 'gemini-3.1-flash-lite'}, "
    "'quotaValue': '15'}]}, {'@type': 'type.googleapis.com/google.rpc.RetryInfo', "
    "'retryDelay': '5s'}]}}"
)

V = KeyView


class TestChooseKey:
    def test_no_usable_key(self) -> None:
        assert choose_key([], 0) is None
        assert choose_key([V(0, math.inf, 500, 500), V(1, 0, 0, 500), V(2, 0, 500, 500, usable=False)], 0) is None

    def test_keeps_current_when_balanced(self) -> None:
        assert choose_key([V(0, 0, 450, 500), V(1, 0, 500, 500)], 0) == 0

    def test_moves_to_the_clearly_less_used_key(self) -> None:
        # đúng ca thật: một key 500/500, key khác mới dùng 81
        assert choose_key([V(0, 0, 180, 500), V(1, 0, 419, 500), V(2, 0, 455, 500)], 0) == 2

    def test_scarce_model_is_not_balanced(self) -> None:
        # 20 request/ngày: 3 request đã là 15% nhưng chỉ chênh 3 request → không đáng đổi key
        assert choose_key([V(0, 0, 17, 20), V(1, 0, 20, 20)], 0) == 0

    def test_prefers_a_ready_key_over_waiting(self) -> None:
        assert choose_key([V(0, 45.0, 500, 500), V(1, 0, 100, 500)], 0) == 1

    def test_when_all_wait_takes_the_shortest(self) -> None:
        assert choose_key([V(0, 40.0, 500, 500), V(1, 12.0, 500, 500), V(2, 30.0, 500, 500)], 0) == 1

    def test_small_gain_does_not_justify_switching(self) -> None:
        assert choose_key([V(0, 30.0, 500, 500), V(1, 29.5, 500, 500)], 0) == 0

    def test_skips_exhausted_and_blocked(self) -> None:
        assert choose_key([V(0, 0, 0, 500), V(1, math.inf, 400, 500), V(2, 0, 10, 500)], 0) == 2


class TestQuotaManagerAdditions:
    def test_estimate_wait_does_not_reserve(self) -> None:
        manager = GeminiQuotaManager(rate_limits={MODEL: RateLimit(2, 250_000, 500)})
        assert manager.estimate_wait(MODEL, 1000, key_id="a") == 0.0
        assert manager.get_remaining(MODEL, key_id="a")["rpd_used"] == 0
        manager.acquire(MODEL, 1000, key_id="a")
        manager.acquire(MODEL, 1000, key_id="a")
        assert 0 < manager.estimate_wait(MODEL, 1000, key_id="a") <= 60
        assert manager.estimate_wait(MODEL, 1000, key_id="b") == 0.0     # key khác vẫn trống

    def test_tpm_wait_counts_the_request_size(self) -> None:
        manager = GeminiQuotaManager(rate_limits={MODEL: RateLimit(15, 100_000, 500)})
        manager.acquire(MODEL, 80_000, key_id="a")
        assert manager.estimate_wait(MODEL, 10_000, key_id="a") == 0.0
        assert manager.estimate_wait(MODEL, 30_000, key_id="a") > 0.0

    def test_server_blocked_key_is_infinite_and_cooldown_counts(self) -> None:
        manager = GeminiQuotaManager(rate_limits={MODEL: RateLimit(15, 250_000, 5)})
        manager.note_rate_limited(MODEL, 20.0, key_id="a")
        assert 19 < manager.estimate_wait(MODEL, 1, key_id="a") <= 20
        manager.mark_daily_exhausted(MODEL, key_id="b")
        assert math.isinf(manager.estimate_wait(MODEL, 1, key_id="b"))

    def test_invalid_key_is_remembered_across_restarts_within_the_day(self, tmp_path: Path) -> None:
        path = tmp_path / "q.json"
        first = GeminiQuotaManager(state_path=path)
        first.mark_key_invalid("abc", "API key not valid")
        assert first.is_key_invalid("abc") and math.isinf(first.estimate_wait(MODEL, 1, key_id="abc"))
        second = GeminiQuotaManager(state_path=path)
        assert second.is_key_invalid("abc") and "not valid" in second.invalid_key_reason("abc")
        second.clear_invalid_key("abc")
        assert not GeminiQuotaManager(state_path=path).is_key_invalid("abc")

    def test_yesterdays_invalid_mark_expires(self, tmp_path: Path) -> None:
        import json

        path = tmp_path / "q.json"
        path.write_text(json.dumps({"__meta__": {"invalid_keys": {"abc": ["x", "2001-01-01"]}}}))
        assert not GeminiQuotaManager(state_path=path).is_key_invalid("abc")

    def test_learned_minute_limits_replace_the_guess_and_persist(self, tmp_path: Path) -> None:
        path = tmp_path / "q.json"
        manager = GeminiQuotaManager(state_path=path)
        before = manager.get_remaining(MODEL)["rpm_limit"]
        assert manager.learn_minute_limit_from_server(MODEL, rpm=before + 7, tpm=123_456)
        assert not manager.learn_minute_limit_from_server(MODEL, rpm=before + 7)
        again = GeminiQuotaManager(state_path=path).get_remaining(MODEL)
        assert again["rpm_limit"] == before + 7 and again["tpm_limit"] == 123_456
        assert not manager.learn_minute_limit_from_server(MODEL)


class TestErrorParsing:
    def test_real_invalid_key_error_is_recognised_and_other_400s_are_not(self) -> None:
        is_bad = GeminiSubtitleTranslator._is_invalid_key_error
        assert is_bad(RuntimeError(REAL_INVALID_KEY))
        assert is_bad(RuntimeError("403 PERMISSION_DENIED. Your API key was reported as leaked."))
        assert is_bad(RuntimeError("400 API key expired. Please renew the API key."))
        assert not is_bad(RuntimeError("400 INVALID_ARGUMENT. Request contains an invalid argument."))
        assert not is_bad(RuntimeError("403 PERMISSION_DENIED. You do not have permission to access the File"))
        assert not is_bad(RuntimeError("429 RESOURCE_EXHAUSTED quota"))

    def test_real_429_exposes_the_minute_limit(self) -> None:
        found = GeminiSubtitleTranslator._QUOTA_VIOLATION_RE.findall(REAL_RPM_429)
        assert found == [("GenerateRequestsPerMinutePerProjectPerModel-FreeTier", "15")]


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    return install_fake_clock(monkeypatch)


class _PerKeyServer:
    """Gemini giả theo TỪNG key: key hỏng thì 400, key bận thì trả lời, ghi lại key đã dùng."""

    def __init__(self, bad: set[str] = frozenset(), errors: dict[str, str] | None = None) -> None:
        self.bad = set(bad)
        self.errors = errors or {}
        self.calls: list[str] = []

    def client(self, key: str):
        server = self

        class _Models:
            def generate_content(self, model, contents, config):
                server.calls.append(key)
                if key in server.bad:
                    raise RuntimeError(REAL_INVALID_KEY)
                if key in server.errors:
                    raise RuntimeError(server.errors[key])
                return SimpleNamespace(text='{"subtitles": []}', usage_metadata=None)

        return SimpleNamespace(models=_Models())


def _pool(monkeypatch, clock, tmp_path, server, keys="KEY_A\nKEY_B\nKEY_C", limits=None):
    manager = GeminiQuotaManager(rate_limits=limits, state_path=tmp_path / "q.json")
    translator, slept = build_translator(monkeypatch, clock, manager, server, keys=keys)
    monkeypatch.setattr(
        translator, "_client_for_index", lambda i: server.client(translator._api_keys[i]))
    return translator, manager, slept


class TestInvalidKeyIsQuarantined:
    def test_one_bad_key_no_longer_fails_the_call(self, monkeypatch, clock, tmp_path) -> None:
        server = _PerKeyServer(bad={"KEY_A"})
        translator, manager, _ = _pool(monkeypatch, clock, tmp_path, server)
        assert call_gemini(translator, MODEL) == {"subtitles": []}
        assert server.calls == ["KEY_A", "KEY_B"]               # chỉ MỘT lượt phí cho key hỏng
        assert manager.is_key_invalid(GeminiQuotaManager.key_fingerprint("KEY_A"))

    def test_bad_key_is_not_tried_again_by_later_calls(self, monkeypatch, clock, tmp_path) -> None:
        server = _PerKeyServer(bad={"KEY_A"})
        translator, _, _ = _pool(monkeypatch, clock, tmp_path, server)
        for _ in range(4):
            call_gemini(translator, MODEL)
        assert server.calls.count("KEY_A") == 1

    def test_all_keys_bad_raises_a_clear_error(self, monkeypatch, clock, tmp_path) -> None:
        server = _PerKeyServer(bad={"KEY_A", "KEY_B"})
        translator, _, _ = _pool(monkeypatch, clock, tmp_path, server, keys="KEY_A\nKEY_B")
        with pytest.raises(SubtitleTranslationError, match="API key"):
            call_gemini(translator, MODEL)
        assert server.calls == ["KEY_A", "KEY_B"]

    def test_single_bad_key_fails_fast_without_retries(self, monkeypatch, clock, tmp_path) -> None:
        server = _PerKeyServer(bad={"KEY_A"})
        translator, _, slept = _pool(monkeypatch, clock, tmp_path, server, keys="KEY_A")
        with pytest.raises(SubtitleTranslationError):
            call_gemini(translator, MODEL)
        assert server.calls == ["KEY_A"] and slept == []

    def test_rotation_helpers_skip_a_bad_key(self, monkeypatch, clock, tmp_path) -> None:
        server = _PerKeyServer()
        translator, manager, _ = _pool(monkeypatch, clock, tmp_path, server)
        manager.mark_key_invalid(GeminiQuotaManager.key_fingerprint("KEY_B"), "x")
        assert translator._rpd_remaining_for_key(MODEL, "KEY_B") == 0
        assert translator._best_available_key_index(MODEL) in (0, 2)


class TestSmartKeyChoice:
    def test_rotates_instead_of_waiting_when_the_current_key_is_rate_limited(
        self, monkeypatch, clock, tmp_path
    ) -> None:
        """Trước đây: key #1 chạm RPM → ngồi chờ tới 60 giây dù hai key kia trống."""
        server = _PerKeyServer()
        limits = {MODEL: RateLimit(rpm=2, tpm=250_000, rpd=500)}
        translator, _, slept = _pool(monkeypatch, clock, tmp_path, server, limits=limits)
        for _ in range(6):
            call_gemini(translator, MODEL)
        assert server.calls == ["KEY_A", "KEY_A", "KEY_B", "KEY_B", "KEY_C", "KEY_C"]
        assert clock.now == FakeClock().now                       # không chờ giây nào
        assert slept == []

    def test_waits_only_when_every_key_is_busy(self, monkeypatch, clock, tmp_path) -> None:
        server = _PerKeyServer()
        limits = {MODEL: RateLimit(rpm=1, tpm=250_000, rpd=500)}
        translator, _, _ = _pool(monkeypatch, clock, tmp_path, server, keys="KEY_A\nKEY_B", limits=limits)
        for _ in range(3):
            call_gemini(translator, MODEL)
        assert server.calls[:2] == ["KEY_A", "KEY_B"]
        assert clock.now - FakeClock().now >= 59                  # lượt 3 phải chờ cửa sổ phút

    def test_video_calls_keep_their_key(self, monkeypatch, clock, tmp_path) -> None:
        server = _PerKeyServer()
        limits = {MODEL: RateLimit(rpm=1, tpm=250_000, rpd=500)}
        translator, _, _ = _pool(monkeypatch, clock, tmp_path, server, limits=limits)
        translator._video_handle_cache["x"] = object()
        call_gemini(translator, MODEL)
        call_gemini(translator, MODEL)
        assert set(server.calls) == {"KEY_A"}                     # handle video gắn theo key

    def test_single_key_behaviour_is_unchanged(self, monkeypatch, clock, tmp_path) -> None:
        server = _PerKeyServer()
        translator, _, _ = _pool(monkeypatch, clock, tmp_path, server, keys="KEY_A")
        call_gemini(translator, MODEL)
        assert server.calls == ["KEY_A"]

    def test_daily_load_is_spread_when_one_key_is_much_more_used(
        self, monkeypatch, clock, tmp_path
    ) -> None:
        server = _PerKeyServer()
        translator, manager, _ = _pool(monkeypatch, clock, tmp_path, server)
        key_a = GeminiQuotaManager.key_fingerprint("KEY_A")
        for _ in range(200):
            manager.acquire(MODEL, 10, key_id=key_a)               # key #1 đã dùng 200/500
        call_gemini(translator, MODEL)
        assert server.calls[0] != "KEY_A"


class TestPinnedKeyAccounting:
    def test_daily_exhaustion_marks_the_key_that_was_actually_used(
        self, monkeypatch, clock, tmp_path
    ) -> None:
        from tests.unit.infrastructure.gemini_call_harness import real_daily_429

        server = _PerKeyServer(errors={"KEY_A": real_daily_429()})
        translator, manager, _ = _pool(monkeypatch, clock, tmp_path, server, keys="KEY_A\nKEY_B")
        assert call_gemini(translator, "gemini-3.8-flash") == {"subtitles": []}
        assert manager.is_server_daily_blocked("gemini-3.8-flash", key_id=GeminiQuotaManager.key_fingerprint("KEY_A"))
        assert not manager.is_server_daily_blocked("gemini-3.8-flash", key_id=GeminiQuotaManager.key_fingerprint("KEY_B"))
        assert server.calls == ["KEY_A", "KEY_B"]

    def test_429_per_minute_teaches_the_real_rpm(self, monkeypatch, clock, tmp_path) -> None:
        server = _PerKeyServer(errors={"KEY_A": REAL_RPM_429})
        translator, manager, _ = _pool(monkeypatch, clock, tmp_path, server, keys="KEY_A\nKEY_B")
        call_gemini(translator, MODEL)
        assert manager.get_remaining(MODEL)["rpm_limit"] == 15


class TestInvalidKeyWithoutQuotaManager:
    """Hồi quy tìm được khi chạy bộ test cũ: adapter KHÔNG có bộ quản lý quota + key sai
    → lỗi 400 API_KEY_INVALID bị coi là "đổi key rồi thử lại" mãi mãi (vòng lặp vô tận)."""

    def test_single_bad_key_fails_fast(self, monkeypatch, clock) -> None:
        server = _PerKeyServer(bad={"KEY_A"})
        translator, _ = build_translator(monkeypatch, clock, None, server, keys="KEY_A")
        monkeypatch.setattr(translator, "_client_for_index", lambda i: server.client(translator._api_keys[i]))
        with pytest.raises(SubtitleTranslationError):
            call_gemini(translator, MODEL)
        assert server.calls == ["KEY_A"]

    def test_bad_key_then_good_key_without_manager(self, monkeypatch, clock) -> None:
        server = _PerKeyServer(bad={"KEY_A"})
        translator, _ = build_translator(monkeypatch, clock, None, server, keys="KEY_A\nKEY_B")
        monkeypatch.setattr(translator, "_client_for_index", lambda i: server.client(translator._api_keys[i]))
        assert call_gemini(translator, MODEL) == {"subtitles": []}
        assert server.calls == ["KEY_A", "KEY_B"]

    def test_all_bad_keys_without_manager_stop_after_one_try_each(self, monkeypatch, clock) -> None:
        server = _PerKeyServer(bad={"KEY_A", "KEY_B", "KEY_C"})
        translator, _ = build_translator(monkeypatch, clock, None, server, keys="KEY_A\nKEY_B\nKEY_C")
        monkeypatch.setattr(translator, "_client_for_index", lambda i: server.client(translator._api_keys[i]))
        with pytest.raises(SubtitleTranslationError):
            call_gemini(translator, MODEL)
        assert sorted(server.calls) == ["KEY_A", "KEY_B", "KEY_C"]
