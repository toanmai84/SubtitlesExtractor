"""[v3.23.642] Trước khi coi một model là HẾT trên mọi key (rồi chuyển sang model dự phòng), ứng dụng
phải thử mọi key mà SERVER chưa từ chối.

Nhật ký thật 21/10: gemini-3.8-flash (20 request/ngày) — server từ chối key …0d53 lúc ứng dụng mới đếm
18/20; ngay sau đó ứng dụng kết luận "hết ở mọi API key (server xác nhận)" mà KHÔNG gửi thử key nào
khác, vì các key kia có bộ đếm nội bộ >= 20. Bộ đếm nội bộ chỉ là ƯỚC LƯỢNG (xem v3.23.526), nên
key có bộ đếm đầy nhưng server chưa từ chối vẫn phải được thử một lần.
"""

from __future__ import annotations

import pytest

from subtitles_extractor.infrastructure.translation.gemini_quota_manager import (
    GeminiQuotaManager,
    QuotaExhaustedError,
    RateLimit,
)
from tests.unit.infrastructure.gemini_call_harness import (
    REAL_DAILY_429,
    FakeClock,
    call_gemini,
    install_fake_clock,
)
from tests.unit.infrastructure.test_key_pool_v323639 import _pool, _PerKeyServer

MODEL = "gemini-3.8-flash"
LIMITS = {MODEL: RateLimit(5, 250_000, 20)}


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    return install_fake_clock(monkeypatch)


def _fill(manager: GeminiQuotaManager, key: str, count: int) -> None:
    model_key = manager._compose_key(GeminiQuotaManager.key_fingerprint(key), MODEL)
    with manager._lock:
        manager._daily_locked(model_key)["count"] = count


def _setup(monkeypatch, clock, tmp_path, server, soft_counts):
    translator, manager, _ = _pool(monkeypatch, clock, tmp_path, server, limits=LIMITS)
    for key, count in soft_counts.items():
        _fill(manager, key, count)
    return translator, manager


def test_soft_exhausted_key_is_probed_and_used_when_the_server_accepts(
    monkeypatch, clock, tmp_path
) -> None:
    server = _PerKeyServer(errors={"KEY_A": REAL_DAILY_429})
    translator, manager = _setup(monkeypatch, clock, tmp_path, server, {"KEY_B": 20, "KEY_C": 20})
    assert call_gemini(translator, MODEL) == {"subtitles": []}
    assert server.calls == ["KEY_A", "KEY_B"]          # B thử sau khi A bị từ chối, C không cần


def test_every_unconfirmed_key_is_probed_exactly_once_before_giving_up(
    monkeypatch, clock, tmp_path
) -> None:
    server = _PerKeyServer(errors={k: REAL_DAILY_429 for k in ("KEY_A", "KEY_B", "KEY_C")})
    translator, manager = _setup(monkeypatch, clock, tmp_path, server, {"KEY_B": 20, "KEY_C": 25})
    with pytest.raises(QuotaExhaustedError):
        call_gemini(translator, MODEL)
    assert sorted(server.calls) == ["KEY_A", "KEY_B", "KEY_C"]
    for key in ("KEY_A", "KEY_B", "KEY_C"):            # lúc này MỌI key đều do server xác nhận
        assert manager.is_server_daily_blocked(MODEL, key_id=GeminiQuotaManager.key_fingerprint(key))
    assert not translator.has_any_daily_quota(MODEL)
    server.calls.clear()
    with pytest.raises(QuotaExhaustedError):
        call_gemini(translator, MODEL)
    assert server.calls == []                           # đã xác nhận → không thử lại trong ngày


def test_has_any_daily_quota_counts_unconfirmed_full_counters(monkeypatch, clock, tmp_path) -> None:
    server = _PerKeyServer()
    translator, manager = _setup(
        monkeypatch, clock, tmp_path, server, {"KEY_A": 20, "KEY_B": 20, "KEY_C": 20}
    )
    assert translator.has_any_daily_quota(MODEL)       # chưa key nào bị server từ chối
    for key in ("KEY_A", "KEY_B", "KEY_C"):
        manager.mark_daily_exhausted(MODEL, key_id=GeminiQuotaManager.key_fingerprint(key))
    assert not translator.has_any_daily_quota(MODEL)


def test_key_with_real_remaining_quota_is_preferred_over_a_probe(monkeypatch, clock, tmp_path) -> None:
    server = _PerKeyServer(errors={"KEY_A": REAL_DAILY_429})
    translator, manager = _setup(monkeypatch, clock, tmp_path, server, {"KEY_B": 20, "KEY_C": 3})
    call_gemini(translator, MODEL)
    assert server.calls == ["KEY_A", "KEY_C"]


def test_invalid_key_is_never_probed(monkeypatch, clock, tmp_path) -> None:
    server = _PerKeyServer(errors={"KEY_A": REAL_DAILY_429})
    translator, manager = _setup(monkeypatch, clock, tmp_path, server, {"KEY_B": 20, "KEY_C": 20})
    manager.mark_key_invalid(GeminiQuotaManager.key_fingerprint("KEY_B"), "hỏng")
    call_gemini(translator, MODEL)
    assert server.calls == ["KEY_A", "KEY_C"]


def test_ensure_viable_key_with_only_unconfirmed_full_counters_does_not_crash(
    monkeypatch, clock, tmp_path
) -> None:
    server = _PerKeyServer()
    translator, manager = _setup(
        monkeypatch, clock, tmp_path, server, {"KEY_A": 20, "KEY_B": 25, "KEY_C": 22}
    )
    assert translator.ensure_viable_key(MODEL, 6)
    assert translator._best_available_key_index(MODEL) is not None


def test_random_server_behaviour_always_terminates_and_never_retries_a_confirmed_key(
    monkeypatch, clock, tmp_path
) -> None:
    """Mô phỏng ngẫu nhiên: mỗi key có trạng thái server riêng; mỗi key bị từ chối vì hết ngày tối đa
    MỘT lần, mọi lời gọi hoặc thành công hoặc ném đúng QuotaExhaustedError (không treo, không lỗi lạ)."""
    import random

    from subtitles_extractor.domain.ports.subtitle_translator_port import SubtitleTranslationError

    for seed in range(40):
        rng = random.Random(seed)
        dead = {k for k in ("KEY_A", "KEY_B", "KEY_C") if rng.random() < 0.6}
        server = _PerKeyServer(errors={k: REAL_DAILY_429 for k in dead})
        sub = tmp_path / str(seed)
        sub.mkdir()
        translator, manager = _setup(
            monkeypatch, clock, sub, server,
            {k: rng.choice([0, 5, 19, 20, 30]) for k in ("KEY_A", "KEY_B", "KEY_C")},
        )
        for _ in range(4):
            try:
                call_gemini(translator, MODEL)
                assert len(dead) < 3
            except QuotaExhaustedError:
                assert len(dead) == 3
            except SubtitleTranslationError:  # pragma: no cover - không mong đợi
                raise
        for key in dead:
            assert server.calls.count(key) <= 1, (seed, key, server.calls)
