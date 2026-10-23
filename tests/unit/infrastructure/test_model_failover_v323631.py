"""[v3.23.631] Bộ ngắt mạch quá tải theo MODEL + model dự phòng; không xoay key vì quá tải khi có video.

Nhật ký thật 10/2026 (phim 86 tập, free tier, 3 API key): 142/184 phút chờ thử lại là lỗi 503 của
CÙNG một model — một chuỗi 68 lần 503 trên ``gemini-3.1-flash-lite`` kéo dài 41 phút, hai lượt
dịch hỏng hẳn sau 12 lần thử mà không model nào khác được thử. 17 lần xoay API key vì quá tải
thì chỉ 2 lần được trả lời ngay (15/17 vẫn 503), mà mỗi lần kèm video buộc tải lại đoạn (~40 s).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from subtitles_extractor.domain.ports.subtitle_translator_port import ModelOverloadedError
from subtitles_extractor.infrastructure.translation.model_health import (
    BASE_COOLDOWN_S,
    MAX_COOLDOWN_S,
    TRIP_AFTER,
    ModelHealth,
    model_tier,
    model_version,
    rank_fallbacks,
)
from tests.unit.infrastructure.gemini_call_harness import (
    REAL_503,
    FakeClock,
    build_translator,
    call_gemini,
    install_fake_clock,
    quota_manager,
)

LITE_31 = "gemini-3.1-flash-lite"
LITE_35 = "gemini-3.5-flash-lite"
FLASH_38 = "gemini-3.8-flash"
FLASH_25 = "gemini-2.5-flash"


class _Time:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class TestModelHealth:
    def test_trips_after_consecutive_overloads_then_cools(self) -> None:
        clock = _Time()
        health = ModelHealth(clock)
        assert [health.record_overload(LITE_31) for _ in range(TRIP_AFTER - 1)] == [0.0] * (TRIP_AFTER - 1)
        assert health.cooling_remaining(LITE_31) == 0
        assert health.record_overload(LITE_31) == BASE_COOLDOWN_S
        assert health.cooling_remaining(LITE_31) == BASE_COOLDOWN_S
        clock.now += BASE_COOLDOWN_S + 1
        assert health.cooling_remaining(LITE_31) == 0

    def test_success_resets_everything(self) -> None:
        health = ModelHealth(_Time())
        health.record_overload(LITE_31)
        health.record_overload(LITE_31)
        health.record_success(LITE_31)
        assert health.record_overload(LITE_31) == 0.0       # đếm lại từ 1

    def test_probe_failure_after_cooldown_retrips_with_double_time(self) -> None:
        clock = _Time()
        health = ModelHealth(clock)
        for _ in range(TRIP_AFTER):
            health.record_overload(LITE_31)
        clock.now += BASE_COOLDOWN_S + 1
        assert health.record_overload(LITE_31) == BASE_COOLDOWN_S * 2   # lượt thăm dò hỏng
        clock.now += BASE_COOLDOWN_S * 2 + 1
        assert health.record_overload(LITE_31) == BASE_COOLDOWN_S * 4

    def test_cooldown_is_capped(self) -> None:
        clock = _Time()
        health = ModelHealth(clock)
        for _ in range(TRIP_AFTER):
            health.record_overload(LITE_31)
        for _ in range(10):
            clock.now += MAX_COOLDOWN_S + 1
            health.record_overload(LITE_31)
        assert health.cooling_remaining(LITE_31) <= MAX_COOLDOWN_S

    def test_parallel_failures_during_cooldown_do_not_compound(self) -> None:
        health = ModelHealth(_Time())
        for _ in range(TRIP_AFTER):
            health.record_overload(LITE_31)
        assert health.record_overload(LITE_31) == 0.0
        assert health.cooling_remaining(LITE_31) == BASE_COOLDOWN_S

    def test_models_are_independent_and_known_models_replace(self) -> None:
        health = ModelHealth(_Time(), seed_models=[LITE_31])
        for _ in range(TRIP_AFTER):
            health.record_overload(LITE_31)
        assert health.cooling_remaining(LITE_35) == 0
        assert health.known_models == (LITE_31,)
        health.set_known_models([LITE_35, FLASH_38])
        assert health.known_models == (LITE_35, FLASH_38)
        health.set_known_models([])                       # danh sách rỗng không xoá cái đang biết
        assert health.known_models == (LITE_35, FLASH_38)


class TestRanking:
    def test_helpers(self) -> None:
        assert model_tier(LITE_31) == "lite" and model_tier(FLASH_38) == "flash"
        assert model_version(LITE_35) == 3.5 and model_version("gemini-flash-latest") == 0.0

    def test_same_tier_then_higher_daily_limit_then_newer(self) -> None:
        known = [FLASH_38, LITE_31, LITE_35, FLASH_25, "gemini-3.7-flash-lite"]
        limits = {LITE_31: 500, LITE_35: 20, "gemini-3.7-flash-lite": 20, FLASH_38: 20, FLASH_25: 20}
        ranked = rank_fallbacks(
            "gemini-3.6-flash-lite", known, daily_limit=lambda m: limits.get(m, 0)
        )
        assert ranked[:3] == [LITE_31, "gemini-3.7-flash-lite", LITE_35]
        assert ranked[3:] == [FLASH_38, FLASH_25]

    def test_excludes_self_cooling_and_unavailable(self) -> None:
        ranked = rank_fallbacks(
            LITE_31, [LITE_31, LITE_35, FLASH_38, FLASH_25],
            cooling=lambda m: 30.0 if m == LITE_35 else 0.0,
            available=lambda m: m != FLASH_38,
        )
        assert ranked == [FLASH_25]

    def test_preview_comes_after_stable_even_with_a_higher_guessed_limit(self) -> None:
        ranked = rank_fallbacks(
            LITE_31, ["gemini-3.1-flash-lite-preview", LITE_35],
            daily_limit=lambda m: 500 if "preview" in m else 20,
        )
        assert ranked == [LITE_35, "gemini-3.1-flash-lite-preview"]

    def test_real_api_list_never_offers_audio_live_or_alias_models(self) -> None:
        """Danh sách free-tier THẬT lấy từ API (18 model) có cả native-audio, live, omni, -latest."""
        real = [
            "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-flash-native-audio-latest",
            "gemini-2.5-flash-native-audio-preview-09-2025", "gemini-3-flash-preview",
            "gemini-3.1-flash-lite", "gemini-3.1-flash-lite-preview",
            "gemini-3.1-flash-live-preview", "gemini-3.5-flash", "gemini-3.5-flash-lite",
            "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash", "gemini-flash-latest",
            "gemini-flash-lite-latest", "gemini-omni-1.1-flash", "gemini-omni-flash-preview",
        ]
        chain = rank_fallbacks("gemini-3.8-flash", real)
        assert chain[:4] == ["gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash", "gemini-2.5-flash"]
        assert not any(
            marker in name for name in chain
            for marker in ("native-audio", "live", "omni", "-latest")
        )
        lite = rank_fallbacks(LITE_31, real)
        assert lite[:2] == [LITE_35, "gemini-2.5-flash-lite"]


class _ModelServer:
    """Gemini giả: model nào quá tải là chuyện riêng của model đó."""

    def __init__(self, overloaded: set[str]) -> None:
        self.overloaded = overloaded
        self.models: list[str] = []
        self.keys: list[str] = []
        self.translator = None

    def generate_content(self, model: str, contents: list, config: object) -> object:
        self.models.append(model)
        self.keys.append(self.translator._api_key)
        if model in self.overloaded:
            raise RuntimeError(REAL_503)
        return SimpleNamespace(text='{"subtitles": []}', usage_metadata=None)


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    return install_fake_clock(monkeypatch)


def _setup(monkeypatch, clock, tmp_path: Path, overloaded: set[str], known, **kw):
    server = _ModelServer(overloaded)
    translator, slept = build_translator(
        monkeypatch, clock, quota_manager(tmp_path), server, **kw
    )
    server.translator = translator
    health_clock = _Time()
    translator._model_health = ModelHealth(health_clock, seed_models=known)
    return translator, server, slept, health_clock


class TestFailover:
    def test_switches_after_two_overloads_instead_of_waiting(
        self, monkeypatch, clock, tmp_path: Path
    ) -> None:
        translator, server, slept, _ = _setup(
            monkeypatch, clock, tmp_path, {LITE_31}, [LITE_31, LITE_35]
        )
        assert call_gemini(translator, LITE_31) == {"subtitles": []}
        assert server.models == [LITE_31, LITE_31, LITE_35]
        assert slept == [30.0]            # MỘT lần chờ, không phải 5-11 lượt
        assert translator._model_health.cooling_remaining(LITE_35) == 0

    def test_without_health_the_old_behaviour_is_kept(
        self, monkeypatch, clock, tmp_path: Path
    ) -> None:
        translator, server, _slept, _ = _setup(
            monkeypatch, clock, tmp_path, {LITE_31}, [LITE_31, LITE_35]
        )
        translator._model_health = None
        with pytest.raises(ModelOverloadedError):
            call_gemini(translator, LITE_31)
        assert set(server.models) == {LITE_31}

    def test_tripped_model_is_skipped_by_the_next_call(
        self, monkeypatch, clock, tmp_path: Path
    ) -> None:
        translator, server, slept, _ = _setup(
            monkeypatch, clock, tmp_path, {LITE_31}, [LITE_31, LITE_35]
        )
        for _ in range(TRIP_AFTER):
            translator._model_health.record_overload(LITE_31)
        assert call_gemini(translator, LITE_31) == {"subtitles": []}
        assert server.models == [LITE_35] and slept == []   # sang dự phòng NGAY

    def test_recovered_model_is_used_again(self, monkeypatch, clock, tmp_path: Path) -> None:
        translator, server, _slept, health_clock = _setup(
            monkeypatch, clock, tmp_path, set(), [LITE_31, LITE_35]
        )
        for _ in range(TRIP_AFTER):
            translator._model_health.record_overload(LITE_31)
        health_clock.now += BASE_COOLDOWN_S + 1
        call_gemini(translator, LITE_31)
        assert server.models == [LITE_31]
        assert translator._model_health.record_overload(LITE_31) == 0.0   # đã bình phục

    def test_chain_goes_on_to_a_third_model(self, monkeypatch, clock, tmp_path: Path) -> None:
        translator, server, _slept, _ = _setup(
            monkeypatch, clock, tmp_path, {LITE_31, LITE_35}, [LITE_31, LITE_35, FLASH_38]
        )
        call_gemini(translator, LITE_31)
        assert server.models == [LITE_31, LITE_31, LITE_35, LITE_35, FLASH_38]

    def test_nobody_left_means_the_normal_retry_then_overload_error(
        self, monkeypatch, clock, tmp_path: Path
    ) -> None:
        translator, server, _slept, _ = _setup(
            monkeypatch, clock, tmp_path, {LITE_31, LITE_35}, [LITE_31, LITE_35]
        )
        with pytest.raises(ModelOverloadedError):
            call_gemini(translator, LITE_31)
        assert set(server.models) == {LITE_31, LITE_35}

    def test_exhausted_daily_quota_models_are_not_chosen(
        self, monkeypatch, clock, tmp_path: Path
    ) -> None:
        translator, server, _slept, _ = _setup(
            monkeypatch, clock, tmp_path, {LITE_31}, [LITE_31, LITE_35, FLASH_38]
        )
        manager = translator._quota_manager
        manager.set_limit(LITE_35, rpm=10, tpm=250_000, rpd=1)
        manager.acquire(LITE_35, 10, key_id=manager.key_fingerprint("KEY_A"))
        assert manager.get_remaining(LITE_35, key_id=manager.key_fingerprint("KEY_A"))["rpd_remaining"] == 0
        call_gemini(translator, LITE_31)
        assert server.models[-1] == FLASH_38 and LITE_35 not in server.models

    def test_thinking_is_stripped_only_when_the_family_changes(
        self, monkeypatch, clock, tmp_path: Path
    ) -> None:
        translator, server, _slept, _ = _setup(
            monkeypatch, clock, tmp_path, {LITE_31}, [LITE_31, "gemini-2.5-flash-lite"]
        )
        stripped: list[object] = []
        monkeypatch.setattr(
            translator, "_strip_thinking_from_config",
            lambda config: stripped.append(config) or config,
        )
        call_gemini(translator, LITE_31)
        assert server.models[-1] == "gemini-2.5-flash-lite" and len(stripped) == 1

        translator2, server2, _s, _ = _setup(
            monkeypatch, clock, tmp_path, {LITE_31}, [LITE_31, LITE_35]
        )
        stripped2: list[object] = []
        monkeypatch.setattr(
            translator2, "_strip_thinking_from_config",
            lambda config: stripped2.append(config) or config,
        )
        call_gemini(translator2, LITE_31)
        assert stripped2 == []

    def test_success_clears_the_overload_count(self, monkeypatch, clock, tmp_path: Path) -> None:
        translator, _server, _slept, _ = _setup(
            monkeypatch, clock, tmp_path, set(), [LITE_31, LITE_35]
        )
        translator._model_health.record_overload(LITE_31)
        translator._model_health.record_overload(LITE_31)
        call_gemini(translator, LITE_31)
        assert translator._model_health.record_overload(LITE_31) == 0.0


class TestNoKeyRotationWithVideo:
    KEYS = "KEY_A\nKEY_B\nKEY_C"

    def test_overload_with_video_stays_on_the_key(
        self, monkeypatch, clock, tmp_path: Path
    ) -> None:
        translator, server, _slept, _ = _setup(
            monkeypatch, clock, tmp_path, {LITE_31}, [LITE_31], keys=self.KEYS
        )
        translator._model_health = None                       # chỉ xét chính sách xoay key
        handle = SimpleNamespace(uri="files/a", mime_type="video/mp4")
        with pytest.raises(ModelOverloadedError):
            call_gemini(translator, LITE_31, video_files=[handle])
        assert set(server.keys) == {"KEY_A"}                  # không xoay → không phải tải lại video

    def test_overload_without_video_still_rotates(
        self, monkeypatch, clock, tmp_path: Path
    ) -> None:
        translator, server, _slept, _ = _setup(
            monkeypatch, clock, tmp_path, {LITE_31}, [LITE_31], keys=self.KEYS
        )
        translator._model_health = None
        with pytest.raises(ModelOverloadedError):
            call_gemini(translator, LITE_31)
        assert len(set(server.keys)) > 1
