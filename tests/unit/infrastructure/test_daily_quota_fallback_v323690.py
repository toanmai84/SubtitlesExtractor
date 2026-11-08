"""[v3.23.690] Model CHÍNH hết hạn mức NGÀY ở mọi key → chuyển sang model dự phòng (nếu bật), không dừng cả lượt.

Ô "Model dự phòng" hứa: "khi model đang dùng quá tải (503) HOẶC hết hạn mức ngày, ứng dụng chuyển sang model dự phòng".
Trước bản này chỉ quá tải mới chuyển; hết hạn mức ngày của model chính thì ném lỗi dừng. Phim bộ 86 tập: phân tích mỗi
tập ~2 request bằng gemini-3.8-flash (20 request/ngày/key) → 3 key cạn sau ~30 tập và "Dịch cả bộ" đứng lại tới hôm sau.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from subtitles_extractor.infrastructure.translation.gemini_quota_manager import (
    GeminiQuotaManager,
    QuotaExhaustedError,
)
from subtitles_extractor.infrastructure.translation.model_health import NO_FALLBACK, ModelHealth
from tests.unit.infrastructure.gemini_call_harness import (
    OK,
    REAL_DAILY_429,
    SCARCE_MODEL,
    FakeClock,
    ScriptedGemini,
    build_translator,
    call_gemini,
    confirmed_scarce_manager,
    install_fake_clock,
)

FALLBACK = "gemini-3.1-flash-lite"


class RecordingGemini(ScriptedGemini):
    def __init__(self, script: list[str]) -> None:
        super().__init__(script)
        self.models: list[str] = []

    def generate_content(self, model: str, contents: list, config: object) -> object:
        self.models.append(model)
        return super().generate_content(model, contents, config)


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    return install_fake_clock(monkeypatch)


def _translator(monkeypatch, clock, manager, server, keys="KEY_A"):
    translator, _slept = build_translator(monkeypatch, clock, manager, server, keys=keys)
    translator._model_health = ModelHealth(seed_models=(SCARCE_MODEL, FALLBACK))
    return translator


def test_server_daily_429_on_last_key_switches_to_fallback(monkeypatch, clock, tmp_path: Path) -> None:
    server = RecordingGemini([REAL_DAILY_429, OK])
    translator = _translator(monkeypatch, clock, confirmed_scarce_manager(tmp_path), server)
    assert call_gemini(translator, SCARCE_MODEL) == {"subtitles": []}
    assert server.models == [SCARCE_MODEL, FALLBACK]


def test_local_counter_exhausted_goes_straight_to_fallback(monkeypatch, clock, tmp_path: Path) -> None:
    manager = confirmed_scarce_manager(tmp_path)
    for key in ("KEY_A", "KEY_B"):
        manager.mark_daily_exhausted(SCARCE_MODEL, key_id=GeminiQuotaManager.key_fingerprint(key))
    server = RecordingGemini([OK])
    translator = _translator(monkeypatch, clock, manager, server, keys="KEY_A\nKEY_B")
    assert call_gemini(translator, SCARCE_MODEL) == {"subtitles": []}
    assert server.models == [FALLBACK]                     # không tốn request nào cho model đã cạn


def test_fallback_off_still_stops(monkeypatch, clock, tmp_path: Path) -> None:
    server = RecordingGemini([REAL_DAILY_429, OK])
    translator = _translator(monkeypatch, clock, confirmed_scarce_manager(tmp_path), server)
    translator._model_health.set_preferred_fallback(NO_FALLBACK)
    with pytest.raises(QuotaExhaustedError):
        call_gemini(translator, SCARCE_MODEL)
    assert server.models == [SCARCE_MODEL]


def test_fallback_also_exhausted_stops_without_looping(monkeypatch, clock, tmp_path: Path) -> None:
    server = RecordingGemini([REAL_DAILY_429])                 # mọi lượt đều hết hạn mức ngày
    translator = _translator(monkeypatch, clock, confirmed_scarce_manager(tmp_path), server)
    with pytest.raises(QuotaExhaustedError):
        call_gemini(translator, SCARCE_MODEL)
    assert server.models == [SCARCE_MODEL, FALLBACK]
