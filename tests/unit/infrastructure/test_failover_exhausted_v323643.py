"""[v3.23.643] Model DỰ PHÒNG hết hạn mức ngày không được làm hỏng cả lượt dịch.

Nhật ký thật 21/10 (phim 86 tập): ``gemini-3.1-flash-lite`` quá tải (503) → dự phòng ``3.5-flash-lite`` →
``3.1-flash-lite-preview`` → ``gemini-3.7-flash`` (20 request/ngày). Server báo ``3.7-flash`` hết hạn mức
ngày → ``QuotaExhaustedError`` thoát khỏi vòng thử lại và DỪNG cả giai đoạn dịch (sau 1 giờ 40), trong khi
``3.1-flash-lite`` và ``3.5-flash-lite`` vẫn còn hạn mức (chỉ đang quá tải tạm thời).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from subtitles_extractor.domain.ports.subtitle_translator_port import ModelOverloadedError
from subtitles_extractor.infrastructure.translation.gemini_quota_manager import QuotaExhaustedError
from subtitles_extractor.infrastructure.translation.model_health import ModelHealth
from tests.unit.infrastructure.gemini_call_harness import (
    REAL_503,
    FakeClock,
    build_translator,
    call_gemini,
    install_fake_clock,
    quota_manager,
    real_daily_429,
)

LITE_31 = "gemini-3.1-flash-lite"
LITE_35 = "gemini-3.5-flash-lite"
FLASH_37 = "gemini-3.7-flash"


class _Server:
    def __init__(self, overloaded_for: dict[str, int], daily_dead: set[str]) -> None:
        self.overloaded_for = dict(overloaded_for)      # model → số lần 503 đầu tiên
        self.daily_dead = daily_dead
        self.models: list[str] = []

    def generate_content(self, model: str, contents: list, config: object) -> object:
        self.models.append(model)
        if model in self.daily_dead:
            raise RuntimeError(real_daily_429())
        if self.overloaded_for.get(model, 0) > 0:
            self.overloaded_for[model] -= 1
            raise RuntimeError(REAL_503)
        return SimpleNamespace(text='{"subtitles": []}', usage_metadata=None)


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    return install_fake_clock(monkeypatch)


def _setup(monkeypatch, clock, tmp_path: Path, server: _Server, known):
    translator, slept = build_translator(
        monkeypatch, clock, quota_manager(tmp_path), server, retry_count=6
    )
    translator._model_health = ModelHealth(lambda: clock.now, seed_models=known)
    return translator, slept


def test_exhausted_fallback_returns_to_the_primary_instead_of_aborting(
    monkeypatch, clock, tmp_path: Path
) -> None:
    # 3.1 quá tải 4 lần đầu; 3.5 quá tải mãi; 3.7 hết hạn mức ngày (đúng ca nhật ký)
    server = _Server({LITE_31: 4, LITE_35: 99}, {FLASH_37})
    translator, _ = _setup(monkeypatch, clock, tmp_path, server, [LITE_31, LITE_35, FLASH_37])
    assert call_gemini(translator, LITE_31) == {"subtitles": []}
    assert FLASH_37 in server.models                       # đã thử và bị từ chối
    assert server.models.count(FLASH_37) == 1              # chỉ MỘT lần
    assert server.models[-1] == LITE_31                    # rồi quay lại model chính và thành công


def test_exhausted_fallback_moves_on_to_the_next_candidate(
    monkeypatch, clock, tmp_path: Path
) -> None:
    server = _Server({LITE_31: 99}, {LITE_35})
    translator, _ = _setup(monkeypatch, clock, tmp_path, server, [LITE_31, LITE_35, FLASH_37])
    assert call_gemini(translator, LITE_31) == {"subtitles": []}
    assert server.models[-1] == FLASH_37


def test_primary_exhausted_still_raises(monkeypatch, clock, tmp_path: Path) -> None:
    server = _Server({}, {LITE_31})
    translator, _ = _setup(monkeypatch, clock, tmp_path, server, [LITE_31, LITE_35])
    with pytest.raises(QuotaExhaustedError):
        call_gemini(translator, LITE_31)


def test_everything_overloaded_or_exhausted_ends_in_a_clear_error_not_a_loop(
    monkeypatch, clock, tmp_path: Path
) -> None:
    server = _Server({LITE_31: 999, LITE_35: 999}, {FLASH_37})
    translator, _ = _setup(monkeypatch, clock, tmp_path, server, [LITE_31, LITE_35, FLASH_37])
    with pytest.raises((ModelOverloadedError, QuotaExhaustedError)):
        call_gemini(translator, LITE_31)
    assert len(server.models) < 60                          # kết thúc, không quay vô hạn
