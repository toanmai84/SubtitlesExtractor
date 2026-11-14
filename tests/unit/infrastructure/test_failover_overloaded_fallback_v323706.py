"""[v3.23.706] Model DỰ PHÒNG cũng quá tải thì quay về model chính, không bỏ cả lượt dịch.

Nhật ký thật 10/11/2026 (A/B 361 câu): ``gemini-3.5-flash-lite`` quá tải thoáng qua → lời gọi chuyển sang
``gemini-3.1-flash-lite`` đúng lúc 3.1 sập (503 ở cả 3 key) → "đã thử 12 lượt trên 3 API key trong khoảng 11 phút"
rồi ``ModelOverloadedError`` làm hỏng cả lượt dịch, trong khi 3.5 đã trả lời bình thường (HTTP 200) từ lâu.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from subtitles_extractor.domain.ports.subtitle_translator_port import ModelOverloadedError
from subtitles_extractor.infrastructure.translation.model_health import ModelHealth
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


class _Server:
    def __init__(self, overloaded_for: dict[str, int]) -> None:
        self.overloaded_for = dict(overloaded_for)      # model → số lần 503 đầu tiên
        self.models: list[str] = []

    def generate_content(self, model: str, contents: list, config: object) -> object:
        self.models.append(model)
        if self.overloaded_for.get(model, 0) > 0:
            self.overloaded_for[model] -= 1
            raise RuntimeError(REAL_503)
        return SimpleNamespace(text='{"subtitles": []}', usage_metadata=None)


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    return install_fake_clock(monkeypatch)


def _setup(monkeypatch, clock, tmp_path: Path, server: _Server):
    translator, slept = build_translator(
        monkeypatch, clock, quota_manager(tmp_path), server, retry_count=6
    )
    translator._model_health = ModelHealth(lambda: clock.now, seed_models=[LITE_31, LITE_35])
    return translator, slept


def test_overloaded_fallback_returns_to_recovered_primary(monkeypatch, clock, tmp_path: Path) -> None:
    # 3.5 (chính) quá tải 2 lượt rồi hồi; 3.1 (dự phòng) sập hẳn — đúng ca nhật ký.
    server = _Server({LITE_35: 2, LITE_31: 99})
    translator, _ = _setup(monkeypatch, clock, tmp_path, server)
    assert call_gemini(translator, LITE_35) == {"subtitles": []}
    assert LITE_31 in server.models                       # đã thử dự phòng
    assert server.models[-1] == LITE_35                   # rồi quay về model chính và thành công
    assert server.models.count(LITE_31) <= 3              # không đốt 12 lượt trên model đang sập


def test_both_models_down_still_fails_and_is_bounded(monkeypatch, clock, tmp_path: Path) -> None:
    server = _Server({LITE_35: 999, LITE_31: 999})
    translator, _ = _setup(monkeypatch, clock, tmp_path, server)
    with pytest.raises(ModelOverloadedError):
        call_gemini(translator, LITE_35)
    assert len(server.models) < 60                        # không qua lại vô hạn
    switches = sum(1 for a, b in zip(server.models, server.models[1:]) if a != b)
    assert switches <= 2                                  # chính → dự phòng → chính, hết
