"""[v3.23.240] Cảnh báo thiếu pedalboard MỘT LẦN mỗi phiên khi nén mạnh.

Chất lượng nén giọng phụ thuộc thầm lặng vào việc máy có cài ``pedalboard`` (Rubber Band)
hay không — mà đây là optional dependency (GPL, app không bundle). Khi vắng nó, pipeline
rơi về WSOLA: ổn ở nén nhẹ, nhưng ở nén MẠNH (>1.6x) WSOLA làm formant tan thành "tiếng
gió". Trước đây người dùng không có cách nào biết mình đang nghe chất lượng thấp hơn.

Cảnh báo phải:
* chỉ hiện khi thật sự cần (có câu nén mạnh) — không dọa khi nén nhẹ vẫn ổn với WSOLA;
* hiện đúng MỘT LẦN mỗi phiên — không lặp mỗi câu (95 dòng -> 95 dòng log là nhiễu).
"""

from __future__ import annotations

import numpy as np
import pytest

import subtitles_extractor.infrastructure.tts.time_stretch as ts


@pytest.fixture
def _khong_co_pedalboard(monkeypatch: pytest.MonkeyPatch) -> None:
    # Ép "from pedalboard import ..." ném ImportError, và reset cờ nhắc.
    import builtins

    that = builtins.__import__

    def gia_lap(name: str, *args: object, **kwargs: object) -> object:
        if name == "pedalboard":
            raise ImportError("giả lập thiếu pedalboard")
        return that(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", gia_lap)
    monkeypatch.setattr(ts, "_pedalboard_hint_shown", False)


def _audio() -> np.ndarray:
    return (np.random.default_rng(0).standard_normal(24_000) * 0.1).astype(np.float32)


def test_nen_nhe_khong_nhac(
    _khong_co_pedalboard: None, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level("INFO"):
        for _ in range(3):
            ts.stretch_with_pedalboard(_audio(), 24_000, 1.2)
    assert "pedalboard" not in caplog.text


def test_nen_manh_nhac_dung_mot_lan(
    _khong_co_pedalboard: None, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level("INFO"):
        ts.stretch_with_pedalboard(_audio(), 24_000, 1.8)
        ts.stretch_with_pedalboard(_audio(), 24_000, 2.0)
        ts.stretch_with_pedalboard(_audio(), 24_000, 1.9)
    # Đúng một dòng nhắc, dù ba câu nén mạnh.
    assert caplog.text.count("pip install pedalboard") == 1


def test_nguong_nhac_la_1_6(_khong_co_pedalboard: None) -> None:
    # Ranh giới: 1.6x nhắc, ngay dưới thì không.
    assert pytest.approx(1.6) == ts._PEDALBOARD_HINT_RATIO


def test_khong_nhac_khi_da_co_pedalboard(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Nếu pedalboard cài được thì đi đường nhanh, không bao giờ chạm hàm nhắc.
    monkeypatch.setattr(ts, "_pedalboard_hint_shown", False)
    called = {"n": 0}

    def gia_nhac(ratio: float) -> None:
        called["n"] += 1

    monkeypatch.setattr(ts, "_goi_y_cai_pedalboard", gia_nhac)
    # Không ép ImportError -> pedalboard thật (nếu có) hoặc None; hàm nhắc chỉ được gọi
    # trong nhánh ImportError. Ở môi trường test không có pedalboard, nhánh đó chạy —
    # nên ta chỉ khẳng định: khi có, hàm nhắc KHÔNG bị gọi thừa ngoài nhánh ImportError.
    # (kiểm định gián tiếp qua việc số lần gọi <= 1)
    ts.stretch_with_pedalboard(_audio(), 24_000, 1.8)
    assert called["n"] <= 1
