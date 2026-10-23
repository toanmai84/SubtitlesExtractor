"""[v3.23.408] Test _coerce_voice_id — worker ép mọi dạng giọng về voice_id chuỗi."""

from __future__ import annotations

import sys
import types

import pytest


@pytest.fixture(autouse=True)
def _fake_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dựng torch giả để import được module subprocess (không cần torch thật)."""
    for name in ("torch", "torch.nn", "torch.nn.functional", "torch._C"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["torch"].nn = sys.modules["torch.nn"]
    sys.modules["torch.nn"].functional = sys.modules["torch.nn.functional"]


def _coerce():
    from subtitles_extractor.infrastructure.tts.vieneu_gpu_subprocess import (
        _coerce_voice_id,
    )

    return _coerce_voice_id


def test_tuple_returns_voice_id() -> None:
    coerce = _coerce()
    assert coerce(("Minh Đức — Nam · Bắc", "Minh Đức")) == "Minh Đức"


def test_stringified_tuple_returns_voice_id() -> None:
    coerce = _coerce()
    assert coerce("('Minh Đức — Nam · Bắc', 'Minh Đức')") == "Minh Đức"


def test_plain_string_unchanged() -> None:
    assert _coerce()("Thái Sơn") == "Thái Sơn"


def test_list_returns_last() -> None:
    assert _coerce()(["Phạm Tuyên — Nam", "Phạm Tuyên"]) == "Phạm Tuyên"


def test_empty_returns_empty() -> None:
    assert _coerce()("") == ""
    assert _coerce()(None) == ""


def test_malformed_tuple_string_falls_back_to_text() -> None:
    # Không parse được → trả chuỗi đã strip (không ném lỗi).
    assert _coerce()("(khong-phai-tuple") == "(khong-phai-tuple"
