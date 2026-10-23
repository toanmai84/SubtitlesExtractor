"""[v3.23.412] Test logic cache batch TTS (không cần GPU/subprocess thật)."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from subtitles_extractor.infrastructure.tts.vieneu_tts_adapter import (
    VieNeuTtsAdapter,
)


def _make_adapter(batch_size: int) -> VieNeuTtsAdapter:
    """Tạo adapter tối giản (bỏ __init__ nặng) chỉ với thuộc tính batch cần thiết."""
    adapter = object.__new__(VieNeuTtsAdapter)
    adapter._batch_size = batch_size
    adapter._batch_texts = ["câu 0", "câu 1", "câu 2", "câu 3"]
    adapter._batch_cache = {}
    adapter._force_cpu = False
    adapter._gpu_worker_disabled = False
    return adapter


def test_batch_fills_group_and_pops(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_adapter(batch_size=2)
    calls: list[list[str]] = []

    def fake_batch(texts, kwargs):
        calls.append(texts)
        return [np.full(10, float(i), dtype=np.float32) for i in range(len(texts))]

    monkeypatch.setattr(adapter, "_infer_batch_via_gpu_worker", fake_batch)

    # idx=0 → điền nhóm [0,1], trả audio idx 0.
    a0 = adapter._get_batched_audio(0, {})
    assert a0 is not None and a0[0] == 0.0
    assert calls == [["câu 0", "câu 1"]]  # gọi batch MỘT lần cho nhóm
    # idx=1 → HIT cache (không gọi batch lần nữa).
    a1 = adapter._get_batched_audio(1, {})
    assert a1 is not None and a1[0] == 1.0
    assert len(calls) == 1
    # idx=2 → nhóm mới [2,3].
    a2 = adapter._get_batched_audio(2, {})
    assert a2 is not None
    assert calls[-1] == ["câu 2", "câu 3"]


def test_batch_failure_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_adapter(batch_size=2)
    monkeypatch.setattr(
        adapter, "_infer_batch_via_gpu_worker", lambda texts, kwargs: None
    )
    assert adapter._get_batched_audio(0, {}) is None


def test_batch_out_of_range_returns_none() -> None:
    adapter = _make_adapter(batch_size=2)
    assert adapter._get_batched_audio(99, {}) is None


def test_worker_handle_batch_writes_and_replies(
    monkeypatch: pytest.MonkeyPatch, tmp_path, capsys
) -> None:
    # torch giả để import module subprocess.
    for name in ("torch", "torch.nn", "torch.nn.functional", "torch._C"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["torch"].nn = sys.modules["torch.nn"]
    sys.modules["torch.nn"].functional = sys.modules["torch.nn.functional"]
    import importlib

    sub = importlib.import_module(
        "subtitles_extractor.infrastructure.tts.vieneu_gpu_subprocess"
    )

    written = {}

    def fake_write_wav(path, audio, sr):
        written[path] = (audio, sr)

    monkeypatch.setattr(sub, "_write_wav", fake_write_wav)

    class FakeEngine:
        def infer_batch(self, texts, **kwargs):
            return [np.ones(5, dtype=np.float32) for _ in texts]

    out0 = str(tmp_path / "b0.wav")
    out1 = str(tmp_path / "b1.wav")
    batch = [{"text": "a", "output": out0}, {"text": "b", "output": out1}]
    sub._handle_batch(FakeEngine(), batch, voice={}, supported=frozenset(),
                      sample_rate=48000, item={})
    import json

    reply = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert reply["ok"] is True
    assert reply["batch"] == [out0, out1]
    assert out0 in written and out1 in written


def test_worker_handle_batch_no_infer_batch(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    for name in ("torch", "torch.nn", "torch.nn.functional", "torch._C"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["torch"].nn = sys.modules["torch.nn"]
    sys.modules["torch.nn"].functional = sys.modules["torch.nn.functional"]
    import importlib

    sub = importlib.import_module(
        "subtitles_extractor.infrastructure.tts.vieneu_gpu_subprocess"
    )

    class NoBatchEngine:
        pass

    sub._handle_batch(NoBatchEngine(), [{"text": "a", "output": "x"}], voice={},
                      supported=frozenset(), sample_rate=48000, item={})
    import json

    reply = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert reply["ok"] is False  # SDK không có infer_batch → adapter lùi per-câu


def test_worker_no_surrogates_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    """[v3.23.414] Worker tự làm sạch lone surrogate (bảo hiểm phía nhận)."""
    for name in ("torch", "torch.nn", "torch.nn.functional", "torch._C"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["torch"].nn = sys.modules["torch.nn"]
    sys.modules["torch.nn"].functional = sys.modules["torch.nn.functional"]
    import importlib

    sub = importlib.import_module(
        "subtitles_extractor.infrastructure.tts.vieneu_gpu_subprocess"
    )
    dirty = "Cách kinh thành\udc9d Lâm An"
    clean = sub._no_surrogates(dirty)
    clean.encode("utf-8")  # không ném lỗi
    assert "\udc9d" not in clean
    assert sub._no_surrogates("Xin chào") == "Xin chào"


def test_worker_batch_passes_gpu_subbatch(monkeypatch: pytest.MonkeyPatch, tmp_path, capsys) -> None:
    """[v3.23.420] Worker truyền gpu_subbatch → infer_batch qua tên param đúng (batch_size)."""
    for name in ("torch", "torch.nn", "torch.nn.functional", "torch._C"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["torch"].nn = sys.modules["torch.nn"]
    sys.modules["torch.nn"].functional = sys.modules["torch.nn.functional"]
    import importlib
    sub = importlib.import_module(
        "subtitles_extractor.infrastructure.tts.vieneu_gpu_subprocess"
    )
    monkeypatch.setattr(sub, "_write_wav", lambda p, a, sr: None)

    captured = {}

    class EngineV3Turbo:
        def infer_batch(self, texts, voice=None, batch_size=None, **kw):
            captured["batch_size"] = batch_size
            return [np.ones(5, dtype=np.float32) for _ in texts]

    batch = [{"text": "a", "output": str(tmp_path / "b0.wav")}]
    sub._handle_batch(EngineV3Turbo(), batch, voice={}, supported=frozenset(),
                      sample_rate=48000, item={"gpu_subbatch": 32})
    capsys.readouterr()
    assert captured["batch_size"] == 32  # truyen dung 32 chunk/forward


def test_worker_batch_max_batch_size_param(monkeypatch: pytest.MonkeyPatch, tmp_path, capsys) -> None:
    """Mode dùng max_batch_size (fast/turbo) → truyền qua tên do."""
    for name in ("torch", "torch.nn", "torch.nn.functional", "torch._C"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["torch"].nn = sys.modules["torch.nn"]
    sys.modules["torch.nn"].functional = sys.modules["torch.nn.functional"]
    import importlib
    sub = importlib.import_module(
        "subtitles_extractor.infrastructure.tts.vieneu_gpu_subprocess"
    )
    monkeypatch.setattr(sub, "_write_wav", lambda p, a, sr: None)
    captured = {}

    class EngineFast:
        def infer_batch(self, texts, voice=None, max_batch_size=None, **kw):
            captured["max_batch_size"] = max_batch_size
            return [np.ones(5, dtype=np.float32) for _ in texts]

    sub._handle_batch(EngineFast(), [{"text": "a", "output": str(tmp_path / "b.wav")}],
                      voice={}, supported=frozenset(), sample_rate=48000,
                      item={"gpu_subbatch": 16})
    capsys.readouterr()
    assert captured["max_batch_size"] == 16


def test_batch_all_fetches_everything(monkeypatch: pytest.MonkeyPatch) -> None:
    """[v3.23.422] batch_size cực lớn (Một lô cho TẤT CẢ) → gom hết câu vào 1 lần gọi."""
    adapter = _make_adapter(batch_size=10_000_000)
    adapter._batch_texts = [f"câu {i}" for i in range(5)]
    calls = []

    def fake_batch(texts, kwargs):
        calls.append(list(texts))
        return [np.full(3, float(i), dtype=np.float32) for i in range(len(texts))]

    monkeypatch.setattr(adapter, "_infer_batch_via_gpu_worker", fake_batch)
    a0 = adapter._get_batched_audio(0, {})
    assert a0 is not None
    # Một lần gọi gom HẾT 5 câu.
    assert len(calls) == 1 and len(calls[0]) == 5
    # Các câu còn lại đều HIT cache (không gọi thêm).
    for i in range(1, 5):
        assert adapter._get_batched_audio(i, {}) is not None
    assert len(calls) == 1
