"""[v3.23.655] VieNeu GPU: lô gom các câu cùng giọng ở bất kỳ đâu phía sau (không chỉ cửa sổ kế tiếp), sắp theo độ dài."""

from __future__ import annotations

import numpy as np
import pytest

from tests.unit.infrastructure.test_vieneu_batch_v323412 import _make_adapter


def test_alternating_voices_still_fill_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_adapter(batch_size=4)
    adapter._batch_texts = [f"câu {i}" + "x" * (i % 3) for i in range(16)]
    adapter._batch_voices = ["A", "B"] * 8                       # hai vai nói xen kẽ
    calls: list[tuple[str, list[str]]] = []

    def fake_batch(texts, kwargs):
        calls.append((kwargs.get("voice_preset", ""), list(texts)))
        return [np.full(4, float(len(t)), dtype=np.float32) for t in texts]

    monkeypatch.setattr(adapter, "_infer_batch_via_gpu_worker", fake_batch)
    for idx in range(16):
        audio = adapter._get_batched_audio(idx, {})
        assert audio is not None and audio[0] == float(len(adapter._batch_texts[idx]))  # đúng audio của đúng câu
    # Trước đây cửa sổ 4 câu chỉ có 2 câu cùng giọng → 8 lần gọi; nay mỗi lần đủ 4 câu → 4 lần gọi.
    assert len(calls) == 4
    assert all(len(texts) == 4 for _voice, texts in calls)
    assert all([len(t) for t in texts] == sorted(len(t) for t in texts) for _voice, texts in calls)


def test_length_cap_follows_longest_line_in_batch_v323655() -> None:
    """[v3.23.655] Rà soát: trần max_new_frames tính theo câu idx — câu dài gom từ xa vào lô bị cắt cụt."""
    from subtitles_extractor.infrastructure.tts.vieneu_tts_adapter import _scale_length_caps

    caps = {"max_new_frames": 300, "temperature": 0.7}
    out = _scale_length_caps(caps, "xin chào", ["xin chào", "hôm nay trời đẹp quá đi thôi bạn ơi"])
    assert out["max_new_frames"] > 300 * 4 and out["temperature"] == 0.7
    assert _scale_length_caps(caps, "hôm nay trời đẹp", ["xin chào"]) is caps
