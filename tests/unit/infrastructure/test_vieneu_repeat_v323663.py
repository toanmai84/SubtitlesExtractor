"""[v3.23.663] VieNeu đọc LẶP câu ngắn ("Đúng vậy" → "Đúng vậy, đúng vậy") — bắt và cắt bỏ lần đọc lại.

Đo thật 28/10 (v3 Turbo, giọng Ngọc Huyền, tập 1–3, 296 câu): 17 câu 1–3 tiếng bị đọc lặp, cả 17 lọt lưới cũ vì sàn
"dài bất thường" 1,8 s cao hơn trần khung của SDK (1,04 / 1,44 / 1,84 s).
"""

from __future__ import annotations

import numpy as np
import pytest

from subtitles_extractor.infrastructure.tts.audio_utils import (
    speech_seconds,
    trim_repeated_reading,
)
from subtitles_extractor.infrastructure.tts.timing_math import (
    VIENEU_MIN_BASE_S,
    VIENEU_MIN_PER_SYLLABLE_S,
    is_short_line_overlong,
)
from subtitles_extractor.infrastructure.tts.vieneu_tts_adapter import VieNeuTtsAdapter

SR = 48000


def _syllables(n: int, gap_s: float = 0.04, syl_s: float = 0.2) -> np.ndarray:
    t = np.arange(int(SR * syl_s)) / SR
    tone = (0.3 * np.sin(2 * np.pi * 180 * t) * np.hanning(t.size)).astype(np.float32)
    parts = []
    for _ in range(n):
        parts += [tone, np.zeros(int(SR * gap_s), np.float32)]
    return np.concatenate(parts[:-1])


def _pad(audio: np.ndarray, lead: float = 0.06, tail: float = 0.2) -> np.ndarray:
    return np.concatenate([np.zeros(int(SR * lead), np.float32), audio, np.zeros(int(SR * tail), np.float32)])


def _repeated(n: int, pause_s: float = 0.35) -> np.ndarray:
    return _pad(np.concatenate([_syllables(n), np.zeros(int(SR * pause_s), np.float32), _syllables(n)]))


def _overlong(speech: float, syl: int) -> bool:
    return is_short_line_overlong(speech, syl, min_base_s=VIENEU_MIN_BASE_S, min_per_syllable_s=VIENEU_MIN_PER_SYLLABLE_S)


@pytest.mark.parametrize(("syl", "ok", "bad"), [(1, 0.52, 0.94), (2, 0.76, 1.18), (3, 1.10, 1.46)])
def test_threshold_splits_measured_clusters(syl: int, ok: float, bad: float) -> None:
    assert not _overlong(ok, syl)
    assert _overlong(bad, syl)


def test_long_lines_use_the_general_net() -> None:
    assert not _overlong(5.0, 4)


@pytest.mark.parametrize("syl", [1, 2, 3])
def test_trim_keeps_first_reading(syl: int) -> None:
    audio = _repeated(syl)
    trimmed = trim_repeated_reading(audio, SR, syl)
    assert trimmed is not None
    single = speech_seconds(_pad(_syllables(syl)), SR)
    assert abs(speech_seconds(trimmed, SR) - single) < 0.12
    assert abs(float(trimmed[-1])) < 1e-3  # fade-out, không click


def test_no_cut_inside_first_reading() -> None:
    # Ngừng sớm khi mới đọc 1 tiếng của câu 3 tiếng → chưa đủ lời, không được cắt ở đó.
    t = _syllables(1, syl_s=0.08)
    audio = _pad(np.concatenate([t, np.zeros(int(SR * 0.3), np.float32), _syllables(3)]))
    trimmed = trim_repeated_reading(audio, SR, 3)
    assert trimmed is None or speech_seconds(trimmed, SR) > 0.6


def test_single_drawl_is_not_cut() -> None:
    assert trim_repeated_reading(_pad(_syllables(1, syl_s=1.0)), SR, 1) is None


class _Engine:
    sample_rate = SR


class _Request:
    retry_count = 5
    retry_delay_s = 0.0


def _adapter(monkeypatch: pytest.MonkeyPatch, takes: list[np.ndarray]) -> tuple[VieNeuTtsAdapter, dict]:
    adapter = object.__new__(VieNeuTtsAdapter)
    adapter._batch_size = 0
    adapter._batch_texts = []
    adapter._batch_cache = {}
    adapter._force_cpu = True
    adapter._gpu_worker_disabled = True
    calls = {"n": 0, "ref": []}

    def fake(engine, text, voice_data, idx=-1, **kw):
        calls["n"] += 1
        calls["ref"].append(kw.get("use_ref_codes", True))
        return takes[min(calls["n"], len(takes)) - 1]

    monkeypatch.setattr(adapter, "_infer_once", fake)
    monkeypatch.setattr(adapter, "_to_mono_pipeline_rate", lambda raw, sr=None: raw)
    return adapter, calls


def test_adapter_trims_repeated_short_line(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, calls = _adapter(monkeypatch, [_repeated(2)])
    out = adapter._synthesize_with_retry(engine=_Engine(), text="Đúng vậy", request=_Request(),
                                         voice_data={}, cancel_cb=None, idx=0)
    assert out is not None
    # Trước đây: trả nguyên bản đọc hai lần (~1,2 s tiếng); một lần đọc ~0,56 s.
    assert speech_seconds(out, SR) < 0.7 < speech_seconds(_repeated(2), SR)
    assert calls["n"] == 1


def test_adapter_resamples_when_repeat_has_no_pause(monkeypatch: pytest.MonkeyPatch) -> None:
    glued = _pad(_syllables(4))  # "thích thích" dính liền: không cắt được → sinh lại
    clean = _pad(_syllables(2))
    adapter, calls = _adapter(monkeypatch, [glued, clean])
    out = adapter._synthesize_with_retry(engine=_Engine(), text="Đúng vậy", request=_Request(),
                                         voice_data={}, cancel_cb=None, idx=0)
    assert calls["n"] == 2
    assert speech_seconds(out, SR) == pytest.approx(speech_seconds(clean, SR))
    # Lần sinh lại bỏ mã tham chiếu trong ngữ cảnh (đo thật: đọc lặp 23% → 9,4%).
    assert calls["ref"] == [True, False]


def test_infer_once_forwards_use_ref_codes() -> None:
    seen: dict = {}

    class _Eng:
        sample_rate = SR

        def infer(self, text, voice=None, max_new_frames=300, repetition_penalty=1.2, use_ref_codes=True):
            seen.update(use_ref_codes=use_ref_codes)
            return np.zeros(10, np.float32)

    adapter = object.__new__(VieNeuTtsAdapter)
    adapter._force_cpu = True
    adapter._mode = "v3turbo"
    adapter._emotion = "natural"
    adapter._temperature = None
    adapter._infer_once(_Eng(), "Đúng vậy", {}, use_ref_codes=False)
    assert seen["use_ref_codes"] is False
    adapter._infer_once(_Eng(), "Đúng vậy", {})
    assert seen["use_ref_codes"] is True


def test_gpu_worker_forwards_use_ref_codes() -> None:
    import inspect

    from subtitles_extractor.infrastructure.tts import vieneu_gpu_subprocess as worker

    assert inspect.getsource(worker).count('"temperature", "use_ref_codes"') == 2


def test_normal_short_line_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    clean = _pad(_syllables(2))
    adapter, calls = _adapter(monkeypatch, [clean])
    out = adapter._synthesize_with_retry(engine=_Engine(), text="Đúng vậy", request=_Request(),
                                         voice_data={}, cancel_cb=None, idx=0)
    assert out is clean and calls["n"] == 1


def test_comma_line_with_pause_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    # "Ơ, khoan đã" (đo thật 0,87 s tiếng, có ngừng 0,32 s sau "Ơ") — không vượt ngưỡng nên không bị cắt.
    audio = _pad(np.concatenate([_syllables(1), np.zeros(int(SR * 0.3), np.float32), _syllables(2)]))
    adapter, calls = _adapter(monkeypatch, [audio])
    out = adapter._synthesize_with_retry(engine=_Engine(), text="Ơ, khoan đã,", request=_Request(),
                                         voice_data={}, cancel_cb=None, idx=0)
    assert out is audio and calls["n"] == 1


# ── Rà độc lập: các ca không được cắt / không được tốn retry ────────────────────


def test_text_that_really_repeats_is_not_cut(monkeypatch: pytest.MonkeyPatch) -> None:
    # "Hả? Hả?" đọc chậm: dài quá ngưỡng 2 tiếng nhưng khoảng ngừng là của lời thoại.
    audio = _pad(np.concatenate([_syllables(1, syl_s=0.4), np.zeros(int(SR * 0.35), np.float32),
                                 _syllables(1, syl_s=0.4)]))
    adapter, calls = _adapter(monkeypatch, [audio])
    out = adapter._synthesize_with_retry(engine=_Engine(), text="Hả? Hả?", request=_Request(),
                                         voice_data={}, cancel_cb=None, idx=0)
    assert speech_seconds(out, SR) > 1.0  # còn đủ hai tiếng


def test_leading_click_does_not_move_the_cut() -> None:
    click = np.zeros(int(SR * 0.03), np.float32)
    click[::40] = 0.3
    slow = np.concatenate([_syllables(1, syl_s=0.3), np.zeros(int(SR * 0.35), np.float32), _syllables(2, syl_s=0.3)])
    audio = _pad(np.concatenate([click, np.zeros(int(SR * 0.3), np.float32), slow]))
    trimmed = trim_repeated_reading(audio, SR, 3)
    assert trimmed is None or speech_seconds(trimmed, SR) > 0.9


def test_stereo_input_is_left_alone() -> None:
    assert trim_repeated_reading(np.zeros((SR, 2), np.float32), SR, 2) is None


def test_slow_valid_short_line_costs_one_retry_at_most(monkeypatch: pytest.MonkeyPatch) -> None:
    slow = _pad(_syllables(2, syl_s=0.5))  # giọng đọc chậm hợp lệ, không có khoảng ngừng để cắt
    adapter, calls = _adapter(monkeypatch, [slow])
    out = adapter._synthesize_with_retry(engine=_Engine(), text="Đúng vậy", request=_Request(),
                                         voice_data={}, cancel_cb=None, idx=0)
    assert out is slow and calls["n"] == 2
