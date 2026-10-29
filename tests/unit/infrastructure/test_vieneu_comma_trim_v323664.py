"""[v3.23.664] "Sư huynh, em..." bị TTS thành "Sư huynh" — bước cắt-lặp (v3.23.663) cắt nhầm ở dấu phẩy.

Đo thật 20 lần (VieNeu v3 Turbo, giọng Ngọc Huyền): 20/20 lần model nói thêm sau "em" ("em em sư huynh",
"em em mệt mỏi"…). Đảo năng lượng: "sư huynh" 0,05–0,47 s | ngừng 0,25 s | "em" 0,70–0,95 s | ngừng | nói thêm
1,20–1,84 s. Khi "sư huynh" dài ≥0,45 s (= 0,15 s × 3 tiếng), mã cũ cắt ngay ở dấu phẩy → 4/20 bản mất chữ "em".
"""

from __future__ import annotations

import numpy as np

from subtitles_extractor.infrastructure.tts.audio_utils import speech_seconds, trim_repeated_reading

SR = 48000


def _tone(seconds: float, freq: float = 180.0) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    env = np.minimum(1.0, np.minimum(t, t[::-1]) / 0.01)  # biên phẳng, ramp 10 ms như tiếng nói thật
    return (0.3 * np.sin(2 * np.pi * freq * t) * env).astype(np.float32)


def _gap(seconds: float) -> np.ndarray:
    return np.zeros(int(SR * seconds), np.float32)


def test_comma_pause_is_not_taken_for_a_repeat() -> None:
    # Đúng cấu trúc bản thật số 2 (bị mất chữ "em" ở v3.23.663).
    audio = np.concatenate([_gap(0.06), _tone(0.45), _gap(0.20), _tone(0.21, 220), _gap(0.28), _tone(0.55), _gap(0.1)])
    trimmed = trim_repeated_reading(audio, SR, 3)
    assert trimmed is not None
    # Giữ cả "sư huynh" lẫn "em" (~0,9 s); trước đây chỉ còn ~0,5 s.
    assert speech_seconds(trimmed, SR) > 0.8


def test_real_repeat_still_cut() -> None:
    # "Hóa Thần | Hóa Thần" (0,09–0,50 | 0,94–1,33): phần bỏ ~bằng phần giữ.
    audio = np.concatenate([_gap(0.09), _tone(0.41), _gap(0.44), _tone(0.39), _gap(0.1)])
    trimmed = trim_repeated_reading(audio, SR, 2)
    assert trimmed is not None and speech_seconds(trimmed, SR) < 0.6


def test_one_syllable_long_second_reading_still_cut() -> None:
    # "con | con…" (0,06–0,27 | 0,61–1,04): lần đọc lại dài gấp đôi — câu 1 tiếng không có ngừng bên trong.
    audio = np.concatenate([_gap(0.06), _tone(0.21), _gap(0.34), _tone(0.43), _gap(0.1)])
    trimmed = trim_repeated_reading(audio, SR, 1)
    assert trimmed is not None and speech_seconds(trimmed, SR) < 0.4


# ── "Sư muội, ta..." (thử nghiệm lớn 836 câu): phần sau dấu phẩy NGẮN nên luật tỉ lệ không chặn được ────────────


def test_internal_break_count() -> None:
    from subtitles_extractor.infrastructure.tts.text_prep import count_internal_breaks

    assert count_internal_breaks("Sư muội, ta...") == 1  # "..." cuối câu không tính
    assert count_internal_breaks("Ngươi... ngươi dám!") == 1
    assert count_internal_breaks("Hóa Thần") == 0
    assert count_internal_breaks("Việc đầu tiên,") == 0
    # Rà độc lập: chấm đơn giữa câu, chữ mở bằng ngoặc/thẻ cảm xúc; số có dấu phân cách không phải chỗ ngắt.
    assert count_internal_breaks("Sư muội. Ta...") == 1
    assert count_internal_breaks('Sư muội, "ta"...') == 1
    assert count_internal_breaks("Ừ, [cười]") == 1
    assert count_internal_breaks("Hả?! Sao") == 1
    assert count_internal_breaks("1,000 lượng") == 0
    assert count_internal_breaks("10:30 rồi") == 0
    assert count_internal_breaks("3.5 cân") == 0
    assert count_internal_breaks("A…B") == 1


def _su_muoi_ta(tail: float = 0.0) -> np.ndarray:
    # Đo thật: "sư muội" 0,05–0,50 | ngừng 0,30 (dấu phẩy) | "ta" 0,80–1,00 (+ nói thêm nếu có).
    parts = [_gap(0.05), _tone(0.45), _gap(0.30), _tone(0.20, 220)]
    if tail:
        parts += [_gap(0.30), _tone(tail)]
    return np.concatenate([*parts, _gap(0.1)])


def test_comma_pause_skipped_when_last_word_is_short() -> None:
    audio = _su_muoi_ta()
    # Không biết dấu phẩy (như v3.23.663/đầu 664): cắt ở dấu phẩy → mất "ta".
    lost = trim_repeated_reading(audio, SR, 3)
    assert lost is not None and speech_seconds(lost, SR) < 0.6
    # Biết câu có 1 chỗ ngắt: không cắt (không có lần đọc lại).
    assert trim_repeated_reading(audio, SR, 3, internal_breaks=1) is None


def test_babble_after_last_word_still_cut_with_breaks() -> None:
    trimmed = trim_repeated_reading(_su_muoi_ta(tail=0.5), SR, 3, internal_breaks=1)
    assert trimmed is not None and 0.85 < speech_seconds(trimmed, SR) < 1.1  # giữ "sư muội, ta"


def test_short_comma_pause_counts_as_the_break() -> None:
    # Ngừng ở dấu phẩy chỉ 0,19 s (đo thật 2/20 bản "Sư huynh, em...") — vẫn phải được coi là chỗ ngắt.
    audio = np.concatenate([_gap(0.05), _tone(0.46), _gap(0.19), _tone(0.23, 220), _gap(0.24), _tone(0.21), _gap(0.1)])
    trimmed = trim_repeated_reading(audio, SR, 3, internal_breaks=1)
    assert trimmed is not None and speech_seconds(trimmed, SR) > 0.8


def test_adapter_keeps_last_word_after_comma(monkeypatch) -> None:
    from .test_vieneu_repeat_v323663 import _adapter, _Engine, _Request

    adapter, calls = _adapter(monkeypatch, [_su_muoi_ta(tail=0.5)])
    out = adapter._synthesize_with_retry(engine=_Engine(), text="Sư muội, ta...", request=_Request(),
                                         voice_data={}, cancel_cb=None, idx=0)
    assert speech_seconds(out, SR) > 0.85  # còn "ta"; trước đây cắt ở dấu phẩy còn ~0,5 s
    assert calls["n"] == 1
