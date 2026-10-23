"""[v3.23.627] Trần ``max_new_frames`` theo frame codec THẬT của từng chế độ + tag cảm xúc 3.8."""

from __future__ import annotations

from subtitles_extractor.infrastructure.tts.emotion_cues import (
    EMOTION_CUES,
    strip_brackets_keep_cues,
)
from subtitles_extractor.infrastructure.tts.vieneu_tts_adapter import (
    frames_per_second_for_mode,
)


def test_v3_turbo_codec_runs_at_12_5_frames_per_second() -> None:
    # 3840 mẫu/frame @ 48 kHz (vieneu_utils.core_utils.CODEC_SAMPLES_PER_FRAME) = 12,5 frame/s.
    assert frames_per_second_for_mode("v3turbo") == 48_000 / 3840
    assert frames_per_second_for_mode("standard") == 100      # neucodec: giữ nguyên
    assert frames_per_second_for_mode("") == 100


def test_cap_for_short_line_is_still_generous_but_real() -> None:
    from subtitles_extractor.infrastructure.tts import vieneu_tts_adapter as m

    cap_s = m.generation_time_cap_seconds(
        12, syllable_count=3, min_base_s=m.VIENEU_MIN_BASE_S,
        min_per_syllable_s=m.VIENEU_MIN_PER_SYLLABLE_S,
    )
    frames = int(cap_s * frames_per_second_for_mode("v3turbo")) + 50
    need = round(3 * 0.25 * 12.5)                  # ~0,25 s/âm tiết
    assert frames >= 5 * need                       # chỉ chặn ca thảm hoạ, câu hợp lệ không chạm
    assert frames < int(cap_s * 100) + 50           # nhưng không còn lỏng gấp 8 lần


def test_english_and_unaccented_cues_are_kept() -> None:
    for cue in ("[cười]", "[thở dài]", "[hắng giọng]", "[chuckle]", "[sigh]", "[clear throat]",
                "[cuoi]", "[tho dai]", "[hang giong]"):
        assert cue in EMOTION_CUES
        assert strip_brackets_keep_cues(f"Chào {cue} bạn") == f"Chào {cue} bạn"
    assert "[rác]" not in strip_brackets_keep_cues("Chào [rác] bạn")
