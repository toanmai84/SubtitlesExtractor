"""[v3.23.128] Test 3 sửa lỗi chất lượng/hiệu năng:

1. Ghép-theo-vị-trí CHỈ khi đủ số dòng (chống dồn lệch khi model rớt dòng giữa).
2. Khử nhãn người nói lặp '[X:] [X:]'.
3. Tầng nén GPU mới (NVDEC giải mã + scale CPU + NVENC) + nhãn phân biệt.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from subtitles_extractor.application.use_cases.translate_subtitles import (
    TranslateSubtitlesUseCase,
)
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
    _BatchPartialError,
    _BatchValidationError,
)

# ── Fix 1: ghép vị trí an toàn ────────────────────────────────────────────


def _vstub() -> SimpleNamespace:
    return SimpleNamespace(
        _renumber_items_by_position=GeminiSubtitleTranslator._renumber_items_by_position
    )


def _payload(pairs: list[tuple[int, str]]) -> dict:
    return {
        "subtitles": [
            {"line_no": ln, "text": t, "speaker": "", "description": ""}
            for ln, t in pairs
        ]
    }


def test_exact_count_offset_is_renumbered_by_position() -> None:
    # Model trả ĐỦ 3 dòng nhưng đánh số từ 1 (lệch) → ghép vị trí AN TOÀN.
    payload = _payload([(1, "a"), (2, "b"), (3, "c")])
    GeminiSubtitleTranslator._validate_batch(_vstub(), payload, 3, 10)  # không raise
    assert [it["line_no"] for it in payload["subtitles"]] == [11, 12, 13]


def test_short_count_offset_does_not_position_merge() -> None:
    # Model trả THIẾU (2/3) + đánh số lệch → KHÔNG được ghép vị trí (sẽ dồn lệch).
    payload = _payload([(1, "a"), (2, "b")])
    with pytest.raises((_BatchValidationError, _BatchPartialError)):
        GeminiSubtitleTranslator._validate_batch(_vstub(), payload, 3, 10)
    # Quan trọng: KHÔNG bị đánh số lại theo vị trí (giữ nguyên số gốc của model).
    assert [it["line_no"] for it in payload["subtitles"]] == [1, 2]


def test_short_count_large_batch_offset_raises_for_halving() -> None:
    # Batch lớn (20), model trả 18 đánh số lệch 1..18 → thiếu toàn bộ tập kỳ vọng
    # (385..404) → raise để chia đôi, KHÔNG dồn lệch.
    payload = _payload([(i, f"t{i}") for i in range(1, 19)])
    with pytest.raises((_BatchValidationError, _BatchPartialError)):
        GeminiSubtitleTranslator._validate_batch(_vstub(), payload, 20, 384)


def test_correct_linenos_missing_two_middle_patches_by_lineno() -> None:
    # Đánh số ĐÚNG (11..25) nhưng thiếu 2 dòng giữa (13,16) → vá theo line_no
    # (2/15 ≤ 15% nên vá cửa sổ thay vì chia đôi).
    pairs = [(ln, f"t{ln}") for ln in range(11, 26) if ln not in (13, 16)]
    payload = _payload(pairs)
    with pytest.raises(_BatchPartialError) as ei:
        GeminiSubtitleTranslator._validate_batch(_vstub(), payload, 15, 10)
    # Các dòng thiếu phải đúng theo line_no (13,16) — KHÔNG theo vị trí.
    missing = ei.value.missing_line_nos
    assert 13 in missing and 16 in missing


# ── Fix 2: khử nhãn người nói lặp ─────────────────────────────────────────


def test_dedupe_double_speaker_tag() -> None:
    f = TranslateSubtitlesUseCase._dedupe_leading_speaker_tag
    assert f("[Người dẫn chuyện:] [Người dẫn chuyện:] Đây là...") == (
        "[Người dẫn chuyện:] Đây là..."
    )
    assert f("[A:] [A:] [A:] x") == "[A:] x"


def test_dedupe_keeps_different_tags() -> None:
    f = TranslateSubtitlesUseCase._dedupe_leading_speaker_tag
    assert f("[A:] [B:] x") == "[A:] [B:] x"
    assert f("Không có nhãn") == "Không có nhãn"


# ── Fix 3: nhãn chế độ nén GPU ────────────────────────────────────────────


def test_encode_mode_labels_distinguish_tiers() -> None:
    from subtitles_extractor.infrastructure.translation.gemini_video_context import (
        GeminiVideoContextProvider as P,
    )
    full = ["-hwaccel", "cuda", "-vf", "scale_cuda=w=-2:h=640", "-c:v", "h264_nvenc"]
    decode = ["-hwaccel", "cuda", "-vf", "scale=-2:640", "-c:v", "h264_nvenc"]
    nvenc = ["-vf", "scale=-2:640", "-c:v", "h264_nvenc"]
    cpu = ["-vf", "scale=-2:640", "-c:v", "libx264"]
    assert "scale_cuda" in P._encode_mode_label(full)
    lbl = P._encode_mode_label(decode)
    assert "NVDEC" in lbl and "scale CPU" in lbl
    assert "một phần" in P._encode_mode_label(nvenc)
    assert "CPU" in P._encode_mode_label(cpu)
