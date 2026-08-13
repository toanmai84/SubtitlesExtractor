"""[v3.23.262] Rà soát toàn ứng dụng: chuyển các tính toán TTS từ KÝ TỰ sang ÂM TIẾT.

Sau khi Toan phát hiện lưới hallucination VieNeu đo bằng ký tự sai (v261), rà toàn bộ các
chỗ dùng ký tự cho tính toán liên quan nhịp đọc:

1. **Gemini hallucination:** bỏ nhánh fallback ký tự (R²=0.07) — nay THUẦN âm tiết.
   Text không có âm tiết (rỗng/toàn dấu) -> không coi là hallucination.
2. **generation_time_cap_seconds (trần max_new_frames):** ưu tiên âm tiết -> câu cùng
   số âm có cùng trần dù khác số ký tự (dấu câu).

Các chỗ dùng ký tự HỢP LỆ (không đổi): budget dịch quy đổi âm tiết->ký tự ở bước
cuối (model đếm ký tự đầu ra); "N nhân vật" ở UI (số nhân vật phim, không phải ký tự).
"""

from __future__ import annotations

from subtitles_extractor.infrastructure.tts.text_prep import dem_am_tiet
from subtitles_extractor.infrastructure.tts.timing_math import (
    VIENEU_MIN_BASE_S,
    VIENEU_MIN_PER_SYLLABLE_S,
    generation_time_cap_seconds,
)


def test_cap_theo_âm_tiết_nhất_quán() -> None:
    # Hai câu CÙNG số âm nhưng khác số ký tự -> cap phải BẰNG NHAU (theo âm tiết).
    t1 = "Ừ, ừ, ừ, ừ, ừ."  # 5 âm, 14 ký tự
    t2 = "không không không không không"  # 5 âm, 29 ký tự
    assert dem_am_tiet(t1) == dem_am_tiet(t2) == 5
    cap1 = generation_time_cap_seconds(
        len(t1), syllable_count=dem_am_tiet(t1),
        min_base_s=VIENEU_MIN_BASE_S, min_per_syllable_s=VIENEU_MIN_PER_SYLLABLE_S,
    )
    cap2 = generation_time_cap_seconds(
        len(t2), syllable_count=dem_am_tiet(t2),
        min_base_s=VIENEU_MIN_BASE_S, min_per_syllable_s=VIENEU_MIN_PER_SYLLABLE_S,
    )
    assert abs(cap1 - cap2) < 0.01  # bằng nhau vì cùng số âm


def test_cap_ký_tự_khác_nhau_khi_không_có_âm_tiết() -> None:
    # Không truyền syllable_count -> fallback ký tự (tương thích ngược).
    cap_char = generation_time_cap_seconds(20)
    assert cap_char > 0


def test_cap_âm_tiết_không_bị_dấu_câu_làm_loãng() -> None:
    # "Chú..." (6 ký tự, 1 âm) và "Chú" (3 ký tự, 1 âm) -> cùng cap theo âm tiết.
    cap_dots = generation_time_cap_seconds(
        len("Chú..."), syllable_count=dem_am_tiet("Chú..."),
        min_base_s=VIENEU_MIN_BASE_S, min_per_syllable_s=VIENEU_MIN_PER_SYLLABLE_S,
    )
    cap_plain = generation_time_cap_seconds(
        len("Chú"), syllable_count=dem_am_tiet("Chú"),
        min_base_s=VIENEU_MIN_BASE_S, min_per_syllable_s=VIENEU_MIN_PER_SYLLABLE_S,
    )
    assert abs(cap_dots - cap_plain) < 0.01


def test_cap_floor_vẫn_áp_dụng() -> None:
    # Câu cực ngắn (1 âm) vẫn có sàn tối thiểu (không quá nhỏ).
    cap = generation_time_cap_seconds(
        3, syllable_count=1,
        min_base_s=VIENEU_MIN_BASE_S, min_per_syllable_s=VIENEU_MIN_PER_SYLLABLE_S,
    )
    assert cap >= 3.0  # GENERATION_CAP_FLOOR_S


def test_gemini_thuần_âm_tiết() -> None:
    # Gemini adapter bỏ nhánh ký tự -> chỉ còn lời gọi âm tiết.
    import pathlib

    src = pathlib.Path(
        "src/subtitles_extractor/infrastructure/tts/gemini_tts_adapter.py"
    ).read_text(encoding="utf-8")
    assert "syllable_count > 0 and is_abnormally_long_vs_floor(" in src
    # Không còn gọi is_abnormally_long (bản ký tự) trong logic.
    assert "hallucination = is_abnormally_long(" not in src
