"""[v3.23.261] VieNeu đo "ngân dài" (hallucination) theo ÂM TIẾT thay vì KÝ TỰ.

**Toan phát hiện từ log thật:** lưới "DÀI BẤT THƯỜNG" đo bằng ký tự cho kết quả
loạn. "Chú..." = 6 ký tự nhưng 1 âm tiết (3 dấu chấm không phát âm); "Ồ," = 2 ký tự
(1 âm + dấu phẩy). Ký tự đếm cả dấu câu -> ngưỡng sai.

**Phân tích:** biên dưới theo ký tự R²=0.07 (vô dụng); theo âm tiết R²=0.98. Adapter
đã có sẵn hàm đúng ``is_abnormally_long_vs_floor`` (âm tiết) nhưng vẫn gọi hàm cũ
``is_abnormally_long`` (ký tự). Fix: chuyển sang hàm âm tiết + hằng số VieNeu.

**Bằng chứng khác biệt (đo thật):**
- "Chú..............." (18 ký tự, 1 âm, 2.5s): lưới ký tự BỎ LỌT (18 ký tự đẩy
  ngưỡng cao); lưới âm tiết BẮT ĐÚNG (1 âm mà 2.5s là ngân dài).
- "Ừ, ừ, ừ, ừ, ừ." (14 ký tự, 5 âm, 3.0s): lưới ký tự BÁO SAI; lưới âm tiết ĐÚNG
  (5 âm lặp đọc 3s là bình thường).
"""

from __future__ import annotations

from subtitles_extractor.infrastructure.tts.text_prep import dem_am_tiet
from subtitles_extractor.infrastructure.tts.timing_math import (
    VIENEU_MIN_BASE_S,
    VIENEU_MIN_PER_SYLLABLE_S,
    is_abnormally_long_vs_floor,
)


def _vieneu_floor_check(text: str, duration_s: float) -> bool:
    return is_abnormally_long_vs_floor(
        duration_s,
        max(1, dem_am_tiet(text.strip())),
        min_base_s=VIENEU_MIN_BASE_S,
        min_per_syllable_s=VIENEU_MIN_PER_SYLLABLE_S,
    )


def test_câu_1_âm_ngân_dài_bị_bắt() -> None:
    # "Chú..." 1 âm nhưng 12s (log thật) -> phải bắt là ngân dài.
    assert _vieneu_floor_check("Chú...", 12.0) is True


def test_câu_1_âm_nhiều_dấu_vẫn_bắt() -> None:
    # Nhiều dấu chấm KHÔNG che giấu được ngân dài (khác lưới ký tự).
    assert _vieneu_floor_check("Chú...............", 2.5) is True


def test_câu_nhiều_âm_lặp_không_báo_sai() -> None:
    # 5 âm lặp đọc 3s là BÌNH THƯỜNG -> không được báo động (lưới ký tự báo sai).
    assert _vieneu_floor_check("Ừ, ừ, ừ, ừ, ừ.", 3.0) is False


def test_câu_dài_bình_thường_không_báo() -> None:
    # Câu nhiều âm đọc với nhịp bình thường -> không báo.
    assert _vieneu_floor_check("Chúc mừng ký chủ đã nhận nhiệm vụ.", 3.8) is False


def test_adapter_dùng_hàm_âm_tiết() -> None:
    # Adapter phải gọi is_abnormally_long_vs_floor (âm tiết), KHÔNG phải bản ký tự.
    import pathlib

    src = pathlib.Path(
        "src/subtitles_extractor/infrastructure/tts/vieneu_tts_adapter.py"
    ).read_text(encoding="utf-8")
    # Lời gọi thực tế trong _synthesize_with_retry phải là bản _vs_floor.
    assert "is_abnormally_long_vs_floor(" in src
    assert "syllable_count = dem_am_tiet(" in src
    # Không còn gọi bản ký tự cũ trong logic (chỉ còn re-export cho test).
    assert "if not is_abnormally_long(duration_s" not in src


def test_log_dùng_âm_tiết_không_ký_tự() -> None:
    # Thông báo log đổi từ "ký tự" sang "âm tiết".
    import pathlib

    src = pathlib.Path(
        "src/subtitles_extractor/infrastructure/tts/vieneu_tts_adapter.py"
    ).read_text(encoding="utf-8")
    assert "cho %d âm tiết" in src
