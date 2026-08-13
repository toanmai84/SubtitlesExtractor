"""[v3.23.222] Ngân sách ký tự (``max_chars``) gửi cho model dịch — tính theo VẬT LÝ đọc.

Công thức cũ ở tầng dịch: ``max(8, khung_gốc * 16 ký_tự/giây)``. Hai sai lầm, đo trên
phiên VieNeu thật (95 câu):

1. **Tốc độ đọc KHÔNG phải hằng số.** Thực đo: ``thời_lượng = 0.302 + 0.0486 * số_ký_tự``
   -> mỗi câu có chi phí cố định ~0.3s (lấy hơi, đuôi âm). Hệ quả: khung 2.0s bị ép cắt
   xuống 32 ký tự dù đọc kịp 34; khung 3.0s lệch tới 35%. Ngược lại khung 0.20s vẫn hứa
   8 ký tự — trong khi khung đó KHÔNG đủ đọc nổi một ký tự.
2. **Bỏ qua gap.** TTS đọc trong khung + phần gap tới câu sau (``effective_available_
   seconds``), nhưng ngân sách chỉ tính khung gốc -> ép cắt nghĩa ở dòng vốn có khoảng
   lặng rộng phía sau.

Kết quả: 57/95 dòng (60%) vượt ngân sách cũ — model phớt lờ vì con số phi lý. Ngân sách
mới có TỔNG gần y hệt (+1%) nhưng PHÂN BỔ đúng chỗ.
"""

from __future__ import annotations

import pytest

from subtitles_extractor.domain.ports.subtitle_translator_port import TranslationLine
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
)
from subtitles_extractor.infrastructure.tts.timing_math import (
    MIN_CHAR_BUDGET,
    expected_speech_seconds,
    readable_char_budget,
    readable_syllable_budget,
    syllable_budget_to_chars,
)


def _line(start_ms: int, end_ms: int, text: str = "x") -> TranslationLine:
    return TranslationLine(index=1, start_ms=start_ms, end_ms=end_ms, text=text)


# ── Hàm thuần: ngân sách phải NHẤT QUÁN với mô hình độ dài của TTS ───────────
@pytest.mark.parametrize("n_char", [6, 10, 20, 34, 50])
def test_ngan_sach_nhat_quan_voi_mo_hinh_do_dai(n_char: int) -> None:
    # Câu dài đúng bằng ngân sách thì phải đọc VỪA khung ở tốc độ tự nhiên (1.0x).
    khung = expected_speech_seconds(n_char)
    assert readable_char_budget(khung) >= n_char - 1  # sai số làm tròn 1 ký tự


def test_khung_cuc_ngan_tra_ve_san() -> None:
    # Khung 0.20s không đủ đọc nổi một ký tự -> trả SÀN, không hứa hẹn điều bất khả thi.
    assert readable_char_budget(0.20) == MIN_CHAR_BUDGET
    assert readable_char_budget(0.0) == MIN_CHAR_BUDGET
    assert readable_char_budget(-1.0) == MIN_CHAR_BUDGET


def test_ngan_sach_tang_theo_khung() -> None:
    assert readable_char_budget(1.0) < readable_char_budget(2.0)
    assert readable_char_budget(2.0) < readable_char_budget(3.0)


def test_sua_duoc_sai_lech_cua_cong_thuc_cu() -> None:
    cps_cu = lambda t: max(8, int(t * 16.0))  # noqa: E731 — công thức cũ, để đối chiếu
    # Khung DÀI: công thức cũ ép cắt nghĩa vô ích -> ngân sách mới phải RỘNG hơn.
    assert readable_char_budget(2.0) > cps_cu(2.0)
    assert readable_char_budget(3.0) > cps_cu(3.0)
    # Khung cực NGẮN: công thức cũ hứa hẹn 8 ký tự cho khung 0.2s (bất khả thi) ->
    # ngân sách mới trả SÀN thay vì con số ảo.
    assert readable_char_budget(0.20) == MIN_CHAR_BUDGET


# ── Tầng dịch: hint dùng KHUNG HIỆU DỤNG ────────────────────────────────────
def test_hint_dung_gap_toi_cau_sau() -> None:
    # Ca thật #91: khung gốc 0.96s nhưng có gap -> khung hiệu dụng 1.34s.
    line = _line(147_960, 148_920)
    hint_khong_gap = GeminiSubtitleTranslator._length_hint(line)
    hint_co_gap = GeminiSubtitleTranslator._length_hint(line, next_start_ms=149_400)
    assert hint_co_gap > hint_khong_gap  # gap rộng -> được dịch dài hơn, không cắt oan


def test_hint_khong_phong_khi_khong_biet_dong_ke_tiep() -> None:
    """Không biết dòng sau -> KHÔNG được mượn ``max_gap_use_s`` (2s) như câu cuối phim.

    Bug bắt được lúc phát triển: dùng thẳng ``effective_available_seconds(next=None)``
    khiến dòng khung 0.64s nhận ngân sách 46 ký tự (thay vì 6) -> model dịch dài gấp 7
    lần khung.
    """
    line = _line(123_960, 124_600)  # ca thật #72: khung 0.64s
    hint_khong_biet = GeminiSubtitleTranslator._length_hint(line)
    # KHÔNG được phồng như câu cuối phim (bug: 46 ký tự cho khung 0.64s).
    # [v3.23.238] Ngân sách nay theo ÂM TIẾT rồi quy sang ký tự — mô hình theo KÝ TỰ đã bị
    # bác bỏ bằng dữ liệu (R²=0.07 so với 0.89). Điều test này bảo vệ vẫn nguyên: không
    # được mượn gap khi chưa biết dòng kế tiếp.
    assert hint_khong_biet == max(
        MIN_CHAR_BUDGET, syllable_budget_to_chars(readable_syllable_budget(0.64))
    )
    assert hint_khong_biet < 20


def test_hint_ca_that_72_chat_nhung_khong_bop_nghet() -> None:
    """#72 "cháu đã nói về chú thế nào," (27 ký tự) — khung 0.64s, gap chỉ 0.08s.

    [v3.23.223] Ngân sách phải CHẶT hơn 27 (để model biết cần rút gọn) nhưng KHÔNG được
    bóp tới mức 6 ký tự như v222 — con số đó buộc model dịch thành "Chú?", mất sạch nghĩa.
    """
    line = _line(123_960, 124_600)
    hint = GeminiSubtitleTranslator._length_hint(line, next_start_ms=124_680)
    assert hint < 27  # vẫn báo cho model biết dòng này chật
    assert hint > MIN_CHAR_BUDGET  # nhưng đủ chỗ giữ nghĩa (v222 chỉ cho 6 -> hỏng)


def test_hint_bang_0_khi_moc_thoi_gian_hong() -> None:
    assert GeminiSubtitleTranslator._length_hint(_line(1000, 1000)) == 0
    assert GeminiSubtitleTranslator._length_hint(_line(2000, 1000)) == 0


def test_payload_kem_max_chars() -> None:
    payload = GeminiSubtitleTranslator._line_to_payload(
        _line(10_000, 12_000, "xin chào"), next_start_ms=12_500
    )
    assert payload["max_chars"] > MIN_CHAR_BUDGET
    assert payload["line_no"] == 1


def test_prompt_giai_thich_max_chars_kem_hau_qua() -> None:
    import pathlib

    src = pathlib.Path(
        "src/subtitles_extractor/infrastructure/translation/gemini_translation_adapter.py"
    ).read_text(encoding="utf-8")
    # [v3.23.223] Chỉ thị phải nêu ngân sách VÀ khẳng định nghĩa quan trọng hơn — chỉ nói
    # "TRẦN CỨNG" (v222) khiến model cắt nghĩa để đạt chỉ tiêu.
    assert "max_chars" in src
    assert "NGHĨA LUÔN THẮNG GIỚI HẠN" in src
