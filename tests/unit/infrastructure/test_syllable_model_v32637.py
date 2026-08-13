"""[v3.23.237] Mô hình độ dài giọng: BIÊN DƯỚI theo ÂM TIẾT (không phải trung bình/ký tự).

Toan đặt câu hỏi: *"Tại sao xét theo ký tự mà không xét theo từ?"* — và câu hỏi đó dẫn tới
phát hiện lớn nhất của cả chuỗi phiên này.

**1. Đơn vị sai.** Tiếng Việt ĐƠN ÂM TIẾT: thời gian đọc tỉ lệ với số âm tiết, không phải
số ký tự. "nghiêng" (7 ký tự) đọc mất đúng bằng "ta" (2 ký tự) — một nhịp. Số ký tự mỗi âm
tiết dao động 2-6 (median 4.2), nên dùng ký tự là tự bơm nhiễu vào mô hình.

**2. Nhưng đổi đơn vị thôi KHÔNG cứu được gì** — và đây mới là phần quan trọng. Đo trên 95
câu Gemini thật:

==================================  ========
cách mô hình hoá                    R²
==================================  ========
trung bình theo KÝ TỰ (đang dùng)   0.069
trung bình theo ÂM TIẾT             0.059
**BIÊN DƯỚI theo ÂM TIẾT**          **0.894**
==================================  ========

Lý do: Gemini có phương sai KHỔNG LỒ. Cùng 2 âm tiết::

    "Vương thúc,"  ->  0.46s
    "Tu vi:"       ->  3.56s      (chênh 7,7 lần!)

Phần phía trên biên dưới là model **ngân/kéo dài ngẫu nhiên**. Hồi quy vào TRUNG BÌNH
chính là mô hình hoá cái nhiễu đó. Biên dưới (p10 mỗi nhóm âm tiết) thì ổn định::

    do_dai_toi_thieu = 0.231 + 0.217 x so_am_tiet     (R² = 0.894)

**3. Hệ quả: fix v236 của tôi sai.** "Sàn vật lý" = audio NGẮN NHẤT model sinh được ->
phải lấy biên dưới. Tôi lại lấy mô hình trung bình:

=========================  ==========
sàn                        giá trị
=========================  ==========
v236 (trung bình/ký tự)    0.45s
**đúng (biên dưới/âm tiết)**  **0.23s**
=========================  ==========

Sàn cao GẤP ĐÔI thực tế -> hai dòng (khung 0.40s và 0.44s) bị tước quyền lấy mẫu lại oan,
dù chúng vẫn còn cứu được. Sau khi sửa, chỉ còn 1/95 dòng thật sự dưới sàn (沦为, 0.20s).
"""

from __future__ import annotations

import pathlib

import pytest

from subtitles_extractor.infrastructure.tts.text_prep import dem_am_tiet
from subtitles_extractor.infrastructure.tts.timing_math import (
    GEMINI_MIN_BASE_S,
    GEMINI_MIN_PER_SYLLABLE_S,
    min_speech_seconds,
    window_below_engine_floor,
)

_GEMINI_SRC = pathlib.Path(
    "src/subtitles_extractor/infrastructure/tts/gemini_tts_adapter.py"
).read_text(encoding="utf-8")


# ── Đếm âm tiết: số KÝ TỰ không nói lên điều gì ─────────────────────────────
@pytest.mark.parametrize(
    ("text", "so_am_tiet"),
    [
        ("nghiêng", 1),  # 7 ký tự nhưng chỉ MỘT nhịp
        ("ta", 1),  # 2 ký tự, cũng MỘT nhịp — đọc mất thời gian như nhau
        ("Tu vi:", 2),
        ("Thanh Linh Đan", 3),
        ("Vương Kiến Cường", 3),  # 16 ký tự, vẫn chỉ 3 nhịp
        ("Đừng dùng đạo đức.", 4),
        ("", 0),
        ("   ", 0),
        ("!!!", 0),  # dấu câu không phát âm
    ],
)
def test_dem_am_tiet(text: str, so_am_tiet: int) -> None:
    assert dem_am_tiet(text) == so_am_tiet


def test_ky_tu_khong_du_de_doan_nhip_doc() -> None:
    """Hai câu cùng số âm tiết, số ký tự chênh gấp đôi -> đọc mất thời gian như nhau."""
    assert dem_am_tiet("Ý ta là,") == dem_am_tiet("Vương Kiến Cường")
    assert len("Ý ta là,") * 2 == len("Vương Kiến Cường")  # 8 vs 16 ký tự


# ── Biên dưới: audio ngắn nhất engine sinh được ─────────────────────────────
def test_bien_duoi_tang_tuyen_tinh_theo_am_tiet() -> None:
    m = lambda n: min_speech_seconds(  # noqa: E731
        n, GEMINI_MIN_BASE_S, GEMINI_MIN_PER_SYLLABLE_S
    )
    # do_dai_toi_thieu = 0.23 + 0.22 x n
    assert m(1) == pytest.approx(0.45, abs=0.01)
    assert m(4) == pytest.approx(1.11, abs=0.01)
    assert m(7) == pytest.approx(1.77, abs=0.01)
    # Khoảng cách giữa các bậc là hằng số (tuyến tính).
    assert (m(5) - m(4)) == pytest.approx(m(3) - m(2), abs=1e-6)


def test_khong_am_khi_so_am_tiet_bang_khong() -> None:
    assert min_speech_seconds(0) > 0.0  # vẫn còn chi phí vào câu
    assert min_speech_seconds(-5) == min_speech_seconds(0)  # không cho phép âm


# ── Sàn vật lý: v236 tính CAO GẤP ĐÔI ───────────────────────────────────────
def _duoi_san(available_s: float, max_ratio: float = 2.0) -> bool:
    return window_below_engine_floor(
        available_s,
        max_ratio,
        min_base_s=GEMINI_MIN_BASE_S,
        min_per_syllable_s=GEMINI_MIN_PER_SYLLABLE_S,
    )


def test_san_dung_la_023_khong_phai_045() -> None:
    # Sàn = min_speech_seconds(1) / max_ratio = 0.45 / 2.0 = 0.225s
    assert _duoi_san(0.22) is True
    assert _duoi_san(0.23) is False


def test_hai_dong_bi_tuoc_quyen_retry_oan_o_v236() -> None:
    """便差不多 (khung 0.40s) và 既然是个 (0.44s).

    Sàn sai của v236 (0.45s) coi chúng là bất khả thi -> nhận luôn bản đầu, không lấy mẫu
    lại. Nhưng sàn thật chỉ 0.23s: chúng vẫn còn cứu được, và ĐÁNG được lấy mẫu lại.
    """
    assert _duoi_san(0.40) is False
    assert _duoi_san(0.44) is False


def test_dong_that_su_bat_kha_thi_van_bi_chan() -> None:
    # 沦为 -> "Trở thành", khung 0.20s. Đây là lỗi TẦNG PHỤ ĐỀ, TTS không cứu nổi.
    assert _duoi_san(0.20) is True


def test_tran_nen_cao_hon_thi_san_thap_hon() -> None:
    assert _duoi_san(0.20, max_ratio=2.0) is True
    assert _duoi_san(0.20, max_ratio=3.0) is False


# ── Adapter dùng đúng mô hình ───────────────────────────────────────────────
def test_adapter_dung_bien_duoi_khong_dung_trung_binh() -> None:
    assert "min_base_s=GEMINI_MIN_BASE_S" in _GEMINI_SRC
    assert "min_per_syllable_s=GEMINI_MIN_PER_SYLLABLE_S" in _GEMINI_SRC
    # Mô hình trung bình KHÔNG được dùng cho câu hỏi về sàn nữa.
    assert "base_overhead_s=GEMINI_BASE_OVERHEAD_S,\n            per_char_s" not in (
        _GEMINI_SRC.split("window_below_engine_floor(")[1][:300]
        if "window_below_engine_floor(" in _GEMINI_SRC
        else ""
    )


def test_ghi_ro_sai_lam_v236_tai_code() -> None:
    assert "CAO GẤP ĐÔI thực tế" in _GEMINI_SRC
