"""[v3.23.221] Lưới "audio DÀI BẤT THƯỜNG" — bắt hallucination ngân dài của model neural.

Bằng chứng (FLAC + CSV cùng phiên, VieNeu, 95 câu / 154.90s):

* Hồi quy độ dài trên 94 câu: ``do_dai ~ 0.302 + 0.0486 * so_ky_tu``
  (sai số chuẩn 0.273s).
* Câu #88 "Ừm." (3 ký tự, kỳ vọng 0.45s) được model sinh **2.55s** -> lệch **+7.7 lần
  sai số chuẩn**, trong khi câu bất thường thứ nhì chỉ 3.9 ("Đi đi, đi đi đi." — dài
  HỢP LỆ vì lặp từ).
* Đo trên FLAC: một cụm tiếng liên tục 1.34s -> model NGÂN kéo dài, không phải đọc lặp.
* Hệ quả dây chuyền: #88 là câu DUY NHẤT bị nén kịch trần 2.0x (tan formant) VÀ là câu
  DUY NHẤT còn đè lên mốc câu sau (0.23s) trong toàn bộ file.

Lưới ``is_effectively_silent`` (v205) chỉ bắt dạng lỗi NGƯỢC LẠI (audio câm). Cùng một cơ
chế lấy mẫu ngẫu nhiên sinh ra cả hai dạng -> cần cả hai lưới.
"""

from __future__ import annotations

import numpy as np
import pytest

from subtitles_extractor.infrastructure.tts.audio_utils import shorter_take
from subtitles_extractor.infrastructure.tts.timing_math import (
    expected_speech_seconds,
    is_abnormally_long,
)

# Các ca THẬT trích từ phiên debug (văn bản, số ký tự, độ dài model sinh, có bất thường)
_CA_THAT = [
    ("Ừm.", 3, 2.55, True),  # +7.7 sigma — hallucination ngân dài
    ("Đi đi, đi đi đi.", 16, 2.15, False),  # 3.9 sigma nhưng DÀI HỢP LỆ (lặp từ)
    ("Từ nay ta và cháu không liên quan.", 34, 2.89, False),  # 3.4 sigma
    ("Này, này, này.", 14, 1.81, False),  # 3.0 sigma
    ("Sao chú có thể vô ơn bạc nghĩa thế", 34, 2.69, False),
    ("Không", 5, 0.33, False),  # câu 1 âm tiết BÌNH THƯỜNG
    ("Hả?", 3, 0.34, False),  # cùng độ dài văn bản với #88, audio bình thường
    ("Chú...", 6, 0.37, False),
]


@pytest.mark.parametrize(("text", "n_char", "duration_s", "bat_thuong"), _CA_THAT)
def test_lua_chon_dung_tren_du_lieu_that(
    text: str, n_char: int, duration_s: float, bat_thuong: bool
) -> None:
    assert is_abnormally_long(duration_s, n_char) is bat_thuong, text


def test_chi_bat_dung_mot_cau_trong_95() -> None:
    # Toàn bộ phiên chỉ được phép có 1 câu bị bắt — nếu lưới bắt nhiều hơn thì nó đang
    # báo động giả và sẽ làm chậm mọi phiên TTS bằng retry vô ích.
    so_cau_bi_bat = sum(
        1 for _, n, d, _ in _CA_THAT if is_abnormally_long(d, n)
    )
    assert so_cau_bi_bat == 1


def test_mo_hinh_do_dai_ky_vong() -> None:
    # Hệ số chặn ~0.3s: câu RẤT NGẮN không được kỳ vọng ngắn tỉ lệ thuận (chi phí cố định
    # lấy hơi + đuôi âm), nếu không mọi câu 1 âm tiết sẽ bị báo động giả.
    assert expected_speech_seconds(0) == pytest.approx(0.30)
    assert expected_speech_seconds(3) == pytest.approx(0.45)
    assert expected_speech_seconds(34) == pytest.approx(2.00)
    # Đơn điệu tăng theo số ký tự.
    assert expected_speech_seconds(10) < expected_speech_seconds(20)


def test_can_ca_hai_dieu_kien_moi_bao_dong() -> None:
    """Phải vượt CẢ tỉ lệ LẪN phần dư mới báo động.

    [v3.23.227] Ngưỡng dư nay THU NHỎ theo câu ngắn (``min(1.0s, kỳ_vọng * 2)``): câu 2 ký
    tự ngân 1.30s là dài GẤP BA và đủ đè lên câu sau, nên NAY BỊ BẮT — trước đây lọt lưới
    vì phần dư (0.90s) chưa tới ngưỡng tuyệt đối 1.0s. Xem ``test_tts_bugs_v32627.py``.
    """
    assert expected_speech_seconds(2) == pytest.approx(0.40)
    # Câu CỰC NGẮN: ngưỡng dư hiệu dụng = min(1.0, 0.40*2) = 0.80s -> 1.30s bị bắt.
    assert is_abnormally_long(1.30, 2) is True
    # Nhưng ngân NHẸ (2.0x) thì vẫn bỏ qua — không báo động giả.
    assert is_abnormally_long(0.80, 2) is False
    # Câu DÀI: ngưỡng dư vẫn là 1.0s -> hành vi KHÔNG đổi.
    assert is_abnormally_long(4.20, 60) is False  # dư 0.90s, tỉ lệ 1.3x
    # Vượt CẢ HAI -> báo.
    assert is_abnormally_long(2.55, 3) is True


def test_dau_vao_suy_bien_khong_bao_dong() -> None:
    assert is_abnormally_long(0.0, 10) is False
    assert is_abnormally_long(-1.0, 10) is False
    assert is_abnormally_long(5.0, 0) is False  # văn bản rỗng -> lưới khác lo


# ── shorter_take: KHÔNG BAO GIỜ mất thoại ────────────────────────────────────
def test_shorter_take_chon_ban_ngan_hon() -> None:
    dai = np.ones(1000, dtype=np.float32)
    ngan = np.ones(400, dtype=np.float32)
    assert shorter_take(None, dai) is dai  # ứng viên đầu tiên luôn được giữ
    assert shorter_take(dai, ngan) is ngan
    assert shorter_take(ngan, dai) is ngan  # bản dài hơn KHÔNG thay thế bản tốt


def test_shorter_take_giu_ban_cu_khi_bang_nhau() -> None:
    a = np.ones(500, dtype=np.float32)
    b = np.zeros(500, dtype=np.float32)
    assert shorter_take(a, b) is a


# ── Quyết định CÓ CHỦ Ý: không áp lưới cho Edge ──────────────────────────────
def test_edge_khong_ap_luoi_dai_bat_thuong() -> None:
    """Edge KHÔNG dùng lưới này — đây là quyết định có căn cứ, không phải bỏ sót.

    Edge Neural Voices là dịch vụ nối ghép/điều tốc qua tham số ``rate`` của API: cùng
    văn bản + cùng rate cho ra độ dài gần như XÁC ĐỊNH. Không có cơ chế lấy mẫu ngẫu
    nhiên -> không sinh hallucination ngân dài, và retry sẽ trả về kết quả y hệt (chỉ tốn
    thời gian mạng). Lưới chỉ áp cho hai engine neural có sampling: VieNeu và Gemini.
    """
    import pathlib

    edge_src = pathlib.Path(
        "src/subtitles_extractor/infrastructure/tts/edge_tts_adapter.py"
    ).read_text(encoding="utf-8")
    assert "is_abnormally_long" not in edge_src

    for engine in ("vieneu_tts_adapter.py", "gemini_tts_adapter.py"):
        src = pathlib.Path(
            f"src/subtitles_extractor/infrastructure/tts/{engine}"
        ).read_text(encoding="utf-8")
        assert "is_abnormally_long" in src, engine
        assert "shorter_take" in src, engine
