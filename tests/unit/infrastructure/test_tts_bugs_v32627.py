"""[v3.23.227] Ba bug TTS phát hiện qua rà soát log chạy thật.

Phiên đo (VieNeu, 95 câu) ghi nhận **9 lần retry do hallucination** — nhiều gấp ba các
phiên trước. Phân tích tỉ lệ theo độ dài văn bản::

    1-4 ký tự    :  6 câu ->  3 hallucination  (50%!)
    5-12 ký tự   : 25 câu ->  2                (8%)
    13-20 ký tự  : 34 câu ->  1                (3%)
    >20 ký tự    : 30 câu ->  1                (3%)

Yếu tố quyết định là ĐỘ DÀI VĂN BẢN (giả thuyết "dấu phẩy cuối câu" đã được kiểm chứng
và BÁC BỎ: 11% so với 6% — quá yếu). VieNeu dùng LLM backbone (log: ``llama_context``);
prompt cực ngắn -> ít token điều kiện -> phương sai sinh lớn -> model "trôi".

Ba bug:

1. **Lưới bắt hụt câu cực ngắn.** Điều kiện AND với ngưỡng dư TUYỆT ĐỐI 1.0s quá lỏng cho
   câu ngắn: "Ơ," (kỳ vọng 0.40s) ngân 1.20s là dài GẤP BA, đủ đè lên câu sau, nhưng phần
   dư chỉ 0.80s -> LỌT.
2. **Backoff vô ích.** Retry do CHẤT LƯỢNG audio bị áp cùng backoff tăng dần (1s, 2s, 3s…)
   như retry do LỖI HỆ THỐNG. VieNeu chạy OFFLINE: không rate limit, không lỗi mạng, model
   lấy mẫu ngẫu nhiên -> chờ là lãng phí thuần tuý.
3. **Không có trần thời lượng sinh.** Log: "Kiều Kiều à," (12 ký tự) khiến model sinh
   **32.10 giây** audio — ~30s CPU đốt vô ích trước khi lưới hậu kiểm phát hiện.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from subtitles_extractor.infrastructure.tts.timing_math import (
    expected_speech_seconds,
    generation_time_cap_seconds,
    is_abnormally_long,
)
from subtitles_extractor.infrastructure.tts.vieneu_tts_adapter import VieNeuTtsAdapter


# ── BUG 1: lưới bắt hụt câu cực ngắn ────────────────────────────────────────
@pytest.mark.parametrize(
    ("text", "duration_s", "bat_thuong"),
    [
        ("Ơ,", 1.20, True),  # 3.0x kỳ vọng — v226 LỌT LƯỚI (dư 0.80 < 1.0)
        ("Ồ,", 1.35, True),  # 3.4x — v226 lọt
        ("Ừm.", 2.02, True),  # ca thật trong log
        ("Kiều Kiều à,", 32.10, True),  # ca thảm hoạ trong log
        # KHÔNG được báo động giả:
        ("Ơ,", 0.80, False),  # 2.0x — ngân nhẹ, chấp nhận được
        ("Hả?", 0.34, False),  # bình thường
        ("Đi đi, đi đi đi.", 2.15, False),  # 2.0x nhưng dài HỢP LỆ (lặp từ)
        ("cháu đã nói về chú thế nào,", 2.90, False),  # 1.8x
    ],
)
def test_luoi_bat_dung_ca_ngan(text: str, duration_s: float, bat_thuong: bool) -> None:
    assert is_abnormally_long(duration_s, len(text)) is bat_thuong


def test_hanh_vi_cau_dai_khong_doi() -> None:
    # Với câu dài, ngưỡng dư hiệu dụng vẫn là 1.0s -> hành vi y hệt v226.
    n = len("Không phải chú không cho cháu cơ hội.")  # 36 ký tự, kỳ vọng ~2.1s
    exp = expected_speech_seconds(n)
    assert is_abnormally_long(exp + 0.9, n) is False  # dư < 1.0s -> bỏ qua
    assert is_abnormally_long(exp * 3.0 + 0.1, n) is True


# ── BUG 3: trần thời lượng sinh ─────────────────────────────────────────────
def test_tran_sinh_rat_rong_khong_cat_cut_cau_hop_le() -> None:
    # Câu hợp lệ dài ~1.0-1.2x kỳ vọng -> phải cách trần rất xa (không có rủi ro cắt cụt).
    for text in ("Ơ,", "Kiều Kiều à,", "Không phải chú không cho cháu cơ hội."):
        n = len(text)
        assert generation_time_cap_seconds(n) >= expected_speech_seconds(n) * 5


def test_tran_sinh_chan_duoc_ca_tham_hoa() -> None:
    # "Kiều Kiều à," -> model sinh 32.10s. Trần phải chặn từ rất sớm.
    cap = generation_time_cap_seconds(len("Kiều Kiều à,"))
    assert cap < 8.0
    assert cap * 3 < 32.10  # ca thảm hoạ vượt trần nhiều lần


def test_tran_co_san_cho_cau_cuc_ngan() -> None:
    # Câu 2 ký tự kỳ vọng 0.40s; trần 6x = 2.4s là quá ngặt -> phải có sàn.
    assert generation_time_cap_seconds(2) >= 3.0


# ── BUG 3b: truyền trần vào SDK — AN TOÀN với mọi phiên bản SDK ─────────────
class _SdkKhongHoTro:
    """SDK cũ: ``infer`` không nhận tham số trần."""

    def __init__(self) -> None:
        self.kwargs: dict[str, object] = {}

    def infer(self, text: str, voice: object) -> np.ndarray:
        self.kwargs = {"text": text, "voice": voice}
        return np.ones(100, dtype=np.float32)


class _SdkHoTro:
    """SDK mới: ``infer`` có nhận trần thời lượng."""

    def __init__(self) -> None:
        self.kwargs: dict[str, object] = {}

    def infer(
        self, text: str, voice: object, max_duration_s: float | None = None
    ) -> np.ndarray:
        self.kwargs = {"text": text, "voice": voice, "max_duration_s": max_duration_s}
        return np.ones(100, dtype=np.float32)


def test_sdk_khong_ho_tro_thi_khong_truyen_gi_them() -> None:
    # Nguyên tắc: dò bằng introspection, KHÔNG giả định tên tham số -> SDK cũ chạy y hệt.
    engine = _SdkKhongHoTro()
    adapter = VieNeuTtsAdapter.__new__(VieNeuTtsAdapter)
    adapter._infer_once(engine, "Kiều Kiều à,", {})
    assert set(engine.kwargs) == {"text", "voice"}


def test_sdk_ho_tro_thi_nhan_duoc_tran() -> None:
    from subtitles_extractor.infrastructure.tts.text_prep import dem_am_tiet
    from subtitles_extractor.infrastructure.tts.timing_math import (
        VIENEU_MIN_BASE_S,
        VIENEU_MIN_PER_SYLLABLE_S,
    )

    engine = _SdkHoTro()
    adapter = VieNeuTtsAdapter.__new__(VieNeuTtsAdapter)
    adapter._infer_once(engine, "Kiều Kiều à,", {})
    # [v3.23.262] Trần nay tính theo ÂM TIẾT (nhất quán lưới hallucination), không phải
    # ký tự — câu cùng số âm có cùng trần dù khác dấu câu.
    text = "Kiều Kiều à,"
    assert engine.kwargs["max_duration_s"] == pytest.approx(
        generation_time_cap_seconds(
            len(text),
            syllable_count=dem_am_tiet(text),
            min_base_s=VIENEU_MIN_BASE_S,
            min_per_syllable_s=VIENEU_MIN_PER_SYLLABLE_S,
        )
    )


def test_introspection_chiu_duoc_ham_khong_co_chu_ky() -> None:
    """SDK có ``infer`` không nội soi được -> trả RỖNG, tuyệt đối không ném lỗi.

    Một số hàm C-extension không có ``__text_signature__``; nếu để lỗi thoát ra thì toàn
    bộ việc tổng hợp giọng sập chỉ vì một tính năng phụ (truyền trần thời lượng).
    """

    class _KhongNoiSoiDuoc:
        @property
        def infer(self) -> object:
            raise ValueError("hàm C-extension không có chữ ký")

    assert VieNeuTtsAdapter._supported_infer_params(_KhongNoiSoiDuoc()) == frozenset()


# ── BUG 2: backoff chỉ dành cho LỖI THẬT, không dành cho retry chất lượng ───
def test_khong_cho_khi_retry_do_chat_luong() -> None:
    src = inspect.getsource(VieNeuTtsAdapter._synthesize_with_retry)
    # Có cờ phân biệt hai loại retry, và sleep bị chặn bởi cờ đó.
    assert "wait_before_next" in src
    assert "if wait_before_next and attempt < retry_count - 1:" in src
    # Nhánh exception (lỗi thật) mới bật cờ.
    assert "wait_before_next = True" in src


def test_gemini_cung_phan_biet_hai_loai_retry() -> None:
    # Kỷ luật parity: cùng một lớp bug phải được xét cho MỌI engine. Gemini là API có rate
    # limit nên KHÔNG bỏ chờ hẳn, mà dùng delay CƠ BẢN (không nhân tăng dần) — quyết định
    # khác VieNeu nhưng có căn cứ, không phải sao chép mù.
    import pathlib

    src = pathlib.Path(
        "src/subtitles_extractor/infrastructure/tts/gemini_tts_adapter.py"
    ).read_text(encoding="utf-8")
    assert "quality_retry" in src
    assert "if quality_retry" in src
