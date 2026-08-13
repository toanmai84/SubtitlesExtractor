"""[v3.23.249] Danh sách giọng Gemini CHÍNH THỨC + xử lý audio rỗng.

**Bug giọng bịa (nghiêm trọng):** Phiên trước ``GEMINI_TTS_VOICES`` có 16/30 tên KHÔNG tồn
tại (Perseus, Electra, Polaris, Vega, Rigel, Deneb, Alula, Altair, Tethys, Dione, Ankaa,
Izar, Albireo, Acamar, Aljanah, Seginus) — người dùng chọn phải sẽ lỗi API. Đồng thời
thiếu 16 giọng thật (Zephyr, Algieba, Despina, Autonoe...). Nay đồng bộ đúng 30 giọng
chính thức theo tài liệu Google (GoogleCloudPlatform/generative-ai notebook).

**Phát hiện temperature (đo trên FLAC thật, 2 bộ cùng giọng Orus):**

========  ==========  ================  ============
bộ        lấn tổng    hallucination     câu bị bỏ
========  ==========  ================  ============
Tự động   0.37s       0                 0
temp 0.7  1.05s       0                 2 ("Ơ.", "Hả?")
========  ==========  ================  ============

Temperature thấp làm model DỄ TỪ CHỐI câu ngắn (trả audio rỗng) -> bị bỏ. "Tự động" (mặc
định model) ổn định hơn cho phụ đề (vốn nhiều câu ngắn). Đây là bằng chứng thực nghiệm
khuyên dùng "Tự động".

**Sửa kèm:** khi audio RỖNG (model từ chối), log rõ + thử tắt affective từ lần sau (câu
ngắn đôi khi bị affective làm model "diễn" quá rồi trả rỗng) — thay vì âm thầm retry.
"""

from __future__ import annotations

import pathlib

from subtitles_extractor.infrastructure.tts.gemini_tts_adapter import GEMINI_TTS_VOICES

_GEMINI_SRC = pathlib.Path(
    "src/subtitles_extractor/infrastructure/tts/gemini_tts_adapter.py"
).read_text(encoding="utf-8")

# 30 giọng chính thức (GoogleCloudPlatform/generative-ai notebook).
_OFFICIAL = {
    "Achernar", "Achird", "Algenib", "Algieba", "Alnilam", "Aoede", "Autonoe",
    "Callirrhoe", "Charon", "Despina", "Enceladus", "Erinome", "Fenrir", "Gacrux",
    "Iapetus", "Kore", "Laomedeia", "Leda", "Orus", "Puck", "Pulcherrima", "Rasalgethi",
    "Sadachbia", "Sadaltager", "Schedar", "Sulafat", "Umbriel", "Vindemiatrix", "Zephyr",
    "Zubenelgenubi",
}
# Các tên BỊA đã bị xóa.
_BIA = {
    "Perseus", "Electra", "Polaris", "Vega", "Rigel", "Deneb", "Alula", "Altair",
    "Tethys", "Dione", "Ankaa", "Izar", "Albireo", "Acamar", "Aljanah", "Seginus",
}


def test_đúng_30_giọng() -> None:
    assert len(GEMINI_TTS_VOICES) == 30


def test_mọi_giọng_đều_chính_thức() -> None:
    assert set(GEMINI_TTS_VOICES) == _OFFICIAL


def test_không_còn_giọng_bịa() -> None:
    for bia in _BIA:
        assert bia not in GEMINI_TTS_VOICES, f"{bia} là giọng bịa, phải xóa"


def test_có_zephyr() -> None:
    # Zephyr là giọng phổ biến, trước bị thiếu.
    assert "Zephyr" in GEMINI_TTS_VOICES


def test_giọng_orus_đang_dùng_còn() -> None:
    # Giọng thực tế trong dữ liệu test.
    assert "Orus" in GEMINI_TTS_VOICES


def test_không_trùng_lặp() -> None:
    assert len(GEMINI_TTS_VOICES) == len(set(GEMINI_TTS_VOICES))


# ── Xử lý audio rỗng ────────────────────────────────────────────────────────
def test_log_khi_audio_rỗng() -> None:
    assert "audio RỖNG cho" in _GEMINI_SRC


def test_tắt_affective_khi_audio_rỗng() -> None:
    # Nhánh else (audio rỗng) phải hạ current_affective để lần sau thử không affective.
    idx = _GEMINI_SRC.find("audio RỖNG cho")
    assert idx != -1
    doan = _GEMINI_SRC[idx : idx + 300]
    assert "current_affective = False" in doan
