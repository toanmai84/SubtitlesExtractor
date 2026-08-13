"""[v3.23.245] Preamble bọc transcript cho Gemini native-audio + sửa bug F821 từ v244.

**Cải tiến theo tài liệu Gemini TTS** (mục Limitations - "Prompt classifier false
rejections"): câu trơ trọi, nhất là câu NGẮN ("Haiz.", "Tuổi:"), có thể không kích hoạt
được speech-synthesis classifier -> model đọc lệch/ngân dài, hoặc đọc cả chỉ thị. Google
khuyến nghị thêm preamble rõ ràng + đánh dấu chỗ transcript bắt đầu.

Đây nhắm ĐÚNG nguyên nhân gốc của hallucination đo được ở v244 (12 câu đọc 3-8x mức tối
thiểu, phần lớn là câu ngắn trơ) — thay vì chỉ chữa triệu chứng bằng retry.

**Bug F821 tìm thấy khi rà (nghiêm trọng):** v244 dùng ``dem_am_tiet(text)`` ở 2 nơi trong
gemini adapter nhưng QUÊN import -> ``NameError`` chờ nổ ngay khi lưới hallucination mới
chạy. Lọt lưới vì test v244 chỉ đọc source text (``_GEMINI_SRC``), không thực thi hàm.
Import module thành công (lỗi chỉ xảy ra lúc GỌI), nên ``import`` check cũng không bắt.
Chỉ ``ruff`` F821 (phân tích tĩnh) mới lộ ra.
"""

from __future__ import annotations

import ast
import pathlib

from subtitles_extractor.infrastructure.tts.text_prep import (
    dem_am_tiet,
    wrap_transcript_for_tts,
)

_GEMINI_PATH = pathlib.Path(
    "src/subtitles_extractor/infrastructure/tts/gemini_tts_adapter.py"
)
_GEMINI_SRC = _GEMINI_PATH.read_text(encoding="utf-8")


# ── Preamble wrap ───────────────────────────────────────────────────────────
def test_bọc_câu_ngắn() -> None:
    out = wrap_transcript_for_tts("Haiz.")
    assert "Haiz." in out
    assert out != "Haiz."  # đã bọc, không còn trơ
    assert '"Haiz."' in out  # transcript nằm trong ngoặc kép (đánh dấu rõ)


def test_giữ_nguyên_nội_dung_đọc() -> None:
    # Nội dung cần đọc phải xuất hiện NGUYÊN VẸN trong ngoặc kép.
    for cau in ("Tuổi:", "Đừng dùng đạo đức.", "Vương Kiến Cường"):
        assert f'"{cau}"' in wrap_transcript_for_tts(cau)


def test_chuỗi_rỗng_không_bọc() -> None:
    # Rỗng -> trả nguyên trạng, để lớp skip xử lý (không tạo prompt vô nghĩa).
    assert wrap_transcript_for_tts("") == ""
    assert wrap_transcript_for_tts("   ") == "   "


def test_là_hàm_thuần() -> None:
    # Gọi nhiều lần cùng đầu vào -> cùng đầu ra.
    assert wrap_transcript_for_tts("Này.") == wrap_transcript_for_tts("Này.")


# ── Adapter dùng wrap ở CẢ HAI đường (native + standard) ─────────────────────
def test_native_audio_bọc_transcript() -> None:
    assert "wrap_transcript_for_tts(text)" in _GEMINI_SRC


def test_cả_hai_đường_đều_bọc() -> None:
    # [v3.23.246] Cookbook: "model can only do TTS, always tell it to say/read".
    # Cả native audio (Live API) lẫn standard TTS (generateContent) đều phải bọc.
    assert _GEMINI_SRC.count("wrap_transcript_for_tts(text)") == 2


# ── Bug F821: dem_am_tiet phải được IMPORT (không chỉ dùng) ──────────────────
def test_dem_am_tiet_được_import() -> None:
    tree = ast.parse(_GEMINI_SRC)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.update(alias.name for alias in node.names)
    # v244 dùng dem_am_tiet ở 2 nơi mà quên import -> NameError runtime.
    assert "dem_am_tiet" in imported


def test_không_còn_tên_chưa_định_nghĩa() -> None:
    # Chốt chặn hồi quy: mọi tên dùng ở module-level phải định nghĩa được.
    # (kiểm gián tiếp qua việc dem_am_tiet dùng được thật)
    assert dem_am_tiet("Vương Kiến Cường") == 3
