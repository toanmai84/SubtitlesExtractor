"""[v3.23.258] Sửa bug double-decode base64 ở Gemini standard TTS (audio hỏng).

**Phát hiện khi cài google-genai 2.12.1 thật + nội soi ``types.Blob``:**
``Blob.data`` là ``Optional[bytes]`` và SDK 2.x **tự b64decode** khi parse response
-> ``inline_data.data`` đã là BYTES audio thô (PCM 16-bit), KHÔNG phải chuỗi base64.

**Bug:** đường standard TTS gọi ``base64.b64decode(inline_data.data)`` LẦN NỮA ->
double-decode -> audio rác hoặc ném lỗi "Invalid base64-encoded string".

**Vì sao chưa lộ:** model TTS mặc định là ``gemini-2.5-flash-preview-tts``
(STANDARD), nhưng
các phiên nghiệm thu trước dùng model native audio -> đi đường Live API
(``chunks.append(idata.data)`` dùng data TRỰC TIẾP, đúng). Chỉ khi chọn model standard
TTS mới gặp bug.

**Sửa:** standard TTS dùng ``inline_data.data`` trực tiếp (như đường native), bỏ
``base64.b64decode`` thừa + bỏ ``import base64`` không còn dùng.
"""

from __future__ import annotations

import base64
import pathlib

import numpy as np

_GEMINI_SRC = pathlib.Path(
    "src/subtitles_extractor/infrastructure/tts/gemini_tts_adapter.py"
).read_text(encoding="utf-8")


def test_không_còn_b64decode_thừa() -> None:
    # Standard TTS KHÔNG được b64decode data (SDK đã tự decode).
    assert "base64.b64decode(" not in _GEMINI_SRC


def test_không_còn_import_base64() -> None:
    # base64 không còn dùng -> không import.
    assert "import base64" not in _GEMINI_SRC


def test_standard_dùng_data_trực_tiếp() -> None:
    # Standard TTS đọc inline_data.data trực tiếp (đồng bộ đường native).
    assert "inline_data.data" in _GEMINI_SRC


def test_double_decode_làm_hỏng_audio() -> None:
    # Minh hoạ HẬU QUẢ của bug: b64decode trên bytes audio thô -> hỏng.
    real_audio = np.array([1000, -2000, 3000, -4000], dtype=np.int16)
    raw_bytes = real_audio.tobytes()  # SDK 2.x trả cái này (đã decode)

    # Dùng trực tiếp -> đúng.
    correct = np.frombuffer(raw_bytes, dtype=np.int16)
    assert np.array_equal(correct, real_audio)

    # b64decode lần nữa -> hỏng (rác hoặc lỗi).
    got_error_or_garbage = False
    try:
        double = base64.b64decode(raw_bytes)
        garbage = np.frombuffer(double, dtype=np.int16)
        got_error_or_garbage = not np.array_equal(garbage, real_audio)
    except Exception:
        got_error_or_garbage = True
    assert got_error_or_garbage


def test_native_audio_vẫn_dùng_data_trực_tiếp() -> None:
    # Đường native (đã đúng từ trước) không bị đụng.
    assert "chunks.append(idata.data)" in _GEMINI_SRC
