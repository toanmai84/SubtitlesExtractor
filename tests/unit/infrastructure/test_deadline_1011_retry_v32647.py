"""[v3.23.247] Lỗi 1011 Deadline: RETRY thay vì bỏ dòng (mất tiếng).

**Bug từ log thật:** Gemini native audio gặp 1011 "Deadline expired before operation
could complete" rải rác vài dòng. Code cũ ``return None`` NGAY khi gặp 1011 -> bỏ luôn
dòng đó -> **mất tiếng** (đúng triệu chứng "âm thanh TTS bị mất nội dung").

**Chẩn đoán (xác nhận qua tài liệu Google):** 1011 Deadline là lỗi SERVER-SIDE TẠM THỜI
(tương đương 503 UNAVAILABLE — server quá tải/chậm phản hồi), KHÔNG phải lỗi vĩnh viễn.
Diễn đàn Google khuyến nghị retry với backoff. Lần thử sau thường thành công.

**Sửa:** nhánh 1011 nay chỉ log rồi để vòng lặp retry tiếp (với backoff tăng dần
``retry_delay_s * (attempt+1)``). Mỗi lần gọi ``_async_native_single`` mở SESSION MỚI
hoàn toàn (``async with client.aio.live.connect``), nên retry không tái dùng session
hỏng. Chỉ bỏ dòng khi đã CẠN lượt retry — khi đó lớp ngoài đánh dấu ``was_skipped=True``
+ ``error_msg`` rõ ràng, không mất tiếng âm thầm.
"""

from __future__ import annotations

import pathlib

_GEMINI_SRC = pathlib.Path(
    "src/subtitles_extractor/infrastructure/tts/gemini_tts_adapter.py"
).read_text(encoding="utf-8")


def test_1011_không_còn_return_none_ngay() -> None:
    # Bug cũ: "1011 Deadline dòng '%s…' — bỏ qua." + return None ngay trong nhánh 1011.
    assert "— bỏ qua." not in _GEMINI_SRC
    # Lấy các dòng CODE (bỏ comment) trong nhánh 1011 tới hết elif.
    lines = _GEMINI_SRC.splitlines()
    start = next(
        i for i, ln in enumerate(lines) if '"1011" in exc_str' in ln
    )
    end = next(
        i for i in range(start + 1, len(lines)) if lines[i].strip().startswith("else:")
    )
    code_lines = [
        ln for ln in lines[start:end] if not ln.strip().startswith("#")
    ]
    # Trong nhánh 1011 (bỏ comment) KHÔNG được có return None trực tiếp.
    assert not any("return None" in ln for ln in code_lines)


def test_1011_có_log_thử_lại() -> None:
    assert "1011 Deadline lần %d/%d" in _GEMINI_SRC
    assert "thử lại" in _GEMINI_SRC


def test_1011_nằm_trong_vòng_retry() -> None:
    # Nhánh 1011 phải nằm trong for-loop retry để backoff + thử lại có tác dụng.
    for_idx = _GEMINI_SRC.find("for attempt in range(max(1, request.retry_count)):")
    e1011_idx = _GEMINI_SRC.find('"1011" in exc_str')
    # Nhánh backoff sau nhánh 1011 (tìm từ vị trí 1011 trở đi).
    backoff_idx = _GEMINI_SRC.find(
        "if attempt < request.retry_count - 1:", e1011_idx
    )
    assert for_idx != -1 and e1011_idx != -1 and backoff_idx != -1
    # 1011 phải nằm SAU khai báo for và TRƯỚC nhánh backoff (trong thân vòng lặp).
    assert for_idx < e1011_idx < backoff_idx


def test_session_mới_mỗi_lần_gọi() -> None:
    # Retry chỉ có ý nghĩa nếu mỗi lần mở session MỚI (không tái dùng session hỏng).
    assert "async with client.aio.live.connect(" in _GEMINI_SRC


def test_cạn_lượt_thì_đánh_dấu_skip_rõ_ràng() -> None:
    # Hết retry: dòng None được đánh dấu skip + có error_msg (không mất tiếng thầm).
    assert "was_skipped=True" in _GEMINI_SRC
    assert "Thất bại sau tất cả lần retry." in _GEMINI_SRC
