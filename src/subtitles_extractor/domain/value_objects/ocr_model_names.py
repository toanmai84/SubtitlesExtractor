"""[v3.23.733] Tên mô hình OCR dạng thị giác–ngôn ngữ (VLM) dùng chung giữa các tầng.

PaddleOCR-VL-1.6 (0,9B, Apache-2.0, chạy trên Paddle — không cần torch) là bộ ĐỌC dòng chữ chất lượng cao nhất đo được
trên phụ đề thật (148 dòng, 3 phim: 146/148 đúng, CER 0,26 %; PP-OCRv6 small 143/148, 0,66 %), nhưng chậm hơn nhiều.
Nó không có bộ dò chữ riêng cho từng dòng: ứng dụng dùng bộ DÒ PP-OCRv6 để tìm hộp chữ, rồi VL đọc chữ trong hộp.
"""

from __future__ import annotations

__all__ = [
    "GLM_OCR_QA_MODEL",
    "QA_READER_GLM_OCR",
    "QA_READER_PADDLE_VL",
    "PADDLE_VL_DETECTION_MODEL",
    "PADDLE_VL_DRAFT_RECOGNITION_MODEL",
    "PADDLE_VL_RECOGNITION_MODEL",
    "PADDLE_VL_VERSION",
    "is_vl_recognizer",
]

#: Khoá "phiên bản OCR" trong Cài đặt.
PADDLE_VL_VERSION = "PaddleOCR-VL-1.6"
#: Tên mô hình nhận dạng (đúng tên mô hình chính thức của PaddleX).
PADDLE_VL_RECOGNITION_MODEL = "PaddleOCR-VL-1.6-0.9B"
#: Bộ dò chữ đi kèm (tìm hộp từng dòng).
PADDLE_VL_DETECTION_MODEL = "PP-OCRv6_small_det"
#: Bộ nhận dạng nháp chạy cùng bộ dò (độ tin cậy, đường lùi khi VL lỗi, khoá bộ đệm bản đọc VL).
PADDLE_VL_DRAFT_RECOGNITION_MODEL = "PP-OCRv6_small_rec"

#: [v3.23.745] Khoá Cài đặt "Bộ đọc kiểm định" (gói kiểm định chất lượng của "Xuất dữ liệu thô OCR").
QA_READER_PADDLE_VL = "paddle_vl"
QA_READER_GLM_OCR = "glm_ocr"
#: Nhãn bộ đọc GLM-OCR ghi vào gói kiểm định (bản GGUF chạy bằng llama.cpp — xem ``glm_ocr_runtime``).
GLM_OCR_QA_MODEL = "GLM-OCR (GGUF Q8_0, llama.cpp b12387)"


def is_vl_recognizer(model_name: str | None) -> bool:
    """``model_name`` là bộ nhận dạng VLM (PaddleOCR-VL) chứ không phải mô hình PP-OCR."""
    return bool(model_name) and str(model_name).startswith("PaddleOCR-VL")
