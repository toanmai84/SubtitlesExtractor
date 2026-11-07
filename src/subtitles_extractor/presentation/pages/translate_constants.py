"""[v3.23.689] Hằng số dùng chung của trang Dịch và phần "Dịch cả bộ" (tách khỏi ``translate_page`` để tránh import vòng)."""

from __future__ import annotations

#: [v3.23.621] Model phân tích mặc định (dự phòng: ANALYSIS_FALLBACK_MODEL của view model).
#: [v3.23.622] Tên THẬT thay cho bí danh đã ngừng ``gemini-3.5-flash`` (chính model đã đo).
#: [v3.23.623] 3.8-flash: với prompt có mốc MM:SS cả 3.6 và 3.8 đúng ~100% người nói (tập 1:
#: 100/100%, tập 2: 99/100%), nhưng 3.8 nhanh gấp ~2,4 lần (45 s/74 s so với 108 s/186 s) và
#: ít token suy nghĩ hơn (1,3k/2,0k so với 6,1k/4,8k).
DEFAULT_ANALYSIS_MODEL = "gemini-3.8-flash"

#: [v3.23.609] Lỗi khiến MỌI tập của "Dịch cả bộ" đều hỏng như nhau (chữ thường).
_BATCH_STOP_MARKERS = ("chưa sẵn sàng", "api key gemini chưa hợp lệ", "google-genai")

__all__ = ["_BATCH_STOP_MARKERS", "DEFAULT_ANALYSIS_MODEL"]
