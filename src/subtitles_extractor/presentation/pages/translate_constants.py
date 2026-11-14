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

# Map tên ngôn ngữ hiển thị → mã ISO ngắn để đặt tên file (<tên>.<mã>.srt).
_LANGUAGE_CODES = {
    "Vietnamese": "vi", "English": "en", "Chinese (Simplified)": "zh",
    "Chinese (Traditional)": "zh", "Japanese": "ja", "Korean": "ko",
    "Thai": "th", "French": "fr", "German": "de", "Spanish": "es",
    "Portuguese": "pt", "Russian": "ru", "Indonesian": "id",
    "Italian": "it", "Arabic": "ar",
}

# [v3.23.694] Chuyển từ translate_page (U6): dùng chung với các phần đã tách.
# Danh sách ngôn ngữ đích phổ biến (tên đầy đủ — dùng trực tiếp trong prompt).
_TARGET_LANGUAGES = [
    "Vietnamese", "English", "Chinese (Simplified)", "Chinese (Traditional)",
    "Japanese", "Korean", "Thai", "French", "German", "Spanish",
    "Portuguese", "Russian", "Indonesian", "Italian", "Arabic",
]


# Danh sách model CHỈ gồm các model đang còn quota MIỄN PHÍ (theo bảng rate-limit thực
# tế của tài khoản, tháng 6/2026). Các model limit 0/0 (Gemini 2 Flash, 2 Flash Lite,
# 2.5 Pro, 3.1 Pro) KHÔNG còn free nên KHÔNG đưa vào.
# Xếp theo RPD giảm dần (RPD cao = dịch được nhiều/ngày, ưu tiên cho bản free).
#   gemini-3.1-flash-lite : RPM 15 · TPM 250K · RPD 500  — KHUYẾN NGHỊ (RPD cao nhất)
#   gemini-3.5-flash-lite : RPD 500 (người dùng xác nhận 10/2026; tài liệu Google không công bố RPD theo
#                           model). [v3.23.635] Đo thật tập 1-2 phim mẫu (3 lượt, chấm mù, dịch 3 giai đoạn):
#                           3.1-flash-lite điểm 30 và ~250 s/lượt; 3.5-flash-lite 42-52 điểm và ~510-545 s/lượt
#                           (không suy nghĩ ở mức low/medium, hay 503) → 3.1 là mặc định dịch.
#   gemini-3.8-flash       : MỚI NHẤT (GA 2/9/2026) — hạn mức xem trong AI Studio
#   gemini-3.6-flash        : token-efficient, chất lượng cao (bản đo 100% người nói)
#   gemini-2.5-flash(-lite) : đời cũ, chỉ còn phục vụ tài khoản đã dùng trước đây
# [v3.23.622] KHÔNG liệt kê gemini-3.5-flash / gemini-3.7-flash: tài liệu (deprecations.md,
# 8/10/2026) ghi đã ngừng và tự định tuyến sang 3.6-flash / 3.8-flash, không ngày đảm bảo.
# Tên cũ đã lưu trong cài đặt vẫn chạy (combo cho gõ tay) — xem domain.value_objects.model_alias.
# [v3.23.724] Lấy từ DANH MỤC model văn bản (miễn phí trước, rồi trả phí) — đúng bảng AI Studio 14/11/2026.
from subtitles_extractor.infrastructure.translation.gemini_model_catalog import default_model_ids  # noqa: E402

_DEFAULT_MODELS = default_model_ids()

#: [v3.23.667] Giá trị "không chuyển model dự phòng" (khớp ``model_health.NO_FALLBACK``).
NO_FALLBACK_CHOICE = "-"
#: Model Flash đầy đủ mặc định cho các giai đoạn tinh chỉnh (STYLE/LOCALIZE).
#: [v3.23.635] Đổi từ ``gemini-3.6-flash`` (20 request/ngày/key) sang ``gemini-3.1-flash-lite`` (500): một phim
#: 3649 dòng cần ~92 lô mỗi giai đoạn tinh chỉnh → ~184 request cho phong cách + bản địa hoá, trong khi
#: 3 key chỉ có 60 request/ngày với model Flash đầy đủ. Đo thật tập 1-2 (3 lượt, chấm mù) cả ba giai đoạn
#: bằng 3.1-flash-lite: lỗi nặng 2, điểm 30 — tốt hơn 3.5-flash-lite (42-52). Chưa đo sạch được 3.6-flash
#: (lượt thử bị quá tải 503 nên tự chuyển sang lite). Vẫn chọn được 3.6/3.8-flash trong ô model.
DEFAULT_REFINE_MODEL = "gemini-3.1-flash-lite"

_STYLE_PRESETS = [
    "Trung tính", "Kiếm hiệp", "Tiên hiệp", "Huyền huyễn", "Khoa học viễn tưởng",
    "Tài liệu", "Thiếu nhi", "Cổ trang", "Hiện đại", "Hài", "Trinh thám",
]

_SETTINGS_ORG = "SubtitlesExtractor"
_SETTINGS_APP = "TranslatePage"

# [v3.7 U5] Trên ngưỡng này, không resize từng dòng (tránh đứng UI), dùng chiều
# cao mặc định ~2 dòng cho mỗi hàng.
# [v3.23.131] Ngưỡng số dòng mà còn gọi resizeRowsToContents(): trên ngưỡng này việc đo
# chiều cao từng dòng word-wrap rất chậm (đứng UI vài giây với phim ~1000+ dòng) → dùng
# chiều cao mặc định cho mượt. Hạ từ 1500 -> 300 vì phim tài liệu/phim bộ thường > 800 dòng.
_RESIZE_ROWS_THRESHOLD = 300

_DEFAULT_ROW_HEIGHT_PX = 44


#: [v3.23.681] Mặc định "chỉ trả dòng cần sửa" cho giai đoạn tinh chỉnh / bản địa hoá (xem CHANGELOG 3.23.681).
EDITS_ONLY_DEFAULT = True

__all__ = [
    "_BATCH_STOP_MARKERS",
    "_LANGUAGE_CODES",
    "DEFAULT_ANALYSIS_MODEL",
    "_TARGET_LANGUAGES",
    "_DEFAULT_MODELS",
    "NO_FALLBACK_CHOICE",
    "DEFAULT_REFINE_MODEL",
    "_STYLE_PRESETS",
    "_SETTINGS_ORG",
    "_SETTINGS_APP",
    "_RESIZE_ROWS_THRESHOLD",
    "_DEFAULT_ROW_HEIGHT_PX",
    "EDITS_ONLY_DEFAULT",
]
