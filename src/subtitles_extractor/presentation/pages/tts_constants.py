"""[v3.23.692] Hằng số dùng chung của trang Lồng tiếng và các lớp trộn tách khỏi nó (tránh import vòng)."""

from __future__ import annotations

from subtitles_extractor.infrastructure.tts.edge_tts_adapter import _EDGE_VOICE_MAP

# Các trường TTSRequest CHỈ Edge đọc (đã quét mã 3 adapter). Khi engine khác chạy,
# debug config phải ĐÁNH DẤU rõ để không gây hiểu nhầm lúc phân tích kết quả.
_EDGE_ONLY_FIELDS = frozenset({
    "edge_concurrency", "timing_strategy", "double_pass",
    "high_quality", "last_line_max_extend_s",
    "min_stretch_ratio", "min_pause_ratio",
    "max_intra_gap_s", "anchor_gap_s", "max_segment_s",
})
# [v3.23.218] ``lead_in_s`` ("Ăn gian đầu") ĐÃ RA KHỎI danh sách trên: VieNeu/Gemini nay
# cũng dùng nó để đọc sớm vào khoảng lặng câu trước -> nén nhẹ hơn, giọng rõ hơn.
# [v3.23.596] ``elastic_timing`` / ``max_drift_s`` / ``comfort_speed_ratio`` cũng ra:
# VieNeu và Gemini nay ngân trễ câu chật (xem ``infrastructure.tts.line_placement``).


_ENGINE_EDGE   = "edge"
_ENGINE_GEMINI = "gemini"
_ENGINE_VIENEU = "vieneu"

# [v3.23.695] Chuyển từ tts_page (U6): dùng chung với các phần đã tách.
#: [v3.23.661] Tốc độ đọc tối đa mặc định: đo thật Edge tập 1–3 (296 dòng) — 1,5× xoá hết dòng đọc quá nhanh (trước: 0,7%
#: dòng >1,5×, có dòng 1,83×) mà chỉ thêm 1 dòng chồng tiếng.
_DEFAULT_MAX_SPEED = 1.5

# [v3.23.206] Ngưỡng CẢNH BÁO lấn (giây): câu chồng sang câu sau quá mức này được tô
# nhãn "Lấn" trong bảng kết quả + lọt bộ lọc "Có vấn đề" -> người dùng scan nhanh câu
# cần rút gọn bản dịch (ca thực tế #62: thoại ~4.9s trong khung 1.28s, kịch trần nén
# vẫn dư 1.26s). Lấn nhỏ hơn thường rơi vào khoảng lặng, vô hại.
_OVERLAP_WARN_S = 0.5

# [v3.23.214] Các tuỳ chọn CHỈ engine Edge đọc (VieNeu/Gemini bỏ qua hoàn toàn — đã
# đối chiếu mã nguồn 3 adapter). Trước đây UI hiện chúng cho MỌI engine -> người dùng
# chỉnh "Elastic timing", "Double Pass"… khi chạy VieNeu và tưởng có tác dụng, thực tế
# vô hiệu; debug config cũng in ra gây hiểu nhầm khi phân tích. Nay: tự vô hiệu hoá +
# ghi rõ "(chỉ Edge)" khi engine khác được chọn.
_EDGE_ONLY_HINT = " — chỉ áp dụng cho Edge TTS"

_ENGINES = [
    (_ENGINE_EDGE,   "🌐 Edge TTS",        "Microsoft Edge Neural TTS (online, Tiếng Việt)"),
    (_ENGINE_GEMINI, "✨ Gemini TTS",       "Google Gemini Neural TTS (online, 30 voices)"),
    (_ENGINE_VIENEU, "🇻🇳 VieNeu-TTS",      "TTS tiếng Việt on-device, voice cloning (offline)"),
]

_EDGE_LANGUAGES = list(_EDGE_VOICE_MAP.keys())
_EDGE_LANGUAGE_DISPLAY = {
    "vi-VN": "🇻🇳 Tiếng Việt", "zh-CN": "🇨🇳 Trung (Giản thể)",
    "zh-TW": "🇹🇼 Trung (Phồn thể)", "en-US": "🇺🇸 English (US)",
    "en-GB": "🇬🇧 English (UK)", "ja-JP": "🇯🇵 日本語",
    "ko-KR": "🇰🇷 한국어", "fr-FR": "🇫🇷 Français",
    "es-ES": "🇪🇸 Español", "de-DE": "🇩🇪 Deutsch", "th-TH": "🇹🇭 Thai",
}

__all__ = [
    "_EDGE_ONLY_FIELDS",
    "_ENGINE_EDGE",
    "_ENGINE_GEMINI",
    "_ENGINE_VIENEU",
    "_DEFAULT_MAX_SPEED",
    "_OVERLAP_WARN_S",
    "_EDGE_ONLY_HINT",
    "_ENGINES",
    "_EDGE_LANGUAGES",
    "_EDGE_LANGUAGE_DISPLAY",
]
