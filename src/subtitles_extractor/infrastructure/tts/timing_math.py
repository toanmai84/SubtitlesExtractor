"""Toán thời lượng & tốc độ đọc cho TTS — DÙNG CHUNG cho mọi engine (hàm THUẦN).

[v3.23.220] Trước đây toàn bộ nhóm hàm này nằm trong ``vieneu_tts_adapter`` (một ADAPTER
cụ thể) nhưng lại được ``gemini_tts_adapter`` import ngược ở 6 chỗ. Hệ quả: mỗi lần sửa
VieNeu là một rủi ro thầm lặng cho Gemini, và không có nơi nào là "nguồn sự thật" hiển
nhiên cho các quy tắc thời lượng. Tách về module thuần: KHÔNG phụ thuộc SDK/engine, chỉ
là số học -> dễ kiểm thử và là điểm neo cho kỷ luật ĐỒNG BỘ giữa ba engine.

Bốn quy tắc cốt lõi (đã nghiệm thu trên dữ liệu thật):

* :func:`effective_available_seconds` — khung của câu = khung gốc + phần gap tới câu sau.
* :func:`lead_in_seconds` — "ăn gian đầu" ĐÚNG MỨC CẦN vào khoảng lặng câu trước để lại.
* :func:`total_speed_ratio` — tốc độ tổng áp lên audio (base_speed, trần người dùng,
  trần chất lượng 2.0).
* :func:`master_length_samples` / :func:`fit_limit_samples` — biên của master track và
  giới hạn cắt theo cấu hình chồng tiếng.
"""

from __future__ import annotations

__all__ = [
    "ABNORMAL_LENGTH_MIN_EXCESS_S",
    "ABNORMAL_LENGTH_RATIO",
    "ABNORMAL_SHORT_EXCESS_FACTOR",
    "ABNORMAL_VS_FLOOR_RATIO",
    "CHARS_PER_SYLLABLE",
    "EDGE_MIN_BASE_S",
    "EDGE_MIN_PER_SYLLABLE_S",
    "GEMINI_BASE_OVERHEAD_S",
    "GEMINI_MIN_BASE_S",
    "GEMINI_MIN_PER_SYLLABLE_S",
    "GEMINI_PER_CHAR_S",
    "GENERATION_CAP_FACTOR",
    "GENERATION_CAP_FLOOR_S",
    "MASTER_TAIL_PAD_S",
    "MAX_LEAD_IN_S",
    "MIN_CHAR_BUDGET",
    "MIN_SPEECH_BASE_S",
    "MIN_SPEECH_PER_SYLLABLE_S",
    "QUALITY_STRETCH_CAP",
    "SPEECH_BASE_OVERHEAD_S",
    "SPEECH_PER_CHAR_S",
    "SYLLABLE_BUDGET_REF_SPEED",
    "TRANSLATION_REF_SPEED",
    "VIENEU_MIN_BASE_S",
    "VIENEU_MIN_PER_SYLLABLE_S",
    "compute_fit_stretch_ratio",
    "effective_available_seconds",
    "expected_speech_seconds",
    "fit_limit_samples",
    "generation_time_cap_seconds",
    "is_abnormally_long",
    "is_abnormally_long_vs_floor",
    "lead_in_seconds",
    "master_length_samples",
    "min_speech_seconds",
    "readable_char_budget",
    "readable_syllable_budget",
    "stretch_ratio_cap",
    "syllable_budget_to_chars",
    "total_speed_ratio",
    "window_below_engine_floor",
]

# [v3.23.196] Trần nén CHẤT LƯỢNG cho time-stretch giọng nói. Nén quá 2x bằng
# phase-vocoder/WSOLA làm tan năng lượng formant -> giọng thành tiếng gió thều thào
# "như không có tiếng" (người dùng xác nhận trên câu nén 2.4-3.0x). Thà đọc RÕ ở 2x và
# cắt tối thiểu phần dư còn hơn "đọc đủ" mà không nghe được gì.
QUALITY_STRETCH_CAP = 2.0

# [v3.23.218] Trần "ăn gian đầu": đọc sớm tối đa 0.25s khi câu trước đã đọc xong (tận
# dụng khoảng lặng để nén nhẹ hơn). Giữ nhỏ để tiếng không lệch xa khẩu hình.
MAX_LEAD_IN_S = 0.25

# [v3.23.197] Đệm đuôi master (giây): chứa câu cuối được nới khung (max_gap_use 2s) và
# phần audio TRÀN tự nhiên khi cho phép chồng tiếng (không cắt) — tránh chặt biên mảng.
MASTER_TAIL_PAD_S = 5.0

# [v3.23.221] Mô hình độ dài giọng nói kỳ vọng — hồi quy trên 94 câu VieNeu thật (tiếng
# Việt): do_dai ~ 0.302 + 0.0486 * so_ky_tu (sai số chuẩn 0.273s). Dùng để phát hiện
# model "ngân dài" bất thường, KHÔNG dùng để lập lịch.
SPEECH_BASE_OVERHEAD_S = 0.30
SPEECH_PER_CHAR_S = 0.05

# [v3.23.232] MÔ HÌNH ĐỘ DÀI PHẢI THEO TỪNG ENGINE — hằng số ở trên hiệu chỉnh trên
# **VieNeu**, dùng cho Gemini là SAI.
#
# Hồi quy trên 95 câu Gemini thật (gemini-2.5-flash-native-audio):
#     do_dai = 0.839 + 0.0499 x so_ky_tu   (sai số chuẩn 0.725s)
#
# Chi phí cố định mỗi câu **0.84s** so với 0.30s của VieNeu (+178%) — Gemini native audio
# vào câu chậm rãi hơn hẳn. Tốc độ mỗi ký tự thì gần như y hệt (+3%).
#
# Hậu quả của việc dùng nhầm hằng số VieNeu cho Gemini: lưới "audio dài bất thường" bắt
# **6/95 câu OAN** ("Tu vi,", "Haiz,", "Khụ khụ.", "Ừm."…) -> mỗi câu một lượt retry gọi
# API TỐN TIỀN, và bản "ngắn nhất" được chọn lại có thể là bản đọc vội, kém tự nhiên.
# Với hằng số đúng: 0 câu bị bắt oan, mà ca thảm hoạ thật (>= 3x kỳ vọng) vẫn bị chặn.
GEMINI_BASE_OVERHEAD_S = 0.84
GEMINI_PER_CHAR_S = 0.05

# [v3.23.237] MÔ HÌNH BIÊN DƯỚI THEO ÂM TIẾT — nền tảng đúng cho câu hỏi "audio NGẮN NHẤT
# mà engine có thể sinh là bao nhiêu?".
#
# Các hằng số ở trên hồi quy vào TRUNG BÌNH và dùng SỐ KÝ TỰ. Đo lại trên 95 câu Gemini
# thật cho thấy cả hai lựa chọn đó đều sai:
#
#   ==================================  ========
#   cách mô hình hoá                    R²
#   ==================================  ========
#   trung bình theo KÝ TỰ               0.069
#   trung bình theo ÂM TIẾT             0.059
#   **BIÊN DƯỚI theo ÂM TIẾT**          **0.894**
#   ==================================  ========
#
# Vì sao:
#   1. Tiếng Việt ĐƠN ÂM TIẾT — thời gian đọc tỉ lệ với số âm tiết, không phải số ký tự.
#      "nghiêng" (7 ký tự) đọc mất đúng bằng "ta" (2 ký tự): một nhịp. Số ký tự mỗi âm
#      tiết dao động 2-6 (median 4.2) -> dùng ký tự là tự bơm nhiễu vào mô hình.
#   2. Gemini có phương sai KHỔNG LỒ: cùng 2 âm tiết, "Vương thúc," đọc 0.46s còn "Tu vi:"
#      ngân 3.56s (chênh 7,7 lần). Phần phía trên biên dưới là ngân/kéo dài NGẪU NHIÊN —
#      hồi quy vào trung bình chính là mô hình hoá cái nhiễu đó.
#
# Biên dưới (p10 mỗi nhóm âm tiết) thì tăng gần tuyến tính và ổn định:
#
#     do_dai_toi_thieu = 0.231 + 0.217 x so_am_tiet
GEMINI_MIN_BASE_S = 0.23
GEMINI_MIN_PER_SYLLABLE_S = 0.22

# Hằng số biên dưới FALLBACK — dùng khi nơi gọi không truyền hằng số riêng của engine.
# Cả ba engine nay đều có hằng số ĐO THẬT riêng (EDGE_*, GEMINI_*, VIENEU_*). Cặp này chỉ
# còn là mặc định an toàn cho code chưa chỉ định engine.
#
# CẢNH BÁO: KHÔNG dùng cặp này cho ngân sách DỊCH — ngân sách phải theo engine CHẬM NHẤT
# (Gemini). Ngân sách chặt không hại engine nhanh (chỉ thở thoải mái hơn), nhưng ngân sách
# lỏng làm engine chậm tràn khung.
MIN_SPEECH_BASE_S = 0.20
MIN_SPEECH_PER_SYLLABLE_S = 0.20

# [v3.23.242] Biên dưới ĐÃ ĐO cho Edge (giọng vi-VN-NamMinhNeural) trên 95 câu thật:
# do_dai_toi_thieu = 0.132 + 0.217 x so_am_tiet, R²=**0.995** (Edge tất định nên còn ổn
# định hơn Gemini R²=0.89). Đáng chú ý: hệ số mỗi âm tiết TRÙNG Gemini (0.217) — cùng nhịp
# đọc — nhưng Edge vào câu NHANH hơn (chi phí cố định 0.13s vs Gemini 0.23s).
EDGE_MIN_BASE_S = 0.13
EDGE_MIN_PER_SYLLABLE_S = 0.217

# [v3.23.243] Biên dưới ĐÃ ĐO cho VieNeu (giọng Doan) trên 95 câu thật:
# do_dai_toi_thieu = 0.245 + 0.196 x so_am_tiet, R²=**0.983**. VieNeu vào câu CHẬM NHẤT
# (0.245s) nhưng mỗi âm tiết NHANH NHẤT (0.196s). Điểm cắt với Gemini ở n=0.67 âm tiết ->
# với mọi câu thật (n>=1) GEMINI vẫn chậm nhất, nên ngân sách dịch theo Gemini (v239) được
# xác nhận đúng bằng cả BA engine đo thật, không còn giả định nào.
VIENEU_MIN_BASE_S = 0.245
VIENEU_MIN_PER_SYLLABLE_S = 0.196

# [v3.23.256] GHI CHÚ điều tra v3 Turbo: ban đầu đo biên dưới v3 Turbo ~0.513+0.406×n (gấp
# ~2.1x standard) và tưởng model đọc chậm. ĐIỀU TRA SÂU (cài vieneu 3.2.3 thật, đọc source
# v3turbo.py) phát hiện GỐC RỄ khác hẳn: v3 Turbo xuất **48kHz** nhưng ``infer`` trả array
# THUẦN (không kèm sr), app lại mặc định 24kHz -> audio 48kHz bị coi là 24kHz -> phát chậm
# NỬA tốc độ + app tính thời lượng GẤP ĐÔI -> nén 2x kịch trần -> méo tiếng. Bằng chứng:
# chia đôi mô hình đo (0.257+0.203×n) KHỚP standard (0.245+0.196×n). Vậy v3 Turbo
# KHÔNG chậm hơn — chỉ bị đọc sai sr. Đã sửa ở ``_to_mono_pipeline_rate`` (đọc
# ``engine.sample_rate``).
# KHÔNG cần hằng số riêng cho v3 Turbo: sau khi sửa sr, nó cùng nhịp với standard.

# [v3.23.238] Quy đổi âm tiết <-> ký tự cho ngân sách DỊCH. Model dịch đếm ký tự đầu ra,
# nhưng giới hạn VẬT LÝ là âm tiết -> tính ngân sách theo âm tiết (đúng bản chất) rồi quy
# sang ký tự để gửi model. Đo trên bản dịch Việt thật: median 4.17 ký tự/âm tiết.
CHARS_PER_SYLLABLE = 4.2

# Tốc độ tham chiếu cho ngân sách âm tiết — ĐÂY LÀ MỘT QUYẾT ĐỊNH ĐÁNH ĐỔI, không phải
# con số đo được. Ngân sách càng chặt thì TTS càng ít chồng tiếng, nhưng model dịch càng
# bị ép cắt nghĩa (bẫy v222).
#
# Toan chọn cán cân "cân bằng" = khoảng 5/95 dòng bị ép ngắn hơn bản dịch hiện tại.
#
# [v3.23.239] ref phải đi CÙNG hằng số biên dưới đang dùng. Ngân sách này nay dùng hằng số
# GEMINI (engine CHẬM NHẤT đã đo) thay cho hằng số tạm 0.20/0.20 — vì bản dịch dùng chung
# cho cả ba engine nên phải an toàn cho engine chậm nhất. Hằng số Gemini chặt hơn, nên để
# GIỮ NGUYÊN cán cân 5 dòng của Toan, ref phải nới từ 1.7 lên 1.9:
#
#     (dưới hằng số Gemini, đo trên 95 dòng thật)
#     ref=1.7  ->  14/95 dòng bị ép
#     ref=1.8  ->   9/95 dòng bị ép
#     ref=1.9  ->   5/95 dòng bị ép   <- giữ đúng cán cân Toan đã chọn
#     ref=2.0  ->   5/95 dòng bị ép
#     ref=2.1  ->   2/95 dòng bị ép
#
# Con số ref đổi CHỈ để giữ nguyên kết quả thực tế dưới hằng số đúng — không đổi cán cân.
SYLLABLE_BUDGET_REF_SPEED = 1.9

# Ngưỡng coi audio là DÀI BẤT THƯỜNG: phải vượt CẢ HAI (bội số kỳ vọng và phần dư tuyệt
# đối). Hiệu chỉnh trên dữ liệu thật: bắt đúng ca hallucination duy nhất (#88 "Ừm.":
# 5.7x kỳ vọng, dư 2.10s), không chạm câu dài HỢP LỆ do lặp từ (#75: 2.0x, dư 1.07s).
ABNORMAL_LENGTH_RATIO = 3.0
ABNORMAL_LENGTH_MIN_EXCESS_S = 1.0

# [v3.23.244] Ngưỡng cho lưới hallucination MỚI (so với BIÊN DƯỚI theo âm tiết, không phải
# trung bình theo ký tự). Đo trên 95 câu Gemini thật: x2.5 bắt 12/12 câu hallucination gây
# lấn (89% tổng lấn) mà không đụng câu đọc bình thường (audio ≈ 1-2x biên dưới).
ABNORMAL_VS_FLOOR_RATIO = 2.5

# [v3.23.227] Với câu CỰC NGẮN, ngưỡng dư tuyệt đối 1.0s là QUÁ LỎNG: "Ơ," (kỳ vọng
# 0.40s) ngân 1.20s là dài GẤP BA, đủ để đè lên câu sau, nhưng phần dư chỉ 0.80s -> lọt
# lưới. Đo thực: câu 1-4 ký tự có tỉ lệ hallucination **50%** (VieNeu dùng LLM backbone,
# prompt cực ngắn -> phân phối sinh phương sai lớn), nên đây đúng là nhóm cần bắt chặt
# nhất. Ngưỡng dư hiệu dụng = min(1.0s, kỳ_vọng nhân hệ_số) -> câu dài giữ nguyên hành vi.
ABNORMAL_SHORT_EXCESS_FACTOR = 2.0

# [v3.23.227] Trần thời lượng cho MỘT lượt sinh (chặn ca thảm hoạ như "Kiều Kiều à," ->
# 32.10s). Rất rộng: câu hợp lệ chỉ ~1.0-1.2x kỳ vọng nên không bao giờ chạm trần 6x.
GENERATION_CAP_FACTOR = 6.0
GENERATION_CAP_FLOOR_S = 3.0

# Dung sai chống lỗi làm tròn nhị phân khi so sánh ở ĐÚNG ngưỡng.
_EPS = 1e-9

# [v3.23.223] Tốc độ đọc THAM CHIẾU khi tính ngân sách ký tự cho tầng dịch.
#
# v222 đặt 1.0 ("dòng đạt ngân sách sẽ đọc ở tốc độ tự nhiên, không nén chút nào") —
# THẬN TRỌNG QUÁ MỨC và đã trả giá đắt: trần siết tới 55/95 dòng, model buộc phải CẮT
# NGHĨA để đạt chỉ tiêu (đo trên bản dịch thật: "cháu đã nói về chú thế nào," -> "Chú?";
# "Không phải chú không cho cháu cơ hội." -> "Cho cơ hội," — mất cả phủ định kép).
#
# Mục tiêu ĐÚNG không phải "không nén" mà là "không nén QUÁ 2.0x" (ngưỡng tan formant,
# đo ở v201). Thang suy giảm đo được (v216): nén 1.1x mất 5% sắc phụ âm | 1.6x mất 8% |
# 2.0x mất 19%. Nghĩa là nén tới ~1.6x VẪN NGHE TỐT -> đặt trần ở đó, chỉ ép rút gọn
# những dòng thật sự bất khả thi (16/95 thay vì 55/95).
#
# Nguyên tắc bất di bất dịch: NGHĨA quan trọng hơn độ mượt của giọng. Người xem thà nghe
# giọng hơi gấp mà HIỂU ĐÚNG, còn hơn nghe rõ mà nghĩa sai/mất.
TRANSLATION_REF_SPEED = 1.6
# Sàn ngân sách: khung cực ngắn (đo thực: 9/95 dòng có khung < 0.54s — không đủ đọc nổi
# 8 ký tự) vẫn phải cho model đủ chỗ giữ nghĩa; phần thiếu để TTS mượn gap/nén.
MIN_CHAR_BUDGET = 6


def master_length_samples(
    last_end_s: float,
    sr: int,
    media_duration_s: float | None,
    tail_pad_s: float = MASTER_TAIL_PAD_S,
) -> int:
    """[v3.23.207] Độ dài master track (hàm thuần): ĐÚNG thời lượng video nếu biết.

    Bug người dùng báo: file TTS dài hơn video 3.53s (master = end câu cuối 220.52 +
    đệm 5s = 225.52 vs video 221.99) -> mux vào video bị lệch thời lượng. Khi
    ``media_duration_s`` có: master (và file xuất) dài chính xác bằng video; audio câu
    cuối tràn quá biên video bị cắt tại biên (video đã hết hình — mux kiểu gì cũng cắt)
    và được đánh dấu was_truncated trung thực. Khi không biết: hành vi cũ.

    Args:
        last_end_s: Mốc kết thúc câu phụ đề cuối (giây).
        sr: Tần số lấy mẫu.
        media_duration_s: Thời lượng video gốc (giây) hoặc None.
        tail_pad_s: Đệm đuôi khi không biết thời lượng video.

    Returns:
        Số sample của master track.
    """
    if media_duration_s is not None and media_duration_s > 0:
        return round(media_duration_s * sr)
    return int((last_end_s + tail_pad_s) * sr)


def fit_limit_samples(
    effective_available_s: float,
    max_overlap_ms: int,
    allow_audio_overlap: bool,
    sr: int,
) -> int | None:
    """[v3.23.197] Giới hạn cắt audio theo cấu hình chồng tiếng (hàm thuần).

    Người dùng bật "Cho phép chồng tiếng" nghĩa là ƯU TIÊN TRỌN NỘI DUNG: audio dài hơn
    khung được TRÀN tự nhiên sang khoảng kế tiếp (master là mảng cộng dồn nên chồng
    tiếng vốn được hỗ trợ) — KHÔNG cắt (bug cũ: chỉ đọc ``max_overlap_ms`` = 0 -> cắt
    đúng khít khung dù đã bật chồng tiếng, làm mất chữ cuối câu).

    Args:
        effective_available_s: Khung hiệu dụng của câu (giây).
        max_overlap_ms: Trần lấn (ms) khi KHÔNG cho phép chồng tiếng.
        allow_audio_overlap: True = cho phép chồng tiếng (không cắt).
        sr: Tần số lấy mẫu.

    Returns:
        Số sample tối đa được giữ, hoặc ``None`` = không giới hạn (tràn tự do).
    """
    if allow_audio_overlap:
        return None
    return int((effective_available_s + max_overlap_ms / 1000.0) * sr)


def lead_in_seconds(
    start_sec: float,
    prev_audio_end_sec: float,
    max_lead_s: float = MAX_LEAD_IN_S,
    needed_lead_s: float | None = None,
) -> float:
    """[v3.23.218] Số giây được phép ĐỌC SỚM trước mốc phụ đề (hàm thuần).

    Câu trước thường đọc xong SỚM hơn mốc câu kế tiếp, để lại một khoảng lặng. Tận dụng
    khoảng đó (bắt đầu sớm một chút) giúp câu hiện tại có khung rộng hơn -> nén NHẸ hơn
    -> giọng rõ hơn. Chỉ ăn vào chỗ TRỐNG THẬT (không bao giờ đè lên audio câu trước) và
    KHÔNG dời câu sau (nên không gây hiệu ứng domino như phương án dời mốc đã bị bác bỏ
    ở v204).

    [v3.23.219] Chỉ ăn gian ĐÚNG MỨC CẦN (``needed_lead_s``). Bug v218: mọi câu đều ăn
    gian tối đa, kể cả 79/95 câu vốn đã vừa khung ở tốc độ cơ bản -> toàn bộ tiếng lệch
    TRƯỚC phụ đề/khẩu hình 250ms -> người dùng nghe "không đồng bộ". Nay câu nào không
    cần nén thêm thì giữ NGUYÊN mốc phụ đề.

    Args:
        start_sec: Mốc bắt đầu câu theo phụ đề (giây).
        prev_audio_end_sec: Thời điểm audio câu TRƯỚC kết thúc thật (giây); 0.0 nếu là
            câu đầu.
        max_lead_s: Trần ăn gian đầu (giây) — giữ nhỏ để tiếng không lệch xa khẩu hình.
        needed_lead_s: Số giây thực sự CẦN để khỏi phải nén quá tốc độ cơ bản; None =
            lấy tối đa (hành vi cũ, dùng cho test biên).

    Returns:
        Số giây được đọc sớm (>= 0.0, <= max_lead_s).
    """
    free_gap = start_sec - max(0.0, prev_audio_end_sec)
    lead = min(max_lead_s, free_gap)
    if needed_lead_s is not None:
        lead = min(lead, needed_lead_s)
    return max(0.0, lead)


def total_speed_ratio(
    audio_duration_s: float,
    available_s: float,
    base_speed: float,
    max_speed: float,
    quality_cap: float = QUALITY_STRETCH_CAP,
) -> float:
    """[v3.23.215] Tốc độ đọc TỔNG áp lên audio (hàm thuần) — ngữ nghĩa ĐÚNG.

    Bug: VieNeu/Gemini chỉ nén audio khi nó DÀI HƠN khung (``fit_ratio``);
    ``base_speed`` chỉ dùng để gán NHÃN ``speed_used``. Hệ quả: người dùng đặt "Tốc độ
    cơ bản 1.3" nhưng câu vừa khung (đa số — median speed ghi nhận = đúng base) phát ở
    tốc độ GỐC 1.0x, chậm hơn 30%% so với mong đợi, mà báo cáo vẫn ghi 1.30x.

    Ngữ nghĩa đúng: audio luôn được đọc TỐI THIỂU ở ``base_speed``; nếu vẫn không vừa
    khung thì nén thêm cho vừa, chặn bởi ``max_speed`` (trần người dùng) và
    ``quality_cap`` (ngưỡng vật lý 2.0 — nén hơn nữa làm tan formant, đo ở v201).

    Args:
        audio_duration_s: Độ dài audio gốc từ model (giây).
        available_s: Khung hiệu dụng của câu (giây).
        base_speed: Tốc độ đọc cơ bản người dùng đặt (x).
        max_speed: Trần tốc độ tổng người dùng đặt (x).
        quality_cap: Trần nén giữ độ rõ giọng (x).

    Returns:
        Tỉ lệ nén áp lên audio gốc (>= 1.0; 1.0 = giữ nguyên tốc độ model).
    """
    fit_ratio = audio_duration_s / available_s if available_s > 0.0 else 1.0
    desired = max(float(base_speed), fit_ratio)
    ceiling = min(float(max_speed), float(quality_cap))
    return max(1.0, min(desired, max(1.0, ceiling)))


def effective_available_seconds(
    start_sec: float,
    end_sec: float,
    next_start_sec: float | None,
    guard_gap_s: float = 0.10,
    max_gap_use_s: float = 2.0,
) -> float:
    """[v3.23.194] Khung hiệu dụng của câu = khung gốc + PHẦN GAP tới câu sau (hàm thuần).

    Giữa các câu phụ đề thường có khoảng lặng (gap) — đo thực trên 498 câu: 272 vị trí có
    gap >100ms, median 1.79s. Audio của câu TRÀN vào gap là tự nhiên (không đè lời câu
    sau). Tận dụng gap giúp GIẢM MẠNH tỉ lệ nén thời gian -> giọng ít bị gấp/méo (trước
    đây 31 câu phải nén >2x nghe kém vì chỉ dùng khung gốc).

    Args:
        start_sec: Mốc bắt đầu câu (giây).
        end_sec: Mốc kết thúc khung gốc của câu (giây).
        next_start_sec: Mốc bắt đầu câu KẾ TIẾP; None nếu là câu cuối.
        guard_gap_s: Khoảng an toàn chừa lại TRƯỚC câu sau (không đọc sát lời câu sau).
        max_gap_use_s: Trần phần gap được dùng (tránh câu ngắn "nuốt" cả đoạn lặng dài
            làm giọng lệch xa phụ đề).

    Returns:
        Độ dài khung hiệu dụng (giây), luôn >= khung gốc (end - start).
    """
    base = max(0.0, end_sec - start_sec)
    if next_start_sec is None:
        # Câu cuối: cho nới nhẹ bằng trần gap (không có câu sau để đè).
        return base + max_gap_use_s
    gap = next_start_sec - end_sec - guard_gap_s
    usable_gap = min(max(0.0, gap), max_gap_use_s)
    return base + usable_gap


def compute_fit_stretch_ratio(
    audio_duration_s: float,
    available_s: float,
    max_speed: float,
    tolerance_s: float = 0.05,
) -> float:
    """[v3.23.192] Tính tỉ lệ time-stretch để audio VỪA khung phụ đề (hàm thuần).

    .. deprecated:: 3.23.215
        Không còn trong luồng chạy — ``total_speed_ratio`` đã gộp cả trần ngữ nghĩa lẫn
        trần chất lượng và áp ``base_speed`` đúng cách. Giữ cho test lịch sử; KHÔNG dùng
        cho code mới.

    Args:
        audio_duration_s: Độ dài audio engine sinh ra (giây).
        available_s: Độ dài khung phụ đề khả dụng (giây).
        max_speed: Tốc độ nén tối đa cho phép (lần tốc độ gốc).
        tolerance_s: Dung sai — vượt ít hơn mức này thì không nén.

    Returns:
        Tỉ lệ stretch (>1 = nén nhanh hơn); 1.0 nếu không cần nén.
    """
    if available_s <= 0.0 or audio_duration_s <= 0.0:
        return 1.0
    if audio_duration_s <= available_s + tolerance_s:
        return 1.0
    ratio = audio_duration_s / available_s
    return min(ratio, max_speed)


def expected_speech_seconds(
    char_count: int,
    base_overhead_s: float = SPEECH_BASE_OVERHEAD_S,
    per_char_s: float = SPEECH_PER_CHAR_S,
) -> float:
    """[v3.23.221] Độ dài audio KỲ VỌNG của một câu theo số ký tự (hàm thuần).

    Mô hình tuyến tính hiệu chỉnh bằng hồi quy trên 94 câu VieNeu thật (giọng "Doan",
    tiếng Việt): ``do_dai ~ 0.302 + 0.0486 * so_ky_tu``, sai số chuẩn 0.273s. Hệ số
    chặn (~0.3s) phản ánh chi phí cố định đầu/cuối câu (lấy hơi, đuôi âm) — vì thế câu
    RẤT NGẮN không được kỳ vọng ngắn tỉ lệ thuận.

    Dùng làm MỐC SO SÁNH để phát hiện model "ngân dài" bất thường
    (:func:`is_abnormally_long`), KHÔNG dùng để lập lịch — lập lịch luôn theo audio thật.

    Args:
        char_count: Số ký tự của văn bản sẽ đọc (sau tiền xử lý).
        base_overhead_s: Chi phí cố định mỗi câu (giây).
        per_char_s: Thời lượng trung bình mỗi ký tự (giây).

    Returns:
        Độ dài audio kỳ vọng (giây), luôn > 0.
    """
    return base_overhead_s + per_char_s * max(0, char_count)


def generation_time_cap_seconds(
    char_count: int,
    factor: float = GENERATION_CAP_FACTOR,
    floor_s: float = GENERATION_CAP_FLOOR_S,
    base_overhead_s: float = SPEECH_BASE_OVERHEAD_S,
    per_char_s: float = SPEECH_PER_CHAR_S,
    syllable_count: int | None = None,
    min_base_s: float = MIN_SPEECH_BASE_S,
    min_per_syllable_s: float = MIN_SPEECH_PER_SYLLABLE_S,
) -> float:
    """[v3.23.227] Trần THỜI LƯỢNG cho một lượt sinh audio (hàm thuần).

    Lưới :func:`is_abnormally_long` chỉ phát hiện được ngân dài SAU KHI model đã sinh
    xong. Đo trên log thật: một câu 12 ký tự ("Kiều Kiều à,") khiến VieNeu sinh **32.10
    giây** audio — tức ~30 giây CPU đốt vô ích trước khi ta biết là hỏng.

    Nếu SDK cho phép giới hạn thời lượng sinh, truyền trần này vào để chặn thảm hoạ ngay
    từ đầu. Trần đặt RẤT RỘNG (mặc định 6x kỳ vọng, sàn 3s) vì mục tiêu là chặn ca thảm
    hoạ, KHÔNG phải kiểm soát chất lượng — câu hợp lệ chỉ dài ~1.0-1.2x kỳ vọng, còn cách
    trần rất xa, nên không có rủi ro cắt cụt giọng.

    [v3.23.262] Ưu tiên tính theo ÂM TIẾT nếu ``syllable_count`` > 0: đơn vị phát âm
    thật, không bị dấu câu làm loãng (vd "Chú..." 6 ký tự nhưng 1 âm). Ký tự vẫn dùng
    làm fallback khi không có âm tiết (tương thích ngược).

    Args:
        char_count: Số ký tự văn bản sẽ đọc (fallback khi không có âm tiết).
        factor: Bội số so với thời lượng kỳ vọng.
        floor_s: Sàn tối thiểu (giây) — câu cực ngắn vẫn phải có chỗ thở.
        syllable_count: Số âm tiết (ưu tiên nếu > 0). None -> dùng ký tự.
        min_base_s: Chi phí vào câu của biên dưới âm tiết (theo engine).
        min_per_syllable_s: Chi phí mỗi âm tiết của biên dưới (theo engine).

    Returns:
        Trần thời lượng (giây) cho một lượt sinh.
    """
    if syllable_count is not None and syllable_count > 0:
        expected = min_speech_seconds(syllable_count, min_base_s, min_per_syllable_s)
    else:
        expected = expected_speech_seconds(char_count, base_overhead_s, per_char_s)
    return max(floor_s, expected * factor)


def min_speech_seconds(
    syllable_count: int,
    min_base_s: float = MIN_SPEECH_BASE_S,
    min_per_syllable_s: float = MIN_SPEECH_PER_SYLLABLE_S,
) -> float:
    """Audio NGẮN NHẤT engine có thể sinh cho ngần ấy âm tiết (hàm thuần).

    Khác :func:`expected_speech_seconds` (hồi quy vào TRUNG BÌNH — dùng để phát hiện bất
    thường). Hàm này mô tả **biên dưới**: model đọc gọn hết mức thì mất bao lâu. Đó mới là
    con số cần dùng khi hỏi *"câu này có khả năng vừa khung không?"*.

    Args:
        syllable_count: Số âm tiết (dùng ``text_prep.dem_am_tiet``).
        min_base_s: Chi phí cố định vào câu ở nhịp nhanh nhất.
        min_per_syllable_s: Chi phí mỗi âm tiết ở nhịp nhanh nhất.

    Returns:
        Thời lượng tối thiểu (giây).
    """
    return min_base_s + min_per_syllable_s * max(0, syllable_count)


def window_below_engine_floor(
    available_s: float,
    max_ratio: float,
    min_base_s: float = MIN_SPEECH_BASE_S,
    min_per_syllable_s: float = MIN_SPEECH_PER_SYLLABLE_S,
) -> bool:
    """[v3.23.236] Khung hẹp tới mức engine không thể vừa dù đọc một ký tự? (hàm thuần)

    Mỗi engine có một **sàn vật lý**: audio ngắn nhất nó sinh được (chi phí vào câu + một
    ký tự). Với Gemini là ``0.84 + 0.05 = 0.89s``; nén kịch trần 2.0x còn 0.45s. Khung
    nào hẹp hơn thế thì **không bản nào vừa được**, kể cả bản đọc một chữ duy nhất.

    Đo trên SRT gốc thật (95 dòng phim Trung): 3 dòng rơi vào nhóm này::

        沦为      khung 0.20s  ->  "Trở thành"
        便差不多  khung 0.40s  ->  "gần bằng"
        既然是个  khung 0.44s  ->  "Đã là một"

    Với chúng, lấy mẫu lại là **vô nghĩa hoàn toàn** — không phải "khó", mà là bất khả
    thi. Trước đây mỗi dòng vẫn ngốn ~4 lượt gọi API trước khi bỏ cuộc.

    Args:
        available_s: Khung thời gian câu này có (giây).
        max_ratio: Tỉ lệ nén tối đa được phép.
        base_overhead_s: Chi phí cố định vào câu của engine.
        per_char_s: Chi phí mỗi ký tự của engine.

    Returns:
        True nếu khung nằm DƯỚI sàn vật lý của engine.
    """
    if available_s <= 0.0 or max_ratio <= 0.0:
        return False
    san_s = min_speech_seconds(1, min_base_s, min_per_syllable_s)
    return san_s / max_ratio > available_s + _EPS


def exceeds_window_even_compressed(
    audio_duration_s: float,
    available_s: float,
    max_ratio: float,
) -> bool:
    """[v3.23.234] Audio có tràn khung KHÔNG THỂ CỨU dù đã nén hết cỡ? (hàm thuần)

    Khác hẳn :func:`is_abnormally_long`, vốn hỏi *"audio có bất thường so với NHỊP ĐỌC
    của engine không?"*. Câu hỏi ở đây là *"audio có vừa KHUNG PHỤ ĐỀ không?"* — hai
    chuyện hoàn toàn khác nhau, và nhầm lẫn giữa chúng đã gây ra hồi quy ở v232.

    Bối cảnh: Gemini native audio đọc chậm (chi phí cố định 0.84s/câu — đo được, và đã xác
    nhận trên FLAC là NHỊP ĐỌC THẬT chứ không phải khoảng lặng thừa). Nhiều câu vì thế dài
    hơn khung phụ đề tới mức nén kịch trần 2.0x vẫn tràn sang câu sau. Ca thật::

        "ta sẽ không cho cô"  audio 4.24s, khung 0.72s -> nén 2.0x còn 2.12s -> LẤN 1.40s
        "Giới tính"           audio 3.30s, khung 0.48s -> LẤN 1.17s

    Với những câu đó, cách duy nhất còn lại là **lấy mẫu lại và giữ bản ngắn nhất**
    (``shorter_take``): model sinh ngẫu nhiên nên phương sai lớn — đo trên log thật, best-
    of-N rút ngắn được 15-20%.

    Args:
        audio_duration_s: Độ dài audio vừa sinh (giây).
        available_s: Khung thời gian câu này thực sự có (giây).
        max_ratio: Tỉ lệ nén tối đa được phép (đã tính cả trần chất lượng).

    Returns:
        True nếu audio vẫn tràn khung SAU KHI nén hết mức cho phép.
    """
    if audio_duration_s <= 0.0 or available_s <= 0.0 or max_ratio <= 0.0:
        return False
    return audio_duration_s / max_ratio > available_s + _EPS


def is_abnormally_long_vs_floor(
    audio_duration_s: float,
    syllable_count: int,
    ratio_threshold: float = ABNORMAL_VS_FLOOR_RATIO,
    min_base_s: float = MIN_SPEECH_BASE_S,
    min_per_syllable_s: float = MIN_SPEECH_PER_SYLLABLE_S,
) -> bool:
    """[v3.23.244] Hallucination đo so với BIÊN DƯỚI THEO ÂM TIẾT (không phải trung bình).

    Thay thế đúng đắn cho :func:`is_abnormally_long` ở câu hỏi hallucination. Lưới cũ so
    audio với mô hình TRUNG BÌNH THEO KÝ TỰ — mà mô hình đó có R²=0.07 (vô dụng, xem
    :func:`min_speech_seconds`). Hệ quả đo trên FLAC Gemini thật: lưới cũ chỉ bắt 5/12 câu
    hallucination gây lấn, bỏ lọt 7 câu (vd "thật đúng là" đọc 3.64s = 4.1x mức tối thiểu
    mà vẫn thoát vì kỳ vọng trung bình bị thổi cao).

    Câu hỏi đúng: audio có dài gấp nhiều lần mức đọc GỌN NHẤT (biên dưới) không? Biên dưới
    theo âm tiết có R²=0.98 — ổn định, nên bội số so với nó phản ánh đúng mức "ngân lê
    thê". Đo trên 95 câu: ngưỡng x2.5 bắt 12/12 câu lấn (89% tổng lấn), 7 câu "bắt thêm"
    đều đọc dài 2.7-3.3x biên dưới (hallucination nhẹ, khung rộng nên chưa lấn — retry vẫn
    đáng để có bản gọn hơn).

    Args:
        audio_duration_s: Độ dài audio model vừa sinh (đã trim biên lặng).
        syllable_count: Số âm tiết (dùng ``text_prep.dem_am_tiet``).
        ratio_threshold: Bội số so với biên dưới để coi là bất thường.
        min_base_s: Chi phí vào câu của biên dưới (theo engine).
        min_per_syllable_s: Chi phí mỗi âm tiết của biên dưới (theo engine).

    Returns:
        True nếu audio dài bất thường so với mức đọc gọn nhất.
    """
    if audio_duration_s <= 0.0 or syllable_count <= 0:
        return False
    floor = min_speech_seconds(syllable_count, min_base_s, min_per_syllable_s)
    if floor <= 0.0:
        return False
    return audio_duration_s > floor * ratio_threshold + _EPS


def is_abnormally_long(
    audio_duration_s: float,
    char_count: int,
    ratio_threshold: float = ABNORMAL_LENGTH_RATIO,
    min_excess_s: float = ABNORMAL_LENGTH_MIN_EXCESS_S,
    base_overhead_s: float = SPEECH_BASE_OVERHEAD_S,
    per_char_s: float = SPEECH_PER_CHAR_S,
) -> bool:
    """[v3.23.221] True nếu audio DÀI BẤT THƯỜNG so với văn bản — dấu hiệu hallucination.

    Model TTS neural (VieNeu/Gemini) lấy mẫu ngẫu nhiên nên thi thoảng "ngân kéo dài" một
    âm tiết. Đo thực trên 95 câu: câu #88 "Ừm." (3 ký tự, kỳ vọng 0.45s) được sinh
    **2.55s** — lệch **+7.7 lần sai số chuẩn**, trong khi câu bất thường thứ nhì chỉ 3.9
    (và đó là "Đi đi, đi đi đi." — dài HỢP LỆ vì lặp từ). Hệ quả dây chuyền: câu này là
    câu DUY NHẤT bị nén kịch trần 2.0x (tan formant) và là câu DUY NHẤT còn đè lên mốc
    câu sau (0.23s).

    Lưới ``is_effectively_silent`` (v205) chỉ bắt dạng lỗi NGƯỢC LẠI (audio câm). Cùng một
    cơ chế sampling sinh ra cả hai dạng, nên cần cả hai lưới.

    Phải thoả **ĐỒNG THỜI** hai điều kiện để tránh báo động giả với câu ngắn (vốn có
    phương sai tương đối lớn):

    1. Dài gấp ``ratio_threshold`` lần kỳ vọng, VÀ
    2. Phần dư tuyệt đối ≥ ``min_excess_s`` (dư ít thì nén hậu kỳ xử lý được, không đáng
       tổng hợp lại).

    Hiệu chỉnh trên dữ liệu thật: chỉ #88 (5.7x, dư 2.10s) bị bắt; #75 (2.0x, dư 1.07s),
    #62 (1.5x) và #69 (1.8x) KHÔNG bị bắt — đúng như mong muốn.

    Args:
        audio_duration_s: Độ dài audio model vừa sinh (giây, đã trim biên lặng).
        char_count: Số ký tự của văn bản.
        ratio_threshold: Bội số so với kỳ vọng để coi là bất thường.
        min_excess_s: Phần dư tối thiểu (giây) để coi là bất thường.
        base_overhead_s: Tham số mô hình kỳ vọng.
        per_char_s: Tham số mô hình kỳ vọng.

    Returns:
        True nếu audio dài bất thường (nên tổng hợp lại và giữ bản ngắn nhất).
    """
    if audio_duration_s <= 0.0 or char_count <= 0:
        return False
    expected = expected_speech_seconds(char_count, base_overhead_s, per_char_s)
    if expected <= 0.0:
        return False
    # [v3.23.227] Ngưỡng dư THU NHỎ theo câu ngắn (xem ABNORMAL_SHORT_EXCESS_FACTOR).
    # ``_EPS`` chống lỗi làm tròn nhị phân ở ĐÚNG ngưỡng: "Ơ," ngân 1.20s có phần dư
    # 1.20 - 0.40 = 0.7999999999999999 trong float64 -> so với ngưỡng 0.8 sẽ trượt oan.
    effective_excess = min(min_excess_s, expected * ABNORMAL_SHORT_EXCESS_FACTOR)
    over_ratio = audio_duration_s >= expected * ratio_threshold - _EPS
    over_excess = (audio_duration_s - expected) >= effective_excess - _EPS
    return over_ratio and over_excess


def readable_char_budget(
    available_s: float,
    ref_speed: float = TRANSLATION_REF_SPEED,
    min_chars: int = MIN_CHAR_BUDGET,
    base_overhead_s: float = SPEECH_BASE_OVERHEAD_S,
    per_char_s: float = SPEECH_PER_CHAR_S,
) -> int:
    """[v3.23.222] Số ký tự tối đa TTS đọc KỊP trong ``available_s`` giây (hàm thuần).

    Dùng làm ngân sách độ dài (``max_chars``) gửi cho model DỊCH: bản dịch dài quá ngân
    sách buộc TTS nén giọng, mà nén > 2x thì formant tan (giọng thều thào).

    Công thức cũ ở tầng dịch giả định tốc độ đọc là HẰNG SỐ (16 ký tự/giây) — sai bản
    chất. Đo thực 95 câu: ``thời_lượng = 0.302 + 0.0486 x số_ký_tự``, tức có **chi phí cố
    định ~0.3s mỗi câu** (lấy hơi, đuôi âm). Hệ quả của giả định sai:

    * Khung 2.0s: công thức cũ cho 32 ký tự, thực tế đọc kịp 35 -> ép model CẮT NGHĨA
      không cần thiết (càng khung dài càng sai: khung 3s lệch tới 35%).
    * Khung 0.20s: công thức cũ cho 8 ký tự (sàn), thực tế **không đủ đọc nổi MỘT ký tự**
      -> ngân sách hứa hẹn điều bất khả thi.

    Đo trên bản dịch thật: 57/95 dòng (60%) vượt ngân sách cũ -> model phớt lờ vì ngân
    sách phi lý. Ngân sách theo công thức này có TỔNG gần y hệt (+1%) nhưng PHÂN BỔ LẠI
    đúng chỗ: nới cho dòng có khung/gap rộng, siết đúng những dòng thật sự chật.

    Args:
        available_s: Khung HIỆU DỤNG của dòng (nên tính bằng
            :func:`effective_available_seconds` — gồm cả gap tới câu sau mà TTS
            được dùng).
        ref_speed: Tốc độ đọc tham chiếu. Mặc định 1.0 = dòng đạt ngân sách sẽ đọc ở tốc
            độ TỰ NHIÊN, không cần nén chút nào (thận trọng: không giả định người dùng
            bật nén).
        min_chars: Sàn ngân sách — khung cực ngắn vẫn phải cho model đủ chỗ giữ nghĩa,
            phần thiếu để TTS mượn gap/nén.
        base_overhead_s: Chi phí cố định mỗi câu (giây).
        per_char_s: Thời lượng trung bình mỗi ký tự (giây).

    Returns:
        Số ký tự tối đa nên dùng (>= ``min_chars``).
    """
    if available_s <= 0.0 or per_char_s <= 0.0:
        return min_chars
    speakable_s = available_s * max(0.1, ref_speed) - base_overhead_s
    return max(min_chars, int(speakable_s / per_char_s))


def readable_syllable_budget(
    available_s: float,
    ref_speed: float = SYLLABLE_BUDGET_REF_SPEED,
    min_syllables: int = 1,
    min_base_s: float = GEMINI_MIN_BASE_S,
    min_per_syllable_s: float = GEMINI_MIN_PER_SYLLABLE_S,
) -> int:
    """[v3.23.238] Số ÂM TIẾT tối đa TTS đọc kịp trong ``available_s`` giây (hàm thuần).

    Nền tảng đúng cho ngân sách dịch: tiếng Việt đơn âm tiết nên thời gian đọc tỉ lệ với
    số âm tiết, không phải số ký tự (xem :func:`min_speech_seconds`). Dùng mô hình BIÊN
    DƯỚI (nhịp đọc gọn nhất) làm nền, rồi ``ref_speed`` cho phép nới thoải mái hơn.

    Args:
        available_s: Khung hiệu dụng của dòng.
        ref_speed: Tốc độ tham chiếu (>1 = cho phép nén nhẹ, ngân sách rộng hơn).
        min_syllables: Sàn — luôn cho ít nhất ngần này âm tiết.
        min_base_s: Chi phí cố định vào câu (nhịp nhanh nhất).
        min_per_syllable_s: Chi phí mỗi âm tiết (nhịp nhanh nhất).

    Returns:
        Số âm tiết tối đa nên dùng (>= ``min_syllables``).
    """
    if available_s <= 0.0 or min_per_syllable_s <= 0.0:
        return min_syllables
    speakable_s = available_s * max(0.1, ref_speed) - min_base_s
    return max(min_syllables, int(speakable_s / min_per_syllable_s))


def syllable_budget_to_chars(
    syllable_budget: int, chars_per_syllable: float = CHARS_PER_SYLLABLE
) -> int:
    """[v3.23.238] Quy ngân sách âm tiết sang ký tự để gửi model dịch (hàm thuần).

    Model dịch nhận ``max_chars`` (đếm ký tự đầu ra), nên ngân sách âm tiết — vốn đúng về
    vật lý — phải quy sang ký tự trước khi gửi.

    Args:
        syllable_budget: Ngân sách âm tiết (từ :func:`readable_syllable_budget`).
        chars_per_syllable: Số ký tự trung bình mỗi âm tiết tiếng Việt.

    Returns:
        Ngân sách ký tự tương đương.
    """
    return max(1, round(syllable_budget * chars_per_syllable))


def stretch_ratio_cap(
    base_speed: float, max_speed: float, quality_cap: float = QUALITY_STRETCH_CAP
) -> float:
    """[v3.23.196] Trần tỉ lệ time-stretch ĐÚNG NGỮ NGHĨA max_speed (hàm thuần).

    .. deprecated:: 3.23.215
        Không còn dùng trong luồng chạy — ``total_speed_ratio`` đã thay thế. Giữ lại cho
        test lịch sử; KHÔNG dùng cho code mới.

    Args:
        base_speed: Tốc độ đọc cơ bản (x), vd 1.3.
        max_speed: Trần tốc độ TỔNG người dùng đặt (x), vd 3.0.
        quality_cap: Trần nén chất lượng (x ratio), mặc định 2.0.

    Returns:
        Trần ratio cho time-stretch, tối thiểu 1.0 (không bao giờ ép giãn ngược).
    """
    semantic_cap = max_speed if base_speed <= 0.0 else max_speed / base_speed
    return max(1.0, min(semantic_cap, quality_cap))
