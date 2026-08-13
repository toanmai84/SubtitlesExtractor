"""Adapter Edge TTS — TTS online dùng Microsoft Edge Neural Voices.

Kiến trúc V7.5 Ultimate — Bảo toàn 100% Core Gốc + Nâng cấp DSP & Lập lịch Cấp Điện Ảnh:
- **Song song**: _EDGE_CONCURRENCY requests đồng thời (asyncio.gather + Semaphore)
- **Double Pass**: Pass 1 đo duration, Pass 2 điều tốc chính xác.
- **Ultra DSP (Khảm thêm, không phá vỡ logic gốc)**:
  * VAD Trim (Gọt khoảng lặng vô hình Edge TTS).
  * Phase-Locked WSOLA (Time-stretch ưu tiên bảo vệ thanh quản).
  * Zero-Phase EQ & Split-band De-Esser.
  * Lookahead Soft-Limiter an toàn RAM (làm mượt gain bằng uniform_filter1d).
- **Tuyệt đối an toàn**: Giữ nguyên Fallback Pydub, thuật toán quản lý RAM LUFS (Chunking), và mọi Log hệ thống.
- **[CẢI TIẾN TỐI THƯỢNG] Elastic Timeline**: Smooth Non-linear Catchup, chống giật cục tốc độ đỉnh, chặn chia zero.

Cài đặt: pip install edge-tts soundfile pydub scipy numpy librosa
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import os
import random
from subtitles_extractor.infrastructure.process.hidden_process import no_window_kwargs
import tempfile
import time
from pathlib import Path
import numpy as np

from subtitles_extractor.domain.ports.subtitle_tts_port import (
    SubtitleTTSPort, TTSCancelledError, TTSGenerationError,
    TTSProgressCallback, TTSCancellationCallback, TTSRequest,
    TTSSegmentResult, TTSUnavailableError,
)
from subtitles_extractor.infrastructure.tts.adaptive_limiter import (
    AdaptiveConcurrencyLimiter,
)
from subtitles_extractor.infrastructure.tts.audio_utils import is_effectively_silent
from subtitles_extractor.infrastructure.tts.dsp_primitives import (
    clamp_smoothed_gain as _clamp_smoothed_gain,
)
from subtitles_extractor.infrastructure.tts.dsp_primitives import (
    fit_length_no_silence as _fit_length_no_silence,  # noqa: F401
)
from subtitles_extractor.infrastructure.tts.dsp_primitives import (
    gated_loudness_from_kweighted as _gated_loudness_from_kweighted,  # noqa: F401
)
from subtitles_extractor.infrastructure.tts.dsp_primitives import (
    inter_node_soft_clip as _inter_node_soft_clip,
)
from subtitles_extractor.infrastructure.tts.dsp_primitives import (
    loudness_gain_linear as _loudness_gain_linear,
)
from subtitles_extractor.infrastructure.tts.dsp_primitives import (
    noise_gate_threshold_linear as _noise_gate_threshold_linear,
)
from subtitles_extractor.infrastructure.tts.dsp_primitives import (
    true_peak_chunked_overlap as _true_peak_chunked_overlap,
)
from subtitles_extractor.infrastructure.tts import text_prep as _text_prep
from subtitles_extractor.infrastructure.tts.text_prep import (
    SkipOptions as _SkipOptions,  # noqa: F401
)
from subtitles_extractor.infrastructure.tts.text_prep import (
    has_speakable_content as _has_speakable_content,
)
from subtitles_extractor.infrastructure.tts.text_prep import (
    preprocess_tts_text as _preprocess_tts_text,
)
from subtitles_extractor.infrastructure.tts.text_prep import (
    skip_from_request as _skip_from_request,
)

logger = logging.getLogger(__name__)

# [v3.23.220] Các tên gạch-dưới ở trên là ALIAS tương thích ngược: nguyên hàm DSP và bộ
# tiền xử lý văn bản đã chuyển sang module thuần dùng chung (``dsp_primitives`` /
# ``text_prep``) để cắt vòng phụ thuộc ``audio_mastering -> edge_tts_adapter`` và
# ``vieneu/gemini -> edge_tts_adapter``. Giữ alias để test và monkeypatch hiện có không
# phải sửa; code MỚI nên import thẳng từ module thuần.

_EDGE_VOICE_MAP: dict[str, list[str]] = {
    "vi-VN": ["vi-VN-HoaiMyNeural", "vi-VN-NamMinhNeural"],
    "zh-CN": ["zh-CN-XiaoxiaoNeural", "zh-CN-YunxiNeural", "zh-CN-YunyangNeural"],
    "zh-TW": ["zh-TW-HsiaoChenNeural", "zh-TW-YunJheNeural"],
    "en-US": ["en-US-JennyNeural", "en-US-GuyNeural", "en-US-AriaNeural"],
    "en-GB": ["en-GB-SoniaNeural", "en-GB-RyanNeural"],
    "ja-JP": ["ja-JP-NanamiNeural", "ja-JP-KeitaNeural"],
    "ko-KR": ["ko-KR-SunHiNeural", "ko-KR-InJoonNeural"],
    "fr-FR": ["fr-FR-DeniseNeural", "fr-FR-HenriNeural"],
    "es-ES": ["es-ES-ElviraNeural", "es-ES-AlvaroNeural"],
    "de-DE": ["de-DE-KatjaNeural", "de-DE-ConradNeural"],
    "th-TH": ["th-TH-PremwadeeNeural", "th-TH-NiwatNeural"],
}

# [v3.23.220] Regex tiền xử lý văn bản là ALIAS của module thuần ``text_prep`` (một nguồn
# sự thật cho cả ba engine). Giữ tên cũ tại đây cho test/monkeypatch hiện có.
_ASS_TAG_RE = _text_prep._ASS_TAG_RE
_HTML_TAG_RE = _text_prep._HTML_TAG_RE
_PAREN_RE = _text_prep._PAREN_RE
_SQUARE_RE = _text_prep._SQUARE_RE
_CURLY_RE = _text_prep._CURLY_RE
_MUSIC_PAIR_RE = _text_prep._MUSIC_PAIR_RE
_MULTISPACE_RE = _text_prep._MULTISPACE_RE
_DIALOG_DASHES = _text_prep._DIALOG_DASHES
_MUSIC_NOTE = _text_prep._MUSIC_NOTE
_SPEAKABLE_CHAR_RE = _text_prep._SPEAKABLE_CHAR_RE
_SPEAKER_TAG_RE = _text_prep._SPEAKER_TAG_RE

_EDGE_CONCURRENCY = 16
_EDGE_API_SPEED_MAX = 3.0
_EDGE_API_SPEED_MIN = 0.5
# [v3.23.162] Ngưỡng residual (audio_thực / khung_cần) kích hoạt Pass 2.5 vi chỉnh.
# Hạ 1.25 -> 1.15 (chiều nhanh) để né stretch TRIỆT ĐỂ hơn; thêm chiều CHẬM 0.87
# (audio ngắn hơn khung ~15%) -> giảm rate cho Edge đọc chậm, khỏi giãn DSP.
_CALIBRATE_TRIGGER_FAST = 1.15
_CALIBRATE_TRIGGER_SLOW = 0.87


def _is_silent_audio(audio: np.ndarray) -> bool:
    """[v3.23.210] Audio "coi như im lặng" — delegate sang hàm thuần dùng chung.

    Engine sinh audio CÓ độ dài nhưng toàn im lặng là dạng lỗi thật (đo được ở VieNeu:
    4 câu/video). Lưới ``len(audio) > 0`` để lọt -> mất thoại âm thầm. Dùng chung ngưỡng
    với hai engine kia (:func:`...tts.audio_utils.is_effectively_silent`, RMS < 0.005).

    [v3.23.220] Gọi thẳng module thuần ``audio_utils`` thay vì lazy-import ngược từ
    ``vieneu_tts_adapter`` (adapter <-> adapter, vòng tròn phụ thuộc).

    Args:
        audio: Tín hiệu mono float32 vừa tổng hợp.

    Returns:
        True nếu không có tiếng nghe được (nên retry).
    """
    return is_effectively_silent(audio)


def _stretch_with_pedalboard(
    audio: np.ndarray, sr: int, ratio: float
) -> np.ndarray | None:
    """[v3.23.199] Delegate sang module dùng chung ``time_stretch`` (một nguồn sự thật).

    Giữ tên module-level tại đây để tương thích test/monkeypatch hiện có; logic thật
    nằm ở :func:`subtitles_extractor.infrastructure.tts.time_stretch.stretch_with_pedalboard`
    — nơi MỌI engine (Edge/Gemini/VieNeu) cùng hưởng Rubber Band.

    Args:
        audio: Tín hiệu mono float32.
        sr: Tần số lấy mẫu.
        ratio: Hệ số tốc độ (>1 = nhanh hơn/ngắn lại).

    Returns:
        Audio đã stretch (mono 1D float32), hoặc ``None`` để caller fallback.
    """
    from subtitles_extractor.infrastructure.tts.time_stretch import (
        stretch_with_pedalboard,
    )

    return stretch_with_pedalboard(audio, sr, ratio)


def _should_insert_dialog_pause(
    is_dialog: bool,
    dialog_pause_ms: int,
    speech_dur: float,
    safe_window_dur: float,
    pause_dur: float,
    crowded_ratio: float = 1.30,
) -> bool:
    """[v3.23.169] Quyết định có chèn khoảng nghỉ hội thoại đầu câu hay không.

    Khoảng nghỉ hội thoại (tách hai lượt thoại "- A. / - B.") là trang trí giúp tự
    nhiên hơn. Nhưng khi khung thời gian ĐÃ quá chật (giọng phải nén mạnh để vừa),
    chèn thêm nghỉ sẽ ăn vào khung -> đẩy tỉ lệ nén lên cao hơn -> GIẢM chất lượng
    giọng. Ưu tiên giữ giọng rõ hơn khoảng nghỉ: bỏ nghỉ khi việc chèn khiến khung
    còn lại cho giọng phải nén vượt ``crowded_ratio``.

    Args:
        is_dialog: Câu có phải lượt thoại (bắt đầu bằng gạch đầu dòng) không.
        dialog_pause_ms: Độ dài nghỉ hội thoại cấu hình (mili-giây).
        speech_dur: Thời lượng audio giọng đã tổng hợp (giây).
        safe_window_dur: Khung thời gian "miễn phí" cho câu (giây).
        pause_dur: Độ dài nghỉ sẽ chèn (giây).
        crowded_ratio: Ngưỡng nén coi là "chật" — trên mức này thì bỏ nghỉ.

    Returns:
        True nếu nên chèn nghỉ; False nếu nên bỏ để bảo toàn chất lượng giọng.
    """
    if not is_dialog or dialog_pause_ms <= 0:
        return False
    room_for_speech = safe_window_dur - pause_dur
    if room_for_speech <= 0.05:
        return False  # chèn nghỉ sẽ không còn chỗ cho giọng
    # Nếu sau khi trừ nghỉ, giọng vẫn phải nén vượt ngưỡng chật -> bỏ nghỉ.
    return speech_dur / room_for_speech <= crowded_ratio


def _speech_only_stretch_ratio(
    total_dur: float,
    pause_dur: float,
    target_total_dur: float,
) -> float:
    """[v3.23.173] Tỉ lệ time-stretch áp riêng cho PHẦN GIỌNG (giữ nguyên pause đầu).

    Khi câu hội thoại có khoảng nghỉ im lặng ở đầu, MAX SQUEEZE trước đây stretch CẢ
    khối (nghỉ + giọng) cùng một tỉ lệ -> (1) nghỉ bị co lại sai độ dài mong muốn,
    (2) stretch một đoạn IM LẶNG là vô ích và có thể tạo artifact ở ranh giới
    nghỉ->giọng. Hàm này tính tỉ lệ chỉ cho phần giọng sao cho TỔNG (nghỉ giữ nguyên +
    giọng đã nén) khớp ``target_total_dur``.

    Args:
        total_dur: Tổng thời lượng hiện tại (nghỉ + giọng), giây.
        pause_dur: Thời lượng nghỉ im lặng ở đầu (giữ nguyên), giây.
        target_total_dur: Tổng thời lượng đích mong muốn, giây.

    Returns:
        Tỉ lệ stretch cho phần giọng (>1 = nén nhanh hơn). Trả 1.0 nếu không cần nén
        hoặc phần giọng/đích không hợp lệ.
    """
    speech_dur = total_dur - pause_dur
    target_speech_dur = target_total_dur - pause_dur
    if speech_dur <= 0.001 or target_speech_dur <= 0.001:
        return 1.0
    ratio = speech_dur / target_speech_dur
    return ratio if ratio > 1.0 else 1.0


def _overlap_length_samples(
    region: np.ndarray,
    abs_floor: float = 0.01,
    rel_floor_db: float = -40.0,
) -> int:
    """[v3.23.172] Độ dài (mẫu) vùng câu trước còn tín hiệu đáng kể trong ``region``.

    Xác định tới đâu câu trước còn "đáng kể" để ducking hạ đúng phần chồng. Trước đây
    dùng ngưỡng TUYỆT ĐỐI cứng 0.01 (-40dBFS): khi câu trước nhỏ (giọng thì thầm, hoặc
    khi tắt chuẩn hoá nên giữ biên gốc Edge thấp ~0.008), mọi mẫu dưới ngưỡng -> coi
    như KHÔNG chồng -> KHÔNG duck -> hai giọng cộng cùng độ to -> đục tiếng, mất rõ.
    Nay ngưỡng lấy MAX giữa sàn tuyệt đối và mức tương đối theo đỉnh region (-40dB so
    đỉnh) -> luôn phát hiện đúng vùng chồng dù câu trước to hay nhỏ.

    Args:
        region: Lát master tại vị trí câu mới (chứa đuôi câu trước nếu chồng).
        abs_floor: Sàn biên độ tuyệt đối tối thiểu (chống nhiễu nền cực nhỏ).
        rel_floor_db: Ngưỡng tương đối so đỉnh region (dB, âm).

    Returns:
        Số mẫu tính từ đầu ``region`` tới mẫu đáng kể cuối cùng (0 nếu không có chồng).
    """
    if region.size == 0:
        return 0
    peak = float(np.max(np.abs(region)))
    if peak <= 0.0:
        return 0
    threshold = min(abs_floor, peak * (10.0 ** (rel_floor_db / 20.0)))
    significant = np.nonzero(np.abs(region) > threshold)[0]
    if significant.size == 0:
        return 0
    return int(significant[-1]) + 1


def _voiced_bounds_from_rms(
    rms: np.ndarray,
    noise_floor: float,
    peak_rms: float,
    floor_multiplier: float = 3.5,
    peak_ratio_db: float = -45.0,
    dynamic_guard_db: float = -42.0,
) -> tuple[int, int] | None:
    """[v3.23.171] Tìm biên [đầu, cuối] vùng có tiếng trong chuỗi RMS (an toàn động học).

    Ngưỡng VAD tổ hợp phải ĐỦ THẤP để không cắt nhầm phụ âm cuối nhẹ ('s', 'ch', 'th')
    hay âm tắt dần ở câu có động học lớn. Trước đây ngưỡng lấy ``max(noise_floor×k,
    peak×10^(-45/20))`` — khi câu có một từ RẤT TO, ``peak`` kéo nhánh thứ hai lên cao
    và ``noise_floor`` (percentile toàn cục) cũng bị đội lên -> cắt nhầm đuôi nhỏ (đo
    thực: cắt 230ms đuôi biên 0.015 so đỉnh 1.0). Nay thêm CHẶN TRÊN theo dynamic_guard
    (mặc định -32dB so đỉnh): ngưỡng không bao giờ vượt mức này -> phụ âm nhẹ được giữ.

    Args:
        rms: Chuỗi RMS trượt của audio (mode='valid').
        noise_floor: Sàn nhiễu ước lượng (percentile thấp của rms).
        peak_rms: RMS đỉnh (max của rms).
        floor_multiplier: Hệ số nhân sàn nhiễu cho ngưỡng.
        peak_ratio_db: Ngưỡng tương đối so đỉnh (dB, giá trị âm).
        dynamic_guard_db: CHẶN TRÊN ngưỡng so đỉnh (dB, âm) — bảo vệ đuôi/đầu nhỏ.

    Returns:
        Cặp ``(chỉ_số_đầu, chỉ_số_cuối)`` trong ``rms`` (bao gồm), hoặc None nếu toàn
        bộ dưới ngưỡng (không có tiếng).
    """
    if rms.size == 0 or peak_rms <= 0.0:
        return None
    # Ngưỡng cơ bản: lớn hơn giữa (sàn nhiễu × hệ số) và (mức tương đối so đỉnh).
    base_threshold = max(
        noise_floor * floor_multiplier, peak_rms * (10.0 ** (peak_ratio_db / 20.0))
    )
    # CHẶN TRÊN cứng theo đỉnh: ở câu động học lớn, ``noise_floor`` (percentile thấp)
    # có thể rơi trúng vùng ĐUÔI NHỎ (phụ âm cuối) thay vì im lặng thật -> nhánh
    # ``noise_floor × k`` đội ngưỡng lên và cắt nhầm đuôi. Guard giữ ngưỡng không vượt
    # mức -42dB so đỉnh; phụ âm cuối nhẹ (-30..-40dB) nhờ đó được giữ. Guard chỉ có
    # tác dụng khi đỉnh đủ lớn (động học cao) — với câu đều tiếng, base_threshold vốn
    # đã dưới guard nên không đổi.
    guard = peak_rms * (10.0 ** (dynamic_guard_db / 20.0))
    threshold = min(base_threshold, guard)
    mask = rms > threshold
    if not np.any(mask):
        return None
    voiced = np.nonzero(mask)[0]
    return (int(voiced[0]), int(voiced[-1]))


def _master_track_length_samples(
    last_end_s: float,
    sample_rate: int,
    dialog_pause_s: float,
    last_line_extend_s: float,
    drift_s: float,
    extra_tail_s: float,
    base_tail_s: float = 1.0,
    media_duration_s: float | None = None,
) -> int:
    """[v3.23.164] Số mẫu cấp phát cho master track (đủ chỗ đuôi câu cuối).

    Master phải dài hơn ``last_end_s`` một khoảng đệm đủ để câu cuối — sau khi cộng
    khoảng lặng hội thoại, phần nới cuối và dung sai trôi — KHÔNG bị slice cắt cụt
    (mất chữ cuối phim). Hàm thuần để kiểm thử biên chính xác.

    [v3.23.207] ``media_duration_s`` (thời lượng video gốc) có -> trả ĐÚNG số mẫu của
    video: file xuất bằng đúng thời lượng video, mux không lệch (bug người dùng báo:
    file dài hơn video 3.53s do đệm đuôi).

    Args:
        last_end_s: Mốc kết thúc muộn nhất trong các câu (giây).
        sample_rate: Tần số lấy mẫu (Hz).
        dialog_pause_s: Khoảng lặng hội thoại chèn trước câu (giây).
        last_line_extend_s: Phần nới thêm cho câu cuối (giây).
        drift_s: Dung sai trôi lipsync khi bật elastic_timing (giây).
        extra_tail_s: Đệm đuôi bổ sung khi bật elastic_timing (giây).
        base_tail_s: Đệm đuôi tối thiểu luôn có (giây).

    Returns:
        Số mẫu (>= 1) cho mảng master.
    """
    if media_duration_s is not None and media_duration_s > 0:
        return max(1, int(round(media_duration_s * sample_rate)))
    tail_s = base_tail_s + dialog_pause_s + max(0.0, last_line_extend_s)
    tail_s += max(0.0, drift_s) + max(0.0, extra_tail_s)
    return max(1, int((last_end_s + tail_s) * sample_rate))


def _stretch_ratio_with_extended(
    speech_dur: float,
    desired_speech_dur: float,
    extended_speech_dur: float,
    max_speed: float,
) -> float:
    """[v3.23.163] Tỉ lệ time-stretch CÒN LẠI sau khi mượn thời gian tới câu kế.

    Câu bị kẹp trần rate Edge (không đọc nhanh hơn được) trước đây phải stretch DSP
    theo khung CHẶT (``desired_speech_dur``). Nếu có khoảng trống tới câu kế
    (``extended_speech_dur`` >= desired), cho câu mượn khung rộng hơn -> chỉ stretch
    phần VƯỢT khung rộng -> ratio nhỏ hơn hẳn, nhiều câu về dưới ngưỡng cảnh báo 1.5x.

    Args:
        speech_dur: Thời lượng audio thực (giây).
        desired_speech_dur: Khung chặt theo tốc độ mục tiêu (giây).
        extended_speech_dur: Khung nới rộng tới sát câu kế đã trừ pause/guard (giây).
        max_speed: Trần time-stretch người dùng cho phép.

    Returns:
        ts_ratio trong ``[1.0, max_speed]``; 1.0 nghĩa là KHÔNG cần stretch.
    """
    roomy = max(desired_speech_dur, extended_speech_dur)
    if roomy <= 0.001 or speech_dur <= roomy * 1.02:
        return 1.0
    return min(speech_dur / roomy, max_speed)


def _calibrated_rate(
    api_speed: float,
    actual_dur: float,
    target_dur: float,
    cap: float,
    floor: float = _EDGE_API_SPEED_MIN,
) -> float | None:
    """[v3.23.155] Rate Edge HIỆU CHỈNH theo thời lượng ĐO ĐƯỢC sau Pass 2.

    Quan hệ rate<->thời lượng của Edge KHÔNG tuyến tính chính xác, nên câu đã tổng
    hợp ở ``api_speed`` vẫn có thể lệch khung. Trả về rate mới = api_speed x
    (dài_thực / dài_cần) kẹp trong ``[floor, cap]`` để tổng hợp lại; None khi không
    cần/không thể hiệu chỉnh.

    [v3.23.162] ĐỐI XỨNG hai chiều để né time-stretch TRIỆT ĐỂ (yêu cầu người dùng:
    câu nào còn hiệu chỉnh được bằng Edge thì PHẢI dùng Edge, tránh cả stretch nén
    LẪN stretch giãn — cả hai đều giảm chất lượng):
    - Chiều NHANH (audio dài hơn khung, residual > 1): tăng rate cho Edge đọc nhanh
      hơn, kích hoạt khi residual >= ``_CALIBRATE_TRIGGER_FAST`` và rate chưa kịch trần.
    - Chiều CHẬM (audio ngắn hơn khung, residual < 1): GIẢM rate cho Edge đọc chậm
      hơn (Edge phát tốt tới rate ~0.5), kích hoạt khi residual <=
      ``_CALIBRATE_TRIGGER_SLOW`` và rate chưa kịch sàn -> khỏi phải giãn DSP.

    Args:
        api_speed: rate đã dùng ở Pass 2.
        actual_dur: thời lượng audio thực đo được (giây).
        target_dur: thời lượng khung cần khớp (giây).
        cap: trần rate API (thường ``_EDGE_API_SPEED_MAX``).
        floor: sàn rate API (mặc định ``_EDGE_API_SPEED_MIN``).

    Returns:
        Rate mới để tổng hợp lại, hoặc ``None`` nếu không cần/không thể.
    """
    if target_dur <= 0.05 or actual_dur <= 0.0:
        return None
    residual = actual_dur / target_dur
    proposed = api_speed * residual
    if residual >= _CALIBRATE_TRIGGER_FAST:  # audio dài hơn khung -> đọc nhanh hơn
        if api_speed >= cap - 0.01:
            return None
        new_rate = min(proposed, cap)
        return new_rate if new_rate > api_speed + 0.04 else None
    if residual <= _CALIBRATE_TRIGGER_SLOW:  # audio ngắn hơn khung -> đọc chậm hơn
        if api_speed <= floor + 0.01:
            return None
        new_rate = max(proposed, floor)
        return new_rate if new_rate < api_speed - 0.04 else None
    return None
_TS_QUALITY_WARN = 1.5
_DUCK_FLOOR = 0.4
_GAP_GUARD_S = 0.08
_LAST_EXTEND_S = 3.0
_ANCHOR_GAP_S = 0.7
_MAX_SEGMENT_S = 10.0
_COMFORT_SPEED_RATIO = 1.25
_COMFORT_STRETCH_MIN = 0.90
_MIN_PAUSE_RATIO = 0.35
_DRIFT_DISASTER_S = 2.0
_CATCHUP_SPEED_BOOST = 1.3
_SPEED_EMA_ALPHA = 0.65    # Hệ số làm mượt tốc độ giữa các câu liền nhau (EMA blend).
_MIN_MICRO_PAUSE_S = 0.05  # "Khoảng thở sinh lý" tối thiểu, giữ cả khi ép thời gian.

# [V11.6] Dung sai thị giác lấy theo "Dung sai cộng dồn" (max_drift_s) người dùng
# đặt, NHƯNG kẹp trần ở mức khẩu hình còn chấp nhận được (lệch quá mức này sẽ thấy
# rõ mất đồng bộ). max_drift vẫn dùng đầy đủ cho catch-up ở chế độ dồn mốc.
_LIPSYNC_LATE_CAP_S = 0.5  # trần dung sai ngân trễ dùng cho lipsync/overlap
_LIPSYNC_EARLY_S = 0.15    # fallback nếu không có lead_in_s
_MAX_SPEED_JUMP = 0.35     # giới hạn nhảy tốc độ giữa 2 câu liền nhau (chống giật)


def _rate_from_speed(speed: float) -> str:
    speed = max(0.1, min(speed, _EDGE_API_SPEED_MAX))
    pct = int((speed - 1.0) * 100)
    return f"+{pct}%" if pct >= 0 else f"{pct}%"


class EdgeTTSAdapter(SubtitleTTSPort):
    def __init__(self) -> None:
        # [SỬA] Bỏ ThreadPoolExecutor tạo ở __init__ (leak threads mỗi lần khởi tạo
        # adapter, không bao giờ shutdown, và thực tế không dùng — mọi tác vụ nền
        # đều qua asyncio.to_thread). Đây là một nguyên nhân treo khi chạy nhiều lần.
        self._ts_heavy_count: int = 0
        self._has_librosa: bool = False

    @staticmethod
    def _check_librosa() -> bool:
        try:
            import librosa  # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    def _compute_extended_windows(
        events: list,
        gap_guard_s: float = _GAP_GUARD_S,
        last_extend_s: float = 0.0,
    ) -> dict[int, float]:
        n = len(events)
        windows: dict[int, float] = {}
        order = sorted(range(n), key=lambda i: events[i].start_sec)
        for pos, idx in enumerate(order):
            ev = events[idx]
            strict = ev.end_sec - ev.start_sec
            if pos + 1 < n:
                next_start = events[order[pos + 1]].start_sec
                extended = next_start - ev.start_sec - gap_guard_s
            else:
                extended = strict + max(0.0, last_extend_s)
            windows[idx] = max(strict, extended)
        return windows

    @staticmethod
    def _schedule_timeline(
        events: list,
        p1_speech_durs: dict[int, float],
        request: TTSRequest,
        gap_guard_s: float = _GAP_GUARD_S,
    ) -> dict[int, tuple[float, float, float]]:
        """Lập lịch timeline co giãn với chiến lược 2 tầng + catch-up phi tuyến.

        - lipsync : bám cả mốc đầu & cuối từng câu, vượt max mới mượn khoảng lặng.
        - balanced: bám mốc đầu, mượn khoảng lặng tới câu kế (đọc thong thả).
        - smooth  : dồn mốc, KHÔNG chồng tiếng, catch-up phi tuyến mượt mà.
        - Khắc phục triệt để lỗi chia cho số âm hoặc 0 khi tính khoảng trống.
        """
        n = len(events)
        if n == 0:
            return {}

        order = sorted(range(n), key=lambda i: events[i].start_sec)
        pause_s = request.dialog_pause_ms / 1000.0
        max_drift = request.max_drift_s
        base_speed = request.base_speed
        max_speed = request.max_speed

        anchor_gap_s = getattr(request, "anchor_gap_s", _ANCHOR_GAP_S)
        max_segment_s = getattr(request, "max_segment_s", _MAX_SEGMENT_S)
        comfort_ratio = getattr(request, "comfort_speed_ratio", _COMFORT_SPEED_RATIO)
        min_pause_ratio = getattr(request, "min_pause_ratio", _MIN_PAUSE_RATIO)
        max_intra_gap_s = getattr(request, "max_intra_gap_s", 0.5)
        allow_audio_overlap = getattr(request, "allow_audio_overlap", True)
        min_stretch_ratio = getattr(request, "min_stretch_ratio", 0.75)
        strategy = getattr(request, "timing_strategy", "lipsync")

        # [V11.6] Dung sai thị giác lấy ĐÚNG từ cấu hình người dùng, KHÔNG hardcode:
        # - "Dung sai cộng dồn" (max_drift_s) = tối đa lời thoại được dời TRỄ so mốc
        #   gốc → chính là dung sai ngân trễ cho lipsync.
        # - "Ăn gian đầu" (lead_in_s) = tối đa cho câu chớm SỚM.
        lipsync_late = min(max_drift, _LIPSYNC_LATE_CAP_S) if allow_audio_overlap else 0.0
        lipsync_early = max(0.0, getattr(request, "lead_in_s", 0.0))

        if strategy == "smooth":
            allow_audio_overlap = False
            lipsync_late = 0.0

        # Đệm cuối khung dùng cho fit/guard (dung sai dời-trễ người dùng cho phép).
        tail_slack = lipsync_late

        # Sàn giãn thực tế: không chậm hơn cả ngưỡng người dùng (min_stretch_ratio)
        # lẫn ngưỡng chống lè nhè (_COMFORT_STRETCH_MIN) — lấy giá trị CAO hơn.
        stretch_floor = base_speed * max(min_stretch_ratio, _COMFORT_STRETCH_MIN)

        # [v3.23.148] pause_for phải tính is_dialog trên VĂN BẢN ÂM THANH THẬT (đã bỏ
        # tag người nói + đã áp skip ngoặc/ký hiệu) — GIỐNG HỆT nơi chèn pause ở
        # _build_results_from_map. Trước đây thiếu skip + strip_speaker_tag nên
        # "[Nam:] - Chào" hoặc "(cười) - Chào" (bật skip ngoặc) bị coi là KHÔNG hội
        # thoại → lịch không dành chỗ cho khoảng lặng → audio dài hơn lịch → cắt oan.
        _skip_opts = _skip_from_request(request)

        def pause_for(idx: int) -> float:
            _text, is_dialog = _preprocess_tts_text(
                events[idx].text, request.clean_tags, _skip_opts,
                strip_speaker_tag=True,
            )
            return pause_s if is_dialog else 0.0

        # ── BƯỚC 1: Phân đoạn theo neo đồng bộ ──────────────────────────────
        raw_segments: list[list[int]] = []
        current: list[int] = [order[0]]
        for pos in range(1, n):
            prev_idx, cur_idx = order[pos - 1], order[pos]
            gap = events[cur_idx].start_sec - events[prev_idx].end_sec
            if gap >= anchor_gap_s:
                raw_segments.append(current)
                current = [cur_idx]
            else:
                current.append(cur_idx)
        raw_segments.append(current)

        def split_long(seg: list[int]) -> list[list[int]]:
            if len(seg) <= 1:
                return [seg]
            span = events[seg[-1]].end_sec - events[seg[0]].start_sec
            if span <= max_segment_s:
                return [seg]
            best_k, best_gap = -1, -1.0
            for k in range(1, len(seg)):
                gap = events[seg[k]].start_sec - events[seg[k - 1]].end_sec
                if gap > best_gap:
                    best_gap, best_k = gap, k
            if best_k <= 0:
                return [seg]
            return split_long(seg[:best_k]) + split_long(seg[best_k:])

        segments: list[list[int]] = []
        for seg in raw_segments:
            segments.extend(split_long(seg))

        # ── BƯỚC 2: Lập lịch từng segment ───────────────────────────────────
        schedule: dict[int, tuple[float, float, float]] = {}
        cursor = 0.0
        resync_count = 0
        max_drift_seen = 0.0
        compressed_segs = 0
        catchup_segs = 0
        overlap_total_s = 0.0
        overlap_count = 0
        peak_speed = base_speed
        prev_eff_speed = base_speed  # động lượng EMA (mượt chuỗi tốc độ thực tế)
        n_segs = len(segments)

        for seg_pos, seg in enumerate(segments):
            first_idx, last_idx = seg[0], seg[-1]
            seg_orig_start = events[first_idx].start_sec

            if seg_pos + 1 < n_segs:
                next_seg_start = events[segments[seg_pos + 1][0]].start_sec
                natural_end = next_seg_start - gap_guard_s
            else:
                tail = max(0.0, request.last_line_max_extend_s)
                natural_end = events[last_idx].end_sec + tail

            total_speech = sum(p1_speech_durs.get(i, 0.0) for i in seg)
            total_pause = sum(pause_for(i) for i in seg)
            inter_guard = gap_guard_s * (len(seg) - 1)

            room_at_orig = (natural_end - seg_orig_start) - total_pause - inter_guard
            overloaded = total_speech > room_at_orig
            lead = 0.0
            if overloaded and request.lead_in_s > 0:
                free_before = seg_orig_start - (cursor + gap_guard_s)
                # [v3.23.229] CHỈ ĂN GIAN ĐỦ BÙ PHẦN THIẾU — port fix v219 từ VieNeu.
                #
                # Trước đây lấy lead TỐI ĐA (`min(lead_in_s, free_before)`), nên cụm chỉ
                # thiếu 30ms vẫn bị dời sớm nguyên 250ms. Đo trên FLAC Edge thật: **34/95
                # câu bị dời đúng kịch trần 250ms**, và **36/95 câu bị dời sớm dù bản thân
                # đã vừa khung** -> tiếng vang TRƯỚC khẩu hình, đúng hiện tượng "không
                # đồng bộ" mà người dùng từng báo ở VieNeu (bug v218).
                #
                # VieNeu đã sửa ở v219 nhưng chưa bao giờ được port sang Edge — đây chính
                # là dạng bug parity mà kỷ luật "một fix phải xét cho cả ba engine" sinh ra
                # để chặn.
                needed_lead = max(0.0, total_speech - room_at_orig)
                free_before = max(0.0, free_before)
                lead = min(request.lead_in_s, free_before, needed_lead)

            earliest = seg_orig_start - lead
            seg_start = max(earliest, cursor)
            if seg_start - seg_orig_start <= 1e-3:
                resync_count += 1

            available_natural = max(0.05, natural_end - seg_start)
            speech_room_natural = available_natural - total_pause - inter_guard

            comfort_speed = min(base_speed * comfort_ratio, max_speed)
            pause_scale = 1.0

            # [V8.5] Segment Base Stress: áp lực tốc độ trung bình của cả cụm, dùng
            # để cân bằng lực căng giữa các câu (tránh câu quá nhanh, câu quá chậm).
            if speech_room_natural > 0.05:
                segment_stress = (total_speech * base_speed) / speech_room_natural
            else:
                segment_stress = max_speed
            segment_stress = min(max(segment_stress, stretch_floor), max_speed)

            if total_speech <= speech_room_natural or speech_room_natural <= 0.05:
                seg_speed = base_speed
            else:
                deficit = total_speech - speech_room_natural
                pause_freeable = total_pause * (1.0 - min_pause_ratio)
                if total_pause > 0 and deficit <= pause_freeable:
                    pause_scale = (total_pause - deficit) / total_pause
                    seg_speed = base_speed
                    compressed_segs += 1
                else:
                    pause_scale = min_pause_ratio if total_pause > 0 else 1.0
                    room = available_natural - total_pause * pause_scale - inter_guard
                    room = max(0.05, room)
                    speed_no_drift = (total_speech * base_speed) / room
                    if speed_no_drift <= comfort_speed:
                        seg_speed = max(speed_no_drift, base_speed)
                    else:
                        room_max = room + max_drift
                        speed_with_drift = (total_speech * base_speed) / room_max
                        seg_speed = min(max(comfort_speed, speed_with_drift), max_speed)
                    compressed_segs += 1

            seg_cursor = seg_start
            catchup_used = False
            n_in_seg = len(seg)

            # [V9] NON-CAUSAL LOOKAHEAD: pre-tính tốc độ lý tưởng (bám khung theo
            # chiến lược) cho từng câu, rồi làm mượt HAI CHIỀU (forward + backward)
            # để hệ thống "biết trước" câu dài sắp tới mà tăng tốc nhẹ dần từ sớm,
            # tạo nhịp điệu tự nhiên. Guard phía dưới đảm bảo không tăng chồng tiếng.
            raw_ideal_list: list[float] = []
            for pp, ii in enumerate(seg):
                sb = p1_speech_durs.get(ii, 0.0) * base_speed
                tp = pause_for(ii) * pause_scale
                st_est = seg_start if pp == 0 else events[ii].start_sec
                if pp + 1 < n_in_seg:
                    nx_est = events[seg[pp + 1]].start_sec
                else:
                    nx_est = natural_end + gap_guard_s
                if allow_audio_overlap and sb > 0:
                    if strategy == "balanced":
                        ri = sb / max(0.05, nx_est - st_est - tp + tail_slack)
                    else:
                        fo = sb / max(0.05, events[ii].end_sec - st_est - tp + tail_slack)
                        ri = fo if fo <= max_speed else sb / max(0.05, nx_est - st_est - tp + tail_slack)
                else:
                    ri = seg_speed
                raw_ideal_list.append(min(max(ri, stretch_floor), max_speed))
            smoothed_ideal_list = raw_ideal_list.copy()
            for i in range(1, n_in_seg):
                smoothed_ideal_list[i] = smoothed_ideal_list[i - 1] * 0.4 + smoothed_ideal_list[i] * 0.6
            for i in range(n_in_seg - 2, -1, -1):
                smoothed_ideal_list[i] = smoothed_ideal_list[i + 1] * 0.3 + smoothed_ideal_list[i] * 0.7

            for pos_in_seg, idx in enumerate(seg):
                orig = events[idx].start_sec
                speech_dur = p1_speech_durs.get(idx, 0.0)
                this_pause = pause_for(idx) * pause_scale
                # [V8.5] Khoảng thở sinh lý nội suy theo độ dài câu (4%, kẹp 50–250ms).
                dynamic_micro_pause = min(0.25, max(_MIN_MICRO_PAUSE_S, speech_dur * 0.04))

                if pos_in_seg + 1 < n_in_seg:
                    next_anchor = events[seg[pos_in_seg + 1]].start_sec
                else:
                    next_anchor = natural_end + gap_guard_s

                if allow_audio_overlap:
                    # [v3.23.233] CHỈ CHỚM SỚM KHI CÂU THẬT SỰ CẦN — ĐÃ NGHIỆM THU.
                    #
                    # Đo trên FLAC + CSV **Edge** thật (95 câu, cùng video):
                    #
                    #   v228 (ăn gian vô điều kiện):
                    #       dời median -71ms | dời OAN 36 câu
                    #       nén max 1.62x    | chồng tiếng 0.95s / 8 câu
                    #   v233 (fix này):
                    #       dời median 0ms   | dời OAN 0 câu
                    #       nén max 1.62x    | chồng tiếng 0.73s / 9 câu
                    #
                    # Nén KHÔNG tăng, chồng tiếng còn GIẢM. Onset thật trên FLAC: median
                    # +40ms — đúng bằng chuẩn VieNeu. Ba câu còn vang sớm đều là câu THẬT
                    # SỰ cần khung rộng hơn (audio dài hơn khung), tức hợp lệ.
                    #
                    # LỊCH SỬ (đừng lặp lại): v231 từng hoàn nguyên fix này vì tưởng nó
                    # gây chồng tiếng gấp 7 lần — nhưng số liệu đó lấy nhầm từ phiên
                    # **Gemini** (file config ghi rõ ``Engine: Gemini TTS``). Dữ liệu Edge
                    # thật ở trên đã bác bỏ hoàn toàn giả thuyết đó.
                    #
                    # Mã cũ: hễ có >= 50ms trống phía trước là dời sớm nguyên
                    # ``lipsync_early`` (250ms), BẤT KỂ câu đó chật hay không -> 36/95 câu
                    # bị dời oan, tiếng vang trước khẩu hình. Nay chỉ ăn gian đúng phần
                    # THIẾU của riêng câu đó — cùng nguyên tắc ``needed_lead_s`` của
                    # VieNeu v219.
                    if pos_in_seg == 0:
                        start_k = seg_cursor
                    else:
                        start_k = orig
                        if orig - seg_cursor > 0.05:
                            window_at_orig = max(
                                0.05,
                                next_anchor - orig - this_pause + tail_slack,
                            )
                            needed_early = max(
                                0.0, speech_dur * base_speed - window_at_orig
                            )
                            early = min(lipsync_early, needed_early)
                            if early > 0.0:
                                start_k = max(seg_cursor + 0.05, orig - early)
                    speech_base = speech_dur * base_speed

                    # Khung mở rộng theo dung sai thị giác (tail_slack) → fit thấp hơn.
                    ext_window = max(0.05, next_anchor - start_k - this_pause + tail_slack)
                    fit_ext = speech_base / ext_window

                    if strategy == "balanced":
                        eff_speed = max(fit_ext, stretch_floor)
                        # Blend nhẹ với tốc độ cụm để tránh nhảy tốc độ quá gắt.
                        eff_speed = (eff_speed * 0.7) + (seg_speed * 0.3)
                    else:
                        own_window = max(0.05, events[idx].end_sec - start_k - this_pause + tail_slack)
                        fit_own = speech_base / own_window
                        if fit_own <= max_speed:
                            eff_speed = max(fit_own, stretch_floor)
                        else:
                            eff_speed = max(fit_ext, stretch_floor)

                    drift_k = start_k - orig
                else:
                    if pos_in_seg == 0:
                        start_k = seg_cursor
                    else:
                        start_k = max(orig, seg_cursor)
                        if max_intra_gap_s >= 0:
                            start_k = min(start_k, seg_cursor + max_intra_gap_s)

                    drift_k = start_k - orig
                    eff_speed = seg_speed

                    # CATCH-UP PHI TUYẾN (hàm mũ 1.5): tăng tốc mượt theo độ trễ,
                    # không bật max ngay, tránh giật cục tốc độ.
                    if drift_k > 0 and max_drift > 0:
                        urgency_ratio = min(1.0, max(0.0, drift_k / max_drift))
                        smooth_urgency = urgency_ratio ** 1.5
                        catchup_multiplier = 1.0 + (smooth_urgency * (_CATCHUP_SPEED_BOOST - 1.0))
                        eff_speed = eff_speed * catchup_multiplier

                    # Khẩn cấp: trượt vượt ngưỡng thảm hoạ → ép max, rút pause còn
                    # mức "thở sinh lý" nội suy theo độ dài câu thay vì triệt tiêu hẳn.
                    if drift_k > _DRIFT_DISASTER_S and speech_dur > 0:
                        eff_speed = max_speed
                        this_pause = dynamic_micro_pause
                        catchup_used = True

                    space_to_next = max(0.05, next_anchor - start_k - this_pause)
                    if speech_dur > 0:
                        fit_speed = (speech_dur * base_speed) / space_to_next
                        if fit_speed < eff_speed and fit_speed >= stretch_floor:
                            eff_speed = max(fit_speed, stretch_floor)

                # [V8.5] Cân bằng lực căng: trộn nhẹ với áp lực tốc độ của cả cụm để
                # không có câu quá nhanh xen câu quá chậm trong cùng cụm hội thoại.
                if n_in_seg > 1:
                    eff_speed = (eff_speed * 0.8) + (segment_stress * 0.2)

                # Chặn trần & sàn: đây là tốc độ "lý tưởng" (bám khung) đã clamp.
                ideal_speed = min(max(eff_speed, stretch_floor), max_speed)

                # [V8.5+] Elastic Peak Soft-Clipping THÔNG MINH: nén mềm đỉnh tốc độ
                # bằng tanh cho êm tai, NHƯNG chỉ trong phần "dư" — không bao giờ nén
                # xuống dưới mức tối thiểu cần để câu vừa khung. Nhờ vậy câu đang phải
                # chạy đua với khung vẫn giữ đồng bộ (không bị tanh làm tràn → khỏi
                # rơi vào vòng nén-đỉnh-rồi-cắt-chữ).
                if ideal_speed > comfort_speed:
                    headroom = max_speed - comfort_speed
                    if headroom > 0.01:
                        ideal_before_clip = ideal_speed
                        excess = ideal_speed - comfort_speed
                        ideal_speed = comfort_speed + headroom * math.tanh(excess / headroom)
                        ideal_speed = min(max(ideal_speed, stretch_floor), max_speed)
                        if speech_dur > 0:
                            fit_needed = (speech_dur * base_speed) / max(
                                0.05, next_anchor - start_k - this_pause + tail_slack
                            )
                            if fit_needed <= max_speed:
                                ideal_speed = max(ideal_speed, min(fit_needed, ideal_before_clip))

                # [V9] Kết hợp Non-Causal + EMA: trước hết trộn tốc độ lý tưởng (chính
                # xác theo start_k động) với mục tiêu đã làm mượt 2 chiều của cụm —
                # đây là phần "nhìn trước" giúp tăng tốc nhẹ trước câu dài. Sau đó EMA
                # với tốc độ thực của câu liền trước để mượt chuỗi phát thực tế.
                ideal_target = 0.5 * ideal_speed + 0.5 * smoothed_ideal_list[pos_in_seg]
                smoothed_speed = (
                    prev_eff_speed * (1.0 - _SPEED_EMA_ALPHA) + ideal_target * _SPEED_EMA_ALPHA
                )
                # Guard: làm mượt KHÔNG bao giờ khiến chồng tiếng tệ hơn tốc độ lý
                # tưởng (xét theo khung đã nới dung sai thị giác); chỉ êm hơn khi dư chỗ.
                speech_base_dur = speech_dur * base_speed
                if speech_base_dur > 0:
                    end_ideal = start_k + this_pause + speech_base_dur / ideal_speed
                    if end_ideal <= next_anchor + tail_slack:
                        floor_no_worse = speech_base_dur / max(
                            0.05, next_anchor - start_k - this_pause + tail_slack
                        )
                    else:
                        floor_no_worse = ideal_speed
                    smoothed_speed = max(smoothed_speed, floor_no_worse)
                eff_speed = min(max(smoothed_speed, stretch_floor), max_speed)

                # [V11.5] Dynamic Clamp: giới hạn nhảy tốc độ giữa 2 câu liền nhau cho
                # đỡ giật; nhưng nếu việc kìm lại khiến câu tràn quá khung (kể cả dung
                # sai thị giác) thì nâng lên "tốc độ sống còn" để vẫn kịp khung.
                if pos_in_seg > 0:
                    allowable_jump = _MAX_SPEED_JUMP if drift_k <= _DRIFT_DISASTER_S else 0.8
                    clamped = min(
                        max(eff_speed, prev_eff_speed - allowable_jump),
                        prev_eff_speed + allowable_jump,
                    )
                    clamped = min(max(clamped, stretch_floor), max_speed)
                    if speech_base_dur > 0:
                        end_if_clamped = start_k + this_pause + speech_base_dur / clamped
                        max_allowed_end = next_anchor + tail_slack
                        if end_if_clamped > max_allowed_end:
                            survival = speech_base_dur / max(
                                0.05, max_allowed_end - start_k - this_pause
                            )
                            clamped = max(clamped, min(survival, max_speed))
                    eff_speed = clamped
                prev_eff_speed = eff_speed
                max_drift_seen = max(max_drift_seen, abs(drift_k))

                dur_k = this_pause + (speech_dur * base_speed) / eff_speed
                end_k = start_k + dur_k

                if end_k > next_anchor + 0.01:
                    overlap_total_s += end_k - next_anchor
                    overlap_count += 1

                schedule[idx] = (start_k, end_k, eff_speed)
                peak_speed = max(peak_speed, eff_speed)
                seg_cursor = end_k + gap_guard_s

            if catchup_used:
                catchup_segs += 1
            cursor = seg_cursor

        if allow_audio_overlap:
            logger.info(
                "Auto-Dubbing scheduling (neo mốc gốc/lipsync, chiến lược=%s): %d câu / %d cụm, "
                "tốc độ đỉnh %.2f×, %d câu chồng tiếng (tổng %.1fs), re-sync %d. "
                "Mỗi câu neo đúng mốc, tốc độ trong [%.2f×, %.2f×].",
                strategy, n, n_segs, peak_speed, overlap_count, overlap_total_s,
                resync_count, stretch_floor, max_speed,
            )
        else:
            logger.info(
                "Auto-Dubbing scheduling (dồn mốc, Catchup phi tuyến): %d câu / %d cụm, %d cụm nén, "
                "%d cụm catch-up, tốc độ đỉnh %.2f×, drift tối đa %.2fs (dung sai %.1fs), "
                "re-sync %d. Phân bổ nén thông minh trong [%.2f×, %.2f×].",
                n, n_segs, compressed_segs, catchup_segs, peak_speed, max_drift_seen,
                max_drift, resync_count, stretch_floor, max_speed,
            )
        return schedule

    def is_available(self) -> bool:
        # [v3.23.328] Ưu tiên chạy edge-tts trong SUBPROCESS để lỗi/treo của nó không
        # ảnh hưởng ứng dụng chính. Nếu bản đóng gói không có Python ngoài thì lùi về
        # chạy trong tiến trình (xem ``_run_edge_in_process``) — được phép vì dự án đã
        # là mã nguồn mở và spec đã gom edge_tts vào bundle.
        # Dùng find_spec: chỉ kiểm module CÓ TỒN TẠI, không nạp code lúc kiểm.
        import importlib.util

        try:
            import scipy  # noqa: F401
            import soundfile  # noqa: F401
        except ImportError:
            return False
        return importlib.util.find_spec("edge_tts") is not None

    def get_engine_name(self) -> str:
        # [SỬA] Giữ nguyên tên cũ để VM/registry khớp đúng engine đã lưu.
        return "Edge TTS (Online)"

    def list_languages(self) -> list[str]:
        return list(_EDGE_VOICE_MAP.keys())

    def list_speakers(self, language: str) -> list[str]:
        return _EDGE_VOICE_MAP.get(language, [])

    # ── Public generate ───────────────────────────────────────────────────────

    def generate(
        self, request: TTSRequest, output_path: Path,
        progress_cb: TTSProgressCallback | None = None,
        cancel_cb: TTSCancellationCallback | None = None,
    ) -> list[TTSSegmentResult]:
        if not self.is_available():
            raise TTSUnavailableError(
                "edge-tts chưa cài đủ. Chạy: pip install edge-tts soundfile scipy numpy librosa"
            )

        voice = request.speaker
        if not voice or voice == "default":
            voices = _EDGE_VOICE_MAP.get(request.language, [])
            voice = voices[0] if voices else "vi-VN-HoaiMyNeural"

        if progress_cb:
            progress_cb(0.0, "Đang kết nối Edge TTS…")

        valid = [e for e in request.events if e.text.strip()]
        if not valid:
            return []

        if progress_cb:
            progress_cb(0.01, "Kiểm tra kết nối và sample rate…")
        probe = self._sync_generate_with_retry("a", voice, request.base_speed, request)
        if probe is None:
            # [v3.23.342] Thông điệp cũ chỉ nói "Kiểm tra mạng" — SAI LỆCH. Log thực tế
            # cho thấy nguyên nhân là Python ngoài thiếu thư viện edge-tts (worker thoát
            # mã 3), mạng hoàn toàn bình thường. Nay nêu CẢ HAI khả năng và cách phân biệt.
            raise TTSGenerationError(
                f"Edge TTS không tạo được âm thanh sau {request.retry_count} lần thử.\n\n"
                "Hai nguyên nhân thường gặp:\n"
                "• Thiếu thư viện edge-tts ở môi trường chạy phụ — mở trang Nhật ký, "
                "nếu thấy “EDGE_TTS_MISSING” thì đúng là lỗi này (KHÔNG phải lỗi mạng).\n"
                "• Không ra được Internet: dịch vụ cần truy cập "
                "speech.platform.bing.com.\n\n"
                "Có thể dùng VieNeu-TTS (chạy hoàn toàn offline) để không phụ thuộc mạng."
            )
        sr = probe[1]

        last_end = max(e.end_sec for e in valid)
        # [v3.23.164] Đuôi master phải đủ chỗ cho câu CUỐI sau khi cộng dialog_pause +
        # phần nới cuối (last_line_max_extend) + dung sai lipsync, nếu không câu cuối bị
        # slice ``min(es, len(master))`` CẮT CỤT ÂM THẦM (mất chữ cuối phim) mà không
        # đánh dấu was_truncated. Tính bằng hàm thuần để test được biên này.
        master_len = _master_track_length_samples(
            last_end_s=last_end,
            sample_rate=sr,
            dialog_pause_s=request.dialog_pause_ms / 1000.0,
            last_line_extend_s=request.last_line_max_extend_s,
            drift_s=request.max_drift_s if request.elastic_timing else 0.0,
            extra_tail_s=_LAST_EXTEND_S if request.elastic_timing else 0.0,
            media_duration_s=getattr(request, "media_duration_s", None),
        )
        master = np.zeros(master_len, dtype=np.float32)

        mode = "Concurrent Double Pass" if request.double_pass else "Concurrent Single Pass"
        n_valid = len(valid)
        logger.info(
            "Edge TTS [%s]: voice=%s events=%d concurrency=%d",
            mode, voice, n_valid, getattr(request, "edge_concurrency", _EDGE_CONCURRENCY),
        )

        self._ts_heavy_count = 0
        self._has_librosa = self._check_librosa()

        results, stats = self._run_concurrent(
            voice=voice, sr=sr, request=request,
            master=master, cancel_cb=cancel_cb, progress_cb=progress_cb,
        )

        # [V8.6] Báo cáo thống kê chuyên sâu cuối quá trình.
        _max_idx = stats["max_overlap_idx"]
        _max_text = request.events[_max_idx].text if _max_idx >= 0 else ""
        if len(_max_text) > 40:
            _max_text = _max_text[:37] + "..."
        logger.info(
            "===== BÁO CÁO AUTO-DUBBING =====\n"
            "- %d câu xử lý | Tốc độ đỉnh (dự kiến): %.2fx\n"
            "- Chồng tiếng: %d câu (tổng %.2fs) | Lần dài nhất: %.2fs ('%s')\n"
            "- Cứu hộ vượt rào %dms: Ép xung %d câu | Cắt mượt %d câu",
            len(results), request.max_speed,
            stats["overlap_count"], stats["overlap_total_s"],
            stats["max_overlap_s"], _max_text, request.max_overlap_ms,
            stats["emergency_stretch_count"], stats["smart_cut_count"],
        )

        if self._ts_heavy_count > 0:
            quality_hint = "" if self._has_librosa else (
                " Cài 'pip install librosa' để cải thiện chất lượng time-stretch."
            )
            logger.warning(
                "Edge TTS: %d/%d dòng cần time-stretch ≥%.1fx (chất lượng có thể giảm). "
                "Cân nhắc tăng max_speed hoặc chỉnh timing phụ đề.%s",
                self._ts_heavy_count, n_valid, _TS_QUALITY_WARN, quality_hint,
            )

        if request.normalize:
            # [V12.0] DC blocker: loại lệch DC (nếu có) trước khi chuẩn loudness/limit.
            # DC làm hao headroom đỉnh và khiến vài thiết bị kêu lụp bụp ở đầu/cuối.
            # [v3.23.213] Dùng high-pass 20Hz (module chung, fix v184) thay TRỪ MEAN
            # toàn cục: trừ mean chỉ loại DC KHÔNG ĐỔI, còn TTS neural sinh DC DRIFT
            # CHẬM trên audio dài -> đo thực: Edge để sót DC gấp ~1449x, ăn 23%%
            # headroom -> limiter kích hoạt oan (nén tiếng không cần thiết).
            # VieNeu/Gemini đã dùng dc_block từ v184; đây là đồng bộ parity.
            from subtitles_extractor.infrastructure.tts.audio_mastering import dc_block

            master = dc_block(master, sr)
            if request.target_lufs < 0.0:
                measured = self._measure_lufs(master, sr)
                if measured != float("-inf"):
                    # [V11.9] Đẩy âm lượng tới mục tiêu LUFS bằng gain ĐẦY ĐỦ, KHÔNG
                    # kéo tụt theo đỉnh. Trước đây chuẩn hoá theo đỉnh khiến tín hiệu
                    # có crest factor cao (≈20dB) bị giảm gain → độ to thực thấp hơn
                    # mục tiêu. Đỉnh nhọn (transient rất ngắn) để limiter phía dưới
                    # ghìm — gần như không ảnh hưởng độ to cảm nhận (LUFS).
                    # [v3.23.178] Gain ĐẦY ĐỦ nhưng CÓ TRẦN (+15dB) qua hàm thuần: đẩy
                    # tới target LUFS mà không khuếch đại quá tay làm nổi nhiễu nền ở
                    # master loudness rất thấp (phim thoại thưa/nhỏ). Đỉnh nhọn để
                    # limiter phía dưới ghìm cục bộ.
                    gain = _loudness_gain_linear(measured, request.target_lufs)
                    master *= np.float32(gain)
                    _actual_gain_db = 20.0 * float(np.log10(gain)) if gain > 0 else 0.0
                    logger.info(
                        "Master loudness: %.1f → mục tiêu %.1f LUFS "
                        "(gain %+.1f dB, chuẩn EBU R128 có trần)",
                        measured, request.target_lufs, _actual_gain_db,
                    )
            # [V12.0] Soft-limit per-mẫu (tanh, trần 0.92 ≈ -0.7 dBFS): chừa đủ
            # headroom cho true-peak liên-mẫu (đo thực tế ≤ -0.5 dBTP ở -12 LUFS) để
            # khi ENCODE LOSSY (AAC/MP3 lúc lồng vào video) KHÔNG clip/méo — yếu tố
            # quyết định chất lượng cuối cho thuyết minh lồng phim. Nhẹ RAM nhất.
            master = self._soft_limit_master(master, threshold=0.85, ceiling=0.92)
            np.clip(master, -1.0, 1.0, out=master)
            # [V13.0] TRUE-PEAK LIMITER CỤC BỘ: limiter per-mẫu không khống chế đỉnh
            # liên-mẫu (inter-sample). Trước đây ta đo true-peak rồi HẠ ĐỀU cả master
            # — an toàn nhưng mất ~2-3 dB độ to ở TOÀN bộ chỉ vì vài đỉnh nhọn. Nay
            # chỉ ghìm CỤC BỘ đúng các đỉnh vượt trần (chiếm rất ít thời lượng), giữ
            # nguyên độ to phần còn lại → đạt gần đúng LUFS mục tiêu mà vẫn an toàn
            # encode lossy.
            tp_before = self._measure_true_peak(master, sr)
            master = self._true_peak_limit(master, sr, ceiling=0.95)
            if tp_before > 0.95:
                tp_after = self._measure_true_peak(master, sr)
                logger.info(
                    "True-peak limiter cục bộ: đỉnh %.2f → %.2f dBTP "
                    "(ghìm riêng đỉnh, giữ nguyên độ to phần còn lại)",
                    20.0 * np.log10(tp_before + 1e-12),
                    20.0 * np.log10(tp_after + 1e-12),
                )
        else:
            np.clip(master, -1.0, 1.0, out=master)

        fmt = getattr(request, "output_format", "wav").lower()
        if progress_cb:
            progress_cb(0.97, f"Đang ghi file {fmt.upper()}…")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        final_path = self._write_audio(master, sr, output_path, request)
        try:
            _peak = float(np.max(np.abs(master))) if len(master) else 0.0
            _sz = final_path.stat().st_size / 1e6
            _lufs_txt = (
                f", {self._measure_lufs(master, sr):.1f} LUFS"
                if request.normalize and request.target_lufs < 0 else ""
            )
            logger.info(
                "Chất lượng output: %s · %.1f MB · đỉnh %.3f%s",
                final_path.name, _sz, _peak, _lufs_txt,
            )
        except (OSError, ValueError):
            pass
        if progress_cb:
            progress_cb(1.0, "Hoàn tất!")
        # Giải phóng mảng master lớn (có thể tới hàng trăm MB với video dài) và thu
        # gom rác ngay, trả bộ nhớ về hệ điều hành trước khi trả kết quả. Nếu không,
        # bộ nhớ đỉnh của lần chạy này còn bị giữ và cộng dồn khi người dùng bấm chạy
        # TTS lần kế tiếp — nguyên nhân gây treo/đơ trên máy RAM hạn chế.
        del master
        import gc as _gc
        _gc.collect()
        return results

    def _write_audio(
        self, master: np.ndarray, sr: int, output_path: Path, request: TTSRequest
    ) -> Path:
        """Ghi master (float32) ra đúng định dạng người dùng chọn.

        - wav/flac/ogg: ghi trực tiếp bằng soundfile (libsndfile) — lossless với
          wav/flac, không qua khâu trung gian.
        - mp3/opus/m4a: encode TRỰC TIẾP từ float32 bằng ffmpeg (pipe f32le) để giữ
          chất lượng tối đa, tránh chuyển đổi ngoài làm nhiễu khi debug.
        Trả về đường dẫn file thực đã ghi (đuôi có thể đổi theo định dạng).
        """
        import soundfile as sf

        fmt = getattr(request, "output_format", "wav").lower()
        subtype = getattr(request, "wav_subtype", "PCM_16")
        bitrate = max(32, int(getattr(request, "output_bitrate_kbps", 320)))
        target = output_path.with_suffix(f".{fmt}")

        if fmt in ("wav", "flac"):
            # FLAC KHÔNG hỗ trợ mẫu Float 32-bit (chỉ PCM nguyên). Nếu người dùng
            # chọn FLAC + FLOAT, tự hạ về PCM 24-bit để tránh lỗi soundfile
            # "Invalid combination of format, endianness and bits per sample".
            if fmt == "flac" and subtype == "FLOAT":
                logger.info("FLAC không hỗ trợ Float 32-bit → dùng PCM_24 thay thế.")
                subtype = "PCM_24"
            sf.write(str(target), master, sr, subtype=subtype)
            return target

        # Các định dạng nén có mất (kể cả ogg) → ffmpeg, encode trực tiếp từ float32
        # để áp đúng bitrate và giữ chất lượng tối đa (không qua khâu trung gian).
        if self._encode_with_ffmpeg(master, sr, target, fmt, bitrate):
            return target

        # Lùi an toàn: nếu ffmpeg không có, ghi WAV để không mất kết quả.
        logger.warning(
            "Không encode được %s (thiếu ffmpeg?). Ghi WAV thay thế để giữ kết quả.", fmt
        )
        wav_path = output_path.with_suffix(".wav")
        sf.write(str(wav_path), master, sr, subtype=subtype)
        return wav_path

    @staticmethod
    def _encode_with_ffmpeg(
        master: np.ndarray, sr: int, target: Path, fmt: str, bitrate_kbps: int
    ) -> bool:
        """Pipe float32 PCM vào ffmpeg để encode lossy chất lượng cao. True nếu OK."""
        import shutil
        import subprocess

        from subtitles_extractor.infrastructure.media import find_ffmpeg
        ffmpeg = find_ffmpeg()
        if not ffmpeg:
            return False

        codec_map = {
            "mp3": ("libmp3lame", []),
            "opus": ("libopus", []),
            "ogg": ("libvorbis", []),
            "m4a": ("aac", ["-movflags", "+faststart"]),
            "aac": ("aac", ["-movflags", "+faststart"]),
        }
        codec, extra = codec_map.get(fmt, ("libmp3lame", []))
        data = np.ascontiguousarray(np.clip(master, -1.0, 1.0), dtype=np.float32)
        cmd = [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0",
            "-c:a", codec, "-b:a", f"{bitrate_kbps}k", *extra, str(target),
        ]
        try:
            proc = subprocess.run(
                cmd, input=data.tobytes(),
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=False,
                **no_window_kwargs(),
            )
            if proc.returncode == 0 and target.exists() and target.stat().st_size > 0:
                return True
            logger.warning("ffmpeg encode %s lỗi: %s", fmt, proc.stderr.decode("utf-8", "ignore")[:200])
            return False
        except (OSError, ValueError) as exc:
            logger.warning("Gọi ffmpeg thất bại: %s", exc)
            return False

    # ── Concurrent orchestration ──────────────────────────────────────────────

    def _run_concurrent(
        self, *, voice, sr, request, master, cancel_cb, progress_cb
    ) -> tuple[list[TTSSegmentResult], dict]:
        coro = self._async_all(
            voice=voice, sr=sr, request=request,
            master=master, cancel_cb=cancel_cb, progress_cb=progress_cb,
        )
        try:
            return asyncio.run(coro)
        except RuntimeError as exc:
            if "event loop" in str(exc).lower():
                loop = asyncio.new_event_loop()
                try:
                    return loop.run_until_complete(
                        self._async_all(
                            voice=voice, sr=sr, request=request,
                            master=master, cancel_cb=cancel_cb, progress_cb=progress_cb,
                        )
                    )
                finally:
                    loop.close()
            raise

    async def _async_all(
        self, *, voice, sr, request, master, cancel_cb, progress_cb
    ) -> tuple[list[TTSSegmentResult], dict]:
        events = request.events
        n = len(events)
        concurrency = max(1, min(getattr(request, "edge_concurrency", _EDGE_CONCURRENCY), 64))
        # [v3.23.229] Giới hạn song song TỰ ĐIỀU CHỈNH thay cho Semaphore cố định.
        #
        # Đo trên log thật (concurrency=64): 20 lỗi "audio rỗng" DỒN trong 2.4 giây ngay
        # sau khi phóng request, rồi 9 timeout 30s cùng lúc -> ~80/87 giây của cả phiên
        # TTS là retry và chờ. Toàn bộ lỗi dồn vào burst đầu (không rải rác) = dấu hiệu
        # dịch vụ TỪ CHỐI vì quá tải kết nối, chứ không phải lỗi lẻ của từng dòng.
        #
        # Mã cũ retry ở CÙNG mức tải -> lại bị chặn (log: lần 2, 3, 4 vẫn rỗng). Edge TTS
        # là dịch vụ miễn phí của Microsoft, giới hạn kết nối KHÔNG công bố và có thể đổi
        # -> hardcode một con số "an toàn" là đoán mò. Cách đúng: tự dò và lùi (AIMD).
        semaphore = AdaptiveConcurrencyLimiter(concurrency)

        skipped: dict[int, TTSSegmentResult] = {}
        to_process: list[int] = []
        for idx, event in enumerate(events):
            # text_srt: cho FILE PHỤ ĐỀ → GIỮ tag người nói (người xem biết ai nói).
            text_srt, _ = _preprocess_tts_text(event.text, request.clean_tags)
            # text_tts: cho AUDIO → BỎ tag người nói (không đọc tên nhân vật).
            text_tts, _ = _preprocess_tts_text(
                event.text, request.clean_tags, _skip_from_request(request),
                strip_speaker_tag=True,
            )
            if not text_srt:
                skipped[idx] = TTSSegmentResult(
                    event_index=idx, start_sec=event.start_sec, end_sec=event.end_sec,
                    text="", was_skipped=True, error_msg="Văn bản rỗng.",
                )
            elif not text_tts:
                skipped[idx] = TTSSegmentResult(
                    event_index=idx, start_sec=event.start_sec, end_sec=event.end_sec,
                    text=text_srt, was_skipped=True, error_msg="Mô tả âm thanh (không đọc).",
                    adjusted_start_sec=event.start_sec, adjusted_end_sec=event.end_sec,
                )
            elif not _has_speakable_content(text_tts):
                # [v3.23.38] Văn bản chỉ gồm dấu câu/ký hiệu (♪, "(nói tiếng X)" sau khi
                # bỏ ngoặc còn rỗng, "(vật rơi)"…) → EdgeTTS sẽ trả NoAudioReceived.
                # Bỏ qua NGAY, KHÔNG retry 3 lần rồi mới báo "Thất bại sau retry".
                skipped[idx] = TTSSegmentResult(
                    event_index=idx, start_sec=event.start_sec, end_sec=event.end_sec,
                    text=text_srt, was_skipped=True,
                    error_msg="Không có nội dung đọc được (chỉ ký hiệu/dấu câu).",
                    adjusted_start_sec=event.start_sec, adjusted_end_sec=event.end_sec,
                )
            else:
                to_process.append(idx)

        n_valid = len(to_process)
        p1_done = 0

        extended_windows = self._compute_extended_windows(
            events, last_extend_s=request.last_line_max_extend_s
        )

        async def _p1(idx: int) -> tuple[int, tuple[np.ndarray, int] | None]:
            nonlocal p1_done
            event = events[idx]
            text, _ = _preprocess_tts_text(
                event.text, request.clean_tags, _skip_from_request(request),
                strip_speaker_tag=True,
            )
            async with semaphore:
                if cancel_cb and cancel_cb():
                    return idx, None
                result = await self._async_gen_with_retry(text, voice, request.base_speed, request)
                if result[0] is not None and len(result[0]) > 0:
                    semaphore.report_success()
                    trimmed = await asyncio.to_thread(self._trim_silence, result[0], result[1])
                    result = (trimmed, result[1])
                else:
                    # Audio rỗng: nhiều khả năng dịch vụ đang chặn vì quá tải -> lùi tải
                    # xuống thay vì tiếp tục đè (mã cũ retry ở cùng mức và lại bị chặn).
                    semaphore.report_failure()

            p1_done += 1
            if progress_cb:
                lbl = "Pass 1" if request.double_pass else "⚡ TTS"
                ratio = 0.03 + (0.42 if request.double_pass else 0.92) * (p1_done / max(1, n_valid))
                progress_cb(ratio, f"{lbl}: {p1_done}/{n_valid} dòng  (×{concurrency} song song)")
            return idx, result

        p1_raw = await asyncio.gather(*[_p1(i) for i in to_process], return_exceptions=True)
        p1_results: dict[int, tuple[np.ndarray, int] | None] = {}
        for _item in p1_raw:
            if isinstance(_item, Exception):
                logger.error("Pass 1 task exception (bỏ qua): %s", _item)
                continue
            _idx, _res = _item
            p1_results[_idx] = _res

        if not request.double_pass:
            schedule = self._build_schedule(events, p1_results, sr, request)
            results_map, stats = self._build_results_from_map(
                p1_results, {}, events, sr, request, master, extended_windows, schedule
            )
            return self._merge_results(n, skipped, results_map, events, request.clean_tags), stats

        schedule = self._build_schedule(events, p1_results, sr, request)

        p2_needed: list[tuple[int, str, float, float]] = []
        for idx in to_process:
            if cancel_cb and cancel_cb():
                raise TTSCancelledError("Người dùng đã huỷ TTS.")
            p1 = p1_results.get(idx)
            if p1 is None or len(p1[0]) == 0:
                continue
            event = events[idx]
            text, is_dialog = _preprocess_tts_text(
                event.text, request.clean_tags, _skip_from_request(request),
                strip_speaker_tag=True,
            )

            if schedule is not None:
                target_speed = schedule[idx][2]
                api_speed = max(_EDGE_API_SPEED_MIN, min(target_speed, _EDGE_API_SPEED_MAX))
                # [v3.23.233] ĐỪNG HẠ NGƯỠNG 0.04 NÀY — đã đo và bác bỏ.
                #
                # Nhìn CSV thấy 9/95 câu phải nén DSP thêm 1.02-1.14x (giọng kéo giãn số,
                # kém tự nhiên hơn gọi lại API), rất dễ kết luận "ngưỡng quá cao, hạ xuống
                # để Pass 2 lo nốt". SAI. Đo trên chính các câu ĐÃ chạy Pass 2:
                #
                #   #19  rate 1.556x: dự kiến 1.132s, THỰC TẾ 1.471s  (+30%)
                #   #20  rate 1.412x: dự kiến 0.642s, THỰC TẾ 0.834s  (+30%)
                #   #21  rate 1.329x: dự kiến 0.821s, THỰC TẾ 1.067s  (+30%)
                #
                # Edge API **không** trả độ dài theo công thức tuyến tính ``dur / rate``
                # — sai số ổn định khoảng **+30%**. Với câu chỉ chênh 2-14%, sai số đó
                # lớn hơn cả phần cần chỉnh: gọi Pass 2 tốn thêm một request + một vòng
                # chờ mà VẪN phải nén DSP, thậm chí lệch xa hơn. Ngưỡng 0.04 chính là
                # ranh giới "chênh đủ lớn để rate đáng tin hơn sai số của chính nó".
                #
                # Ngoài ra 1.02-1.14x là mức nén RẤT NHẸ (VieNeu/Gemini nén tới 2.0x).
                if abs(api_speed - request.base_speed) > 0.04:
                    # Khung SPEECH mục tiêu = thời lượng lịch trừ khoảng lặng hội thoại
                    # (lịch v148 đã dành chỗ pause nhất quán) — dùng cho Pass 2.5.
                    pause_dur = (request.dialog_pause_ms / 1000.0) if is_dialog else 0.0
                    slot_dur = schedule[idx][1] - schedule[idx][0]
                    # [v3.23.163] Câu bị kẹp TRẦN rate (2.55x trong log nhưng khung vẫn
                    # đòi nhanh hơn) sẽ không hạ được residual bằng rate -> rơi vào
                    # time-stretch. Cho các câu đó MƯỢN thời gian tới sát câu kế
                    # (extended_windows đã trừ gap_guard, không đụng câu sau) làm khung
                    # mục tiêu -> nhiều câu thoát ngưỡng stretch mà KHÔNG cần giãn DSP.
                    effective_slot = slot_dur
                    if api_speed >= _EDGE_API_SPEED_MAX - 0.01:
                        effective_slot = max(slot_dur, extended_windows.get(idx, slot_dur))
                    p2_needed.append(
                        (idx, text, api_speed, max(0.05, effective_slot - pause_dur))
                    )
            else:
                available_ext = extended_windows[idx]
                pause_dur = (request.dialog_pause_ms / 1000.0) if is_dialog else 0.0
                speech_dur = len(p1[0]) / sr
                target_speech = max(0.05, available_ext - pause_dur)
                if speech_dur > target_speech * 1.05:
                    exact_speed = (speech_dur * request.base_speed) / target_speech
                    api_speed = min(exact_speed, _EDGE_API_SPEED_MAX)
                    if api_speed > request.base_speed + 0.04:
                        p2_needed.append((idx, text, api_speed, target_speech))

        n_p2 = len(p2_needed)
        p2_done = 0
        if progress_cb and n_p2 > 0:
            progress_cb(0.45, f"⚡ Pass 2: {n_p2}/{n_valid} dòng cần tăng tốc (×{concurrency})…")
        elif progress_cb and n_p2 == 0 and request.double_pass:
            progress_cb(0.45, "✅ Pass 1 hoàn tất — tất cả dòng đã vừa vặn, không cần Pass 2")

        async def _p2(idx: int, text: str, api_speed: float) -> tuple[int, tuple[np.ndarray, int] | None]:
            nonlocal p2_done
            async with semaphore:
                if cancel_cb and cancel_cb():
                    return idx, None
                result = await self._async_gen_with_retry(text, voice, api_speed, request)
                if result[0] is not None and len(result[0]) > 0:
                    trimmed = await asyncio.to_thread(self._trim_silence, result[0], result[1])
                    result = (trimmed, result[1])

            p2_done += 1
            if progress_cb:
                ratio = 0.46 + 0.44 * (p2_done / max(1, n_p2))
                progress_cb(ratio, f"⚡ Pass 2: {p2_done}/{n_p2} dòng…  (×{concurrency})")
            return idx, result

        if n_p2 > 0:
            p2_raw = await asyncio.gather(
                *[_p2(i, t, s) for i, t, s, _tgt in p2_needed], return_exceptions=True
            )
            p2_results: dict[int, tuple[np.ndarray, int] | None] = {}
            for _item in p2_raw:
                if isinstance(_item, Exception):
                    logger.error("Pass 2 task exception (bỏ qua): %s", _item)
                    continue
                _idx, _res = _item
                p2_results[_idx] = _res
        else:
            p2_results = {}

        # ── Pass 2.5: VI CHỈNH rate theo thời lượng ĐO ĐƯỢC (đối xứng nhanh+chậm) ──
        # [v3.23.162] Edge rate<->thời lượng không tuyến tính -> sau Pass 2 câu vẫn lệch
        # khung: DÀI hơn (residual >= 1.15) hoặc NGẮN hơn (residual <= 0.87). Trước đây
        # phần lệch rơi vào time-stretch DSP (cả nén lẫn giãn đều giảm chất lượng). Nay
        # tổng hợp LẠI với rate hiệu chỉnh (còn dư địa trong [0.5, 3.0]) -> Edge tự đọc
        # nhanh/chậm cho khớp khung, hạn chế TRIỆT ĐỂ stretch. Rate đã dùng cho từng câu
        # được theo dõi để lần hiệu chỉnh tính từ ĐÚNG gốc.
        current_rate: dict[int, float] = {
            idx: api_speed for idx, _text, api_speed, _tgt in p2_needed
        }
        target_speech_of: dict[int, float] = {
            idx: tgt for idx, _text, _api, tgt in p2_needed
        }
        text_of: dict[int, str] = {idx: text for idx, text, _api, _tgt in p2_needed}
        recal_needed: list[tuple[int, str, float]] = []
        for idx, text, api_speed, target_speech in p2_needed:
            p2_audio = p2_results.get(idx)
            if p2_audio is None or len(p2_audio[0]) == 0:
                continue
            new_rate = _calibrated_rate(
                api_speed, len(p2_audio[0]) / sr, target_speech,
                _EDGE_API_SPEED_MAX, _EDGE_API_SPEED_MIN,
            )
            if new_rate is not None:
                recal_needed.append((idx, text, new_rate))
        if recal_needed:
            logger.info(
                "Pass 2.5 (vi chỉnh Edge rate): %d câu còn lệch khung sau Pass 2 "
                "-> tổng hợp lại (nhanh/chậm) để né time-stretch triệt để.",
                len(recal_needed),
            )
            p25_done = 0

            async def _p25(
                idx: int, text: str, rate: float
            ) -> tuple[int, float, tuple[np.ndarray, int] | None]:
                nonlocal p25_done
                async with semaphore:
                    if cancel_cb and cancel_cb():
                        return idx, rate, None
                    result = await self._async_gen_with_retry(text, voice, rate, request)
                    if result[0] is not None and len(result[0]) > 0:
                        trimmed = await asyncio.to_thread(
                            self._trim_silence, result[0], result[1]
                        )
                        result = (trimmed, result[1])
                p25_done += 1
                if progress_cb:
                    progress_cb(
                        0.90,
                        f"🎯 Pass 2.5: {p25_done}/{len(recal_needed)} câu vi chỉnh tốc độ…",
                    )
                return idx, rate, result

            p25_raw = await asyncio.gather(
                *[_p25(i, t, r) for i, t, r in recal_needed], return_exceptions=True
            )

            def _residual_gap(audio_len: int, target: float) -> float:
                """Khoảng lệch tuyệt đối |residual - 1| của một bản audio so với khung."""
                if target <= 0.05:
                    return float("inf")
                return abs(audio_len / sr / target - 1.0)

            for _item in p25_raw:
                if isinstance(_item, Exception):
                    logger.error("Pass 2.5 task exception (bỏ qua): %s", _item)
                    continue
                _idx, _rate, _res = _item
                if _res is None or len(_res[0]) == 0:
                    continue
                _old = p2_results.get(_idx)
                _target = target_speech_of.get(_idx, 0.0)
                # [v3.23.162] Chấp nhận bản mới nếu GẦN KHUNG HƠN (đúng cả hai chiều),
                # không chỉ "ngắn hơn" — bản chậm-lại đúng khung cũng phải được nhận.
                if _old is None or _residual_gap(len(_res[0]), _target) < _residual_gap(
                    len(_old[0]), _target
                ):
                    p2_results[_idx] = _res
                    current_rate[_idx] = _rate

            # [v3.23.162] VÒNG 2 (Edge phi tuyến, một vòng đôi khi chưa tới đích): các
            # câu VẪN lệch sau vòng 1 và rate còn dư địa -> hiệu chỉnh thêm một lần.
            round2: list[tuple[int, str, float]] = []
            for _idx, _rate in current_rate.items():
                _audio = p2_results.get(_idx)
                _target = target_speech_of.get(_idx, 0.0)
                if _audio is None or len(_audio[0]) == 0 or _target <= 0.05:
                    continue
                _next = _calibrated_rate(
                    _rate, len(_audio[0]) / sr, _target,
                    _EDGE_API_SPEED_MAX, _EDGE_API_SPEED_MIN,
                )
                if _next is not None:
                    round2.append((_idx, text_of.get(_idx, ""), _next))
            if round2:
                logger.info("Pass 2.5 vòng 2: %d câu tinh chỉnh thêm.", len(round2))
                p25_done = 0
                r2_raw = await asyncio.gather(
                    *[_p25(i, t, r) for i, t, r in round2], return_exceptions=True
                )
                for _item in r2_raw:
                    if isinstance(_item, Exception):
                        logger.error("Pass 2.5 vòng 2 exception (bỏ qua): %s", _item)
                        continue
                    _idx, _rate, _res = _item
                    if _res is None or len(_res[0]) == 0:
                        continue
                    _old = p2_results.get(_idx)
                    _target = target_speech_of.get(_idx, 0.0)
                    if _old is None or _residual_gap(
                        len(_res[0]), _target
                    ) < _residual_gap(len(_old[0]), _target):
                        p2_results[_idx] = _res

        results_map, stats = self._build_results_from_map(
            p1_results, p2_results, events, sr, request, master, extended_windows, schedule
        )
        return self._merge_results(n, skipped, results_map, events, request.clean_tags), stats

    def _build_schedule(
        self,
        events: list,
        p1_results: dict[int, tuple[np.ndarray, int] | None],
        sr: int,
        request: TTSRequest,
    ) -> dict[int, tuple[float, float, float]] | None:
        if not request.elastic_timing:
            return None
        p1_speech_durs: dict[int, float] = {}
        for idx, p1 in p1_results.items():
            if p1 is not None and len(p1[0]) > 0:
                p1_speech_durs[idx] = len(p1[0]) / sr
            else:
                p1_speech_durs[idx] = 0.0
        return self._schedule_timeline(events, p1_speech_durs, request)

    # ── Post-processing ───────────────────────────────────────────────────────

    def _build_results_from_map(
        self,
        p1_map: dict[int, tuple[np.ndarray, int] | None],
        p2_map: dict[int, tuple[np.ndarray, int] | None],
        events: list,
        sr: int,
        request: TTSRequest,
        master: np.ndarray,
        extended_windows: dict[int, float],
        schedule: dict[int, tuple[float, float, float]] | None = None,
    ) -> tuple[dict[int, TTSSegmentResult], dict]:
        results: dict[int, TTSSegmentResult] = {}

        # [V8.6] Bộ thống kê chuyên sâu cho báo cáo cuối quá trình.
        stats: dict = {
            "overlap_count": 0,
            "overlap_total_s": 0.0,
            "max_overlap_s": 0.0,
            "max_overlap_idx": -1,
            "emergency_stretch_count": 0,
            "smart_cut_count": 0,
        }
        # Nếu người dùng tắt "Cho phép chồng tiếng" → giới hạn lấn = 0s.
        allow_overlap = getattr(request, "allow_audio_overlap", True)
        strategy = getattr(request, "timing_strategy", "lipsync")
        user_max_overlap_s = (request.max_overlap_ms / 1000.0) if allow_overlap else 0.0
        # [V11.6] Dung sai thị giác PHẢI khớp với scheduler: câu được phép ngân tới
        # mốc câu kế + lipsync_late mà KHÔNG bị tính là "lấn cần cắt". Trước đây khâu
        # này đo theo extended_windows (thiếu lipsync_late) nên cắt oan đúng các câu
        # mà scheduler cố ý cho đọc chậm bằng dung sai → tốc độ thấp vẫn bị cắt.
        lipsync_late_eff = min(request.max_drift_s, _LIPSYNC_LATE_CAP_S) if (allow_overlap and strategy != "smooth") else 0.0

        # [V11.7] Xử lý theo THỨ TỰ THỜI GIAN tăng dần (không theo thứ tự index) để
        # ducking luôn dìm đúng câu liền TRƯỚC đã nằm trong master — an toàn kể cả khi
        # events bị xáo trộn thứ tự sau chỉnh sửa/gộp hoặc khi câu được dời sớm.
        time_order = sorted(p1_map.keys(), key=lambda i: events[i].start_sec)
        for idx in time_order:
            p1 = p1_map[idx]
            event = events[idx]
            # Đo độ dài theo TEXT đã bỏ tag (giống audio thực) để căn thời gian chính xác.
            # [v3.23.148] Truyền THÊM skip options để is_dialog (quyết định CHÈN pause)
            # tính trên đúng văn bản đã tổng hợp âm thanh — nhất quán với Pass 1/2 và
            # scheduler; "(cười) - Chào" (bật skip ngoặc) nay được chèn khoảng lặng đúng.
            text, is_dialog = _preprocess_tts_text(
                event.text, request.clean_tags, _skip_from_request(request),
                strip_speaker_tag=True,
            )
            # [v3.23.77] Bản GIỮ tag người nói cho FILE PHỤ ĐỀ (không strip_speaker_tag).
            text_srt, _ = _preprocess_tts_text(event.text, request.clean_tags)
            start_sec, end_sec = event.start_sec, event.end_sec
            available_strict = end_sec - start_sec
            pause_dur = (request.dialog_pause_ms / 1000.0) if is_dialog else 0.0
            available_ext = extended_windows.get(idx, available_strict)
            # Bộ theo dõi DEBUG cho từng câu (điền dần qua các bước xử lý).
            _dbg_safe = 0.0
            _dbg_overlap = 0.0
            _dbg_stretch = ""
            _dbg_ratio = 1.0
            _dbg_cut_ms = 0.0
            _dbg_ducked = 0.0

            sched = schedule.get(idx) if schedule is not None else None
            if sched is not None:
                actual_start, sched_end, target_speed = sched
                place_start = actual_start
                target_speech = None
            else:
                actual_start = start_sec
                place_start = start_sec
                available_ext = extended_windows.get(idx, available_strict)
                target_speech = max(0.05, available_ext - pause_dur)
                target_speed = None

            p2 = p2_map.get(idx)
            # Giải phóng tham chiếu segment trong map ngay sau khi đã lấy ra biến cục
            # bộ (speech_audio bên dưới là bản .copy() độc lập). Tránh giữ toàn bộ
            # 2000+ segment cùng lúc đến cuối vòng lặp — giảm mạnh bộ nhớ đỉnh.
            p1_map[idx] = None
            if idx in p2_map:
                p2_map[idx] = None
            if p2 is not None and len(p2[0]) > 0:
                speech_audio = p2[0].copy()
            elif p1 is not None and len(p1[0]) > 0:
                speech_audio = p1[0].copy()
            else:
                results[idx] = TTSSegmentResult(
                    event_index=idx, start_sec=start_sec, end_sec=end_sec,
                    text=text, subtitle_text=text_srt, was_skipped=True,
                    error_msg="Thất bại sau retry.",
                )
                continue

            speech_dur = len(speech_audio) / sr
            p1_speech_dur = (len(p1[0]) / sr) if (p1 and len(p1[0]) > 0) else speech_dur

            used_p2 = p2 is not None and len(p2[0]) > 0
            if used_p2 and p1 is not None and len(p1[0]) > 0:
                speed_api = p1_speech_dur / max(speech_dur, 1e-6)
            else:
                speed_api = request.base_speed

            if sched is not None:
                speed_used = target_speed
            else:
                speed_used = max(speed_api, request.base_speed)

            if sched is not None and not request.high_quality:
                if target_speed > speed_api + 0.02:
                    desired_speech_dur = (p1_speech_dur * request.base_speed) / target_speed
                    # [v3.23.163] TRƯỚC khi time-stretch, cho câu MƯỢN thời gian tới sát
                    # câu kế (extended_windows đã trừ gap_guard nên không đụng câu sau):
                    # nếu audio thật vừa khít trong khung nới rộng thì KHÔNG stretch nữa;
                    # nếu vẫn dài thì chỉ stretch phần vượt KHUNG NỚI RỘNG -> ts_ratio nhỏ
                    # hơn hẳn, nhiều câu thoát ngưỡng cảnh báo 1.5x. Đây là lý do log
                    # còn 7 câu stretch: khung strict quá chật dù có gap trống liền kề.
                    ts_ratio = _stretch_ratio_with_extended(
                        speech_dur, desired_speech_dur, available_ext - pause_dur,
                        request.max_speed,
                    )
                    if ts_ratio > 1.02:
                        stretched = self._time_stretch_vocal(speech_audio, sr, ts_ratio)
                        if len(stretched) > 0:
                            speech_audio = stretched
                            speech_dur = len(speech_audio) / sr
                            _dbg_stretch = "librosa" if ts_ratio > 2.0 else "WSOLA"
                            _dbg_ratio = ts_ratio
                            speed_used = target_speed
                            if ts_ratio >= _TS_QUALITY_WARN:
                                self._ts_heavy_count += 1
            elif sched is None and not request.high_quality:
                if target_speech is not None and speech_dur > target_speech * 1.05:
                    ts_ratio = min(speech_dur / max(0.001, target_speech), request.max_speed)
                    if ts_ratio > 1.02:
                        stretched = self._time_stretch_vocal(speech_audio, sr, ts_ratio)
                        if len(stretched) > 0:
                            speech_audio = stretched
                            speech_dur = len(speech_audio) / sr
                            speed_used = ts_ratio
                            if ts_ratio >= _TS_QUALITY_WARN:
                                self._ts_heavy_count += 1

            if request.voice_clarity:
                # [v3.23.181] Inter-Node Safe Clip: chặn vọt peak ảo (xô lệch pha) giữa
                # các tầng DSP bằng SOFT-CLIP (tanh knee) thay cho hard clip — hard clip
                # cắt phẳng đỉnh sinh méo hài chói tai (EQ clarity +70% có thể đẩy 0.9 ->
                # 1.05). Soft-clip giữ nguyên phần dưới ngưỡng, chỉ nén mềm phần vượt.
                speech_audio = self._apply_noise_gate(speech_audio, sr)
                speech_audio = self._zero_phase_eq(speech_audio, sr)
                speech_audio = _inter_node_soft_clip(speech_audio)
                speech_audio = self._split_band_de_esser(speech_audio, sr)
                speech_audio = self._cinematic_air_exciter(speech_audio, sr)
                speech_audio = _inter_node_soft_clip(speech_audio)

            if request.normalize:
                speech_audio = self._normalize_segment(speech_audio)

            speech_audio = self._apply_equal_power_fades(speech_audio, sr)

            # [v3.23.169] Chỉ chèn nghỉ hội thoại khi khung KHÔNG quá chật — nếu chèn
            # khiến giọng phải nén mạnh hơn (vd câu "- A. / - B." nhồi 0.8s), bỏ nghỉ
            # để bảo toàn chất lượng giọng (dữ liệu thật: STT 567 nén 1.57x vì 100ms
            # nghỉ ăn vào khung 0.81s). safe_window ở đây = khung mở rộng + dung sai.
            safe_window_for_pause = available_ext + lipsync_late_eff
            if _should_insert_dialog_pause(
                is_dialog, request.dialog_pause_ms, speech_dur,
                safe_window_for_pause, pause_dur,
            ):
                pause_samples = int(pause_dur * sr)
                audio_final = np.concatenate(
                    (np.zeros(pause_samples, dtype=np.float32), speech_audio)
                )
            else:
                audio_final = speech_audio
                pause_dur = 0.0  # không chèn -> cập nhật để các phép tính sau nhất quán

            audio_dur = len(audio_final) / sr
            was_truncated = False
            available_ext = extended_windows.get(idx, available_strict)

            # [V11.6] KHUNG NHẤT QUÁN VỚI SCHEDULER:
            # - safe_window: khung "miễn phí" = tới mốc câu kế + dung sai thị giác.
            #   Câu fit trong đây KHÔNG bị coi là lấn (khớp đúng tốc độ scheduler đặt).
            # - allowed_dur: + phần lấn người dùng cố ý cho phép (max_overlap_ms).
            safe_window = available_ext + lipsync_late_eff
            allowed_dur = safe_window + user_max_overlap_s
            _dbg_safe = safe_window

            overlap_s = audio_dur - safe_window
            if overlap_s > 0:
                _dbg_overlap = overlap_s
                if request.skip_overlap_ms > 0 and overlap_s > request.skip_overlap_ms / 1000.0:
                    results[idx] = TTSSegmentResult(
                        event_index=idx, start_sec=start_sec, end_sec=end_sec,
                        text=text, subtitle_text=text_srt, was_skipped=True,
                        error_msg=f"Overlap {overlap_s*1000:.0f}ms > skip {request.skip_overlap_ms}ms",
                    )
                    continue
                if overlap_s > stats["max_overlap_s"]:
                    stats["max_overlap_s"] = overlap_s
                    stats["max_overlap_idx"] = idx
                stats["overlap_count"] += 1
                stats["overlap_total_s"] += overlap_s

                # Chỉ can thiệp khi lấn VƯỢT mức người dùng cho phép.
                if audio_dur > allowed_dur:
                    # [V11.6/V11.8] MAX SQUEEZE: vắt kiệt tốc độ đọc lên tới mức cần
                    # (≤ max_speed) để bảo toàn TRỌN lời TRƯỚC khi nghĩ tới cắt. Áp
                    # dụng cả khi bật "chất lượng cao" vì giữ đủ nội dung quan trọng
                    # hơn (số câu này rất ít sau khi đã dùng dung sai).
                    dur_at_max = pause_dur + (p1_speech_dur * request.base_speed) / request.max_speed
                    target_dur = max(allowed_dur, dur_at_max)
                    # [V11.8] Hạ ngưỡng kích hoạt: trước đây bỏ qua khi câu chỉ hơi
                    # dài (≤1%) khiến nó bị CẮT oan ở tốc độ thấp dù còn dư địa. Với
                    # tỉ lệ nén nhỏ (≤5%) dùng OLA giữ pitch (WSOLA bỏ qua <2%); nén
                    # mạnh hơn mới dùng WSOLA. Nhờ đó câu hơi dài được nén êm thay vì
                    # cụt đuôi → giữ trọn lời, gần như không còn câu bị cắt.
                    if audio_dur > target_dur * 1.003:
                        # [v3.23.173] Tách NGHỈ im lặng ở đầu (nếu có) ra khỏi phần
                        # giọng: chỉ stretch phần GIỌNG, giữ nguyên độ dài nghỉ. Tránh
                        # co nhầm khoảng nghỉ hội thoại + không stretch vô ích đoạn im
                        # lặng (gây artifact ở ranh giới nghỉ->giọng).
                        pause_samples_head = int(pause_dur * sr) if pause_dur > 0 else 0
                        pause_samples_head = min(pause_samples_head, len(audio_final))
                        head_silence = audio_final[:pause_samples_head]
                        speech_part = audio_final[pause_samples_head:]
                        speech_ratio = _speech_only_stretch_ratio(
                            audio_dur, pause_dur, target_dur
                        )
                        if speech_ratio > 1.05:
                            squeezed_speech = self._time_stretch_vocal(
                                speech_part, sr, speech_ratio
                            )
                            _dbg_stretch = "librosa" if speech_ratio > 2.0 else "WSOLA"
                        else:
                            squeezed_speech = self._ola_time_stretch(
                                speech_part, sr, speech_ratio
                            )
                            _dbg_stretch = "OLA"
                        if len(squeezed_speech) > 0:
                            _dbg_ratio = speech_ratio
                            audio_final = (
                                np.concatenate((head_silence, squeezed_speech))
                                if pause_samples_head > 0
                                else squeezed_speech
                            )
                            audio_dur = len(audio_final) / sr
                            stats["emergency_stretch_count"] += 1
                            new_speech = max(0.001, audio_dur - pause_dur)
                            speed_used = min(request.max_speed, (p1_speech_dur * request.base_speed) / new_speech)
                            if speed_used >= _TS_QUALITY_WARN:
                                self._ts_heavy_count += 1

                    # SMART DISCIPLINE CUT: chỉ cắt khi ĐÃ vắt hết cỡ mà VẪN vượt khung
                    # cho phép (chống sụp đổ Domino + tôn trọng ngưỡng lấn người dùng).
                    if audio_dur > allowed_dur + 0.01:
                        _dbg_cut_ms = (audio_dur - allowed_dur) * 1000.0
                        max_samples = int(allowed_dur * sr)
                        if max_samples > 0:
                            audio_final = audio_final[:max_samples]
                            fade_samples = min(int(sr * 0.04), max_samples)
                            if fade_samples > 0:
                                fade_curve = np.cos(
                                    np.linspace(0, np.pi / 2, fade_samples)
                                ).astype(np.float32)
                                audio_final[-fade_samples:] *= fade_curve
                        else:
                            audio_final = np.array([], dtype=np.float32)
                        audio_dur = len(audio_final) / sr
                        was_truncated = True
                        stats["smart_cut_count"] += 1

            ss = int(place_start * sr)
            es = min(ss + len(audio_final), len(master))
            # [v3.23.164] An toàn kép: nếu slice master VẪN cắt bớt audio (đuôi vượt
            # master dù đã cấp phát rộng) thì ghi nhận truncate + cập nhật thời lượng
            # thực để báo cáo/subtitle khớp âm thanh thật, không im lặng mất chữ.
            if es - ss < len(audio_final) and ss < len(master):
                was_truncated = True
                audio_dur = max(0.0, (es - ss) / sr)
            if ss < len(master):
                # Ducking chống chồng tiếng đục: nếu vùng câu này đè lên đuôi câu
                # trước (master đã có tín hiệu), hạ âm lượng câu trước xuống sàn
                # bằng đường cong S (sigmoid) cho mượt như studio — câu trước đầy ở
                # đầu vùng rồi giảm dần xuống sàn, giúp câu MỚI nổi rõ, tránh méo.
                region = master[ss:es]
                # [v3.23.172] Độ dài chồng dùng ngưỡng TƯƠNG ĐỐI theo đỉnh region (hàm
                # thuần) thay cho ngưỡng cứng 0.01: câu trước nhỏ (thì thầm / tắt chuẩn
                # hoá) vẫn được phát hiện đúng để duck, tránh hai giọng chồng cùng độ to.
                ov_end = _overlap_length_samples(region)
                if ov_end > 0:
                    _dbg_ducked = ov_end / sr
                    # [V10] Adaptive Ducking: chồng càng dài (gần "cướp lời") thì hạ
                    # câu trước càng sâu; chồng ngắn (chỉ chạm đuôi) chỉ giảm nhẹ để
                    # không bóp mất tự nhiên. Nội suy sàn từ 0.72 (≈0s) → _DUCK_FLOOR.
                    ov_dur = ov_end / sr
                    adapt_floor = max(
                        _DUCK_FLOOR, 0.72 - (0.72 - _DUCK_FLOOR) * min(1.0, ov_dur / 0.6)
                    )
                    fade_len = min(int(0.06 * sr), ov_end)  # S-curve mượt 60ms
                    duck = np.full(ov_end, adapt_floor, dtype=np.float32)
                    if fade_len > 0:
                        # Sigmoid giảm 1.0 → floor: 0.5*(1+cos(pi*t)) với t: 0→1.
                        t = np.linspace(0.0, 1.0, fade_len)
                        s_curve = 0.5 * (1.0 + np.cos(np.pi * t))  # 1 → 0
                        duck[:fade_len] = adapt_floor + (1.0 - adapt_floor) * s_curve
                    region[:ov_end] *= duck
                master[ss:es] += audio_final[:es - ss]

            adj_start = -1.0
            adj_end = -1.0
            if sched is not None:
                final_dur = len(audio_final) / sr
                adj_start = place_start
                adj_end = place_start + final_dur

            results[idx] = TTSSegmentResult(
                event_index=idx, start_sec=start_sec, end_sec=end_sec,
                text=text, subtitle_text=text_srt, audio_duration_s=len(audio_final) / sr,
                speed_used=speed_used, was_truncated=was_truncated,
                adjusted_start_sec=adj_start, adjusted_end_sec=adj_end,
                pass1_dur_s=p1_speech_dur,
                pass2_dur_s=(len(p2[0]) / sr) if used_p2 else 0.0,
                used_pass2=used_p2,
                scheduled_speed=(target_speed if sched is not None else 0.0),
                api_speed=(speed_api * request.base_speed) if used_p2 else request.base_speed,
                window_strict_s=available_strict,
                window_ext_s=available_ext,
                safe_window_s=_dbg_safe,
                overlap_s=_dbg_overlap,
                stretch_method=_dbg_stretch,
                stretch_ratio=_dbg_ratio,
                cut_amount_ms=_dbg_cut_ms,
                pause_ms=pause_dur * 1000.0,
                is_dialog=is_dialog,
                ducked_prev_s=_dbg_ducked,
            )

        return results, stats

    @staticmethod
    def _merge_results(
        n: int,
        skipped: dict[int, TTSSegmentResult],
        results_map: dict[int, TTSSegmentResult],
        events: list,
        clean_tags: bool = True,
    ) -> list[TTSSegmentResult]:
        out: list[TTSSegmentResult] = []
        for i in range(n):
            if i in skipped:
                out.append(skipped[i])
            elif i in results_map:
                out.append(results_map[i])
            else:
                ev = events[i]
                text, _ = _preprocess_tts_text(ev.text, clean_tags)
                out.append(TTSSegmentResult(
                    event_index=i, start_sec=ev.start_sec, end_sec=ev.end_sec,
                    text=text, was_skipped=True,
                    error_msg="Lỗi không xác định trong quá trình generate.",
                ))
        return out

    # ── AUDIO ENGINEERING DSP CẤP ĐỘ C ────────────────────────────────────────

    @staticmethod
    def _trim_silence(audio: np.ndarray, sr: int, top_db: float = 38.0) -> np.ndarray:
        """[V9] Adaptive Percentile VAD: tự dò sàn nhiễu (noise floor) thực tế của
        Edge TTS bằng bách phân vị thay vì ngưỡng dB cứng — gọt khoảng lặng chặt
        hơn nhưng không lẹm âm gió đầu/cuối ('s', 'th')."""
        if len(audio) == 0:
            return audio
        max_val = np.max(np.abs(audio))
        if max_val < 1e-4:
            return audio

        window_size = int(sr * 0.01)
        if len(audio) < window_size * 2:
            return audio

        rms = np.sqrt(np.convolve(audio ** 2, np.ones(window_size) / window_size, mode='valid'))
        # [v3.23.171] Sàn nhiễu = 5% năng lượng thấp nhất; ngưỡng có CHẶN TRÊN theo
        # dynamic guard để không cắt nhầm phụ âm cuối nhẹ ở câu động học lớn.
        noise_floor = float(np.percentile(rms, 5))
        peak_rms = float(np.max(rms))
        bounds = _voiced_bounds_from_rms(rms, noise_floor, peak_rms)
        if bounds is None:
            return audio
        first_voiced, last_voiced = bounds

        pad = int(sr * 0.04)  # chừa 40ms hai đầu, không lẹm âm gió
        start = max(0, first_voiced - pad)
        end = min(len(audio), last_voiced + window_size + pad)
        return audio[start:end].astype(np.float32)

    @classmethod
    def _time_stretch_vocal(cls, audio: np.ndarray, sr: int, ratio: float) -> np.ndarray:
        """Co giãn giọng nói (pedalboard -> librosa -> WSOLA -> OLA).

        [v3.23.220] Uỷ cho :func:`...tts.time_stretch.vocal_time_stretch` — thuật toán đã
        chuyển sang module thuần dùng chung (hành vi giữ NGUYÊN). Trước đây nó là
        staticmethod của adapter Edge, buộc VieNeu import ngược ``EdgeTTSAdapter`` chỉ để
        nén giọng. Giữ classmethod tại đây cho test/monkeypatch hiện có.

        Args:
            audio: Tín hiệu mono float32.
            sr: Tần số lấy mẫu.
            ratio: >1 = nhanh hơn (ngắn lại); <1 = chậm hơn.

        Returns:
            Tín hiệu đã co giãn, cao độ giữ nguyên.
        """
        from subtitles_extractor.infrastructure.tts.time_stretch import (
            vocal_time_stretch,
        )

        return vocal_time_stretch(audio, sr, ratio)

    @staticmethod
    def _apply_noise_gate(audio: np.ndarray, sr: int, threshold_db: float = -42.0) -> np.ndarray:
        if len(audio) < sr * 0.05:
            return audio
        from scipy.signal import butter, filtfilt
        b, a = butter(2, 20.0 / (sr * 0.5), btype='lowpass')
        env = filtfilt(b, a, np.abs(audio))
        # [v3.23.180] Ngưỡng THÍCH ỨNG theo đỉnh câu (hàm thuần): câu giọng NHỎ hợp lệ
        # (thì thầm/yếu) không bị hạ oan như ngưỡng tuyệt đối cũ; câu to vẫn khử nhiễu
        # nền hiệu quả. threshold_db truyền vào là ngưỡng tuyệt đối cận trên.
        env_peak = float(np.max(env)) if env.size else 0.0
        threshold_linear = _noise_gate_threshold_linear(
            env_peak, absolute_threshold_db=threshold_db
        )
        gain = np.clip(env / max(threshold_linear, 1e-9), 0.0, 1.0)
        return (audio * filtfilt(b, a, gain)).astype(np.float32)

    @staticmethod
    def _zero_phase_eq(audio: np.ndarray, sr: int) -> np.ndarray:
        if len(audio) < 150:
            return audio
        from scipy.signal import butter, sosfiltfilt
        nyquist = sr * 0.5
        # 1) Cắt rumble/ầm tần cực thấp (90Hz).
        audio = sosfiltfilt(butter(4, min(0.99, 90.0 / nyquist), btype="highpass", output="sos"), audio)
        # 2) Hạ vùng "ồm/đục" (mud) 130-380Hz → giọng gọn, bớt nặng tần thấp. Giọng
        #    nam (vd NamMinh) dồn nhiều năng lượng ở dải này nên cần hạ tay hơn.
        mud = sosfiltfilt(butter(2, [130.0 / nyquist, 380.0 / nyquist], btype="bandpass", output="sos"), audio)
        audio = audio - mud * 0.60
        # 3) Nâng dải ĐỘ RÕ LỜI 2-4kHz (vùng tai người nhạy nhất) để tiếng nói "cắt"
        #    rõ, dễ nghe hơn khi mix lên nền phim. Tăng tay vì giọng nam thiếu presence.
        clarity = sosfiltfilt(butter(2, [2000.0 / nyquist, 4000.0 / nyquist], btype="bandpass", output="sos"), audio)
        audio = audio + clarity * 0.70
        # 4) Thêm "air" nhẹ >7kHz cho thoáng (vừa phải, tránh chói/xì vì exciter đã thêm).
        air = sosfiltfilt(butter(2, min(0.99, 7000.0 / nyquist), btype="highpass", output="sos"), audio)
        return (audio + air * 0.18).astype(np.float32)

    @staticmethod
    def _split_band_de_esser(audio: np.ndarray, sr: int) -> np.ndarray:
        if len(audio) < 150:
            return audio
        from scipy.signal import butter, sosfiltfilt, filtfilt
        nyq = sr * 0.5
        sos_split = butter(4, min(0.99, 5500.0 / nyq), btype='highpass', output='sos')
        high_band = sosfiltfilt(sos_split, audio)
        low_band = audio - high_band

        b_env, a_env = butter(2, 50.0 / nyq, btype='lowpass')
        high_env = filtfilt(b_env, a_env, np.abs(high_band))

        gain_reduction = np.clip(0.06 / (high_env + 1e-6), 0.25, 1.0)
        gain_smooth = filtfilt(b_env, a_env, gain_reduction)
        return (low_band + high_band * gain_smooth).astype(np.float32)

    @staticmethod
    def _cinematic_air_exciter(audio: np.ndarray, sr: int) -> np.ndarray:
        if len(audio) < 150:
            return audio
        from scipy.signal import butter, sosfiltfilt
        nyq = sr * 0.5
        source_band = sosfiltfilt(butter(2, [3000.0 / nyq, 6000.0 / nyq], btype='bandpass', output='sos'), audio)
        harmonics = np.tanh(source_band * 2.0) - source_band
        try:
            air_band = sosfiltfilt(butter(2, min(0.99, 10000.0 / nyq), btype='highpass', output='sos'), harmonics)
        except ValueError:
            return audio
        return (audio + air_band * 0.3).astype(np.float32)

    @staticmethod
    def _apply_equal_power_fades(audio: np.ndarray, sr: int, fade_ms: float = 6.0) -> np.ndarray:
        frames = int(sr * fade_ms / 1000.0)
        if len(audio) < frames * 2:
            return audio
        out = audio.copy()
        if frames > 0:
            curve_in = np.sin(np.linspace(0, np.pi / 2, frames))
            curve_out = np.cos(np.linspace(0, np.pi / 2, frames))
            out[:frames] *= curve_in.astype(np.float32)
            out[-frames:] *= curve_out.astype(np.float32)
        return out

    # ── BỘ HÀM GỐC (GIỮ NGUYÊN — fallback / tương thích) ─────────────────────

    @staticmethod
    def _high_pass_filter(audio: np.ndarray, sr: int, cutoff_hz: float = 85.0) -> np.ndarray:
        if len(audio) < 16:
            return audio
        from scipy.signal import butter, sosfilt
        nyquist = sr * 0.5
        norm_cutoff = min(0.99, cutoff_hz / nyquist)
        sos = butter(4, norm_cutoff, btype="highpass", output="sos")
        return sosfilt(sos, audio).astype(np.float32)

    @staticmethod
    def _normalize_segment(
        audio: np.ndarray, target_rms: float = 0.16, max_peak: float = 0.97,
        gain_min: float = 0.5, gain_max: float = 4.0,
    ) -> np.ndarray:
        if len(audio) == 0:
            return audio
        audio = audio - float(np.mean(audio))
        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
        if rms < 1e-6:
            return audio.astype(np.float32)
        # [V11.7] Giới hạn hệ số khuếch đại: câu rất nhỏ (im lặng/hơi thở) KHÔNG bị
        # đẩy lên nhiều lần làm nổi nhiễu nền; câu to KHÔNG bị nén mất uy lực →
        # giữ động học/biểu cảm tự nhiên giữa các câu thay vì cào bằng tuyệt đối.
        gain = target_rms / rms
        gain = max(gain_min, min(gain, gain_max))
        normalized = audio * gain
        peak = float(np.max(np.abs(normalized)))
        if peak > max_peak:
            normalized = normalized * (max_peak / peak)
        return normalized.astype(np.float32)

    @staticmethod
    def _apply_fades(
        audio: np.ndarray, sr: int, fade_in_ms: float = 4.0, fade_out_ms: float = 8.0
    ) -> np.ndarray:
        n = len(audio)
        if n == 0:
            return audio
        fi = int(sr * fade_in_ms / 1000.0)
        fo = int(sr * fade_out_ms / 1000.0)
        if n < fi + fo + 2:
            return audio
        out = audio.copy()
        if fi > 0:
            out[:fi] *= np.linspace(0.0, 1.0, fi, dtype=np.float32)
        if fo > 0:
            out[-fo:] *= np.linspace(1.0, 0.0, fo, dtype=np.float32)
        return out

    @classmethod
    def _measure_true_peak(cls, master: np.ndarray, sr: int, oversample: int = 4) -> float:
        """Đo true-peak (đỉnh liên-mẫu) bằng oversample ``oversample``×.

        [v3.23.179] Uỷ cho hàm thuần ``_true_peak_chunked_overlap`` — xử lý theo chunk
        CÓ overlap ở ranh giới để không bỏ sót đỉnh liên-mẫu vắt qua ranh giới chunk
        (an toàn RAM cho master dài, đo đúng cả ở biên chunk).

        Args:
            master: Tín hiệu cần đo (không bị sửa đổi).
            sr: Tần số lấy mẫu (Hz).
            oversample: Hệ số oversample (4 = chuẩn ITU-R BS.1770).

        Returns:
            Giá trị true-peak tuyến tính (>= 0.0).
        """
        return _true_peak_chunked_overlap(master, sr, oversample=oversample)

    @staticmethod
    def _true_peak_limit(
        master: np.ndarray, sr: int, ceiling: float = 0.95, lookahead_ms: float = 1.5
    ) -> np.ndarray:
        """Ghìm CỤC BỘ chỉ các đỉnh liên-mẫu (true-peak) vượt ``ceiling``.

        [v3.23.220] Uỷ cho :func:`...tts.audio_mastering.true_peak_limit` — bản cài đặt
        tại đây là BẢN SAO Y HỆT (đã đối chiếu từng dòng) của hàm dùng chung mà
        VieNeu/Gemini vẫn gọi qua ``master_finalize``. Hai bản sao song song là mầm mống
        lệch parity giữa các engine (đúng dạng bug đã trả giá ở v215-v219) -> gộp về một
        nguồn sự thật. Giữ nguyên chữ ký + hành vi (sửa in-place và trả về mảng).

        Args:
            master: tín hiệu float32 (bị sửa in-place và cũng trả về).
            sr: tần số lấy mẫu.
            ceiling: trần biên độ tuyến tính cho true-peak (0.95 ≈ -0.45 dBTP).
            lookahead_ms: cửa sổ nhìn trước để bắt đỉnh trước khi nó tới.

        Returns:
            Mảng master đã giới hạn true-peak cục bộ.
        """
        from subtitles_extractor.infrastructure.tts.audio_mastering import (
            true_peak_limit,
        )

        return true_peak_limit(master, sr, ceiling=ceiling, lookahead_ms=lookahead_ms)

    @staticmethod
    def _soft_limit_master(
        master: np.ndarray, threshold: float = 0.90, ceiling: float = 0.97
    ) -> np.ndarray:
        """Nén mềm (tanh knee) mọi mẫu vượt ``threshold`` về sát ``ceiling``.

        [v3.23.220] Uỷ cho :func:`...tts.audio_mastering.soft_limit` (bản sao y hệt, chỉ
        khác giá trị mặc định) — một nguồn sự thật cho cả ba engine. Đỉnh ra LUÔN
        ≤ ``ceiling`` nên an toàn true-peak liên-mẫu khi phát/encode, không méo PCM_16.

        Args:
            master: tín hiệu float32 (không bị sửa in-place).
            threshold: ngưỡng bắt đầu nén mềm.
            ceiling: trần biên độ đầu ra.

        Returns:
            Mảng float32 đã nén mềm phần vượt ngưỡng.
        """
        from subtitles_extractor.infrastructure.tts.audio_mastering import soft_limit

        return soft_limit(master, threshold=threshold, ceiling=ceiling)

    @staticmethod
    def _lookahead_soft_limiter(
        master: np.ndarray, sr: int, threshold: float = 0.95, lookahead_ms: float = 5.0
    ) -> np.ndarray:
        """Soft-limiter có lookahead, AN TOÀN RAM cho master rất dài.

        [SỬA] Làm mượt đường gain bằng ``uniform_filter1d`` (chạy C, gần như
        in-place, ít RAM) thay cho ``filtfilt`` — vốn cấp phát bộ đệm gấp nhiều
        lần và có thể gây OOM/treo trên file dài (hàng trăm triệu mẫu), nhất là
        khi tạo nhiều lần liên tiếp trong cùng phiên.
        """
        abs_m = np.abs(master)
        if not np.any(abs_m > threshold):
            return master

        from scipy.ndimage import maximum_filter1d, uniform_filter1d

        lookahead_frames = max(1, int(sr * lookahead_ms / 1000.0))
        window_size = lookahead_frames * 2 + 1
        env = maximum_filter1d(abs_m, size=window_size)

        gain = np.ones_like(master)
        over = env > threshold
        if np.any(over):
            gain[over] = (
                threshold + (1.0 - threshold) * np.tanh((env[over] - threshold) / (1.0 - threshold))
            ) / env[over]

        # Làm mượt gain ~20ms để tránh "pumping" mà không tốn RAM như filtfilt.
        smooth_win = max(1, int(sr * 0.02))
        smoothed = uniform_filter1d(gain, size=smooth_win)
        # [v3.23.176] KẸP: làm mượt chỉ được GIẢM gain, không nới lỏng mức ghìm tại
        # đỉnh transient (nếu không, đỉnh nhọn đơn lẻ lọt méo — đo thực 2.5 -> 1.747).
        gain = _clamp_smoothed_gain(smoothed, gain)
        return (master * gain).astype(np.float32)

    @classmethod
    def _measure_lufs(cls, audio: np.ndarray, sr: int) -> float:
        """Đo loudness tích hợp (LUFS) theo EBU R128 CÓ gating (bỏ im lặng).

        [v3.23.177] Trước đây lấy mean-square TOÀN CỤC (gồm cả khoảng lặng giữa câu) ->
        LUFS đo thấp hơn thực -> gain chuẩn hoá bị đẩy cao oan -> giọng to quá mức. Nay
        áp K-weighting theo chunk (RAM thấp cho master dài) rồi tính loudness có
        absolute + relative gating qua hàm thuần ``_gated_loudness_from_kweighted``.

        Args:
            audio: Tín hiệu master (float).
            sr: Tần số lấy mẫu (Hz).

        Returns:
            Loudness tích hợp (LUFS); ``-inf`` nếu tín hiệu rỗng/toàn im lặng.
        """
        # [v3.23.213] Delegate sang bộ đo dùng chung (``audio_mastering.measure_lufs``):
        # CÙNG hệ số K-weighting (đã đối chiếu trùng khít) + cùng gating, nhưng đã tối
        # ưu RAM ở v212 (ghi thẳng vào mảng float32 cấp sẵn thay vì tích luỹ list
        # float64 rồi concatenate ~9x RAM audio). Bản sao cũ tại đây dính ĐÚNG bug đó
        # -> phim dài ngốn hàng GB. Một nguồn sự thật (DRY).
        from subtitles_extractor.infrastructure.tts.audio_mastering import measure_lufs

        return measure_lufs(audio, sr)

    @staticmethod
    def _ola_time_stretch(audio: np.ndarray, sr: int, ratio: float) -> np.ndarray:
        """Overlap-Add có dò tương quan — dùng cho tỉ lệ nhỏ hoặc đoạn ngắn.

        [v3.23.220] Uỷ cho :func:`...tts.time_stretch.ola_time_stretch` (bản dùng chung
        nay đã khớp độ dài bằng ``fit_length_no_silence`` y hệt bản cũ tại đây).

        Args:
            audio: Tín hiệu mono float32.
            sr: Tần số lấy mẫu.
            ratio: >1 = nhanh hơn (ngắn lại); <1 = chậm hơn.

        Returns:
            Tín hiệu đã co giãn.
        """
        from subtitles_extractor.infrastructure.tts.time_stretch import ola_time_stretch

        return ola_time_stretch(audio, sr, ratio)

    @classmethod
    def _time_stretch(cls, audio: np.ndarray, sr: int, ratio: float) -> np.ndarray:
        if abs(ratio - 1.0) < 0.02:
            return audio
        try:
            import librosa
            return np.asarray(
                librosa.effects.time_stretch(audio, rate=float(ratio)), dtype=np.float32
            )
        except ImportError:
            pass
        return cls._ola_time_stretch(audio, sr, ratio)

    # ── ASYNC NETWORK CORE ────────────────────────────────────────

    async def _async_gen_with_retry(
        self, text: str, voice: str, speed: float, request: TTSRequest
    ) -> tuple[np.ndarray, int]:
        for attempt in range(max(1, request.retry_count)):
            try:
                if not text.strip():
                    return np.array([], dtype=np.float32), 24000
                rate = _rate_from_speed(speed)
                audio, sr = await self._async_generate(text, voice, rate)
                # [v3.23.210] Audio toàn im lặng = thất bại -> retry (đồng bộ VieNeu
                # v205: lưới len>0 để lọt audio CÓ độ dài nhưng không tiếng -> mất
                # thoại mà vẫn báo OK).
                if len(audio) > 0 and not _is_silent_audio(audio):
                    return audio, sr
                logger.warning(
                    "Edge TTS async lần %d/%d: audio %s cho '%s…'",
                    attempt + 1, request.retry_count,
                    "rỗng" if len(audio) == 0 else "toàn im lặng", text[:25],
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Edge TTS async lần %d/%d: %s", attempt + 1, request.retry_count, exc)
            if attempt < request.retry_count - 1:
                # [v3.20.3 #3net] Exponential backoff + JITTER ngẫu nhiên: chống
                # "bão kết nối lại" khi 64 luồng cùng rớt mạng và retry đồng loạt
                # (gây 429 Too Many Requests / bị Anti-DDoS chặn). Jitter rải các
                # luồng thức dậy lệch nhau → mượt băng thông.
                base_delay = request.retry_delay_s * (attempt + 1)
                await asyncio.sleep(base_delay + random.uniform(0.0, request.retry_delay_s))

        # [v3.23.38] Fallback cuối: một số văn bản bị EdgeTTS từ chối khi BỌC TRONG
        # NGOẶC hoặc kèm ký hiệu (vd '(nói tiếng Quan Thoại)'). Thử lại MỘT lần với
        # phần chữ bên trong (bỏ ngoặc/ký hiệu) — thường đọc được. Chỉ thử nếu khác
        # văn bản gốc và còn nội dung đọc được.
        stripped = _MULTISPACE_RE.sub(
            " ",
            _CURLY_RE.sub(" ", _SQUARE_RE.sub(" ", _PAREN_RE.sub(" ", text)))
            .replace(_MUSIC_NOTE, " "),
        ).strip()
        if stripped and stripped != text.strip() and _has_speakable_content(stripped):
            try:
                audio, sr = await self._async_generate(stripped, voice, _rate_from_speed(speed))
                if len(audio) > 0:
                    logger.info("Edge TTS đọc được sau khi bỏ ngoặc/ký hiệu: %r", stripped[:40])
                    return audio, sr
            except Exception as exc:  # noqa: BLE001
                logger.warning("Edge TTS fallback bỏ ngoặc vẫn lỗi: %s", exc)
        return np.array([], dtype=np.float32), 24000

    #: Mã thoát worker báo THIẾU THƯ VIỆN (không phải lỗi mạng, không nên thử lại).
    _EXIT_CODE_LIBRARY_MISSING: int = 3

    #: [v3.23.342] Đã biết Python ngoài thiếu edge-tts -> khỏi thử subprocess nữa.
    #: Thử lại vô ích tốn ~2 giây mỗi câu; với 55 câu là gần 2 phút.
    _prefer_in_process: bool = False

    @staticmethod
    async def _run_edge_in_process(
        text: str, voice: str, rate: str, output_path: str
    ) -> bool:
        """Chạy edge-tts NGAY trong tiến trình này (dự phòng khi không có Python ngoài).

        Dùng khi bản đóng gói không tìm được Python hệ thống. Kém cách ly hơn subprocess
        (lỗi/treo của edge-tts ảnh hưởng trực tiếp) nên chỉ dùng làm phương án cuối —
        nhưng vẫn hơn hẳn việc báo thất bại khi thư viện đã nằm sẵn trong bundle.

        Args:
            text: Câu cần đọc.
            voice: Mã giọng (vd ``vi-VN-NamMinhNeural``).
            rate: Tốc độ dạng ``"+0%"``.
            output_path: Tệp WAV đích.

        Returns:
            ``True`` nếu tổng hợp thành công.
        """
        import asyncio
        import tempfile

        try:
            import edge_tts
        except ImportError:
            logger.warning(
                "Edge TTS: không có Python ngoài VÀ thư viện edge-tts cũng không nằm "
                "trong bản đóng gói. Cài 'pip install edge-tts', hoặc dùng engine khác "
                "(VieNeu/Gemini)."
            )
            return False

        mp3_path = ""
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as handle:
                mp3_path = handle.name
            communicate = edge_tts.Communicate(text, voice, rate=rate)
            # Giới hạn thời gian giống worker: mạng chậm không được treo vô hạn.
            await asyncio.wait_for(communicate.save(mp3_path), timeout=30.0)
            # Tái dùng CHÍNH hàm chuyển đổi của worker để hai đường chạy cho kết quả
            # y hệt nhau (soundfile BSD, lùi về pydub MIT).
            from subtitles_extractor.infrastructure.tts.edge_tts_subprocess import (
                _mp3_to_wav,
            )

            return bool(_mp3_to_wav(mp3_path, output_path))
        except (TimeoutError, asyncio.TimeoutError):
            logger.warning("Edge TTS (trong tiến trình): quá hạn 30s — bỏ qua câu này.")
            return False
        except Exception as exc:  # noqa: BLE001 — lỗi mạng/dịch vụ không được phá TTS
            logger.warning("Edge TTS (trong tiến trình) lỗi: %s", exc)
            return False
        finally:
            if mp3_path:
                with contextlib.suppress(OSError):
                    Path(mp3_path).unlink(missing_ok=True)

    @staticmethod
    def _resolve_python_for_worker() -> str | None:
        """Tìm trình thông dịch Python để chạy worker edge-tts (cách ly GPL).

        - Chạy từ source (không đóng gói): dùng ``sys.executable`` (chính Python đang chạy).
        - Chạy đóng gói (PyInstaller ``sys.frozen``): ``sys.executable`` là app exe, KHÔNG
          phải Python. edge-tts KHÔNG được đóng gói kèm (để cách ly GPL) → tìm Python hệ
          thống ngoài. Trả None nếu không có (app báo dùng engine khác).

        Returns:
            Đường dẫn Python chạy được worker, hoặc None nếu không tìm thấy.
        """
        import shutil
        import sys

        if not getattr(sys, "frozen", False):
            return sys.executable

        # Đóng gói: tìm Python hệ thống (PATH). edge-tts cài ngoài như một tuỳ chọn.
        for candidate in ("python", "python3", "py"):
            found = shutil.which(candidate)
            if found:
                return found
        return None

    @staticmethod
    async def _async_generate(text: str, voice: str, rate: str) -> tuple[np.ndarray, int]:
        # [v3.23.268] Gọi edge-tts qua SUBPROCESS RIÊNG để cách ly GPL (edge-tts là GPL v3;
        # chạy tiến trình riêng để không liên kết vào ứng dụng thương mại). Tiến trình chính
        # KHÔNG import edge_tts — chỉ chạy worker + đọc WAV kết quả. Xem docs/LICENSE_ANALYSIS.
        import soundfile as sf

        fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            ok = await EdgeTTSAdapter._run_edge_subprocess(text, voice, rate, tmp_path)
            if not ok:
                return np.array([], dtype=np.float32), 24000
            try:
                audio, sr = sf.read(tmp_path, dtype="float32")
            except Exception as read_exc:  # noqa: BLE001 - kết quả rỗng -> bỏ dòng, thử lại
                logger.debug("Đọc WAV Edge subprocess lỗi: %s", read_exc)
                return np.array([], dtype=np.float32), 24000
            if isinstance(audio, np.ndarray) and audio.ndim > 1:
                audio = audio.mean(axis=1)
            return np.asarray(audio, dtype=np.float32), int(sr)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    @staticmethod
    async def _run_edge_subprocess(
        text: str, voice: str, rate: str, output_path: str
    ) -> bool:
        """Chạy worker edge-tts trong tiến trình riêng. True nếu tạo được WAV.

        Cách ly GPL: edge-tts chỉ được import trong worker subprocess, không trong tiến
        trình chính. Timeout cứng 35s (worker tự timeout 30s + biên an toàn).
        """
        from pathlib import Path

        worker = Path(__file__).with_name("edge_tts_subprocess.py")
        python_exe = EdgeTTSAdapter._resolve_python_for_worker()
        if EdgeTTSAdapter._prefer_in_process:
            # Lần trước đã xác định Python ngoài thiếu edge-tts — đi thẳng đường trong
            # tiến trình, khỏi tốn ~2 giây/câu cho một subprocess chắc chắn thất bại.
            return await EdgeTTSAdapter._run_edge_in_process(
                text, voice, rate, output_path
            )
        if python_exe is None or not worker.is_file():
            # [v3.23.328] Trước đây tới đây là THẤT BẠI: thiết kế cũ cố ý KHÔNG đóng gói
            # edge-tts (cách ly GPL) nên buộc phải có Python ngoài. Dự án nay là mã nguồn
            # mở và spec ĐÃ gom edge_tts vào bundle, nên chạy thẳng trong tiến trình được.
            # Vẫn ưu tiên subprocess khi có Python ngoài, vì cách ly giúp edge-tts treo
            # hoặc lỗi mạng không làm sập ứng dụng chính.
            return await EdgeTTSAdapter._run_edge_in_process(
                text, voice, rate, output_path
            )
        args = [
            python_exe, str(worker),
            "--text", text,
            "--voice", voice,
            "--rate", rate,
            "--output", output_path,
        ]
        try:
            # [v3.23.350] Trước đây dict ``flags`` cục bộ đặt ``creationflags`` RỒI
            # còn nối thêm ``**no_window_kwargs()`` (cũng đặt ``creationflags``) →
            # "got multiple values for keyword argument 'creationflags'" → Edge TTS
            # SẬP HOÀN TOÀN trên Windows. Dùng MỘT nguồn duy nhất.
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
                **no_window_kwargs(),
            )
            try:
                _stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=35.0)
            except asyncio.TimeoutError:
                proc.kill()
                logger.warning("Edge TTS subprocess quá thời gian (35s) — bỏ dòng.")
                return False
            if proc.returncode != 0:
                detail = stderr.decode("utf-8", "replace").strip() if stderr else ""
                logger.debug(
                    "Edge TTS subprocess lỗi (mã %s): %s", proc.returncode, detail
                )
                # [v3.23.342] SỬA LỖI THẬT: mã 3 = Python NGOÀI không có thư viện
                # edge-tts. Nhưng bản đóng gói CÓ gom edge_tts vào bundle, nên chạy
                # trong tiến trình là được. Trước đây chỉ trả False rồi thử lại 10 lần
                # cùng một cách sai — tốn ~50 giây rồi báo "kiểm tra mạng", trong khi
                # mạng hoàn toàn bình thường.
                if proc.returncode == EdgeTTSAdapter._EXIT_CODE_LIBRARY_MISSING:
                    logger.info(
                        "Edge TTS: Python ngoài không có thư viện edge-tts — chuyển "
                        "sang chạy trong tiến trình (thư viện đã nằm trong bản đóng gói)."
                    )
                    EdgeTTSAdapter._prefer_in_process = True
                    return await EdgeTTSAdapter._run_edge_in_process(
                        text, voice, rate, output_path
                    )
                return False
            return Path(output_path).exists() and Path(output_path).stat().st_size > 0
        except (OSError, ValueError) as exc:
            logger.warning("Không chạy được Edge TTS subprocess: %s", exc)
            return False

    def _sync_generate_with_retry(
        self, text: str, voice: str, speed: float, request: TTSRequest
    ) -> tuple[np.ndarray, int] | None:
        for attempt in range(max(1, request.retry_count)):
            try:
                rate = _rate_from_speed(speed)
                try:
                    audio, sr = asyncio.run(self._async_generate(text, voice, rate))
                except RuntimeError as exc:
                    if "event loop" in str(exc).lower():
                        loop = asyncio.new_event_loop()
                        try:
                            audio, sr = loop.run_until_complete(
                                self._async_generate(text, voice, rate)
                            )
                        finally:
                            loop.close()
                    else:
                        raise
                if len(audio) > 0 and not _is_silent_audio(audio):
                    return audio, sr
                logger.warning(
                    "Edge TTS probe lần %d/%d: audio %s cho '%s…'",
                    attempt + 1, request.retry_count,
                    "rỗng" if len(audio) == 0 else "toàn im lặng", text[:25],
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Edge TTS probe lần %d/%d: %s", attempt + 1, request.retry_count, exc)
            if attempt < request.retry_count - 1:
                # [v3.20.3 #3net] Backoff + jitter (xem ghi chú ở _async_gen_with_retry).
                base_delay = request.retry_delay_s * (attempt + 1)
                time.sleep(base_delay + random.uniform(0.0, request.retry_delay_s))
        return None


__all__ = ["EdgeTTSAdapter"]
