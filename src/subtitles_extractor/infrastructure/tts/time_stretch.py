"""Co giãn thời gian GIỮ NGUYÊN CAO ĐỘ (pitch-preserving time-stretch).

Tách thành module dùng chung để mọi engine TTS (Edge, Gemini, VieNeu) tăng/giảm tốc độ
đọc mà giọng KHÔNG bị méo cao độ ("giọng sóc chuột"). Trước đây các engine thiếu
``librosa`` rơi về ``scipy.signal.resample`` — vốn đổi cả cao độ. Ở đây dùng:

* WSOLA (Waveform Similarity Overlap-Add) có dò tương quan chéo để giữ pha thanh quản,
* OLA cho tỉ lệ nhỏ hoặc đoạn quá ngắn,
* librosa cho tỉ lệ rất lớn (>2×) nếu có sẵn (chất lượng tốt nhất).

Quy ước ``ratio``: >1 = đọc NHANH hơn (audio ngắn lại); <1 = chậm hơn (dài ra).
"""

from __future__ import annotations

import logging

import numpy as np

from subtitles_extractor.infrastructure.tts.dsp_primitives import fit_length_no_silence

logger = logging.getLogger(__name__)

# Dưới ngưỡng này coi như không cần co giãn (tránh xử lý thừa).
_NEGLIGIBLE = 0.02

# [v3.23.240] Chất lượng nén phụ thuộc pedalboard có được cài hay không — mà đây là
# OPTIONAL dependency (GPL, app không bundle). Khi vắng nó, pipeline lặng lẽ rơi về WSOLA:
# ổn ở nén nhẹ, nhưng ở nén MẠNH (>1.6x) WSOLA làm formant tan thành "tiếng gió". Trước
# đây người dùng KHÔNG có cách nào biết mình đang nghe chất lượng thấp hơn.
#
# Cảnh báo đúng MỘT LẦN mỗi phiên (không lặp mỗi câu -> không gây nhiễu log), và chỉ khi
# thật sự cần: có câu nén mạnh mà pedalboard lại vắng.
_PEDALBOARD_HINT_RATIO = 1.6
_pedalboard_hint_shown = False


def _goi_y_cai_pedalboard(ratio: float) -> None:
    """Nhắc cài pedalboard MỘT LẦN mỗi phiên khi nén mạnh mà thiếu nó (hàm side-effect).

    Args:
        ratio: Hệ số nén của câu hiện tại (chỉ nhắc khi vượt ngưỡng đáng kể).
    """
    global _pedalboard_hint_shown
    if _pedalboard_hint_shown or ratio < _PEDALBOARD_HINT_RATIO:
        return
    _pedalboard_hint_shown = True
    logger.info(
        "Có câu phải nén %.2fx nhưng thiếu 'pedalboard' — đang dùng WSOLA (giọng nén "
        "mạnh dễ mất độ rõ). Cài 'pip install pedalboard' để giữ formant chuẩn studio. "
        "Nhắc một lần cho mỗi phiên.",
        ratio,
    )


def ola_time_stretch(audio: np.ndarray, sr: int, ratio: float) -> np.ndarray:
    """Overlap-Add có dò tương quan — nhẹ, dùng cho tỉ lệ nhỏ hoặc đoạn ngắn."""
    frame_len = int(sr * 0.04)
    if frame_len < 4 or len(audio) < frame_len * 2:
        new_len = max(1, int(len(audio) / ratio))
        return np.interp(
            np.linspace(0, len(audio) - 1, new_len), np.arange(len(audio)), audio
        ).astype(np.float32)

    hop_syn = frame_len // 2
    hop_ana = max(1, int(hop_syn * ratio))
    search = max(1, frame_len // 4)
    window = np.hanning(frame_len).astype(np.float32)

    n_frames = max(1, (len(audio) - frame_len) // hop_ana + 1)
    out_len = (n_frames - 1) * hop_syn + frame_len
    out = np.zeros(out_len, dtype=np.float32)
    norm = np.zeros(out_len, dtype=np.float32)

    prev_tail: np.ndarray | None = None
    for i in range(n_frames):
        nominal = i * hop_ana
        if prev_tail is None or i == 0:
            a0 = nominal
        else:
            lo = max(0, nominal - search)
            hi = min(len(audio) - frame_len, nominal + search)
            if hi <= lo:
                a0 = min(nominal, len(audio) - frame_len)
            else:
                seg_len = len(prev_tail)
                candidates = audio[lo:hi + seg_len]
                if len(candidates) >= seg_len and seg_len > 0:
                    xcorr = np.correlate(candidates, prev_tail, mode="valid")
                    a0 = lo + int(np.argmax(xcorr))
                else:
                    a0 = lo
        frame = audio[a0:a0 + frame_len]
        if len(frame) < frame_len:
            frame = np.pad(frame, (0, frame_len - len(frame)))
        s0 = i * hop_syn
        out[s0:s0 + frame_len] += frame * window
        norm[s0:s0 + frame_len] += window
        prev_tail = frame[hop_syn:hop_syn + (frame_len - hop_syn)].copy()

    norm[norm < 1e-6] = 1.0
    result = (out / norm).astype(np.float32)
    target_len = max(1, int(len(audio) / ratio))
    # [v3.23.220] Khớp độ dài KHÔNG chèn im lặng cuối. Bản cũ ``np.pad`` zeros phần
    # THIẾU -> chèn tới 17ms im lặng vào đuôi giọng (câu nghe "hụt hơi"). Bug này đã sửa
    # cho Edge ở v3.23.174 nhưng KHÔNG được port sang module dùng chung -> Gemini (và mọi
    # caller không có pedalboard) vẫn dính. Nay ba engine cùng một nguồn sự thật.
    return fit_length_no_silence(result, target_len)


def wsola_time_stretch(audio: np.ndarray, sr: int, ratio: float) -> np.ndarray:
    """WSOLA giữ pha thanh quản — chất lượng tốt cho dải tỉ lệ vừa phải."""
    frame_len = int(sr * 0.03)
    if frame_len < 4 or len(audio) < frame_len * 2:
        return ola_time_stretch(audio, sr, ratio)

    hop_syn = frame_len // 2
    hop_ana = max(1, int(hop_syn * ratio))
    search = max(1, frame_len // 3)

    n_frames = max(1, (len(audio) - frame_len) // hop_ana + 1)
    out_len = (n_frames - 1) * hop_syn + frame_len
    out = np.zeros(out_len, dtype=np.float32)
    norm = np.zeros(out_len, dtype=np.float32)
    window = np.hanning(frame_len).astype(np.float32)

    prev_tail: np.ndarray | None = None
    for i in range(n_frames):
        nominal = i * hop_ana
        if prev_tail is None or i == 0:
            a0 = nominal
        else:
            lo = max(0, nominal - search)
            hi = min(len(audio) - frame_len, nominal + search)
            if hi <= lo:
                a0 = min(nominal, len(audio) - frame_len)
            else:
                seg_len = len(prev_tail)
                candidates = audio[lo:hi + seg_len]
                if len(candidates) >= seg_len and seg_len > 0:
                    xcorr = np.correlate(candidates, prev_tail, mode="valid")
                    a0 = lo + int(np.argmax(xcorr))
                else:
                    a0 = lo
        frame = audio[a0:a0 + frame_len]
        if len(frame) < frame_len:
            frame = np.pad(frame, (0, frame_len - len(frame)))
        s0 = i * hop_syn
        out[s0:s0 + frame_len] += frame * window
        norm[s0:s0 + frame_len] += window
        prev_tail = frame[hop_syn:hop_syn + (frame_len - hop_syn)].copy()

    norm[norm < 1e-6] = 1.0
    result = (out / norm).astype(np.float32)
    target_len = max(1, int(len(audio) / ratio))
    # [v3.23.220] Khớp độ dài KHÔNG chèn im lặng cuối. Bản cũ ``np.pad`` zeros phần
    # THIẾU -> chèn tới 17ms im lặng vào đuôi giọng (câu nghe "hụt hơi"). Bug này đã sửa
    # cho Edge ở v3.23.174 nhưng KHÔNG được port sang module dùng chung -> Gemini (và mọi
    # caller không có pedalboard) vẫn dính. Nay ba engine cùng một nguồn sự thật.
    return fit_length_no_silence(result, target_len)


def stretch_with_pedalboard(
    audio: np.ndarray, sr: int, ratio: float
) -> np.ndarray | None:
    """[v3.23.199] Time-stretch bằng pedalboard (Rubber Band engine) nếu đã cài.

    Rubber Band là engine time-stretch formant-preserving chuẩn studio: giữ trọn năng
    lượng/độ rõ giọng nói kể cả nén 2-3x (nơi phase-vocoder/WSOLA làm tan formant thành
    "tiếng gió"). Là OPTIONAL dependency lazy-import: app không bundle (pedalboard GPL-3,
    Rubber Band dual-license GPL/thương mại) — người dùng tự ``pip install pedalboard``.
    Đặt tại module dùng chung để MỌI engine (Edge/Gemini/VieNeu) cùng hưởng.

    Args:
        audio: Tín hiệu mono float32.
        sr: Tần số lấy mẫu.
        ratio: Hệ số tốc độ (>1 = nhanh hơn/ngắn lại) — cùng semantics stretch_factor.

    Returns:
        Audio đã stretch (mono float32), hoặc ``None`` nếu pedalboard chưa cài / lỗi
        (caller fallback librosa/WSOLA/OLA).
    """
    try:
        # [v3.23.266] Cờ tắt pedalboard cho build THƯƠNG MẠI: pedalboard là GPL v3 (nhúng
        # JUCE/VST3). Đặt SUBEXT_DISABLE_PEDALBOARD=1 để ép fallback librosa (ISC license,
        # an toàn thương mại) — xem docs/LICENSE_ANALYSIS.md. Chất lượng librosa tốt, chỉ
        # chậm hơn chút.
        import os

        if os.environ.get("SUBEXT_DISABLE_PEDALBOARD") == "1":
            return None
        from pedalboard import time_stretch as _pb_time_stretch
    except ImportError:
        _goi_y_cai_pedalboard(ratio)
        return None
    try:
        stretched = np.asarray(
            _pb_time_stretch(audio, float(sr), stretch_factor=float(ratio)),
            dtype=np.float32,
        )
    except (RuntimeError, ValueError) as exc:
        logger.warning("pedalboard time_stretch lỗi (fallback WSOLA/librosa): %s", exc)
        return None
    # pedalboard trả (channels, samples) — ép về mono 1D cho pipeline.
    if stretched.ndim > 1:
        stretched = (
            stretched.reshape(-1) if stretched.shape[0] == 1 else stretched.mean(axis=0)
        )
    # [v3.23.201] Rubber Band làm RMS giảm theo mức nén (đo thực: -16%% @2x) -> câu nén
    # nghe NHỎ tương đối. Bù về RMS gốc ngay tại nguồn — mọi caller cùng hưởng.
    source_rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    return match_rms(stretched.astype(np.float32), source_rms)


def match_rms(
    stretched: np.ndarray, target_rms: float, max_gain: float = 2.0
) -> np.ndarray:
    """[v3.23.201] Bù âm lượng sau time-stretch về mức RMS tham chiếu (hàm thuần).

    Đo thực nghiệm trên giọng VieNeu thật: Rubber Band (pedalboard) làm RMS GIẢM dần
    theo mức nén (-16%% @2x, -22%% @3x) trong khi WSOLA giữ nguyên. Master chuẩn hoá
    loudness TOÀN file nên câu nén sâu nghe NHỎ tương đối so với câu xung quanh -> giảm
    khả năng truyền đạt nội dung. Bù RMS per-segment khắc phục; trần ``max_gain`` chống
    khuếch đại nhiễu khi đoạn gần im lặng.

    Args:
        stretched: Audio sau time-stretch (mono float32).
        target_rms: RMS của audio TRƯỚC stretch (mức tham chiếu).
        max_gain: Trần hệ số khuếch đại (an toàn nhiễu).

    Returns:
        Audio đã bù âm lượng (float32); trả nguyên bản nếu không cần/không thể bù.
    """
    if stretched.size == 0 or target_rms <= 0.0:
        return stretched
    current_rms = float(np.sqrt(np.mean(stretched.astype(np.float64) ** 2)))
    if current_rms <= 1e-8:
        return stretched  # gần im lặng — khuếch đại chỉ ra nhiễu
    gain = min(target_rms / current_rms, max_gain)
    if abs(gain - 1.0) < 0.02:
        return stretched  # chênh không đáng kể
    return (stretched * gain).astype(np.float32)


def vocal_time_stretch(audio: np.ndarray, sr: int, ratio: float) -> np.ndarray:
    """[v3.23.220] Co giãn GIỌNG NÓI — đường chất lượng cao dùng chung cho mọi engine.

    Trước đây thuật toán này là ``EdgeTTSAdapter._time_stretch_vocal`` (staticmethod nằm
    trong một ADAPTER), khiến VieNeu phải import ngược ``EdgeTTSAdapter`` chỉ để nén
    giọng — adapter phụ thuộc adapter. Chuyển về đây NGUYÊN VĂN hành vi:

    1. Bỏ qua nếu tỉ lệ không đáng kể (< 2%).
    2. Rubber Band (pedalboard) nếu người dùng đã cài — giữ formant kể cả nén 2-3x.
    3. librosa (phase-vocoder) cho biến đổi MẠNH cả hai chiều (>2x hoặc <0.5x).
    4. WSOLA phase-locked (khung 30ms) cho dải vừa; OLA khi đoạn quá ngắn.

    Khác với :func:`time_stretch_preserve_pitch` ở hai điểm CÓ CHỦ Ý (giữ nguyên hành vi
    đã nghiệm thu của Edge/VieNeu): dùng librosa cho cả chiều GIÃN mạnh (<0.5x — Pass 2.5
    của Edge sinh yêu cầu này), và không chuyển sang OLA ở dải nén nhẹ.

    Args:
        audio: Tín hiệu mono float32.
        sr: Tần số lấy mẫu.
        ratio: >1 = đọc nhanh hơn (audio ngắn lại); <1 = chậm hơn.

    Returns:
        Tín hiệu đã co giãn (float32), cao độ giữ nguyên.
    """
    if abs(ratio - 1.0) < _NEGLIGIBLE:
        return audio

    stretched = stretch_with_pedalboard(audio, sr, ratio)
    if stretched is not None:
        return stretched

    if ratio > 2.0 or ratio < 0.5:
        try:
            import librosa

            return np.asarray(
                librosa.effects.time_stretch(audio, rate=float(ratio)),
                dtype=np.float32,
            )
        except ImportError:
            pass

    frame_len = int(sr * 0.03)
    if frame_len < 4 or len(audio) < frame_len * 2:
        return ola_time_stretch(audio, sr, ratio)
    return wsola_time_stretch(audio, sr, ratio)


def time_stretch_preserve_pitch(audio: np.ndarray, sr: int, ratio: float) -> np.ndarray:
    """Co giãn thời gian giữ cao độ — điểm vào chung cho mọi engine.

    Chọn thuật toán theo thứ tự ưu tiên: pedalboard/Rubber Band (nếu cài — chất lượng
    cao nhất, giữ formant ở mọi tỉ lệ); bỏ qua khi không đáng kể; OLA cho nén/giãn rất
    nhỏ (≤5%); librosa cho tỉ lệ rất lớn (>2×) nếu có; còn lại dùng WSOLA.

    Args:
        audio: tín hiệu mono float32.
        sr: tần số lấy mẫu.
        ratio: >1 nhanh hơn, <1 chậm hơn.

    Returns:
        Tín hiệu đã co giãn (float32), cao độ giữ nguyên.
    """
    audio = np.asarray(audio, dtype=np.float32)
    if abs(ratio - 1.0) < _NEGLIGIBLE or len(audio) < 8:
        return audio
    # [v3.23.199] Ưu tiên Rubber Band (pedalboard) — Gemini/mọi caller cùng hưởng
    # (trước đây chỉ đường _time_stretch_vocal của Edge/VieNeu có, Gemini bị bỏ sót).
    stretched = stretch_with_pedalboard(audio, sr, ratio)
    if stretched is not None:
        return stretched
    if ratio > 2.0:
        try:
            import librosa
            return np.asarray(
                librosa.effects.time_stretch(audio, rate=float(ratio)), dtype=np.float32
            )
        except ImportError:
            pass
    if abs(ratio - 1.0) <= 0.05:
        return ola_time_stretch(audio, sr, ratio)
    return wsola_time_stretch(audio, sr, ratio)
