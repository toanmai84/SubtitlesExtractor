"""Bộ xử lý âm thanh "phòng thu" + ghi đa định dạng — DÙNG CHUNG cho mọi engine TTS.

Trước đây chỉ Edge TTS có bộ lọc chuẩn phát thanh (LUFS, True-Peak, Voice-Clarity) và
xuất đa định dạng; các engine khác thì hard-code ``.wav`` và không xử lý. Module này
gom các thuật toán đó thành hàm thuần, tái dùng cho cả ba engine để chất lượng đồng
nhất.

Gồm bốn nhóm:
* Đo lường: :func:`measure_lufs`, :func:`measure_true_peak`.
* Làm rõ giọng: :func:`voice_clarity` (high-pass rumble + hạ mud + nâng presence +
  air, de-esser tách dải, cinematic air exciter).
* Chuẩn hoá độ to & giới hạn đỉnh: :func:`normalize_to_lufs`, :func:`soft_limit`,
  :func:`true_peak_limit`, và :func:`master_finalize` (chuỗi hoàn chỉnh).
* Ghi file: :func:`write_audio` (WAV/FLAC qua soundfile; MP3/OPUS/OGG/M4A qua ffmpeg
  pipe, giữ float32 trước khi nén).

Mọi hàm xử lý theo chunk khi cần để an toàn RAM với audio dài hàng giờ.

.. note::
    [v3.23.220] Module này KHÔNG được phép import từ bất kỳ ``*_tts_adapter`` nào (trước
    đây import ngược từ ``edge_tts_adapter`` -> vòng tròn phụ thuộc). Các nguyên hàm DSP
    nay nằm ở :mod:`subtitles_extractor.infrastructure.tts.dsp_primitives`.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from subtitles_extractor.infrastructure.process.hidden_process import no_window_kwargs
from pathlib import Path

from collections.abc import Callable

import numpy as np

from subtitles_extractor.infrastructure.tts.dsp_primitives import (
    gated_loudness_from_kweighted as _gated_loudness_from_kweighted,
)
from subtitles_extractor.infrastructure.tts.dsp_primitives import (
    inter_node_soft_clip as _inter_node_soft_clip,
)
from subtitles_extractor.infrastructure.tts.dsp_primitives import (
    loudness_gain_linear as _loudness_gain_linear,
)
from subtitles_extractor.infrastructure.tts.dsp_primitives import (
    true_peak_chunked_overlap as _true_peak_chunked_overlap,
)

logger = logging.getLogger(__name__)

# Hệ số K-weighting (ITU-R BS.1770) cho phép đo LUFS không cần pyloudnorm.
_KW_B1 = [1.53512485958697, -2.69169618940638, 1.19839281085285]
_KW_A1 = [1.0, -1.69065929318241, 0.73248077421585]
_KW_B2 = [1.0, -2.0, 1.0]
_KW_A2 = [1.0, -1.99004745483398, 0.99007225036621]


# ── Xử lý audio DÀI theo khối (tiết kiệm RAM) ────────────────────────────────
# [v3.23.212] Ngưỡng bật chế độ khối. Đo thực: chuỗi master ngốn RAM đỉnh ~15.8x kích
# thước audio (nhiều mảng trung gian float64 của sosfiltfilt/pyloudnorm cùng tồn tại)
# -> phim 2h (master 691MB) cần ~10.9GB, tập 45 phút cần ~4GB -> treo/crash máy đang
# chạy PaddleOCR. Video ngắn (< ngưỡng) giữ NGUYÊN đường cũ -> zero regression.
_LONG_AUDIO_S = 300.0
_BLOCK_S = 60.0
# Đệm warm-up mỗi biên khối: IIR zero-phase (sosfiltfilt) suy giảm rất nhanh nên 1s đã
# đủ để kết quả gần như đồng nhất với xử lý toàn cục (đo sai số ở test).
_BLOCK_PAD_S = 1.0


def process_in_blocks(
    audio: np.ndarray,
    sr: int,
    transform: "Callable[[np.ndarray, int], np.ndarray]",
    block_s: float = _BLOCK_S,
    pad_s: float = _BLOCK_PAD_S,
) -> np.ndarray:
    """[v3.23.212] Áp ``transform`` theo KHỐI có đệm warm-up (hàm thuần, tiết kiệm RAM).

    Mỗi khối được mở rộng ``pad_s`` về hai phía trước khi lọc rồi cắt bỏ phần đệm — bộ
    lọc IIR zero-phase "nguội" trong vùng đệm nên đường ghép không tạo click và kết quả
    gần như đồng nhất với xử lý toàn cục. RAM đỉnh chỉ còn tỉ lệ với KHỐI, không với
    toàn bộ phim.

    Args:
        audio: Tín hiệu mono float32.
        sr: Tần số lấy mẫu.
        transform: Hàm lọc thuần ``(audio, sr) -> audio`` giữ nguyên độ dài.
        block_s: Độ dài mỗi khối (giây).
        pad_s: Đệm warm-up mỗi biên (giây).

    Returns:
        Tín hiệu đã xử lý (float32), CÙNG độ dài với đầu vào.
    """
    n = audio.size
    block = max(1, int(block_s * sr))
    pad = max(1, int(pad_s * sr))
    if n <= block + 2 * pad:
        return transform(audio, sr)
    out = np.empty(n, dtype=np.float32)
    for start in range(0, n, block):
        end = min(start + block, n)
        lo = max(0, start - pad)
        hi = min(n, end + pad)
        chunk = transform(audio[lo:hi], sr)
        out[start:end] = chunk[start - lo : start - lo + (end - start)]
    return out


def clip_for_output(audio: np.ndarray, ceiling: float = 1.0) -> np.ndarray:
    """[v3.23.183] Chặn biên độ về [-ceiling, ceiling] trước khi GHI ra file.

    Bước bảo vệ cuối trước khi ghi: master đã qua limiter nhưng có thể còn mẫu vượt
    ±1.0 (engine ngoài không qua limiter, hoặc sai số cộng dồn). Với WAV/FLAC subtype
    FLOAT, soundfile GIỮ NGUYÊN mẫu vượt biên -> méo/clip khi phát lại hoặc encode tiếp
    (vd lồng vào video qua AAC). Path ffmpeg đã clip tường minh; hàm này bảo đảm path
    soundfile (WAV/FLAC) cũng nhất quán. Hàm THUẦN: không sửa mảng gốc.

    Args:
        audio: Tín hiệu cần chặn biên (không bị sửa in-place).
        ceiling: Ngưỡng biên độ tuyệt đối tối đa (mặc định 1.0 = full-scale).

    Returns:
        Mảng float32 với mọi mẫu trong [-ceiling, ceiling].
    """
    return np.clip(audio, -ceiling, ceiling).astype(np.float32)


# ── Đo lường ─────────────────────────────────────────────────────────────────
def measure_lufs(audio: np.ndarray, sr: int) -> float:
    """Đo độ to tích hợp (LUFS) theo EBU R128 CÓ gating (bỏ im lặng).

    [v3.23.182] Đồng bộ với ``edge_tts_adapter`` (fix v177): K-weighting theo chunk rồi
    tính loudness có absolute + relative gating -> im lặng giữa câu không kéo LUFS xuống
    sai (trước đây mean-square toàn cục gây gain chuẩn hoá cao oan).

    Args:
        audio: Tín hiệu master (float).
        sr: Tần số lấy mẫu (Hz).

    Returns:
        Loudness tích hợp (LUFS); ``-inf`` nếu rỗng/toàn im lặng.
    """
    n = len(audio)
    if n == 0:
        return float("-inf")
    from scipy.signal import lfilter, lfilter_zi

    zi1 = lfilter_zi(_KW_B1, _KW_A1) * float(audio[0])
    zi2 = None
    chunk = sr * 60
    # [v3.23.212] Ghi THẲNG vào mảng cấp sẵn float32 thay vì tích luỹ list float64 rồi
    # concatenate (trước: parts 2x + bản ghép 2x = ~9x RAM audio -> phim 2h ngốn hàng
    # GB). Lọc vẫn chạy float64 trong từng chunk (giữ độ chính xác K-weighting); chỉ
    # LƯU ở float32 — sai số LUFS < 0.01 LU, không ảnh hưởng gain chuẩn hoá.
    kweighted = np.empty(n, dtype=np.float32)
    for start in range(0, n, chunk):
        block = audio[start:start + chunk].astype(np.float64)
        s1, zi1 = lfilter(_KW_B1, _KW_A1, block, zi=zi1)
        if zi2 is None:
            zi2 = lfilter_zi(_KW_B2, _KW_A2) * float(s1[0])
        s2, zi2 = lfilter(_KW_B2, _KW_A2, s1, zi=zi2)
        kweighted[start:start + len(s2)] = s2.astype(np.float32)
    return _gated_loudness_from_kweighted(kweighted, sr)


def measure_true_peak(audio: np.ndarray, sr: int, oversample: int = 4) -> float:
    """Đo true-peak (đỉnh liên-mẫu) bằng oversample, theo chunk CÓ overlap.

    [v3.23.182] Đồng bộ fix v179: chunk có overlap ở ranh giới để không bỏ sót đỉnh
    liên-mẫu vắt qua ranh giới chunk.

    Args:
        audio: Tín hiệu cần đo.
        sr: Tần số lấy mẫu (Hz).
        oversample: Hệ số oversample (4 = chuẩn ITU-R BS.1770).

    Returns:
        Giá trị true-peak tuyến tính (>= 0.0).
    """
    return _true_peak_chunked_overlap(audio, sr, oversample=oversample)


# ── Làm rõ giọng ───────────────────────────────────────────────────────────
def _zero_phase_eq(audio: np.ndarray, sr: int) -> np.ndarray:
    if len(audio) < 150:
        return audio
    from scipy.signal import butter, sosfiltfilt
    nyq = sr * 0.5
    audio = sosfiltfilt(butter(4, min(0.99, 90.0 / nyq), btype="highpass", output="sos"), audio)
    mud = sosfiltfilt(butter(2, [130.0 / nyq, 380.0 / nyq], btype="bandpass", output="sos"), audio)
    audio = audio - mud * 0.60
    clarity = sosfiltfilt(butter(2, [2000.0 / nyq, 4000.0 / nyq], btype="bandpass", output="sos"), audio)
    audio = audio + clarity * 0.70
    air = sosfiltfilt(butter(2, min(0.99, 7000.0 / nyq), btype="highpass", output="sos"), audio)
    return (audio + air * 0.18).astype(np.float32)


def _split_band_de_esser(audio: np.ndarray, sr: int) -> np.ndarray:
    if len(audio) < 150:
        return audio
    from scipy.signal import butter, filtfilt, sosfiltfilt
    nyq = sr * 0.5
    high_band = sosfiltfilt(butter(4, min(0.99, 5500.0 / nyq), btype="highpass", output="sos"), audio)
    low_band = audio - high_band
    b_env, a_env = butter(2, 50.0 / nyq, btype="lowpass")
    high_env = filtfilt(b_env, a_env, np.abs(high_band))
    gain_reduction = np.clip(0.06 / (high_env + 1e-6), 0.25, 1.0)
    gain_smooth = filtfilt(b_env, a_env, gain_reduction)
    return (low_band + high_band * gain_smooth).astype(np.float32)


def _cinematic_air_exciter(audio: np.ndarray, sr: int) -> np.ndarray:
    if len(audio) < 150:
        return audio
    from scipy.signal import butter, sosfiltfilt
    nyq = sr * 0.5
    source = sosfiltfilt(butter(2, [3000.0 / nyq, 6000.0 / nyq], btype="bandpass", output="sos"), audio)
    harmonics = np.tanh(source * 2.0) - source
    try:
        air = sosfiltfilt(butter(2, min(0.99, 10000.0 / nyq), btype="highpass", output="sos"), harmonics)
    except ValueError:
        return audio
    return (audio + air * 0.3).astype(np.float32)


def voice_clarity(audio: np.ndarray, sr: int) -> np.ndarray:
    """Làm trong trẻo giọng đọc: EQ rõ lời → de-esser → air exciter.

    Lọc rumble <90Hz, hạ vùng đục 130-380Hz, nâng presence 2-4kHz, thêm air; sau đó
    khử xì (de-esser tách dải) và thêm hài âm "air" điện ảnh cho thoáng tiếng.
    """
    audio = np.asarray(audio, dtype=np.float32)
    # [v3.23.212] Audio DÀI (phim/tập) -> lọc theo KHỐI: RAM đỉnh giảm từ ~15.8x kích
    # thước audio xuống mức tỉ lệ với KHỐI (phim 2h: ~10.9GB -> vài trăm MB, hết treo
    # máy). Video ngắn đi đường cũ -> zero regression.
    if audio.size > _LONG_AUDIO_S * sr:
        return process_in_blocks(audio, sr, _voice_clarity_core)
    return _voice_clarity_core(audio, sr)


def _voice_clarity_core(audio: np.ndarray, sr: int) -> np.ndarray:
    """Chuỗi làm rõ giọng trên MỘT đoạn (dùng trực tiếp hoặc qua ``process_in_blocks``).

    Args:
        audio: Đoạn tín hiệu mono float32.
        sr: Tần số lấy mẫu.

    Returns:
        Đoạn đã làm rõ giọng (cùng độ dài).
    """
    audio = _zero_phase_eq(audio, sr)
    # [v3.23.182] Đồng bộ fix v181: soft-clip (tanh knee) giữa các tầng thay hard clip
    # -> EQ nâng presence có thể đẩy vượt ±1.0, soft-clip tránh méo hài chói tai.
    audio = _inter_node_soft_clip(audio)
    audio = _split_band_de_esser(audio, sr)
    audio = _cinematic_air_exciter(audio, sr)
    audio = _inter_node_soft_clip(audio)
    return audio


# ── Chuẩn hoá độ to & giới hạn đỉnh ──────────────────────────────────────────
def normalize_to_lufs(
    audio: np.ndarray, sr: int, target_lufs: float,
    max_peak: float = 0.97, measured_lufs: float | None = None,
) -> np.ndarray:
    """Chuẩn hoá độ to về ``target_lufs`` (vd -16 web, -23 EBU R128)."""
    if len(audio) == 0:
        return audio
    if measured_lufs is not None:
        current = measured_lufs
    else:
        # [v3.23.212] Audio DÀI: pyloudnorm yêu cầu bản float64 TOÀN BỘ (2x RAM + mảng
        # nội bộ) -> dùng bộ đo K-weighted gated tự viết (measure_lufs, xử lý theo
        # chunk, cùng chuẩn ITU-R BS.1770). Audio ngắn giữ pyloudnorm như cũ.
        if audio.size > _LONG_AUDIO_S * sr:
            current = measure_lufs(audio, sr)
        else:
            try:
                import pyloudnorm as pyln
                current = pyln.Meter(sr).integrated_loudness(audio.astype(np.float64))
            except ImportError:
                current = measure_lufs(audio, sr)
    if current == float("-inf"):
        return audio
    # [v3.23.182] Đồng bộ fix v178: gain có TRẦN (+15dB) chống nổi nhiễu nền ở master
    # loudness rất thấp (phim thoại thưa/nhỏ).
    gain = _loudness_gain_linear(current, target_lufs)
    out = (audio if audio.dtype == np.float32 else audio.astype(np.float32)) * np.float32(gain)
    peak = float(np.max(np.abs(out)))
    if peak > max_peak:
        out *= np.float32(max_peak / peak)
    return out


def soft_limit(audio: np.ndarray, threshold: float = 0.85, ceiling: float = 0.92) -> np.ndarray:
    """Nén mềm (tanh knee) mọi mẫu vượt ``threshold`` về sát ``ceiling``."""
    abs_a = np.abs(audio)
    over = abs_a > threshold
    if not np.any(over):
        return audio
    result = audio.copy()
    room = max(ceiling - threshold, 1e-6)
    limited = threshold + room * np.tanh((abs_a[over] - threshold) / room)
    result[over] = np.sign(audio[over]) * limited
    return result.astype(np.float32)


def true_peak_limit(
    audio: np.ndarray, sr: int, ceiling: float = 0.95, lookahead_ms: float = 1.5
) -> np.ndarray:
    """Ghìm CỤC BỘ chỉ các đỉnh liên-mẫu vượt ``ceiling`` (giữ độ to phần còn lại)."""
    n = len(audio)
    if n < 8:
        return audio
    from scipy.ndimage import maximum_filter1d, uniform_filter1d
    from scipy.signal import resample_poly

    os_factor = 4
    look = max(1, int(sr * lookahead_ms / 1000.0))
    smooth_win = max(1, int(sr * 0.003))
    chunk = sr * 30
    for start in range(0, n, chunk):
        end = min(n, start + chunk)
        seg = np.asarray(audio[start:end], dtype=np.float32)
        if seg.size < 8:
            continue
        up = resample_poly(seg, os_factor, 1)
        env_up = maximum_filter1d(np.abs(up), size=os_factor * look * 2 + 1)
        env = env_up[::os_factor][: seg.size]
        if env.size < seg.size:
            env = np.pad(env, (0, seg.size - env.size), mode="edge")
        if not np.any(env > ceiling):
            continue
        gain = np.ones(seg.size, dtype=np.float32)
        over = env > ceiling
        gain[over] = (ceiling / env[over]).astype(np.float32)
        gain = uniform_filter1d(gain, size=smooth_win)
        audio[start:end] = seg * gain
    return audio


def dc_block(audio: np.ndarray, sr: int, cutoff_hz: float = 20.0) -> np.ndarray:
    """[v3.23.184] Loại thành phần một chiều (DC) bằng high-pass bậc 2.

    DC blocker cũ trừ MEAN TOÀN CỤC (``audio - mean(audio)``) chỉ loại được DC KHÔNG
    ĐỔI; với audio dài (cả phim), TTS neural thường có DC DRIFT chậm (offset thay đổi
    theo thời gian) -> trừ một hằng số không loại được (đo thực: mỗi đoạn vẫn lệch tới
    ±0.09) -> chiếm headroom, gây méo bass/rung loa. High-pass 20Hz loại sạch DC + drift
    mà KHÔNG suy giảm giọng (giọng nói > 80Hz). Hàm THUẦN: không sửa mảng gốc.

    Args:
        audio: Tín hiệu cần loại DC (không bị sửa in-place).
        sr: Tần số lấy mẫu (Hz).
        cutoff_hz: Tần số cắt high-pass (Hz); 20Hz dưới ngưỡng nghe, an toàn cho giọng.

    Returns:
        Mảng float32 đã loại DC. Trả nguyên bản nếu quá ngắn để lọc.
    """
    n = len(audio)
    if n < 16:
        # Quá ngắn để high-pass ổn định -> trừ mean (đủ tốt cho đoạn cực ngắn).
        return (audio - np.float32(np.mean(audio))).astype(np.float32) if n else audio
    from scipy.signal import butter, sosfilt

    sos = butter(2, cutoff_hz / (sr * 0.5), btype="highpass", output="sos")
    return sosfilt(sos, audio).astype(np.float32)


def master_finalize(
    audio: np.ndarray, sr: int,
    target_lufs: float = -16.0,
    apply_clarity: bool = True,
    tp_ceiling: float = 0.95,
) -> np.ndarray:
    """Chuỗi master hoàn chỉnh: (clarity) → DC-block → LUFS → soft-limit → true-peak.

    Args:
        audio: tín hiệu mono float32.
        sr: tần số lấy mẫu.
        target_lufs: độ to mục tiêu (-16 web, -23 EBU R128, -12 to hơn cho lồng phim).
        apply_clarity: có chạy bộ làm rõ giọng không.
        tp_ceiling: trần true-peak tuyến tính (0.95 ≈ -0.45 dBTP).

    Returns:
        Master đã hoàn thiện (float32), an toàn encode lossy.
    """
    out = np.asarray(audio, dtype=np.float32)
    if out.size == 0:
        return out
    if apply_clarity:
        out = voice_clarity(out, sr)
    # [v3.23.184] DC blocker chuẩn (high-pass 20Hz) thay cho trừ mean toàn cục — loại
    # được cả DC drift chậm ở audio dài, không chiếm headroom.
    out = dc_block(out, sr)
    # Chuẩn hoá LUFS nhưng KHÔNG hạ đều theo đỉnh (max_peak rất cao) — để
    # soft-limit + true-peak-limit ghìm CỤC BỘ đỉnh, giữ đúng độ to mục tiêu.
    out = normalize_to_lufs(out, sr, target_lufs, max_peak=1e9)
    out = soft_limit(out, threshold=0.85, ceiling=0.92)
    np.clip(out, -1.0, 1.0, out=out)
    out = true_peak_limit(out, sr, ceiling=tp_ceiling)
    return out


# ── Ghi file đa định dạng ────────────────────────────────────────────────────
def _encode_with_ffmpeg(
    audio: np.ndarray, sr: int, target: Path, fmt: str, bitrate_kbps: int
) -> bool:
    """Pipe float32 vào ffmpeg để encode lossy chất lượng cao. True nếu thành công."""
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
    data = np.ascontiguousarray(clip_for_output(audio))
    # libvorbis (ogg) encode VBR theo chất lượng ổn định hơn nhiều so với bitrate cố
    # định ở mono/sample-rate thấp — ánh xạ bitrate sang mức chất lượng tương đương.
    if fmt == "ogg":
        q = 3 if bitrate_kbps <= 128 else (5 if bitrate_kbps <= 224 else 7)
        rate_args = ["-q:a", str(q)]
    else:
        rate_args = ["-b:a", f"{bitrate_kbps}k"]

    # [BrokenPipeError Fix] KHÔNG bơm vài GB float32 vào stdin (pipe:0) — Windows dễ
    # đóng pipe giữa chừng làm file ra 0 byte (mất tiếng). Chiến thuật Temp File
    # Buffer: ghi WAV float32 ra ổ cứng rồi để ffmpeg đọc từ file, xong xoá file rác.
    import os
    import tempfile

    import soundfile as sf

    fd, tmp_wav = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        sf.write(tmp_wav, data, sr, subtype="FLOAT")
        cmd = [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-i", tmp_wav,
            "-c:a", codec, *rate_args, *extra, str(target),
        ]
        # [v3.23.25] Timeout chống treo: encode audio dài (FLAC/AAC) hiếm khi >10 phút;
        # nếu ffmpeg kẹt (đĩa đầy, codec lỗi) không có timeout sẽ treo cả tiến trình TTS.
        try:
            proc = subprocess.run(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=False,
                timeout=600,
                **no_window_kwargs(),
            )
        except subprocess.TimeoutExpired:
            logger.warning("ffmpeg encode %s quá 10 phút — huỷ (đĩa/codec có thể lỗi).", fmt)
            return False
        if proc.returncode == 0 and target.exists() and target.stat().st_size > 0:
            return True
        logger.warning("ffmpeg encode %s lỗi: %s", fmt,
                       proc.stderr.decode("utf-8", "ignore")[:200])
        return False
    except (OSError, ValueError) as exc:
        logger.warning("Gọi ffmpeg thất bại: %s", exc)
        return False
    finally:
        try:
            os.remove(tmp_wav)
        except OSError:
            pass


def write_audio(
    audio: np.ndarray, sr: int, output_path: Path,
    fmt: str = "wav", subtype: str = "PCM_16", bitrate_kbps: int = 320,
) -> Path:
    """Ghi audio ra đúng định dạng yêu cầu; trả về đường dẫn thực đã ghi.

    WAV/FLAC ghi trực tiếp bằng soundfile (lossless); MP3/OPUS/OGG/M4A encode bằng
    ffmpeg từ float32. Thiếu ffmpeg sẽ lùi an toàn về WAV để không mất kết quả.
    """
    import soundfile as sf

    fmt = (fmt or "wav").lower()
    bitrate_kbps = max(32, int(bitrate_kbps))
    target = output_path.with_suffix(f".{fmt}")

    if fmt in ("wav", "flac"):
        # FLAC không hỗ trợ Float 32-bit → tự hạ về PCM_24 tránh crash soundfile.
        if fmt == "flac" and subtype == "FLOAT":
            logger.info("FLAC không hỗ trợ Float 32-bit → dùng PCM_24 thay thế.")
            subtype = "PCM_24"
        # [v3.23.183] Chặn biên trước khi ghi: nhất quán với path ffmpeg, tránh mẫu
        # vượt ±1.0 gây méo (đặc biệt subtype FLOAT không tự clip như PCM).
        sf.write(str(target), clip_for_output(audio), sr, subtype=subtype)
        return target

    if _encode_with_ffmpeg(audio, sr, target, fmt, bitrate_kbps):
        return target

    logger.warning("Không encode được %s (thiếu ffmpeg?). Ghi WAV thay thế.", fmt)
    wav_path = output_path.with_suffix(".wav")
    sf.write(str(wav_path), clip_for_output(audio), sr, subtype=subtype)
    return wav_path
