"""Nguyên hàm DSP thuần — DÙNG CHUNG cho mọi engine TTS và bộ master.

[v3.23.220] Trước đây các nguyên hàm này nằm trong ``edge_tts_adapter`` (một ADAPTER cụ
thể), khiến module hạ tầng dùng chung ``audio_mastering`` phải ``import`` ngược từ
adapter — phụ thuộc đảo chiều, sinh vòng tròn import (``edge -> audio_mastering ->
edge``) chỉ né được bằng lazy import, và mọi thay đổi ở Edge trở thành rủi ro cho hai
engine còn lại. Gom về đây: mọi hàm là **thuần** (không side-effect, không phụ thuộc
adapter/Qt/SDK), chỉ dựa vào ``numpy``/``scipy``.

Quan hệ phụ thuộc sau khi tách (một chiều, không vòng)::

    edge/vieneu/gemini_tts_adapter
        -> audio_mastering -> dsp_primitives
        -> time_stretch / timing_math / audio_utils / text_prep

Gồm bốn nhóm:

* Đo lường: :func:`true_peak_chunked_overlap`, :func:`gated_loudness_from_kweighted`.
* Tính gain: :func:`loudness_gain_linear`, :func:`clamp_smoothed_gain`,
  :func:`noise_gate_threshold_linear`.
* Giới hạn biên độ: :func:`inter_node_soft_clip`.
* Khớp độ dài: :func:`fit_length_no_silence`.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "clamp_smoothed_gain",
    "fit_length_no_silence",
    "gated_loudness_from_kweighted",
    "inter_node_soft_clip",
    "loudness_gain_linear",
    "noise_gate_threshold_linear",
    "true_peak_chunked_overlap",
]


def fit_length_no_silence(stretched: np.ndarray, target_len: int) -> np.ndarray:
    """[v3.23.174] Khớp mảng audio về đúng ``target_len`` mà KHÔNG chèn im lặng cuối.

    Các engine time-stretch (OLA/WSOLA) sinh độ dài lệch nhẹ so với đích do phép chia
    nguyên khi tính số khung. Trước đây phần THIẾU được ``np.pad`` bằng zeros -> chèn
    im lặng vào ĐUÔI giọng (đo thực: tới 17ms ở nén nhẹ) -> câu nghe "hụt hơi" cuối.
    Nay: nếu dài hơn đích thì cắt; nếu NGẮN hơn thì NỘI SUY TUYẾN TÍNH giãn nhẹ phần
    giọng về đúng đích -> giữ âm liên tục, không có khoảng lặng nhân tạo.

    Args:
        stretched: Mảng audio sau stretch (có thể lệch độ dài so với đích).
        target_len: Số mẫu mong muốn (> 0).

    Returns:
        Mảng float32 có đúng ``target_len`` mẫu.
    """
    current_len = len(stretched)
    if current_len == target_len or target_len <= 0:
        return stretched.astype(np.float32)
    if current_len > target_len:
        return stretched[:target_len].astype(np.float32)
    if current_len < 2:
        # Không đủ mẫu để nội suy -> lặp mẫu cuối (vẫn hơn chèn zeros im lặng).
        pad_value = stretched[-1] if current_len == 1 else np.float32(0.0)
        return np.full(target_len, pad_value, dtype=np.float32)
    return np.interp(
        np.linspace(0, current_len - 1, target_len),
        np.arange(current_len),
        stretched,
    ).astype(np.float32)


def true_peak_chunked_overlap(
    master: np.ndarray,
    sample_rate: int,
    oversample: int = 4,
    chunk_seconds: int = 30,
    overlap_samples: int = 64,
) -> float:
    """[v3.23.179] Đo true-peak (đỉnh liên-mẫu) theo chunk CÓ OVERLAP ở ranh giới.

    Đo true-peak bằng cách chia master thành chunk rồi ``resample_poly`` từng chunk giúp
    tiết kiệm RAM cho master rất dài. Nhưng nếu các chunk RỜI NHAU, đỉnh liên-mẫu VẮT
    QUA ranh giới chunk bị bỏ sót (mỗi chunk không thấy mẫu lân cận ở chunk kế) -> đo
    THẤP hơn thực (đo thực: bỏ sót 0.148, báo 0.90 trong khi đỉnh thật 1.05). Khắc phục:
    mỗi chunk lấy THÊM ``overlap_samples`` mẫu ở hai biên để phép nội suy tái tạo đúng
    đỉnh tại ranh giới. Hàm THUẦN (không side-effect) để kiểm thử độc lập.

    Args:
        master: Tín hiệu cần đo (không bị sửa đổi).
        sample_rate: Tần số lấy mẫu (Hz).
        oversample: Hệ số oversample cho nội suy (4 = chuẩn ITU-R BS.1770).
        chunk_seconds: Độ dài mỗi chunk xử lý (giây) để giới hạn RAM.
        overlap_samples: Số mẫu chồng lấn ở mỗi biên chunk (>= độ dài đuôi filter).

    Returns:
        Giá trị true-peak tuyến tính (>= 0.0).
    """
    from scipy.signal import resample_poly

    n = len(master)
    if n == 0:
        return 0.0
    if n < 8:
        return float(np.max(np.abs(master)))

    chunk = max(1, sample_rate * chunk_seconds)
    true_peak = 0.0
    for start in range(0, n, chunk):
        # Mở rộng biên trái/phải để đỉnh liên-mẫu ở ranh giới được tái tạo đúng.
        read_start = max(0, start - overlap_samples)
        read_end = min(n, start + chunk + overlap_samples)
        seg = np.asarray(master[read_start:read_end], dtype=np.float64)
        if seg.size < 8:
            true_peak = max(true_peak, float(np.max(np.abs(seg))))
            continue
        upsampled = resample_poly(seg, oversample, 1)
        true_peak = max(true_peak, float(np.max(np.abs(upsampled))))
    return true_peak


def loudness_gain_linear(
    measured_lufs: float,
    target_lufs: float,
    max_gain_db: float = 15.0,
) -> float:
    """[v3.23.178] Tính hệ số gain tuyến tính đưa loudness về target, CÓ GIỚI HẠN.

    Gain = ``10^((target - measured)/20)`` nhưng CHẶN TRÊN theo ``max_gain_db``. Nếu
    không chặn, master có loudness rất thấp (phim thoại thưa/nhỏ, vd -40 LUFS) sẽ bị
    khuếch đại tới +26dB -> NỔI nhiễu nền Edge (hơi thở, nền số) + đẩy vô số đỉnh vượt
    trần khiến true-peak limiter phải ghìm liên tục gây méo/pumping. Chuẩn phát thanh
    giới hạn gain chuẩn hoá (thường +12..+15dB) để bảo toàn tỉ lệ tín hiệu/nhiễu.

    Args:
        measured_lufs: Loudness đo được của master (LUFS); ``-inf`` nếu im lặng.
        target_lufs: Loudness mục tiêu (LUFS, thường -14).
        max_gain_db: Trần khuếch đại (dB) để không nổi nhiễu nền.

    Returns:
        Hệ số gain tuyến tính (>= 0.0). Trả 1.0 nếu ``measured_lufs`` là ``-inf``.
    """
    if measured_lufs == float("-inf"):
        return 1.0
    gain_db = target_lufs - measured_lufs
    if gain_db > max_gain_db:
        gain_db = max_gain_db
    return float(10.0 ** (gain_db / 20.0))


def gated_loudness_from_kweighted(
    kweighted: np.ndarray,
    sample_rate: int,
    block_ms: float = 400.0,
    absolute_gate_lufs: float = -70.0,
    relative_gate_lu: float = -10.0,
) -> float:
    """[v3.23.177] Tính loudness tích hợp (LUFS) có GATING theo chuẩn EBU R128.

    Tín hiệu ``kweighted`` (đã qua K-weighting) được chia thành block ``block_ms``, mỗi
    block tính mean-square. Áp hai cổng loại block im lặng/nền để KHÔNG kéo loudness
    xuống sai:

    1. Absolute gate: loại block có loudness < ``absolute_gate_lufs`` (-70 LUFS).
    2. Relative gate: từ các block còn lại, tính loudness sơ bộ rồi loại thêm block
       dưới (loudness_sơ_bộ + ``relative_gate_lu``) (-10 LU).

    Loudness cuối = trung bình mean-square các block qua cả hai cổng.

    Đây là điểm KHÁC BIỆT cốt lõi so với đo mean-square TOÀN CỤC: master lồng tiếng có
    nhiều khoảng lặng giữa câu; nếu gộp cả im lặng, mean-square bị pha loãng -> LUFS đo
    THẤP hơn thực -> gain chuẩn hoá bị đẩy cao oan -> giọng to quá mức (đo thực: 4s im
    lặng kéo -17.6 xuống -24.6 LUFS, lệch 7 LU).

    Args:
        kweighted: Tín hiệu đã qua bộ lọc K-weighting (float).
        sample_rate: Tần số lấy mẫu (Hz).
        block_ms: Độ dài block phân tích (ms), chuẩn R128 là 400ms.
        absolute_gate_lufs: Ngưỡng cổng tuyệt đối (LUFS).
        relative_gate_lu: Ngưỡng cổng tương đối so loudness sơ bộ (LU).

    Returns:
        Loudness tích hợp (LUFS); ``-inf`` nếu không có block nào qua cổng.
    """
    n = len(kweighted)
    if n == 0:
        return float("-inf")
    block_size = max(1, int(sample_rate * block_ms / 1000.0))
    n_blocks = n // block_size
    if n_blocks == 0:
        # Tín hiệu ngắn hơn một block: đo trực tiếp toàn bộ (không đủ để gating).
        mean_square = float(np.mean(kweighted.astype(np.float64) ** 2))
        return (
            -0.691 + 10.0 * np.log10(mean_square)
            if mean_square > 1e-12
            else float("-inf")
        )

    # [v3.23.213] Tính công suất từng block theo LÔ NHỎ: bản cũ ép TOÀN BỘ chuỗi
    # K-weighted sang float64 rồi bình phương (~4x RAM audio, cộng dồn thành ~12x cho cả
    # bộ đo) -> phim dài ngốn hàng GB. Lô nhỏ giữ RAM tạm ở mức vài chục MB; kết quả
    # BIT-IDENTICAL (cùng phép toán, chỉ khác thứ tự gom).
    block_ms_powers = np.empty(n_blocks, dtype=np.float64)
    batch = 200  # ~200 x 400ms = 80s dữ liệu mỗi lô
    for i in range(0, n_blocks, batch):
        j = min(i + batch, n_blocks)
        segment = kweighted[i * block_size : j * block_size].astype(np.float64)
        block_ms_powers[i:j] = np.mean(segment.reshape(j - i, block_size) ** 2, axis=1)
    block_loudness = np.full(n_blocks, -np.inf)
    valid = block_ms_powers > 1e-12
    block_loudness[valid] = -0.691 + 10.0 * np.log10(block_ms_powers[valid])

    absolute_pass = block_loudness > absolute_gate_lufs
    if not np.any(absolute_pass):
        return float("-inf")

    provisional_power = float(np.mean(block_ms_powers[absolute_pass]))
    if provisional_power <= 1e-12:
        return float("-inf")
    provisional_lufs = -0.691 + 10.0 * np.log10(provisional_power)
    relative_threshold = provisional_lufs + relative_gate_lu

    final_pass = absolute_pass & (block_loudness > relative_threshold)
    if not np.any(final_pass):
        final_pass = absolute_pass  # giữ lại kết quả absolute nếu relative loại hết

    gated_power = float(np.mean(block_ms_powers[final_pass]))
    if gated_power <= 1e-12:
        return float("-inf")
    return -0.691 + 10.0 * np.log10(gated_power)


def inter_node_soft_clip(audio: np.ndarray, threshold: float = 0.90) -> np.ndarray:
    """[v3.23.181] Soft-clip (tanh knee) giữa các tầng DSP thay cho hard clip.

    Giữa các tầng voice_clarity (EQ nâng clarity +70%, exciter thêm hài), biên độ có
    thể vọt qua ±1.0 do cộng dồn pha. Trước đây dùng ``np.clip`` (hard clip) cắt PHẲNG
    đỉnh -> sinh hài bậc cao chói tai (harsh distortion) ngay tại đỉnh giọng (đo thực:
    EQ đẩy 0.9 -> 1.048, hard clip cắt cứng gây méo). Nay dùng knee mềm tanh: mẫu dưới
    ``threshold`` giữ NGUYÊN (không đụng phần lớn tín hiệu), chỉ nén mềm phần vượt ->
    đỉnh ra luôn < 1.0 mà không sinh méo cứng.

    Args:
        audio: Tín hiệu cần giới hạn (không bị sửa in-place).
        threshold: Ngưỡng bắt đầu nén mềm (mẫu dưới mức này giữ nguyên).

    Returns:
        Mảng float32 với đỉnh < 1.0, phần dưới ngưỡng không đổi.
    """
    abs_audio = np.abs(audio)
    over = abs_audio > threshold
    if not np.any(over):
        return audio.astype(np.float32)
    result = audio.astype(np.float32).copy()
    room = 1.0 - threshold
    excess = abs_audio[over] - threshold
    limited = threshold + room * np.tanh(excess / room)
    result[over] = np.sign(audio[over]) * limited
    return result


def noise_gate_threshold_linear(
    envelope_peak: float,
    absolute_threshold_db: float = -42.0,
    relative_floor_db: float = -32.0,
) -> float:
    """[v3.23.180] Ngưỡng noise gate tuyến tính, THÍCH ỨNG theo đỉnh câu.

    Noise gate cũ dùng ngưỡng TUYỆT ĐỐI (-42dBFS): vùng dưới mức đó bị hạ gain. Với câu
    giọng NHỎ hợp lệ (thì thầm, giọng yếu — đỉnh envelope ~0.008), toàn bộ câu nằm dưới
    ngưỡng tuyệt đối -> bị hạ oan 36%% -> càng nhỏ đi, mất lời. Khắc phục: ngưỡng thực
    = MIN giữa ngưỡng tuyệt đối và mức tương đối (-32dB SO đỉnh câu). Nhờ đó câu nhỏ
    được đo theo chính đỉnh của nó (chỉ hạ phần thực sự im lặng giữa từ), câu to vẫn
    dùng ngưỡng tuyệt đối để khử nhiễu nền hiệu quả.

    Args:
        envelope_peak: Đỉnh đường bao (envelope) của câu.
        absolute_threshold_db: Ngưỡng tuyệt đối (dBFS, âm).
        relative_floor_db: Ngưỡng tương đối so đỉnh câu (dB, âm).

    Returns:
        Ngưỡng biên độ tuyến tính (> 0) để so với envelope trong noise gate.
    """
    absolute_linear = 10.0 ** (absolute_threshold_db / 20.0)
    if envelope_peak <= 0.0:
        return absolute_linear
    relative_linear = envelope_peak * (10.0 ** (relative_floor_db / 20.0))
    return min(absolute_linear, relative_linear)


def clamp_smoothed_gain(
    smoothed_gain: np.ndarray,
    raw_gain: np.ndarray,
) -> np.ndarray:
    """[v3.23.176] Kẹp gain đã làm mượt để KHÔNG vượt gain thô cần thiết tại mỗi mẫu.

    Làm mượt đường gain (trung bình trượt) tránh "pumping", nhưng có tác dụng phụ: tại
    một đỉnh transient đơn lẻ cần gain THẤP, các mẫu lân cận có gain ~1.0 sẽ kéo gain
    sau làm mượt LÊN CAO hơn mức cần -> đỉnh không được ghìm đủ -> lọt méo/clip (đo
    thực: đỉnh 2.5 chỉ giảm còn 1.747, vẫn > 1.0). Khắc phục: gain cuối = min(mượt,
    thô) theo từng mẫu -> làm mượt chỉ được phép GIẢM gain thêm (an toàn hơn), không
    bao giờ nới lỏng mức ghìm tại đỉnh.

    Args:
        smoothed_gain: Đường gain sau khi làm mượt (uniform_filter1d).
        raw_gain: Đường gain thô trước làm mượt (đã đảm bảo ghìm đỉnh ≤ ceiling).

    Returns:
        Đường gain đã kẹp, cùng độ dài, luôn ``<= raw_gain`` từng mẫu.
    """
    return np.minimum(smoothed_gain, raw_gain).astype(np.float32)
