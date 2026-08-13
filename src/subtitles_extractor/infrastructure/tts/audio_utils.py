"""Tiện ích xử lý tín hiệu audio thô cho TTS — DÙNG CHUNG cho mọi engine (hàm THUẦN).

[v3.23.220] Tách khỏi ``vieneu_tts_adapter``: ba hàm dưới đây vốn là hạ tầng chung nhưng
lại được định nghĩa bên trong một ADAPTER cụ thể, khiến Edge phải lazy-import ngược (né
vòng tròn) và Gemini phải import từ VieNeu. Nay đứng độc lập: chỉ phụ thuộc numpy/scipy.

* :func:`is_effectively_silent` — bắt lỗi engine sinh audio "có độ dài nhưng câm".
* :func:`trim_edge_silence` — cắt khoảng lặng đầu/cuối TRƯỚC khi tính stretch.
* :func:`resample_audio` — đưa mọi engine về một tần số lấy mẫu chung của pipeline.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from math import gcd

import numpy as np

logger = logging.getLogger(__name__)


@contextmanager
def _torch_none_bypassed() -> Iterator[None]:
    """Tạm gỡ các entry ``sys.modules[...] = None`` của torch trong khối ``with``.

    [v3.23.257] ``torch_import_blocker`` đặt ``sys.modules["torch"]=None`` ép engine
    (VieNeu) chạy ONNX thay vì torch. Nhưng scipy >=1.15 (array-API compat) dò kiểu tensor
    bằng ``getattr(sys.modules["torch"], "Tensor")`` — với torch=None thành
    ``getattr(None, "Tensor")`` -> ``AttributeError``. Hàm này tạm XOÁ entry None
    (chỉ entry đang là None, KHÔNG đụng torch thật nếu đã nạp) để scipy thấy torch vắng
    (KeyError -> xử lý an toàn), rồi KHÔI PHỤC nguyên trạng khi ra khối — giữ blocker hoạt
    động cho phần còn lại của pipeline.

    Yields:
        None. Trong khối, các entry torch=None tạm biến mất khỏi ``sys.modules``.
    """
    blocked_roots = ("torch", "torchvision", "torchaudio")
    removed: list[str] = []
    for root in blocked_roots:
        if sys.modules.get(root, "absent") is None:
            del sys.modules[root]
            removed.append(root)
    try:
        yield
    finally:
        # Khôi phục đúng trạng thái None để blocker tiếp tục chặn import torch thật.
        for root in removed:
            sys.modules[root] = None  # type: ignore[assignment]

__all__ = [
    "RESAMPLE_PATIENCE",
    "cap_nhat_ban_tot_nhat",
    "is_effectively_silent",
    "resample_audio",
    "shorter_take",
    "trim_edge_silence",
]


def shorter_take(
    current_best: np.ndarray | None, candidate: np.ndarray
) -> np.ndarray:
    """[v3.23.221] Chọn bản thu NGẮN HƠN giữa hai lần tổng hợp (hàm thuần).

    Dùng cho lưới "audio dài bất thường": khác với audio CÂM (vứt đi được vì vô giá trị),
    audio quá dài VẪN là audio hợp lệ — chỉ là model ngân lê thê. Vì thế không bao giờ
    được trả ``None`` khi hết lượt thử (sẽ MẤT THOẠI); thay vào đó giữ ứng viên tốt nhất
    qua các lần thử và trả bản ngắn nhất.

    Args:
        current_best: Ứng viên tốt nhất tới hiện tại (None nếu chưa có).
        candidate: Bản thu vừa sinh.

    Returns:
        Bản ngắn hơn trong hai bản (ưu tiên ``current_best`` khi bằng nhau).
    """
    if current_best is None:
        return candidate
    return candidate if candidate.size < current_best.size else current_best


def is_effectively_silent(audio: np.ndarray, rms_threshold: float = 0.005) -> bool:
    """[v3.23.205] Audio "coi như im lặng" — model sinh nhưng không có tiếng (hàm thuần).

    Phát hiện từ đối chiếu FLAC thực: VieNeu thi thoảng sinh audio CÓ ĐỘ DÀI nhưng toàn
    im lặng (câu #58 "Ý chú là": 2.2s, RMS ~0.003 — mất thoại mà vẫn báo OK). Dạng lỗi
    này lọt qua lưới ``size > 0`` (chỉ bắt audio RỖNG như câu "Ưm"). Ngưỡng 0.005 nằm
    giữa nhiễu sàn đo được (~0.003) và giọng nhỏ nhất thực tế (~0.04).

    Args:
        audio: Tín hiệu mono float32.
        rms_threshold: Ngưỡng RMS toàn đoạn coi là im lặng.

    Returns:
        True nếu audio không có tiếng nghe được (nên retry như audio rỗng).
    """
    if audio.size == 0:
        return True
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    return rms < rms_threshold


def trim_edge_silence(
    audio: np.ndarray,
    sr: int,
    threshold: float = 0.008,
    keep_head_s: float = 0.05,
    keep_tail_s: float = 0.08,
    adaptive: bool = False,
) -> np.ndarray:
    """[v3.23.204] Cắt khoảng LẶNG đầu/cuối audio TTS vừa sinh (hàm thuần).

    Đo thực trên 95 câu VieNeu: model sinh im lặng ĐẦU câu rất dài (median 160ms, 28
    câu >300ms, max 1420ms). Hệ quả kép: (1) tiếng vang TRỄ so với mốc phụ đề/khẩu
    hình; (2) im lặng chiếm khung đẩy phần tiếng dồn về cuối -> TRÀN/chồng lên câu sau
    (14.8s im lặng lãng phí trong một video 155s); (3) stretch nén mạnh hơn mức cần vì
    tính cả im lặng. Trim biên (giữ đệm nhỏ cho tự nhiên) giải quyết cả ba.

    Args:
        audio: Tín hiệu mono float32 vừa tổng hợp.
        sr: Tần số lấy mẫu.
        threshold: Ngưỡng RMS cửa sổ 20ms coi là "có tiếng" (trên nhiễu nền ~0.001,
            dưới giọng nhỏ nhất đo được ~0.04). Chỉ dùng khi ``adaptive=False``.
        keep_head_s: Đệm giữ lại trước tiếng (vào nhịp tự nhiên).
        keep_tail_s: Đệm giữ lại sau tiếng (đuôi âm không bị hụt).
        adaptive: [v3.23.241] Nếu True, ngưỡng TỰ DÒ theo sàn nhiễu của chính câu đó
            (giống VAD adaptive của Edge) thay vì hằng số tuyệt đối. Bền hơn với câu
            model sinh biên độ thấp — ngưỡng cố định 0.008 bỏ sót câu có RMS < 0.008,
            còn adaptive vẫn cắt được. Chặn trên theo đỉnh để không lẹm phụ âm/âm gió.

    Returns:
        Audio đã cắt biên; trả NGUYÊN BẢN nếu toàn im lặng (để retry/skip xử lý).
    """
    if audio.size == 0:
        return audio
    window = max(1, int(0.02 * sr))
    usable = (len(audio) // window) * window
    if usable == 0:
        return audio
    frame_rms = np.sqrt((audio[:usable].reshape(-1, window) ** 2).mean(axis=1))
    thr = _adaptive_voice_threshold(frame_rms) if adaptive else threshold
    voiced = np.nonzero(frame_rms > thr)[0]
    if voiced.size == 0:
        return audio  # toàn im lặng — giữ nguyên cho lớp retry/skip quyết định
    start = max(0, voiced[0] * window - int(keep_head_s * sr))
    end = min(len(audio), (voiced[-1] + 1) * window + int(keep_tail_s * sr))
    return audio[start:end]


def _adaptive_voice_threshold(frame_rms: np.ndarray) -> float:
    """[v3.23.241] Ngưỡng "có tiếng" tự dò theo sàn nhiễu của chính câu (hàm thuần).

    Cùng nguyên lý VAD adaptive percentile mà Edge dùng (``_trim_silence``), nay đưa về
    hàm dùng chung cho VieNeu/Gemini cùng hưởng. Ngưỡng cố định tuyệt đối bỏ sót câu
    model sinh biên độ thấp (RMS < ngưỡng -> không cắt được im lặng đầu -> tiếng trễ).

    Ngưỡng = bội số nhỏ của sàn nhiễu (p5 của RMS), nhưng CHẶN TRÊN theo đỉnh (15%) để
    câu động học lớn không bị cắt nhầm phụ âm cuối nhẹ ('s', 'ch', 'th').

    Args:
        frame_rms: RMS theo cửa sổ của tín hiệu.

    Returns:
        Ngưỡng RMS tuyệt đối cho câu này.
    """
    if frame_rms.size == 0:
        return 0.008
    noise_floor = float(np.percentile(frame_rms, 5))
    peak = float(np.max(frame_rms))
    return min(noise_floor * 4.0, peak * 0.15)


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample tín hiệu về ``target_sr`` (hàm thuần, ưu tiên chất lượng).

    Dùng ``resample_poly`` (poly-phase, chống aliasing) khi có scipy; nếu không có thì
    lùi về nội suy tuyến tính. Là hàm THUẦN: chỉ phụ thuộc đầu vào, không side-effect.

    Args:
        audio: Tín hiệu nguồn (mono, float).
        orig_sr: Tần số lấy mẫu gốc (Hz).
        target_sr: Tần số lấy mẫu đích (Hz).

    Returns:
        Tín hiệu float32 ở ``target_sr``. Trả nguyên bản nếu hai tần số bằng nhau.
    """
    if orig_sr == target_sr or audio.size == 0:
        return audio.astype(np.float32)
    try:
        # [v3.23.257] Bọc CẢ lần import scipy: scipy >=1.15 khởi tạo array-API
        # compat ngay lúc IMPORT (scipy.stats dựng docstring ví dụ -> is_torch_array
        # -> getattr(torch, "Tensor")). Khi blocker đặt torch=None, lần import scipy
        # ĐẦU TIÊN trong khối torch_isolation crash NGAY tại import, không phải lúc
        # gọi resample_poly. Vậy phải gỡ torch=None quanh cả import lẫn gọi.
        with _torch_none_bypassed():
            from scipy.signal import resample_poly

            divisor = gcd(orig_sr, target_sr)
            return resample_poly(
                audio, target_sr // divisor, orig_sr // divisor
            ).astype(np.float32)
    except ImportError:
        ratio = target_sr / orig_sr
        new_length = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, new_length)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


# [v3.23.235] Số lần lấy mẫu lại LIÊN TIẾP không cho ra bản ngắn hơn thì dừng.
#
# Đo trên log Gemini thật (v234): nhiều câu retry đủ 10/10 lần mà bản ngắn nhất đã tìm
# được từ lần 2-3, ví dụ::
#
#     "gần như"       0.84s ở lần 3, rồi 7 lần sau đều y hệt 0.84s
#     "Chỉ cần cháu"  1.24s ở lần 2, rồi 8 lần sau không lần nào ngắn hơn
#
# Nguyên nhân: khung phụ đề của những câu này NHỎ HƠN cả nhịp đọc chuẩn của engine (câu
# "biến thành" chỉ có khung 0.20s, trong khi Gemini đọc 10 ký tự mất tối thiểu ~1.34s ->
# nén 2.0x vẫn còn 0.67s). Không bản nào vừa được, nên vòng retry chạy hết 10 lượt rồi mới
# chịu thua — tốn 10 lượt gọi API và kéo phiên TTS từ 3.5 phút lên 7.5 phút.
#
# Vì sao KHÔNG cắt cứng ở 3 lượt: ca "biến thành" có lần đầu bị hallucination 8.08s, mãi
# tới lượt thứ 10 mới ra bản 2.64s (ngắn hơn 5.44s). Đếm LIÊN TIẾP giữ được ca đó, trong
# khi vẫn cắt sạch các ca lặp lại vô ích.
#
# Mô phỏng trên 10 ca thật: patience=3 tiết kiệm **46% lượt gọi API**, độ dài audio thu
# được hầu như không đổi (median +0.00s).
RESAMPLE_PATIENCE = 3


def cap_nhat_ban_tot_nhat(
    best_duration_s: float,
    new_duration_s: float,
    no_improve_streak: int,
    patience: int = RESAMPLE_PATIENCE,
) -> tuple[float, int, bool]:
    """Cập nhật bản ngắn nhất và quyết định có nên DỪNG lấy mẫu lại không (hàm thuần).

    Args:
        best_duration_s: Độ dài bản ngắn nhất tìm được tới giờ (``inf`` nếu chưa có).
        new_duration_s: Độ dài bản vừa sinh.
        no_improve_streak: Số lượt LIÊN TIẾP trước đó không cải thiện.
        patience: Ngưỡng kiên nhẫn.

    Returns:
        Bộ ba ``(best_moi, streak_moi, nen_dung)``.
    """
    if new_duration_s < best_duration_s - 1e-3:
        return new_duration_s, 0, False
    streak = no_improve_streak + 1
    return best_duration_s, streak, streak >= patience
