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
    "EDGE_ISLAND_CORE_DB",
    "EDGE_ISLAND_MAX_S",
    "EDGE_ISLAND_MIN_GAP_S",
    "RESAMPLE_PATIENCE",
    "cap_nhat_ban_tot_nhat",
    "is_effectively_silent",
    "resample_audio",
    "shorter_take",
    "speech_seconds",
    "speech_span",
    "trim_edge_silence",
]

#: [v3.23.615] Đảo tiếng ở BIÊN audio ngắn hơn mức này (giây) mà cách phần lời ít nhất
#: :data:`EDGE_ISLAND_MIN_GAP_S` là tạp âm model sinh thêm, không phải lời. Âm tiết ngắn
#: nhất ("à", "ừ") dài trên 80ms; tiếng bật của phụ âm tắc nằm sát nguyên âm (dưới 80ms).
EDGE_ISLAND_MAX_S = 0.06

#: [v3.23.615] Khoảng lặng tối thiểu (giây) tách đảo tạp âm khỏi phần lời.
EDGE_ISLAND_MIN_GAP_S = 0.15

#: [v3.23.616] Độ dài một đảo đo trên LÕI của nó: các khung to trong vòng chừng này dB
#: dưới đỉnh của CHÍNH đảo. Đếm mọi khung trên ngưỡng "có tiếng" (rất thấp, sát sàn nhiễu)
#: thì đuôi hiện/tắt dần và hai mép cửa sổ 20ms bị tính cả: tiếng "tách" 52ms thành 80ms.
EDGE_ISLAND_CORE_DB = -20.0


def speech_span(
    voiced: np.ndarray,
    frame_s: float,
    max_island_s: float = EDGE_ISLAND_MAX_S,
    min_gap_s: float = EDGE_ISLAND_MIN_GAP_S,
    levels: np.ndarray | None = None,
) -> tuple[int, int] | None:
    """[v3.23.615] Biên ``[đầu, cuối]`` của phần LỜI, bỏ đảo tạp âm lẻ ở hai đầu.

    Hàm thuần.

    Đo trên FLAC lồng tiếng thật 14/10 (VieNeu, 95 câu): câu "Hôm nay muội đến đây" nói
    xong ở 0.78s, im 305ms rồi có một tiếng "tách" 30ms (-24 dB) đúng ở cuối; câu "Hả?!"
    cũng vậy (230ms lặng rồi 20ms tạp âm). Cắt lặng theo khung CUỐI có tiếng giữ nguyên cả
    khoảng lặng lẫn tiếng tách: người nghe nghe một tiếng "click" sau lời thoại, và khung
    lời bị tính dài hơn thực tế (khoảng lặng bị nén theo).

    Args:
        voiced: Cờ "có tiếng" theo từng khung.
        frame_s: Độ dài một khung (giây) — để đổi ngưỡng giây sang số khung.
        max_island_s: Đảo biên dài tối đa chừng này mới bị coi là tạp âm.
        min_gap_s: Đảo phải cách phần lời ít nhất chừng này.
        levels: [v3.23.616] Mức (RMS) từng khung, cùng độ dài ``voiced``. Có thì độ dài
            đảo chỉ tính các khung to trong vòng :data:`EDGE_ISLAND_CORE_DB` dưới đỉnh
            của chính đảo; không có thì đếm mọi khung "có tiếng" như cũ.

    Returns:
        ``(khung đầu, khung cuối)`` (bao gồm) của phần lời; ``None`` nếu không có khung
        nào có tiếng. Không bao giờ bỏ đảo cuối cùng còn lại.
    """
    indices = np.flatnonzero(voiced)
    if indices.size == 0:
        return None
    breaks = np.flatnonzero(np.diff(indices) > 1)
    starts = np.concatenate(([indices[0]], indices[breaks + 1]))
    ends = np.concatenate((indices[breaks], [indices[-1]]))
    if frame_s <= 0:
        return int(starts[0]), int(ends[-1])
    max_island = max(1, round(max_island_s / frame_s))
    min_gap = max(1, round(min_gap_s / frame_s))
    core_ratio = 10.0 ** (EDGE_ISLAND_CORE_DB / 20.0)

    def island_frames(position: int) -> int:
        if levels is None:
            return int(ends[position] - starts[position] + 1)
        island_levels = levels[starts[position]: ends[position] + 1]
        return int(np.count_nonzero(island_levels >= island_levels.max() * core_ratio))

    first, last = 0, len(starts) - 1
    while (
        first < last
        and island_frames(first) <= max_island
        and starts[first + 1] - ends[first] - 1 >= min_gap
    ):
        first += 1
    while (
        last > first
        and island_frames(last) <= max_island
        and starts[last] - ends[last - 1] - 1 >= min_gap
    ):
        last -= 1
    return int(starts[first]), int(ends[last])


def shorter_take(
    current_best: np.ndarray | None, candidate: np.ndarray, sr: int | None = None
) -> np.ndarray:
    """[v3.23.221] Chọn bản thu NGẮN HƠN giữa hai lần tổng hợp (hàm thuần).

    Dùng cho lưới "audio dài bất thường": khác với audio CÂM (vứt đi được vì vô giá trị),
    audio quá dài VẪN là audio hợp lệ — chỉ là model ngân lê thê. Vì thế không bao giờ
    được trả ``None`` khi hết lượt thử (sẽ MẤT THOẠI); thay vào đó giữ ứng viên tốt nhất
    qua các lần thử và trả bản ngắn nhất.

    Args:
        current_best: Ứng viên tốt nhất tới hiện tại (None nếu chưa có).
        candidate: Bản thu vừa sinh.
        sr: [v3.23.601] Có tần số lấy mẫu thì so độ dài PHẦN TIẾNG
            (:func:`speech_seconds`) thay vì độ dài thô — khâu đặt câu cắt lặng biên nên
            im lặng không làm bản thu "dài" hơn. None giữ cách so độ dài thô như cũ.

    Returns:
        Bản ngắn hơn trong hai bản (ưu tiên ``current_best`` khi bằng nhau).
    """
    if current_best is None:
        return candidate
    if sr:
        shorter = speech_seconds(candidate, sr) < speech_seconds(current_best, sr)
        return candidate if shorter else current_best
    return candidate if candidate.size < current_best.size else current_best


def speech_seconds(audio: np.ndarray, sr: int) -> float:
    """[v3.23.601] Độ dài PHẦN TIẾNG của bản thu — sau khi cắt lặng biên (hàm thuần).

    Lưới "dài bất thường" phải đo đúng thứ người nghe sẽ nghe. Khâu đặt câu cắt lặng
    đầu/cuối (:func:`trim_edge_silence`, ngưỡng tự dò) ngay sau khi tổng hợp, nên một bản
    thô 2.2 giây gồm 0.7 giây tiếng và 1.5 giây im lặng là bản đọc GỌN, không phải bản
    "ngân dài".

    Args:
        audio: Tín hiệu mono float32 vừa tổng hợp.
        sr: Tần số lấy mẫu (Hz).

    Returns:
        Số giây tiếng sau khi cắt lặng biên; 0.0 khi ``sr`` không hợp lệ.
    """
    if sr <= 0:
        return 0.0
    return trim_edge_silence(audio, sr, adaptive=True).size / sr


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
    # [v3.23.615] Bỏ đảo tạp âm lẻ ở hai đầu (xem :func:`speech_span`).
    # [v3.23.616] Đo đảo trên lõi của nó (``levels``): FLAC lồng tiếng thật 14/10 câu
    # "nhưng đường tu tiên" còn một tiếng 52ms (-13 dB) sau 270ms lặng ở cuối — đếm theo
    # ngưỡng "có tiếng" nó dài 80ms, quá ngưỡng đảo 60ms nên không bị cắt.
    span = speech_span(frame_rms > thr, window / sr, levels=frame_rms)
    if span is None:
        return audio  # toàn im lặng — giữ nguyên cho lớp retry/skip quyết định
    first_voiced, last_voiced = span
    start = max(0, first_voiced * window - int(keep_head_s * sr))
    end = min(len(audio), (last_voiced + 1) * window + int(keep_tail_s * sr))
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
