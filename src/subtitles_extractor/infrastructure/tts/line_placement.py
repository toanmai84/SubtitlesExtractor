"""[v3.23.596] Đặt MỘT câu lên trục thời gian: đọc sớm, ngân trễ, tốc độ (hàm thuần).

Dùng chung cho VieNeu và Gemini. Hai engine này từng chép cùng một khối tính "khung hiệu
dụng → ăn gian đầu → tốc độ" trong ba hàm riêng (VieNeu, Gemini thường, Gemini native).
Gom về đây để thêm elastic timing một lần cho cả ba, đúng kỷ luật đồng bộ engine.

ELASTIC TIMING (đồng bộ ngữ nghĩa với Edge): câu quá chật được nén tới mức "êm"
(``base x Nén êm``) trước; còn thiếu mới NGÂN TRỄ qua mốc câu sau tối đa
``min(Dung sai cộng dồn, 0,5 s)`` — câu sau đọc lùi lại đúng phần đó (KHÔNG chồng tiếng).
Đo trên lượt lồng tiếng thật 10/10/2026 (1245 dòng VieNeu, base 1,3x, trần 2,5x, Dung sai
0,25 s, Nén êm 1,5): dòng phải nén ≥ 2,0x 29 → 13, dòng chồng tiếng 6 → 0; đổi lại 34
dòng (2,7%) đọc trễ, trung vị 0,19 s. Mô phỏng tái lập đúng lượt thật khi tắt elastic
(1073 dòng ở tốc độ cơ bản, khớp từng dòng). Phương án "dời mốc không giới hạn" bị bác ở
v3.23.204 (đè tiếng 16 → 32 câu do domino) — ở đây trễ có trần và câu sau KHÔNG đè.
"""

from __future__ import annotations

from dataclasses import dataclass

from subtitles_extractor.domain.ports.subtitle_tts_port import TTSRequest
from subtitles_extractor.infrastructure.tts.timing_math import (
    effective_available_seconds,
    lead_in_seconds,
    total_speed_ratio,
)

#: Trần ngân trễ (giây) — cùng trần khẩu hình với Edge (``_LIPSYNC_LATE_CAP_S``).
ELASTIC_LATE_CAP_S = 0.5
#: Khoảng nghỉ giữa tiếng câu trước và câu bị đọc lùi (giây).
ELASTIC_JOIN_GAP_S = 0.05


@dataclass(frozen=True, slots=True)
class LinePlacement:
    """Kết quả đặt một câu.

    Attributes:
        play_start_sec: Mốc phát tiếng (sớm hơn phụ đề khi ăn gian đầu, trễ hơn khi bị
            câu trước ngân trễ đẩy lùi).
        effective_available_s: Khung từ ``play_start_sec`` tới mép khung của câu (gồm cả
            khoảng nghỉ hội thoại) — dùng tính phần lấn.
        speed: Tỉ lệ nén áp lên giọng (≥ 1,0).
    """

    play_start_sec: float
    effective_available_s: float
    speed: float


def elastic_late_seconds(elastic_timing: bool, max_drift_s: float) -> float:
    """Phần ngân trễ tối đa cho phép (giây).

    Args:
        elastic_timing: Người dùng bật "Elastic timing".
        max_drift_s: "Dung sai cộng dồn" (giây).

    Returns:
        ``min(max_drift_s, ELASTIC_LATE_CAP_S)`` khi bật; 0 khi tắt.
    """
    if not elastic_timing:
        return 0.0
    return max(0.0, min(float(max_drift_s), ELASTIC_LATE_CAP_S))


def comfort_speed(request: TTSRequest) -> float:
    """Tốc độ "êm" của yêu cầu: ``min(base x Nén êm, trần tốc độ)`` — như Edge.

    Args:
        request: Yêu cầu TTS.

    Returns:
        Tốc độ êm (x).
    """
    base = float(request.base_speed)
    ratio = max(1.0, float(getattr(request, "comfort_speed_ratio", 1.0)))
    return min(base * ratio, max(base, float(request.max_speed)))


@dataclass(frozen=True, slots=True)
class LineContext:
    """Biên thời gian của một dòng trong vòng tổng hợp tuần tự.

    Attributes:
        next_start_sec: Mốc câu kế tiếp (``None`` = không có biên).
        prev_audio_end_sec: Tiếng câu trước kết thúc lúc nào.
        late_s: Phần dòng này được ngân trễ qua mốc câu sau.
        defer_s: Phần dòng này được đọc lùi khi câu trước ngân trễ sang (câu cuối phim
            không được ngân trễ nhưng vẫn phải lùi sau tiếng câu trước).
    """

    next_start_sec: float | None
    prev_audio_end_sec: float
    late_s: float
    defer_s: float


def line_context(
    request: TTSRequest, position: int, running_prev_end_sec: float
) -> LineContext:
    """Biên thời gian của dòng ``position`` — dùng chung ba vòng tổng hợp.

    * Lượt đầy đủ: câu kế là ``events[position + 1]``; câu cuối lấy thời lượng video làm
      biên và KHÔNG ngân trễ (tiếng không được tràn qua cuối phim).
    * TTS lại tập con (``request.neighbors``): dùng mốc thật của dòng kề trong cả phim;
      dòng kề không được TTS lại thì không ngân trễ qua nó.

    Args:
        request: Yêu cầu TTS.
        position: Vị trí dòng trong ``request.events``.
        running_prev_end_sec: Tiếng câu tổng hợp ngay trước (trong lượt này) kết thúc lúc
            nào.

    Returns:
        :class:`LineContext`.
    """
    late_s = elastic_late_seconds(
        bool(getattr(request, "elastic_timing", False)),
        float(getattr(request, "max_drift_s", 0.0)),
    )
    neighbors = request.neighbors_at(position)
    if neighbors is not None:
        prev_end = (
            running_prev_end_sec
            if neighbors.prev_audio_end_sec is None
            else neighbors.prev_audio_end_sec
        )
        no_late = neighbors.next_fixed or neighbors.next_start_sec is None
        return LineContext(
            neighbors.next_start_sec, prev_end, 0.0 if no_late else late_s, late_s
        )
    if position + 1 < len(request.events):
        return LineContext(
            request.events[position + 1].start_sec, running_prev_end_sec, late_s, late_s
        )
    return LineContext(
        getattr(request, "media_duration_s", None), running_prev_end_sec, 0.0, late_s
    )


def place_line(
    *,
    start_sec: float,
    end_sec: float,
    next_start_sec: float | None,
    prev_audio_end_sec: float,
    voice_duration_s: float,
    pause_s: float,
    base_speed: float,
    max_speed: float,
    max_lead_s: float,
    late_s: float = 0.0,
    comfort_speed: float | None = None,
    defer_s: float | None = None,
) -> LinePlacement:
    """Đặt một câu: mốc phát, khung hiệu dụng và tốc độ.

    ``late_s = 0`` cho kết quả y hệt cách tính trước v3.23.596 (khung tới câu sau + ăn
    gian đầu đúng mức cần).

    Args:
        start_sec: Mốc bắt đầu phụ đề của câu.
        end_sec: Mốc kết thúc phụ đề của câu.
        next_start_sec: Mốc câu kế tiếp (``None`` = câu cuối, không có biên).
        prev_audio_end_sec: Tiếng câu trước kết thúc lúc nào (0 nếu là câu đầu).
        voice_duration_s: Độ dài giọng model sinh (chưa nén, đã cắt im lặng biên).
        pause_s: Khoảng nghỉ hội thoại chèn trước giọng (không bị nén).
        base_speed: Tốc độ đọc cơ bản.
        max_speed: Trần tốc độ người dùng đặt.
        max_lead_s: Trần ăn gian đầu.
        late_s: Phần ngân trễ cho phép (:func:`elastic_late_seconds`); 0 = tắt elastic.
        comfort_speed: Tốc độ "êm" — nén tới mức này trước khi ngân trễ; ``None`` = luôn
            ngân trễ khi vượt tốc độ cơ bản.
        defer_s: Phần được đọc lùi khi tiếng câu trước còn kéo dài qua mốc câu này;
            ``None`` = bằng ``late_s``.

    Returns:
        :class:`LinePlacement`.
    """
    comfort = float(base_speed) if comfort_speed is None else float(comfort_speed)
    max_defer = late_s if defer_s is None else defer_s
    deferred = max_defer > 0.0 and prev_audio_end_sec > start_sec

    def _place(extra_s: float) -> LinePlacement:
        limit = None if next_start_sec is None else next_start_sec + extra_s
        window_end = start_sec + effective_available_seconds(start_sec, end_sec, limit)
        if deferred:
            # Câu trước đã ngân trễ qua mốc câu này: đọc lùi sau tiếng của nó (không
            # đè), nhưng không lùi quá trần — vượt trần thì chấp nhận chồng như khi tắt
            # elastic.
            play_start = min(
                prev_audio_end_sec + ELASTIC_JOIN_GAP_S, start_sec + max_defer
            )
        else:
            needed_window = voice_duration_s / max(1.0, float(base_speed)) + pause_s
            play_start = start_sec - lead_in_seconds(
                start_sec,
                prev_audio_end_sec,
                max_lead_s=max_lead_s,
                needed_lead_s=max(0.0, needed_window - (window_end - start_sec)),
            )
        available = max(0.0, window_end - play_start)
        speed = total_speed_ratio(
            voice_duration_s, max(0.05, available - pause_s), base_speed, max_speed
        )
        return LinePlacement(play_start, available, speed)

    placement = _place(0.0)
    if late_s <= 0.0 or next_start_sec is None or placement.speed <= comfort:
        return placement
    stretched = _place(late_s)
    # Nén tới mức êm trước, chỉ phần còn thiếu mới ngân trễ: câu sau lùi ít nhất có thể.
    return LinePlacement(
        stretched.play_start_sec,
        stretched.effective_available_s,
        max(stretched.speed, min(placement.speed, comfort)),
    )


__all__ = [
    "ELASTIC_JOIN_GAP_S",
    "ELASTIC_LATE_CAP_S",
    "LineContext",
    "LinePlacement",
    "comfort_speed",
    "elastic_late_seconds",
    "line_context",
    "place_line",
]
