"""[v3.23.648] Khung thời gian VIDEO thực sự cần cho một lô dịch (hàm thuần).

Log thật 22/10: 111 request dịch đính video tốn **7,48 triệu token prompt (67 nghìn/request)** so với 268 nghìn token
đầu ra. Mỗi lô 40–50 dòng (~2 phút phim) nhưng bị đính TRỌN đoạn video phủ nó (~6 phút ở độ phân giải thấp ≈ 100 token/giây),
tức ~3× phần cần xem, và đó là nguyên nhân gần như toàn bộ thời gian chờ TPM và phần lớn lỗi 503.

Gemini nhận ``video_metadata.start_offset/end_offset`` để chỉ xử lý một khoảng của tệp. Đo thật bằng ``count_tokens`` trên
đoạn 138 s: cả đoạn 14 218 token; cắt 30 s còn 2 384–3 094 token.
"""

from __future__ import annotations

#: Lề thêm hai phía khung lô (giây): người đang nói thường vào/ra khung hình ngay trước/sau dòng đầu/cuối lô.
DEFAULT_MARGIN_S = 8.0
#: Chỉ cắt khi tiết kiệm đáng kể (khoảng cắt ≤ tỉ lệ này của đoạn) — cắt gần hết đoạn thì gửi nguyên cho đỡ rủi ro.
MAX_CLIP_FRACTION = 0.8
#: Khoảng cắt tối thiểu (giây): API từ chối khoảng quá ngắn.
MIN_CLIP_S = 4.0


def clip_window(
    chunk_start_s: float,
    chunk_end_s: float,
    batch_start_s: float,
    batch_end_s: float,
    margin_s: float = DEFAULT_MARGIN_S,
) -> tuple[float, float] | None:
    """Khoảng ``(bắt đầu, kết thúc)`` TÍNH TỪ ĐẦU ĐOẠN video cần gửi cho lô, hoặc ``None`` = gửi nguyên đoạn.

    Args:
        chunk_start_s: Mốc đầu của đoạn video trong phim (giây).
        chunk_end_s: Mốc cuối của đoạn.
        batch_start_s: Mốc bắt đầu dòng đầu lô.
        batch_end_s: Mốc kết thúc dòng cuối lô.
        margin_s: Lề thêm hai phía.

    Returns:
        Khoảng tương đối so với đầu đoạn, đã kẹp trong ``[0, độ dài đoạn]``; ``None`` khi đoạn quá ngắn, khoảng cắt không
        đủ ngắn so với đoạn, hoặc dữ liệu không hợp lệ.
    """
    length = chunk_end_s - chunk_start_s
    if length <= MIN_CLIP_S or batch_end_s < batch_start_s:
        return None
    start = max(0.0, batch_start_s - margin_s - chunk_start_s)
    end = min(length, batch_end_s + margin_s - chunk_start_s)
    if end - start < MIN_CLIP_S:
        # Lô nằm sát mép đoạn / rất ngắn: nới tới độ dài tối thiểu rồi kẹp lại trong đoạn.
        end = min(length, start + MIN_CLIP_S)
        start = max(0.0, end - MIN_CLIP_S)
    if end <= start or (end - start) > MAX_CLIP_FRACTION * length:
        return None
    return round(start, 2), round(end, 2)


__all__ = ["DEFAULT_MARGIN_S", "MAX_CLIP_FRACTION", "MIN_CLIP_S", "clip_window"]
