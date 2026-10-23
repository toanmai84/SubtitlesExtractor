"""[v3.23.585] Thử lại khi Gemini QUÁ TẢI (503) mà hạn mức ngày của model ÍT.

Nhật ký thật v3.23.584 (phim "灭门惨案II", phân tích 17 đoạn bằng gemini-3.8-flash, hạn
mức 20 request/ngày):

* Mỗi lượt gọi trả 503 VẪN bị server tính vào hạn mức ngày. Bảy lần server báo hết hạn
  mức trong nhật ký, tổng "request ứng dụng đếm + số lần 503" là 21, 20, 19, 21, 16, 21,
  17 — luôn xấp xỉ 20 — trong khi ứng dụng (hoàn lại bộ đếm sau mỗi 503) chỉ đếm 0, 0, 1,
  8, 5, 8, 4.
* Ngân sách chờ quá tải 600 giây cho tới ~11 lượt thử MỖI lời gọi. Với model 20
  request/ngày, hai đoạn hỏng là hết sạch hạn mức của một key: lần phân tích đầu đốt 60
  request (3 key) trong 93 phút và chỉ xong 1/17 đoạn.

Thống kê mọi nhật ký đã nhận (phân tích theo đoạn), số đoạn thành công sau k lần 503:
k=0: 195, 1: 50, 2: 10, 3: 13, 4: 5, 5: 1, 6: 3, 7: 2; còn 11 đoạn hỏng hẳn sau 10-11
lần 503. Tức 273/279 đoạn (97,8%) xong trong 5 lượt thử; từ lượt thứ 9 trở đi KHÔNG
lượt nào thành công — 3 lượt cuối của mỗi lời gọi hỏng chỉ đốt hạn mức.

Quy tắc ở đây — chỉ áp cho model hạn mức ÍT (hạn mức dồi dào giữ nguyên ngân sách thời
gian): một lời gọi được thử tối đa 5 lượt, các lần chờ thưa dần (30, 60, 120, 120 giây).
Bốn lần chờ đó dài 330 giây — bằng quãng mà cách cũ (30, 30, 30, 42, 82, 90…) cần 7
lượt thử mới phủ được, nên ít lượt hơn mà vẫn vượt qua các đợt quá tải ngắn.

KHÔNG siết thêm theo số request còn lại của key: đã thử "không quá 1/3 số còn lại" trên
chính nhật ký này thì lượt phân tích sáng hôm sau (16/17 đoạn) sẽ dừng ngay ở đoạn 4 —
đoạn ấy thành công ở lượt thử thứ 5 — tức dừng oan trong một ngày quá tải bình thường.

[v3.23.592] Quá tải kéo dài thì THỬ KEY KHÁC trước khi bỏ cuộc. Trước bản này ứng dụng
chỉ xoay key khi hết hạn mức; gặp 503 thì đứng yên trên một key, với lý do "quá tải là
của model, đổi key không hết". Nhật ký thật v3.23.591 (ba API key):

* Phân tích: key #1 hết hạn mức ở đoạn 12 → xoay sang #2 → 5 lượt 503 liền trên #2 →
  DỪNG, còn 6/17 đoạn, trong khi key #3 còn nguyên 20 request. Người dùng bấm phân tích
  lại sau 100 giây: ứng dụng chọn key #3 và đoạn 12 xong ngay lượt đầu.
* Dịch (gemini-3.5-flash-lite): 11 lượt 503 trong 14 phút trên key #1 rồi cả lượt dịch
  thất bại ở giai đoạn 4; key #2 và #3 không được thử lần nào.

Giả định "đổi key không hết" không khớp dữ liệu: trên 54 nhật ký, sau một lượt 503, lượt
kế tiếp trên CÙNG key thành công 127/360 lần (35%); bốn lần lượt kế tiếp rơi vào key
KHÁC (vì key cũ vừa hết hạn mức) thì thành công 3. Mẫu nhỏ, nhưng thử key khác tốn rất
ít nên đáng làm:

* Model hạn mức ÍT: key đầu chạm trần 5 lượt thì mỗi key còn hạn mức được THĂM DÒ tối đa
  :data:`SCARCE_PROBE_ATTEMPTS` lượt (mỗi lượt 503 vẫn bị tính, nên chỉ thăm dò, không
  thử đủ 5). Tệ nhất tốn thêm 2 request cho mỗi key khác.
* Model hạn mức dồi dào: sau :data:`PLENTIFUL_ROTATE_AFTER` lượt 503 liên tiếp trên một
  key thì chuyển sang key chưa thử; số request không phải là điều đáng tiếc ở đây.

Toàn bộ module là HÀM THUẦN: không I/O.
"""

from __future__ import annotations

#: Model có hạn mức ngày từ mức này trở xuống bị coi là ÍT (20, 50, 100 request/ngày).
SCARCE_DAILY_LIMIT = 100

#: Số lượt thử tối đa cho một lời gọi khi quá tải, ở model hạn mức ít.
SCARCE_MAX_ATTEMPTS = 5

#: Lần chờ đầu và trần mỗi lần chờ (giây) khi thử lại ở model hạn mức ít.
SCARCE_BASE_DELAY_S = 30.0
SCARCE_MAX_DELAY_S = 120.0

#: [v3.23.592] Số lượt thăm dò trên MỖI key khác sau khi key đầu chạm trần quá tải (model
#: hạn mức ít).
SCARCE_PROBE_ATTEMPTS = 2

#: [v3.23.592] Model hạn mức dồi dào: số lượt quá tải LIÊN TIẾP trên một key trước khi
#: chuyển sang key chưa thử.
PLENTIFUL_ROTATE_AFTER = 3


def overload_attempt_cap(rpd_limit: int) -> int | None:
    """Số lượt thử tối đa cho MỘT lời gọi khi server quá tải.

    Args:
        rpd_limit: Hạn mức request/ngày của model (≤ 0 = không biết).

    Returns:
        Số lượt thử (kể cả lượt đầu), hoặc ``None`` khi hạn mức dồi dào/không biết —
        lúc đó chỉ ngân sách thời gian quyết định.
    """
    if rpd_limit <= 0 or rpd_limit > SCARCE_DAILY_LIMIT:
        return None
    return SCARCE_MAX_ATTEMPTS


def scarce_overload_delay(failures: int) -> float:
    """Thời gian chờ trước lượt thử kế tiếp, sau ``failures`` lần quá tải liên tiếp.

    Args:
        failures: Số lần 503 đã gặp trong lời gọi này (≥ 1).

    Returns:
        30, 60, 120, 120… giây.
    """
    exponent = max(0, failures - 1)
    return min(SCARCE_MAX_DELAY_S, SCARCE_BASE_DELAY_S * (2**exponent))


def overload_key_cap(first_key_cap: int | None, keys_tried_before: int) -> int | None:
    """[v3.23.592] Trần số lượt quá tải trên key ĐANG DÙNG của một lời gọi (hàm thuần).

    Args:
        first_key_cap: Kết quả :func:`overload_attempt_cap` (``None`` = hạn mức dồi dào).
        keys_tried_before: Số key đã chạm trần quá tải trước key này trong lời gọi.

    Returns:
        Trần đầy đủ cho key đầu, :data:`SCARCE_PROBE_ATTEMPTS` cho các key thăm dò;
        ``None`` khi hạn mức dồi dào (chỉ ngân sách thời gian quyết định).
    """
    if first_key_cap is None:
        return None
    if keys_tried_before <= 0:
        return first_key_cap
    return min(first_key_cap, SCARCE_PROBE_ATTEMPTS)


def should_try_other_key(key_cap: int | None, failures_on_key: int) -> bool:
    """[v3.23.592] Đã đến lúc thử key khác vì quá tải kéo dài chưa (hàm thuần).

    Args:
        key_cap: Trần của key đang dùng (:func:`overload_key_cap`).
        failures_on_key: Số lượt quá tải liên tiếp trên key đang dùng.

    Returns:
        True khi key hạn mức ít đã chạm trần, hoặc key hạn mức dồi dào đã quá tải
        :data:`PLENTIFUL_ROTATE_AFTER` lượt liền.
    """
    if key_cap is None:
        return failures_on_key >= PLENTIFUL_ROTATE_AFTER
    return failures_on_key >= key_cap


#: [v3.23.615] Lượt thử chạy tới ít nhất tỉ lệ này của thời hạn mỗi lời gọi rồi mới hỏng
#: thì coi là CHẠM HẠN CHÓT (máy chủ chưa kịp trả lời), không phải lỗi mạng tức thời.
DEADLINE_HIT_RATIO = 0.8

#: [v3.23.615] Hệ số nới thời hạn cho lượt thử sau mỗi lần chạm hạn chót.
DEADLINE_TIMEOUT_GROWTH = 1.5

#: [v3.23.615] Trần thời hạn mỗi lời gọi khi tự nới (giây) — bằng mức tối đa ô Cài đặt cho
#: phép, để không lời gọi nào được treo lâu hơn mức người dùng có thể tự chọn.
REQUEST_TIMEOUT_CEILING_S = 600.0


def deadline_was_hit(elapsed_s: float, timeout_s: float) -> bool:
    """[v3.23.615] Lượt thử hỏng SAU KHI đã chạy gần hết thời hạn mỗi lời gọi (hàm thuần).

    Nhật ký thật (44 tệp app_log): 56 lượt thử hỏng đúng ở mốc 300 giây — 504
    DEADLINE_EXCEEDED, 499 CANCELLED, "The read operation timed out" — tức máy chủ còn
    đang làm thì hết thời hạn ứng dụng đặt. Sau một lần như vậy, lượt kế tiếp cùng thời
    hạn lại chạm mốc 17/62 lần (mất thêm 5 phút và một request mỗi lần).

    Args:
        elapsed_s: Thời gian lượt thử đã chạy tới khi hỏng (giây).
        timeout_s: Thời hạn mỗi lời gọi đang áp (giây).

    Returns:
        True nếu lượt thử đã chạy ít nhất :data:`DEADLINE_HIT_RATIO` thời hạn.
    """
    if timeout_s <= 0:
        return False
    return elapsed_s >= DEADLINE_HIT_RATIO * timeout_s


def escalated_request_timeout(
    current_s: float, ceiling_s: float = REQUEST_TIMEOUT_CEILING_S
) -> float:
    """[v3.23.615] Thời hạn cho lượt thử sau một lần chạm hạn chót (hàm thuần).

    Args:
        current_s: Thời hạn đang áp (giây).
        ceiling_s: Trần thời hạn.

    Returns:
        ``current_s`` nhân :data:`DEADLINE_TIMEOUT_GROWTH`, không vượt trần và không bao
        giờ ngắn hơn thời hạn hiện tại (người dùng đặt cao hơn trần thì giữ nguyên).
    """
    if current_s <= 0:
        return current_s
    return max(current_s, min(current_s * DEADLINE_TIMEOUT_GROWTH, ceiling_s))


def overload_failure_message(
    model_name: str, *, attempts: int, keys: int, waited_s: float, reason: str
) -> str:
    """[v3.23.616] Thông điệp tiếng Việt khi bỏ cuộc vì model quá tải (hàm thuần).

    Nhật ký 14/10: ba lượt dịch liền báo lên giao diện nguyên khối JSON tiếng Anh của máy
    chủ (``503 UNAVAILABLE. {'error': {'code': 503, 'message': 'This model is currently
    experiencing high demand…``) sau 8-14 phút chờ; người dùng tự đổi lần lượt ba model
    mới dịch xong. Thông điệp cần nói đã thử bao nhiêu lượt, bao lâu, và nguyên nhân gọn.

    Args:
        model_name: Model đang quá tải.
        attempts: Số lượt gửi bị từ chối vì quá tải.
        keys: Số API key đã thử.
        waited_s: Tổng thời gian đã chờ giữa các lượt (giây).
        reason: Lỗi máy chủ đã rút gọn (vd ``"503 UNAVAILABLE"``).

    Returns:
        Thông điệp một câu, giữ mã lỗi gọn để tầng giao diện nhận dạng nhóm lỗi.
    """
    minutes = max(1, round(waited_s / 60.0))
    return (
        f"Gemini đang quá tải với model '{model_name}': đã thử {attempts} lượt trên "
        f"{keys} API key trong khoảng {minutes} phút mà máy chủ vẫn báo bận ({reason})."
    )


__all__ = [
    "DEADLINE_HIT_RATIO",
    "DEADLINE_TIMEOUT_GROWTH",
    "PLENTIFUL_ROTATE_AFTER",
    "REQUEST_TIMEOUT_CEILING_S",
    "SCARCE_DAILY_LIMIT",
    "SCARCE_MAX_ATTEMPTS",
    "SCARCE_PROBE_ATTEMPTS",
    "deadline_was_hit",
    "escalated_request_timeout",
    "overload_attempt_cap",
    "overload_failure_message",
    "overload_key_cap",
    "scarce_overload_delay",
    "should_try_other_key",
]
