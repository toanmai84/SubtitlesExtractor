"""[v3.23.611] Chọn model cho các lượt DỊCH LẠI phụ (trọng tài, dọn CJK, dọn dòng trùng).

Mọi lượt dịch lại phụ dùng cấu hình của giai đoạn dịch thô — dịch lại từ câu gốc, sát
nghĩa, không kèm video. Nhưng model của giai đoạn đó có thể đã hết hạn mức NGÀY từ lâu:
người dùng chạy dịch thô xong thì hết lượt, đổi model cho giai đoạn sau rồi chạy tiếp.
Ca thật v3.23.609 (phim "灭门惨案II"): giai đoạn phong cách chạy trên ``gemini-3.6-flash``
nhưng trọng tài vẫn gọi ``gemini-3.7-flash`` (model dịch thô, đã chạm 20 request/ngày)
nên thất bại — 4 dòng lẻ bị viết lại sai không được phân xử, đi thẳng tới bản cuối.

Ở đây chỉ chọn TÊN model; các thiết lập khác (nhiệt độ, thinking, cỡ lô) vẫn lấy từ
giai đoạn dịch thô để bản dịch lại giữ đúng tính chất "dịch sát từ câu gốc".
"""

from __future__ import annotations

from collections.abc import Callable, Iterable


def choose_retranslation_model(
    preferred: str,
    alternatives: Iterable[str],
    has_quota: Callable[[str], bool],
) -> str:
    """Chọn model đầu tiên còn hạn mức ngày, ưu tiên model của giai đoạn dịch thô.

    Args:
        preferred: Model của giai đoạn dịch thô.
        alternatives: Các model thay thế theo thứ tự ưu tiên (thường là model của giai
            đoạn đang chạy rồi tới các giai đoạn khác); tên trống/trùng được bỏ qua.
        has_quota: Hàm cho biết model còn ít nhất một key có hạn mức ngày.

    Returns:
        Model đầu tiên còn hạn mức; ``preferred`` khi không model nào còn (để lời gọi
        thất bại với đúng thông báo hết hạn mức của model dịch thô như trước).
    """
    seen: set[str] = set()
    for model_name in (preferred, *alternatives):
        name = (model_name or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        if has_quota(name):
            return name
    return preferred
