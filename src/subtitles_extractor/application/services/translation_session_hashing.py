"""Hàm băm nội dung phục vụ nhận biết tính hợp lệ của phiên dịch (Bước 2).

Ý tưởng: mỗi kết quả trung gian (phân tích ngữ cảnh, bản dịch từng giai đoạn) được
gắn HASH của ĐẦU VÀO sinh ra nó. Khi mở lại video, ta băm lại đầu vào hiện tại và so
với hash đã lưu — khớp thì khôi phục (dùng lại), lệch thì biết là cũ và cần làm lại.

Các hàm ở đây là HÀM THUẦN (pure): cùng đầu vào → cùng hash, không phụ thuộc trạng
thái ngoài, dễ kiểm thử.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable


def hash_text_lines(lines: Iterable[str]) -> str:
    """Băm một chuỗi dòng văn bản (ổn định theo thứ tự & nội dung).

    Dùng để nhận diện phụ đề nguồn: nếu người dùng sửa nội dung, hash đổi → biết là
    nguồn đã thay đổi so với lúc dịch trước.

    Args:
        lines: Các dòng văn bản theo thứ tự.

    Returns:
        Chuỗi hex SHA-256 (64 ký tự).
    """
    hasher = hashlib.sha256()
    for line in lines:
        hasher.update(line.encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def hash_analysis_input(
    source_lines: Iterable[str], target_lang: str, video_signature: str = ""
) -> str:
    """Băm đầu vào của bước PHÂN TÍCH ngữ cảnh.

    Phân tích phụ thuộc: nội dung phụ đề nguồn + ngôn ngữ đích + (tuỳ chọn) chữ ký
    video kèm theo. Mọi yếu tố này đổi → cần phân tích lại.

    Args:
        source_lines: Các dòng phụ đề nguồn (chỉ nội dung văn bản).
        target_lang: Mã ngôn ngữ đích.
        video_signature: Chữ ký video ngữ cảnh (nếu có); rỗng nếu không kèm video.

    Returns:
        Chuỗi hex SHA-256.
    """
    hasher = hashlib.sha256()
    hasher.update(b"analysis::")
    hasher.update(target_lang.encode("utf-8"))
    hasher.update(b"::")
    hasher.update(video_signature.encode("utf-8"))
    hasher.update(b"::")
    for line in source_lines:
        hasher.update(line.encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def hash_stage_input(
    previous_stage_hash: str, stage_id: str, target_lang: str, context_hash: str = ""
) -> str:
    """Băm đầu vào của MỘT giai đoạn dịch (chuỗi giai đoạn phụ thuộc nhau).

    Mỗi giai đoạn nhận kết quả giai đoạn trước làm đầu vào, nên hash của nó gồm: hash
    giai đoạn trước + id giai đoạn + ngôn ngữ đích + hash ngữ cảnh. Nhờ tính dây
    chuyền này, nếu một giai đoạn sớm thay đổi thì mọi giai đoạn sau tự động "lệch
    hash" → biết phải dịch lại từ điểm đó trở đi.

    Args:
        previous_stage_hash: Hash đầu vào/kết quả của giai đoạn liền trước (hoặc hash
            phụ đề nguồn nếu đây là giai đoạn đầu).
        stage_id: Định danh giai đoạn.
        target_lang: Mã ngôn ngữ đích.
        context_hash: Hash của ngữ cảnh dịch áp dụng (nhân vật/tóm tắt) nếu có.

    Returns:
        Chuỗi hex SHA-256.
    """
    hasher = hashlib.sha256()
    hasher.update(b"stage::")
    hasher.update(stage_id.encode("utf-8"))
    hasher.update(b"::")
    hasher.update(target_lang.encode("utf-8"))
    hasher.update(b"::")
    hasher.update(context_hash.encode("utf-8"))
    hasher.update(b"::")
    hasher.update(previous_stage_hash.encode("utf-8"))
    return hasher.hexdigest()


__all__ = ["hash_text_lines", "hash_analysis_input", "hash_stage_input"]
