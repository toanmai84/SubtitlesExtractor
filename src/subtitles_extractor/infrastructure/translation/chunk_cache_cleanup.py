"""[v3.23.166] Lập kế hoạch dọn cache file đoạn video nén cục bộ (thuần, test được).

Từ v3.23.157, file đoạn nén (``*.ctxpartNN.hXXXfY.mp4``) được GIỮ lại để tái dùng khi
xoay API key / chạy lại cùng phim (tránh nén lại ~5 phút/lần). Hệ quả: file tích tụ
dần, chiếm ổ cứng. Module này quyết định file nào cần XOÁ theo ngân sách, tách khỏi
thao tác I/O để kiểm thử luật dọn độc lập (đầu vào -> đầu ra thuần).

Luật ưu tiên:
1. KHÔNG BAO GIỜ xoá file thuộc phiên hiện tại (``protected_paths``) — đang/sắp dùng.
2. Xoá file QUÁ HẠN TUỔI trước (theo ``max_age_seconds``).
3. Nếu tổng dung lượng còn lại VẪN vượt ``max_total_bytes``: xoá tiếp theo LRU
   (mtime cũ nhất trước) tới khi về dưới ngân sách.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ChunkFileInfo:
    """Thông tin một file đoạn nén cục bộ phục vụ quyết định dọn.

    Attributes:
        path: Đường dẫn tuyệt đối (dạng chuỗi) tới file đoạn.
        size_bytes: Kích thước file (byte).
        mtime: Thời điểm sửa đổi lần cuối (epoch giây) — dùng cho LRU và tuổi.
    """

    path: str
    size_bytes: int
    mtime: float


def plan_chunk_cache_cleanup(
    files: list[ChunkFileInfo],
    max_total_bytes: int,
    max_age_seconds: float,
    now_epoch: float,
    protected_paths: frozenset[str] = frozenset(),
) -> list[str]:
    """Quyết định danh sách đường dẫn file đoạn cần XOÁ theo ngân sách.

    Hàm thuần: không chạm đĩa, kết quả chỉ phụ thuộc đầu vào -> dễ kiểm thử.

    Args:
        files: Danh sách file đoạn hiện có (kèm size, mtime).
        max_total_bytes: Ngân sách dung lượng tối đa cho cache (byte). ``<= 0`` nghĩa
            là không giới hạn theo dung lượng (chỉ dọn theo tuổi).
        max_age_seconds: Tuổi tối đa của file (giây). File cũ hơn mức này bị xoá.
            ``<= 0`` nghĩa là không dọn theo tuổi.
        now_epoch: Thời điểm hiện tại (epoch giây) để tính tuổi file.
        protected_paths: Tập đường dẫn KHÔNG được xoá (phiên hiện tại đang dùng).

    Returns:
        Danh sách đường dẫn cần xoá, thứ tự cũ->mới (an toàn để unlink tuần tự).
    """
    candidates = [info for info in files if info.path not in protected_paths]
    to_delete_paths: set[str] = set()

    if max_age_seconds > 0:
        for info in candidates:
            if now_epoch - info.mtime > max_age_seconds:
                to_delete_paths.add(info.path)

    if max_total_bytes > 0:
        # Dung lượng còn lại sau khi đã loại các file quá hạn tuổi.
        surviving = [info for info in candidates if info.path not in to_delete_paths]
        remaining_bytes = sum(info.size_bytes for info in surviving)
        if remaining_bytes > max_total_bytes:
            # LRU: xoá file cũ nhất trước cho tới khi về dưới ngân sách.
            for info in sorted(surviving, key=lambda item: item.mtime):
                if remaining_bytes <= max_total_bytes:
                    break
                to_delete_paths.add(info.path)
                remaining_bytes -= info.size_bytes

    # Trả về theo thứ tự cũ -> mới để log/unlink có trật tự ổn định.
    ordered = sorted(
        (info for info in candidates if info.path in to_delete_paths),
        key=lambda item: item.mtime,
    )
    return [info.path for info in ordered]
