"""[v3.23.601] Callback "tải lại video bằng key hiện tại" dùng chung cho worker Gemini.

Adapter Gemini xoay API key khi key cũ cạn hạn mức; file video đã tải bằng key cũ trả 403
với key mới, nên adapter gọi lại callback này để tải đúng các đoạn cần (v3.23.132/550).
Trước bản này, worker phân tích và worker dịch mỗi bên chép một bản giống hệt; worker "Bổ
sung người nói" là bên thứ ba cần đúng hành vi đó.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

VideoReuploadCallback = Callable[..., list]


def make_video_reupload_callback(
    video_provider: Any,
    video_path: str,
    cancel_cb: Callable[[], bool],
    *,
    on_uploaded: Callable[[list], None] | None = None,
) -> VideoReuploadCallback:
    """Tạo callback tải lại video cho adapter dịch/phân tích.

    Args:
        video_provider: Bộ chuẩn bị/tải video ngữ cảnh (``set_active_key``,
            ``prepare_and_upload``, tuỳ chọn ``upload_chunks_for_ranges``).
        video_path: Đường dẫn video gốc.
        cancel_cb: Callback huỷ hợp tác.
        on_uploaded: [v3.23.609] Nhận các tham chiếu vừa tải, để ghi vào phiên — nút
            "Xoá file cloud" chỉ xoá được những file đã được ghi lại.

    Returns:
        Hàm ``(api_key, ranges=None) -> list`` trả các tham chiếu video đã tải.
    """

    def upload(api_key: str, ranges: list[tuple[float, float]] | None) -> list:
        video_provider.set_active_key(api_key)
        # [v3.23.550] Có danh sách khoảng cần → chỉ tải đúng các đoạn đó (tải lười),
        # thay vì tải lại cả phim mỗi lần xoay key.
        upload_ranges = getattr(video_provider, "upload_chunks_for_ranges", None)
        if ranges and callable(upload_ranges):
            partial = list(upload_ranges(Path(video_path), ranges, cancel_cb=cancel_cb))
            if partial:
                return partial
        return list(
            video_provider.prepare_and_upload(Path(video_path), cancel_cb=cancel_cb)
        )

    def reupload(
        api_key: str, ranges: list[tuple[float, float]] | None = None
    ) -> list:
        refs = upload(api_key, ranges)
        if refs and on_uploaded is not None:
            on_uploaded(list(refs))
        return refs

    return reupload


__all__ = ["VideoReuploadCallback", "make_video_reupload_callback"]
