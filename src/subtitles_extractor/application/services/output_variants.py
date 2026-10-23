"""[v3.23.596] Tìm các biến thể ngôn ngữ của một tệp đầu ra (``<tên>.tts.vi.srt``…).

Tên phim tải về thường mang ngoặc vuông —
``灭门惨案II借种..AI修复版[老舅电影laody.net发布]`` — mà ``[...]`` trong mẫu ``glob`` là
TẬP KÝ TỰ chứ không phải chữ thường: mẫu ``<tên>.tts.*.srt`` không bao giờ khớp chính tệp
đó. Nhật ký thật 10/10/2026: khâu TTS đã ghi ``<tên>.tts.vi.srt`` cạnh tệp giọng đọc,
nhưng trang Xuất bản báo "Không tìm thấy phụ đề đã chỉnh giờ theo giọng đọc" và xuất phim
với phụ đề mốc GỐC, lệch giọng thuyết minh.
"""

from __future__ import annotations

import glob
from pathlib import Path


def find_output_variants(
    folder: Path, stem: str, marker: str, suffix: str = ".srt"
) -> list[Path]:
    """Các tệp ``<stem>.<marker>.*<suffix>`` trong ``folder``, sắp theo tên.

    Args:
        folder: Thư mục chứa tệp.
        stem: Tên gốc (có thể chứa ``[``, ``]``, ``*``, ``?`` — được thoát trước khi dò).
        marker: Đoạn đánh dấu loại tệp, vd ``"tts"`` hoặc ``"translate"``.
        suffix: Phần mở rộng, gồm dấu chấm.

    Returns:
        Danh sách đường dẫn khớp (rỗng nếu không có).

    Raises:
        OSError: Không đọc được thư mục.
    """
    return sorted(folder.glob(f"{glob.escape(stem)}.{marker}.*{suffix}"))


__all__ = ["find_output_variants"]
