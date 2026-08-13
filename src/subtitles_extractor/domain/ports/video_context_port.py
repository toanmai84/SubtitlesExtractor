"""Cổng (port) cung cấp NGỮ CẢNH VIDEO cho khâu dịch phụ đề.

Ý tưởng: cho mô hình dịch (vd Gemini) "xem" video gốc để nắm bối cảnh hình ảnh,
nhân vật, sắc thái — nhờ đó dịch chính xác và tự nhiên hơn so với chỉ đọc văn bản.

Vấn đề thực tế: video phim dài (hàng giờ) vượt cả giới hạn DUNG LƯỢNG tải lên lẫn
ngân sách TOKEN của mô hình. Vì vậy port định nghĩa bước "chuẩn bị" để **tự cắt
video thành nhiều đoạn** vừa với giới hạn, rồi tải từng đoạn lên dịch vụ đám mây và
tái sử dụng (cache) theo nội dung để khỏi tải lại.

Tầng domain chỉ khai báo hợp đồng; chi tiết ffmpeg/đám mây nằm ở tầng hạ tầng.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class VideoChunk:
    """Một đoạn video đã cắt để dùng làm ngữ cảnh.

    Attributes:
        index: thứ tự đoạn (0-based).
        path: đường dẫn file đoạn trên đĩa.
        start_sec: mốc bắt đầu của đoạn trong video gốc (giây).
        end_sec: mốc kết thúc của đoạn trong video gốc (giây).
        is_full_video: True nếu đoạn chính là toàn bộ video (không cần cắt).
    """

    index: int
    path: Path
    start_sec: float
    end_sec: float
    is_full_video: bool = False

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)

    def covers(self, t_sec: float) -> bool:
        """True nếu mốc thời gian ``t_sec`` (theo video gốc) nằm trong đoạn này."""
        return self.start_sec <= t_sec < self.end_sec


@dataclass(frozen=True)
class RemoteVideoRef:
    """Tham chiếu tới một đoạn video ĐÃ tải lên dịch vụ đám mây."""

    chunk_index: int
    remote_name: str          # định danh file phía dịch vụ (vd Gemini files/xxx)
    start_sec: float
    end_sec: float
    state: str = "ACTIVE"


@dataclass(frozen=True)
class VideoContextPlan:
    """Kế hoạch chuẩn bị ngữ cảnh video cho một file."""

    source_path: Path
    duration_sec: float
    estimated_tokens: int
    chunks: list[VideoChunk]
    is_truncated: bool = False

    @property
    def needs_split(self) -> bool:
        return len(self.chunks) > 1


class VideoContextPort(Protocol):
    """Hợp đồng cung cấp ngữ cảnh video cho khâu dịch."""

    def plan_chunks(self, video_path: Path) -> VideoContextPlan:
        """Lập kế hoạch cắt video thành các đoạn vừa giới hạn token/dung lượng."""
        ...

    def upload_chunk(self, chunk: VideoChunk) -> RemoteVideoRef:
        """Tải một đoạn lên đám mây (tái dùng nếu đã có), trả tham chiếu từ xa."""
        ...

    def cleanup_local_chunks(self, plan: VideoContextPlan) -> None:
        """Xoá các file đoạn tạm trên đĩa sau khi đã tải lên."""
        ...
