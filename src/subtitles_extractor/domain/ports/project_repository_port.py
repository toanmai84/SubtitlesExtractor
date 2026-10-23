"""Cổng (port) cho kho lưu trữ dự án Auto-Dubbing theo hash video."""

from __future__ import annotations

from abc import ABC, abstractmethod

from subtitles_extractor.domain.entities.project_record import ProjectRecord


class ProjectRepositoryPort(ABC):
    """Giao diện lưu/đọc/liệt kê/xoá các :class:`ProjectRecord`.

    Các adapter (vd SQLite) hiện thực giao diện này. Tầng nghiệp vụ và giao
    diện chỉ phụ thuộc vào port, không phụ thuộc công nghệ lưu trữ cụ thể.
    """

    @abstractmethod
    def get(self, video_hash: str) -> ProjectRecord | None:
        """Trả về dự án theo hash, hoặc None nếu chưa tồn tại."""

    @abstractmethod
    def save(self, record: ProjectRecord) -> None:
        """Tạo mới hoặc cập nhật dự án (upsert theo video_hash)."""

    @abstractmethod
    def list_all(self) -> list[ProjectRecord]:
        """Liệt kê tất cả dự án, mới cập nhật xếp trước."""

    def list_summaries(self) -> list[ProjectRecord]:
        """[v3.23.643] Như :meth:`list_all` nhưng KHÔNG nạp các cột lớn (phụ đề, dữ liệu OCR, cài đặt).

        Dành cho danh sách trên giao diện; muốn dữ liệu đầy đủ của một dự án thì dùng :meth:`get`.
        Mặc định rơi về :meth:`list_all` để các hiện thực cũ vẫn chạy đúng.
        """
        return self.list_all()

    @abstractmethod
    def delete(self, video_hash: str) -> None:
        """Xoá dự án theo hash (không lỗi nếu không tồn tại)."""

    @abstractmethod
    def close(self) -> None:
        """Đóng kết nối, dọn tài nguyên."""


__all__ = ["ProjectRepositoryPort"]
