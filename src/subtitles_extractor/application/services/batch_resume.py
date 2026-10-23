"""[v3.23.584] Tiếp tục một giai đoạn dịch DỞ: dùng lại các lô đã dịch xong ở lần trước.

Nhật ký thật (v3.23.583, phim "he thong sung the", 5540 dòng): lần dịch thứ nhất dừng vì
hết hạn mức ngày sau 43 lượt gọi, lần thứ hai dừng vì mạng chập chờn sau 25 lượt; lần thứ
ba phải dịch lại TỪ ĐẦU giai đoạn dịch thô (96 lượt). Checkpoint chỉ lưu sau khi CẢ giai
đoạn xong, nên 68 lượt gọi — 70% một giai đoạn — bị vứt bỏ, trong khi hạn mức ngày chính
là giới hạn của Gemini với người dùng gói miễn phí. Các nhật ký khác cũng có lần dịch đứt
giữa chừng vì 503 hay mất kết nối.

Module này chỉ là phần THUẦN của việc tiếp tục: tra dòng đã lưu theo chỉ số gốc. Lưu xuống
đĩa là việc của tầng ứng dụng (checkpoint của use case).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field, replace

from subtitles_extractor.domain.ports.subtitle_translator_port import TranslationLine


@dataclass(frozen=True)
class BatchResume:
    """Dòng đã dịch ở lần chạy dở, tra theo chỉ số GỐC của ``input_lines``.

    Attributes:
        saved: ``chỉ số gốc → dòng đã dịch``.
        reindex_map: ``chỉ số trong lô → chỉ số gốc`` (khi adapter đánh số lại liên tục
            sau khi bỏ dòng rác); rỗng nghĩa là hai cách đánh số trùng nhau.
    """

    saved: dict[int, TranslationLine] = field(default_factory=dict)
    reindex_map: dict[int, int] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        resume_lines: Iterable[TranslationLine] | None,
        reindex_map: dict[int, int] | None = None,
    ) -> BatchResume:
        """Dựng bảng tra từ các dòng đã lưu.

        Args:
            resume_lines: Dòng đã dịch ở lần chạy dở (chỉ số gốc); ``None`` = không có.
            reindex_map: ``chỉ số trong lô → chỉ số gốc``.

        Returns:
            Bảng tra (rỗng nếu không có gì để tiếp tục).
        """
        return cls(
            saved={line.index: line for line in resume_lines or ()},
            reindex_map=dict(reindex_map or {}),
        )

    def __bool__(self) -> bool:
        return bool(self.saved)

    def original_index(self, index: int) -> int:
        """Chỉ số gốc của một dòng trong lô."""
        return self.reindex_map.get(index, index)

    def lines_for(self, batch: list[TranslationLine]) -> list[TranslationLine] | None:
        """Bản dịch đã lưu cho CẢ lô, hoặc ``None`` nếu thiếu dù chỉ một dòng.

        Dòng đã lưu phải cùng bản gốc (``original_text``) với dòng đang dịch — chống dùng
        nhầm khi đầu vào đã khác (đổi phụ đề, đánh số lại).

        Args:
            batch: Các dòng của lô (chỉ số trong lô).

        Returns:
            Dòng đã dịch, mang chỉ số trong lô; ``None`` khi phải gọi API.
        """
        if not self.saved or not batch:
            return None
        restored: list[TranslationLine] = []
        for line in batch:
            saved = self.saved.get(self.original_index(line.index))
            if saved is None or (saved.original_text or "") != (line.original_text or ""):
                return None
            restored.append(replace(saved, index=line.index))
        return restored

    def to_original(self, lines: list[TranslationLine]) -> list[TranslationLine]:
        """Đổi chỉ số trong lô về chỉ số gốc (để lưu)."""
        if not self.reindex_map:
            return list(lines)
        return [replace(line, index=self.original_index(line.index)) for line in lines]


__all__ = ["BatchResume"]
