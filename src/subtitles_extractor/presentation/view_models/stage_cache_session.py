"""[v3.23.608] Phiên làm việc của hộp thoại "Quản lý giai đoạn".

Gắn MỘT bộ phụ đề + MỘT cấu hình dịch (đang chọn lúc mở hộp thoại) với dịch vụ bộ nhớ
đệm, và báo ViewModel khi nội dung một giai đoạn đổi (để bảng so sánh, phiên đã lưu và
bản dịch hiện hành theo kịp). Không phụ thuộc Qt — hộp thoại chỉ gọi các phương thức này.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping

from subtitles_extractor.application.services.stage_cache import (
    StageCacheOverview,
    StageCacheScope,
    StageCacheService,
    StageEditorRow,
    StageEditOutcome,
)
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    TranslationLine,
    TranslationStageKind,
)

__all__ = ["StageCacheSession", "StageLinesCallback"]

#: Callback nhận ``(giai đoạn, các dòng hiện tại của giai đoạn)``.
StageLinesCallback = Callable[[TranslationStageKind, list[TranslationLine]], None]


class StageCacheSession:
    """Cầu nối hộp thoại Quản lý giai đoạn ↔ :class:`StageCacheService`.

    Args:
        service: Dịch vụ bộ nhớ đệm.
        scope: Phụ đề + cấu hình dịch đang chọn.
        on_stage_edited: Gọi sau khi nội dung một giai đoạn đổi (sửa tay / hoàn tác).
        on_apply_stage: Gọi khi người dùng chọn dùng một giai đoạn làm bản dịch hiện hành.
    """

    def __init__(
        self,
        service: StageCacheService,
        scope: StageCacheScope,
        *,
        on_stage_edited: StageLinesCallback,
        on_apply_stage: StageLinesCallback,
    ) -> None:
        self._service = service
        self._scope = scope
        self._on_stage_edited = on_stage_edited
        self._on_apply_stage = on_apply_stage

    @property
    def scope(self) -> StageCacheScope:
        """Phạm vi (phụ đề + cấu hình) của phiên."""
        return self._scope

    def overview(self) -> StageCacheOverview:
        """Trạng thái các giai đoạn và dự đoán lần dịch tới."""
        return self._service.overview(self._scope)

    def editor_rows(self, kind: TranslationStageKind) -> list[StageEditorRow]:
        """Các dòng của một giai đoạn để xem/sửa.

        Raises:
            StageNotCachedError: Giai đoạn chưa có trong bộ nhớ đệm.
        """
        return self._service.editor_rows(self._scope, kind)

    def save_edits(
        self, kind: TranslationStageKind, texts: Mapping[int, str]
    ) -> StageEditOutcome:
        """Lưu câu sửa của một giai đoạn và báo ViewModel nếu có dòng đổi.

        Raises:
            StageNotCachedError: Giai đoạn chưa có trong bộ nhớ đệm.
        """
        outcome = self._service.save_edits(self._scope, kind, texts)
        if outcome.result.touched:
            self._notify_edited(kind)
        return outcome

    def discard_edits(
        self, kind: TranslationStageKind, indices: Iterable[int] | None = None
    ) -> int:
        """Trả dòng sửa tay về câu model (``None`` = mọi dòng). Trả số dòng."""
        reverted = self._service.discard_edits(self._scope, kind, indices)
        if reverted:
            self._notify_edited(kind)
        return reverted

    def delete_stage(self, kind: TranslationStageKind, *, keep_edits: bool) -> None:
        """Xoá bản lưu của một giai đoạn (lần dịch tới dịch lại nó)."""
        self._service.delete_stage(self._scope, kind, keep_edits=keep_edits)

    def delete_all(self) -> int:
        """Xoá toàn bộ bộ nhớ đệm của phụ đề. Trả số file đã xoá."""
        return self._service.delete_all(self._scope)

    def apply_stage(self, kind: TranslationStageKind) -> int:
        """Dùng nội dung đã lưu của giai đoạn làm bản dịch hiện hành. Trả số dòng.

        Raises:
            StageNotCachedError: Giai đoạn chưa có trong bộ nhớ đệm.
        """
        lines = self._service.stage_lines(self._scope, kind)
        self._on_apply_stage(kind, lines)
        return len(lines)

    def _notify_edited(self, kind: TranslationStageKind) -> None:
        self._on_stage_edited(kind, self._service.stage_lines(self._scope, kind))
