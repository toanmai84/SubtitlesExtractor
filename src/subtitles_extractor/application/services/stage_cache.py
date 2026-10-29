"""[v3.23.608] Quản lý bộ nhớ đệm từng giai đoạn dịch (màn "Quản lý giai đoạn").

Cho người dùng:

* Xem giai đoạn nào đã lưu, dịch bằng model nào, lúc nào, bao nhiêu dòng sửa tay, và
  LẦN DỊCH TỚI sẽ làm gì với nó (dùng lại nguyên, chỉ dịch các lô có dòng đổi, dịch lại
  — kèm lý do) theo đúng cấu hình đang chọn — :meth:`StageCacheService.overview`.
* Xem và sửa từng dòng của một giai đoạn — :meth:`StageCacheService.editor_rows`,
  :meth:`StageCacheService.save_edits`. Dòng sửa tay luôn được giữ; các giai đoạn sau chỉ
  dịch lại các lô có dòng đổi.
* Xoá một giai đoạn (giữ hoặc bỏ dòng sửa tay) hay xoá toàn bộ.

Dự đoán "lần dịch tới" dùng CHÍNH các quy tắc của use case dịch
(:func:`stage_checkpoint.stage_reuse_verdict`, :meth:`StageCheckpoint.carry_over`) nên
khớp với những gì thực sự xảy ra.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from subtitles_extractor.application.services.model_capability import (
    result_reusable_for,
)
from subtitles_extractor.application.services.stage_checkpoint import (
    PIPELINE_ORDER,
    CheckpointIdentity,
    Clock,
    ReuseVerdict,
    StageCheckpoint,
    StageEditResult,
    StageNotCachedError,
    StageRun,
    legacy_checkpoint_key,
    line_input_digest,
    stage_config_signature,
    stage_input_fingerprint,
)
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    TranslationLine,
    TranslationStageConfig,
    TranslationStageKind,
)

logger = logging.getLogger(__name__)

__all__ = [
    "STAGE_LABELS",
    "StageAction",
    "StageCacheEntry",
    "StageCacheOverview",
    "StageCacheScope",
    "StageCacheService",
    "StageEditOutcome",
    "StageEditorRow",
    "StageReason",
    "adopt_previous_checkpoints",
]

#: Tên ngắn của từng giai đoạn cho giao diện.
STAGE_LABELS: dict[TranslationStageKind, str] = {
    TranslationStageKind.PREPROCESS: "Tiền xử lý",
    TranslationStageKind.LITERAL: "Dịch thô",
    TranslationStageKind.STYLE: "Tinh chỉnh",
    TranslationStageKind.LOCALIZE: "Bản địa hoá",
}


class StageAction(StrEnum):
    """Việc lần dịch tới sẽ làm với một giai đoạn."""

    REUSE = "reuse"
    PARTIAL = "partial"
    RETRANSLATE = "retranslate"
    TRANSLATE = "translate"
    SKIPPED = "skipped"
    ABSENT = "absent"


class StageReason(StrEnum):
    """Vì sao một giai đoạn không được dùng lại nguyên (giá trị như ``ReuseVerdict``)."""

    NONE = "reuse"
    NOT_CACHED = "missing"
    NO_RUN_INFO = "no_run_info"
    CONFIG_CHANGED = "config_changed"
    INPUT_CHANGED = "input_changed"
    UNVERIFIED_INPUT = "unverified_input"
    WEAKER_MODEL = "weaker_model"
    CONTEXT_CHANGED = "context_changed"
    UPSTREAM_PENDING = "upstream_pending"
    LINE_COUNT_CHANGED = "line_count_changed"


@dataclass(frozen=True)
class StageCacheScope:
    """Bộ phụ đề + cấu hình dịch đang chọn — đủ để mở bộ nhớ đệm và dự đoán lần dịch tới.

    Attributes:
        checkpoint_dir: Thư mục bộ nhớ đệm.
        identity: Định danh file (:class:`CheckpointIdentity`).
        source_lines: Câu gốc dạng dòng dịch (chỉ số từ 1).
        stages: Các giai đoạn đang bật, theo thứ tự chạy.
        attach_video_stages: Giai đoạn được đính video.
        context_fingerprint: Vân tay bảng nhân vật/thuật ngữ đang dùng; rỗng = không so.
    """

    checkpoint_dir: Path
    identity: CheckpointIdentity
    source_lines: tuple[TranslationLine, ...]
    stages: tuple[TranslationStageConfig, ...]
    attach_video_stages: frozenset[TranslationStageKind] = frozenset()
    context_fingerprint: str = ""
    #: [v3.23.656] Visual Cues của yêu cầu — dòng có người nói NGƯỜI DÙNG chọn là đầu vào khác (như lượt dịch thật).
    visual_cues: str = ""

    @property
    def key(self) -> str:
        """Khoá file hiện hành."""
        return self.identity.key

    @property
    def previous_key(self) -> str:
        """Khoá file của bản v3.23.606-607."""
        return self.identity.previous_key

    @property
    def legacy_key(self) -> str:
        """Khoá file của bản ≤ v3.23.605."""
        return legacy_checkpoint_key(
            self.identity, self.stages, self.attach_video_stages
        )

    def run_for(
        self, stage: TranslationStageConfig, input_lines: list[TranslationLine]
    ) -> StageRun:
        """Điều kiện chạy giai đoạn ``stage`` trên ``input_lines`` với cấu hình này."""
        return StageRun.for_stage(
            stage, input_lines, attach_video=stage.kind in self.attach_video_stages
        )


@dataclass(frozen=True)
class StageCacheEntry:
    """Trạng thái một giai đoạn trong bộ nhớ đệm và dự đoán cho lần dịch tới.

    Attributes:
        kind: Giai đoạn.
        label: Tên hiển thị.
        enabled: Giai đoạn đang bật trong cấu hình.
        stored: Có kết quả đã lưu.
        line_count: Số dòng đã lưu.
        stored_model: Model đã dịch kết quả đã lưu (rỗng nếu không rõ).
        requested_model: Model đang chọn cho giai đoạn (rỗng nếu giai đoạn tắt).
        completed_at: Thời điểm lưu (ISO, UTC; rỗng nếu không rõ).
        edited_lines: Số dòng sửa tay.
        partial_lines: Số dòng của lần chạy dở (0 nếu không có).
        partial_model: Model đã dịch phần chạy dở.
        action: Việc lần dịch tới sẽ làm.
        reason: Lý do không dùng lại nguyên (``StageReason.NONE`` khi dùng lại).
        reusable_lines: Số dòng giữ lại khi ``action`` là PARTIAL; -1 = chưa biết trước
            (phụ thuộc giai đoạn trước dịch ra gì).
        total_lines: Số dòng của phụ đề.
    """

    kind: TranslationStageKind
    label: str
    enabled: bool
    stored: bool
    line_count: int
    stored_model: str
    requested_model: str
    completed_at: str
    edited_lines: int
    partial_lines: int
    partial_model: str
    action: StageAction
    reason: StageReason
    reusable_lines: int
    total_lines: int


@dataclass(frozen=True)
class StageCacheOverview:
    """Toàn cảnh bộ nhớ đệm của một bộ phụ đề.

    Attributes:
        key: Khoá file.
        exists: Có bất kỳ dữ liệu nào trên đĩa.
        total_lines: Số dòng phụ đề.
        context_changed: Bảng nhân vật/thuật ngữ đã đổi so với lúc lưu.
        entries: Bốn giai đoạn theo thứ tự pipeline.
        updated_at: Lần ghi gần nhất (ISO, UTC; rỗng nếu không rõ).
    """

    key: str
    exists: bool
    total_lines: int
    context_changed: bool
    entries: tuple[StageCacheEntry, ...]
    updated_at: str

    @property
    def stored_kinds(self) -> list[TranslationStageKind]:
        """Các giai đoạn có kết quả đã lưu."""
        return [entry.kind for entry in self.entries if entry.stored]

    def entry(self, kind: TranslationStageKind) -> StageCacheEntry:
        """Trạng thái của một giai đoạn."""
        return next(entry for entry in self.entries if entry.kind is kind)


@dataclass(frozen=True)
class StageEditorRow:
    """Một dòng trong trình sửa giai đoạn.

    Attributes:
        index: Chỉ số dòng (từ 1).
        start_ms: Mốc bắt đầu.
        end_ms: Mốc kết thúc.
        source_text: Câu gốc.
        input_text: Đầu vào của giai đoạn (bản của giai đoạn trước, hoặc câu gốc).
        text: Nội dung hiện tại của giai đoạn.
        model_text: Câu model dịch ra (khác ``text`` khi dòng đã sửa tay).
        edited: Dòng đã sửa tay.
    """

    index: int
    start_ms: int
    end_ms: int
    source_text: str
    input_text: str
    text: str
    model_text: str
    edited: bool


@dataclass(frozen=True)
class StageEditOutcome:
    """Kết quả lưu sửa đổi một giai đoạn.

    Attributes:
        result: Các dòng đã đổi / hoàn tác / không tồn tại.
        downstream: Giai đoạn đã lưu phía sau — lần dịch tới sẽ dịch lại các lô có dòng
            đổi của chúng.
    """

    result: StageEditResult
    downstream: tuple[TranslationStageKind, ...]


def adopt_previous_checkpoints(
    checkpoint: StageCheckpoint, scope: StageCacheScope, *, context: str
) -> list[str]:
    """Nhận file bộ nhớ đệm do bản cũ ghi (khoá khác) cho cùng bộ phụ đề.

    * v3.23.606-607: khoá chứa model quét Visual Cues (= model giai đoạn đầu) — nhận
      nguyên vì đã có đủ điều kiện dùng lại từng giai đoạn.
    * ≤ v3.23.605: khoá chứa cấu hình mọi giai đoạn; khoá khớp nghĩa là lần chạy dở trước
      dùng ĐÚNG cấu hình đang chọn, nên các giai đoạn được ghi kèm :class:`StageRun` của
      cấu hình này (vân tay đầu vào tính lại theo chuỗi từ câu gốc).

    Args:
        checkpoint: Bộ nhớ đệm theo khoá hiện hành (chỉ nhận khi còn trống).
        scope: Phụ đề + cấu hình.
        context: Vân tay ngữ cảnh để mở file cũ (rỗng = không so).

    Returns:
        Các giai đoạn đã nhận (theo ``kind.value``).
    """
    if not checkpoint.is_empty:
        return []
    if scope.previous_key != scope.key:
        previous = StageCheckpoint(scope.checkpoint_dir, scope.previous_key, context)
        if not previous.is_empty:
            adopted = checkpoint.adopt(previous)
            logger.info(
                "Chuyển bộ nhớ đệm dịch của bản 3.23.606-607 sang khoá mới (khoá cũ chứa "
                "model quét Visual Cues): %d giai đoạn (%s).",
                len(adopted), ", ".join(adopted) or "không có",
            )
            return adopted
    legacy = StageCheckpoint(scope.checkpoint_dir, scope.legacy_key, context)
    if legacy.is_empty:
        return []
    runs: dict[str, StageRun] = {}
    previous_lines: list[TranslationLine] | None = list(scope.source_lines)
    for stage in scope.stages:
        input_fingerprint = (
            stage_input_fingerprint(stage.kind, previous_lines)
            if previous_lines is not None
            else ""
        )
        runs[stage.kind.value] = StageRun(
            input_fingerprint=input_fingerprint,
            config_signature=stage_config_signature(
                stage, stage.kind in scope.attach_video_stages
            ),
            model_name=stage.model_name,
        )
        previous_lines = legacy.get_stage(stage.kind)
    adopted = checkpoint.adopt_legacy(legacy, runs)
    logger.info(
        "Chuyển bộ nhớ đệm dịch của bản cũ sang khoá mới: %d giai đoạn đã xong (%s), "
        "tiến độ dở: %s.",
        len(adopted), ", ".join(adopted) or "không có",
        "có" if checkpoint.has_partial else "không",
    )
    return adopted


@dataclass(frozen=True)
class _Prediction:
    action: StageAction
    reason: StageReason
    reusable_lines: int
    output: list[TranslationLine] | None
    chain_intact: bool


class StageCacheService:
    """Đọc, sửa, xoá bộ nhớ đệm từng giai đoạn của một bộ phụ đề.

    Args:
        clock: Đồng hồ (để test); mặc định giờ UTC hiện tại.
    """

    def __init__(self, clock: Clock | None = None) -> None:
        self._clock = clock

    def open(self, scope: StageCacheScope) -> StageCheckpoint:
        """Mở bộ nhớ đệm (không so ngữ cảnh, không xoá gì), nhận file khoá cũ nếu có."""
        checkpoint = StageCheckpoint(scope.checkpoint_dir, scope.key, clock=self._clock)
        adopt_previous_checkpoints(checkpoint, scope, context="")
        return checkpoint

    # ── Xem ──────────────────────────────────────────────────────────────────
    def overview(self, scope: StageCacheScope) -> StageCacheOverview:
        """Trạng thái từng giai đoạn và việc lần dịch tới (cấu hình ``scope``) sẽ làm.

        Args:
            scope: Phụ đề + cấu hình đang chọn.

        Returns:
            Toàn cảnh bộ nhớ đệm.
        """
        checkpoint = self.open(scope)
        total = len(scope.source_lines)
        context_changed = bool(
            checkpoint.stored_context
            and scope.context_fingerprint
            and checkpoint.stored_context != scope.context_fingerprint
        )
        partial = checkpoint.partial_progress()
        enabled = {stage.kind: stage for stage in scope.stages}
        current: list[TranslationLine] | None = list(scope.source_lines)
        chain_intact = True
        entries: list[StageCacheEntry] = []
        for kind in PIPELINE_ORDER:
            lines = checkpoint.get_stage(kind)
            stored_run = checkpoint.stored_run(kind)
            stage = enabled.get(kind)
            partial_here = partial is not None and partial.kind is kind
            edited = checkpoint.edited_line_count(kind)
            if stage is None:
                prediction = _Prediction(
                    StageAction.SKIPPED if lines or partial_here or edited
                    else StageAction.ABSENT,
                    StageReason.NONE, 0, current, chain_intact,
                )
            else:
                if current is not None and scope.visual_cues:
                    # [v3.23.656] B8: lượt dịch thật gắn người nói người dùng chọn vào dòng ở MỖI giai đoạn trước khi
                    # so vân tay — dự đoán phải làm y hệt, không thì báo "dùng lại hết" dù lượt tới sẽ dịch lại vài lô.
                    from subtitles_extractor.application.services.visual_cue_serializer import tag_user_cues

                    current = tag_user_cues(current, scope.visual_cues)
                prediction = self._predict(
                    checkpoint, scope, stage, lines, current,
                    chain_intact=chain_intact, context_changed=context_changed,
                )
                current = prediction.output
                chain_intact = prediction.chain_intact
            entries.append(
                StageCacheEntry(
                    kind=kind,
                    label=STAGE_LABELS[kind],
                    enabled=stage is not None,
                    stored=lines is not None,
                    line_count=len(lines or []),
                    stored_model=stored_run.model_name if stored_run else "",
                    requested_model=stage.model_name if stage else "",
                    completed_at=checkpoint.stage_completed_at(kind),
                    edited_lines=edited,
                    partial_lines=partial.line_count if partial_here and partial else 0,
                    partial_model=partial.model_name if partial_here and partial else "",
                    action=prediction.action,
                    reason=prediction.reason,
                    reusable_lines=prediction.reusable_lines,
                    total_lines=total,
                )
            )
        return StageCacheOverview(
            key=scope.key,
            exists=checkpoint.exists_on_disk,
            total_lines=total,
            context_changed=context_changed,
            entries=tuple(entries),
            updated_at=str(checkpoint.meta.get("updated_at") or ""),
        )

    @staticmethod
    def _predict(
        checkpoint: StageCheckpoint,
        scope: StageCacheScope,
        stage: TranslationStageConfig,
        lines: list[TranslationLine] | None,
        current: list[TranslationLine] | None,
        *,
        chain_intact: bool,
        context_changed: bool,
    ) -> _Prediction:
        """Mô phỏng đúng quyết định của use case dịch cho một giai đoạn đang bật."""
        kind = stage.kind
        if context_changed:
            action = StageAction.RETRANSLATE if lines else StageAction.TRANSLATE
            return _Prediction(action, StageReason.CONTEXT_CHANGED, 0, None, False)
        if current is None:
            return StageCacheService._predict_unknown_input(
                checkpoint, scope, stage, lines
            )
        run = scope.run_for(stage, current)
        verdict = checkpoint.reuse_verdict(kind, run, chain_intact=chain_intact)
        if verdict is ReuseVerdict.REUSE and lines is not None:
            if len(lines) == len(scope.source_lines):
                return _Prediction(
                    StageAction.REUSE, StageReason.NONE, len(lines), lines, chain_intact
                )
            reason = StageReason.LINE_COUNT_CHANGED
        else:
            reason = StageReason(verdict.value)
        carried = checkpoint.carry_over(kind, run, current)
        resumed = checkpoint.get_partial(kind, run.input_fingerprint, run) or []
        kept = {line.index for line in carried} | {line.index for line in resumed}
        if kept:
            return _Prediction(StageAction.PARTIAL, reason, len(kept), None, False)
        action = StageAction.RETRANSLATE if lines is not None else StageAction.TRANSLATE
        return _Prediction(action, reason, 0, None, False)

    @staticmethod
    def _predict_unknown_input(
        checkpoint: StageCheckpoint,
        scope: StageCacheScope,
        stage: TranslationStageConfig,
        lines: list[TranslationLine] | None,
    ) -> _Prediction:
        """Giai đoạn trước sẽ dịch lại → chưa biết đầu vào; xét cấu hình và model."""
        if lines is None:
            return _Prediction(
                StageAction.TRANSLATE, StageReason.NOT_CACHED, 0, None, False
            )
        stored = checkpoint.stored_run(stage.kind)
        if stored is None:
            return _Prediction(
                StageAction.RETRANSLATE, StageReason.NO_RUN_INFO, 0, None, False
            )
        run = scope.run_for(stage, [])
        if stored.config_signature not in run.config_signatures:
            return _Prediction(
                StageAction.RETRANSLATE, StageReason.CONFIG_CHANGED, 0, None, False
            )
        if not result_reusable_for(stored.model_name, stage.model_name):
            return _Prediction(
                StageAction.RETRANSLATE, StageReason.WEAKER_MODEL, 0, None, False
            )
        if not checkpoint.input_digests(stage.kind):
            return _Prediction(
                StageAction.RETRANSLATE, StageReason.UPSTREAM_PENDING, 0, None, False
            )
        return _Prediction(
            StageAction.PARTIAL, StageReason.UPSTREAM_PENDING, -1, None, False
        )

    def stage_lines(
        self, scope: StageCacheScope, kind: TranslationStageKind
    ) -> list[TranslationLine]:
        """Kết quả đã lưu của một giai đoạn.

        Raises:
            StageNotCachedError: Giai đoạn chưa có trong bộ nhớ đệm.
        """
        lines = self.open(scope).get_stage(kind)
        if lines is None:
            raise StageNotCachedError(kind)
        return lines

    def editor_rows(
        self, scope: StageCacheScope, kind: TranslationStageKind
    ) -> list[StageEditorRow]:
        """Các dòng của một giai đoạn kèm câu gốc, đầu vào và câu model (để sửa).

        Raises:
            StageNotCachedError: Giai đoạn chưa có trong bộ nhớ đệm.
        """
        checkpoint = self.open(scope)
        lines = checkpoint.get_stage(kind)
        if lines is None:
            raise StageNotCachedError(kind)
        edits = checkpoint.stage_edits(kind)
        inputs = self._stage_input_texts(checkpoint, scope, kind)
        sources = {line.index: line.text for line in scope.source_lines}
        return [
            StageEditorRow(
                index=line.index,
                start_ms=line.start_ms,
                end_ms=line.end_ms,
                source_text=sources.get(line.index, line.original_text),
                input_text=inputs.get(line.index, ""),
                text=line.text,
                model_text=(
                    edits[line.index].model_text if line.index in edits else line.text
                ),
                edited=line.index in edits,
            )
            for line in lines
        ]

    @staticmethod
    def _stage_input_texts(
        checkpoint: StageCheckpoint,
        scope: StageCacheScope,
        kind: TranslationStageKind,
    ) -> dict[int, str]:
        """Đầu vào của giai đoạn lúc dịch: bản giai đoạn trước khớp vân tay, hoặc câu gốc.

        Giai đoạn trước có thể đang tắt hoặc đã bị sửa sau đó, nên chọn ứng viên khớp
        vân tay từng dòng nhiều nhất; file không ghi vân tay thì lấy giai đoạn liền trước.
        """
        position = PIPELINE_ORDER.index(kind)
        candidates: list[list[TranslationLine]] = []
        for earlier in reversed(PIPELINE_ORDER[:position]):
            earlier_lines = checkpoint.get_stage(earlier)
            if earlier_lines is not None:
                candidates.append(earlier_lines)
        candidates.append(list(scope.source_lines))
        digests = checkpoint.input_digests(kind)
        if digests:
            best = max(
                candidates,
                key=lambda candidate: sum(
                    1 for line in candidate
                    if digests.get(line.index) == line_input_digest(line.text)
                ),
            )
        else:
            best = candidates[0]
        return {line.index: line.text for line in best}

    # ── Sửa ──────────────────────────────────────────────────────────────────
    def save_edits(
        self,
        scope: StageCacheScope,
        kind: TranslationStageKind,
        texts: Mapping[int, str],
    ) -> StageEditOutcome:
        """Lưu câu người dùng sửa cho một giai đoạn.

        Args:
            scope: Phụ đề + cấu hình.
            kind: Giai đoạn.
            texts: ``chỉ số dòng → câu mới`` (câu trùng câu model = hoàn tác).

        Returns:
            Kết quả và các giai đoạn phía sau bị ảnh hưởng.

        Raises:
            StageNotCachedError: Giai đoạn chưa có trong bộ nhớ đệm.
        """
        checkpoint = self.open(scope)
        result = checkpoint.edit_stage(kind, texts)
        position = PIPELINE_ORDER.index(kind)
        downstream = tuple(
            later for later in checkpoint.stage_kinds()
            if PIPELINE_ORDER.index(later) > position
        )
        return StageEditOutcome(result, downstream if result.touched else ())

    def discard_edits(
        self,
        scope: StageCacheScope,
        kind: TranslationStageKind,
        indices: Iterable[int] | None = None,
    ) -> int:
        """Trả các dòng sửa tay về câu model (``None`` = mọi dòng). Trả số dòng."""
        return len(self.open(scope).discard_edits(kind, indices))

    # ── Xoá ──────────────────────────────────────────────────────────────────
    def delete_stage(
        self,
        scope: StageCacheScope,
        kind: TranslationStageKind,
        *,
        keep_edits: bool,
    ) -> None:
        """Xoá một giai đoạn đã lưu — lần dịch tới dịch lại nó (dòng sửa tay tuỳ chọn)."""
        self.open(scope).delete_stage(kind, keep_edits=keep_edits)

    def delete_all(self, scope: StageCacheScope) -> int:
        """Xoá toàn bộ bộ nhớ đệm của bộ phụ đề (kể cả file khoá của bản cũ).

        Returns:
            Số file bộ nhớ đệm đã xoá.
        """
        removed = 0
        for key in dict.fromkeys((scope.key, scope.previous_key, scope.legacy_key)):
            checkpoint = StageCheckpoint(scope.checkpoint_dir, key)
            if checkpoint.exists_on_disk:
                removed += 1
            checkpoint.delete()
        logger.info("Đã xoá %d file bộ nhớ đệm dịch của phụ đề hiện tại.", removed)
        return removed
