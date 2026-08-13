"""Use case: dịch danh sách ``SubtitleEvent`` qua pipeline đa giai đoạn.

Điều phối thứ tự các giai đoạn (tiền xử lý → dịch thô → phong cách → bản địa hoá),
chuyển đổi giữa ``SubtitleEvent`` (domain) và ``TranslationLine`` (trung gian),
nối đầu ra giai đoạn trước làm đầu vào giai đoạn sau, và báo tiến độ tổng.

**Resume checkpoint (Stage-level)**:
Nếu ``checkpoint_dir`` được cung cấp, sau mỗi giai đoạn hoàn tất use case lưu
kết quả xuống ``{checkpoint_dir}/{key}.json``. Lần chạy tiếp theo cùng request
(cùng source + cấu hình) sẽ đọc checkpoint và bỏ qua giai đoạn đã xong — không
dịch lại từ đầu khi bị ngắt giữa chừng.

Tầng này KHÔNG biết Gemini là gì — chỉ phụ thuộc ``SubtitleTranslatorPort``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    SubtitleTranslationError,
    SubtitleTranslatorPort,
    TranslationCancelledError,
    TranslationContext,
    TranslationLine,
    TranslationStageConfig,
    TranslationStageKind,
    apply_visual_cues_to_lines,
)
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval

logger = logging.getLogger(__name__)

# Callback tiến độ tổng: (phần_trăm_0_đến_1, mô_tả_giai_đoạn).
OverallProgressCallback = Callable[[float, str], None]
# Callback kiểm tra huỷ.
CancellationCallback = Callable[[], bool]

# [v3.23.350] Ngưỡng tỉ lệ dòng trùng-hệt-nguồn để coi cả lần dịch là "không tác dụng".
# Đặt cao (0.98) để KHÔNG báo oan: bản dịch tốt vẫn có thể giữ nguyên vài dòng là tên
# riêng/số/thán từ ("OK", "Alice", "911"). Chỉ khi gần như MỌI dòng trùng mới cảnh báo.
_PASSTHROUGH_WARN_RATIO = 0.98


@dataclass(frozen=True)
class TranslateSubtitlesRequest:
    """Đầu vào cho use case dịch.

    Attributes:
        events:  Danh sách phụ đề gốc cần dịch (đã sắp xếp theo thời gian).
        stages:  Danh sách giai đoạn theo đúng thứ tự thực thi (đã lọc bật/tắt).
        context: Ngữ cảnh toàn cục dùng chung.
    """

    events: list[SubtitleEvent]
    stages: list[TranslationStageConfig]
    context: TranslationContext
    # Ngữ cảnh video tuỳ chọn: danh sách đoạn đã tải lên + tập giai đoạn được đính
    # video. Phân tích ngữ cảnh xử lý riêng ở use case phân tích (gộp mọi đoạn);
    # ở đây chỉ áp cho các giai đoạn dịch (LITERAL/STYLE/LOCALIZE…) được chọn.
    video_refs: tuple[Any, ...] = ()
    attach_video_stages: frozenset[TranslationStageKind] = frozenset()
    # [v3.23.31] Visual Cues (Vision Director): nếu bật VÀ có video_refs, quét video
    # MỘT LẦN trước khi dịch để xác định ai nói / nói với ai từng dòng, rồi bơm vào
    # speaker/addressee giúp dịch xưng hô đúng vai vế. TỐN TOKEN (mỗi ~150 dòng 1 lần
    # gọi) nên là tuỳ chọn người dùng tự bật.
    enable_visual_cues: bool = False
    visual_cues_model: str = "gemini-flash-lite-latest"
    visual_cues_batch_size: int = 150


@dataclass(frozen=True)
class TranslateSubtitlesResponse:
    """Kết quả dịch.

    Attributes:
        events:           Phụ đề sau khi dịch (giữ uid/timing gốc).
        completed_stages: Các giai đoạn đã chạy thành công, theo thứ tự.
        resumed_from:     Giai đoạn resume đầu tiên (None nếu chạy mới).
        elapsed_seconds:  Tổng thời gian thực thi (giây).
    """

    events: list[SubtitleEvent]
    completed_stages: list[TranslationStageKind]
    resumed_from: TranslationStageKind | None = None
    elapsed_seconds: float = 0.0
    # [v3.23.47] Kết quả văn bản sau MỖI giai đoạn (để so sánh giữa các giai đoạn).
    # Khoá = TranslationStageKind, giá trị = danh sách SubtitleEvent của giai đoạn đó.
    stage_outputs: dict[TranslationStageKind, list[SubtitleEvent]] = field(
        default_factory=dict
    )


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _compute_checkpoint_key(request: TranslateSubtitlesRequest) -> str:
    """Hash ngắn xác định duy nhất một yêu cầu dịch để dùng làm tên file checkpoint.

    Bao gồm ``source_lang`` từ context để phân biệt checkpoint khi cùng bộ phụ đề
    nhưng ngôn ngữ gốc khác nhau (vd: zh → vi vs ko → vi).
    """
    source_texts = [e.text for e in request.events]
    # [v3.23.135] Khóa phải bao gồm MỌI cài đặt người dùng kiểm soát mà ẢNH HƯỞNG kết quả
    # dịch — nếu thiếu, đổi phong cách/locale/thinking/ngữ cảnh-size/visual-cues rồi chạy
    # lại sẽ bị RESUME nhầm bản dịch CŨ. Cố ý KHÔNG đưa ngữ cảnh tự-phân-tích (nhân vật/
    # tóm tắt) vào khóa để checkpoint còn dùng được khi tiếp tục một lần chạy dở.
    stage_ids = [
        f"{s.kind.value}:{s.model_name}:{s.batch_size}:{s.temperature:.2f}"
        f":ctx{s.context_size}:st={s.style_name or ''}:loc={s.locale_notes or ''}"
        f":th={s.enable_thinking}/{s.thinking_budget}/{s.thinking_level or ''}"
        f":rt={s.allow_retime}"
        for s in request.stages
    ]
    payload = json.dumps(
        {
            "texts": source_texts,
            "stages": stage_ids,
            "target": request.context.target_lang,
            "source": request.context.source_lang or "",   # [Q1] phân biệt ngôn ngữ gốc
            # Phân biệt checkpoint khi có đính video (kết quả dịch khác bản text-only).
            "attach_video": sorted(s.value for s in request.attach_video_stages),
            # [v3.23.135] Visual Cues làm giàu speaker/addressee -> đổi kết quả dịch.
            "visual_cues": (
                [request.visual_cues_model, request.visual_cues_batch_size]
                if request.enable_visual_cues
                else False
            ),
        },
        ensure_ascii=False, sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:20]


def _lines_to_json(lines: list[TranslationLine]) -> list[dict]:
    # [v3.23.81] Lưu ĐỦ trường, đặc biệt ``o`` (original_text - mỏ neo bản gốc bất
    # biến chống "tam sao thất bản") và ``a`` (addressee - chọn đại từ/kính ngữ). Thiếu
    # chúng, bản dịch RESUME mất mỏ neo nguồn ở STYLE/LOCALIZE -> trôi nghĩa.
    return [
        {"i": l.index, "s": l.start_ms, "e": l.end_ms,
         "t": l.text, "sp": l.speaker, "d": l.description,
         "o": l.original_text, "a": l.addressee, "c": l.scene}
        for l in lines
    ]


def _json_to_lines(data: list[dict]) -> list[TranslationLine]:
    # ``o``/``a``/``c`` dùng .get(default="") để TƯƠNG THÍCH NGƯỢC với checkpoint cũ.
    return [
        TranslationLine(
            index=row["i"], start_ms=row["s"], end_ms=row["e"],
            text=row["t"], speaker=row.get("sp", ""), description=row.get("d", ""),
            original_text=row.get("o", ""), addressee=row.get("a", ""),
            scene=row.get("c", ""),
        )
        for row in data
    ]


def _measure_passthrough_ratio(
    source_events: list[SubtitleEvent], translated_events: list[SubtitleEvent]
) -> tuple[int, int]:
    """Đếm số dòng KẾT QUẢ giống HỆT nguồn (đã trim) — dấu hiệu "dịch không tác dụng".

    Hàm thuần (không I/O, không trạng thái ngoài): chỉ so từng cặp ``text`` theo
    vị trí. Dùng cho guard cuối pipeline nhằm phát hiện trường hợp toàn bộ (hoặc gần
    như toàn bộ) bản dịch trùng bản gốc — thường do pipeline THIẾU khâu LITERAL,
    nguồn đã cùng ngôn ngữ đích, hoặc resume nhầm một checkpoint hỏng.

    Args:
        source_events:     Phụ đề gốc (đầu vào).
        translated_events: Phụ đề sau toàn bộ pipeline.

    Returns:
        Cặp ``(số_dòng_giữ_nguyên, tổng_số_dòng_so_được)``. Chỉ so các dòng có nội
        dung nguồn khác rỗng để tỉ lệ không bị nhiễu bởi dòng trống.
    """
    unchanged = 0
    comparable = 0
    for source, translated in zip(source_events, translated_events, strict=False):
        source_text = (source.text or "").strip()
        if not source_text:
            continue
        comparable += 1
        if source_text == (translated.text or "").strip():
            unchanged += 1
    return unchanged, comparable


class _StageCheckpoint:
    """Quản lý checkpoint stage-level trên disk.

    Format file ``{checkpoint_dir}/{key}.json``:
    ``{"key": "...", "stages": {"LITERAL": [lines...], ...}}``
    """

    def __init__(self, checkpoint_dir: Path, key: str) -> None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._path = checkpoint_dir / f"{key}.json"
        self._data: dict[str, list[dict]] = {}
        self._visual_cues: list[dict] | None = None
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                raw = json.loads(self._path.read_text(encoding="utf-8"))
                self._data = raw.get("stages", {})
                self._visual_cues = raw.get("visual_cues")
                logger.info("Đọc checkpoint dịch: %s (%d giai đoạn đã lưu).",
                            self._path.name, len(self._data))
            except (json.JSONDecodeError, OSError, KeyError) as exc:
                logger.warning("Không đọc được checkpoint %s: %s — bắt đầu mới.", self._path, exc)
                self._data = {}
                self._visual_cues = None

    def get_stage(self, kind: TranslationStageKind) -> list[TranslationLine] | None:
        """Trả kết quả đã lưu của giai đoạn, hoặc None nếu chưa có."""
        raw = self._data.get(kind.value)
        return _json_to_lines(raw) if raw is not None else None

    def get_visual_cues(self) -> list[dict] | None:
        """[v3.23.32] Trả Visual Cues đã cache (dạng dict rút gọn), None nếu chưa có."""
        return self._visual_cues

    def save_visual_cues(self, cues_json: list[dict]) -> None:
        """[v3.23.32] Cache Visual Cues để dịch lại không phải quét video lại."""
        self._visual_cues = cues_json
        self._flush()

    def save_stage(self, kind: TranslationStageKind, lines: list[TranslationLine]) -> None:
        """Lưu kết quả giai đoạn xuống disk."""
        self._data[kind.value] = _lines_to_json(lines)
        self._flush()

    def _flush(self) -> None:
        try:
            payload: dict[str, Any] = {"key": self._path.stem, "stages": self._data}
            if self._visual_cues is not None:
                payload["visual_cues"] = self._visual_cues
            self._path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=None),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning("Không lưu được checkpoint: %s — tiếp tục không checkpoint.", exc)

    def delete(self) -> None:
        """Xoá checkpoint sau khi dịch hoàn tất thành công."""
        try:
            if self._path.exists():
                self._path.unlink()
        except OSError as exc:
            logger.warning("Không xoá được checkpoint %s: %s", self._path, exc)

    @property
    def exists_on_disk(self) -> bool:
        return self._path.exists()


class TranslateSubtitlesUseCase:
    """Điều phối dịch phụ đề đa giai đoạn qua một ``SubtitleTranslatorPort``."""

    _STAGE_LABELS = {
        TranslationStageKind.PREPROCESS: "Giai đoạn 1: Tiền xử lý bản gốc",
        TranslationStageKind.LITERAL: "Giai đoạn 2: Dịch thô sát nghĩa",
        TranslationStageKind.STYLE: "Giai đoạn 3: Tinh chỉnh phong cách",
        TranslationStageKind.LOCALIZE: "Giai đoạn 4: Bản địa hoá",
    }

    def __init__(
        self,
        translator: SubtitleTranslatorPort,
        checkpoint_dir: Path | None = None,
    ) -> None:
        self._translator = translator
        self._checkpoint_dir = checkpoint_dir

    def ensure_viable_key(
        self,
        model_name: str,
        needed_requests: int = 1,
        *,
        avoid_reupload: bool = False,
    ) -> str | None:
        """[v3.23.145] Chọn API key còn quota ngày TRƯỚC khi upload video (uỷ quyền adapter).

        Cùng cơ chế như phân tích: tránh upload dưới key hết quota rồi phải xoay + tải lại.

        Args:
            model_name: Model của giai đoạn có đính video (mốc xét quota).
            needed_requests: Số request dự trù cho trọn phiên dịch (>= 1).
            avoid_reupload: [v3.23.294] True khi video đã upload dưới key hiện tại và
                phiên này tái dùng nó → adapter bỏ xoay key cơ hội, tránh upload lại.
        """
        fn = getattr(self._translator, "ensure_viable_key", None)
        if fn is None:
            return None
        try:
            return fn(model_name, needed_requests, avoid_reupload=avoid_reupload)
        except TypeError:
            # Adapter cũ chưa hỗ trợ tham số mới → fallback gọi tối giản (1 tham số).
            try:
                return fn(model_name)
            except (AttributeError, TypeError, ValueError):
                return None
        except (AttributeError, ValueError):
            return None

    def has_any_daily_quota(self, model_name: str) -> bool:
        """[v3.23.146] True nếu còn ÍT NHẤT một key có quota ngày (để fail nhanh nếu cạn)."""
        fn = getattr(self._translator, "has_any_daily_quota", None)
        if fn is None:
            return True
        try:
            return bool(fn(model_name))
        except (AttributeError, TypeError, ValueError):
            return True

    def execute(
        self,
        request: TranslateSubtitlesRequest,
        progress_cb: OverallProgressCallback | None = None,
        cancel_cb: CancellationCallback | None = None,
    ) -> TranslateSubtitlesResponse:
        """Chạy toàn bộ pipeline dịch, tự động resume từ checkpoint nếu có.

        Args:
            request:     Đầu vào (events, stages, context).
            progress_cb: Callback tiến độ tổng ``(0..1, mô tả)``.
            cancel_cb:   Callback kiểm tra huỷ; trả True để dừng sớm.

        Returns:
            ``TranslateSubtitlesResponse`` với danh sách event đã dịch.
        """
        if not request.events:
            raise SubtitleTranslationError("Không có phụ đề nào để dịch.")
        if not request.stages:
            raise SubtitleTranslationError("Chưa bật giai đoạn dịch nào.")

        # ── Khởi tạo checkpoint ──────────────────────────────────────────
        checkpoint: _StageCheckpoint | None = None
        if self._checkpoint_dir is not None:
            key = _compute_checkpoint_key(request)
            checkpoint = _StageCheckpoint(self._checkpoint_dir, key)

        source_lines = self._events_to_lines(request.events)

        # [v3.23.31] Visual Cues (tuỳ chọn): quét video MỘT LẦN để xác định ai nói/nói
        # với ai từng dòng, bơm vào speaker/addressee giúp dịch xưng hô đúng vai vế.
        # Chỉ chạy khi bật VÀ có video; lỗi/huỷ ở bước này KHÔNG làm hỏng dịch (bỏ qua).
        if request.enable_visual_cues and request.video_refs:
            source_lines = self._enrich_with_visual_cues(
                source_lines, request, checkpoint, progress_cb, cancel_cb
            )

        reference_lines = source_lines
        current_lines = source_lines
        completed: list[TranslationStageKind] = []
        stage_outputs: dict[TranslationStageKind, list[SubtitleEvent]] = {}
        has_prior_translation = False
        total_stages = len(request.stages)
        resumed_from: TranslationStageKind | None = None
        _start_time = time.monotonic()

        for stage_position, stage in enumerate(request.stages):
            if cancel_cb is not None and cancel_cb():
                raise TranslationCancelledError("Người dùng đã huỷ tiến trình dịch.")

            stage_label = self._STAGE_LABELS.get(stage.kind, stage.kind.value)

            # ── Thử resume từ checkpoint ─────────────────────────────────
            if checkpoint is not None:
                cached = checkpoint.get_stage(stage.kind)
                if cached is not None and len(cached) == len(source_lines):
                    logger.info("Resume checkpoint: bỏ qua %s (đã có %d dòng).",
                                stage_label, len(cached))
                    stage_output = cached
                    if resumed_from is None:
                        resumed_from = stage.kind
                    if progress_cb is not None:
                        overall = (stage_position + 1) / total_stages
                        progress_cb(overall, f"[Resume] {stage_label}")
                    # [v3.23.134] Giai đoạn RESUME cũng phải ghi vào stage_outputs như
                    # giai đoạn chạy thường — nếu không, màn "so sánh từng giai đoạn"
                    # (cần >=2 giai đoạn) sẽ THIẾU giai đoạn đã resume, mất khả năng đối
                    # chiếu trước/sau khi tiếp tục dịch sau khi huỷ/treo.
                    stage_outputs[stage.kind] = self._lines_to_events(
                        request.events, stage_output, request.context
                    )
                    current_lines = stage_output
                    completed.append(stage.kind)
                    if stage.kind is TranslationStageKind.PREPROCESS:
                        reference_lines = stage_output
                    else:
                        has_prior_translation = True
                    continue

            # ── Dịch bình thường ─────────────────────────────────────────
            logger.info("Bắt đầu %s với model '%s'.", stage_label, stage.model_name)

            def _stage_progress(
                fraction: float, _pos: int = stage_position, _label: str = stage_label
            ) -> None:
                if progress_cb is not None:
                    overall = (_pos + max(0.0, min(1.0, fraction))) / total_stages
                    progress_cb(overall, _label)

            stage_output = self._translator.translate_stage(
                stage=stage,
                context=request.context,
                source_lines=reference_lines,
                input_lines=current_lines,
                has_prior_translation=has_prior_translation,
                progress_cb=_stage_progress,
                cancel_cb=cancel_cb,
                video_refs=list(request.video_refs),
                attach_video=stage.kind in request.attach_video_stages,
            )

            # ── Lưu checkpoint sau khi giai đoạn hoàn tất ────────────────
            if checkpoint is not None:
                checkpoint.save_stage(stage.kind, stage_output)

            # [v3.23.47] Giữ kết quả giai đoạn (dạng events) để so sánh sau khi dịch xong.
            stage_outputs[stage.kind] = self._lines_to_events(
                request.events, stage_output, request.context
            )

            current_lines = stage_output
            completed.append(stage.kind)

            if stage.kind is TranslationStageKind.PREPROCESS:
                reference_lines = stage_output
            else:
                has_prior_translation = True

        translated_events = self._lines_to_events(
            request.events, current_lines, request.context
        )
        if progress_cb is not None:
            progress_cb(1.0, "Hoàn tất")

        # [v3.23.350] GUARD "dịch không tác dụng": trước đây khi toàn bộ bản dịch
        # trùng bản gốc (pipeline thiếu khâu LITERAL, nguồn đã cùng ngôn ngữ đích,
        # hoặc resume nhầm checkpoint hỏng), các khâu vẫn "thành công" và chỉ để lại
        # log INFO/WARNING rải rác → người dùng thấy "trả về bản gốc mà không báo lỗi".
        # Ở đây so KẾT QUẢ CUỐI với nguồn MỘT LẦN và cảnh báo NỔI BẬT kèm gợi ý.
        unchanged, comparable = _measure_passthrough_ratio(
            request.events, translated_events
        )
        if comparable > 0 and unchanged / comparable >= _PASSTHROUGH_WARN_RATIO:
            has_literal = any(
                s.kind is TranslationStageKind.LITERAL for s in request.stages
            )
            translating_stages = [
                s.kind for s in request.stages
                if s.kind is not TranslationStageKind.PREPROCESS
            ]
            if not translating_stages:
                likely_cause = (
                    "pipeline KHÔNG có khâu dịch nào (chỉ có Tiền xử lý)"
                )
            elif not has_literal:
                likely_cause = (
                    "pipeline THIẾU khâu 'Dịch thô sát nghĩa' (LITERAL) — các khâu "
                    "Tinh chỉnh/Bản địa hoá coi việc giữ nguyên bản nháp là HỢP LỆ nên "
                    "không đổi ngôn ngữ"
                )
            elif resumed_from is not None:
                likely_cause = (
                    f"đã RESUME từ checkpoint (khâu '{resumed_from.value}') — "
                    "checkpoint có thể lưu kết quả hỏng từ lần chạy trước; hãy xoá "
                    "checkpoint rồi dịch lại"
                )
            else:
                likely_cause = (
                    "model có thể liên tục trả về nguyên văn nguồn (xem WARNING "
                    "'giữ text gốc' ở tầng dịch phía trên)"
                )
            logger.warning(
                "DỊCH KHÔNG TÁC DỤNG: %d/%d dòng (%.0f%%) TRÙNG HỆT bản gốc. "
                "Nguyên nhân khả dĩ: %s.",
                unchanged, comparable, 100.0 * unchanged / comparable, likely_cause,
            )

        # ── Xoá checkpoint sau khi hoàn tất thành công ───────────────────
        if checkpoint is not None:
            checkpoint.delete()

        return TranslateSubtitlesResponse(
            events=translated_events,
            completed_stages=completed,
            resumed_from=resumed_from,
            elapsed_seconds=time.monotonic() - _start_time,
            stage_outputs=stage_outputs,
        )

    # ── Visual Cues (Vision Director) ────────────────────────────────────
    def _enrich_with_visual_cues(
        self,
        source_lines: list[TranslationLine],
        request: TranslateSubtitlesRequest,
        checkpoint: "_StageCheckpoint | None",
        progress_cb: OverallProgressCallback | None,
        cancel_cb: CancellationCallback | None,
    ) -> list[TranslationLine]:
        """Quét video để xác định ai nói/nói với ai, bơm vào speaker/addressee.

        An toàn-trước-tiên: mọi lỗi (trừ huỷ) đều được nuốt và trả về dòng gốc, để
        bước phụ trợ này KHÔNG bao giờ làm hỏng tiến trình dịch chính.
        """
        analyze = getattr(self._translator, "analyze_visual_cues", None)

        # [v3.23.35] ƯU TIÊN CAO NHẤT: dùng Visual Cues đã phân tích chung với ngữ cảnh
        # toàn cục (người dùng có thể đã xem/sửa) — không quét lại video.
        ctx_cues = (request.context.visual_cues or "").strip()
        if ctx_cues:
            from subtitles_extractor.application.services.visual_cue_serializer import (
                deserialize_visual_cues,
            )
            cues = deserialize_visual_cues(ctx_cues)
            if cues:
                logger.info(
                    "Dùng %d Visual Cues từ phân tích ngữ cảnh (đã có sẵn, không quét "
                    "lại video).", len(cues),
                )
                return apply_visual_cues_to_lines(source_lines, cues)

        if analyze is None:
            return source_lines

        # Kế đến: dùng cache trong checkpoint (cùng source+video) nếu có.
        if checkpoint is not None:
            cached = checkpoint.get_visual_cues()
            if cached is not None:
                from subtitles_extractor.application.services.visual_cue_serializer import (
                    deserialize_visual_cues,
                )
                cues = deserialize_visual_cues(json.dumps(cached))
                if cues:
                    logger.info(
                        "Dùng lại %d Visual Cues từ checkpoint (bỏ qua quét video).",
                        len(cues),
                    )
                    return apply_visual_cues_to_lines(source_lines, cues)

        if progress_cb is not None:
            progress_cb(0.0, "Phân tích hình ảnh (ai nói với ai)")
        try:
            cues = analyze(
                source_lines,
                request.context.target_lang,
                model_name=request.visual_cues_model,
                video_refs=list(request.video_refs),
                batch_size=request.visual_cues_batch_size,
                cancel_cb=cancel_cb,
            )
        except TranslationCancelledError:
            raise
        except SubtitleTranslationError as exc:
            logger.warning(
                "Phân tích Visual Cues thất bại — bỏ qua, dịch không có gợi ý hình "
                "ảnh: %s", exc,
            )
            return source_lines
        if not cues:
            return source_lines
        # Cache lại để lần dịch sau (hoặc resume) không phải quét video lại.
        if checkpoint is not None:
            from subtitles_extractor.application.services.visual_cue_serializer import (
                serialize_visual_cues,
            )
            checkpoint.save_visual_cues(json.loads(serialize_visual_cues(cues)))
        enriched = apply_visual_cues_to_lines(source_lines, cues)
        logger.info(
            "Đã bơm gợi ý hình ảnh cho %d/%d dòng (ai nói / nói với ai).",
            sum(1 for c in cues if c.speaker or c.addressee), len(source_lines),
        )
        return enriched

    # ── Chuyển đổi domain ↔ trung gian ───────────────────────────────────
    @staticmethod
    def _events_to_lines(events: list[SubtitleEvent]) -> list[TranslationLine]:
        return [
            TranslationLine(
                index=position,
                start_ms=int(round(event.start_sec * 1000)),
                end_ms=int(round(event.end_sec * 1000)),
                text=event.text,
            )
            for position, event in enumerate(events, start=1)
        ]

    @staticmethod
    def _lines_to_events(
        original_events: list[SubtitleEvent],
        translated_lines: list[TranslationLine],
        context: TranslationContext,
    ) -> list[SubtitleEvent]:
        result: list[SubtitleEvent] = []
        last_speaker = ""  # [v3.23.20] theo dõi người nói để chỉ tag khi ĐỔI người
        # [v3.23.25] Ánh xạ tên CJK → phiên âm (Hán Việt…) lấy từ roster phân tích,
        # để tag tên người nói CJK được phiên âm sang tiếng đích, Latin giữ nguyên.
        roster_map = TranslateSubtitlesUseCase._build_roster_pronunciation_map(
            context.characters or ""
        )
        # [v3.23.84] Map tên CHUẨN từ roster để chuẩn hoá tag người nói nhất quán
        # (đồng nhất cách viết + ánh xạ alias CJK -> tên đích).
        canonical_map = TranslateSubtitlesUseCase._build_canonical_name_map(
            context.characters or ""
        )
        for position, original in enumerate(original_events, start=1):
            line = translated_lines[position - 1] if position - 1 < len(translated_lines) else None
            new_text = line.text if line is not None else original.text
            display_text, last_speaker = TranslateSubtitlesUseCase._compose_display_text(
                line, new_text, context, last_speaker, roster_map, canonical_map
            )
            result.append(
                SubtitleEvent(
                    index=original.index,
                    text=display_text,
                    interval=TimeInterval(original.start_sec, original.end_sec),
                    confidence=original.confidence,
                    frame_count=original.frame_count,
                    position=original.position,
                    bounding_box=original.bounding_box,
                    uid=original.uid,
                )
            )
        return TranslateSubtitlesUseCase._suppress_repeated_speaker_tags(result)

    @classmethod
    def _suppress_repeated_speaker_tags(
        cls, events: list[SubtitleEvent]
    ) -> list[SubtitleEvent]:
        """[v3.23.137] Gỡ nhãn '[Người nói:]' ở ĐẦU dòng khi TRÙNG người nói dòng ngay
        trước (theo khoá chuẩn hoá), BẤT KỂ nhãn do app gắn hay do model tự chèn vào text.

        Vì model lite gán người nói KHÔNG nhất quán giữa các giai đoạn (lúc dùng trường
        'speaker', lúc chèn thẳng '[Tên:]' vào text), khử trùng chỉ dựa trên trường speaker
        (``_compose_display_text``) không đủ → nhãn bị lặp mỗi dòng dù cùng một người. Lượt
        hợp nhất này bảo đảm: chuỗi dòng liên tiếp CÙNG người chỉ hiện nhãn ở dòng đầu; dòng
        hiệu ứng âm thanh/nhạc NGẮT mạch (người nói kế sau được tag lại).
        """
        tag_re = re.compile(r"^\s*\[([^\]]+?):\]\s*")
        last_key = ""
        out: list[SubtitleEvent] = []
        for ev in events:
            text = ev.text
            if cls._is_sound_effect(text):
                last_key = ""  # ngắt mạch người nói qua hiệu ứng âm thanh/nhạc
                out.append(ev)
                continue
            match = tag_re.match(text)
            if match:
                key = cls._normalize_name_key(match.group(1))
                if key and key == last_key:
                    stripped = text[match.end():].lstrip()
                    # Chỉ gỡ khi PHẦN CÒN LẠI không rỗng (tránh tạo dòng trống).
                    if stripped:
                        out.append(replace(ev, text=stripped))
                        continue
                else:
                    last_key = key
            out.append(ev)
        return out

    # Người nói "rỗng" không nên tag: N/A, unknown, dấu chấm hỏi…
    _EMPTY_SPEAKERS = frozenset({
        "", "n/a", "na", "unknown", "không rõ", "khong ro", "?", "-", "none",
        "người nói", "speaker", "nhân vật", "narrator unknown",
    })

    # [v3.23.23] Dịch các nhãn người nói CHUNG sang tiếng Việt. Tên riêng (John,
    # Tiến sĩ Teven…) KHÔNG dịch. Khoá là dạng thường, đã bỏ ngoặc/dấu hai chấm.
    _GENERIC_SPEAKER_VI: dict[str, str] = {
        "man": "Người đàn ông", "woman": "Người phụ nữ",
        "boy": "Cậu bé", "girl": "Cô bé", "child": "Đứa trẻ", "kid": "Đứa trẻ",
        "narrator": "Người dẫn chuyện", "announcer": "Phát thanh viên",
        "reporter": "Phóng viên", "interviewer": "Người phỏng vấn",
        "host": "Người dẫn chương trình", "presenter": "Người dẫn chương trình",
        "crowd": "Đám đông", "audience": "Khán giả", "voice": "Giọng nói",
        "voiceover": "Lời dẫn", "male voice": "Giọng nam", "female voice": "Giọng nữ",
        "old man": "Ông lão", "old woman": "Bà lão",
        "doctor": "Bác sĩ", "teacher": "Giáo viên", "student": "Học sinh",
        "soldier": "Người lính", "officer": "Sĩ quan", "nurse": "Y tá",
        "all": "Mọi người", "both": "Cả hai", "everyone": "Mọi người",
        "computer": "Máy tính", "radio": "Đài", "tv": "Ti-vi", "phone": "Điện thoại",
        # [v3.23.28] Bổ sung từ phân tích bản dịch thực tế (PBS NOVA…).
        "astronaut": "Phi hành gia", "pilot": "Phi công",
        "nasa announcer": "Phát thanh viên NASA", "mission control": "Trung tâm điều khiển",
        "controller": "Nhân viên điều phối", "scientist": "Nhà khoa học",
        "engineer": "Kỹ sư", "commander": "Chỉ huy", "captain": "Đội trưởng",
        "automated voice": "Giọng tự động", "computer voice": "Giọng máy tính",
        "robot": "Người máy", "ai": "Trí tuệ nhân tạo",
        "operator": "Nhân viên trực tổng đài", "news anchor": "Phát thanh viên",
        "translator": "Phiên dịch", "interpreter": "Phiên dịch",
        "boy and girl": "Cậu bé và cô bé", "men": "Những người đàn ông",
        "women": "Những người phụ nữ", "group": "Nhóm người", "people": "Mọi người",
    }

    # [v3.23.90] Chức danh/kính ngữ CJK phổ biến trong phim cổ trang -> Hán-Việt. Dùng
    # làm FALLBACK cho tag người nói khi thuật ngữ không có trong roster, tránh rò ký tự
    # Trung vào phụ đề tiếng Việt (vd cue trả về "大臣"/"皇上/陛下").
    _CJK_ROLE_VI: dict[str, str] = {
        "皇上": "Hoàng thượng", "陛下": "Bệ hạ", "朕": "Trẫm", "皇帝": "Hoàng đế",
        "太后": "Thái hậu", "皇后": "Hoàng hậu", "皇贵妃": "Hoàng quý phi",
        "贵妃": "Quý phi", "娘娘": "Nương nương", "公主": "Công chúa",
        "太子": "Thái tử", "王爷": "Vương gia", "世子": "Thế tử", "殿下": "Điện hạ",
        "大臣": "Đại thần", "丞相": "Thừa tướng", "宰相": "Tể tướng",
        "将军": "Tướng quân", "大将军": "Đại tướng quân", "大人": "Đại nhân",
        "将士": "Tướng sĩ", "士兵": "Binh sĩ", "士卒": "Binh sĩ", "侍卫": "Thị vệ",
        "太监": "Thái giám", "公公": "Công công", "宫女": "Cung nữ", "婢女": "Tỳ nữ",
        "老奴": "Lão nô", "奴婢": "Nô tỳ", "百姓": "Bách tính", "大夫": "Đại phu",
        "钦差": "Khâm sai", "王妃": "Vương phi", "侧妃": "Trắc phi", "嫔妃": "Tần phi",
    }

    # [v3.23.28] Chú thích KÊNH THOẠI đi kèm tên (vd 'on computer', 'on radio').
    # Tách riêng để dịch chú thích nhưng GIỮ tên chính + dấu ngoặc đúng.
    _SPEAKER_CHANNEL_VI: dict[str, str] = {
        "on computer": "trên máy tính", "on radio": "qua radio",
        "on phone": "qua điện thoại", "on tv": "trên ti-vi",
        "on screen": "trên màn hình", "on speaker": "qua loa",
        "on intercom": "qua bộ đàm", "on monitor": "trên màn hình",
        "voiceover": "lời dẫn", "v.o.": "lời dẫn", "o.s.": "ngoài hình",
        "off screen": "ngoài hình", "on recording": "trong bản ghi",
        "via video": "qua video", "on video": "qua video",
        "translated": "đã dịch", "in english": "bằng tiếng Anh",
    }

    @staticmethod
    def _has_cjk(text: str) -> bool:
        """True nếu chuỗi chứa ký tự CJK (Trung/Nhật Kanji/Hàn Hanja)."""
        for ch in text:
            o = ord(ch)
            if (0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF
                    or 0x3040 <= o <= 0x30FF or 0xAC00 <= o <= 0xD7AF):
                return True
        return False

    @staticmethod
    def _normalize_name_key(name: str) -> str:
        """Khoá chuẩn hoá để so tên: hạ hoa-thường + gộp khoảng trắng."""
        return " ".join(name.strip().lower().split())

    @classmethod
    def _build_canonical_name_map(cls, characters: str) -> dict[str, str]:
        """Map ``normalized(alias) -> TÊN CHUẨN`` từ roster, để chuẩn hoá tag người nói.

        Mỗi dòng roster có dạng ``Tên (alias)`` — trong đó MỘT bên là tên đích
        (Latinh/Việt) và bên kia là tên gốc CJK; thứ tự có thể đảo tuỳ model. Hàm tự nhận
        diện phần CJK để lấy phần KHÔNG-CJK làm TÊN CHUẨN, và đăng ký:
          * chính tên chuẩn (chuẩn hoá hoa-thường/khoảng trắng) -> tên chuẩn
          * mỗi alias CJK (tách theo '/', '、', ',') -> tên chuẩn

        CHỈ khớp CHÍNH XÁC sau chuẩn hoá — KHÔNG gộp mờ (không trộn nhân vật khác nhau).
        """
        mapping: dict[str, str] = {}
        if not characters:
            return mapping
        for raw_line in characters.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            match = re.match(r"^(.+?)\s*[(（]\s*([^)）]+?)\s*[)）]", line)
            if not match:
                continue
            left = match.group(1).strip()
            right = match.group(2).strip()
            if not left or not right:
                continue
            # Tên CHUẨN = phần KHÔNG phải CJK (Latinh/Việt); alias = phần CJK.
            if cls._has_cjk(left) and not cls._has_cjk(right):
                canonical, cjk_part = right, left
            elif cls._has_cjk(right) and not cls._has_cjk(left):
                canonical, cjk_part = left, right
            else:
                # Không phân biệt được (cả hai cùng loại) -> dùng phần trái làm chuẩn.
                canonical, cjk_part = left, right
            aliases = [canonical, *re.split(r"[/、,]", cjk_part)]
            for alias in aliases:
                key = cls._normalize_name_key(alias)
                if key and key not in mapping:
                    mapping[key] = canonical
        return mapping

    @staticmethod
    def _build_roster_pronunciation_map(characters: str) -> dict[str, str]:
        """Trích ánh xạ tên-gốc-CJK → tên-phiên-âm từ roster phân tích.

        Roster thường có dạng '林昆 (Lâm Côn), 王语嫣 (Vương Ngữ Yên)'. Hàm tách cặp
        'CJK (phiên âm)' để khi gặp tag tên CJK ta thay bằng bản phiên âm tiếng đích.
        """
        mapping: dict[str, str] = {}
        if not characters:
            return mapping
        # Bắt mẫu 'TÊN_GỐC (phiên âm)' hoặc 'TÊN_GỐC（phiên âm）' (ngoặc CJK).
        for match in re.finditer(r"([^,;()（）]+?)\s*[(（]\s*([^,;()（）]+?)\s*[)）]", characters):
            original = match.group(1).strip()
            romanized = match.group(2).strip()
            if original and romanized:
                mapping[original] = romanized
        return mapping

    @classmethod
    def _localize_one_name(
        cls,
        name: str,
        roster_map: dict[str, str] | None,
        canonical_map: dict[str, str] | None = None,
    ) -> str:
        """Chuẩn hoá MỘT tên người nói (không gồm chú thích).

        0. [v3.23.84] Nếu khớp CHÍNH XÁC một alias trong roster (canonical_map) -> trả về
           TÊN CHUẨN của roster (đồng nhất cách viết + ánh xạ alias CJK sang tên đích).
        1. Nhãn CHUNG (Woman/Astronaut…) → dịch tiếng Việt.
        2. Tên CJK → phiên âm từ roster (khớp chính xác rồi khớp dài nhất); roster
           không có → giữ nguyên CJK.
        3. Tên Latin → GIỮ NGUYÊN.
        """
        cleaned = name.strip().strip("[]():").strip()
        if not cleaned:
            return ""
        # [v3.23.84] Ưu tiên tên chuẩn từ roster (khớp chính xác sau chuẩn hoá).
        if canonical_map:
            canonical = canonical_map.get(cls._normalize_name_key(cleaned))
            if canonical:
                return canonical
        key = cleaned.lower()
        if key in cls._GENERIC_SPEAKER_VI:
            return cls._GENERIC_SPEAKER_VI[key]
        if cls._has_cjk(cleaned) and roster_map:
            if cleaned in roster_map:
                return roster_map[cleaned]
            best_original = ""
            for original in roster_map:
                if original and (original in cleaned or cleaned in original):
                    if len(original) > len(best_original):
                        best_original = original
            if best_original:
                return roster_map[best_original]
        # [v3.23.90] Fallback chức danh/kính ngữ CJK phổ biến (cổ trang) -> Hán-Việt,
        # tránh rò ký tự Trung vào tag tiếng Việt khi không có trong roster. Xử lý cả
        # dạng ghép "皇上/陛下" bằng cách thử từng phần (canonical map trước, rồi map).
        if cls._has_cjk(cleaned):
            for part in re.split(r"[/、,]", cleaned):
                part = part.strip()
                if not part:
                    continue
                if canonical_map:
                    mapped = canonical_map.get(cls._normalize_name_key(part))
                    if mapped:
                        return mapped
                if part in cls._CJK_ROLE_VI:
                    return cls._CJK_ROLE_VI[part]
        return cleaned

    @classmethod
    def _localize_speaker(
        cls,
        speaker: str,
        roster_map: dict[str, str] | None = None,
        canonical_map: dict[str, str] | None = None,
    ) -> str:
        """Chuẩn hoá nhãn người nói cho tag, GIỮ chú thích kênh thoại.

        Xử lý dạng 'TÊN (chú thích)' (vd 'MAN (on computer)', 'ASTRONAUT (on radio)'):
        - Dịch TÊN qua :meth:`_localize_one_name` (nhãn chung sang VI, CJK phiên âm).
        - Dịch CHÚ THÍCH kênh thoại qua bảng :attr:`_SPEAKER_CHANNEL_VI` nếu nhận ra,
          ngược lại giữ nguyên; luôn trả về đúng định dạng 'Tên (chú thích)'.
        Ví dụ: 'MAN (on computer)' → 'Người đàn ông (trên máy tính)'.
        """
        raw = speaker.strip().strip("[]:").strip()
        # Tách 'TÊN (chú thích)' — chú thích là cụm trong ngoặc đơn cuối chuỗi.
        match = re.match(r"^(.*?)\s*[(（]\s*(.*?)\s*[)）]\s*$", raw)  # noqa: RUF001
        if match:
            name_part = match.group(1).strip()
            note_part = match.group(2).strip()
            localized_name = cls._localize_one_name(name_part, roster_map, canonical_map)
            # [v3.23.86] Bỏ chú thích khi nó chỉ là ECHO của TÊN (model hay xuất
            # 'Tần Chính (Tần Chính)' hoặc 'Tần Chính (秦政)') — tránh tag lặp tên.
            if cls._is_redundant_name_note(
                name_part, note_part, localized_name, canonical_map
            ):
                return localized_name
            note_key = note_part.lower().strip()
            localized_note = cls._SPEAKER_CHANNEL_VI.get(note_key, note_part)
            if localized_name and localized_note:
                return f"{localized_name} ({localized_note})"
            return localized_name or localized_note
        return cls._localize_one_name(raw, roster_map, canonical_map)

    @classmethod
    def _is_redundant_name_note(
        cls,
        name_part: str,
        note_part: str,
        localized_name: str,
        canonical_map: dict[str, str] | None,
    ) -> bool:
        """True nếu chú thích trong ngoặc chỉ là echo của TÊN (không phải kênh thoại).

        Bắt các trường hợp: trùng tên gốc/đã địa phương hoá; là tên CJK gốc (chú thích
        kênh thoại không bao giờ là CJK); hoặc cùng quy về MỘT tên chuẩn trong roster.
        """
        if not note_part:
            return False
        note_key = cls._normalize_name_key(note_part)
        # 1. Trùng tên gốc hoặc tên đã địa phương hoá.
        if note_key in {
            cls._normalize_name_key(name_part),
            cls._normalize_name_key(localized_name),
        }:
            return True
        # 2. Chú thích là CJK -> echo tên gốc (kênh thoại luôn ở tiếng đích).
        if cls._has_cjk(note_part):
            return True
        # 3. Cùng quy về một tên chuẩn trong roster.
        if canonical_map:
            note_canon = canonical_map.get(note_key)
            name_canon = canonical_map.get(cls._normalize_name_key(localized_name))
            if note_canon and note_canon == name_canon:
                return True
        return False

    @staticmethod
    def _is_sound_effect(text: str) -> bool:
        """True nếu dòng CHỈ là hiệu ứng âm thanh/nhạc (toàn bộ trong ngoặc đơn).

        Vd '(nhạc kịch tính)', '(tiếng chuông)', '[âm thanh nền]'. Các dòng này KHÔNG
        gắn với người nói nên không bao giờ tag tên.
        """
        stripped = text.strip()
        if not stripped:
            return False
        return (
            (stripped.startswith("(") and stripped.endswith(")"))
            or (stripped.startswith("[") and stripped.endswith("]"))
            or (stripped.startswith("♪") or stripped.endswith("♪"))
        )

    @classmethod
    def _compose_display_text(
        cls,
        line: TranslationLine | None,
        fallback_text: str,
        context: TranslationContext,
        last_speaker: str = "",
        roster_map: dict[str, str] | None = None,
        canonical_map: dict[str, str] | None = None,
    ) -> tuple[str, str]:
        """Ghép text hiển thị, trả về (text, người_nói_hiện_tại để theo dõi tiếp).

        [v3.23.20] Quy tắc tag tên người nói:
          * KHÔNG tag dòng hiệu ứng âm thanh/nhạc (toàn bộ trong ngoặc).
          * KHÔNG tag người nói rỗng/N/A/không xác định.
          * CHỈ tag khi người nói ĐỔI so với dòng liền trước (cùng người → bỏ tag).
          * [v3.23.25] Nhãn chung dịch sang VI; tên CJK phiên âm qua roster; Latin giữ.
        """
        if line is None:
            return fallback_text, last_speaker

        body = line.text
        speaker = line.speaker.strip()
        speaker_norm = speaker.lower()
        is_sfx = cls._is_sound_effect(body)

        # Hiệu ứng âm thanh → reset người nói (lời thoại tiếp theo coi như đổi người),
        # không tag, vẫn giữ phần mô tả nếu bật.
        if is_sfx:
            prefix = ""
            if context.include_desc and line.description.strip():
                prefix += f"({line.description.strip()}) "
            return (f"{prefix}{body}".strip() or fallback_text), ""

        prefix = ""
        valid_speaker = bool(speaker) and speaker_norm not in cls._EMPTY_SPEAKERS
        # [v3.23.136] So sánh "đổi người nói" theo tên ĐÃ CHUẨN HOÁ (localized), không
        # theo chuỗi thô. Cùng một người dưới biến thể tên (tên CJK vs phiên âm, hoa/
        # thường, alias trong roster) trước đây bị coi là người KHÁC -> lặp nhãn thừa.
        localized = (
            cls._localize_speaker(speaker, roster_map, canonical_map)
            if valid_speaker
            else ""
        )
        current_speaker = localized
        # Chỉ tag khi có người nói hợp lệ VÀ khác người nói trước đó (theo tên chuẩn hoá).
        if context.enable_tags and valid_speaker and localized != last_speaker:
            prefix += f"[{localized}:] "
        if context.include_desc and line.description.strip():
            prefix += f"({line.description.strip()}) "
        composed = f"{prefix}{body}".strip() or fallback_text
        composed = cls._dedupe_leading_speaker_tag(composed)
        return composed, current_speaker

    @staticmethod
    def _dedupe_leading_speaker_tag(text: str) -> str:
        """[v3.23.128] Gộp nhãn người nói '[X:] [X:]' bị lặp liền nhau thành một.

        Nguồn có nhãn người nói INLINE (vd 'NARRATOR:') khiến model vừa dịch nhãn vào
        text, vừa điền trường speaker → khi ghép prefix bị lặp '[Tên:] [Tên:]'. Hàm này
        gộp các nhãn dạng '[...:]' GIỐNG HỆT đứng liền đầu câu về một nhãn duy nhất.
        """
        if not text:
            return text
        return re.sub(r"^(\[[^\]]+:\]\s*)\1+", r"\1", text)


__all__ = [
    "TranslateSubtitlesRequest",
    "TranslateSubtitlesResponse",
    "TranslateSubtitlesUseCase",
    "OverallProgressCallback",
    "_compute_checkpoint_key",
    "_StageCheckpoint",
    "_events_to_lines_helper",
]


def _events_to_lines_helper(events: list[SubtitleEvent]) -> list[TranslationLine]:
    """Helper module-level để tầng ngoài chuyển đổi event → line."""
    return TranslateSubtitlesUseCase._events_to_lines(events)
