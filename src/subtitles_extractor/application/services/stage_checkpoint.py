"""[v3.23.608] Bộ nhớ đệm từng giai đoạn dịch: lưu trên đĩa và quy tắc dùng lại.

Tách khỏi ``translate_subtitles`` (trước đây là lớp nội bộ ``_StageCheckpoint``) vì nay
ngoài use case dịch còn có màn "Quản lý giai đoạn" đọc, sửa và xoá từng giai đoạn.

Mỗi bộ phụ đề (cùng câu gốc, cặp ngôn ngữ, Visual Cues) có một file ``{key}.json``:

* ``stages``: kết quả từng giai đoạn đã xong (đủ trường: thời gian, người nói, bản gốc…).
* ``stage_runs``: điều kiện lúc dịch (vân tay đầu vào, cấu hình, model) — kết quả chỉ được
  dùng lại khi khớp (:func:`stage_reuse_verdict`).
* ``stage_inputs``: vân tay TỪNG DÒNG đầu vào. Đầu vào chỉ đổi vài dòng (người dùng sửa
  giai đoạn trước) thì giai đoạn này chỉ dịch lại các lô có dòng đổi
  (:func:`carry_over_lines`).
* ``stage_edits``: dòng người dùng sửa tay. Luôn được áp lại, kể cả khi giai đoạn bị dịch
  lại vì đổi model, đổi cấu hình hay đổi bảng nhân vật — câu sửa tay là bản dịch của chính
  câu gốc dòng đó, mà câu gốc không đổi trong cùng một file.
* ``stage_times``, ``events`` (nhật ký diễn biến), ``visual_cues``, ``context``, ``meta``.

Tiến độ GIỮA giai đoạn (các lô đã xong của lần chạy dở) nằm ở ``{key}.partial.json``.

Từ v3.23.608 bộ nhớ đệm được GIỮ sau khi dịch xong (trước đây bị xoá, nên đổi một giai
đoạn là phải dịch lại cả lượt) và được dọn theo tuổi/số lượng
(:func:`prune_stage_checkpoints`).
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from subtitles_extractor.application.services.model_capability import (
    result_reusable_for,
)
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    TranslationContext,
    TranslationLine,
    TranslationStageConfig,
    TranslationStageKind,
)

logger = logging.getLogger(__name__)

__all__ = [
    "CHECKPOINT_FORMAT",
    "PIPELINE_ORDER",
    "CheckpointIdentity",
    "Clock",
    "PartialProgress",
    "ReuseVerdict",
    "StageCheckpoint",
    "StageEdit",
    "StageEditResult",
    "StageNotCachedError",
    "StageRun",
    "carry_over_lines",
    "context_fingerprint",
    "json_to_lines",
    "legacy_checkpoint_key",
    "legacy_stage_config_signature",
    "line_input_digest",
    "line_key_text",
    "lines_to_json",
    "prune_stage_checkpoints",
    "stage_config_signature",
    "stage_input_digests",
    "stage_input_fingerprint",
    "stage_reuse_verdict",
]

#: Phiên bản định dạng file (1 = trước v3.23.608, không ghi trường này).
CHECKPOINT_FORMAT = 2

#: Thứ tự các giai đoạn trong pipeline.
PIPELINE_ORDER: tuple[TranslationStageKind, ...] = (
    TranslationStageKind.PREPROCESS,
    TranslationStageKind.LITERAL,
    TranslationStageKind.STYLE,
    TranslationStageKind.LOCALIZE,
)

_PARTIAL_SUFFIX = ".partial.json"
_SECONDS_PER_DAY = 86_400

Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _iso(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat(timespec="seconds")


def _short_digest(payload: Any, length: int) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:length]


class StageNotCachedError(LookupError):
    """Giai đoạn được hỏi tới không có trong bộ nhớ đệm."""

    def __init__(self, kind: TranslationStageKind) -> None:
        super().__init__(f"Bộ nhớ đệm chưa có giai đoạn '{kind.value}'.")
        self.kind = kind


# ── Chuyển đổi dòng ⇄ JSON ────────────────────────────────────────────────────

def lines_to_json(lines: list[TranslationLine]) -> list[dict[str, Any]]:
    """Dòng dịch → dạng lưu file (khoá ngắn để file nhỏ).

    [v3.23.81] Lưu ĐỦ trường, đặc biệt ``o`` (bản gốc — mỏ neo chống "tam sao thất bản")
    và ``a`` (người nghe — chọn đại từ/kính ngữ). Thiếu chúng, giai đoạn dùng lại từ bộ
    nhớ đệm mất mỏ neo nguồn ở các giai đoạn sau.
    """
    return [
        {"i": line.index, "s": line.start_ms, "e": line.end_ms,
         "t": line.text, "sp": line.speaker, "d": line.description,
         "o": line.original_text, "a": line.addressee, "c": line.scene}
        for line in lines
    ]


def json_to_lines(rows: list[dict[str, Any]]) -> list[TranslationLine]:
    """Dạng lưu file → dòng dịch (đọc được file của mọi bản cũ)."""
    return [
        TranslationLine(
            index=row["i"], start_ms=row["s"], end_ms=row["e"],
            text=row["t"], speaker=row.get("sp", ""), description=row.get("d", ""),
            original_text=row.get("o", ""), addressee=row.get("a", ""),
            scene=row.get("c", ""),
        )
        for row in rows
    ]


# ── Dấu vân tay ───────────────────────────────────────────────────────────────

def stage_input_fingerprint(
    kind: TranslationStageKind, lines: list[TranslationLine]
) -> str:
    """[v3.23.584] Dấu vân tay ĐẦU VÀO của một giai đoạn (loại + chỉ số + nội dung).

    Kết quả đã lưu của giai đoạn chỉ được dùng NGUYÊN khi đầu vào y hệt lần chạy trước.
    """
    payload = json.dumps(
        [kind.value, [[line.index, line_key_text(line)] for line in lines]], ensure_ascii=False
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def line_key_text(line: TranslationLine) -> str:
    """[v3.23.653] Phần của một dòng đầu vào quyết định bản dịch: chữ, kèm người nói/người nghe NGƯỜI DÙNG chọn.

    Dòng không có cue người dùng → đúng chữ như trước (bộ nhớ đệm đã lưu vẫn khớp); đổi người nói ở trang TTS → dòng
    đó khác vân tay → chỉ lô chứa nó được dịch lại (xưng hô theo người nói mới).
    """
    user_cue = getattr(line, "user_cue", "")
    return line.text if not user_cue else f"{line.text}\x1f{user_cue}"


def line_input_digest(text: str) -> str:
    """[v3.23.608] Dấu vân tay nội dung MỘT dòng đầu vào (10 ký tự hex)."""
    return hashlib.sha256((text or "").encode()).hexdigest()[:10]


def stage_input_digests(lines: Iterable[TranslationLine]) -> dict[int, str]:
    """[v3.23.608] Vân tay từng dòng đầu vào, theo chỉ số dòng."""
    return {line.index: line_input_digest(line_key_text(line)) for line in lines}


def context_fingerprint(context: TranslationContext) -> str:
    """[v3.23.586] Dấu vân tay của BẢNG NHÂN VẬT + BẢNG THUẬT NGỮ dùng cho lượt dịch.

    Hai bảng này quyết định tên riêng, thuật ngữ và xưng hô của MỌI dòng. Khoá file cố ý
    không chứa chúng (Visual Cues bổ sung trong lúc dịch đổi theo từng lượt), nên phải so
    riêng: kết quả dịch bằng bảng KHÁC thì không được dùng lại.

    Args:
        context: Ngữ cảnh của lượt dịch.

    Returns:
        Chuỗi hex 16 ký tự.
    """
    payload = json.dumps(
        [(context.characters or "").strip(), (context.glossary or "").strip()],
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _config_digest(fields: list[Any]) -> str:
    payload = json.dumps(fields, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def stage_config_signature(stage: TranslationStageConfig, attach_video: bool) -> str:
    """[v3.23.606] Dấu vân tay cấu hình một giai đoạn, TRỪ tên model (hàm thuần).

    Đổi phong cách, ghi chú bản địa hoá, mức suy nghĩ, cỡ ngữ cảnh, nhiệt độ, cỡ lô hay
    việc đính video là đổi yêu cầu dịch → kết quả đã lưu không dùng lại. Riêng model được
    so theo hạng (``model_capability.result_reusable_for``).

    [v3.23.609] Chỉ tính những gì THẬT SỰ gửi tới model: suy nghĩ tắt (hoặc ngân sách 0)
    thì mức và ngân sách suy nghĩ không được gửi, nên không còn nằm trong dấu vân tay.
    Trước đây bật suy nghĩ, đổi mức rồi tắt lại là cả giai đoạn bị dịch lại dù yêu cầu y
    hệt.

    Args:
        stage: Cấu hình giai đoạn.
        attach_video: Giai đoạn có đính video không.

    Returns:
        Chuỗi hex 16 ký tự.
    """
    thinking = bool(stage.enable_thinking) and stage.thinking_budget != 0
    return _config_digest(
        [
            stage.kind.value, stage.batch_size, f"{stage.temperature:.2f}",
            stage.context_size, stage.style_name or "", stage.locale_notes or "",
            thinking,
            stage.thinking_budget if thinking else 0,
            (stage.thinking_level or "") if thinking else "",
            stage.allow_retime, attach_video,
        ]
    )


def legacy_stage_config_signature(
    stage: TranslationStageConfig, attach_video: bool
) -> str:
    """Dấu vân tay cấu hình kiểu v3.23.606-608 (gồm cả thông số suy nghĩ khi đang tắt).

    Chỉ dùng để nhận kết quả đã lưu bởi các bản đó.
    """
    return _config_digest(
        [
            stage.kind.value, stage.batch_size, f"{stage.temperature:.2f}",
            stage.context_size, stage.style_name or "", stage.locale_notes or "",
            stage.enable_thinking, stage.thinking_budget, stage.thinking_level or "",
            stage.allow_retime, attach_video,
        ]
    )


@dataclass(frozen=True)
class StageRun:
    """[v3.23.606] Điều kiện để dùng lại kết quả đã lưu của một giai đoạn.

    Attributes:
        input_fingerprint: :func:`stage_input_fingerprint` của đầu vào giai đoạn; rỗng =
            không rõ (file chuyển từ bản cũ — chỉ tin khi chuỗi giai đoạn trước còn
            nguyên).
        config_signature: :func:`stage_config_signature`.
        model_name: Model đã/sẽ dịch giai đoạn.
        accepted_configs: [v3.23.609] Dấu vân tay cấu hình kiểu cũ cũng được coi là
            khớp (kết quả do bản trước lưu) — không ghi vào file.
    """

    input_fingerprint: str
    config_signature: str
    model_name: str
    accepted_configs: tuple[str, ...] = ()

    @property
    def config_signatures(self) -> tuple[str, ...]:
        """Mọi dấu vân tay cấu hình được coi là khớp lần chạy này."""
        return (self.config_signature, *self.accepted_configs)

    @classmethod
    def for_stage(
        cls,
        stage: TranslationStageConfig,
        input_lines: list[TranslationLine],
        *,
        attach_video: bool,
    ) -> StageRun:
        """[v3.23.609] Điều kiện chạy ``stage`` trên ``input_lines`` (nhận cả vân tay cũ).

        Args:
            stage: Cấu hình giai đoạn.
            input_lines: Đầu vào của giai đoạn.
            attach_video: Giai đoạn có đính video không.

        Returns:
            :class:`StageRun`.
        """
        signature = stage_config_signature(stage, attach_video)
        legacy = legacy_stage_config_signature(stage, attach_video)
        return cls(
            input_fingerprint=stage_input_fingerprint(stage.kind, input_lines),
            config_signature=signature,
            model_name=stage.model_name,
            accepted_configs=() if legacy == signature else (legacy,),
        )

    def to_json(self) -> dict[str, str]:
        """Dạng lưu file."""
        return {
            "input": self.input_fingerprint,
            "config": self.config_signature,
            "model": self.model_name,
        }

    @classmethod
    def from_json(cls, raw: Any) -> StageRun | None:
        """Đọc từ file; ``None`` nếu không phải dữ liệu hợp lệ."""
        if not isinstance(raw, dict):
            return None
        return cls(
            input_fingerprint=str(raw.get("input") or ""),
            config_signature=str(raw.get("config") or ""),
            model_name=str(raw.get("model") or ""),
        )


class ReuseVerdict(StrEnum):
    """[v3.23.608] Kết luận có dùng lại NGUYÊN kết quả đã lưu của một giai đoạn không."""

    REUSE = "reuse"
    MISSING = "missing"
    NO_RUN_INFO = "no_run_info"
    CONFIG_CHANGED = "config_changed"
    INPUT_CHANGED = "input_changed"
    UNVERIFIED_INPUT = "unverified_input"
    WEAKER_MODEL = "weaker_model"


def stage_reuse_verdict(
    stored: StageRun | None,
    has_lines: bool,
    run: StageRun,
    *,
    chain_intact: bool,
) -> ReuseVerdict:
    """[v3.23.608] Quy tắc dùng lại nguyên một giai đoạn (hàm thuần).

    Args:
        stored: Điều kiện lúc lưu kết quả (``None`` nếu file không ghi).
        has_lines: Có kết quả đã lưu.
        run: Đầu vào, cấu hình và model của lần chạy này.
        chain_intact: Mọi giai đoạn trước đều dùng lại kết quả đã lưu — điều kiện để tin
            kết quả không ghi vân tay đầu vào (file chuyển từ bản cũ).

    Returns:
        :attr:`ReuseVerdict.REUSE` hoặc lý do phải dịch lại.
    """
    if not has_lines:
        return ReuseVerdict.MISSING
    if stored is None:
        return ReuseVerdict.NO_RUN_INFO
    if stored.config_signature not in run.config_signatures:
        return ReuseVerdict.CONFIG_CHANGED
    if stored.input_fingerprint:
        if stored.input_fingerprint != run.input_fingerprint:
            return ReuseVerdict.INPUT_CHANGED
    elif not chain_intact:
        return ReuseVerdict.UNVERIFIED_INPUT
    if not result_reusable_for(stored.model_name, run.model_name):
        return ReuseVerdict.WEAKER_MODEL
    return ReuseVerdict.REUSE


def carry_over_lines(
    stored_lines: list[TranslationLine],
    stored_digests: Mapping[int, str],
    input_lines: list[TranslationLine],
) -> list[TranslationLine]:
    """[v3.23.608] Dòng đã dịch còn dùng được vì đầu vào của dòng đó không đổi.

    Người dùng sửa vài dòng của giai đoạn dịch thô → đầu vào giai đoạn tinh chỉnh chỉ khác
    ở các dòng đó. Trả về các dòng còn lại để bộ dịch chỉ dịch lại những lô có dòng đổi.

    Args:
        stored_lines: Kết quả đã lưu của giai đoạn.
        stored_digests: Vân tay từng dòng đầu vào lúc dịch kết quả đó.
        input_lines: Đầu vào của lần chạy này.

    Returns:
        Dòng đã lưu (theo thứ tự đầu vào) có đầu vào trùng; rỗng nếu không biết đầu
        vào cũ.
    """
    if not stored_digests:
        return []
    by_index = {line.index: line for line in stored_lines}
    carried: list[TranslationLine] = []
    for line in input_lines:
        saved = by_index.get(line.index)
        digest = stored_digests.get(line.index)
        if saved is None or digest is None or digest != line_input_digest(line_key_text(line)):
            continue
        carried.append(saved)
    return carried


# ── Khoá file ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CheckpointIdentity:
    """[v3.23.608] Những gì xác định file bộ nhớ đệm của một bộ phụ đề.

    Attributes:
        texts: Câu gốc theo thứ tự.
        target_lang: Ngôn ngữ đích.
        source_lang: Ngôn ngữ gốc (đã ghi trong ngữ cảnh).
        visual_cues: Có bật Visual Cues (đổi người nói/người nghe → đổi kết quả dịch).
        visual_cues_batch_size: Cỡ lô quét Visual Cues.
        visual_cues_model: Model quét Visual Cues — CHỈ để tìm file của bản v3.23.606-607
            (khi đó khoá chứa nó, nên đổi model giai đoạn đầu là mất mọi giai đoạn đã
            xong).
    """

    texts: tuple[str, ...]
    target_lang: str
    source_lang: str = ""
    visual_cues: bool = False
    visual_cues_batch_size: int = 150
    visual_cues_model: str = ""

    def _payload(self, *, with_cue_model: bool) -> dict[str, Any]:
        cues: Any = False
        if self.visual_cues:
            cues = (
                [self.visual_cues_model, self.visual_cues_batch_size]
                if with_cue_model
                else [self.visual_cues_batch_size]
            )
        return {
            "texts": list(self.texts),
            "target": self.target_lang,
            "source": self.source_lang or "",
            "visual_cues": cues,
        }

    @property
    def key(self) -> str:
        """Khoá file hiện hành (20 ký tự hex)."""
        return _short_digest(self._payload(with_cue_model=False), 20)

    @property
    def previous_key(self) -> str:
        """Khoá file của bản v3.23.606-607 (trùng :attr:`key` khi tắt Visual Cues)."""
        return _short_digest(self._payload(with_cue_model=True), 20)

    @property
    def texts_fingerprint(self) -> str:
        """Vân tay bộ câu gốc (ghi vào ``meta`` để nhận ra file của phụ đề nào)."""
        return _short_digest(list(self.texts), 16)

    def legacy_payload(self) -> dict[str, Any]:
        """Phần chung của khoá file các bản ≤ v3.23.605."""
        return self._payload(with_cue_model=True)


def legacy_checkpoint_key(
    identity: CheckpointIdentity,
    stages: Iterable[TranslationStageConfig],
    attach_video_stages: Iterable[TranslationStageKind],
) -> str:
    """Khoá file của v3.23.135 đến v3.23.605 (gồm cấu hình mọi giai đoạn).

    Chỉ dùng để nhận file của lần chạy dở do bản cũ ghi.
    """
    stage_ids = [
        f"{s.kind.value}:{s.model_name}:{s.batch_size}:{s.temperature:.2f}"
        f":ctx{s.context_size}:st={s.style_name or ''}:loc={s.locale_notes or ''}"
        f":th={s.enable_thinking}/{s.thinking_budget}/{s.thinking_level or ''}"
        f":rt={s.allow_retime}"
        for s in stages
    ]
    return _short_digest(
        {
            **identity.legacy_payload(),
            "stages": stage_ids,
            "attach_video": sorted(kind.value for kind in attach_video_stages),
        },
        20,
    )


# ── Sửa tay ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class StageEdit:
    """[v3.23.608] Một dòng người dùng sửa tay trong một giai đoạn.

    Attributes:
        text: Câu người dùng chốt.
        model_text: Câu model dịch ra (bản gần nhất) — để hoàn tác và so sánh.
        edited_at: Thời điểm sửa (ISO, UTC).
    """

    text: str
    model_text: str
    edited_at: str


@dataclass(frozen=True)
class StageEditResult:
    """[v3.23.608] Kết quả lưu các sửa đổi của một giai đoạn.

    Attributes:
        changed: Dòng nhận câu sửa mới.
        reverted: Dòng trở về đúng câu model (bỏ đánh dấu sửa tay).
        unknown: Chỉ số dòng không có trong giai đoạn (bị bỏ qua).
        blank: [v3.23.609] Dòng bị xoá trắng — không nhận (dòng rỗng mất khỏi phụ đề và
            lồng tiếng, rồi được "giữ" qua mọi lần dịch lại).
    """

    changed: tuple[int, ...] = ()
    reverted: tuple[int, ...] = ()
    unknown: tuple[int, ...] = ()
    blank: tuple[int, ...] = ()

    @property
    def touched(self) -> int:
        """Số dòng thực sự đổi nội dung."""
        return len(self.changed) + len(self.reverted)


@dataclass(frozen=True)
class PartialProgress:
    """[v3.23.608] Tiến độ dở của một giai đoạn (lô đã xong của lần chạy bị ngắt).

    Attributes:
        kind: Giai đoạn đang dở (``None`` nếu file ghi giá trị lạ).
        line_count: Số dòng đã dịch xong.
        model_name: Model đã dịch các lô đó.
        input_fingerprint: Vân tay đầu vào lúc chạy dở.
    """

    kind: TranslationStageKind | None
    line_count: int
    model_name: str
    input_fingerprint: str


def _stage_kind(value: Any) -> TranslationStageKind | None:
    try:
        return TranslationStageKind(str(value))
    except ValueError:
        return None


def _dict_or_empty(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


# ── Lưu trữ ───────────────────────────────────────────────────────────────────

class StageCheckpoint:
    """Bộ nhớ đệm các giai đoạn dịch của một bộ phụ đề (một file + một file tiến độ dở).

    [v3.23.586] ``context`` là :func:`context_fingerprint` của lượt dịch. Chuẩn đoán thật
    v3.23.585: 360 dòng dịch thô của lần chạy dở (bảng nhân vật cũ) bị nối vào lượt dịch
    dùng bảng MỚI — bản dịch mang hai bộ tên. Ngữ cảnh đã đổi thì bỏ kết quả cũ, dịch lại;
    [v3.23.608] riêng các dòng sửa tay được giữ để áp lại.

    [v3.23.592] ``events`` giữ nhật ký diễn biến của TỪNG giai đoạn đã xong để giai đoạn
    dùng lại từ bộ nhớ đệm vẫn hiện trong file chuẩn đoán.

    Args:
        checkpoint_dir: Thư mục bộ nhớ đệm (tự tạo).
        key: Khoá file (:attr:`CheckpointIdentity.key`).
        context: :func:`context_fingerprint` của lượt dịch; rỗng = không so (màn quản lý,
            nút xoá).
        meta: Thông tin mô tả ghi kèm file (số dòng, ngôn ngữ…).
        clock: Đồng hồ (để test); mặc định giờ UTC hiện tại.
    """

    def __init__(
        self,
        checkpoint_dir: Path,
        key: str,
        context: str = "",
        *,
        meta: Mapping[str, Any] | None = None,
        clock: Clock | None = None,
    ) -> None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._key = key
        self._path = checkpoint_dir / f"{key}.json"
        self._partial_path = checkpoint_dir / f"{key}{_PARTIAL_SUFFIX}"
        self._context = context
        self._clock: Clock = clock or _utc_now
        self._stored_context = ""
        self._data: dict[str, list[dict[str, Any]]] = {}
        self._visual_cues: list[dict[str, Any]] | None = None
        self._partial: dict[str, Any] | None = None
        self._stage_events: dict[str, list[dict[str, Any]]] = {}
        self._stage_runs: dict[str, dict[str, str]] = {}
        self._stage_inputs: dict[str, dict[str, str]] = {}
        self._stage_times: dict[str, str] = {}
        self._stage_edits: dict[str, dict[str, dict[str, str]]] = {}
        self._meta: dict[str, Any] = {}
        self._load()
        if meta:
            self._meta.update(meta)

    # ── Đọc file ─────────────────────────────────────────────────────────────
    def _reset_stages(self) -> None:
        self._data = {}
        self._visual_cues = None
        self._stage_events = {}
        self._stage_runs = {}
        self._stage_inputs = {}
        self._stage_times = {}

    def _load(self) -> None:
        if self._path.exists():
            try:
                raw = json.loads(self._path.read_text(encoding="utf-8"))
                if not isinstance(raw, dict):
                    raise ValueError("nội dung không phải đối tượng JSON")
                self._data = _dict_or_empty(raw.get("stages"))
                cues = raw.get("visual_cues")
                self._visual_cues = cues if isinstance(cues, list) else None
                self._stage_events = _dict_or_empty(raw.get("events"))
                self._stage_runs = _dict_or_empty(raw.get("stage_runs"))
                self._stage_inputs = _dict_or_empty(raw.get("stage_inputs"))
                self._stage_times = _dict_or_empty(raw.get("stage_times"))
                self._stage_edits = _dict_or_empty(raw.get("stage_edits"))
                self._meta = _dict_or_empty(raw.get("meta"))
                self._stored_context = str(raw.get("context") or "")
                logger.info(
                    "Đọc bộ nhớ đệm dịch: %s (%d giai đoạn đã lưu, %d giai đoạn có dòng "
                    "sửa tay).",
                    self._path.name, len(self._data), len(self._stage_edits),
                )
            except (json.JSONDecodeError, OSError, ValueError) as exc:
                logger.warning(
                    "Không đọc được bộ nhớ đệm %s: %s — bắt đầu mới.", self._path, exc
                )
                self._reset_stages()
                self._stage_edits = {}
                self._meta = {}
                self._stored_context = ""
        if self._partial_path.exists():
            try:
                raw_partial = json.loads(self._partial_path.read_text(encoding="utf-8"))
            except (ValueError, OSError) as exc:   # [v3.23.638] gồm UnicodeDecodeError
                logger.warning(
                    "Không đọc được tiến độ dở %s: %s — giai đoạn đó dịch lại từ đầu.",
                    self._partial_path.name, exc,
                )
            else:
                self._partial = raw_partial if isinstance(raw_partial, dict) else None
        partial_context = str((self._partial or {}).get("context") or "")
        self._discard_if_context_changed(self._stored_context, partial_context)

    def _discard_if_context_changed(self, stored: str, partial: str) -> None:
        """[v3.23.586] Bỏ phần đã lưu được dịch bằng bảng nhân vật/thuật ngữ KHÁC.

        File do bản cũ ghi (không có dấu vân tay) được giữ như trước. [v3.23.608] Dòng sửa
        tay được giữ: chúng là bản dịch người dùng chốt cho chính câu gốc của dòng.

        Args:
            stored: Dấu vân tay ghi trong file các giai đoạn đã xong.
            partial: Dấu vân tay ghi trong file tiến độ dở.
        """
        if not self._context:
            return
        stale_stages = bool(stored) and stored != self._context
        stale_partial = bool(partial) and partial != self._context
        if not stale_stages and not stale_partial:
            return
        dropped_stages = sorted(self._data) if stale_stages else []
        partial_lines = (self._partial or {}).get("lines") or []
        dropped_lines = len(partial_lines) if stale_partial else 0
        kept_edits = sum(len(edits) for edits in self._stage_edits.values())
        if stale_stages:
            self._reset_stages()
            self._stored_context = self._context
            if self._stage_edits:
                self._flush()
            else:
                self._remove(self._path)
        if stale_partial:
            self._partial = None
            self._remove(self._partial_path)
        logger.warning(
            "Bảng nhân vật/thuật ngữ đã ĐỔI so với lúc lưu → bỏ %d giai đoạn đã lưu "
            "và %d dòng của giai đoạn đang dở, dịch lại bằng ngữ cảnh mới để tên riêng, "
            "thuật ngữ và xưng hô nhất quán (giữ %d dòng sửa tay để áp lại).",
            len(dropped_stages), dropped_lines, kept_edits,
        )
        from subtitles_extractor.application.services.translation_process_log import (
            record_event,
        )

        record_event(
            "checkpoint_discarded",
            reason="context_changed",
            stages=dropped_stages,
            partial_lines=dropped_lines,
            kept_edits=kept_edits,
        )

    # ── Thông tin ────────────────────────────────────────────────────────────
    @property
    def key(self) -> str:
        """Khoá file."""
        return self._key

    @property
    def stored_context(self) -> str:
        """Vân tay ngữ cảnh ghi trong file (rỗng nếu file bản cũ / chưa có)."""
        return self._stored_context

    @property
    def meta(self) -> dict[str, Any]:
        """Thông tin mô tả ghi kèm file (bản sao)."""
        return dict(self._meta)

    def stage_kinds(self) -> list[TranslationStageKind]:
        """[v3.23.608] Các giai đoạn có kết quả đã lưu, theo thứ tự pipeline."""
        return [kind for kind in PIPELINE_ORDER if kind.value in self._data]

    def get_stage(self, kind: TranslationStageKind) -> list[TranslationLine] | None:
        """Kết quả đã lưu của giai đoạn, hoặc ``None`` nếu chưa có."""
        raw = self._data.get(kind.value)
        if raw is None:
            return None
        try:
            return json_to_lines(raw)
        except (KeyError, TypeError) as exc:
            logger.warning(
                "Kết quả lưu của giai đoạn '%s' bị hỏng (%s) — coi như chưa có.",
                kind.value, exc,
            )
            return None

    def stored_run(self, kind: TranslationStageKind) -> StageRun | None:
        """[v3.23.606] Điều kiện lúc lưu kết quả của giai đoạn (None nếu không ghi)."""
        return StageRun.from_json(self._stage_runs.get(kind.value))

    def stage_completed_at(self, kind: TranslationStageKind) -> str:
        """[v3.23.608] Thời điểm lưu kết quả giai đoạn (ISO, UTC; rỗng = file bản cũ)."""
        return str(self._stage_times.get(kind.value) or "")

    def input_digests(self, kind: TranslationStageKind) -> dict[int, str]:
        """[v3.23.608] Vân tay từng dòng đầu vào lúc dịch giai đoạn (rỗng = không ghi)."""
        digests: dict[int, str] = {}
        for index, digest in _dict_or_empty(self._stage_inputs.get(kind.value)).items():
            try:
                digests[int(index)] = str(digest)
            except ValueError:
                continue
        return digests

    def stage_edits(self, kind: TranslationStageKind) -> dict[int, StageEdit]:
        """[v3.23.608] Dòng sửa tay của giai đoạn, theo chỉ số dòng."""
        edits: dict[int, StageEdit] = {}
        for index, raw in _dict_or_empty(self._stage_edits.get(kind.value)).items():
            if not isinstance(raw, dict):
                continue
            try:
                edits[int(index)] = StageEdit(
                    text=str(raw.get("text") or ""),
                    model_text=str(raw.get("model") or ""),
                    edited_at=str(raw.get("at") or ""),
                )
            except ValueError:
                continue
        return edits

    def edited_line_count(self, kind: TranslationStageKind) -> int:
        """[v3.23.608] Số dòng sửa tay của giai đoạn."""
        return len(_dict_or_empty(self._stage_edits.get(kind.value)))

    def reuse_verdict(
        self,
        kind: TranslationStageKind,
        run: StageRun,
        *,
        chain_intact: bool,
    ) -> ReuseVerdict:
        """[v3.23.608] :func:`stage_reuse_verdict` cho giai đoạn đã lưu."""
        return stage_reuse_verdict(
            self.stored_run(kind),
            kind.value in self._data,
            run,
            chain_intact=chain_intact,
        )

    def reusable_stage(
        self,
        kind: TranslationStageKind,
        run: StageRun,
        *,
        chain_intact: bool,
    ) -> list[TranslationLine] | None:
        """[v3.23.606] Kết quả đã lưu của giai đoạn nếu dùng lại NGUYÊN được.

        Args:
            kind: Giai đoạn sắp chạy.
            run: Đầu vào, cấu hình và model của lần chạy này.
            chain_intact: Mọi giai đoạn trước đều dùng lại kết quả đã lưu.

        Returns:
            Các dòng đã lưu, hoặc None khi chưa có / khác đầu vào / khác cấu hình / model
            đã dịch yếu hơn model đang chọn.
        """
        if self.reuse_verdict(kind, run, chain_intact=chain_intact) is not (
            ReuseVerdict.REUSE
        ):
            return None
        return self.get_stage(kind)

    def carry_over(
        self,
        kind: TranslationStageKind,
        run: StageRun,
        input_lines: list[TranslationLine],
    ) -> list[TranslationLine]:
        """[v3.23.608] Dòng đã lưu còn dùng được khi đầu vào giai đoạn chỉ đổi một phần.

        Chỉ khi cùng cấu hình và model đã dịch mạnh hơn hoặc ngang model đang chọn — như
        điều kiện dùng lại nguyên giai đoạn, trừ điều kiện đầu vào giống hệt.

        Args:
            kind: Giai đoạn sắp chạy.
            run: Cấu hình và model của lần chạy này.
            input_lines: Đầu vào của lần chạy này.

        Returns:
            Các dòng còn dùng được (rỗng nếu không có).
        """
        stored = self.stored_run(kind)
        lines = self.get_stage(kind)
        if lines is None or stored is None:
            return []
        if stored.config_signature not in run.config_signatures:
            return []
        if not result_reusable_for(stored.model_name, run.model_name):
            return []
        return carry_over_lines(lines, self.input_digests(kind), input_lines)

    def get_visual_cues(self) -> list[dict[str, Any]] | None:
        """[v3.23.32] Visual Cues đã lưu (dạng rút gọn), None nếu chưa có."""
        return self._visual_cues

    def save_visual_cues(self, cues_json: list[dict[str, Any]]) -> None:
        """[v3.23.32] Lưu Visual Cues để dịch lại không phải quét video lại."""
        self._visual_cues = cues_json
        self._flush()

    def get_stage_events(self, kind: TranslationStageKind) -> list[dict[str, Any]]:
        """[v3.23.592] Nhật ký diễn biến đã lưu của một giai đoạn (rỗng nếu không có)."""
        events = self._stage_events.get(kind.value)
        if not isinstance(events, list):
            return []
        return [item for item in events if isinstance(item, dict)]

    # ── Ghi kết quả giai đoạn ────────────────────────────────────────────────
    def apply_edits(
        self, kind: TranslationStageKind, lines: list[TranslationLine]
    ) -> tuple[list[TranslationLine], list[int]]:
        """[v3.23.608] Áp các dòng sửa tay lên kết quả (mới dịch) của giai đoạn.

        Câu model mới cho dòng đã sửa được ghi lại làm ``model_text`` để người dùng còn
        hoàn tác về bản model mới nhất. Không ghi file — :meth:`save_stage` ghi.

        Args:
            kind: Giai đoạn.
            lines: Kết quả của giai đoạn.

        Returns:
            ``(dòng sau khi áp, chỉ số các dòng vừa được áp đè lên câu model khác)``.
        """
        raw_edits = _dict_or_empty(self._stage_edits.get(kind.value))
        if not raw_edits:
            return list(lines), []
        from dataclasses import replace

        applied: list[int] = []
        result: list[TranslationLine] = []
        for line in lines:
            edit = raw_edits.get(str(line.index))
            if not isinstance(edit, dict):
                result.append(line)
                continue
            edited_text = str(edit.get("text") or "")
            if line.text != edited_text:
                edit["model"] = line.text
                applied.append(line.index)
                line = replace(line, text=edited_text)
            result.append(line)
        return result, applied

    def save_stage(
        self,
        kind: TranslationStageKind,
        lines: list[TranslationLine],
        events: list[dict[str, Any]] | None = None,
        run: StageRun | None = None,
        input_lines: list[TranslationLine] | None = None,
    ) -> None:
        """Lưu kết quả giai đoạn xuống đĩa (và bỏ tiến độ dở của giai đoạn đó).

        Dòng sửa tay của giai đoạn luôn được áp lên trước khi lưu.

        Args:
            kind: Giai đoạn vừa xong.
            lines: Kết quả của giai đoạn.
            events: [v3.23.592] Nhật ký diễn biến của riêng giai đoạn; ``None`` thì giữ
                phần đã lưu (nếu có).
            run: [v3.23.606] Đầu vào, cấu hình và model đã dịch giai đoạn.
            input_lines: [v3.23.608] Đầu vào của giai đoạn — để lần sau chỉ dịch lại các
                dòng có đầu vào đổi; ``None`` = không ghi.
        """
        lines, _ = self.apply_edits(kind, lines)
        self._data[kind.value] = lines_to_json(lines)
        if events is not None:
            self._stage_events[kind.value] = events
        if run is not None:
            self._stage_runs[kind.value] = run.to_json()
        else:
            self._stage_runs.pop(kind.value, None)
        if input_lines is not None:
            self._stage_inputs[kind.value] = {
                str(index): digest
                for index, digest in stage_input_digests(input_lines).items()
            }
        else:
            self._stage_inputs.pop(kind.value, None)
        self._stage_times[kind.value] = _iso(self._clock())
        self._flush()
        if self._partial is not None and self._partial.get("stage") == kind.value:
            self._partial = None
            self._remove(self._partial_path)

    # ── Sửa / xoá theo yêu cầu người dùng ────────────────────────────────────
    def edit_stage(
        self, kind: TranslationStageKind, texts: Mapping[int, str]
    ) -> StageEditResult:
        """[v3.23.608] Ghi câu người dùng sửa cho các dòng của một giai đoạn đã lưu.

        Câu trùng câu model dịch ra thì bỏ đánh dấu sửa tay (hoàn tác). Các giai đoạn sau
        tự nhận ra đầu vào đã đổi ở lần dịch tới và chỉ dịch lại các lô có dòng đổi.

        Args:
            kind: Giai đoạn.
            texts: ``chỉ số dòng → câu mới``.

        Returns:
            Các dòng đã đổi / đã hoàn tác / không tồn tại.

        Raises:
            StageNotCachedError: Giai đoạn chưa có trong bộ nhớ đệm.
        """
        rows = self._data.get(kind.value)
        if rows is None:
            raise StageNotCachedError(kind)
        by_index = {row.get("i"): row for row in rows if isinstance(row, dict)}
        edits = dict(_dict_or_empty(self._stage_edits.get(kind.value)))
        changed: list[int] = []
        reverted: list[int] = []
        unknown: list[int] = []
        blank: list[int] = []
        now = _iso(self._clock())
        for index, raw_text in sorted(texts.items()):
            row = by_index.get(index)
            if row is None:
                unknown.append(index)
                continue
            new_text = (raw_text or "").strip()
            if not new_text:
                blank.append(index)
                continue
            current_text = str(row.get("t") or "")
            existing = edits.get(str(index))
            model_text = (
                str(existing.get("model") or "")
                if isinstance(existing, dict)
                else current_text
            )
            if new_text == model_text:
                if existing is not None:
                    edits.pop(str(index))
                    row["t"] = model_text
                    reverted.append(index)
                continue
            if new_text == current_text and existing is not None:
                continue
            edits[str(index)] = {"text": new_text, "model": model_text, "at": now}
            row["t"] = new_text
            changed.append(index)
        if edits:
            self._stage_edits[kind.value] = edits
        else:
            self._stage_edits.pop(kind.value, None)
        if changed or reverted:
            self._flush()
            logger.info(
                "Giai đoạn '%s': lưu %d dòng sửa tay, hoàn tác %d dòng.",
                kind.value, len(changed), len(reverted),
            )
        if unknown:
            logger.warning(
                "Giai đoạn '%s': bỏ qua %d dòng không có trong bộ nhớ đệm: %s.",
                kind.value, len(unknown), unknown[:10],
            )
        return StageEditResult(
            tuple(changed), tuple(reverted), tuple(unknown), tuple(blank)
        )

    def discard_edits(
        self,
        kind: TranslationStageKind,
        indices: Iterable[int] | None = None,
    ) -> list[int]:
        """[v3.23.608] Trả các dòng sửa tay về câu model (``None`` = mọi dòng).

        Args:
            kind: Giai đoạn.
            indices: Dòng cần hoàn tác.

        Returns:
            Chỉ số các dòng đã hoàn tác.
        """
        edits = self.stage_edits(kind)
        wanted = set(edits) if indices is None else set(indices) & set(edits)
        if not wanted or kind.value not in self._data:
            return []
        result = self.edit_stage(
            kind, {index: edits[index].model_text for index in wanted}
        )
        return list(result.reverted)

    def delete_stage(self, kind: TranslationStageKind, *, keep_edits: bool) -> None:
        """[v3.23.608] Xoá kết quả đã lưu của một giai đoạn (lần dịch tới dịch lại nó).

        Args:
            kind: Giai đoạn.
            keep_edits: Giữ các dòng sửa tay để áp lên bản dịch mới.
        """
        for store in (
            self._data, self._stage_runs, self._stage_events, self._stage_inputs,
            self._stage_times,
        ):
            store.pop(kind.value, None)
        if not keep_edits:
            self._stage_edits.pop(kind.value, None)
        if self._partial is not None and self._partial.get("stage") == kind.value:
            self._partial = None
            self._remove(self._partial_path)
        if self._data or self._stage_edits or self._visual_cues is not None:
            self._flush()
        else:
            self._remove(self._path)
        logger.info(
            "Đã xoá giai đoạn '%s' khỏi bộ nhớ đệm %s (%s dòng sửa tay).",
            kind.value, self._path.name, "giữ" if keep_edits else "xoá cả",
        )

    # ── Tiến độ dở ───────────────────────────────────────────────────────────
    @staticmethod
    def _partial_matches(
        partial: dict[str, Any] | None,
        kind: TranslationStageKind,
        input_fingerprint: str,
        run: StageRun | None,
    ) -> bool:
        """Tiến độ dở có đúng giai đoạn, đúng đầu vào (và đúng điều kiện ``run``)."""
        if (
            not partial
            or partial.get("stage") != kind.value
            or partial.get("input") != input_fingerprint
        ):
            return False
        if run is None:
            return True
        return partial.get("config") in run.config_signatures and result_reusable_for(
            str(partial.get("model") or ""), run.model_name
        )

    def partial_model(self, kind: TranslationStageKind, input_fingerprint: str) -> str:
        """[v3.23.606] Model đã dịch tiến độ dở của giai đoạn (rỗng nếu không có)."""
        partial = self._partial
        if not self._partial_matches(partial, kind, input_fingerprint, None):
            return ""
        return str((partial or {}).get("model") or "")

    @property
    def has_partial(self) -> bool:
        """[v3.23.606] Có tiến độ dở của một giai đoạn nào đó."""
        return bool(self._partial)

    def partial_progress(self) -> PartialProgress | None:
        """[v3.23.608] Tóm tắt tiến độ dở (để hiển thị), ``None`` nếu không có."""
        partial = self._partial
        if not partial:
            return None
        lines = partial.get("lines")
        return PartialProgress(
            kind=_stage_kind(partial.get("stage")),
            line_count=len(lines) if isinstance(lines, list) else 0,
            model_name=str(partial.get("model") or ""),
            input_fingerprint=str(partial.get("input") or ""),
        )

    def get_partial(
        self,
        kind: TranslationStageKind,
        input_fingerprint: str,
        run: StageRun | None = None,
    ) -> list[TranslationLine] | None:
        """[v3.23.584] Dòng đã dịch xong ở lần chạy DỞ của giai đoạn này.

        Args:
            kind: Giai đoạn sắp chạy.
            input_fingerprint: :func:`stage_input_fingerprint` của đầu vào hiện tại.
            run: [v3.23.606] Cấu hình và model của lần chạy này; ``None`` = không so.

        Returns:
            Các dòng đã lưu, hoặc ``None`` khi không có / của giai đoạn khác / đầu vào
            khác / cấu hình khác / dịch bằng model yếu hơn model đang chọn.
        """
        partial = self._partial
        if not self._partial_matches(partial, kind, input_fingerprint, run):
            return None
        try:
            lines = json_to_lines((partial or {}).get("lines") or [])
        except (KeyError, TypeError) as exc:
            logger.warning(
                "Tiến độ dở trong bộ nhớ đệm bị hỏng (%s) — dịch lại từ đầu.", exc
            )
            return None
        return lines or None

    def add_partial(
        self,
        kind: TranslationStageKind,
        input_fingerprint: str,
        lines: list[TranslationLine],
        run: StageRun | None = None,
    ) -> None:
        """[v3.23.584] Ghi thêm các dòng vừa dịch xong của giai đoạn đang chạy.

        Tiến độ của giai đoạn khác, đầu vào khác — [v3.23.606] hay cấu hình khác, model
        yếu hơn — bị thay bằng tiến độ mới.

        Args:
            kind: Giai đoạn đang chạy.
            input_fingerprint: :func:`stage_input_fingerprint` của đầu vào.
            lines: Các dòng của một lô vừa xong (chỉ số theo đầu vào giai đoạn).
            run: [v3.23.606] Cấu hình và model đang dịch giai đoạn.
        """
        partial = self._partial
        if not self._partial_matches(partial, kind, input_fingerprint, run):
            partial = {"stage": kind.value, "input": input_fingerprint, "lines": []}
        partial = dict(partial or {})
        if run is not None:
            # Gộp với lô của model mạnh hơn: ghi model ĐANG dịch (yếu nhất trong số đã
            # góp) để lần sau đổi lên model mạnh hơn thì vẫn dịch lại.
            partial["config"] = run.config_signature
            partial["model"] = run.model_name
        if self._context:
            partial["context"] = self._context
        by_index = {row["i"]: row for row in partial.get("lines") or []}
        by_index.update({row["i"]: row for row in lines_to_json(lines)})
        partial["lines"] = [by_index[index] for index in sorted(by_index)]
        self._partial = partial
        try:
            self._write_atomic(self._partial_path, partial)
        except OSError as exc:
            logger.warning("Không lưu được tiến độ dở: %s — vẫn dịch tiếp.", exc)

    # ── Nhận file của bản cũ ─────────────────────────────────────────────────
    @property
    def is_empty(self) -> bool:
        """Không có giai đoạn đã xong, tiến độ dở hay dòng sửa tay nào."""
        return not self._data and not self._partial and not self._stage_edits

    def adopt(self, previous: StageCheckpoint) -> list[str]:
        """[v3.23.608] Nhận TOÀN BỘ nội dung một file khoá khác (cùng phụ đề) rồi xoá nó.

        Dùng cho file của bản v3.23.606-607 (khoá chứa model quét Visual Cues): nội dung
        đã có đủ điều kiện dùng lại từng giai đoạn nên nhận nguyên.

        Args:
            previous: Bộ nhớ đệm mở bằng khoá cũ.

        Returns:
            Các giai đoạn đã xong được nhận (theo ``kind.value``).
        """
        self._data = dict(previous._data)
        self._stage_runs = dict(previous._stage_runs)
        self._stage_events = dict(previous._stage_events)
        self._stage_inputs = dict(previous._stage_inputs)
        self._stage_times = dict(previous._stage_times)
        self._stage_edits = dict(previous._stage_edits)
        self._visual_cues = previous._visual_cues
        self._meta = {**previous._meta, **self._meta}
        if not self._stored_context:
            self._stored_context = previous._stored_context
        self._take_partial(previous._partial)
        self._flush()
        previous.delete()
        return list(self._data)

    def adopt_legacy(
        self,
        legacy: StageCheckpoint,
        runs: dict[str, StageRun],
    ) -> list[str]:
        """[v3.23.606] Nhận nội dung file do bản ≤ v3.23.605 ghi rồi xoá file đó.

        Khoá cũ chứa cấu hình mọi giai đoạn, nên file cũ tìm được bằng cấu hình hiện tại
        chắc chắn được dịch với đúng cấu hình đó; ``runs`` mang điều kiện tương ứng.

        Args:
            legacy: Bộ nhớ đệm mở bằng khoá cũ.
            runs: :class:`StageRun` theo ``kind.value`` của các giai đoạn trong yêu cầu.

        Returns:
            Các giai đoạn đã xong được nhận (theo ``kind.value``).
        """
        adopted = [kind for kind in legacy._data if kind in runs]
        self._data = {kind: legacy._data[kind] for kind in adopted}
        self._stage_runs = {kind: runs[kind].to_json() for kind in adopted}
        self._stage_events = {
            kind: events for kind, events in legacy._stage_events.items() if kind in runs
        }
        self._visual_cues = legacy._visual_cues
        if not self._stored_context:
            self._stored_context = legacy._stored_context
        partial = dict(legacy._partial) if legacy._partial else None
        if partial is not None and partial.get("stage") in runs:
            run = runs[str(partial["stage"])]
            partial["config"] = run.config_signature
            partial["model"] = run.model_name
            self._take_partial(partial)
        self._flush()
        legacy.delete()
        return adopted

    def _take_partial(self, partial: dict[str, Any] | None) -> None:
        if not partial:
            return
        self._partial = dict(partial)
        try:
            self._write_atomic(self._partial_path, self._partial)
        except OSError as exc:
            logger.warning("Không chuyển được tiến độ dở: %s", exc)

    # ── Ghi / xoá file ───────────────────────────────────────────────────────
    @staticmethod
    def _write_atomic(path: Path, payload: Mapping[str, Any]) -> None:
        """Ghi file tạm rồi thay thế: mất điện giữa chừng không làm hỏng bản cũ."""
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=None), encoding="utf-8"
        )
        temporary.replace(path)

    def _flush(self) -> None:
        context = self._context or self._stored_context
        payload: dict[str, Any] = {
            "format": CHECKPOINT_FORMAT,
            "key": self._key,
            "stages": self._data,
            "meta": {**self._meta, "updated_at": _iso(self._clock())},
        }
        if context:
            payload["context"] = context
        optional = {
            "visual_cues": self._visual_cues,
            "events": self._stage_events,
            "stage_runs": self._stage_runs,
            "stage_inputs": self._stage_inputs,
            "stage_times": self._stage_times,
            "stage_edits": self._stage_edits,
        }
        payload.update({name: value for name, value in optional.items() if value})
        try:
            self._write_atomic(self._path, payload)
        except OSError as exc:
            logger.warning(
                "Không lưu được bộ nhớ đệm dịch: %s — tiếp tục không lưu.", exc
            )
        else:
            self._stored_context = context

    @staticmethod
    def _remove(path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Không xoá được bộ nhớ đệm %s: %s", path, exc)

    def delete(self) -> None:
        """Xoá cả file (kết quả các giai đoạn, dòng sửa tay) lẫn tiến độ dở."""
        self._remove(self._path)
        self._remove(self._partial_path)
        self._reset_stages()
        self._stage_edits = {}
        self._partial = None

    @property
    def exists_on_disk(self) -> bool:
        """Có file trên đĩa."""
        return self._path.exists() or self._partial_path.exists()

    def touch(self) -> None:
        """[v3.23.609] Ghi nhận vừa được DÙNG (dọn theo tuổi/số lượng xét giờ sửa file).

        Lượt dịch lấy nguyên mọi giai đoạn từ bộ nhớ đệm không ghi gì vào file, nên
        trước đây phim vừa dịch lại hôm qua vẫn bị coi là "cũ" và bị dọn mất.
        """
        if not self._path.exists():
            return
        try:
            self._path.touch()
        except OSError as exc:
            logger.debug(
                "Không cập nhật được giờ dùng bộ nhớ đệm %s: %s", self._path, exc
            )


def prune_stage_checkpoints(
    checkpoint_dir: Path,
    *,
    max_files: int,
    max_age_days: int,
    keep: Iterable[str] = (),
    now: datetime | None = None,
) -> list[str]:
    """[v3.23.608] Dọn bộ nhớ đệm cũ: giữ ``max_files`` file mới nhất, không quá tuổi.

    Bộ nhớ đệm nay được giữ sau khi dịch xong nên phải có giới hạn; phim dịch gần đây
    (hay được sửa/dịch lại) được ưu tiên giữ theo thời điểm ghi file.

    Args:
        checkpoint_dir: Thư mục bộ nhớ đệm.
        max_files: Số file tối đa được giữ.
        max_age_days: Tuổi tối đa (ngày, theo lần ghi cuối).
        keep: Khoá luôn giữ (lượt dịch đang chạy).
        now: Thời điểm hiện tại (để test).

    Returns:
        Các khoá đã xoá.
    """
    if not checkpoint_dir.is_dir():
        return []
    protected = set(keep)
    entries: list[tuple[float, Path]] = []
    partial_only: list[tuple[float, Path]] = []
    for path in checkpoint_dir.glob("*.json"):
        try:
            modified = path.stat().st_mtime
        except OSError:
            continue
        if path.name.endswith(_PARTIAL_SUFFIX):
            main = checkpoint_dir / (path.name[: -len(_PARTIAL_SUFFIX)] + ".json")
            if not main.exists():
                partial_only.append((modified, path))
            continue
        entries.append((modified, path))
    entries.extend(partial_only)
    entries.sort(key=lambda item: item[0], reverse=True)
    cutoff = (now or _utc_now()).timestamp() - max_age_days * _SECONDS_PER_DAY
    removed: list[str] = []
    for position, (modified, path) in enumerate(entries):
        key = path.name.split(".", 1)[0]
        if key in protected or (position < max_files and modified >= cutoff):
            continue
        victims = (
            checkpoint_dir / f"{key}.json",
            checkpoint_dir / f"{key}{_PARTIAL_SUFFIX}",
        )
        for victim in victims:
            try:
                victim.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning("Không dọn được bộ nhớ đệm cũ %s: %s", victim.name, exc)
        removed.append(key)
    if removed:
        logger.info(
            "Dọn %d bộ nhớ đệm dịch cũ (giữ tối đa %d bộ, trong %d ngày).",
            len(removed), max_files, max_age_days,
        )
    return removed
