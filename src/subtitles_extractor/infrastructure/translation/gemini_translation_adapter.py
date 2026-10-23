"""Adapter dịch phụ đề dùng Google Gemini (``google-genai``).

Chuyển thể thuật toán dịch đa giai đoạn từ mã tham khảo sang kiến trúc sạch:
    * Dựng ``system_instruction`` riêng cho từng giai đoạn.
    * Chia lô (batch) + cửa sổ trượt ngữ cảnh (sliding window) trước/sau.
    * Gọi Gemini với JSON schema ràng buộc đầu ra.
    * Retry luỹ thừa cho lỗi server tạm thời + lỗi sai cấu trúc batch.
    * Kiểm tra toàn vẹn ``line_no`` để chống lệch dòng.

Thư viện ``google-genai`` được import LAZY: nếu chưa cài, adapter vẫn khởi tạo
được nhưng ``is_available()`` trả False và mọi lệnh dịch ném
``TranslatorUnavailableError`` — giúp ứng dụng không sập khi thiếu dependency.
"""

from __future__ import annotations

import copy
import json
import logging
import random
import re
import threading
import time
from collections.abc import Callable
from itertools import count
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any

from subtitles_extractor.application.services.analysis_audit import (
    AUDIT_MAX_PROMPT_LINES,
    AUDIT_THINKING_LEVEL,
    AUDIT_WINDOW_LINES,
)
from subtitles_extractor.application.services.batch_resume import BatchResume
from subtitles_extractor.application.services.under_translation_guard import (
    find_regressions,
    find_under_translated,
    log_under_translated,
)
from subtitles_extractor.application.services.address_gloss import address_pronoun_gloss
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    ANALYSIS_STOP_DAILY_QUOTA,
    ANALYSIS_STOP_OVERLOAD,
    BatchDoneCallback,
    CancellationCallback,
    ModelOverloadedError,
    StageProgressCallback,
    SubtitleContextAnalysis,
    SubtitleTranslationError,
    TranslationCancelledError,
    TranslationContext,
    TranslationLine,
    TranslationStageConfig,
    TranslationStageKind,
    TranslatorUnavailableError,
    VisualCue,
)
from subtitles_extractor.domain.value_objects.speaker_label import (
    CROWD_SPEAKER,
    NARRATOR_SPEAKER,
    VOICE_ONLY_SPEAKERS,
)
from subtitles_extractor.application.services.token_usage import (
    TokenUsage,
    TokenUsageLedger,
)
from subtitles_extractor.infrastructure.gemini_model_family import (
    gemini_model_version,
    uses_thinking_level,
)
from subtitles_extractor.infrastructure.translation.key_selection import KeyView, choose_key
from subtitles_extractor.infrastructure.translation.model_health import (
    FAILOVER_AFTER_IN_CALL,
    ModelHealth,
)
from subtitles_extractor.infrastructure.translation.analysis_checkpoint import (
    SHARED_PROGRESS_MODEL,
    AnalysisCheckpointStore,
    AnalysisProgress,
    analysis_checkpoint_key,
    describe_segment_models,
)
from subtitles_extractor.infrastructure.translation.empty_response_policy import (
    EmptyResponse,
    diagnose_empty_response,
    should_drop_video,
)
from subtitles_extractor.infrastructure.translation.gemini_quota_manager import (
    MAX_SERVER_COOLDOWN_SEC,
    QuotaExhaustedError,
    daily_quota_message,
)
from subtitles_extractor.infrastructure.translation.key_rotation_policy import (
    should_switch_for_viability,
)
from subtitles_extractor.infrastructure.translation.overload_retry_policy import (
    deadline_was_hit,
    escalated_request_timeout,
    overload_attempt_cap,
    overload_failure_message,
    overload_key_cap,
    scarce_overload_delay,
    should_try_other_key,
)
from subtitles_extractor.infrastructure.translation.thinking_policy import (
    LOWEST_COMMON_LEVEL,
    MINIMAL_LEVEL,
    THINKING_LEVELS,
    ThinkingSettings,
    resolve_thinking,
    supports_minimal_level,
)
from subtitles_extractor.infrastructure.tts.timing_math import (
    MIN_CHAR_BUDGET,
    effective_available_seconds,
    readable_syllable_budget,
    reading_char_budget,
    syllable_budget_to_chars,
)

logger = logging.getLogger(__name__)

# [v3.23.143] Thư viện sửa JSON hỏng từ LLM (thuần Python, zero phụ thuộc bắt buộc). Import
# TUỲ CHỌN: nếu môi trường chưa cài, tự lùi về bộ vá cắt cụt nội bộ (không sập).
try:
    import json_repair as _json_repair
except ImportError:  # pragma: no cover - môi trường chưa cài optional dep
    _json_repair = None


# ── JSON Schema ràng buộc đầu ra của Gemini ──────────────────────────────────
MAX_ALIGNMENT_RETRIES = 2
"""Số lần tối đa ép dịch lại một lô vì lệch dòng — quá ngưỡng thì nhận kết quả, tránh đốt quota."""

MAX_SAMPLE_LINES = 12
"""Số dòng tối đa trích cho mỗi lô lỗi — đủ để đối chiếu mà không phình tệp chuẩn đoán."""

MAX_SAMPLE_CHARS = 160
"""Trần độ dài mỗi đoạn văn bản trích."""


def _batch_content_sample(
    batch: list[Any],
    payload: dict[str, Any] | None,
    focus_lines: list[int] | None = None,
) -> list[dict[str, Any]]:
    """[v3.23.520] Trích NỘI DUNG của một lô có vấn đề để đối chiếu khi phân tích.

    Ghi cả câu GỐC lẫn bản model TRẢ VỀ (kể cả bản sai) — không có nội dung thì biết lô nào lỗi
    cũng chẳng suy ra được model hiểu sai ở đâu. Chỉ trích cho lô CÓ VẤN ĐỀ và giới hạn số dòng nên
    tệp chuẩn đoán không phình: ghi trọn 5540 dòng × 4 giai đoạn là vô ích.

    Args:
        batch: Các dòng nguồn của lô.
        payload: JSON model trả về (có thể None hoặc dị dạng).

    Returns:
        Danh sách ``{"line_no", "source", "returned"}`` đã cắt ngắn.
    """
    try:
        items = (payload or {}).get("subtitles") or []
        returned: dict[int, str] = {}
        for offset, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            line_no = _safe_int(item.get("line_no"), -1)
            if line_no <= 0 and offset < len(batch):
                line_no = batch[offset].index
            if line_no > 0:
                returned[line_no] = str(item.get("text", "") or "")[:MAX_SAMPLE_CHARS]
        # [v3.23.521] Ưu tiên trích QUANH dòng bị cờ. Bản trước lấy 12 dòng ĐẦU lô, nhưng dòng
        # lệch thường nằm giữa lô (vd lô 541-600 bị cờ ở dòng 575, 587, 591) nên mẫu ghi ra không
        # chứa chỗ cần xem — không đối chiếu được gì.
        chosen = batch
        if focus_lines:
            wanted: set[int] = set()
            for line_no in focus_lines:
                wanted.update(range(line_no - 1, line_no + 2))
            focused = [line for line in batch if line.index in wanted]
            if focused:
                chosen = focused
        sample: list[dict[str, Any]] = []
        # [v3.23.552] Khối lệch cần TRỌN khối làm bằng chứng — nới trần theo số dòng được yêu cầu.
        limit = MAX_SAMPLE_LINES
        if focus_lines:
            limit = min(max(MAX_SAMPLE_LINES, len(focus_lines) + 2), MAX_BLOCK_EVIDENCE_LINES + 2)
        for line in chosen[:limit]:
            sample.append({
                "line_no": line.index,
                "source": str(line.original_text or line.text or "")[:MAX_SAMPLE_CHARS],
                "returned": returned.get(line.index, ""),
            })
        if len(items) > len(batch):
            sample.append({"note": f"model trả {len(items)} mục cho lô {len(batch)} dòng"})
        return sample
    except Exception:  # noqa: BLE001 — trích nội dung là việc phụ
        return []


def _record_process_event(event: str, **fields: Any) -> None:
    """[v3.23.519] Ghi một sự kiện diễn biến dịch; không bao giờ làm hỏng khâu dịch."""
    try:
        from subtitles_extractor.application.services.translation_process_log import (
            record_event,
        )

        record_event(event, **fields)
    except Exception:  # noqa: BLE001 — ghi nhật ký là việc phụ
        pass


def _accepts_ranges_argument(callback: Callable[..., Any]) -> bool:
    """[v3.23.550] True nếu callback tải lại video nhận được đối số thứ hai (``ranges``).

    Dò bằng chữ ký thay vì thử-gọi-rồi-bắt-``TypeError``: một ``TypeError`` phát sinh BÊN TRONG
    callback mới sẽ bị hiểu nhầm là "callback kiểu cũ" và kích hoạt tải lại cả phim.
    """
    import inspect

    try:
        parameters = list(inspect.signature(callback).parameters.values())
    except (TypeError, ValueError):
        return False
    if any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in parameters):
        return True
    positional = [
        p for p in parameters
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    return len(positional) >= 2


MAX_BLOCK_EVIDENCE_LINES = 25
"""Số dòng tối đa ghi làm bằng chứng cho một khối lệch (giữ file chuẩn đoán gọn)."""


def _misalignment_focus_lines(misaligned: list[dict[str, Any]]) -> list[int]:
    """[v3.23.552] Các dòng cần ghi làm BẰNG CHỨNG cho một lần bắt lệch.

    Lệch theo mốc neo: đúng các dòng bị cờ. Lệch theo KHỐI: TRỌN khối — trước đây chỉ ghi dòng đầu
    và dòng cuối (hai mục của kết quả), nên chuẩn đoán thật không kiểm lại được phần giữa: mẫu ghi
    lại của lô 2101-2160 toàn các dòng căn đúng, không chứng minh được lần dịch lại là cần thiết.
    """
    first = misaligned[0]
    if "block_lines" in first:
        start = int(first["index"])
        end = int(misaligned[-1]["index"])
        return list(range(start, min(end, start + MAX_BLOCK_EVIDENCE_LINES - 1) + 1))
    return [int(item["index"]) for item in misaligned]


def _misalignment_message(misaligned: list[dict[str, Any]]) -> str:
    """[v3.23.549] Thông điệp lỗi lệch dòng — phân biệt lệch theo MỐC NEO với lệch theo KHỐI.

    Kết quả từ bộ dò khối không có "dòng đích" cụ thể (tương quan cả đoạn sụp đổ), nên thông điệp
    kiểu "dòng X mang nội dung của dòng X" vừa vô nghĩa vừa làm người đọc log hiểu sai.
    """
    first = misaligned[0]
    if "block_lines" in first:
        return (
            f"Nội dung LỆCH DÒNG theo KHỐI: dòng {first['index']}-{misaligned[-1]['index']} "
            f"({first['block_lines']} dòng), tương quan độ dài nguồn/dịch tụt còn "
            f"{first['min_correlation']} — nghi model gộp/bỏ dòng, cần dịch lại."
        )
    return (
        f"Nội dung LỆCH DÒNG ở {len(misaligned)} dòng "
        f"(vd dòng {first['index']} mang nội dung của dòng "
        f"{first['belongs_to_index']}) — cần dịch lại."
    )


#: [v3.23.620] Cỡ lô lớn nhất còn gắn minItems/maxItems cùng khoảng line_no (đo được).
MAX_ITEM_COUNT_CONSTRAINED_BATCH = 40


def relax_schema_constraints(schema: Any, array_key: str = "subtitles") -> Any | None:
    """[v3.23.620] Nới một bậc ràng buộc schema lô; ``None`` nếu không còn gì để nới.

    Thứ tự: bỏ minItems/maxItems trước (giới hạn chặt nhất), rồi tới khoảng line_no.
    """
    if not isinstance(schema, dict):
        return None
    array_schema = schema.get("properties", {}).get(array_key)
    if not isinstance(array_schema, dict):
        return None
    relaxed = copy.deepcopy(schema)
    target = relaxed["properties"][array_key]
    if "minItems" in target or "maxItems" in target:
        target.pop("minItems", None)
        target.pop("maxItems", None)
        return relaxed
    line_no = target.get("items", {}).get("properties", {}).get("line_no")
    if isinstance(line_no, dict) and ("minimum" in line_no or "maximum" in line_no):
        line_no.pop("minimum", None)
        line_no.pop("maximum", None)
        return relaxed
    return None


DESCRIPTION_FIELD_NOTE = (
    "Chỉ tiếng động/âm thanh KHÔNG LỜI có thật mà dòng gốc ghi rõ (vd chữ gốc '(敲门声)'). "
    "KHÔNG ghi cảm xúc, giọng điệu, suy nghĩ hay giải thích. Mặc định để trống."
)


def _batch_constrained_schema(
    base_schema: dict[str, Any], batch: list[Any], array_key: str = "subtitles"
) -> dict[str, Any]:
    """[v3.23.509] Ràng buộc SỐ DÒNG và KHOẢNG line_no ngay trong schema của từng lô.

    Tài liệu Gemini (Structured outputs) cho phép ``minItems``/``maxItems`` với mảng và
    ``minimum``/``maximum`` với số nguyên. Đây là cách ÉP ở tầng schema thay vì năn nỉ trong
    prompt: model buộc phải trả về ĐÚNG số phần tử bằng số dòng gửi đi, và line_no phải nằm trong
    khoảng của lô — trực tiếp chặn kiểu lỗi gộp/bỏ/lệch dòng vốn là nguồn sai lớn nhất.

    Bổ sung ``description`` cho từng trường theo khuyến nghị "Clear descriptions" của tài liệu.

    Args:
        base_schema: Schema gốc (bất biến, không bị sửa tại chỗ).
        batch: Các dòng nguồn của lô hiện tại.
        array_key: Tên trường mảng chứa kết quả.

    Returns:
        Bản sao schema đã gắn ràng buộc; trả về schema gốc nếu lô rỗng hoặc cấu trúc không khớp.
    """
    if not batch:
        return base_schema
    array_schema = base_schema.get("properties", {}).get(array_key)
    if not isinstance(array_schema, dict):
        return base_schema
    schema = copy.deepcopy(base_schema)
    target = schema["properties"][array_key]
    # [v3.23.620] Gemini từ chối (400 INVALID_ARGUMENT) schema có CẢ minItems/maxItems lẫn
    # khoảng line_no khi lô ≥ 50 dòng — tức lô MẶC ĐỊNH 50: phim ≥ 50 dòng không dịch được.
    # Đo 3.1/3.5-flash-lite: cả hai ràng buộc chạy tới 40 dòng, chỉ số phần tử tới 50, chỉ
    # khoảng line_no tới 200. Lô ≤ 40 giữ cả hai; lô lớn hơn chỉ giữ khoảng line_no (lệch
    # số dòng vẫn bị bắt ở bước kiểm toàn vẹn sau khi nhận kết quả).
    if len(batch) <= MAX_ITEM_COUNT_CONSTRAINED_BATCH:
        target["minItems"] = len(batch)
        target["maxItems"] = len(batch)
    target["description"] = (
        f"Đúng {len(batch)} phần tử, mỗi phần tử ứng với MỘT dòng phụ đề trong lô, "
        "giữ nguyên thứ tự và line_no."
    )
    indices = [int(getattr(line, "index", 0) or 0) for line in batch]
    indices = [i for i in indices if i > 0]
    item_properties = target.get("items", {}).get("properties", {})
    if indices and isinstance(item_properties, dict) and "line_no" in item_properties:
        item_properties["line_no"] = {
            **item_properties["line_no"],
            "minimum": min(indices),
            "maximum": max(indices),
            "description": "Số dòng gốc — phải TRÙNG KHỚP với dòng nguồn tương ứng.",
        }
    for field, note in (
        ("text", "Bản dịch của ĐÚNG dòng này, không mang nội dung dòng khác."),
        ("speaker", "Tên người nói; để trống nếu không chắc chắn."),
        # [v3.23.620] "Chú thích ngắn" khiến model bịa cảm xúc/giọng điệu/độc thoại nội tâm
        # vào ngoặc đầu câu (đo: 11-19 dòng "(…)" mỗi 2 tập dù phụ đề gốc không có).
        ("description", DESCRIPTION_FIELD_NOTE),
    ):
        if isinstance(item_properties, dict) and field in item_properties:
            item_properties[field] = {**item_properties[field], "description": note}
    return schema


_TRANSLATE_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "subtitles": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "line_no": {"type": "NUMBER"},
                    "speaker": {"type": "STRING"},
                    "text": {"type": "STRING"},
                    "description": {"type": "STRING"},
                },
                "required": ["line_no", "speaker", "text", "description"],
            },
        }
    },
    "required": ["subtitles"],
}

_TRANSLATE_NO_DESC_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "subtitles": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "line_no": {"type": "NUMBER"},
                    "speaker": {"type": "STRING"},
                    "text": {"type": "STRING"},
                },
                "required": ["line_no", "speaker", "text"],
            },
        }
    },
    "required": ["subtitles"],
}
"""Schema rút gọn không có ``description`` — dùng khi ``context.include_desc=False``."""

_PREPROCESS_TEXT_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "subtitles": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "line_no": {"type": "NUMBER"},
                    "text": {"type": "STRING"},
                },
                "required": ["line_no", "text"],
            },
        }
    },
    "required": ["subtitles"],
}

# [v3.23.14] Token video/giây để ước lượng quota. Video ngữ cảnh nén 360p/1fps gửi
# tới Gemini 3 (~70 token/frame mặc định) ≈ 100 token/giây. Dùng cận trên an toàn.
_VIDEO_TOKENS_PER_SEC = 100


class ClippedVideoHandle:
    """[v3.23.648] Handle file video kèm khoảng ``[start_s, end_s]`` (giây, tính từ ĐẦU đoạn) để chỉ gửi phần đó.

    Phần còn lại hành xử như handle gốc (``uri``, ``mime_type``, ``name``…), nên mã khác không cần biết. Handle gốc là
    đối tượng dùng chung trong bộ nhớ đệm nên KHÔNG sửa tại chỗ mà bọc lại.
    """

    __slots__ = ("end_s", "handle", "start_s")

    def __init__(self, handle: Any, start_s: float, end_s: float) -> None:
        self.handle = handle
        self.start_s = float(start_s)
        self.end_s = float(end_s)

    def __getattr__(self, name: str) -> Any:
        if name in ClippedVideoHandle.__slots__:        # chưa khởi tạo xong (vd copy/pickle): tránh đệ quy vô hạn
            raise AttributeError(name)
        return getattr(self.handle, name)

    @property
    def length_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)


def replace_refs_by_range(target: list[Any], fresh: list[Any]) -> None:
    """[v3.23.561] Thay tại chỗ ref trong ``target`` bằng ref mới CÙNG khoảng thời gian.

    Ref video khớp nhau theo ``(start_sec, end_sec)`` — tên file đổi sau khi tải lại bằng
    key khác, còn khoảng thời gian thì không.

    Args:
        target: Danh sách ref cần cập nhật (bị sửa tại chỗ).
        fresh: Các ref mới vừa tải lại.
    """
    by_range = {
        (getattr(ref, "start_sec", None), getattr(ref, "end_sec", None)): ref
        for ref in fresh
    }
    for position, ref in enumerate(target):
        key = (getattr(ref, "start_sec", None), getattr(ref, "end_sec", None))
        if key in by_range:
            target[position] = by_range[key]


#: Khoá ghi nhớ ref đã tải lại: (vân tay API key, start_sec, end_sec).
HealedRefKey = tuple[str, Any, Any]


def healed_ref_key(key_id: str, ref: Any) -> HealedRefKey:
    """[v3.23.605] Khoá tra ref đã tải lại của một đoạn video dưới một API key.

    Args:
        key_id: Vân tay API key đã tải lại đoạn đó.
        ref: Ref video (cũ hoặc mới) — chỉ dùng khoảng thời gian của nó.

    Returns:
        Bộ ba (vân tay key, start_sec, end_sec).
    """
    return (key_id, getattr(ref, "start_sec", None), getattr(ref, "end_sec", None))


def substitute_healed_refs(
    refs: list[Any], healed: dict[HealedRefKey, Any], key_id: str
) -> int:
    """[v3.23.605] Thay TẠI CHỖ ref nào đã có bản tải lại bằng key ``key_id``.

    Mỗi giai đoạn dịch nhận lại danh sách ref GỐC của phiên (thuộc key lúc bắt đầu). Sau
    khi xoay key, giai đoạn trước đã tải lại các đoạn bằng key mới, nhưng giai đoạn sau
    vẫn cầm ref của key cũ: nhật ký thật 11/10 21:10 cho thấy giai đoạn 4 lại dò 403 và
    "tải lại" đủ 17 đoạn lần nữa. Bảng ghi nhớ theo key cho phép dùng ngay ref mới.

    Args:
        refs: Danh sách ref (bị sửa tại chỗ).
        healed: Bảng ref đã tải lại, khoá theo :func:`healed_ref_key`.
        key_id: Vân tay API key HIỆN TẠI.

    Returns:
        Số ref đã được thay.
    """
    replaced = 0
    for position, ref in enumerate(refs):
        fresh = healed.get(healed_ref_key(key_id, ref))
        if fresh is None or fresh is ref:
            continue
        if getattr(fresh, "remote_name", None) == getattr(ref, "remote_name", None):
            continue
        refs[position] = fresh
        replaced += 1
    return replaced

# [v3.23.107] Nhận diện ký tự CJK (Hán/Kana/Hangul) để bảo vệ dòng rác OCR khỏi dịch.
_CJK_CHAR_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u30ff\uac00-\ud7af]")
_CJK_LANG_PREFIXES = ("zh", "ja", "ko", "chi", "jpn", "kor", "yue", "cmn")
# [v3.23.350] Nếu bộ lọc "dòng rác CJK" coi >= ngưỡng này số dòng có nội dung là rác thì
# gần như chắc do SAI NHÃN ngôn ngữ (nguồn đặt CJK nhưng phụ đề thực tế là Latin) → bỏ
# qua nhánh lọc để không trả về nguyên bản gốc. 0.90: phụ đề CJK thật hầu như luôn có
# ký tự CJK ở đại đa số dòng nên không chạm ngưỡng; chỉ nội dung KHÔNG-CJK mới vượt.
_CJK_NOISE_MAX_RATIO = 0.90
# [v3.23.352] Nếu nhãn ngôn ngữ nguồn là CJK nhưng tỉ lệ dòng có ký tự CJK thấp hơn
# ngưỡng này thì gần như chắc AI đã nhầm ngôn ngữ thoại/gốc phim với ngôn ngữ CHỮ phụ đề.
_CJK_TEXT_MIN_RATIO = 0.10

_MAX_BATCH_GLOSSARY_TERMS = 25
"""Trần số thuật ngữ nhắc lại mỗi lô — đủ phủ lô 30-60 dòng mà không phình prompt."""

#: [v3.23.615] 499 CANCELLED: máy chủ tự huỷ lời gọi khi hết thời hạn ứng dụng đặt. Ứng
#: dụng không bao giờ tự huỷ một lời gọi HTTP đang chạy (huỷ là hợp tác, giữa các lượt),
#: nên 499 luôn đến từ phía máy chủ. app_log thật 14/10 14:26: lô dịch thô chạy đúng 300
#: giây rồi nhận 499 — bị coi là lỗi vĩnh viễn, cả lượt dịch thất bại, phải dịch lại.
_RETRYABLE_HTTP_CODES = frozenset({408, 429, 499, 500, 502, 503, 504})

#: [v3.23.554] Lỗi QUÁ TẢI phía Gemini (503/504): server bận, không phải hết quota nên
#: xoay key vô ích. Nhật ký thật 2026-09-23: HAI lần dịch hỏng hẳn vì đợt quá tải kéo
#: dài hơn tổng thời gian chờ của 5 lượt thử thường (~83 giây).
#: [v3.23.615] 499 cùng nhóm với 504 DEADLINE_EXCEEDED: máy chủ chưa xong trong thời hạn.
_OVERLOAD_HTTP_CODES = frozenset({499, 503, 504})
_OVERLOAD_MARKERS = (
    "499 cancelled",
    "unavailable",
    "high demand",
    "overloaded",
    "try again later",
    "temporarily unavailable",
    "service unavailable",
    "deadline_exceeded",
)

#: Tổng thời gian được chờ THÊM riêng cho lỗi quá tải, sau khi hết lượt thử thường. Mất
#: cả lượt dịch (hàng chục phút và tiền token) tệ hơn nhiều so với chờ thêm 10 phút.
OVERLOAD_RETRY_BUDGET_S = 600.0

#: Trần mỗi lần chờ trong ngân sách quá tải — chờ thưa, không nã liên tục vào server bận.
OVERLOAD_MAX_DELAY_S = 120.0
_RETRYABLE_MARKERS = (
    "499 cancelled",
    "quota",
    "resource_exhausted",
    "unavailable",
    "deadline_exceeded",
    "internal",
    "high demand",
    "try again later",
    "temporarily unavailable",
    "service unavailable",
    "connection aborted",
    "connection reset",
    "connection closed",
    "timed out",
    "timeout",
    "network",
    "transport",
    # [v3.23.437] Lỗi mạng nhất thời còn thiếu (từ app_log: "Server disconnected without
    # sending a response" của httpx.RemoteProtocolError → trước đây KHÔNG khớp marker nào →
    # cả tập dịch thất bại thay vì thử lại). Bổ sung để retry đúng như 503.
    "disconnect",
    "remoteprotocolerror",
    "remote protocol",
    "protocol error",
    "server disconnected",
    "peer closed",
    "connection error",
    "broken pipe",
    "eof occurred",
    "incomplete read",
    "connection refused",
    # [v3.23.487] 403 loại "FILE" của Gemini (file thuộc key cũ sau khi xoay key, hoặc đã hết
    # hạn 48h): app CÓ cơ chế tự tải lại đoạn video ở lần thử lại → đây là lỗi PHỤC HỒI ĐƯỢC.
    # Trước đây bị xếp VĨNH VIỄN nên v477 bỏ luôn lượt thử lại → mất hẳn ngữ cảnh đoạn đó
    # (app_log: đoạn 21). Lưu ý: 403 do SAI API KEY vẫn là vĩnh viễn vì không khớp các mẫu này.
    "permission to access the file",
    "or it may not exist",
    # [v3.23.480] Lỗi DNS/mất mạng nhất thời (app_log: "[Errno 11001] getaddrinfo failed" khi
    # internet rớt giữa chừng) → TRƯỚC ĐÂY không khớp marker nào → coi là VĨNH VIỄN → bỏ retry cả
    # 16 đoạn. Đây là lỗi TẠM THỜI (mạng có thể hồi) → phải retry như 503.
    "getaddrinfo",
    "errno 11001",  # Windows: DNS name không phân giải được
    "errno 11002",  # Windows: DNS tạm thời lỗi
    "name resolution",
    "name or service not known",
    "temporary failure in name resolution",
    "nodename nor servname",
    "failed to resolve",
    "no address associated",
    "network is unreachable",
    "no route to host",
    "socket",
)


# [v3.23.131] Trần kích thước lô cho model "lite" (nhẹ) — lô lớn hơn gây rớt/lệch dòng.
_LITE_MODEL_BATCH_CAP = 60


def effective_batch_size(model_name: str, batch_size: int) -> int:
    """[v3.23.609] Cỡ lô THỰC dùng cho một model (model "lite" bị kẹp ở trần riêng).

    Dùng chung cho bước dịch và cho ước lượng số request của "Dịch cả bộ" — trước đây
    ước lượng đọc một ô "cỡ lô" không tồn tại và luôn lấy 40.

    Args:
        model_name: Tên model.
        batch_size: Cỡ lô người dùng đặt.

    Returns:
        Cỡ lô ≥ 1.
    """
    size = max(1, batch_size)
    if "lite" in (model_name or "").lower():
        return min(size, _LITE_MODEL_BATCH_CAP)
    return size

#: [v3.23.591] Vai trò của VIDEO đính kèm khi dịch. Trước bản này prompt đính 1-3 đoạn
#: video mà KHÔNG nói video dùng để làm gì. Chuẩn đoán thật (phim "灭门惨案II": nói tiếng
#: Quảng Đông, phụ đề cứng theo văn viết — lời nói và chữ khác nhau): model dịch theo điều
#: NGHE được. Trên 16 lượt dịch phim này, bản thô đúng ở cả 16 lượt nhưng bản cuối mang
#: lời chỉ có trong tiếng:
#:
#: * dòng 68 "不要生氣" → "Mở cửa mau!" / "Khai mau." (8 lượt);
#: * dòng 70 "這麼久" → "Nhanh lên!" (4 lượt);
#: * dòng 77 "你沒有事嗎" → "Mọi người vỗ tay hoan nghênh!" (3 lượt);
#: * dòng 93 "改革開放成功了" → "MƯỜI NĂM cải cách mở cửa…" (6 lượt; câu gốc không có con
#:   số nào, bản thô của cả 16 lượt cũng không).
VIDEO_ROLE_RULE = (
    "- VIDEO KÈM THEO (nếu có) CHỈ LÀ NGỮ CẢNH: dùng để hiểu ai đang nói, nói với ai, "
    "thái độ và hành động trong cảnh. NGUỒN ĐỂ DỊCH luôn là CHỮ của từng dòng phụ đề. "
    "Lời NGHE được trong video có thể KHÁC chữ phụ đề (phim lồng tiếng lại hoặc nói tiếng "
    "địa phương, tiếng người khác chen vào, câu thoại không có phụ đề): TUYỆT ĐỐI KHÔNG "
    "dịch theo lời nghe được, KHÔNG thêm từ ngữ hay con số chỉ có trong âm thanh, KHÔNG "
    "thay nội dung của một dòng bằng câu nghe thấy ở thời điểm đó. Chữ và tiếng khác nhau "
    "→ LUÔN theo CHỮ. (Tag cảm xúc, nếu có quy tắc TAG CẢM XÚC ở trên, vẫn chèn theo quy "
    "tắc đó.)\n"
)

#: [v3.23.592] Vế dịch của bảng thuật ngữ: phần TRƯỚC ngoặc phải là cách nói dùng được.
#:
#: Prompt phân tích vốn chỉ dặn điều này cho "tiếng lóng/thành ngữ". Bản phân tích thật
#: v3.23.591 ghi ``所長 = sở trưởng (trưởng đồn công an)`` — đọc âm từng chữ, còn cách nói
#: người Việt hiểu thì để trong ngoặc — và khâu dịch chép "Sở trưởng" vào cả 15 dòng. Trên
#: 19 bảng thuật ngữ thật có 89 mục như vậy (18/19 bảng); 553 dòng dịch cuối của 56 lượt
#: mang nguyên phần đọc âm. Xem ``glossary_reading_check`` (lưới chỉ ra các mục này).
GLOSSARY_WORDING_RULE = (
    "\n  - Phần TRƯỚC ngoặc là thứ khâu dịch chép NGUYÊN vào lời thoại — đúng với MỌI "
    "mục, kể cả chức danh và thành ngữ. Nếu chính bạn phải ghi một cách nói dễ hiểu hơn "
    "vào trong ngoặc thì cách nói ấy mới là bản dịch: đưa nó ra TRƯỚC ngoặc. "
    "SAI: '所長 => sở trưởng (trưởng đồn công an)', '通姦 => thông gian (ngoại tình)'; "
    "ĐÚNG: '所長 => trưởng đồn', '通姦 => ngoại tình'. "
    "Đọc âm từng chữ chỉ dành cho tên riêng và cho từ mà người nói {target_lang} thật sự "
    "dùng (vd 'công an', 'cục trưởng')."
)


#: [v3.23.612] Tuỳ chọn "Xưng hô thuần Việt" — khâu PHÂN TÍCH (bảng nhân vật, tóm tắt,
#: thuật ngữ). Bản phân tích thật phim "灭门惨案II" ghi ``甄劍聲 / 劍哥 (Chân Kiếm Thanh /
#: Kiếm ca)``, ``洪大嫂 (Hồng đại tẩu)``, danh xưng "Chân tiên sinh"; khâu dịch chép
#: nguyên vào 28 dòng "Kiếm ca", 8 dòng "đại tẩu". Phần model vẫn viết kiểu Hán-Việt
#: được đổi tất định sau đó (``native_address``).
NATIVE_ADDRESS_ANALYSIS_RULE = (
    "\n\u2022 XƯNG HÔ THUẦN VIỆT (người dùng chọn cho phim hiện đại, đời thường): trong "
    "\"aka\", danh xưng, quy tắc xưng hô, overview và glossary, tiếng gọi TÊN kèm vai vế "
    "viết theo cách người Việt nói — từ vai vế đứng TRƯỚC tên, KHÔNG đọc âm Hán-Việt hậu "
    "tố. '劍哥' → 'anh Kiếm' (KHÔNG 'Kiếm ca'); '明月姐' → 'chị Minh Nguyệt' (KHÔNG "
    "'Minh Nguyệt tỷ'); '洪大嫂' → 'chị Hồng' (KHÔNG 'Hồng đại tẩu'); '林先生' → 'anh "
    "Lâm' với người trẻ, 'ông Lâm' với người lớn tuổi (KHÔNG 'Lâm tiên sinh'); '周姨' → "
    "'dì Chu'; '小钱弟弟' → 'em Tiền'; '大嫂' đứng một mình → 'chị dâu'; '表哥' → 'anh "
    "họ'. Danh xưng của từng cặp dùng đúng từ vai vế người nói gọi người kia (cảnh sát "
    "gọi 'cô' thì danh xưng là 'cô Chân'). GIỮ phiên âm tên riêng (Kiếm, Chân) và kiểu "
    "'A + tên' (A Tú)."
)
#: [v3.23.612] Cùng tuỳ chọn — khâu DỊCH (mọi giai đoạn trừ hiệu đính nguồn).
NATIVE_ADDRESS_TRANSLATION_RULE = (
    "- XƯNG HÔ THUẦN VIỆT (BẮT BUỘC — người dùng chọn cho phim hiện đại, đời "
    "thường): tiếng gọi TÊN kèm vai vế dịch theo cách người Việt nói — từ vai vế đứng "
    "TRƯỚC tên, KHÔNG đọc âm Hán-Việt hậu tố. '劍哥' → 'anh Kiếm' (KHÔNG 'Kiếm ca'); "
    "'明月姐' → 'chị Minh Nguyệt' (KHÔNG 'Minh Nguyệt tỷ'); '洪大嫂' → 'chị Hồng' "
    "(KHÔNG 'Hồng đại tẩu'); '林先生' → 'anh Lâm' hoặc 'ông Lâm' (KHÔNG 'Lâm tiên "
    "sinh'); '周姨' → 'dì Chu'; '大嫂' → 'chị dâu'; '表哥' → 'anh họ'. Từ vai vế trước "
    "tên lấy theo ĐÚNG đại từ người nói dùng để gọi người đó trong bảng xưng hô (trường "
    "'address' gọi 'cô' → 'cô Chân'). Giữ nguyên phiên âm tên và kiểu 'A + tên' (A "
    "Tú).\n"
)


def _patch_threshold(expected_count: int) -> int:
    """Số dòng thiếu tối đa của một lô vẫn VÁ được (≤ max(2, 15%)); hơn thì chia đôi."""
    return max(2, expected_count * 15 // 100)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return default if value is None else int(value)
    except (TypeError, ValueError):
        return default


def _compute_patch_windows(
    batch_indices: list[int], missing: set[int], padding: int
) -> list[tuple[int, int]]:
    """Tính các CỬA SỔ liền kề (theo vị trí trong batch) để vá dòng thiếu kèm ngữ cảnh.

    Mỗi dòng thiếu được mở rộng thêm ``padding`` câu trước & sau (lấy từ chính batch),
    rồi các cửa sổ chồng/sát nhau được GỘP lại. Nhờ vậy câu thiếu được dịch lại CÙNG
    các câu lân cận → có đủ ngữ cảnh, đồng thời sửa luôn các câu kề bị lệch/lặp do câu
    thiếu gây ra.

    Args:
        batch_indices: Danh sách line_no của batch theo thứ tự (vị trí 0..n-1).
        missing: Tập line_no bị thiếu cần vá.
        padding: Số câu lân cận mỗi bên đưa vào vùng vá.

    Returns:
        Danh sách (start_pos, end_pos) — chỉ số VỊ TRÍ (nửa mở) trong batch, đã gộp.
    """
    if not missing or not batch_indices:
        return []
    pos_by_lineno = {ln: i for i, ln in enumerate(batch_indices)}
    n = len(batch_indices)
    raw: list[tuple[int, int]] = []
    for ln in missing:
        pos = pos_by_lineno.get(ln)
        if pos is None:
            continue
        start = max(0, pos - padding)
        end = min(n, pos + padding + 1)
        raw.append((start, end))
    if not raw:
        return []
    raw.sort()
    merged: list[tuple[int, int]] = [raw[0]]
    for start, end in raw[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:  # chồng hoặc sát → gộp
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _repair_truncated_json(txt: str) -> str | None:
    """[v3.23.142] Cứu JSON bị CẮT CỤT (model hết token output giữa chừng).

    Thuật toán CẮT-LÙI an toàn: quét một lượt (theo dõi trạng thái trong/ngoài chuỗi và
    stack ngoặc) để thu các "điểm cắt hợp lệ" — vị trí NGAY SAU một GIÁ TRỊ hoàn chỉnh
    (đóng ``}``/``]``, hết một chuỗi-giá-trị, hoặc hết số/true/false/null). Sau đó thử từ
    điểm cắt MỚI NHẤT lùi dần: cắt tới đó, đóng nốt ngoặc còn mở theo stack tại điểm ấy,
    rồi ``json.loads`` thử — trả về bản vá ĐẦU TIÊN parse được. KHÔNG cố sửa lỗi GIỮA cấu
    trúc (để tầng retry lo).

    Returns:
        Chuỗi JSON đã vá, hoặc None nếu không cứu được.
    """
    if not txt:
        return None
    # Mỗi phần tử: (vị_trí_cắt, danh_sách_ngoặc_đóng_cần_thêm) tại điểm value vừa hoàn chỉnh.
    cut_points: list[tuple[int, str]] = []
    stack: list[str] = []
    in_string = False
    escape = False
    i, n = 0, len(txt)
    while i < n:
        ch = txt[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
                # Chuỗi vừa đóng: là GIÁ TRỊ nếu ký tự kế (bỏ trắng) KHÔNG phải ':'.
                j = i + 1
                while j < n and txt[j] in " \t\r\n":
                    j += 1
                if j >= n or txt[j] != ":":
                    cut_points.append((i, "".join(reversed(stack))))
            i += 1
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if stack:
                stack.pop()
            cut_points.append((i, "".join(reversed(stack))))
        elif ch in "0123456789.-tfn":
            # Bắt đầu một số/true/false/null: nuốt tới hết literal rồi ghi điểm cắt.
            k = i
            while k + 1 < n and txt[k + 1] not in ",}]  \t\r\n":
                k += 1
            cut_points.append((k, "".join(reversed(stack))))
            i = k + 1
            continue
        i += 1

    # Thử từ điểm cắt mới nhất lùi dần (giới hạn số lần để không tốn kém).
    for pos, closers in reversed(cut_points[-200:]):
        candidate = txt[: pos + 1].rstrip()
        if candidate.endswith(","):
            candidate = candidate[:-1].rstrip()
        candidate += closers
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            continue
    return None


def _repair_json_text(sanitized: str) -> tuple[Any, str] | None:
    """[v3.23.143] Sửa JSON hỏng theo thứ tự ưu tiên CHẤT LƯỢNG rồi mới đến retry.

    1. ``json_repair`` (thư viện chuyên dụng): xử lý thiếu phẩy/ngoặc/nháy GIỮA cấu trúc,
       cắt cụt, nháy đơn, khóa trần, prose thừa... — mạnh và nhanh nhất.
    2. Bộ vá cắt cụt nội bộ ``_repair_truncated_json`` (zero-dependency): phòng khi thư viện
       chưa cài hoặc trả rỗng.

    Chỉ nhận kết quả KHÔNG rỗng (dict/list có dữ liệu) — nếu cả hai bó tay hoặc chỉ ra rỗng,
    trả None để tầng gọi RETRY (gen lại) thay vì nuốt dữ liệu rỗng.

    Returns:
        (data, tên_phương_pháp) nếu sửa được & không rỗng; None nếu bó tay.
    """
    if _json_repair is not None:
        try:
            # [v3.23.144] Chỉ gọi khi json.loads ĐÃ thất bại (chuỗi chắc chắn hỏng) ->
            # truyền skip_json_loads=True để BỎ bước json.loads dư thừa bên trong thư viện
            # (đúng khuyến nghị README json_repair, tránh antipattern loads-2-lần). Dùng
            # .loads() trả OBJECT nên ký tự CJK/Việt được giữ nguyên (không cần ensure_ascii).
            data = _json_repair.loads(sanitized, skip_json_loads=True)
        except TypeError:
            data = _json_repair.loads(sanitized)  # thư viện quá cũ, không có tham số
        except (ValueError, RecursionError):
            data = None
        if data:  # dict/list không rỗng
            return data, "json_repair"

    repaired = _repair_truncated_json(sanitized)
    if repaired is not None:
        try:
            data = json.loads(repaired)
        except json.JSONDecodeError:
            data = None
        if data:
            return data, "bộ vá nội bộ"

    return None


def _sanitize_json_text(raw_text: str) -> str:
    """[v3.20.3 #3] Bóc tách JSON khỏi văn bản "lắm lời" của AI (Hallucination Shield).

    Model đôi khi bọc JSON trong rào ```json … ``` HOẶC thêm lời dẫn ("Dạ, kết quả
    đây: {...}", "json { ... }"). Hàm:
      1. Bỏ rào markdown ```json / ``` nếu có.
      2. Nếu phần còn lại không bắt đầu bằng ``{`` hoặc ``[``, QUÉT SÂU tìm khối
         JSON ngoài cùng: từ dấu ``{``/``[`` ĐẦU TIÊN tới ``}``/``]`` TƯƠNG ỨNG
         cuối cùng — đủ để ``json.loads`` ăn được, chặn ``JSONDecodeError`` làm sập
         tiến trình dịch.

    Args:
        raw_text: Văn bản thô từ model.

    Returns:
        Chuỗi đã bóc, kỳ vọng là JSON hợp lệ (việc validate để ``json.loads`` lo).
    """
    txt = raw_text.strip()
    # Bước 1: gỡ rào markdown (mọi biến thể ```json / ```JSON / ```).
    fence = re.match(r"^```[a-zA-Z]*\s*", txt)
    if fence:
        txt = txt[fence.end():]
        if txt.rstrip().endswith("```"):
            txt = txt.rstrip()[:-3]
    txt = txt.strip()

    # Bước 2: quét sâu — lấy khối ngoài cùng từ dấu mở đầu tiên tới dấu đóng
    # tương ứng cuối cùng (cắt cả lời dẫn ĐẦU lẫn rác ĐUÔI sau JSON).
    first_obj, first_arr = txt.find("{"), txt.find("[")
    candidates = [i for i in (first_obj, first_arr) if i >= 0]
    if not candidates:
        return txt  # không có JSON → để json.loads báo lỗi rõ ràng phía gọi
    start = min(candidates)
    close_char = "}" if txt[start] == "{" else "]"
    end = txt.rfind(close_char)
    if end > start:
        return txt[start : end + 1].strip()
    return txt[start:].strip()


def _wrap_xml_block(tag: str, payload: Any) -> str:
    content = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return f"<{tag}>\n{content}\n</{tag}>"


class _BatchValidationError(Exception):
    """Lỗi nội bộ: model trả batch sai cấu trúc — kích hoạt retry chuyên biệt."""


class _EmptyResponseError(SubtitleTranslationError):
    """[v3.23.358] Gemini trả về KHÔNG có text (``candidates=None`` / bị safety filter /
    finish_reason=OTHER). Thường là NHẤT THỜI — nhất là khi đính video (video đôi khi
    kích hoạt safety filter hoặc khiến model trả rỗng). Đánh dấu riêng để RETRY (và BỎ
    video ở các lần thử sau) thay vì để một lô làm HỎNG CẢ JOB dịch như trước.

    [v3.23.591] Mang theo LÝ DO máy chủ nêu (xem ``empty_response_policy``): bị chặn thì
    bỏ video ngay, không nêu lý do thì thử lại một lần vẫn kèm video.

    Args:
        message: Mô tả lỗi.
        diagnosis: Lý do máy chủ nêu; mặc định "không nêu lý do".
    """

    def __init__(self, message: str, diagnosis: EmptyResponse | None = None) -> None:
        super().__init__(message)
        self.diagnosis = diagnosis or EmptyResponse()


class _TruncatedOutputError(SubtitleTranslationError):
    """[v3.23.622] Phản hồi dừng vì ``MAX_TOKENS`` (token suy nghĩ ăn hết ngân sách đầu ra).

    Tài liệu thinking.md: lúc đó đầu ra bị cụt hoặc rỗng nhưng vẫn tính tiền token suy nghĩ;
    cách chữa là HẠ ``thinking_level``. Trước đây phản hồi rỗng này bị coi là "rỗng không rõ
    lý do" và sau lần thứ hai ứng dụng BỎ VIDEO (mất căn cứ gán người nói), còn JSON cụt của
    khâu phân tích thì bị json_repair vá lặng lẽ thành "kết quả đầy đủ".
    """


class _BatchPartialError(Exception):
    """[v3.23.11] Model trả ĐỦ phần lớn dòng, chỉ THIẾU ít — vá riêng phần thiếu.

    Thay vì retry cả batch (model thường lại bỏ quên đúng các dòng đó → lãng phí ~10
    retry) hoặc halving cả batch, ta giữ phần đã dịch được và chỉ gọi lại cho các
    line_no còn thiếu. Mang theo data đã nhận + danh sách line_no thiếu.
    """

    def __init__(self, data: dict[str, Any], missing_line_nos: list[int]) -> None:
        super().__init__(f"Thiếu {len(missing_line_nos)} dòng: {missing_line_nos[:5]}")
        self.data = data
        self.missing_line_nos = missing_line_nos


class _BatchCountMismatchError(SubtitleTranslationError):
    """Model liên tục trả sai số dòng sau tất cả retry — kích hoạt halving đệ quy.

    Khác ``_BatchValidationError`` (lỗi tạm thời trong retry) — lỗi này báo hiệu
    rằng batch hiện tại QUÁ LỚN so với giới hạn mô hình và cần được chia đôi.
    """


# Schema phân tích ngữ cảnh toàn cục
def _clean_speaker_label(value: Any) -> str:
    """[v3.23.493] Chuẩn hoá nhãn người nói/người nghe của Visual Cues về TÊN GỐC thuần.

    Prompt (v3.23.490) yêu cầu ``spk``/``to`` chỉ ghi tên gốc để khớp máy móc với roster, nhưng
    model vẫn có thể kèm phiên âm hoặc chức danh (``"林逸（Lâm Dật）"``, ``"Tổng giám đốc 林逸"``).
    Lớp phòng vệ này bỏ phần trong ngoặc và quy dấu câu full-width về ASCII, để khâu dịch tra
    bảng quy đổi tên không bị trượt.

    Args:
        value: Giá trị thô từ JSON của model.

    Returns:
        Nhãn đã làm sạch (có thể rỗng).
    """
    from subtitles_extractor.application.services.roster_format import normalize_punctuation

    text = normalize_punctuation(str(value or "")).strip()
    if not text:
        return ""
    # Bỏ mọi cụm chú thích trong ngoặc: phiên âm, kênh thoại, ghi chú thêm.
    text = re.sub(r"\s*\([^)]*\)", "", text)
    return text.strip(" -–—:·,").strip()


def _segment_fields_directive(target_lang: str) -> str:
    """[v3.23.566] Chỉ thị tách vai trò 'overview' / 'segment_summary' khi phân tích đoạn.

    Không có chỉ thị này, model trả lại tóm tắt TOÀN PHIM từ đầu ở mỗi đoạn (đúng như được
    yêu cầu "hợp nhất thành bản cập nhật đầy đủ") — ca thật: 7/13 đoạn văn của tóm tắt mở
    đầu y hệt nhau, phần giữa phim bị cắt vì chạm trần độ dài.

    Args:
        target_lang: Ngôn ngữ viết tóm tắt.

    Returns:
        Đoạn chỉ thị nối vào cuối prompt của mỗi đoạn.
    """
    return (
        "\n\nHAI TRƯỜNG TÓM TẮT (BẮT BUỘC tách bạch):\n"
        "• \"segment_summary\": diễn biến CHI TIẾT CHỈ trong đoạn video đang xem, theo "
        "thứ tự thời gian (ai làm gì, với ai, vì sao, thái độ), viết bằng "
        f"{target_lang}, "
        "dùng tên phiên âm. KHÔNG kể lại chuyện của các đoạn trước.\n"
        "• \"overview\": bản toàn cảnh NGẮN GỌN của phim TỚI THỜI ĐIỂM NÀY (thế giới "
        "quan, tuyến nhân vật, mạch chính) — sẽ THAY THẾ bản cũ, nên viết trọn vẹn nhưng "
        "súc tích, không liệt kê lại chi tiết từng cảnh."
    )


def analysis_source_lines(
    source_lines: list[TranslationLine], keep_repeats: bool
) -> list[TranslationLine]:
    """[v3.23.603] Dòng phụ đề gửi khâu phân tích ngữ cảnh (hàm thuần).

    Luôn bỏ dòng rỗng. Dòng LẶP Y HỆT dòng liền trước (OCR tách một câu thành hai sự
    kiện, hoặc nhân vật nói lặp) chỉ bỏ khi không cần Visual Cues: khâu phân tích trả
    nhãn người nói theo ``id`` của dòng ĐƯỢC GỬI, nên dòng lặp bị bỏ không bao giờ có
    nhãn. Lượt thật 11/10 (灭门惨案II, gemini-3.8-flash): 6/6 dòng phải hỏi lại người
    nói chính là 6 dòng lặp bị bỏ (207, 217, 356, 366, 455, 714) — bước hỏi lại tốn 5
    phút 36 giây và 4 lời gọi (cộng các lượt 503 thử lại) của model chỉ có 20 lượt/ngày;
    dòng 207 còn bị gán người nói khác hẳn dòng lặp của nó. Phim khác: 10/3902 và 31/5540
    dòng lặp, đều chỉ có nhãn nhờ quét bổ sung về sau.

    Args:
        source_lines: Dòng phụ đề gốc.
        keep_repeats: True khi cần nhãn người nói cho từng dòng (Visual Cues).

    Returns:
        Danh sách dòng giữ nguyên thứ tự.
    """
    kept: list[TranslationLine] = []
    previous_text = ""
    for line in source_lines:
        text = (line.text or "").strip()
        if not text or (text == previous_text and not keep_repeats):
            continue
        kept.append(line)
        previous_text = text
    return kept


def _segment_cue_lines_directive(
    lines: list[TranslationLine], start_sec: float, end_sec: float
) -> str:
    """[v3.23.596] Danh sách dòng phụ đề của ĐÚNG đoạn video đang phân tích, kèm mốc giờ.

    Prompt phân tích chỉ có ``số. chữ`` của cả phim, không mốc thời gian — model phải tự
    dò trong đoạn video ~6 phút xem dòng nào thuộc đoạn và hiện lúc nào. Lượt thật
    10/10/2026 (phim "灭门惨案II", 17 đoạn): khâu phân tích trả nhãn cho 446/1245 dòng
    (36%); đoạn có ≤ 53 dòng gần như đủ, đoạn 100+ dòng chỉ 8-38 dòng — 800 dòng còn lại
    phải quét bổ sung bằng model nhỏ (16 lời gọi, 18 phút), mà nhãn quét bổ sung sai người
    nghe 29% so với 7% của khâu phân tích (đo ở v3.23.581). Bộ quét bổ sung đã gửi mốc
    giờ từ v3.23.581; khâu phân tích nay cũng vậy.

    Args:
        lines: Dòng phụ đề gửi phân tích (:func:`analysis_source_lines`; mốc mili-giây).
        start_sec: Mốc đầu đoạn video (giây).
        end_sec: Mốc cuối đoạn video (giây).

    Returns:
        Đoạn chỉ thị nối vào prompt của đoạn; rỗng nếu đoạn không có dòng nào.
    """
    from subtitles_extractor.application.services.visual_cue_batching import (
        lines_in_clip,
        scan_line_items,
    )

    segment = lines_in_clip(lines, start_sec, end_sec)
    if not segment:
        return ""
    return (
        f"\n\nCÁC DÒNG PHỤ ĐỀ HIỆN TRONG ĐOẠN VIDEO NÀY ({len(segment)} dòng; \"t\" = "
        "thời điểm dòng hiện, tính từ ĐẦU đoạn video đang xem): trả \"cues\" cho TỪNG id "
        "dưới đây — xem đúng khung hình và nghe đúng giọng tại \"t\" để nhận người nói "
        "và người nghe. Không nhận ra người nói thì ghi nhãn theo GIỌNG (\"Giọng nam "
        "chưa rõ\"…) chứ không để trống \"spk\"; \"to\" không chắc thì để trống. Đừng "
        "bỏ qua dòng nào, kể cả khi đoạn video dài.\n"
        + _wrap_xml_block("segment_lines", scan_line_items(segment, start_sec))
    )


def _speaker_fallback_rule() -> str:
    """[v3.23.600] Quy tắc người nói cho phân tích ngữ cảnh và bộ quét Visual Cues.

    Bản trước dặn "không chắc thì để trống" — lồng tiếng không biết đọc dòng trống bằng
    giọng nào (lượt mới nhất của hai phim thật: 103/3902 và 125/5540 dòng). Model vẫn NGHE
    được giọng nam/nữ/trẻ em, một người hay đám đông: ghi đúng điều đó bằng nhãn dự phòng
    cố định (``domain.value_objects.speaker_label``), không bịa danh tính.

    Returns:
        Đoạn chỉ thị.
    """
    voices = "', '".join(VOICE_ONLY_SPEAKERS)
    return (
        "\"spk\" KHÔNG được để trống — mọi dòng có lời đều phải lồng tiếng được. Nhận "
        "ra người nói thì ghi tên theo quy tắc tên. KHÔNG nhận ra danh tính thì ghi ĐÚNG "
        f"MỘT nhãn theo GIỌNG nghe được: '{voices}'; nhiều người cùng nói: "
        f"'{CROWD_SPEAKER}'; lời kể/thuyết minh ngoài câu chuyện: '{NARRATOR_SPEAKER}'. "
        "TUYỆT ĐỐI KHÔNG đoán bừa danh tính: chỉ ghi tên khi nhận ra người nói qua "
        "hình hoặc giọng. \"to\" không chắc thì để trống."
    )


def _visual_cues_system_instruction(target_lang: str, characters: str) -> str:
    """[v3.23.562] Chỉ thị hệ thống cho bộ quét Visual Cues (hàm thuần).

    Bản cũ không đưa bảng nhân vật và nêu ví dụ vai vế cổ trang ('Nam đệ tử', 'Nữ tỳ',
    'Sư phụ'). Đo thật trên một phim đô thị hiện đại: 95% nhãn là vai vế chung chung,
    riêng "Nữ tỳ" 208 dòng, "Nam đệ tử" 131 — model chép nguyên ví dụ. Nay: đưa bảng nhân
    vật, bắt ghi TÊN GỐC đúng như bảng (cùng quy tắc với phân tích ngữ cảnh), bỏ ví dụ.

    Args:
        target_lang: Ngôn ngữ đích (viết phần mô tả ``cue`` và nhãn vai trò).
        characters: Bảng nhân vật; rỗng thì chỉ yêu cầu nhãn vai trò nhất quán.

    Returns:
        Chuỗi system instruction.
    """
    parts = [
        "Bạn là GIÁM SÁT VIDEO (Vision Director). Dựa trên video kèm theo và danh sách "
        "phụ đề, với MỖI dòng hãy xác định ba điều: ai đang nói (spk), nói với ai (to), "
        "và bối cảnh + thái độ/cảm xúc ngắn gọn (cue).",
    ]
    if (characters or "").strip():
        parts.append(
            "BẢNG NHÂN VẬT của phim (tên gốc đứng đầu mỗi dòng, phiên âm trong ngoặc):\n"
            f"{characters.strip()}\n"
            "QUY TẮC TÊN (BẮT BUỘC): khi người nói/người nghe là nhân vật trong bảng, "
            "\"spk\"/\"to\" PHẢI ghi đúng TÊN GỐC như đầu dòng của bảng (vd '宋钱'), "
            "không ghi phiên âm, không thêm chức danh. Chỉ người KHÔNG có trong bảng mới "
            f"dùng nhãn vai trò ngắn bằng {target_lang}, và giữ nguyên nhãn đó ở mọi "
            "dòng."
        )
    else:
        parts.append(
            f"Ghi người nói/người nghe bằng nhãn vai trò ngắn bằng {target_lang}, đúng "
            "bối cảnh của chính phim này, và giữ nguyên một nhãn cho một người ở mọi "
            "dòng."
        )
    # [v3.23.581] Mốc thời gian + dòng tham khảo: quét riêng các dòng thiếu nhãn mà không
    # có hai thứ này thì model gán cả cuộc gọi Lâm Dật - chị dâu cho một cảnh khác.
    parts.append(
        "Trường t (nếu có) là thời điểm dòng bắt đầu hiện, tính từ ĐẦU đoạn video kèm "
        "theo (MM:SS): hãy xem đúng khung hình quanh thời điểm đó (phụ đề cứng trên hình "
        "khớp với text) để biết ai đang nói. Khối <known> (nếu có) là các dòng xen kẽ ĐÃ "
        "biết người nói/người nghe — chỉ dùng để theo mạch hội thoại, KHÔNG trả lại "
        "chúng. Một dòng nằm giữa hai dòng known của CÙNG một cặp nhân vật thường thuộc "
        "chính cuộc nói chuyện đó; chỉ gán cho người khác khi hình/tiếng cho thấy rõ "
        "người khác chen vào."
    )
    # [v3.23.602] Lượt thật (gemini-3.5-flash-lite, hỏi lại 39 dòng thiếu người nói): các
    # câu của đám dân làng được chia LẦN LƯỢT cho ba tên trong bảng (胖公安, 老伯,
    # 朱正軍…), kể cả một câu nói tách qua ba dòng thành ba người.
    parts.append(
        "Dân làng, đám đông, người qua đường… KHÔNG có trong bảng nhân vật: KHÔNG mượn "
        "tên một nhân vật trong bảng chỉ vì người đó có mặt trong cảnh, KHÔNG chia lần "
        "lượt các tên trong bảng cho một loạt dòng — một người không tên ghi nhãn GIỌNG, "
        f"nhiều người ghi '{CROWD_SPEAKER}'. Một câu nói bị tách qua nhiều dòng liền "
        "nhau (dòng sau nói tiếp ý dòng trước) là của CÙNG một người."
    )
    parts.append(
        "KHÔNG bịa chức danh/vai vế không có trong phim. "
        f"{_speaker_fallback_rule()} "
        "Trả JSON rút gọn theo schema, mỗi phần tử gồm id (số dòng), spk, to, cue. "
        "BẮT BUỘC giữ nguyên id, đủ mọi dòng trong <lines>, không gộp/bỏ dòng."
    )
    return " ".join(parts)


def _batch_glossary_block(batch: list[Any], glossary: str) -> list[dict[str, Any]]:
    """[v3.23.500] Lọc các thuật ngữ THỰC SỰ xuất hiện trong lô để nhắc lại ngay cạnh dòng dịch.

    Cả bảng glossary nằm trong ``system_instruction`` khiến model phân tán chú ý và bỏ qua phần
    lớn (đo trên bản dịch thật: 72/113 thuật ngữ bị dịch khác quy định). Lọc theo lô giúp danh
    sách ngắn, sát ngữ cảnh và dễ tuân thủ; lô không chứa thuật ngữ nào thì không tốn token.

    Args:
        batch: Các dòng nguồn của lô hiện tại (mỗi dòng có thuộc tính ``text``).
        glossary: Bảng thuật ngữ quy cách cố định, mỗi dòng ``gốc = bản dịch``.

    Returns:
        Danh sách giới hạn số mục để không phình prompt, mỗi mục
        ``{"term": ..., "must_translate_as": ...}``; mục có nhiều phương án thì phương án
        đầu là ``must_translate_as``, các phương án còn lại nằm trong ``also_acceptable``.
    """
    if not glossary:
        return []
    source_text = "\n".join(str(getattr(line, "text", "") or "") for line in batch)
    if not source_text:
        return []
    from subtitles_extractor.application.services.glossary_options import term_renderings

    hits: list[dict[str, Any]] = []
    # [v3.23.586] Chỉ đưa phần DÙNG TRONG LỜI THOẠI. Trước đây cả chú thích trong ngoặc
    # nằm trong "must_translate_as" nên model chép luôn vào câu dịch — 459 dòng dịch thô
    # trên 50 chuẩn đoán thật ("120 (cấp cứu y tế)"). Chú thích vẫn có trong bảng thuật
    # ngữ đầy đủ ở system_instruction để model hiểu nghĩa.
    # [v3.23.587] Tách hai vế ``A / B`` (xem ``glossary_options``): vế dịch nhiều phương
    # án thành DANH SÁCH để model chọn một thay vì chép cả chuỗi "bốc đồng / manh động";
    # vế gốc nhiều cách viết thành từng thuật ngữ riêng — trước đây "老山 / 魯山" được tìm
    # nguyên chuỗi trong câu gốc nên không bao giờ khớp.
    # [v3.23.589] Mục nhiều phương án vẫn phải mang khoá "must_translate_as". Khối này
    # không có lời giải thích nào khác: TÊN KHOÁ chính là chỉ thị. Cùng bảng thuật ngữ,
    # cùng model, ba lượt chạy thật: khoá "must_translate_as": "tự sướng / thủ dâm"
    # (v3.23.586) thì 性自樂 ra "tự sướng" 4/4 dòng; khoá "must_translate_as_one_of"
    # (v3.23.587, v3.23.588) thì cả hai lượt ra "tính tự lạc" (đọc âm Hán-Việt) 4/4
    # dòng, và số lượt sai của các mục nhiều phương án ở dịch thô tăng 1/40 → 6/40.
    # [v3.23.615] So không phân biệt phồn/giản: OCR đọc lẫn chữ phồn ("楚瑤") vào phụ đề
    # giản thể, còn bảng thuật ngữ do model viết ghi chữ giản.
    from subtitles_extractor.application.services.han_variants import fold_han_variants
    from subtitles_extractor.application.services.term_variants import VariantIndex

    folded_source = fold_han_variants(source_text)
    renderings = term_renderings(glossary)
    known_terms = frozenset(fold_han_variants(r.term) for r in renderings)
    index = VariantIndex(folded_source)
    # [v3.23.645] Tên tông môn viết lệch một chữ (灵霄宗 / 凌霄宗): bảng chỉ giữ một dạng, dạng
    # kia vẫn phải nhận cùng cách dịch, nếu không cùng tông môn có hai tên Việt. Gợi ý biến thể
    # xếp SAU mọi thuật ngữ thật (không chiếm chỗ trong giới hạn), mỗi dạng chỉ một gợi ý, và
    # mang khoá ``variant_of`` để model tự cân nhắc đó có đúng là cùng tổ chức không.
    variant_hits: list[dict[str, Any]] = []
    seen_variants: set[str] = set()
    for rendering in renderings:
        term, options = rendering.term, rendering.options
        if len(term) < 2:
            continue
        folded_term = fold_han_variants(term)
        if folded_term in folded_source and len(hits) < _MAX_BATCH_GLOSSARY_TERMS:
            hit: dict[str, Any] = {"term": term, "must_translate_as": options[0]}
            if len(options) > 1:
                hit["also_acceptable"] = list(options[1:])
            hits.append(hit)
        for variant in index.variants_of(folded_term, known_terms):
            if variant in seen_variants:
                continue
            seen_variants.add(variant)
            variant_hit: dict[str, Any] = {
                "term": variant, "must_translate_as": options[0], "variant_of": term,
            }
            variant_hits.append(variant_hit)
    hits.extend(variant_hits[: max(0, _MAX_BATCH_GLOSSARY_TERMS - len(hits))])
    return hits


def _merge_roster_text(previous: str, current: str) -> str:
    """[v3.23.489] Hợp nhất bảng nhân vật tích luỹ (uỷ quyền cho ``roster_format``)."""
    from subtitles_extractor.application.services.roster_format import merge_roster

    return merge_roster(previous, current)


def _merge_glossary_text(previous: str, current: str) -> str:
    """[v3.23.489] Hợp nhất bảng thuật ngữ tích luỹ."""
    from subtitles_extractor.application.services.roster_format import merge_glossary

    return merge_glossary(previous, current)


def _merge_overview_text(previous: str, current: str) -> str:
    """[v3.23.489] Nối tiếp tóm tắt theo thời gian (thay cho GHI ĐÈ bằng đoạn cuối)."""
    from subtitles_extractor.application.services.roster_format import merge_overview

    return merge_overview(previous, current)


def _normalize_roster(value: Any) -> str:
    """[v3.23.488] Chuẩn hoá bảng nhân vật (mảng đối tượng HOẶC chuỗi cũ) về văn bản QUY CÁCH
    CỐ ĐỊNH — mỗi nhân vật đúng MỘT DÒNG ``Tên (Phiên âm) [giới, tuổi] — vai trò``.

    Nhờ vậy mọi khâu phía sau (giao diện, chuẩn đoán, ngữ cảnh nối tập) đọc TẤT ĐỊNH, không phải
    đoán định dạng tự do như trước.
    """
    from subtitles_extractor.application.services.roster_format import (
        parse_roster, render_roster,
    )

    return render_roster(parse_roster(value))


def _normalize_glossary(value: Any) -> str:
    """[v3.23.488] Chuẩn hoá bảng thuật ngữ về quy cách cố định — mỗi mục MỘT DÒNG ``gốc = dịch``."""
    from subtitles_extractor.application.services.roster_format import (
        drop_subsumed_glossary_entries,
        drop_variant_glossary_entries,
        parse_glossary,
        render_glossary,
        split_half_translated_entries,
    )

    # [v3.23.506] Khử BIẾN THỂ cùng nghĩa ngay khi nhận: hai mục cho cùng một khái niệm khiến
    # khâu dịch nhận chỉ thị mâu thuẫn (dữ liệu thật: 9 cặp).
    entries = drop_subsumed_glossary_entries(parse_glossary(value))
    entries = drop_variant_glossary_entries(entries)
    # [v3.23.549] Loại mục DỊCH DỞ (vế dịch lẫn chữ Hán, vd ``生死道消 = "thân tử đạo消"``). Giữ
    # lại thì prompt dịch mang một ví dụ DẠY MODEL để sót chữ Hán trong bản tiếng Việt — sai về
    # đúng thứ mà cờ ``cjk_leftover`` sinh ra để chặn.
    healthy, broken = split_half_translated_entries(entries)
    if broken:
        logger.warning(
            "Bảng thuật ngữ có %d mục DỊCH DỞ (vế dịch còn chữ Hán) — đã loại khỏi prompt: %s",
            len(broken),
            ", ".join(f"{e.term}={e.translation}" for e in broken[:5]),
        )
    return render_glossary(healthy)


_CONTEXT_ANALYSIS_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "source_lang": {"type": "STRING"},
        # [v3.23.488] CÓ CẤU TRÚC thay cho chuỗi tự do: trước đây model tự do viết mỗi lần một
        # kiểu (khi ngăn bằng phẩy, khi bằng chấm, mô tả lại chứa thêm dấu hai chấm) khiến MỌI
        # bộ đọc phía sau đều phải đoán và sai. Nay trả mảng đối tượng → chuẩn hoá tất định.
        "characters": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "name": {"type": "STRING"},
                    "aka": {"type": "STRING"},
                    "gender": {"type": "STRING"},
                    "age": {"type": "STRING"},
                    "role": {"type": "STRING"},
                },
                "required": ["name"],
            },
        },
        "overview":    {"type": "STRING"},
        # [v3.23.566] Diễn biến CHỈ của đoạn video đang xem (phân tích tuần tự theo đoạn).
        "segment_summary": {"type": "STRING"},
        # [v3.23.23/488] Bảng thuật ngữ để dịch NHẤT QUÁN xuyên suốt — nay là MẢNG đối tượng.
        "glossary": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "term": {"type": "STRING"},
                    "translation": {"type": "STRING"},
                },
                "required": ["term"],
            },
        },
    },
    "required": ["source_lang", "characters", "overview"],
}

# [v3.23.37] Schema phân tích ngữ cảnh KÈM Visual Cues — dùng khi bật phân tích hình
# ảnh, để LẤY LUÔN "ai nói/nói với ai" trong CÙNG request phân tích tuần tự (đã gửi
# video lên rồi) → tiết kiệm quota, KHÔNG cần gọi analyze_visual_cues riêng. Khoá 'cues'
# rút gọn (id/spk/to) để không chạm trần token output.
# [v3.23.622] ``cues`` và các trường của từng cue nay BẮT BUỘC (troubleshooting.md: "Make all
# output fields required"; mảng rỗng vẫn hợp lệ). Trước đây model được phép bỏ hẳn ``cues`` —
# thứ duy nhất khiến bật Visual Cues có ý nghĩa.
_CONTEXT_WITH_CUES_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "source_lang": {"type": "STRING"},
        # [v3.23.488] CÓ CẤU TRÚC thay cho chuỗi tự do: trước đây model tự do viết mỗi lần một
        # kiểu (khi ngăn bằng phẩy, khi bằng chấm, mô tả lại chứa thêm dấu hai chấm) khiến MỌI
        # bộ đọc phía sau đều phải đoán và sai. Nay trả mảng đối tượng → chuẩn hoá tất định.
        "characters": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "name": {"type": "STRING"},
                    "aka": {"type": "STRING"},
                    "gender": {"type": "STRING"},
                    "age": {"type": "STRING"},
                    "role": {"type": "STRING"},
                },
                "required": ["name"],
            },
        },
        "overview":    {"type": "STRING"},
        # [v3.23.566] Diễn biến CHỈ của đoạn video đang xem (phân tích tuần tự theo đoạn).
        "segment_summary": {"type": "STRING"},
        "glossary": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "term": {"type": "STRING"},
                    "translation": {"type": "STRING"},
                },
                "required": ["term"],
            },
        },
        "cues": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "id": {"type": "INTEGER"},
                    "spk": {"type": "STRING"},
                    "to": {"type": "STRING"},
                    "cue": {"type": "STRING"},
                },
                "required": ["id", "spk", "to", "cue"],
            },
        },
    },
    "required": ["source_lang", "characters", "overview", "cues"],
}

# [JSON Minification] Schema Visual Cues khoá rút gọn (id/spk/to/cue) để AI nhả
# được hàng ngàn dòng mà không chạm trần 8192 token output của Gemini.
_VISUAL_CUES_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "cues": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "id": {"type": "INTEGER"},
                    "spk": {"type": "STRING"},
                    "to": {"type": "STRING"},
                    "cue": {"type": "STRING"},
                },
                "required": ["id"],
            },
        },
    },
    "required": ["cues"],
}

# [v3.23.623] Schema gán người nói CHỈ BẰNG CHỮ: mọi trường bắt buộc (troubleshooting.md:
# "Make all output fields required"; chuỗi rỗng vẫn hợp lệ).
_TEXT_CUES_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "cues": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "id": {"type": "INTEGER"},
                    "spk": {"type": "STRING"},
                    "to": {"type": "STRING"},
                    "cue": {"type": "STRING"},
                },
                "required": ["id", "spk", "to", "cue"],
            },
        },
    },
    "required": ["cues"],
}

#: [v3.23.624] Bản vá tự kiểm tra phân tích: CHỈ phần cần sửa, không sinh lại toàn bộ.
_AUDIT_PATCH_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "characters": {"type": "ARRAY", "items": {"type": "STRING"}},
        "remove_characters": {"type": "ARRAY", "items": {"type": "STRING"}},
        "glossary": {"type": "ARRAY", "items": {"type": "STRING"}},
        "remove_glossary": {"type": "ARRAY", "items": {"type": "STRING"}},
        "overview": {"type": "STRING"},
        "cue_fixes": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "id": {"type": "INTEGER"},
                    "spk": {"type": "STRING"},
                    "to": {"type": "STRING"},
                },
                "required": ["id", "spk", "to"],
            },
        },
    },
    "required": ["characters", "remove_characters", "glossary", "remove_glossary",
                 "overview", "cue_fixes"],
}

#: [v3.23.637] Khiếm khuyết cấp bảng trên phim dài: tối đa ngần này dòng, chia thành ngần này đoạn.
AUDIT_NO_FOCUS_MAX_LINES = 1200
AUDIT_NO_FOCUS_CHUNKS = 6
#: Số dòng tối đa gửi kèm khi tự kiểm tra; dài hơn thì chỉ gửi vùng quanh dòng nghi lỗi.
AUDIT_FULL_EPISODE_LINES = 700


def _audit_system_instruction(target_lang: str) -> str:
    """[v3.23.624] Chỉ thị hệ thống cho bước tự kiểm tra phân tích (hàm thuần)."""
    return (
        "<role>\nBạn là biên tập viên kiểm định bản PHÂN TÍCH NGỮ CẢNH của một tập phim "
        "(bảng nhân vật, thuật ngữ, tóm tắt, nhãn người nói). Bản phân tích cũ đã có; việc của "
        "bạn là VÁ khiếm khuyết của nó dựa trên phụ đề gốc, KHÔNG viết lại từ đầu.\n</role>\n"
        "<nguyen_tac>\n"
        "- Chỉ trả những mục CẦN SỬA/THÊM/XOÁ; mục đúng thì không nhắc tới (để mảng rỗng).\n"
        "- Mỗi 'chi_tiet' trong <khiem_khuyet> là một nghi vấn đã đối chiếu máy móc: kiểm lại "
        "bằng phụ đề, chỉ sửa khi CHẮC CHẮN; nghi vấn sai thì bỏ qua.\n"
        "- 'characters': mỗi phần tử MỘT dòng đúng quy cách 'Tên gốc (Phiên âm) [giới, tuổi] — "
        "vai trò và cách xưng hô'. Tên gốc PHẢI trùng chữ gốc trong phụ đề. Dòng có tên đã tồn "
        "tại thì thay thông tin cũ.\n"
        "- 'glossary': 'gốc = bản dịch'. 'remove_*': tên/thuật ngữ gốc cần xoá (chỉ khi sai hẳn "
        "hoặc thừa).\n"
        "- 'overview': chỉ điền khi tóm tắt hiện có sai/thiếu ý chính, viết lại ĐẦY ĐỦ bằng "
        f"{target_lang}; còn lại để chuỗi rỗng.\n"
        "- 'cue_fixes': sửa người nói 'spk' / người nghe 'to' của dòng sai (dùng đúng TÊN GỐC "
        "trong bảng nhân vật); đọc cả mạch hội thoại, nhất là ai đang đáp lại ai.\n"
        "- Khiếm khuyết 'no_address_rules': trả LẠI dòng của từng nhân vật có thoại với phần "
        "'xưng hô: với <Tên người nghe> tự xưng '…' gọi '…'' (tên người nghe viết ĐÚNG như bảng, "
        "KHÔNG pinyin/chữ Hán xen vào câu tiếng Việt). Quy tắc chọn đại từ: "
        f"{ANALYSIS_ADDRESS_REGISTER_RULE}"
        "- Không bịa nhân vật/chi tiết không có trong phụ đề.\n"
        "</nguyen_tac>"
    )


#: [v3.23.629] Rút gọn câu dịch cho vừa khung đọc của TTS.
_SHORTEN_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "lines": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {"id": {"type": "INTEGER"}, "text": {"type": "STRING"}},
                "required": ["id", "text"],
            },
        },
    },
    "required": ["lines"],
}

#: Số dòng tối đa một lượt rút gọn (prompt nhỏ, ít sót dòng).
SHORTEN_BATCH_SIZE = 30


def _shorten_system_instruction(target_lang: str) -> str:
    """[v3.23.629] Chỉ thị hệ thống rút gọn câu dịch để giọng đọc kịp khung thời gian (hàm thuần)."""
    return (
        "<role>\nBạn là biên tập viên LỒNG TIẾNG. Mỗi dòng dưới đây đã được dịch nhưng quá dài so "
        "với khung thời gian giọng đọc: viết lại NGẮN HƠN bằng " + target_lang + ", giữ nguyên nghĩa "
        "và giọng điệu, để giọng đọc không phải đọc gấp.\n</role>\n"
        "<nguyen_tac>\n"
        "- Mỗi dòng có 'toi_da_am_tiet' (số tiếng/âm tiết tối đa). Viết sao cho KHÔNG vượt quá, "
        "nhưng đừng ngắn hơn cần thiết.\n"
        "- GIỮ cách xưng hô (tự xưng / gọi người nghe) và sắc thái của 'nguoi_noi' → 'nguoi_nghe'; "
        "không đổi ngôi, không đảo vai.\n"
        "- Cách rút gọn ưu tiên: bỏ từ đệm / lặp ý / trạng từ thừa; dùng từ ngắn thay cụm dài; "
        "đổi cấu trúc câu cho gọn. KHÔNG bỏ ý chính, KHÔNG thêm ý mới, KHÔNG bịa.\n"
        "- Giữ nguyên tên riêng, thuật ngữ trong 'nguon' đã có cách dịch ở các dòng kề, và tag "
        "người nói ở đầu dòng nếu có (dạng '[Tên:]').\n"
        "- NGHĨA quan trọng hơn độ ngắn: giữ nguyên từ khoá / thuật ngữ (vd 'nhiệm vụ', 'phong ấn', "
        "'sư huynh'), không đổi nghĩa động từ (vd 'phong ấn' ≠ 'phá phong ấn'), không đổi mức độ "
        "('đương nhiên chấp nhận' ≠ 'tất nhận đi'), không đổi 'chúng ta' thành 'đôi ta'. Muốn xem ngữ "
        "cảnh thì đối chiếu 'nguon'.\n"
        "- Nếu KHÔNG thể rút gọn mà vẫn giữ nguyên nghĩa và từ khoá, trả lại ĐÚNG câu cũ (không ép).\n"
        "- Trả đủ MỌI id đã cho, mỗi id một câu hoàn chỉnh, một dòng, không xuống dòng.\n"
        "- 'truoc' / 'sau' chỉ để hiểu mạch hội thoại, KHÔNG viết lại chúng.\n"
        "</nguyen_tac>"
    )


#: Số dòng tối đa của một lô gán người nói bằng chữ, và cửa sổ phụ đề ngữ cảnh quanh lô.
TEXT_CUES_BATCH_SIZE = 150
TEXT_CUES_FULL_EPISODE_LINES = 600
TEXT_CUES_WINDOW_LINES = 250


def _text_cues_system_instruction(
    target_lang: str, characters: str, has_voices: bool = False
) -> str:
    """[v3.23.623] Chỉ thị hệ thống gán người nói/người nghe CHỈ từ chữ (hàm thuần).

    Lỗi dịch nặng nhất còn lại là ĐẢO VAI (你/我 nhầm người). Khi không có video, các lưới
    xưng hô (quy tắc 'tự xưng … gọi …', kiểm tra đại từ) đều thiếu người nói/người nghe nên
    "mù": model dịch phải tự đoán trong cửa sổ 10 dòng. Bước này đọc CẢ TẬP một lần (phụ đề
    tiếng Trung hay lược chủ ngữ, nhưng cả mạch hội thoại thì lộ ra ai nói với ai) rồi gắn
    nhãn cho từng dòng để phần còn lại của đường ống dùng như cues từ video.

    Args:
        target_lang: Ngôn ngữ viết ``cue``.
        characters: Bảng nhân vật; rỗng thì dùng nhãn vai trò ngắn.
        has_voices: Phụ đề kèm nhãn giọng ``[v:N]`` do bộ nhận dạng giọng nói tách từ âm
            thanh (xem :func:`voice_labels_from_words`).

    Returns:
        Chuỗi system instruction.
    """
    voices = "', '".join(VOICE_ONLY_SPEAKERS)
    if (characters or "").strip():
        names = (
            "<ten>\nBảng nhân vật (tên gốc đứng đầu mỗi dòng):\n"
            f"{characters.strip()}\n"
            "\"spk\" và \"to\" PHẢI ghi đúng TÊN GỐC như đầu dòng của bảng (vd '宋钱'), không "
            "phiên âm, không thêm chức danh. Người KHÔNG có trong bảng thì ghi nhãn vai trò ngắn "
            f"bằng {target_lang} và giữ nguyên nhãn đó ở mọi dòng.\n</ten>"
        )
    else:
        names = (
            "<ten>\nGhi người nói/người nghe bằng nhãn vai trò ngắn bằng "
            f"{target_lang}, giữ nguyên một nhãn cho một người ở mọi dòng.\n</ten>"
        )
    voice_rules = (
        # [v3.23.633] Nhãn giọng là bằng chứng MẠNH nhưng bộ nhận dạng hay gộp nhầm nhiều người
        # có giọng giống nhau (đệ tử/đám đông/hệ thống) vào một nhãn. Đo 6 lần/tập trên phim mẫu:
        # nhóm đệ tử giễu nhân vật chính (dòng 71-80) luôn bị gán cho nhân vật chính (0/6 đúng)
        # → thêm bước kiểm nội dung với người đã gán: độ đúng cả tập 86,5% → 93,7%.
        "<giong_noi>\nMỗi dòng có nhãn giọng [v:N] do bộ nhận dạng giọng nói tách từ ÂM THANH. "
        "Cùng nhãn thường là cùng một người, nên lấy đó làm mặc định: gán mỗi nhãn cho một "
        "nhân vật (nhìn cả tập để biết nhãn là ai). NHƯNG bộ nhận dạng hay GỘP NHẦM nhiều người "
        "có giọng giống nhau (đặc biệt các vai phụ: đệ tử, đám đông, hệ thống) vào một nhãn, "
        "hoặc tách một người làm nhiều nhãn. Vì vậy với MỖI dòng hãy kiểm nội dung với người "
        "đã gán; nếu mâu thuẫn thì dòng đó KHÔNG thuộc người ấy dù cùng nhãn, và chọn người "
        "đúng theo lời thoại:\n"
        "  * người nói không tự xưng 你 với chính mình và không gọi tên mình trong lời thoại "
        "('叶然你还不明白吗' không thể do 叶然 nói — đây là người khác nói VỚI 叶然);\n"
        "  * câu có 你/你们 chê bai, giễu cợt, khoe người thứ ba ('楚师姐…而你呢…') là lời của "
        "kẻ đang đối đầu với người nghe, không phải độc thoại của người nghe;\n"
        "  * giới tính, vai vế, quan hệ và thái độ trong câu phải khớp với nhân vật được gán;\n"
        "  * một nhãn xuất hiện xen kẽ giữa nhiều cuộc thoại với các đối tượng khác nhau thì "
        "rất có thể đã gộp nhiều người.\n"
        "Dòng không có nhãn: suy từ chữ và mạch hội thoại. Nhãn giọng chỉ cho biết AI NÓI; "
        "người NGHE (\"to\") vẫn suy từ lời thoại và lượt đối đáp.\n</giong_noi>\n"
        if has_voices else ""
    )
    return (
        "<role>\nBạn gán NGƯỜI NÓI (spk) và NGƯỜI NGHE (to) cho từng dòng phụ đề của một tập "
        "phim, chỉ dựa vào CHỮ"
        f"{' và nhãn giọng nói' if has_voices else ' (chưa có video)'}.\n</role>\n"
        f"{voice_rules}"
        "<cach_suy_luan>\n"
        "- Phụ đề tiếng Trung hay lược chủ ngữ: hãy suy ra từ lời gọi, xưng hô và mạch hội thoại "
        "của CẢ TẬP chứ không chỉ dòng đang xét.\n"
        "- 我/我们 = người nói; 你/您/你们 = người nghe; 他/她 = người thứ ba.\n"
        "- Lời gọi đầu câu như '师兄，我…' nghĩa là người nói GỌI người nghe: người nghe là 师兄, "
        "người nói là người xưng 我 (thường là người gọi bằng 师妹/sư muội, hay là người ít vai "
        "vế hơn).\n"
        "- Hai người đối đáp thì lượt nói LUÂN PHIÊN; một câu bị tách qua nhiều dòng liền nhau "
        "là của CÙNG một người nói.\n"
        "- Độc thoại/suy nghĩ thầm: to = để trống. Lời dẫn truyện hoặc chữ trên màn hình: "
        f"spk = '{NARRATOR_SPEAKER}'. Lời của 'hệ thống' (nếu phim có) là nhân vật riêng.\n"
        "- Không chắc người NGHE thì để \"to\" trống; KHÔNG bịa. Người nói không rõ danh tính: "
        f"ghi ĐÚNG MỘT nhãn theo giọng suy ra từ chữ ('{voices}'); nhiều người cùng nói: "
        f"'{CROWD_SPEAKER}'.\n"
        "</cach_suy_luan>\n"
        f"{names}\n"
        "<dinh_dang>\nTrả JSON theo schema: mỗi phần tử gồm id, spk, to, cue. \"cue\" là mô "
        f"tả RẤT NGẮN (≤ 6 từ, {target_lang}) về hành động nói (ra lệnh, mỉa mai, thì thầm, "
        "nghĩ thầm…) CHỈ khi lời thoại cho thấy rõ, nếu không để trống. Giữ nguyên id, đủ MỌI "
        "id trong <lines_to_label>, không gộp/bỏ dòng.\n</dinh_dang>"
    )


# Số dòng giữ lại khi phân tích context (đã chuyển vào hàm analyze_global_context)
# _CONTEXT_ANALYSIS_MAX_SAMPLES đã bỏ — nay dùng toàn bộ + hard-cap 8000.


# [v3.23.633] GIỮ VAI & NGÔI ở các giai đoạn TINH CHỈNH. Chấm mù tập 1-3 phim mẫu: chuỗi dịch thô →
# phong cách → bản địa hoá có lỗi nặng toàn do ĐẢO NGÔI ("当年你是天纵之资" → "Năm xưa ta là thiên
# tài"): nhãn speaker/cue máy gán sai (đệ tử bị gán là nhân vật chính "tự nhủ") kéo model coi câu
# là độc thoại. Lặp 3-4 lần/cấu hình: lỗi đảo ngôi + gọi sai "huynh" trên dòng bẫy 1,67 → 0,67
# mỗi lượt khi dặn model tin chữ gốc hơn nhãn; cả chuỗi 3 giai đoạn (6 bảng chấm mù): lỗi nặng
# 8 → 1, điểm 56 → 21. KHÔNG áp quy tắc này cho giai đoạn dịch thô: 8 bảng chấm mù cho điểm
# 76 (có quy tắc) so với 47 (không có) — nó ép "ngươi" cả khi 你 hướng tới bề trên (sư thúc).
PERSON_ANCHOR_RULE_REFINE = (
    "- ƯU TIÊN SỐ 1 — GIỮ NGUYÊN VAI & NGÔI: 'text' của khâu trước đã được dịch từ chữ gốc và "
    "đại từ nhân xưng trong đó (ta/tôi/ngươi/huynh/muội/anh/em…) là kết quả ĐÃ KIỂM. Chỉ đổi một "
    "đại từ khi 'original' chứng minh RÕ RÀNG nó sai. Quy tắc cứng theo chữ gốc: 你/您/你们 là "
    "NGÔI THỨ HAI (người nghe) — TUYỆT ĐỐI không viết thành ngôi thứ nhất ('ta', 'tôi', "
    "'chúng ta'); 我/咱/我们 là NGÔI THỨ NHẤT. Ví dụ cấm: 'original' '你们这些人' + 'text' "
    "'Lũ các ngươi' → KHÔNG được sửa thành 'Lũ chúng ta'.\n"
    "- Nhãn 'speaker', 'addressee', 'cue' do máy tự gán và đôi khi SAI (nhất là vai phụ, đám "
    "đông, giọng hệ thống). Khi chúng mâu thuẫn với chữ trong 'original' hoặc ngôi trong 'text' "
    "thì tin 'original' và 'text'. Không đổi ngôi/đại từ chỉ vì nhãn 'cue' (vd 'tự nhủ', 'nghĩ "
    "thầm') hay 'speaker'.\n"
)


# [v3.23.634] Quy tắc xưng hô của bản PHÂN TÍCH. Đo trên tập 1-3 phim mẫu (12 lượt, chấm tự động
# theo đáp án, gemini-3.1-flash-lite, chỉ chữ): ví dụ cũ trong prompt dùng 'em'/'anh' nên phim tiên
# hiệp hay ra "Diệp Nhiên tự xưng 'anh' gọi 'muội'", "Dao Dao tự xưng 'em'"; ngoài ra ghi nhiều
# phương án ('hoặc', '/'), bịa tên chưa từng có trong phụ đề ("Chu Dao", "Triệu Phàm") và trộn chữ
# Hán vào chữ Việt ("Nam宫梦", "sư叔"). Số lỗi trên 12 lượt: 32 → 21; có video + gemini-3.8-flash
# (tập 1-2): 14 → 8; tập giữ riêng 5-9 (chấm mù): 64 → 61 điểm, không hồi quy.
ANALYSIS_ADDRESS_REGISTER_RULE = (
    "ĐẠI TỪ THEO THỂ LOẠI (quyết định bởi thế giới quan của phim, nhận ra từ nội dung): phim cổ "
    "trang/kiếm hiệp/tiên hiệp/huyền huyễn dùng bộ đại từ cổ phong — tự xưng 'ta', 'đệ', 'muội', "
    "'huynh', 'đồ nhi', 'đệ tử', 'con'; gọi 'ngươi', 'huynh', 'muội', 'tỷ', 'đệ', 'sư phụ', 'sư thúc', "
    "'con' — TUYỆT ĐỐI không dùng 'anh', 'em', 'cô', 'tôi', 'bạn' cho phim loại này; phim hiện đại "
    "thì ngược lại. 'Tự xưng' và 'gọi' PHẢI là từ xưng hô thật dùng được trong lời thoại, không phải "
    "chức danh chung chung hay mô tả quan hệ.\n"
    "VAI VẾ QUYẾT ĐỊNH TỰ XƯNG: người DƯỚI nói với bậc TRÊN (đệ tử → sư phụ/sư tôn/sư "
    "thúc/sư bá/chưởng giáo; con → cha mẹ; thuộc hạ → chủ) tự xưng bằng danh xưng khiêm — 'đệ tử', "
    "'đồ nhi', 'con', 'thuộc hạ' — KHÔNG tự xưng 'ta' hay 'tôi', và gọi bằng chức danh ('sư thúc', "
    "'sư phụ'). Bậc TRÊN nói với người dưới tự xưng 'ta' (hoặc 'vi sư', 'bổn tọa') và gọi 'ngươi' hoặc "
    "'đồ nhi'. Vai vế lấy từ LỜI THOẠI (họ gọi nhau là sư thúc/đệ tử/sư phụ), không lấy từ tuổi hay "
    "tu vi (nhân vật chính dù mạnh đến đâu vẫn là đệ tử khi nói với sư thúc).\n"
    "ĐÚNG MỘT phương án: SAI → gọi 'muội/Lâm Nhi', tự xưng 'sư huynh' hoặc 'ta'. ĐÚNG → gọi 'muội' "
    "(danh xưng: Lâm Nhi), tự xưng 'ta'. Đây là cách XƯNG HÔ mặc định của cặp; cách gọi tên riêng chỉ "
    "ghi trong ngoặc.\n"
    "CHỈ GHI TÊN CÓ THẬT TRONG PHỤ ĐỀ: nhân vật không được gọi tên trong phụ đề thì dùng danh xưng "
    "NGẮN đúng như lời thoại gọi ('Sư phụ', 'Đệ tử'), TUYỆT ĐỐI KHÔNG tự đặt tên riêng chưa từng "
    "xuất hiện. Trong câu tiếng Việt (role, mệnh đề 'với…') viết tên theo MỘT dạng trọn vẹn — toàn bộ "
    "bằng chữ Việt (aka) — KHÔNG trộn chữ Hán xen vào từ tiếng Việt (SAI: 'Lâm峰', 'sư叔'; ĐÚNG: "
    "'Lâm Phong', 'sư thúc'). Mỗi cặp ĐÚNG MỘT đại từ, KHÔNG dùng 'hoặc' hay '/'.\n"
)


class GeminiSubtitleTranslator:
    """Adapter dịch phụ đề đa giai đoạn dùng Gemini.

    Args:
        api_key:           Khoá API Gemini.
        retry_count:       Số lần thử lại tối đa khi gặp lỗi tạm thời (≥ 1).
        request_timeout_s: Thời gian chờ tối đa mỗi request (giây); áp qua
            ``http_options.timeout`` để một request treo không làm đứng worker.
    """

    def __init__(
        self,
        api_key: str,
        retry_count: int = 5,
        request_timeout_s: float = 120.0,
        quota_manager: Any = None,
        analysis_media_resolution: str = "medium",
        analysis_thinking_level: str = "medium",
        parallel_batches: int = 1,
        reading_cps_budget: float = 0.0,
        emotion_cues: bool = False,
        analysis_video_fps: float = 1.0,
        analysis_checkpoint_dir: Path | None = None,
        token_ledger: TokenUsageLedger | None = None,
        native_address: bool = False,
        model_health: ModelHealth | None = None,
        clip_video_to_batch: bool = True,
    ) -> None:
        # [v3.23.610] Sổ token dùng chung: ghi mọi lời gọi máy chủ đã trả lời.
        self._token_ledger = token_ledger
        # [v3.23.631] Sức khoẻ model dùng chung (bộ ngắt mạch quá tải + model dự phòng);
        # ``None`` = tắt (chỉ thử lại trên một model như trước).
        self._model_health = model_health
        # [v3.23.612] Xưng hô thuần Việt ("anh Kiếm" thay "Kiếm ca") cho phân tích & dịch.
        self._native_address = bool(native_address)
        # [v3.23.121] Hỗ trợ NHIỀU API key: tách theo dòng/phẩy, bỏ trùng & rỗng.
        # Khi một key chạm quota ngày, tự XOAY sang key kế tiếp còn hạn mức.
        self._api_keys: list[str] = self._parse_keys(api_key)
        self._key_index: int = 0
        self._api_key = self._api_keys[0] if self._api_keys else ""
        self._retry_count = max(1, retry_count)
        self._request_timeout_s = max(5.0, request_timeout_s)
        # [v3.23.439] Dịch CPS-aware (tuỳ chọn): >0 = bật, siết max_chars theo tốc độ MẮT ĐỌC
        # (ký tự/giây) để phụ đề đọc kịp. 0 = tắt (giữ hành vi cũ dựa ngân sách TTS).
        self._reading_cps_budget = max(0.0, reading_cps_budget)
        # [v3.23.448] Sinh tag cảm xúc [cười]/[thở dài]/[hắng giọng] từ NGỮ CẢNH VIDEO (hình +
        # âm thanh) khi dịch — model Gemini "thấy"/"nghe" được. Chỉ có tác dụng khi đính video.
        self._emotion_cues = bool(emotion_cues)
        #: [v3.23.648] Khi dịch có đính video: chỉ gửi khoảng video lô cần (± lề) thay vì cả đoạn.
        self._clip_video_to_batch = bool(clip_video_to_batch)
        self._client: Any = None
        self._genai_module: Any = None
        self._types_module: Any = None
        self._import_error: str = ""
        # Bộ điều tiết quota tuỳ chọn (đặt chỗ token, tránh vượt rate-limit Gemini).
        self._quota_manager = quota_manager
        # [v3.23.129] Độ phân giải video cho PHÂN TÍCH ngữ cảnh & cues (low/medium/high).
        level = (analysis_media_resolution or "medium").lower()
        self._analysis_media_resolution = (
            level if level in ("low", "medium", "high") else "medium"
        )
        # [v3.23.141] Token video/giây tương ứng media_resolution phân tích, dùng để ước
        # lượng quota CHÍNH XÁC (low~100, medium~300, high~400 theo tài liệu Gemini). Trước
        # đây luôn dùng 100 -> ước lượng thấp gấp 3 khi phân tích ở medium -> vượt TPM 429.
        self._analysis_video_tps = {"low": 100, "medium": 300, "high": 400}.get(
            self._analysis_media_resolution, 300
        )
        # [v3.23.476] fps lấy mẫu video khi phân tích. Mặc định 1.0 = Gemini tự lấy ~1fps (hành vi
        # cũ). >1 → BỌC handle với videoMetadata.fps để Gemini lấy mẫu DÀY hơn (nhiều chi tiết
        # thời gian hơn, nhưng nhiều token hơn) — làm cho cài đặt fps của người dùng THỰC SỰ tác
        # dụng khi phân tích (trước đây fps chỉ ảnh hưởng preview).
        # [v3.23.477] Gemini GIỚI HẠN videoMetadata.fps ≤ 24 (đặt cao hơn → 400 INVALID_ARGUMENT,
        # HỎNG mọi call phân tích). Kẹp cứng ≤ 24 để an toàn kể cả khi setting cũ lưu >24.
        self._analysis_video_fps = max(0.2, min(24.0, float(analysis_video_fps)))
        # [v3.23.564] Lưu tiến độ phân tích theo đoạn để chạy tiếp khi hết hạn mức ngày.
        self._analysis_checkpoints = (
            AnalysisCheckpointStore(analysis_checkpoint_dir)
            if analysis_checkpoint_dir is not None
            else None
        )
        # [v3.23.140] Mức Thinking cho PHÂN TÍCH ngữ cảnh toàn cục (low/medium/high).
        # Mặc định "medium": phân tích chạy MỘT LẦN (không theo lô) nhưng quyết định roster/
        # tóm tắt/visual cues/giới tính người nói -> ảnh hưởng TOÀN BỘ bản dịch, nên đáng
        # đầu tư thinking hơn khâu dịch từng lô. Người dùng có thể chọn "high" cho phim khó
        # hoặc "low" để nhanh/tiết kiệm.
        think = (analysis_thinking_level or "medium").lower()
        self._analysis_thinking_level = (
            think if think in ("low", "medium", "high") else "medium"
        )
        # [v3.23.149] Số batch dịch chạy SONG SONG trong mỗi giai đoạn (1 = tuần tự).
        self._parallel_batches = max(1, min(int(parallel_batches or 1), 4))
        # [v3.23.149] Khoá trạng thái key: khi dịch song song, nhiều luồng có thể cùng
        # phát hiện hết quota và cùng muốn xoay key -> chỉ MỘT luồng được xoay tại một
        # thời điểm, các luồng khác thấy ngay key mới (chống xoay chồng chéo/lệch client).
        self._state_lock = threading.RLock()
        # Bộ nhớ đệm handle file video đã resolve theo remote_name (tránh gọi lại API).
        self._video_handle_cache: dict[str, Any] = {}
        # [v3.23.132] Callback UPLOAD LẠI video bằng key HIỆN TẠI. File trên Gemini cô
        # lập theo API key: khi xoay sang key khác, file do key cũ tải lên sẽ bị 403. Khi
        # đó adapter gọi callback này (truyền key hiện tại) để tải lại đoạn bằng key mới
        # và nhận refs mới. Worker cấp callback (nắm video provider + đường dẫn nguồn).
        self._video_reupload_cb: Callable[..., list[Any]] | None = None
        # [v3.23.532] True ngay sau khi đổi key: handle video cũ thuộc project cũ nên vô hiệu.
        self._video_handles_stale = False
        # [v3.23.605] Ref đã tải lại theo từng key — dùng chung cho MỌI giai đoạn.
        self._healed_refs: dict[HealedRefKey, Any] = {}
        self._load_sdk()

    def set_video_reupload_callback(
        self, cb: Callable[..., list[Any]] | None
    ) -> None:
        """Đăng ký callback tải lại video theo key hiện tại (chống 403 khi xoay key).

        [v3.23.550] Callback nhận ``(api_key, ranges)`` — ``ranges`` là các khoảng thời gian
        lời gọi đang cần, để chỉ tải đúng các đoạn đó. Callback kiểu cũ chỉ nhận ``api_key``
        vẫn được hỗ trợ.
        """
        self._video_reupload_cb = cb

    # ── Quản lý nhiều API key ────────────────────────────────────────────
    @staticmethod
    def _parse_keys(raw: str) -> list[str]:
        """Tách chuỗi key thành danh sách (theo dòng/phẩy), giữ thứ tự, bỏ trùng."""
        out: list[str] = []
        seen: set[str] = set()
        for part in re.split(r"[\n,]+", raw or ""):
            key = part.strip()
            if key and key not in seen:
                seen.add(key)
                out.append(key)
        return out

    @staticmethod
    def _fingerprint(key: str) -> str:
        """Vân tay ngắn, KHÔNG đảo ngược của API key (để khoá quota mà không lộ key)."""
        from subtitles_extractor.infrastructure.translation.gemini_quota_manager import (
            GeminiQuotaManager,
        )
        return GeminiQuotaManager.key_fingerprint(key)

    def _current_key_id(self) -> str:
        return self._fingerprint(self._api_key)

    def _switch_to_key(self, index: int) -> None:
        """Chuyển sang key thứ ``index`` và buộc dựng lại client với key mới."""
        self._key_index = index
        self._api_key = self._api_keys[index]
        self._client = None  # buộc _get_client() dựng lại với key mới
        self._video_handle_cache.clear()  # handle video gắn theo từng project/key
        # [v3.23.532] Mọi handle video đã tải bằng key CŨ chắc chắn không dùng được với key mới.
        # Trước đây app vẫn thử dùng chúng rồi mới nhận 403 — mà mỗi lần thử hỏng VẪN TÍNH MỘT
        # REQUEST vào hạn mức của key mới. Với 23 đoạn video và hạn mức 20 request/ngày, riêng
        # các lần thử vô ích đã đủ làm cạn quota ngay sau khi xoay key. Đánh dấu sẵn để bỏ hẳn
        # bước thử đó.
        self._video_handles_stale = True

    def _rpd_remaining_for_key(self, model_name: str, key: str) -> int:
        """[v3.23.146] Số request/ngày CÒN LẠI của một key cho model (an toàn với lỗi)."""
        if self._quota_manager is None:
            return 1_000_000  # không quản quota -> coi như vô hạn
        try:
            # [v3.23.639] Key bị server từ chối vĩnh viễn thì không còn hạn mức nào để xoay tới.
            if self._quota_manager.is_key_invalid(self._fingerprint(key)):
                return 0
            remaining = self._quota_manager.get_remaining(
                model_name, key_id=self._fingerprint(key)
            )
        except (AttributeError, TypeError):
            return 1_000_000  # quota_manager cũ -> coi như còn
        return int(remaining.get("rpd_remaining", 1))

    def _can_probe_key(self, model_name: str, key: str) -> bool:
        """[v3.23.642] Key có bộ đếm nội bộ ĐẦY nhưng SERVER chưa từ chối model này hôm nay.

        Bộ đếm nội bộ chỉ là ước lượng (lệch cả hai chiều: nhật ký thật 21/10 — server từ chối khi app
        mới đếm 18/20; bảng điều khiển Google từng ghi 0/20 khi app đếm đầy). Kết luận "hết ở mọi key"
        chỉ đúng khi CHÍNH SERVER đã từ chối từng key (hoặc key bị đánh dấu hỏng), nên key loại này vẫn
        phải được thử một lần — nếu server từ chối thì nó thành "xác nhận" và không bao giờ bị thử lại
        trong ngày.
        """
        manager = self._quota_manager
        if manager is None:
            return False
        fingerprint = self._fingerprint(key)
        try:
            return not (
                manager.is_key_invalid(fingerprint)
                or manager.is_server_daily_blocked(model_name, key_id=fingerprint)
            )
        except (AttributeError, TypeError):
            return False

    def _has_any_key_capacity(self, model_name: str) -> bool:
        """[v3.23.643] Có key nào còn hạn mức (hoặc đáng thử) cho ``model_name`` không."""
        return any(self._key_has_capacity(model_name, key) for key in self._api_keys)

    def _key_has_capacity(self, model_name: str, key: str) -> bool:
        """[v3.23.642] Còn hạn mức theo bộ đếm HOẶC đáng thử một lần (xem :meth:`_can_probe_key`)."""
        return self._rpd_remaining_for_key(model_name, key) > 0 or self._can_probe_key(
            model_name, key
        )

    def _client_for_index(self, index: int) -> Any:
        """[v3.23.639] Client của đúng key thứ ``index`` — KHÔNG phụ thuộc key "hiện tại" toàn cục.

        Khi dịch song song, luồng khác có thể đổi key toàn cục giữa lúc luồng này đặt chỗ quota và
        lúc gửi: lời gọi đi bằng key khác với key đã tính quota. Ghim client theo key đã đặt chỗ.
        """
        with self._state_lock:
            # [v3.23.641] Kiểm tra "key hiện tại" VÀ dựng client trong cùng một lần giữ khoá: trước
            # đây luồng khác đổi key toàn cục giữa hai bước → client của key sai (đo: 58/600 000).
            if index == self._key_index or not (0 <= index < len(self._api_keys)):
                return self._get_client()
            cache = self.__dict__.setdefault("_pinned_clients", {})
            client = cache.get(index)
            if client is None:
                self._ensure_available()
                timeout_ms = int(self._request_timeout_s * 1000)
                try:
                    client = self._genai_module.Client(
                        api_key=self._api_keys[index], http_options={"timeout": timeout_ms},
                    )
                except TypeError:
                    client = self._genai_module.Client(api_key=self._api_keys[index])
                cache[index] = client
            return client

    def _key_views(self, model_name: str, est_tokens: int) -> list[KeyView]:
        """[v3.23.639] Tình trạng từng key cho lời gọi sắp gửi (xem :mod:`key_selection`)."""
        manager = self._quota_manager
        views: list[KeyView] = []
        for index, key in enumerate(self._api_keys):
            key_id = self._fingerprint(key)
            try:
                invalid = manager.is_key_invalid(key_id)
                wait = manager.estimate_wait(model_name, est_tokens, key_id=key_id)
                remaining = manager.get_remaining(model_name, key_id=key_id)
            except (AttributeError, TypeError):
                return []                      # quota_manager cũ/giả: không chọn thông minh được
            views.append(KeyView(
                index=index, wait_s=wait, usable=not invalid,
                rpd_remaining=int(remaining.get("rpd_remaining", 0)),
                rpd_limit=int(remaining.get("rpd_limit", 0)),
            ))
        return views

    def _pick_key_for_text_call(self, model_name: str, est_tokens: int) -> None:
        """[v3.23.639] Trước khi đặt chỗ: đổi sang key RẢNH / ít dùng hơn nếu có (chỉ lời gọi KHÔNG video).

        Thay cho cách cũ "dính một key tới khi cạn": (1) gặp trần RPM/TPM thì chuyển sang key khác
        đang trống thay vì ngồi chờ tới 60 giây; (2) dàn đều mức dùng ngày giữa các key. Lời gọi
        đính video giữ nguyên key (handle video gắn theo key; đổi key = tải lại cả đoạn).
        """
        if self._quota_manager is None or len(self._api_keys) <= 1 or self._video_handle_cache:
            return
        views = self._key_views(model_name, est_tokens)
        if not views:
            return
        with self._state_lock:
            current = self._key_index
            chosen = choose_key(views, current)
            if chosen is None or chosen == current:
                return
            mine = next(v for v in views if v.index == current)
            target = next(v for v in views if v.index == chosen)
            logger.info(
                "Chọn API key #%d thay #%d cho '%s' (%s).", chosen + 1, current + 1, model_name,
                f"key #{current + 1} phải chờ {mine.wait_s:.0f}s, key #{chosen + 1} {'gửi được ngay' if target.wait_s <= 0.5 else f'chờ {target.wait_s:.0f}s'}"
                if mine.wait_s > 0.5 else
                f"cân bằng hạn mức ngày: còn {target.rpd_remaining}/{target.rpd_limit} so với "
                f"{mine.rpd_remaining}/{mine.rpd_limit}",
            )
            self._switch_to_key(chosen)

    _QUOTA_VIOLATION_RE = re.compile(
        r"quotaId['\"]?\s*[:=]\s*['\"]([^'\"]+)['\"](?:(?!quotaId).)*?quotaValue['\"]?\s*[:=]\s*['\"]?(\d+)",
        re.DOTALL,
    )

    def _learn_minute_limits(self, error: Exception, model_name: str) -> None:
        """[v3.23.639] Lấy RPM / TPM thật từ lỗi 429 ``…PerMinute…`` làm hạn mức cho các lần sau.

        Đo thật (gemini-3.1-flash-lite, 60 lời gọi dồn): ``quotaId: GenerateRequestsPerMinutePerProject
        PerModel-FreeTier``, ``quotaValue: '15'``. Bảng chép tay đoán theo tên model sẽ lạc hậu khi
        Google đổi chính sách; số server nêu thì không.
        """
        manager = self._quota_manager
        if manager is None:
            return
        try:
            for quota_id, value in self._QUOTA_VIOLATION_RE.findall(str(error)):
                low = quota_id.lower()
                if "perminute" not in low:
                    continue
                amount = int(value)
                if "token" in low:
                    manager.learn_minute_limit_from_server(model_name, tpm=amount)
                elif "request" in low:
                    manager.learn_minute_limit_from_server(model_name, rpm=amount)
        except (AttributeError, TypeError, ValueError):
            logger.debug("Không đọc được hạn mức phút từ lỗi 429.", exc_info=True)

    @staticmethod
    def _is_invalid_key_error(error: Exception) -> bool:
        """[v3.23.639] Server từ chối CHÍNH API key (không hợp lệ / hết hạn / bị vô hiệu hoá).

        Đo thật: key sai → ``400 INVALID_ARGUMENT`` với ``reason: API_KEY_INVALID`` ("API key not
        valid. Please pass a valid API key."). Google cũng vô hiệu hoá key bị lộ ("reported as
        leaked") và key hết hạn ("API key expired"). Khác với 403 do FILE thuộc key khác
        (xem ``_is_file_permission_error``) và 400 do schema (``_is_generic_invalid_argument``).
        """
        text = str(error).lower()
        return any(marker in text for marker in (
            "api_key_invalid", "api key not valid", "api key expired", "api_key_expired",
            "reported as leaked", "api key was reported", "consumer_suspended",
            "api key has been suspended", "api_key_service_blocked",
        ))

    def _clear_invalid_mark(self, key_id: str) -> None:
        """[v3.23.641] Key vừa gọi THÀNH CÔNG thì không còn "hỏng" (người dùng đã sửa key phía Google).

        Dấu hỏng lưu cả ngày và không có đường xoá nào: key đã sửa vẫn bị coi là hết hạn mức (chặn
        phân tích video, ⛔ mãi trong bảng key) cho tới ngày hạn mức kế tiếp.
        """
        manager = self._quota_manager
        if manager is None or not key_id:
            return
        try:
            if manager.is_key_invalid(key_id):
                manager.clear_invalid_key(key_id)
        except (AttributeError, TypeError):
            pass

    def _quarantine_key(self, key_index: int, error: Exception) -> bool:
        """[v3.23.639] Cách ly key bị server từ chối vĩnh viễn; ``True`` nếu còn key khác dùng được.

        Ghi dấu hỏng vào bộ quản lý quota (dùng chung + lưu qua phiên trong ngày) để mọi lượt chọn key
        sau đó bỏ qua. Nếu key toàn cục đang là key hỏng thì chuyển sang key tốt nhất còn lại.
        """
        key_id = self._fingerprint(self._api_keys[key_index]) if 0 <= key_index < len(
            self._api_keys) else ""
        reason = self._concise_error(error)
        # Sổ riêng của adapter: chạy đúng cả khi KHÔNG có bộ quản lý quota (không thì key hỏng
        # luôn "còn dùng được" và vòng thử lại quay mãi).
        self.__dict__.setdefault("_rejected_keys", set()).add(self._api_keys[key_index]
                                                              if 0 <= key_index < len(self._api_keys) else "")
        if self._quota_manager is not None and key_id:
            try:
                self._quota_manager.mark_key_invalid(key_id, reason)
            except (AttributeError, TypeError):
                pass
        logger.error(
            "API key #%d (…%s) bị Gemini TỪ CHỐI: %s — bỏ key này, dùng key khác%s.",
            key_index + 1, key_id[-4:] if key_id else "?", reason,
            "" if len(self._api_keys) > 1 else " (không còn key nào khác)",
        )
        with self._state_lock:
            rejected = self.__dict__.get("_rejected_keys", set())
            usable = [
                i for i, key in enumerate(self._api_keys)
                if (key not in rejected if self._quota_manager is None
                    else not self._quota_manager.is_key_invalid(self._fingerprint(key)))
            ]
            if not usable:
                return False
            if self._key_index not in usable:
                self._switch_to_key(usable[0])
            return True

    def _after_daily_exhausted(self, model_name: str, sent_index: int) -> bool:
        """[v3.23.639] Key ``sent_index`` vừa bị server báo hết hạn mức ngày: có key khác để thử không?

        Khi dịch song song, luồng khác có thể đã chuyển key toàn cục sang một key còn hạn mức —
        dùng luôn key đó thay vì kết luận "mọi key đều hết" (trước đây so với key toàn cục và trả
        ``False`` oan).
        """
        with self._state_lock:
            if self._key_index != sent_index and self._key_has_capacity(
                model_name, self._api_keys[self._key_index]
            ):
                return True
            return self._rotate_to_available_key_locked(model_name)

    def _best_available_key_index(self, model_name: str) -> int | None:
        """[v3.23.146] Chỉ số key còn NHIỀU quota ngày NHẤT (>0); None nếu mọi key đã cạn.

        Chọn key dư dả nhất giúp đi TRỌN phiên (vd phân tích 5 đoạn) mà không cạn giữa
        chừng rồi phải xoay + tải lại video -> hành xử tối ưu, dự đoán tốt.
        """
        best_idx: int | None = None
        best_remaining = 0
        for idx, key in enumerate(self._api_keys):
            remaining = self._rpd_remaining_for_key(model_name, key)
            if remaining > best_remaining:
                best_remaining = remaining
                best_idx = idx
        if best_idx is not None:
            return best_idx
        # [v3.23.642] Không key nào còn hạn mức THEO BỘ ĐẾM: thử key mà server chưa từ chối (ưu tiên
        # key có bộ đếm thấp nhất = nhiều khả năng bộ đếm lệch nhất) trước khi bỏ cuộc.
        probes = [
            (self._rpd_used_for_key(model_name, key), idx)
            for idx, key in enumerate(self._api_keys)
            if self._can_probe_key(model_name, key)
        ]
        return min(probes)[1] if probes else None

    def _rpd_used_for_key(self, model_name: str, key: str) -> int:
        """Số request/ngày đã đếm của một key (an toàn với lỗi)."""
        if self._quota_manager is None:
            return 0
        try:
            used = self._quota_manager.get_remaining(model_name, key_id=self._fingerprint(key))
            return int(used.get("rpd_used", 0))
        except (AttributeError, TypeError):
            return 0

    def has_any_daily_quota(self, model_name: str) -> bool:
        """[v3.23.146] True nếu CÓ ÍT NHẤT MỘT key còn quota ngày cho model.

        Cho tầng worker DỰ ĐOÁN trước khi làm việc tốn kém (nén+upload video ~2 phút): nếu
        MỌI key đã cạn quota ngày thì fail NHANH với thông báo rõ, không phí công rồi mới 429.
        """
        if self._quota_manager is None or not self._api_keys:
            return True
        return any(self._key_has_capacity(model_name, key) for key in self._api_keys)

    def ensure_viable_key(
        self,
        model_name: str,
        needed_requests: int = 1,
        *,
        avoid_reupload: bool = False,
    ) -> str:
        """[v3.23.145] CHỦ ĐỘNG chọn API key CÒN quota ngày cho model TRƯỚC khi upload.

        Gốc rễ lãng phí (dò mù): video được upload dưới key khởi tạo (vd #1), nhưng khi
        phân tích adapter mới phát hiện #1 hết quota rồi xoay sang #2/#3 -> file thuộc key
        cũ -> 403 -> phải tải lại (nén lại) nhiều lần. Hàm này để tầng worker gọi TRƯỚC khi
        upload: nếu key hiện tại đã hết quota ngày thì XOAY sang key còn hạn NGAY, và trả
        về api_key sẽ dùng -> worker set CÙNG key đó cho video provider -> upload + phân tích
        đồng nhất một key, không còn 403/tải lại.

        Args:
            model_name: Model sẽ dùng cho phiên (quota tính theo cặp model-key).
            needed_requests: Số request dự trù cho trọn phiên (>= 1) để xét đủ/thiếu.
            avoid_reupload: [v3.23.294] True khi 1 video ĐÃ upload dưới key hiện tại
                (phase phân tích) và phiên này sẽ tái dùng nó — khi đó BỎ xoay cơ hội
                ``much_better`` (xoay = buộc cắt+nén+upload lại, LỖ). Chỉ xoay khi key
                hiện tại cạn hoặc không đủ đi trọn phiên.

        Returns:
            api_key (chuỗi) sẽ được dùng cho phiên sau khi đã chọn. Người gọi phải lấy
            video của phiên SAU lời gọi này, bằng chính key trả về.
        """
        with self._state_lock:
            return self._ensure_viable_key_locked(
                model_name, needed_requests, avoid_reupload=avoid_reupload
            )

    def _ensure_viable_key_locked(
        self,
        model_name: str,
        needed_requests: int = 1,
        *,
        avoid_reupload: bool = False,
    ) -> str:
        """[v3.23.149] Thân ensure_viable_key, gọi DƯỚI ``_state_lock``."""
        if self._quota_manager is not None and len(self._api_keys) > 1:
            current_remaining = self._rpd_remaining_for_key(
                model_name, self._api_key
            )
            best_idx = self._best_available_key_index(model_name)
            if best_idx is not None and best_idx != self._key_index:
                best_remaining = self._rpd_remaining_for_key(
                    model_name, self._api_keys[best_idx]
                )
                # [v3.23.294] Quyết định xoay uỷ quyền cho policy thuần (testable, SRP).
                # avoid_reupload=True -> bỏ nhánh xoay cơ hội để giữ video đã upload,
                # tránh cắt+nén+upload lại (xem key_rotation_policy).
                if should_switch_for_viability(
                    current_remaining=current_remaining,
                    best_remaining=best_remaining,
                    needed_requests=needed_requests,
                    avoid_reupload=avoid_reupload,
                ):
                    logger.info(
                        "Chọn API key #%d (còn %d req/ngày) cho '%s' thay vì #%d (còn %d"
                        "%s) -> đi trọn phiên, tránh xoay+tải lại giữa chừng.",
                        best_idx + 1, best_remaining, model_name,
                        self._key_index + 1, current_remaining,
                        f", cần ~{int(needed_requests)}" if needed_requests > 1 else "",
                    )
                    self._switch_to_key(best_idx)
                    # [v3.23.614] Hàm này chạy TRƯỚC khi phiên lấy video (worker đặt cùng
                    # key cho video provider rồi mới tải/tra cache), nên mọi handle phiên
                    # sắp dùng đều thuộc key MỚI — không có handle cũ nào để vô hiệu. Giữ
                    # cờ "handle cũ" làm lời gọi đầu tiên "tải lại" vô ích: app_log thật
                    # 14/10 ghi "Vừa đổi API key → … tải lại NGAY" ở cả phân tích lẫn dịch
                    # dù file vừa tải bằng chính key đó; phim nhiều đoạn còn phải nén lại
                    # đoạn đó bằng ffmpeg chỉ để nhận về đúng file cũ.
                    self._video_handles_stale = False
        return self._api_key

    def _rotate_to_available_key(self, model_name: str) -> bool:
        """Chuyển sang API key CÒN quota ngày NHIỀU NHẤT cho ``model_name``.

        [v3.23.146] Chọn key dư dả NHẤT (không phải key đầu tiên gặp) để đi trọn phần
        việc còn lại mà không cạn tiếp -> ít xoay + ít tải lại video hơn.

        Returns:
            True nếu đã xoay sang một key khác còn quota; False nếu mọi key đều đã hết
            (hoặc key tốt nhất chính là key hiện tại).
        """
        with self._state_lock:
            return self._rotate_to_available_key_locked(model_name)

    def _rotate_to_available_key_locked(self, model_name: str) -> bool:
        """[v3.23.149] Thân _rotate_to_available_key, gọi DƯỚI ``_state_lock``."""
        if len(self._api_keys) <= 1 or self._quota_manager is None:
            return False
        best_idx = self._best_available_key_index(model_name)
        if best_idx is None or best_idx == self._key_index:
            return False
        # [v3.23.529] Ghi rõ AI ra quyết định xoay key: server thật sự từ chối, hay chỉ là bộ đếm
        # nội bộ. Nhật ký thật cho thấy hai lần xoay đầu xảy ra lúc 19:51 và 19:57 trong khi 429
        # đầu tiên từ server mãi 20:03 mới tới — tức app tự xoay khỏi những key VẪN CÒN quota,
        # dồn tải vào key cuối rồi mới thật sự chạm trần. Không ghi lý do thì không thể truy ra.
        current_key_id = self._fingerprint(self._api_keys[self._key_index])
        server_confirmed = False
        if self._quota_manager is not None:
            server_confirmed = self._quota_manager.is_server_daily_blocked(
                model_name, key_id=current_key_id
            )
        best_remaining = self._rpd_remaining_for_key(model_name, self._api_keys[best_idx])
        logger.info(
            "Xoay API key #%d → #%d (%s) vì key hiện tại hết quota '%s' [nguồn: %s].",
            self._key_index + 1, best_idx + 1,
            f"còn {best_remaining} req/ngày" if best_remaining > 0
            else "bộ đếm đã đầy nhưng SERVER chưa từ chối — thử một lần",
            model_name,
            "SERVER từ chối" if server_confirmed else "bộ đếm nội bộ (server chưa từ chối)",
        )
        self._switch_to_key(best_idx)
        return True

    def _rotate_for_overload(
        self, model_name: str, tried: set[int], failed_key_index: int
    ) -> bool:
        """[v3.23.592] Quá tải kéo dài trên một key → chuyển sang key CHƯA thử.

        Chọn key còn nhiều hạn mức ngày nhất trong các key chưa chạm trần quá tải ở lời
        gọi này. Xem ``overload_retry_policy`` về lý do và số liệu.

        Khi dịch song song, luồng khác có thể đã chuyển key trong lúc lời gọi này còn
        chờ: nếu key đang dùng không phải key vừa quá tải và lời gọi này chưa thử nó thì
        dùng luôn key đó — không chuyển thêm lần nữa (hai luồng tự chuyển sẽ đẩy nhau
        quay về chính key đang quá tải).

        Args:
            model_name: Model đang gọi (hạn mức tính theo cặp model-key).
            tried: Chỉ số các key đã chạm trần quá tải trong lời gọi này; key vừa quá
                tải được thêm vào.
            failed_key_index: Chỉ số key mà các lượt quá tải vừa rồi đã được gửi bằng.

        Returns:
            True nếu lời gọi nên gửi lại ngay bằng một key khác (đã chuyển, hoặc luồng
            khác đã chuyển sẵn); False nếu không còn key nào để thử.
        """
        with self._state_lock:
            tried.add(failed_key_index)
            if len(self._api_keys) <= 1:
                return False
            if self._key_index not in tried:
                return True
            candidates = [
                (self._rpd_remaining_for_key(model_name, key), -index)
                for index, key in enumerate(self._api_keys)
                if index not in tried
            ]
            candidates = [item for item in candidates if item[0] > 0]
            if not candidates:
                return False
            remaining, negative_index = max(candidates)
            best_idx = -negative_index
            logger.warning(
                "Gemini QUÁ TẢI kéo dài trên key #%d → thử key #%d (còn %d req/ngày cho "
                "'%s'); đã thử %d/%d key.",
                failed_key_index + 1, best_idx + 1, remaining, model_name,
                len(tried), len(self._api_keys),
            )
            self._switch_to_key(best_idx)
            return True

    @staticmethod
    def _is_rate_limit_error(error: Exception) -> bool:
        """[v3.23.147] Nhận diện 429 rate-limit nói chung (RPM/TPM/RPD)."""
        text = str(error).lower()
        return (
            _safe_int(getattr(error, "code", None), 0) == 429
            or "429" in text
            or "resource_exhausted" in text
        )

    def _cooldown_remaining_for_key(self, model_name: str, key: str) -> float:
        """[v3.23.147] Giây cooldown còn lại của một key cho model (0.0 nếu không rõ)."""
        if self._quota_manager is None:
            return 0.0
        fn = getattr(self._quota_manager, "cooldown_remaining_s", None)
        if fn is None:
            return 0.0  # quota_manager cũ -> coi như không cooldown
        try:
            return float(fn(model_name, key_id=self._fingerprint(key)))
        except (AttributeError, TypeError, ValueError):
            return 0.0

    def _rotate_for_temporary_429(self, model_name: str) -> bool:
        """[v3.23.147] Xoay key khi bị 429 TẠM THỜI (TPM/RPM) trên lời gọi KHÔNG đính video.

        Với batch text-only, xoay key là MIỄN PHÍ (không có file gắn key, không phải tải
        lại gì) -> thay vì ngồi chờ retryDelay 30-60s trên key hiện tại, chuyển NGAY sang
        key vừa còn quota ngày vừa KHÔNG trong cooldown để tiếp tục tức thời.

        Returns:
            True nếu đã xoay sang key sẵn sàng gửi ngay; False nếu không có key nào phù hợp.
        """
        with self._state_lock:
            return self._rotate_for_temporary_429_locked(model_name)

    def _rotate_for_temporary_429_locked(self, model_name: str) -> bool:
        """[v3.23.149] Thân _rotate_for_temporary_429, gọi DƯỚI ``_state_lock``."""
        if len(self._api_keys) <= 1 or self._quota_manager is None:
            return False
        best_idx: int | None = None
        best_remaining = 0
        for idx, key in enumerate(self._api_keys):
            if idx == self._key_index:
                continue
            if self._cooldown_remaining_for_key(model_name, key) > 0.0:
                continue  # key đang bị server rate-limit -> gửi ngay sẽ 429 tiếp
            remaining = self._rpd_remaining_for_key(model_name, key)
            if remaining > best_remaining:
                best_remaining = remaining
                best_idx = idx
        if best_idx is None:
            return False
        logger.info(
            "429 tạm thời trên key #%d → xoay sang key #%d (còn %d req/ngày, không "
            "cooldown) để tiếp tục NGAY, khỏi chờ retryDelay (batch không đính video).",
            self._key_index + 1, best_idx + 1, best_remaining,
        )
        self._switch_to_key(best_idx)
        return True

    # ── Khả dụng & SDK ───────────────────────────────────────────────────
    def _load_sdk(self) -> None:
        try:
            from google import genai
            from google.genai import types

            self._genai_module = genai
            self._types_module = types
        except ImportError as exc:
            self._import_error = str(exc)
            logger.warning("Không import được google-genai: %s. Trang Dịch sẽ ở chế độ không khả dụng.", exc)

    def is_available(self) -> bool:
        return self._genai_module is not None and bool(self._api_key)

    def _ensure_available(self) -> None:
        if self._genai_module is None:
            raise TranslatorUnavailableError(
                "Chưa cài thư viện 'google-genai'. Cài bằng: pip install google-genai"
            )
        if not self._api_key:
            raise TranslatorUnavailableError("Chưa nhập API Key Gemini.")

    def _get_client(self) -> Any:
        client = self._client
        if client is None:
            self._ensure_available()
            # [v3.7 fix] Đặt timeout HTTP (ms) để 1 request treo không làm đứng
            # worker vĩnh viễn. google-genai nhận http_options dạng dict {timeout: ms}.
            timeout_ms = int(self._request_timeout_s * 1000)
            try:
                client = self._genai_module.Client(
                    api_key=self._api_key,
                    http_options={"timeout": timeout_ms},
                )
            except TypeError:
                # SDK cũ không nhận http_options — fallback không timeout.
                logger.warning("SDK google-genai không hỗ trợ http_options.timeout; bỏ qua timeout.")
                client = self._genai_module.Client(api_key=self._api_key)
            self._client = client
        return client

    # ── Cấu hình sinh nội dung ───────────────────────────────────────────
    @staticmethod
    def _accepts_sampling_params(model_name: str) -> bool:
        """[v3.23.511] True nếu model còn CHẤP NHẬN ``temperature``/``top_p``/``top_k``.

        Tài liệu Gemini (7/2026) nêu rõ: từ Gemini 3.6 Flash trở đi các tham số lấy mẫu bị BỎ QUA,
        và **các thế hệ model sau sẽ trả HTTP 400** nếu vẫn gửi. Cách nhận diện cũ là "không phải
        gemini-3 thì gửi", nên một model tương lai (vd ``gemini-4-flash``) sẽ KHÔNG khớp mẫu
        ``gemini-3`` và vẫn bị gửi ``temperature`` → lỗi 400 làm hỏng toàn bộ khâu dịch.

        Nay đảo chiều: CHỈ gửi cho các họ model CŨ đã biết chắc là còn hỗ trợ. Model lạ/mới mặc
        định KHÔNG gửi — an toàn hơn, vì bỏ sót tham số đã bị bỏ qua thì vô hại, còn gửi thừa thì
        hỏng cả tác vụ.

        Args:
            model_name: Tên model, vd ``"gemini-2.5-flash"``.

        Returns:
            True nếu nên gửi tham số lấy mẫu.
        """
        name = (model_name or "").lower()
        return any(
            marker in name
            for marker in ("gemini-1.", "gemini-1_", "gemini-2.", "gemini-2_", "gemma")
        )

    # [v3.23.52] Nhận diện họ model Gemini 3.x (dùng thinking_level thay thinking_budget).
    @staticmethod
    def _is_gemini_3_family(model_name: str) -> bool:
        """True nếu model dùng ``thinking_level``: Gemini ≥ 3 (3.8-flash, 3.5-flash-lite…).

        Model 3.x dùng ``thinking_level`` (không dùng ``thinking_budget``) và BỎ QUA
        ``temperature/top_p/top_k`` (đã deprecated — xem changelog Gemini API 6/2026).
        """
        name = (model_name or "").lower()
        # [v3.23.595] Mọi thế hệ ≥ 3 (cả gemini-4 sau này) và bí danh "-latest" (đang trỏ
        # tới Gemini 3.x): trước đây chỉ khớp chuỗi "gemini-3" nên model thế hệ sau bị gửi
        # ``thinking_budget`` — tham số cũ mà Gemini 3 chỉ còn nhận để tương thích ngược,
        # và mức suy nghĩ "low" người dùng chọn bị bỏ qua.
        if name.endswith("-latest") and name.startswith("gemini-") and (
            gemini_model_version(name) is None
        ):
            return True
        return uses_thinking_level(name)

    def _resolve_media_resolution(self, level: str) -> Any:
        """Map 'low'/'medium'/'high' sang enum MediaResolution (None nếu SDK thiếu)."""
        mapping = {
            "low": "MEDIA_RESOLUTION_LOW",
            "medium": "MEDIA_RESOLUTION_MEDIUM",
            "high": "MEDIA_RESOLUTION_HIGH",
        }
        enum_name = mapping.get((level or "").lower())
        if enum_name is None:
            return None
        media_enum = getattr(self._types_module, "MediaResolution", None)
        if media_enum is None:
            return None
        return getattr(media_enum, enum_name, None)

    def _build_config(
        self,
        temperature: float,
        response_schema: dict[str, Any],
        system_instruction: str,
        enable_thinking: bool = False,
        thinking_budget: int = -1,
        model_name: str = "",
        thinking_level: str = "low",
        media_resolution: str | None = None,
    ) -> Any:
        """Tạo ``GenerateContentConfig`` cho lần gọi API.

        Cấu hình suy nghĩ (thinking) khác nhau theo họ model:

        * **Gemini 3.x** (3-flash, 3.1-flash-lite, 3.5-flash…): dùng ``thinking_level``
          (low/medium/high). KHÔNG được dùng kèm ``thinking_budget`` (lỗi 400). Với tác
          vụ DỊCH, mặc định ``low`` cho nhanh/rẻ và chất lượng tốt — thinking quá nhiều
          dễ phản tác dụng với dịch thuật.
        * **Gemini 2.5**: dùng ``thinking_budget`` (-1 dynamic, 0 tắt, >0 giới hạn token).

        ``media_resolution`` ('low'/'medium'/'high') kiểm soát số token cấp cho VIDEO/ảnh
        (tính năng Gemini 3.x). Đặt thấp khi video chỉ là ngữ cảnh phụ → tiết kiệm token,
        giảm 429; đặt vừa khi cần nhìn rõ mặt/biểu cảm (phân tích cues). SDK cũ không hỗ
        trợ thì bỏ qua êm.

        Thinking tương thích với ``response_mime_type="application/json"`` +
        ``response_schema`` — ``response.text`` trả JSON sạch, phần suy nghĩ tách riêng.
        """
        types = self._types_module
        kwargs: dict[str, Any] = {
            "response_mime_type": "application/json",
            "response_schema": response_schema,
        }
        # [v3.23.250] Nhiệt độ: tài liệu Gemini 3 khuyến nghị MẠNH để mặc định 1.0 —
        # "if your existing code explicitly sets temperature (especially to low values),
        # we recommend removing this parameter... to avoid potential looping issues or
        # performance degradation on complex tasks". Với Gemini 3.x, KHÔNG truyền
        # temperature (để model tự dùng 1.0). Với Gemini 2.5.x, giữ hành vi cũ (clamp
        # [0,1]) vì model cũ được hiệu chỉnh với temperature thấp cho dịch tất định.
        # [v3.23.511] Chỉ gửi cho họ model CŨ còn hỗ trợ; model mới/lạ thì bỏ qua để tránh
        # HTTP 400 mà tài liệu đã cảnh báo cho các thế hệ sau Gemini 3.6.
        if self._accepts_sampling_params(model_name):
            kwargs["temperature"] = max(0.0, min(1.0, temperature))
        if system_instruction:
            kwargs["system_instruction"] = system_instruction

        if media_resolution:
            resolved = self._resolve_media_resolution(media_resolution)
            if resolved is not None:
                kwargs["media_resolution"] = resolved

        # [v3.23.610] TẮT suy nghĩ cũng phải gửi cấu hình: không gửi gì thì model dùng mức
        # mặc định (gemini-3.8-flash: Medium, 3.1-flash-lite: High) — xem thinking_policy.
        thinking = resolve_thinking(
            model_name,
            enable_thinking=enable_thinking,
            thinking_budget=thinking_budget,
            thinking_level=thinking_level,
            minimal_rejected=model_name in self._minimal_level_rejected_models(),
        )
        if thinking is not None:
            thinking_config = self._thinking_config(types, thinking)
            if thinking_config is not None:
                kwargs["thinking_config"] = thinking_config

        return self._build_gcc(types, kwargs)

    def _minimal_level_rejected_models(self) -> set[str]:
        """[v3.23.610] Model đã trả lỗi khi nhận mức ``minimal`` trong phiên này."""
        rejected = getattr(self, "_minimal_rejected", None)
        if rejected is None:
            rejected = set()
            self._minimal_rejected = rejected
        return rejected

    @staticmethod
    def _thinking_config(types: Any, thinking: ThinkingSettings) -> Any | None:
        """``types.ThinkingConfig`` cho ``thinking``; ``None`` nếu SDK không hỗ trợ."""
        try:
            return types.ThinkingConfig(**thinking.as_kwargs())
        except (AttributeError, TypeError):
            pass
        try:
            # SDK cũ chưa có thinking_level: vẫn bật suy nghĩ mặc định như trước.
            return types.ThinkingConfig()
        except (AttributeError, TypeError):
            logger.debug("SDK này chưa hỗ trợ ThinkingConfig, bỏ qua thinking.")
            return None

    def _build_gcc(self, types: Any, kwargs: dict[str, Any]) -> Any:
        """[v3.23.419] Tạo ``GenerateContentConfig`` và TẮT AFC.

        App KHÔNG dùng function/tool calling cho dịch, nhưng google-genai bật Automatic Function
        Calling (AFC) mặc định → cảnh báo "Direct use of AFC ... not recommended" + overhead vô
        ích mỗi lần gọi. Tắt AFC cho sạch log và nhanh hơn chút. Guard try/except phòng SDK cũ
        chưa có ``AutomaticFunctionCallingConfig``.

        Args:
            types: Module ``google.genai.types``.
            kwargs: Tham số cấu hình đã dựng.

        Returns:
            Instance ``GenerateContentConfig``.
        """
        try:
            kwargs.setdefault(
                "automatic_function_calling",
                types.AutomaticFunctionCallingConfig(disable=True),
            )
        except (AttributeError, TypeError):
            pass
        # [v3.23.425] Nới safety_settings về BLOCK_NONE. Đây là công cụ DỊCH phụ đề phim của
        # NGƯỜI DÙNG (nội dung hợp pháp họ đang xem). Mặc định Gemini hay CHẶN NHẦM cảnh phim
        # cổ trang/hành động (hành quyết, chiến trận, xưng hô gay gắt, tình cảm) → trả None
        # (finish_reason=SAFETY) → mất câu dịch + tốn retry. Nới lọc giảm thất bại giả này.
        # Guard: SDK/loại/threshold nào không hỗ trợ thì bỏ qua, không làm hỏng cấu hình.
        kwargs.setdefault("safety_settings", self._permissive_safety_settings(types))
        return types.GenerateContentConfig(**kwargs)

    @staticmethod
    def _permissive_safety_settings(types: Any) -> list[Any]:
        """[v3.23.425] Danh sách SafetySetting = BLOCK_NONE cho 4 nhóm harm chính.

        Bỏ qua nhóm/threshold nào SDK không hỗ trợ (trả danh sách rỗng nếu SDK quá cũ) — an
        toàn tuyệt đối, không ném lỗi.

        Args:
            types: Module ``google.genai.types``.

        Returns:
            Danh sách ``SafetySetting`` (có thể rỗng nếu SDK không hỗ trợ).
        """
        settings: list[Any] = []
        category_names = (
            "HARM_CATEGORY_HARASSMENT",
            "HARM_CATEGORY_HATE_SPEECH",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "HARM_CATEGORY_DANGEROUS_CONTENT",
        )
        try:
            harm_category = types.HarmCategory
            threshold = types.HarmBlockThreshold.BLOCK_NONE
            safety_setting_cls = types.SafetySetting
        except AttributeError:
            return settings
        for name in category_names:
            category = getattr(harm_category, name, None)
            if category is None:
                continue
            try:
                settings.append(
                    safety_setting_cls(category=category, threshold=threshold)
                )
            except (TypeError, ValueError):
                continue
        return settings

    def _strip_thinking_from_config(self, config: Any) -> Any:
        """Tạo bản sao config nhưng BỎ ``thinking_config`` (dùng khi fallback).

        Trả về config mới để retry với model không hỗ trợ thinking, giữ nguyên các
        thiết lập khác (schema, system_instruction, temperature).

        [v3.23.610] Sao chép NGUYÊN config rồi chỉ bỏ phần suy nghĩ. Trước đây config được
        dựng lại từ 4 trường, nên mất ``media_resolution`` (video gửi ở độ phân giải mặc
        định — nhiều token hơn hẳn) và bộ lọc an toàn đã nới.
        """
        copied = self._copy_config(config, thinking_config=None)
        if copied is not None:
            return copied
        types = self._types_module
        kwargs: dict[str, Any] = {}
        for attr in ("temperature", "response_mime_type", "response_schema", "system_instruction"):
            value = getattr(config, attr, None)
            if value is not None:
                kwargs[attr] = value
        try:
            return self._build_gcc(types, kwargs)
        except (AttributeError, TypeError):
            return config

    def _extend_timeout_after_deadline(
        self, config: Any, timeout_s: float, elapsed_s: float
    ) -> tuple[Any, float]:
        """[v3.23.615] Nới thời hạn cho lượt thử sau khi lượt vừa rồi chạy hết thời hạn.

        Gửi lại với CÙNG thời hạn sau khi máy chủ đã không kịp trả lời là gần như chắc
        mất thêm một lần chờ trọn thời hạn (xem :func:`deadline_was_hit`). Thời hạn mới
        đi kèm chính lời gọi (``http_options`` trong config), không đổi client dùng chung.

        Args:
            config: Config đang gửi.
            timeout_s: Thời hạn đang áp cho lời gọi này (giây).
            elapsed_s: Thời gian lượt thử vừa hỏng đã chạy (giây).

        Returns:
            ``(config, thời hạn)`` cho lượt kế tiếp — giữ nguyên khi lượt vừa rồi hỏng
            sớm, khi đã ở trần, hoặc khi config không sao chép được.
        """
        if not deadline_was_hit(elapsed_s, timeout_s):
            return config, timeout_s
        longer = escalated_request_timeout(timeout_s)
        if longer <= timeout_s:
            return config, timeout_s
        http_options_cls = getattr(self._types_module, "HttpOptions", None)
        if http_options_cls is None:
            return config, timeout_s
        try:
            http_options = http_options_cls(timeout=int(longer * 1000))
        except (TypeError, ValueError):
            return config, timeout_s
        copied = self._copy_config(config, http_options=http_options)
        if copied is None:
            return config, timeout_s
        logger.info(
            "Lượt thử chạy hết %.0f giây mà máy chủ chưa trả lời (thời hạn %.0f giây) — "
            "lượt sau cho phép %.0f giây.",
            elapsed_s, timeout_s, longer,
        )
        return copied, longer

    @staticmethod
    def _copy_config(config: Any, **update: Any) -> Any | None:
        """Bản sao ``GenerateContentConfig`` đổi vài trường (``None``: không sao được)."""
        model_copy = getattr(config, "model_copy", None)
        if not callable(model_copy):
            return None
        try:
            return model_copy(update=update)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _config_thinking_level(config: Any) -> str:
        """Mức ``thinking_level`` đang gửi trong ``config`` (chữ thường, có thể rỗng)."""
        level = getattr(getattr(config, "thinking_config", None), "thinking_level", None)
        if level is None:
            return ""
        return str(getattr(level, "value", level)).lower()

    @staticmethod
    def _is_generic_invalid_argument(error: BaseException) -> bool:
        """[v3.23.620] 400 INVALID_ARGUMENT không kèm lý do cụ thể (dấu hiệu schema quá khó)."""
        message = str(error).lower()
        return (
            "400" in message
            and "invalid_argument" in message
            and "request contains an invalid argument" in message
        )

    @staticmethod
    def _is_thinking_level_rejection(error: BaseException) -> bool:
        """[v3.23.610] Lỗi do model không nhận MỨC suy nghĩ đã gửi (vd ``minimal``)."""
        message = str(error).lower()
        if "thinking" not in message or "level" not in message:
            return False
        return any(
            marker in message
            for marker in ("minimal", "not supported", "unsupported", "invalid")
        )

    def _downgrade_minimal_level(self, model_name: str, config: Any) -> Any | None:
        """[v3.23.610] Model từ chối ``minimal`` → bản sao config dùng ``low`` và nhớ lại.

        Returns:
            Config mới; ``None`` nếu config không gửi ``minimal`` hoặc không sao được.
        """
        if self._config_thinking_level(config) != MINIMAL_LEVEL:
            return None
        types = self._types_module
        try:
            lowered = types.ThinkingConfig(thinking_level=LOWEST_COMMON_LEVEL)
        except (AttributeError, TypeError):
            return None
        copied = self._copy_config(config, thinking_config=lowered)
        if copied is not None:
            self._minimal_level_rejected_models().add(model_name)
        return copied

    def _lower_thinking_level(self, model_name: str, config: Any) -> Any | None:
        """[v3.23.622] Bản sao config với mức suy nghĩ thấp hơn MỘT bậc (``None``: hết đường).

        Dùng khi phản hồi dừng vì ``MAX_TOKENS``. Chỉ áp dụng model dùng ``thinking_level``
        (Gemini 3+); ``minimal`` chỉ là đích nếu model nhận (``supports_minimal_level``).
        """
        current = self._config_thinking_level(config)
        if current not in THINKING_LEVELS:
            return None
        index = THINKING_LEVELS.index(current)
        if index == 0:
            return None
        target = THINKING_LEVELS[index - 1]
        if target == MINIMAL_LEVEL and (
            not supports_minimal_level(model_name)
            or model_name in self._minimal_level_rejected_models()
        ):
            return None
        try:
            lowered = self._types_module.ThinkingConfig(thinking_level=target)
        except (AttributeError, TypeError):
            return None
        return self._copy_config(config, thinking_config=lowered)

    @staticmethod
    def _log_call_summary(model_name: str, config: Any, response: Any) -> None:
        """[v3.23.622] Một dòng nhật ký mỗi lời gọi: model YÊU CẦU và model THẬT SỰ phục vụ.

        Bí danh đã ngừng (3.5-flash → 3.6-flash) chỉ lộ ra ở ``response.model_version``;
        mức suy nghĩ gửi đi so với token suy nghĩ thực tế cho biết mức đó có suy nghĩ không
        (đo: ``low`` trên 3.5-flash-lite không suy nghĩ chút nào).
        """
        meta = getattr(response, "usage_metadata", None)
        served = str(getattr(response, "model_version", "") or "")
        finish = diagnose_empty_response(response).finish_reason or "?"
        level = GeminiSubtitleTranslator._config_thinking_level(config) or "-"
        thoughts = _safe_int(getattr(meta, "thoughts_token_count", None), 0)
        logger.info(
            "Gemini %s%s | suy nghĩ gửi=%s thực=%d token | kết thúc=%s",
            model_name,
            f" (phục vụ bởi {served})" if served and served != model_name else "",
            level, thoughts, finish,
        )

    @staticmethod
    def _is_thinking_unsupported_error(error: BaseException) -> bool:
        """Nhận diện lỗi 'model không hỗ trợ thinking' để fallback tắt thinking.

        Chỉ những model THỰC SỰ không hỗ trợ mới khớp — các model mới (kể cả nhỏ)
        đa số đã hỗ trợ nên vẫn ưu tiên bật thinking trước.
        """
        message = str(error).lower()
        markers = (
            "thinking", "thinking_config", "thinkingbudget",
            "does not support", "not supported", "unknown field",
            "invalid_argument",
        )
        # Phải nhắc tới 'thinking' để tránh nuốt nhầm lỗi khác.
        return "thinking" in message and any(m in message for m in markers)

    def _response_schema_for(
        self, stage: TranslationStageConfig, context: TranslationContext
    ) -> dict[str, Any]:
        """Chọn JSON schema phù hợp với giai đoạn + cờ ngữ cảnh.

        [v3.7] Schema PREPROCESS không cần speaker/description.
        Với LITERAL/STYLE/LOCALIZE: bỏ description khỏi schema khi
        ``context.include_desc=False`` → tiết kiệm token + tránh AI hallucinate.
        """
        if stage.kind is TranslationStageKind.PREPROCESS:
            return _PREPROCESS_TEXT_SCHEMA
        if not context.include_desc:
            return _TRANSLATE_NO_DESC_SCHEMA
        return _TRANSLATE_SCHEMA

    # ── Retry helpers ────────────────────────────────────────────────────
    def _overload_attempt_cap(self, model_name: str) -> int | None:
        """[v3.23.585] Số lượt thử tối đa khi quá tải, theo hạn mức ngày của model.

        Args:
            model_name: Tên model sắp gọi.

        Returns:
            Số lượt thử (kể cả lượt đầu) khi hạn mức ngày của model ít; ``None`` khi dồi
            dào, khi hạn mức chỉ là phỏng đoán theo tên model, hoặc khi không có bộ quản
            lý quota (lúc đó chỉ ngân sách thời gian quyết định).
        """
        if self._quota_manager is None:
            return None
        if not self._quota_manager.is_daily_limit_confirmed(model_name):
            return None
        remaining = self._quota_manager.get_remaining(
            model_name, key_id=self._current_key_id()
        )
        return overload_attempt_cap(int(remaining.get("rpd_limit", 0)))

    def _failover_model(self, model_name: str, tried: set[str]) -> str | None:
        """[v3.23.631] Model dự phòng khi ``model_name`` đang quá tải (``None`` = không có).

        Chỉ chọn model còn hạn mức ngày ở ÍT NHẤT một key, cùng thế hệ suy nghĩ (đổi giữa
        ``thinking_level`` và ngân sách token thì phải bỏ suy nghĩ khỏi config), chưa thử
        trong lời gọi này và không đang bị ngắt mạch. Xem ``model_health``.
        """
        health = self._model_health
        if health is None:
            return None

        def _available(name: str) -> bool:
            # Chọn model DỰ PHÒNG theo bộ đếm: model còn hạn mức thật luôn hơn model chỉ "đáng thử"
            # (key đầy-bộ-đếm của model chính đã được thử ở tầng xoay key, xem ``_can_probe_key``).
            return any(self._rpd_remaining_for_key(name, key) > 0 for key in self._api_keys)

        def _daily_limit(name: str) -> int:
            if self._quota_manager is None:
                return 0
            try:
                return int(self._quota_manager.get_remaining(name).get("rpd_limit", 0))
            except (AttributeError, TypeError):
                return 0

        ranked = health.fallbacks(
            model_name, daily_limit=_daily_limit, available=_available, exclude=tried,
        )
        if not ranked:
            return None
        # [v3.23.638] Model dự phòng người dùng chọn thì dùng đúng nó (kể cả khác thế hệ suy nghĩ).
        if health.preferred_fallback and ranked[0] == health.preferred_fallback:
            return ranked[0]
        same_family = [
            name for name in ranked
            if uses_thinking_level(name) == uses_thinking_level(model_name)
        ]
        return (same_family or ranked)[0]

    @staticmethod
    def _is_overload_error(error: Exception) -> bool:
        """[v3.23.554] Lỗi QUÁ TẢI tạm thời phía server (503/504), không phải quota/lỗi ta.

        Args:
            error: Ngoại lệ nhận được từ SDK Gemini.

        Returns:
            ``True`` nếu là quá tải — loại được chờ thêm theo
            :data:`OVERLOAD_RETRY_BUDGET_S` thay vì bỏ cả lượt dịch.
        """
        if isinstance(error, ModelOverloadedError):
            return True
        if _safe_int(getattr(error, "code", None), 0) in _OVERLOAD_HTTP_CODES:
            return True
        if _safe_int(getattr(error, "status_code", None), 0) in _OVERLOAD_HTTP_CODES:
            return True
        text = str(error).lower()
        if "resource_exhausted" in text or "quota" in text:
            return False
        return any(marker in text for marker in _OVERLOAD_MARKERS)

    @staticmethod
    def _is_retryable(error: Exception) -> bool:
        # [v3.23.142] JSON hỏng từ model thường NHẤT THỜI (một lần gen lỗi cú pháp). Cho
        # retry để gen lại thay vì bỏ cả đoạn/lô. Lớp cứu cắt cụt đã chạy trước đó.
        if isinstance(error, (json.JSONDecodeError, ModelOverloadedError)):
            return True
        if _safe_int(getattr(error, "code", None), 0) in _RETRYABLE_HTTP_CODES:
            return True
        if _safe_int(getattr(error, "status_code", None), 0) in _RETRYABLE_HTTP_CODES:
            return True
        error_text = str(error).lower()
        return any(marker in error_text for marker in _RETRYABLE_MARKERS)

    def _log_daily_quota_context(
        self, error: Exception, model_name: str, key_id: str = ""
    ) -> None:
        """[v3.23.528] Ghi ngữ cảnh khi server báo hết hạn mức NGÀY, để đối chiếu với bảng Google.

        In ra: đuôi API key đang dùng, số request ỨNG DỤNG đã đếm cho key đó, và hạn mức server
        nêu trong lỗi. Khi số đếm của ứng dụng thấp hơn hẳn hạn mức server, gần như chắc chắn các
        API key đang dùng CHUNG một dự án — vì hạn mức tính theo dự án chứ không theo key.

        Args:
            error: Ngoại lệ 429 từ server.
            model_name: Tên model đang gọi.
        """
        try:
            text = str(error)
            quota_value = 0
            match = re.search(r"quotaValue['\"]?\s*[:=]\s*['\"]?(\d+)", text)
            if match:
                quota_value = int(match.group(1))
            key_id = key_id or self._current_key_id() or ""
            key_tail = key_id[-4:] if len(key_id) >= 4 else key_id
            counted = -1
            if self._quota_manager is not None:
                remaining = self._quota_manager.get_remaining(model_name, key_id=key_id)
                counted = int(remaining.get("rpd_used", -1))
            # [v3.23.530] Lấy luôn hạn mức server nêu làm hạn mức thật cho các lần sau — bảng
            # chép tay trong ứng dụng đã lạc hậu (cấu hình 70 trong khi server nêu 20).
            # [v3.23.531] CHỈ học hạn mức khi đúng là quota SỐ REQUEST/ngày. Quota TOKEN/ngày có
            # giá trị hàng triệu — học nhầm sẽ đặt hạn mức request thành 1.000.000 và vô hiệu hoá
            # hoàn toàn lớp bảo vệ. (Lỗi này do chính v3.23.530 tạo ra, chưa kịp phát tác.)
            low_text = text.lower()
            is_request_quota = "request" in low_text and "token" not in low_text
            if quota_value > 0 and is_request_quota and self._quota_manager is not None:
                self._quota_manager.learn_daily_limit_from_server(model_name, quota_value)
            logger.warning(
                "Server báo HẾT HẠN MỨC NGÀY cho '%s' (key …%s). Hạn mức server nêu: %s. "
                "Ứng dụng mới đếm %s request cho key này. Nếu con số của ứng dụng thấp hơn hẳn, "
                "nhiều khả năng các API key của bạn thuộc CÙNG MỘT DỰ ÁN — hạn mức ngày tính "
                "theo DỰ ÁN, không theo key, nên các key dùng chung một hũ quota.",
                model_name, key_tail or "?", quota_value or "không rõ", counted,
            )
        except Exception:  # noqa: BLE001 — ghi chẩn đoán không được làm hỏng luồng lỗi
            logger.debug("Không ghi được ngữ cảnh hạn mức ngày.", exc_info=True)

    @staticmethod
    def _is_daily_quota_error(error: Exception) -> bool:
        """Nhận diện 429 do HẾT QUOTA NGÀY (RPD) — để xoay key thay vì chờ hồi.

        [v3.23.586] Mã quota ``…PerDay…`` quyết định trước; mã ``…PerMinute…`` (hạn mức
        request/phút dùng CHUNG tên metric ``free_tier_requests`` với hạn mức ngày) thì
        không phải hết quota ngày. Từ bản này lỗi hết quota ngày mà không còn key là
        dừng hẳn, nên nhận nhầm hạn mức phút sẽ làm hỏng cả lượt chạy.

        Args:
            error: Ngoại lệ nhận được từ SDK Gemini.

        Returns:
            ``True`` nếu server từ chối vì hạn mức theo NGÀY.
        """
        text = str(error).lower()
        if "perday" in text or "requests_per_day" in text:
            return True
        # 429 kèm dấu hiệu giới hạn 'per day' / 'mỗi ngày'.
        is_429 = _safe_int(getattr(error, "code", None), 0) == 429 or "429" in text
        if is_429 and ("per day" in text or "/ngày" in text):
            return True
        if "perminute" in text or "per minute" in text:
            return False
        return "free_tier_requests" in text and "exceeded" in text

    @staticmethod
    def _is_file_permission_error(error: Exception) -> bool:
        """[v3.23.132] Nhận diện 403 do FILE thuộc API key khác (sau khi xoay key).

        Gemini cô lập file theo key: dùng key B truy cập file của key A → 403
        PERMISSION_DENIED kèm 'do not have permission to access the File'. Cần tải lại
        đoạn bằng key hiện tại thay vì coi là lỗi vĩnh viễn.
        """
        text = str(error).lower()
        code = _safe_int(getattr(error, "code", None), 0)
        is_403 = code == 403 or "403" in text or "permission_denied" in text
        return is_403 and ("file" in text and "permission" in text)

    @staticmethod
    def _concise_error(error: Exception) -> str:
        """[v3.23.478] Rút gọn lỗi Gemini để LOG gọn: '429 RESOURCE_EXHAUSTED (quota token/phút)'
        thay vì in nguyên khối JSON ~1000 ký tự làm rối nhật ký. Chi tiết đầy đủ vẫn nằm trong
        exception gốc khi cần."""
        text = str(error)
        low = text.lower()
        # [v3.23.480] Lỗi mạng/DNS → thông báo thân thiện (giúp người dùng biết là MẤT MẠNG, không
        # phải lỗi app).
        if any(k in low for k in ("getaddrinfo", "name resolution", "11001", "11002",
                                  "name or service not known", "network is unreachable",
                                  "failed to resolve", "no route to host")):
            return "MẤT MẠNG / DNS không phân giải được (kiểm tra kết nối internet)"
        if "timed out" in low or "timeout" in low:
            # [v3.23.615] 34/34 lần "timed out" trong nhật ký thật xảy ra đúng ở mốc thời
            # hạn mỗi lời gọi (300 giây): máy chủ chưa trả lời kịp, không phải mạng chập
            # chờn.
            return "Hết thời hạn chờ trả lời (máy chủ chậm hoặc mạng chập chờn)"
        match = re.match(r"\s*(\d{3})\s+([A-Z_]+)", text)
        if match is None:
            return text if len(text) <= 80 else text[:77] + "…"
        head = f"{match.group(1)} {match.group(2)}"
        if "resource_exhausted" in low or match.group(1) == "429":
            if "input_token" in low or "perminute" in low or "per minute" in low:
                return f"{head} (quota token/phút free-tier)"
            if "perday" in low or "per day" in low:
                # [v3.23.531] Phân biệt hạn mức theo SỐ REQUEST/ngày với hạn mức theo SỐ
                # TOKEN/ngày. Trước đây gộp chung thành "quota token/NGÀY" nên nhật ký nói sai
                # loại: bảng điều khiển hiện RPD mới 0/20 mà ứng dụng lại báo hết request/ngày.
                # Giữ luôn mã quota gốc để không phải đoán khi chẩn đoán về sau.
                quota_id = ""
                id_match = re.search(r"quotaid['\"]?\s*[:=]\s*['\"]([^'\"]+)", text, re.I)
                if id_match:
                    quota_id = id_match.group(1)
                if "token" in low and "request" not in low:
                    kind = "quota TOKEN/ngày"
                elif "request" in low:
                    kind = "quota SỐ REQUEST/ngày"
                else:
                    kind = "quota theo NGÀY"
                suffix = f" [{quota_id}]" if quota_id else ""
                return f"{head} ({kind} free-tier){suffix}"
            return f"{head} (quota)"
        return head

    @staticmethod
    def _retry_delay(attempt: int, base_seconds: float = 5.0, max_seconds: float = 90.0) -> float:
        return min(max_seconds, (2**attempt) * base_seconds + random.uniform(1.0, 3.0))

    @staticmethod
    def _server_retry_delay(error: Exception) -> float | None:
        """[v3.23.9] Trích ``retryDelay`` server gửi kèm 429 (vd "36s") để chờ ĐÚNG.

        Gemini 429 RESOURCE_EXHAUSTED kèm RetryInfo.retryDelay cho biết chính xác bao
        lâu nữa quota mới hồi. Chờ đúng bằng đó (cộng buffer nhỏ) là tối ưu: không chờ
        thiếu (vi phạm tiếp, mất lượt) cũng không chờ thừa (lãng phí). Trả None nếu
        không tìm thấy → caller dùng backoff thường.
        """
        import re

        text = str(error)
        # Mẫu: 'retryDelay': '36s'  hoặc  "Please retry in 36.34s"
        match = re.search(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s", text)
        if not match:
            match = re.search(r"retry in (\d+(?:\.\d+)?)s", text)
        if match:
            try:
                # +1.5s buffer để chắc chắn cửa sổ quota đã reset.
                return float(match.group(1)) + 1.5
            except (ValueError, TypeError):
                return None
        return None

    @staticmethod
    def _interruptible_sleep(seconds: float, cancel_cb: CancellationCallback | None) -> bool:
        """Ngủ ``seconds`` nhưng kiểm tra huỷ mỗi 0.25s.

        [v3.7 fix] Trước đây ``time.sleep(delay)`` (tới 90s) không kiểm tra huỷ →
        người dùng bấm Huỷ phải chờ hết backoff. Nay chia nhỏ để phản hồi nhanh.

        Returns:
            True nếu bị huỷ giữa chừng, False nếu ngủ hết thời gian bình thường.
        """
        deadline = time.monotonic() + max(0.0, seconds)
        while time.monotonic() < deadline:
            if cancel_cb is not None and cancel_cb():
                return True
            time.sleep(min(0.25, max(0.0, deadline - time.monotonic())))
        return False

    @staticmethod
    def _config_system_instruction_len(config: Any) -> int:
        """Độ dài system_instruction trong config (0 nếu không có) — chỉ để log/kiểm."""
        si = getattr(config, "system_instruction", None)
        return len(si) if isinstance(si, str) else 0

    def _apply_healed_refs(self, refs: list[Any]) -> bool:
        """[v3.23.605] Thay tại chỗ ref đã tải lại bằng key hiện tại.

        Args:
            refs: Danh sách ref (bị sửa tại chỗ).

        Returns:
            True nếu MỌI ref trong danh sách là ref đã tải lại bằng key hiện tại (dùng
            được ngay, khỏi dò 403 lẫn khỏi tải lại).
        """
        # getattr: đối tượng dựng bằng ``__new__`` (kiểm thử bộ quét cues) chưa có bảng.
        healed = getattr(self, "_healed_refs", None)
        if not refs or not healed:
            return False
        key_id = self._current_key_id()
        with self._state_lock:
            replaced = substitute_healed_refs(refs, healed, key_id)
            all_current = all(
                healed.get(healed_ref_key(key_id, ref)) is ref for ref in refs
            )
        if replaced:
            logger.debug(
                "Dùng lại %d ref video đã tải lại bằng key hiện tại (khỏi dò 403).",
                replaced,
            )
        return all_current

    def _remember_healed_refs(self, fresh_refs: list[Any]) -> None:
        """[v3.23.605] Ghi nhớ ref vừa tải lại bằng key hiện tại cho các lời gọi sau."""
        key_id = self._current_key_id()
        with self._state_lock:
            healed = self.__dict__.setdefault("_healed_refs", {})
            for ref in fresh_refs:
                healed[healed_ref_key(key_id, ref)] = ref

    def _resolve_handles_with_heal(
        self,
        video_refs: list[Any] | None,
        fallback_handles: list[Any],
        *,
        probe: bool = True,
    ) -> list[Any]:
        """Resolve handle video bằng key HIỆN TẠI; nếu file thuộc key cũ (403) thì tải lại.

        Buộc resolve lại (xoá cache handle vì handle gắn theo key). Nếu thiếu handle so
        với số refs (file của key cũ không truy cập được) và có callback tải lại → tải
        lại đoạn bằng key hiện tại rồi resolve lại. Cập nhật ``video_refs`` TẠI CHỖ để
        các lần gọi sau dùng refs mới.

        Args:
            video_refs: Ref cần resolve (bị sửa tại chỗ thành ref mới nếu tải lại).
            fallback_handles: Handle trả về khi không có ref nào.
            probe: [v3.23.605] False khi người gọi VỪA dò ``files.get`` và thấy thiếu —
                bỏ lượt dò thứ hai (nhật ký thật: mỗi ref cũ bị dò 403 hai lần mỗi lô).

        Returns:
            Handle resolve được (có thể thiếu nếu tải lại cũng thất bại).
        """
        if not video_refs:
            return list(fallback_handles or [])
        self._video_handle_cache.clear()
        all_current = self._apply_healed_refs(video_refs)
        stale = getattr(self, "_video_handles_stale", False)
        if stale and not all_current:
            logger.info(
                "Vừa đổi API key → handle video cũ chắc chắn vô hiệu, tải lại NGAY "
                "(bỏ qua lần thử hỏng để khỏi tốn request)."
            )
            handles = []
        elif not probe and not all_current:
            handles = []
        else:
            handles = self._resolve_video_handles(video_refs)
            if len(handles) >= len(video_refs):
                return handles
        if self._video_reupload_cb is None:
            return handles
        logger.info(
            "Handle video thuộc key cũ (không truy cập được) → tải lại bằng key hiện tại."
        )
        # [v3.23.550] Chỉ xin tải lại ĐÚNG các đoạn lời gọi này cần. Trước đây callback tải lại
        # cả phim: app_log thật mất 5 phút 45 giây tải 22 đoạn chỉ để gửi một yêu cầu cần một đoạn.
        needed_ranges = [
            (float(getattr(ref, "start_sec", 0.0)), float(getattr(ref, "end_sec", 0.0)))
            for ref in video_refs
        ]
        try:
            if _accepts_ranges_argument(self._video_reupload_cb):
                fresh_all = self._video_reupload_cb(self._api_key, needed_ranges)
            else:
                # Callback kiểu cũ (chỉ nhận key) — vẫn chạy được, chỉ là tải lại cả phim.
                fresh_all = self._video_reupload_cb(self._api_key)
        except Exception as exc:  # noqa: BLE001 — tải lại hỏng không được làm sập lời gọi
            logger.warning("Tải lại video bằng key mới thất bại: %s", exc)
            return handles
        if not fresh_all:
            return handles
        self._video_handles_stale = False
        self._remember_healed_refs(list(fresh_all))
        # Khớp từng ref cũ với ref mới theo khoảng thời gian (start,end) — đúng cho cả
        # lời gọi 1 đoạn (map-reduce) lẫn nhiều đoạn.
        matched: list[Any] = []
        for ref in video_refs:
            start = getattr(ref, "start_sec", None)
            end = getattr(ref, "end_sec", None)
            pick = next(
                (
                    f for f in fresh_all
                    if getattr(f, "start_sec", None) == start
                    and getattr(f, "end_sec", None) == end
                ),
                None,
            )
            matched.append(pick if pick is not None else ref)
        video_refs[:] = matched
        self._video_handle_cache.clear()
        return self._resolve_video_handles(matched)

    def _call_gemini(
        self,
        model_name: str,
        prompt: str,
        config: Any,
        validator: Any,
        cancel_cb: CancellationCallback | None = None,
        video_files: list[Any] | None = None,
        est_tokens: int = 0,
        video_refs: list[Any] | None = None,
        reject_truncated: bool = False,
    ) -> dict[str, Any]:
        """Gọi Gemini với retry cho lỗi tạm thời và lỗi sai cấu trúc batch.

        ``reject_truncated``: [v3.23.622] khâu PHÂN TÍCH (không có validator đếm dòng) đặt
        True để phản hồi ``MAX_TOKENS`` không bị vá thành kết quả đầy đủ mà bị hạ mức suy
        nghĩ rồi gọi lại.

        Nếu ``video_files`` được cung cấp, chúng được đính kèm TRƯỚC prompt để mô
        hình "xem" video làm ngữ cảnh. Nếu có ``self._quota_manager``, mỗi lần gọi sẽ
        đặt chỗ token trước (tránh vượt rate-limit) và đối soát token thực sau đó.
        """
        # [v3.7] Bắt sớm model rỗng để tránh retry loop vô nghĩa.
        if not (model_name or "").strip():
            raise SubtitleTranslationError("Tên model Gemini không được để trống.")
        # [v3.23.641] Không có key nào: báo "chưa nhập key" như trước v3.23.639 (đặt chỗ quota theo
        # chỉ số key sẽ ném IndexError khó hiểu).
        if not self._api_keys:
            raise TranslatorUnavailableError("Chưa nhập API Key Gemini.")

        # ── PHÂN LUỒNG 2 KÊNH theo đúng thiết kế Gemini API ──────────────────
        # KÊNH 1 (config.system_instruction): vai trò + quy tắc bền vững — đã nằm trong
        #   `config` (dựng ở `_build_config`), KHÔNG ghép vào prompt.
        # KÊNH 2 (contents): DỮ LIỆU của lượt này — video (nếu có) + prompt (các khối XML
        #   dữ liệu). Hai kênh tách bạch, không trộn lẫn.
        contents = [*self._apply_video_fps(video_files), prompt]
        current_handles = list(video_files or [])
        video_needs_refresh = False
        if logger.isEnabledFor(logging.DEBUG):
            sys_len = self._config_system_instruction_len(config)
            logger.debug(
                "Gemini call '%s': KÊNH system_instruction=%d ký tự | KÊNH contents: "
                "%d phần video + prompt %d ký tự (KHÔNG trộn lẫn).",
                model_name, sys_len, len(video_files or []), len(prompt),
            )
        # Ước lượng token để đặt chỗ quota (thô: ký tự/4 + token video nếu chưa truyền).
        if est_tokens <= 0:
            est_tokens = max(1, len(prompt) // 4)

        last_error: Exception | None = None
        thinking_stripped = False
        video_dropped_for_empty = False
        # [v3.23.591] Số phản hồi rỗng KHÔNG rõ lý do của lời gọi này.
        unexplained_empties = 0
        # [v3.23.554] Ngân sách chờ RIÊNG cho lỗi quá tải server — xem OVERLOAD_RETRY_BUDGET_S.
        overload_waited = 0.0
        # [v3.23.616] Mốc bắt đầu lời gọi — để báo người dùng đã chờ bao lâu khi bỏ cuộc.
        call_started = time.monotonic()
        # [v3.23.585] Model hạn mức ngày ÍT: giới hạn SỐ LƯỢT thử khi quá tải, vì mỗi lượt
        # 503 vẫn bị server tính vào hạn mức ngày (xem overload_retry_policy).
        overload_failures = 0
        overload_cap = self._overload_attempt_cap(model_name)
        # [v3.23.592] Quá tải kéo dài thì thử key khác: số lượt quá tải trên key ĐANG
        # DÙNG và các key đã chạm trần trong lời gọi này (xem overload_retry_policy).
        key_overload_failures = 0
        overload_keys_tried: set[int] = set()
        # Key mà số lượt quá tải ở trên đang đếm cho, và key của lượt gửi hiện tại.
        overload_key_index = self._key_index
        sent_key_index = self._key_index
        # Số lượt thử THƯỜNG tính lại từ 0 sau khi XOAY KEY vì hết hạn mức ngày: key mới
        # phải được thử đủ lượt, không thừa hưởng số lượt key cũ đã tiêu.
        attempt_offset = 0
        # [v3.23.615] Thời hạn MỖI lượt gửi của lời gọi này — nới sau lần chạm hạn chót.
        request_timeout_s = self._request_timeout_s
        # [v3.23.639] Số lần đổi key vì key bị từ chối trong lời gọi này (chặn vòng lặp vô tận).
        invalid_key_rotations = 0
        # [v3.23.631] Model đã thử trong lời gọi này (không quay lại khi chuyển dự phòng).
        failover_tried: set[str] = {model_name}

        primary_model = model_name
        primary_config = config

        def _switch_model(candidate: str, raw: int) -> None:
            """Đổi model của lời gọi và đặt lại mọi bộ đếm thử lại (dùng chung cho dự phòng / quay về)."""
            nonlocal model_name, config, overload_failures, key_overload_failures
            nonlocal overload_keys_tried, overload_waited, overload_cap, attempt_offset
            nonlocal overload_key_index, primary_config
            if model_name == primary_model:
                primary_config = config            # nhớ cấu hình suy nghĩ của model chính
            if candidate == primary_model:
                # [v3.23.643] Quay về model chính: KHÔI PHỤC cấu hình gốc (bản đã bị gỡ ``thinking_config``
                # khi sang dự phòng khác thế hệ làm model chính chạy với mức suy nghĩ mặc định).
                config = primary_config
            elif uses_thinking_level(candidate) != uses_thinking_level(model_name):
                config = self._strip_thinking_from_config(config)
            failover_tried.add(candidate)
            model_name = candidate
            overload_failures = key_overload_failures = 0
            overload_waited = 0.0
            overload_keys_tried = set()
            overload_key_index = self._key_index
            overload_cap = self._overload_attempt_cap(model_name)
            attempt_offset = raw + 1

        def _fail_over(reason: str, raw: int) -> bool:
            """Chuyển SANG model dự phòng cho phần còn lại của lời gọi (``True`` nếu chuyển)."""
            candidate = self._failover_model(model_name, failover_tried)
            if candidate is None:
                return False
            logger.warning(
                "Model '%s' %s → chuyển sang model dự phòng '%s' cho lời gọi này "
                "(không chờ thêm trên model đang bận).", model_name, reason, candidate,
            )
            _switch_model(candidate, raw)
            return True

        def _leave_exhausted_fallback(raw: int) -> bool:
            """[v3.23.643] Model DỰ PHÒNG vừa bị server báo hết hạn mức ngày: đừng bỏ cả lời gọi.

            Quá tải của model chính chỉ là tạm thời, còn hết hạn mức của model dự phòng thì không liên
            quan tới nó — thử dự phòng kế tiếp, không còn thì QUAY VỀ model chính (chấp nhận chờ nó
            hết quá tải). ``failover_tried`` chỉ lớn dần nên không thể quay vòng vô hạn.
            """
            if model_name == primary_model:
                return False
            if _fail_over("hết hạn mức ngày", raw):
                return True
            if self._has_any_key_capacity(primary_model):
                logger.warning(
                    "Model dự phòng '%s' đã hết hạn mức ngày và không còn model dự phòng nào khác → "
                    "quay lại '%s' (còn hạn mức) và chờ nó hết quá tải.", model_name, primary_model,
                )
                _switch_model(primary_model, raw)
                return True
            return False

        def _refresh_video_after_key_switch() -> None:
            """[v3.23.132] Sau khi XOAY KEY, handle video (do key cũ tải lên) sẽ bị 403 →
            resolve lại bằng key hiện tại, tải lại đoạn nếu cần, rồi dựng lại contents."""
            nonlocal current_handles, contents, video_needs_refresh
            if video_needs_refresh and video_refs:
                clips = [
                    (h.start_s, h.end_s) if isinstance(h, ClippedVideoHandle) else None
                    for h in current_handles
                ]
                current_handles = self._resolve_handles_with_heal(
                    video_refs, current_handles
                )
                # [v3.23.648] Handle mới (đã tải lại) không còn khoảng cắt: gắn lại theo vị trí khi số đoạn khớp.
                if len(clips) == len(current_handles):
                    current_handles = [
                        ClippedVideoHandle(h, *clip) if clip else h
                        for h, clip in zip(current_handles, clips, strict=True)
                    ]
                contents = [*self._apply_video_fps(current_handles), prompt]
                video_needs_refresh = False

        for raw_attempt in count():
            attempt = raw_attempt - attempt_offset
            if cancel_cb is not None and cancel_cb():
                raise TranslationCancelledError("Người dùng đã huỷ tiến trình dịch.")
            if self._key_index != sent_key_index:
                # [v3.23.592] Luồng dịch song song khác đã chuyển key từ lượt gửi trước:
                # handle video của key cũ sẽ bị 403 (mà 403 vẫn bị tính một request).
                video_needs_refresh = True
            _refresh_video_after_key_switch()
            # [v3.23.631] Model vừa bị ngắt mạch (các lời gọi trước liên tiếp gặp 503): sang
            # dự phòng NGAY, khỏi tự đốt thêm hai lượt 503 để biết điều đã biết.
            if (
                self._model_health is not None
                and self._model_health.cooling_remaining(model_name) > 0
            ):
                _fail_over("đang quá tải (đã ngắt mạch)", raw_attempt)
            reservation = None
            # [v3.23.639] Lời gọi KHÔNG video: chọn key rảnh / ít dùng nhất TRƯỚC khi đặt chỗ.
            if not (video_refs or current_handles):
                self._pick_key_for_text_call(model_name, est_tokens)
            # [v3.23.639] Key dùng để ĐẶT CHỖ quota cũng chính là key GỬI đi (luồng khác đổi key
            # toàn cục giữa chừng không làm lệch sổ sách).
            with self._state_lock:
                reserve_index = self._key_index
            if self._quota_manager is not None:
                while True:
                    try:
                        reservation = self._quota_manager.acquire(
                            model_name, est_tokens, cancel_cb=cancel_cb,
                            key_id=self._fingerprint(self._api_keys[reserve_index]),
                        )
                        break
                    except InterruptedError as exc:
                        raise TranslationCancelledError(
                            "Người dùng đã huỷ khi đang chờ quota."
                        ) from exc
                    except QuotaExhaustedError:
                        # Key hiện tại hết quota ngày → thử xoay sang key khác còn hạn.
                        if self._after_daily_exhausted(model_name, reserve_index):
                            video_needs_refresh = True
                            with self._state_lock:
                                reserve_index = self._key_index
                            continue
                        raise  # mọi key đều hết → báo lên trên
            # [v3.23.585] Key vừa bị xoay NGAY trong bước đặt chỗ ở trên: bước tải lại ở
            # đầu vòng lặp đã qua, nên lời gọi này sẽ mang handle video của key cũ → 403:
            # mất một request của key mới (403 vẫn bị tính) và một lượt thử kèm thời gian
            # chờ; nếu đó là lượt thử cuối thì hỏng cả lời gọi. Tải lại ngay tại đây.
            _refresh_video_after_key_switch()
            sent_key_index = reserve_index
            sent_key_id = self._fingerprint(self._api_keys[sent_key_index])
            # [v3.23.435] Khởi tạo TRƯỚC try để 'consumed' LUÔN bound khi bất kỳ except nào
            # chạy (tránh NameError che lấp lỗi thật — pyright reportPossiblyUnbound).
            consumed = False
            attempt_started = time.monotonic()
            try:
                client = self._client_for_index(sent_key_index)
                response = client.models.generate_content(
                    model=model_name, contents=contents, config=config
                )
                # [v3.23.123] Generate ĐÃ trả về → server ĐÃ tính quota (kể cả khi text bị
                # safety filter). Đánh dấu để KHÔNG hoàn (release) bộ đếm ngày ở nhánh lỗi.
                consumed = True
                if self._model_health is not None:
                    self._model_health.record_success(model_name)
                self._clear_invalid_mark(sent_key_id)
                self._record_token_usage(model_name, response)
                # Đối soát token thực tế (nếu có) để cửa sổ quota chính xác.
                if reservation is not None and self._quota_manager is not None:
                    actual = self._extract_token_count(response, est_tokens)
                    self._quota_manager.reconcile(reservation, actual)
                self._log_cache_savings(response)
                self._log_call_summary(model_name, config, response)
                # [v3.7] response.text có thể None khi bị safety filter hoặc finish_reason=OTHER.
                raw_text = getattr(response, "text", None)
                if raw_text is None or (
                    reject_truncated and diagnose_empty_response(response).truncated
                ):
                    # [v3.23.591] Ghi LÝ DO máy chủ nêu thay cho ``candidates=None`` vô
                    # nghĩa: bị chặn nội dung và trục trặc nhất thời cần xử lý khác nhau.
                    diagnosis = diagnose_empty_response(response)
                    if diagnosis.truncated:
                        raise _TruncatedOutputError(
                            "Gemini dừng vì hết ngân sách token đầu ra (MAX_TOKENS): token "
                            "suy nghĩ đã chiếm hết."
                        )
                    raise _EmptyResponseError(
                        f"Gemini không trả về text ({diagnosis.describe()}).", diagnosis
                    )
                # [v3.23.142] Parse JSON: nếu hỏng, ƯU TIÊN RETRY gen lại (lỗi JSON của
                # model thường nhất thời). Chỉ ở LẦN THỬ CUỐI mới CỨU một phần từ JSON cắt
                # cụt (hết token output) để không mất trắng cả đoạn/lô.
                # [v3.23.143] Parse JSON: nếu hỏng -> SỬA (json_repair mạnh nhất, fallback
                # bộ vá nội bộ). CHỈ khi sửa bó tay (rỗng) mới ném ra để RETRY gen lại. Đây
                # là ưu tiên chất lượng+hiệu năng: đa số JSON hỏng được sửa NGAY, khỏi tốn
                # thêm lượt gọi (và quota) cho retry.
                sanitized = _sanitize_json_text(raw_text)
                try:
                    data = json.loads(sanitized)
                except json.JSONDecodeError as json_err:
                    repaired = _repair_json_text(sanitized)
                    if repaired is None:
                        raise  # bó tay -> _is_retryable cho retry gen lại
                    data, repair_method = repaired
                    logger.warning(
                        "JSON model hỏng -> đã sửa bằng %s (%s).",
                        repair_method, json_err,
                    )
                try:
                    validator(data)
                except _BatchPartialError as partial:
                    # [v3.23.11] Thiếu ít dòng → KHÔNG retry cả batch (model thường lại
                    # bỏ quên đúng các dòng đó). Trả data một phần kèm cờ để tầng trên
                    # vá riêng phần thiếu. Đối soát token đã dùng bình thường.
                    if reservation is not None and self._quota_manager is not None:
                        actual = self._extract_token_count(response, est_tokens)
                        self._quota_manager.reconcile(reservation, actual)
                    data["_missing_line_nos"] = partial.missing_line_nos
                    return data
                return data
            except _BatchValidationError as exc:
                # [v3.23.519] Ghi diễn biến để đưa vào file chuẩn đoán — trước đây thông tin lô
                # lỗi và số lần thử lại CHỈ nằm trong nhật ký ứng dụng và mất khi đóng chương trình.
                _record_process_event(
                    "batch_retry", attempt=attempt + 1, model=model_name, reason=str(exc)[:160]
                )
                if self._quota_manager is not None:
                    self._quota_manager.reconcile(
                        reservation, self._estimate_response_failure_tokens(est_tokens)
                    )
                last_error = exc
                if attempt < self._retry_count - 1:
                    delay = self._retry_delay(attempt, base_seconds=2.0, max_seconds=20.0)
                    logger.warning(
                        "Model trả batch sai cấu trúc (lần %d/%d): %s. Chờ %.1fs rồi thử lại...",
                        attempt + 1,
                        self._retry_count,
                        exc,
                        delay,
                    )
                    if self._interruptible_sleep(delay, cancel_cb):
                        raise TranslationCancelledError("Người dùng đã huỷ tiến trình dịch.") from exc
                    continue
                # Hết retry batch validation → kích hoạt halving đệ quy ở tầng trên.
                raise _BatchCountMismatchError(
                    f"Model liên tục trả sai số dòng sau {self._retry_count} lần: {exc}"
                ) from exc
            except _TruncatedOutputError as exc:
                last_error = exc
                lowered = self._lower_thinking_level(model_name, config)
                if lowered is not None:
                    config = lowered
                    logger.warning(
                        "%s → hạ mức suy nghĩ xuống '%s' rồi thử lại (video giữ nguyên).",
                        exc, self._config_thinking_level(config),
                    )
                    _record_process_event(
                        "thinking_lowered_on_truncation", model=model_name,
                        level=self._config_thinking_level(config),
                    )
                    continue
                if attempt < self._retry_count - 1:
                    delay = self._retry_delay(attempt, base_seconds=2.0, max_seconds=20.0)
                    logger.warning("%s Chờ %.1fs rồi thử lại...", exc, delay)
                    if self._interruptible_sleep(delay, cancel_cb):
                        raise TranslationCancelledError(
                            "Người dùng đã huỷ tiến trình dịch."
                        ) from exc
                    continue
                raise
            except _EmptyResponseError as exc:
                last_error = exc
                # [v3.23.358] candidates=None ĐÃ tính quota ở server (consumed=True) nên
                # KHÔNG release. Trước đây lỗi này bị raise NGAY → một lô rỗng/bị chặn giết
                # CẢ JOB. Nay: nếu đang đính video (tác nhân thường gặp gây rỗng/safety),
                # BỎ video rồi thử lại NGAY; nếu vẫn rỗng thì retry backoff; hết retry mới
                # bỏ cuộc để tầng trên (halving/keep-original) xử lý.
                # [v3.23.591] Chỉ bỏ video khi máy chủ NÊU lý do chặn, hoặc đã rỗng tới
                # lần thứ hai. Nhật ký thật: hai lần trả rỗng đều giữa lúc quá tải, và
                # cùng đoạn video ấy chạy bình thường ở lần phân tích sau — bỏ video ngay
                # là để mất căn cứ gán người nói của cả đoạn.
                if not exc.diagnosis.blocked:
                    unexplained_empties += 1
                if current_handles and not video_dropped_for_empty:
                    if should_drop_video(exc.diagnosis, unexplained_empties):
                        video_dropped_for_empty = True
                        current_handles = []
                        contents = [prompt]
                        logger.warning(
                            "Gemini trả rỗng khi đính video (%s) — thử lại KHÔNG đính "
                            "video.", exc.diagnosis.describe(),
                        )
                        continue
                    # Lượt thử lại kèm video không tính vào số lượt thử thường, để sau
                    # đó vẫn còn đủ lượt cho phương án không video.
                    attempt_offset += 1
                    delay = self._retry_delay(attempt, base_seconds=2.0, max_seconds=20.0)
                    logger.warning(
                        "Gemini trả rỗng khi đính video (%s) — coi là nhất thời: chờ "
                        "%.1fs rồi thử lại VẪN KÈM video (rỗng lần nữa mới bỏ video).",
                        exc.diagnosis.describe(), delay,
                    )
                    if self._interruptible_sleep(delay, cancel_cb):
                        raise TranslationCancelledError(
                            "Người dùng đã huỷ tiến trình dịch."
                        ) from exc
                    continue
                if attempt < self._retry_count - 1:
                    delay = self._retry_delay(attempt, base_seconds=2.0, max_seconds=20.0)
                    logger.warning(
                        "Gemini trả rỗng (lần %d/%d): %s. Chờ %.1fs rồi thử lại...",
                        attempt + 1, self._retry_count, exc, delay,
                    )
                    if self._interruptible_sleep(delay, cancel_cb):
                        raise TranslationCancelledError(
                            "Người dùng đã huỷ tiến trình dịch."
                        ) from exc
                    continue
                raise
            except SubtitleTranslationError:
                raise
            except Exception as exc:
                last_error = exc
                # [v3.23.123] Generate THẤT BẠI (429/mạng…) → server KHÔNG trừ quota.
                # Hoàn trả chỗ đã đặt (RPM + token + bộ đếm NGÀY) để không báo hết quota oan.
                # [v3.23.585] Trừ lỗi QUÁ TẢI (503): server vẫn tính lượt gọi → giữ
                # nguyên bộ đếm.
                # [v3.23.587] 403 do file video thuộc key khác cũng bị tính. Trên 6 nhật
                # ký có 10 lần một key VỪA XOAY (còn đủ 20 request) bị server chặn: cộng
                # lượt thành công + 503 + 403 thì ra 19-21 lượt ở cả 10 lần; bỏ 403 ra
                # chỉ còn 10-19. Không đếm thì ứng dụng ngỡ key còn hạn mức.
                if not consumed and reservation is not None and self._quota_manager is not None:
                    self._quota_manager.release(
                        reservation,
                        request_counted=self._is_overload_error(exc)
                        or self._is_file_permission_error(exc),
                    )
                    reservation = None
                # [v3.23.639] Server từ chối CHÍNH API key (400 API_KEY_INVALID / key hết hạn / bị
                # vô hiệu hoá vì lộ): cách ly key đó rồi thử lại NGAY bằng key khác. Trước đây lỗi
                # này là 400 "không thể thử lại" → hỏng cả lượt dịch dù các key khác còn tốt. Phải
                # xét TRƯỚC các nhánh 400 khác vì chữ INVALID_ARGUMENT làm chúng nhận nhầm.
                if self._is_invalid_key_error(exc):
                    invalid_key_rotations += 1
                    if (
                        invalid_key_rotations <= len(self._api_keys)
                        and self._quarantine_key(sent_key_index, exc)
                    ):
                        video_needs_refresh = True
                        attempt_offset = raw_attempt + 1
                        continue
                    raise SubtitleTranslationError(
                        "Mọi API key đều bị Gemini từ chối (không hợp lệ / hết hạn / bị vô hiệu "
                        f"hoá): {self._concise_error(exc)}. Kiểm tra lại API key trong phần "
                        "cài đặt."
                    ) from exc
                # [v3.23.620] 400 CHUNG CHUNG ("Request contains an invalid argument") khi gửi
                # schema có ràng buộc → schema quá phức tạp với Gemini. Nới một bậc ràng buộc rồi
                # thử lại NGAY. Trước đây lỗi này bị nhận nhầm là "model không hỗ trợ thinking"
                # (khớp chữ invalid_argument): tắt thinking, vẫn 400, hỏng cả lượt dịch.
                if self._is_generic_invalid_argument(exc):
                    relaxed_schema = relax_schema_constraints(
                        getattr(config, "response_schema", None)
                    )
                    relaxed = (
                        self._copy_config(config, response_schema=relaxed_schema)
                        if relaxed_schema is not None else None
                    )
                    if relaxed is not None:
                        config = relaxed
                        logger.info(
                            "Gemini từ chối schema có ràng buộc (400) với '%s' → nới ràng "
                            "buộc và thử lại.", model_name,
                        )
                        continue
                # [v3.23.610] Model không nhận mức "minimal" (gửi khi TẮT suy nghĩ) → hạ
                # xuống "low" và thử lại NGAY; các lô sau của model này gửi thẳng "low".
                if self._is_thinking_level_rejection(exc):
                    lowered = self._downgrade_minimal_level(model_name, config)
                    if lowered is not None:
                        config = lowered
                        logger.info(
                            "Model '%s' không nhận mức suy nghĩ 'minimal' → dùng 'low'.",
                            model_name,
                        )
                        continue
                # [Thinking fallback] Ưu tiên bật thinking; nếu model THỰC SỰ không
                # hỗ trợ → bỏ thinking_config và thử lại NGAY (không tính một lượt
                # retry), giữ chất lượng cho model có hỗ trợ.
                if not thinking_stripped and self._is_thinking_unsupported_error(exc):
                    config = self._strip_thinking_from_config(config)
                    thinking_stripped = True
                    logger.info(
                        "Model '%s' không hỗ trợ thinking → tự tắt và thử lại.", model_name
                    )
                    continue
                # [v3.23.132] File video thuộc key cũ (đã xoay key) → 403 PERMISSION_DENIED.
                # Tải lại đoạn bằng key HIỆN TẠI rồi thử lại NGAY (không tính 1 lượt retry).
                # [v3.23.449] KHÔNG đòi 'video_refs' của batch hiện tại: chính lỗi 403-file đã
                # CHỨNG MINH có file cũ cần tải lại (đã gặp ca batch có video nhưng trạng thái
                # video_refs bị lệch sau khi xoay key trên 429 → 403 không được cứu → mất tập).
                if (
                    self._video_reupload_cb is not None
                    and not video_needs_refresh
                    and self._is_file_permission_error(exc)
                    and attempt < self._retry_count - 1
                ):
                    logger.info(
                        "File video bị 403 (thuộc key cũ) → tải lại bằng key hiện tại."
                    )
                    video_needs_refresh = True
                    continue
                overload = self._is_overload_error(exc)
                retryable = self._is_retryable(exc)
                # [v3.23.121] Nếu là HẾT QUOTA NGÀY và còn API key khác: đánh dấu key
                # hiện tại đã cạn rồi XOAY sang key kế — thử lại NGAY, khỏi chờ hồi.
                # [v3.23.585] Xoay key KHÔNG phải một lượt thử lại nên không phụ thuộc số
                # lượt còn lại. Nhật ký thật: 429 hết hạn mức ngày rơi đúng lượt thử CUỐI
                # (các lượt trước đã tiêu vào 503) → đoạn phân tích bị bỏ dù còn key khác;
                # đoạn kế tiếp mới xoay key. Xảy ra 4 lần trong một ngày (đoạn 2, 7, 14,
                # 14).
                _daily = retryable and self._is_daily_quota_error(exc)
                if _daily and self._quota_manager is not None:
                    # [v3.23.528] Hạn mức NGÀY của Gemini tính theo DỰ ÁN, không theo key.
                    # Nhiều key cùng một dự án dùng chung MỘT hũ quota, trong khi ứng dụng đếm
                    # riêng từng key nên tưởng còn nhiều. Quan sát thật: bảng điều khiển ghi
                    # 3.8 Flash mới dùng 0/20 và 1/20 ở hai dự án, nhưng server vẫn từ chối cả
                    # ba key — dấu hiệu điển hình của việc các key chung dự án. Ghi rõ để người
                    # dùng đối chiếu thay vì ngỡ ứng dụng đếm sai.
                    self._log_daily_quota_context(exc, model_name, sent_key_id)
                    self._quota_manager.mark_daily_exhausted(model_name, key_id=sent_key_id)
                    if self._after_daily_exhausted(model_name, sent_key_index):
                        video_needs_refresh = True
                        attempt_offset = raw_attempt + 1
                        # Server vừa nêu hạn mức thật → từ đây có thể đã thành "ít".
                        # Số lần 503 thì GIỮ: key mới nhận phần lượt thử còn lại; chạm
                        # trần rồi mới tới lượt thăm dò các key khác (v3.23.592).
                        overload_cap = self._overload_attempt_cap(model_name)
                        overload_key_index = self._key_index
                        logger.info(
                            "Hết quota ngày ở key hiện tại → đã xoay sang key khác, "
                            "thử lại ngay với '%s'.", model_name,
                        )
                        continue
                if _daily and _leave_exhausted_fallback(raw_attempt):
                    continue
                if _daily:
                    # [v3.23.586] Hết hạn mức NGÀY mà không còn key nào: chờ là vô ích —
                    # hạn mức chỉ hồi vào mốc đổi ngày. Nhật ký thật: server gửi kèm
                    # retryDelay 36738 giây và ứng dụng NGỦ theo, đứng im từ 20:47 tới khi
                    # người dùng bấm huỷ lúc 05:44.
                    # [v3.23.607] Không dùng ``retryDelay`` làm mốc hồi: với lỗi hết
                    # hạn mức ngày nó luôn chỉ về 00:00 UTC, còn hạn mức hồi lúc 00:00
                    # giờ Thái Bình Dương (xem ``gemini_quota_clock``).
                    raise QuotaExhaustedError(daily_quota_message(model_name)) from exc
                if overload:
                    overload_failures += 1
                    if sent_key_index != overload_key_index:
                        # Lượt này đã đi bằng key khác (luồng khác vừa chuyển key).
                        overload_key_index = sent_key_index
                        key_overload_failures = 0
                    key_overload_failures += 1
                    if self._model_health is not None:
                        self._model_health.record_overload(model_name)
                        # [v3.23.631] Quá tải là của MODEL: sau FAILOVER_AFTER_IN_CALL lượt (hoặc
                        # model đã bị ngắt mạch) thì thử model khác thay vì chờ 5-11 lượt.
                        if (
                            overload_failures >= FAILOVER_AFTER_IN_CALL
                            or self._model_health.cooling_remaining(model_name) > 0
                        ) and _fail_over("quá tải kéo dài", raw_attempt):
                            continue
                # [v3.23.592] Trần tính theo key đang dùng: key đầu được thử đủ lượt, các
                # key sau chỉ thăm dò. Chạm trần mà còn key chưa thử → chuyển key, gửi
                # lại NGAY (key khác có thể đang không quá tải).
                key_cap = overload_key_cap(overload_cap, len(overload_keys_tried))
                # [v3.23.631] KHÔNG xoay key vì quá tải khi lời gọi đính VIDEO: nhật ký thật 17 lần
                # xoay chỉ 2 lần được trả lời ngay, mà mỗi lần xoay buộc tải lại đoạn video bằng
                # key mới (~40 giây + request bị tính hạn mức).
                if (
                    overload
                    and not (video_refs or current_handles)
                    and should_try_other_key(key_cap, key_overload_failures)
                    and self._rotate_for_overload(
                        model_name, overload_keys_tried, sent_key_index
                    )
                ):
                    video_needs_refresh = True
                    key_overload_failures = 0
                    overload_key_index = self._key_index
                    continue
                capped = (
                    overload
                    and key_cap is not None
                    and key_overload_failures >= key_cap
                )
                has_overload_budget = (
                    overload and not capped and overload_waited < OVERLOAD_RETRY_BUDGET_S
                )
                if retryable and not capped and (
                    attempt < self._retry_count - 1 or has_overload_budget
                ):
                    # [v3.23.9] Ưu tiên retryDelay server gửi kèm 429 (chờ ĐÚNG bằng
                    # thời gian quota hồi); chỉ dùng backoff mù khi server không nói.
                    # [v3.23.586] Kẹp trần như bộ quản lý quota đã làm từ v3.23.534: không
                    # lỗi nào đáng để ứng dụng ngủ hàng giờ theo lời server.
                    server_delay = self._server_retry_delay(exc)
                    delay = (
                        self._retry_delay(attempt)
                        if server_delay is None
                        else min(server_delay, MAX_SERVER_COOLDOWN_SEC)
                    )
                    if overload and overload_cap is not None:
                        # Hạn mức ngày ít: ít lượt hơn nhưng chờ thưa dần (30, 60, 120…).
                        # [v3.23.592] Tính theo key đang dùng: lượt thăm dò thứ hai trên
                        # key mới chờ 30 giây, không phải 120.
                        delay = scarce_overload_delay(key_overload_failures)
                        overload_waited += delay
                    elif overload:
                        # Server bận: chờ thưa (trần riêng), tính vào ngân sách quá tải.
                        delay = min(max(delay, 30.0), OVERLOAD_MAX_DELAY_S)
                        overload_waited += delay
                    # [v3.23.141] Ghi cooldown vào quota manager để các request/đoạn KẾ TIẾP
                    # tới cùng (key,model) cũng CHỜ, thay vì lao vào rồi 429 nối tiếp.
                    if self._is_rate_limit_error(exc):
                        self._learn_minute_limits(exc, model_name)
                    if server_delay is not None and self._quota_manager is not None:
                        self._quota_manager.note_rate_limited(
                            model_name, server_delay, key_id=sent_key_id
                        )
                    # [v3.23.147] 429 TẠM THỜI (TPM/RPM) trên lời gọi KHÔNG đính video:
                    # xoay ngay sang key còn quota + không cooldown (miễn phí vì không có
                    # file gắn key) -> tiếp tục tức thời thay vì chờ retryDelay.
                    if (
                        not video_refs
                        and self._is_rate_limit_error(exc)
                        and self._rotate_for_temporary_429(model_name)
                    ):
                        continue
                    config, request_timeout_s = self._extend_timeout_after_deadline(
                        config, request_timeout_s, time.monotonic() - attempt_started
                    )
                    if attempt >= self._retry_count - 1:
                        logger.warning(
                            "Gemini đang QUÁ TẢI: %s. Hết %d lượt thử thường nhưng vẫn chờ "
                            "thêm (%.0fs/%.0fs ngân sách) để KHÔNG mất cả lượt dịch. "
                            "Chờ %.0fs rồi thử lại...",
                            self._concise_error(exc),
                            self._retry_count,
                            overload_waited,
                            OVERLOAD_RETRY_BUDGET_S,
                            delay,
                        )
                    else:
                        logger.warning(
                            "Lỗi tạm thời từ Gemini (lần %d/%d): %s. Chờ %.1fs rồi thử lại%s...",
                            attempt + 1,
                            self._retry_count,
                            self._concise_error(exc),
                            delay,
                            " (theo retryDelay server)" if server_delay is not None else "",
                        )
                    if self._interruptible_sleep(delay, cancel_cb):
                        raise TranslationCancelledError("Người dùng đã huỷ tiến trình dịch.") from exc
                    continue
                if capped:
                    logger.warning(
                        "Gemini QUÁ TẢI: dừng sau %d lượt thử trên %d key để GIỮ hạn mức "
                        "ngày của '%s' — mỗi lượt 503 vẫn bị server tính vào hạn mức.",
                        overload_failures, max(1, len(overload_keys_tried)), model_name,
                    )
                if overload:
                    # [v3.23.616] Không đẩy nguyên khối JSON tiếng Anh của máy chủ
                    # lên giao diện (nhật ký 14/10: ba lần liền) — nói rõ đã thử bao
                    # nhiêu lượt, trong bao lâu.
                    raise ModelOverloadedError(
                        overload_failure_message(
                            model_name,
                            attempts=overload_failures,
                            keys=max(1, len(overload_keys_tried)),
                            waited_s=time.monotonic() - call_started,
                            reason=self._concise_error(exc),
                        ),
                        model_name=model_name,
                    ) from exc
                raise SubtitleTranslationError(f"Gọi Gemini thất bại: {exc}") from exc

        raise SubtitleTranslationError(f"Gọi Gemini thất bại sau {self._retry_count} lần: {last_error}")

    @staticmethod
    def _log_cache_savings(response: Any) -> None:
        """[v3.23.424] Ghi log token được CACHE (implicit caching của Gemini 2.5+/3.x).

        Cookbook Gemini khuyến nghị theo dõi ``cached_content_token_count`` để xác nhận caching
        đang hoạt động. App đã cấu trúc chuẩn (nội dung ổn định — glossary/roster/luật — nằm ở
        ``system_instruction``; video handle đứng đầu ``contents``), nên các lần gọi liên tiếp
        (xử lý chunk tuần tự) tự động ăn cache prefix → giảm token tính phí. Chỉ LOG (không đổi
        hành vi) để người dùng thấy phần token tiết kiệm.
        """
        meta = getattr(response, "usage_metadata", None)
        if meta is None:
            return
        cached = getattr(meta, "cached_content_token_count", None)
        if isinstance(cached, int) and cached > 0:
            prompt = getattr(meta, "prompt_token_count", None) or 0
            ratio = f" (~{cached * 100 // prompt}% prompt)" if prompt else ""
            logger.info(
                "Gemini implicit caching: %d token được tái dùng từ cache%s — tiết kiệm chi phí.",
                cached, ratio,
            )

    def _record_token_usage(self, model_name: str, response: Any) -> None:
        """[v3.23.610] Ghi token của một lời gọi máy chủ đã trả lời vào sổ token."""
        ledger = getattr(self, "_token_ledger", None)
        if ledger is not None:
            ledger.record(model_name, self._usage_from_response(response))

    @staticmethod
    def _usage_from_response(response: Any) -> TokenUsage:
        """Token theo ``usage_metadata`` (trường thiếu → 0)."""
        meta = getattr(response, "usage_metadata", None)

        def _count(field: str) -> int:
            return max(0, _safe_int(getattr(meta, field, None), 0))

        return TokenUsage(
            requests=1,
            prompt=_count("prompt_token_count"),
            cached=_count("cached_content_token_count"),
            thoughts=_count("thoughts_token_count"),
            output=_count("candidates_token_count"),
        )

    @staticmethod
    def _extract_token_count(response: Any, fallback: int) -> int:
        """Lấy tổng token thực tế từ phản hồi Gemini (usage_metadata)."""
        meta = getattr(response, "usage_metadata", None)
        if meta is not None:
            total = getattr(meta, "total_token_count", None)
            if isinstance(total, int) and total > 0:
                return total
        return max(1, fallback)

    @staticmethod
    def _estimate_response_failure_tokens(est_tokens: int) -> int:
        """Khi gọi lỗi cấu trúc, vẫn coi như đã tiêu thụ ~token đầu vào đã đặt."""
        return max(1, est_tokens)

    def _apply_video_fps(self, video_files: list[Any] | None) -> list[Any]:
        """[v3.23.476] Bọc handle video với ``videoMetadata.fps`` khi fps>1 → Gemini LẤY MẪU DÀY
        hơn (nhiều chi tiết thời gian hơn, nhiều token hơn). fps≈1 (mặc định) → giữ handle THÔ
        (hành vi cũ, đang chạy tốt). Lỗi bọc bất kỳ → fallback handle thô để KHÔNG làm hỏng call."""
        files = list(video_files or [])
        use_fps = self._analysis_video_fps > 1.05
        if not files or not (use_fps or any(isinstance(h, ClippedVideoHandle) for h in files)):
            return [h.handle if isinstance(h, ClippedVideoHandle) else h for h in files]
        try:
            from google.genai import types
        except ImportError:
            return [h.handle if isinstance(h, ClippedVideoHandle) else h for h in files]
        wrapped: list[Any] = []
        for handle in files:
            raw = handle.handle if isinstance(handle, ClippedVideoHandle) else handle
            try:
                uri = getattr(raw, "uri", None)
                mime = getattr(raw, "mime_type", None) or "video/mp4"
                if not uri:
                    wrapped.append(raw)
                    continue
                metadata: dict[str, Any] = {}
                if use_fps:
                    # Kẹp ≤ 24 (giới hạn cứng của Gemini) — an toàn tuyệt đối.
                    metadata["fps"] = min(24.0, self._analysis_video_fps)
                if isinstance(handle, ClippedVideoHandle):
                    # [v3.23.648] Chỉ xử lý khoảng cần xem của đoạn (tiết kiệm token, xem video_clip_window).
                    metadata["start_offset"] = f"{handle.start_s:.2f}s"
                    metadata["end_offset"] = f"{handle.end_s:.2f}s"
                wrapped.append(
                    types.Part(
                        file_data=types.FileData(file_uri=uri, mime_type=mime),
                        video_metadata=types.VideoMetadata(**metadata),
                    )
                )
            except Exception:  # noqa: BLE001 — bọc lỗi thì dùng handle thô (an toàn)
                wrapped.append(raw)
        return wrapped

    def _clip_batch_handles(
        self, handles: list[Any], covering: list[Any], batch_start_s: float, batch_end_s: float,
    ) -> tuple[list[Any], float]:
        """[v3.23.648] Bọc từng handle bằng khoảng cần xem của lô (nếu tiết kiệm đáng kể).

        Args:
            handles: Handle đã resolve của các đoạn phủ lô, theo thứ tự ``covering``.
            covering: Ref các đoạn phủ lô.
            batch_start_s: Mốc bắt đầu dòng đầu lô (giây trong phim).
            batch_end_s: Mốc kết thúc dòng cuối lô.

        Returns:
            ``(handles đã bọc, số giây video thực sự gửi)``. Số handle thiếu so với ``covering`` (resolve hỏng) thì
            không cắt gì — không đoán handle nào ứng với đoạn nào.
        """
        total_s = sum(
            max(0.0, float(getattr(r, "end_sec", 0.0)) - float(getattr(r, "start_sec", 0.0)))
            for r in covering
        )
        if not getattr(self, "_clip_video_to_batch", True) or len(handles) != len(covering):
            return handles, total_s
        from subtitles_extractor.application.services.video_clip_window import clip_window

        clipped: list[Any] = []
        sent = 0.0
        for handle, ref in zip(handles, covering, strict=True):
            chunk_start = float(getattr(ref, "start_sec", 0.0))
            chunk_end = float(getattr(ref, "end_sec", 0.0))
            window = clip_window(chunk_start, chunk_end, batch_start_s, batch_end_s)
            if window is None:
                clipped.append(handle)
                sent += max(0.0, chunk_end - chunk_start)
            else:
                clipped.append(ClippedVideoHandle(handle, *window))
                sent += window[1] - window[0]
        if sent < total_s:
            logger.debug("Lô đính video: gửi %.0f s/%.0f s (cắt theo khung lô).", sent, total_s)
        return clipped, sent

    def _handles_for_batch(
        self, covering: list[Any], all_refs: list[Any], batch_idx: int
    ) -> list[Any]:
        """[v3.23.561] Handle video cho các đoạn phủ MỘT lô — tự tải lại khi thiếu.

        Đường nhanh: handle đã cache/resolve được đủ thì dùng luôn. Thiếu (file thuộc key
        cũ sau khi xoay key, hoặc vừa đổi key) thì tải lại đúng các đoạn đó bằng key hiện
        tại, rồi GHI ref mới NGƯỢC vào ``all_refs`` để các lô sau không 403 lần nữa.

        Args:
            covering: Các ref phủ lô (được cập nhật tại chỗ thành ref mới nếu tải lại).
            all_refs: Danh sách ref của cả giai đoạn (được cập nhật tại chỗ).
            batch_idx: Chỉ số lô (0-based), chỉ để ghi log.

        Returns:
            Danh sách handle đính kèm lô; có thể thiếu nếu tải lại cũng thất bại.
        """
        if not covering:
            return []
        # [v3.23.605] Ref đã tải lại ở lô/giai đoạn trước bằng key hiện tại thì dùng ngay.
        all_current = self._apply_healed_refs(covering)
        if all_current or not getattr(self, "_video_handles_stale", False):
            handles = self._resolve_video_handles(covering)
            if len(handles) >= len(covering):
                return handles
        healed = list(covering)
        handles = self._resolve_handles_with_heal(healed, [], probe=False)
        replace_refs_by_range(all_refs, healed)
        covering[:] = healed
        if len(handles) < len(covering):
            logger.warning(
                "Lô %d gửi THIẾU video (%d/%d đoạn) — tải lại bằng key hiện tại thất "
                "bại; lô này dịch dựa trên chữ.",
                batch_idx + 1, len(handles), len(covering),
            )
            # Vào file chuẩn đoán: lần trước lỗi này chỉ lộ qua 261 dòng INFO nhật ký.
            _record_process_event(
                "video_missing", batch=batch_idx + 1,
                attached=len(handles), needed=len(covering),
            )
        return handles

    def _refresh_segment_handle(
        self,
        idx: int,
        video_handles: list[Any],
        video_refs: list[Any],
        handle_keys: list[str],
    ) -> None:
        """[v3.23.587] Bảo đảm handle video của đoạn ``idx`` thuộc API key HIỆN TẠI.

        Khâu phân tích resolve handle của MỌI đoạn một lần lúc bắt đầu. Khi key bị xoay
        giữa chừng, chỉ đoạn đang gửi được tải lại; các đoạn sau vẫn mang handle của key
        cũ nên lời gọi đầu tiên của từng đoạn nhận 403 rồi mới tải lại — mỗi 403 là một
        request của key mới. Hàm này tải lại TRƯỚC khi gửi.

        Hai key cùng một dự án dùng chung file: khi đó handle resolve lại được ngay,
        không phải tải lên lần nữa.

        Args:
            idx: Chỉ số đoạn (0-based).
            video_handles: Handle của mọi đoạn (cập nhật tại chỗ).
            video_refs: Ref của mọi đoạn (cập nhật tại chỗ thành ref mới nếu tải lại).
            handle_keys: Vân tay của key đã resolve handle từng đoạn (cập nhật tại chỗ).
        """
        key_id = self._current_key_id()
        if idx >= len(video_refs) or handle_keys[idx] == key_id:
            return
        segment_refs = [video_refs[idx]]
        handles = self._resolve_handles_with_heal(segment_refs, [])
        if not handles:
            # Tải lại hỏng: giữ handle cũ, nhánh 403 trong ``_call_gemini`` thử lại sau.
            return
        video_handles[idx] = handles[0]
        video_refs[idx] = segment_refs[0]
        handle_keys[idx] = key_id

    def _resolve_video_handles(self, video_refs: list[Any] | None) -> list[Any]:
        """Đổi danh sách RemoteVideoRef → handle file Gemini để đính vào contents.

        Có cache theo ``remote_name`` để không gọi lại ``files.get`` nhiều lần.
        Bỏ qua (cảnh báo) đoạn nào không resolve được thay vì làm hỏng cả lần dịch.
        """
        if not video_refs:
            return []
        client = self._get_client()
        handles: list[Any] = []
        # [v3.23.631] File đã bị 403 với key HIỆN TẠI thì key đó mãi mãi không đọc được nó (file
        # gắn với dự án đã tải lên): không dò lại ở các lượt thử/lô sau — nhật ký thật 93 lần
        # "cần tải lại (403)", phần lớn là dò lặp lại cùng một file.
        foreign: set[tuple[str, str]] = self.__dict__.setdefault("_foreign_handles", set())
        key_id = self._current_key_id()
        for ref in video_refs:
            name = getattr(ref, "remote_name", None)
            if not name:
                continue
            handle = self._video_handle_cache.get(name)
            if handle is None:
                if (key_id, name) in foreign:
                    continue
                try:
                    handle = client.files.get(name=name)
                    self._video_handle_cache[name] = handle
                except Exception as exc:  # noqa: BLE001
                    # [v3.23.481] Thường là 403 do XOAY KEY (file thuộc key cũ) — đường phân tích
                    # sẽ TỰ tải lại (heal). Hạ về INFO + rút gọn để không gây hiểu lầm "lỗi nặng".
                    if self._is_file_permission_error(exc):
                        foreign.add((key_id, name))
                    logger.info(
                        "Handle video '%s' cần tải lại (%s).", name, self._concise_error(exc)
                    )
                    continue
            handles.append(handle)
        return handles

    # ── Dựng system instruction theo giai đoạn ───────────────────────────
    @staticmethod
    def _canonical_names_directive(characters: str) -> str:
        """[v3.23.491] Dựng BẢNG QUY ĐỔI TÊN (gốc → đích) + chỉ thị dùng nhất quán.

        Vì sao cần bảng quy đổi, không chỉ danh sách tên đích: từ v3.23.490, nhãn ``spk``/``to``
        trong Visual Cues luôn là TÊN GỐC (để khớp máy móc với roster). Nếu prompt dịch chỉ đưa
        danh sách tên tiếng đích, model phải TỰ ĐOÁN "林逸" ứng với "Lâm Dật" — nguồn gây gọi sai
        tên/đảo nhân vật. Nay đưa thẳng cặp ánh xạ kèm giới tính/tuổi để chọn đại từ đúng.

        Uỷ quyền phân tích roster cho ``roster_format`` (nguồn chân lý duy nhất) thay vì tự dò
        bằng biểu thức chính quy riêng như trước.

        Args:
            characters: Bảng nhân vật (quy cách cố định hoặc dữ liệu cũ).

        Returns:
            Chỉ thị chèn vào system prompt; rỗng nếu roster không có tên dùng được.
        """
        if not characters:
            return ""
        from subtitles_extractor.application.services.roster_format import parse_roster

        entries = parse_roster(characters)
        if not entries:
            return ""
        mappings: list[str] = []
        target_only: list[str] = []
        for entry in entries[:60]:  # cap để prompt không phình
            traits = ", ".join(t for t in (entry.gender, entry.age) if t)
            suffix = f" [{traits}]" if traits else ""
            if entry.aka and entry.aka != entry.name:
                mappings.append(f"{entry.name} → {entry.aka}{suffix}")
                target_only.append(entry.aka)
            else:
                mappings.append(f"{entry.name}{suffix}")
                target_only.append(entry.name)
        if not mappings:
            return ""
        return (
            "\n- TÊN NHÂN VẬT (BẮT BUỘC NHẤT QUÁN): mỗi khi nhắc đến nhân vật trong "
            "Roster (kể cả nhãn 'speaker'/'addressee' và trong lời thoại), PHẢI dùng ĐÚNG tên "
            "tiếng đích đã cho, viết GIỐNG HỆT (hoa/thường, dấu); KHÔNG tự tạo cách phiên âm "
            "khác, KHÔNG đổi qua lại giữa các biến thể tên cho CÙNG một người xuyên suốt "
            "phim bộ. Nhân vật chưa có trong Roster: chọn MỘT cách dịch tên rồi giữ nguyên."
            "\n  BẢNG QUY ĐỔI TÊN (tên gốc → tên dùng trong bản dịch): "
            + "; ".join(mappings)
            + ".\n  LƯU Ý: nhãn speaker/addressee kèm mỗi dòng được ghi bằng TÊN GỐC — hãy tra "
            "bảng trên để biết đó là ai, rồi trong BẢN DỊCH chỉ dùng TÊN TIẾNG ĐÍCH tương ứng "
            "(TUYỆT ĐỐI không để lọt chữ gốc vào bản dịch)."
        )

    def _system_instruction(
        self,
        stage: TranslationStageConfig,
        context: TranslationContext,
        has_prior: bool,
        *,
        with_video: bool = False,
    ) -> str:
        overview = context.overview or "chưa có tóm tắt"
        # [v3.23.591] Chỉ nhắc vai trò của video ở giai đoạn THẬT SỰ đính video.
        video_rule = VIDEO_ROLE_RULE if with_video else ""
        characters = context.characters or "chưa có"
        glossary = (context.glossary or "").strip()
        # [v3.23.500] Ghi nhớ glossary của phiên để mỗi LÔ trích ra đúng thuật ngữ xuất hiện trong
        # lô đó và nhắc lại ngay cạnh dòng cần dịch (xem ``_batch_glossary_block``).
        self._active_glossary = glossary
        # [v3.23.516] Ghi nhớ bảng nhân vật để dựng MỐC NEO kiểm căn dòng ngay tại từng lô.
        self._active_characters = characters
        # [v3.23.53] Chỉ thị bám bảng thuật ngữ MẠNH hơn (theo best practice dịch LLM:
        # "force terminology compliance" — cấm tự ý dùng từ đồng nghĩa khác).
        glossary_instr = ""
        if glossary:
            glossary_instr = (
                f"\n- BẢNG THUẬT NGỮ (BẮT BUỘC TUYỆT ĐỐI): khi gặp thuật ngữ trong bảng, "
                f"PHẢI dùng đúng bản dịch đã quy định, KHÔNG tự thay bằng từ đồng nghĩa dù "
                f"bạn cho là hay hơn. Áp dụng nhất quán xuyên suốt; giữ nguyên viết tắt "
                f"thông dụng (FBI/NASA/IBM). LƯU Ý: nếu một mục có chú thích trong NGOẶC "
                f"(vd 'Tên (giải thích)'), phần trong ngoặc CHỈ để bạn HIỂU nghĩa — trong "
                f"bản dịch CHỈ dùng phần TÊN/THUẬT NGỮ chính TRƯỚC ngoặc, TUYỆT ĐỐI KHÔNG "
                f"chèn chú thích đó vào lời thoại. Bảng: {glossary}"
            )

        # [v3.23.85] Chỉ thị bắt buộc dùng TÊN NHÂN VẬT nhất quán theo roster (force name
        # compliance) — kèm danh sách tên chuẩn để model bám sát, tránh đổi tên xuyên tập.
        names_instr = self._canonical_names_directive(context.characters or "")

        # [v3.23.139] CĂN DÒNG 1:1 — chống lỗi model gộp NGỮ NGHĨA khi một câu bị tách
        # qua nhiều dòng phụ đề (vd "...world-famous" + "for one thing.") rồi dồn cả câu
        # vào dòng đầu, kéo nội dung dòng sau lên -> lệch dây chuyền + trùng dòng. Quy tắc
        # cũ ("không gộp/tách dòng") quá chung; model không nhận ra dồn nghĩa CŨNG là gộp.
        line_align = (
            "- CĂN DÒNG 1:1 (CỰC KỲ QUAN TRỌNG): mỗi line_no là MỘT dòng phụ đề độc lập; "
            "bản dịch của line_no X chỉ được mang nội dung của ĐÚNG dòng X nguồn. Khi một "
            "CÂU bị tách qua nhiều dòng (dòng trước chưa có dấu kết câu, ý nối sang dòng "
            "sau), hãy CHIA phần dịch theo ĐÚNG ranh giới từng dòng — mỗi dòng giữ đúng "
            "phần ý của nó (được phép là một mảnh câu chưa trọn vẹn). TUYỆT ĐỐI KHÔNG dồn "
            "nghĩa nhiều dòng vào MỘT dòng rồi bỏ trống/kéo nội dung dòng sau lên; KHÔNG "
            "để dòng X mang nội dung dòng X+1; KHÔNG lặp lại một câu ở hai dòng liền kề. "
            "Giữ nguyên tuyệt đối line_no, không đổi thứ tự, không gộp, không tách dòng.\n"
            # [v3.23.502] Đo trên bản dịch thật: 62 dòng bị LỆCH nội dung, có ca nghiêm trọng
            # như "睁大你们的狗眼" (mở to mắt ra mà nhìn) bị dịch thành "Maybach" — nội dung của
            # dòng khác lọt sang. Quy tắc chung ở trên chưa đủ; bổ sung ví dụ lỗi cụ thể + bước
            # TỰ KIỂM bắt buộc, áp cho MỌI giai đoạn dịch.
            "- CẤM MƯỢN NỘI DUNG DÒNG KHÁC: bản dịch của mỗi dòng phải bắt nguồn từ CHÍNH "
            "chữ trong dòng đó. Ví dụ lỗi PHẢI TRÁNH: dòng gốc nói 'mở to mắt ra mà nhìn' "
            "nhưng lại dịch thành tên một chiếc xe (nội dung của dòng bên cạnh) — đây là lỗi "
            "nặng nhất, làm hỏng cả đoạn thoại.\n"
            "- KHÔNG BỎ NGẮN BẤT THƯỜNG: nếu dòng gốc có nhiều chữ mà bản dịch chỉ còn một hai "
            "tiếng rời rạc ('chứ?', 'Phi!', 'Có lẽ') thì gần như chắc chắn bạn đã đánh rơi nội "
            "dung. Hãy dịch ĐỦ ý của dòng đó. Thà dịch sát chữ còn hơn bỏ mất nội dung.\n"
            "- TỰ KIỂM TRƯỚC KHI TRẢ KẾT QUẢ (bắt buộc, làm thầm): với TỪNG dòng, đối chiếu lại "
            "bản dịch với chữ gốc của đúng dòng đó và tự hỏi: (a) độ dài có tương xứng không? "
            "(b) mọi ý trong bản dịch có xuất phát từ chữ của dòng này không, hay đã mượn của "
            "dòng lân cận? (c) số dòng trả về có ĐÚNG BẰNG số dòng nhận vào và line_no có khớp "
            "từng dòng không? Nếu phát hiện lệch, SỬA LẠI trước khi trả JSON.\n"
        )
        # [v3.23.155] Xưng hô tiếng Việt theo GIỚI TÍNH + vai vế: roster nhân vật nay
        # có [nam/nữ, ~tuổi] (xác định từ giọng nói trong audio + hình ảnh khi phân
        # tích). Bắt model dùng dữ kiện đó + speaker/addressee từng dòng để chọn đại
        # từ đúng — chấm dứt lỗi kinh điển đảo anh/em giữa cặp thoại.
        pronoun_rule = (
            "- XƯNG HÔ THEO GIỚI TÍNH & VAI VẾ (RẤT QUAN TRỌNG): danh sách nhân vật có "
            "ghi [nam/nữ, tuổi] và quan hệ; mỗi dòng có thể kèm speaker (người nói) và "
            "addressee (người nghe). LUÔN đối chiếu hai dữ kiện này để chọn đại từ "
            "anh/em/cô/chú/ông/bà/ngài... cho ĐÚNG người đúng vai: nam nói với nữ trẻ "
            "hơn thường xưng 'anh' gọi 'em', nữ nói với nam lớn hơn xưng 'em' gọi "
            "'anh'; bề trên-bề dưới, chủ-tớ, quan-dân theo đúng vai vế. GIỮ NHẤT QUÁN "
            "cách xưng hô của từng CẶP nhân vật suốt phim, KHÔNG đảo vai giữa chừng; "
            "tag người nói (nếu có) phải là NGƯỜI NÓI của dòng đó, "
            "không nhầm giới tính.\n"
            "- TRƯỜNG 'address' là quy tắc xưng hô ĐÃ TRA SẴN cho cặp người nói → người nghe "
            "(tự xưng … gọi …). Nó chỉ ghi ở dòng ĐẦU TIÊN của cặp đó trong lô, nhưng áp cho MỌI "
            "dòng cùng cặp (cùng speaker + addressee) trong lô. Mặc định DÙNG ĐÚNG "
            "đại từ trong đó. Chỉ được đổi khi lời thoại có BƯỚC NGOẶT rõ (trở mặt, cãi to, khinh "
            "miệt công khai) — và khi đã đổi thì GIỮ cách mới cho HẾT cảnh đó, không đảo qua đảo lại "
            "giữa các dòng. Quy tắc có điều kiện giai đoạn ('trước khi…', 'sau khi…') thì chọn "
            "đúng giai đoạn theo diễn biến.\n"
            "- BÁM BẢNG XƯNG HÔ ĐÃ CHỐT: nếu mục 'xưng hô:' trong Roster đã quy định cặp đại từ "
            "cho hai nhân vật, PHẢI dùng đúng cặp đó, KHÔNG tự chọn lại theo cảm nhận của riêng "
            "lô này. Chỉ đổi khi chính lời thoại thể hiện quan hệ thay đổi (làm lành, trở mặt, "
            "biết được vai vế thật) — và khi đã đổi thì giữ cách mới cho các dòng sau.\n"
            "- TRƯỜNG 'pron' (nếu có) là phép gán ĐÃ CHỐT cho chữ 我/你 của ĐÚNG dòng đó: "
            "'我=muội' nghĩa là 我 phải ra 'muội'; '你=sư huynh' nghĩa là 你 phải ra 'sư huynh'. "
            "Áp cả khi 我 đứng sau lời gọi: '师兄，我…' là người nói GỌI người nghe là sư huynh "
            "rồi tự xưng theo 'pron' ('Sư huynh, muội…'), KHÔNG biến người nói thành sư huynh.\n"
        )
        # [v3.23.621] Chấm mù 1.770 dòng (phim mẫu tiên hiệp): roster đã chốt 'ta/huynh/muội',
        # 'sư thúc' mà model vẫn trôi sang 'anh/em', 'mày/tao' ở 6/10 lượt dịch tập 2 — lỗi
        # văn phong người xem thấy ngay. Hán-Việt khó nghe ('tiểu khả ái', 'nhất ngôn vi
        # định') cũng bị chấm lặp lại vì bản dịch còn được đọc bằng TTS.
        register_rule = (
            "- VĂN PHONG THEO BỐI CẢNH: phim cổ trang/kiếm hiệp/tiên hiệp/huyền huyễn (nhận ra "
            "từ tóm tắt, Roster: tông môn, tu luyện, sư huynh, hoàng thượng…) dùng đại từ cổ "
            "phong (ta/ngươi, huynh/muội/tỷ/đệ, sư phụ/sư thúc/đồ nhi, bổn tọa…); TUYỆT ĐỐI "
            "không chuyển sang 'anh/em', 'mày/tao', 'tôi/bạn' giữa chừng. Phim hiện đại thì "
            "ngược lại: không dùng đại từ cổ phong. Ngôi 2 cổ phong viết 'ngươi' — KHÔNG "
            "nhầm với 'người' (kính ngữ, chỉ dùng khi bề dưới thưa với bề trên).\n"
            "- Hán-Việt chỉ dùng cho thuật ngữ/tên/cảnh giới quen thuộc; câu nói thường dùng "
            "từ thuần Việt nghe hiểu ngay (一言为定 → 'một lời đã định', 小可爱 → 'bé cưng', "
            "không 'nhất ngôn vi định', 'tiểu khả ái').\n"
        )
        line_align = line_align + pronoun_rule + register_rule

        # Chỉ dẫn súc tích — bản dịch sẽ qua TTS đọc theo mốc thời gian phụ đề.
        # Câu quá dài buộc TTS đọc nhanh/nén tốc độ → khó nghe, mất đồng bộ.
        concise = (
            "- SÚC TÍCH (nhưng KHÔNG được đánh đổi bằng nghĩa): bản dịch sẽ được TTS đọc "
            "khít vào thời lượng phụ đề. Câu dịch quá dài khiến TTS đọc gấp gáp. Vì vậy "
            "hãy: (1) bỏ từ đệm/thừa ('thì', 'là', 'mà', 'rằng', 'đã', 'các', 'những', "
            "'một cách', 'của việc' khi không cần); (2) ưu tiên từ "
            "ngắn. TỰ NHIÊN QUAN TRỌNG HƠN NGẮN: với THOẠI ĐỜI THƯỜNG dùng từ "
            "THUẦN VIỆT dễ hiểu (giữ 'giá trên trời', 'tốc độ tay nhanh'), TUYỆT ĐỐI không "
            "đổi thành Hán-Việt cứng/sượng chỉ để ngắn hơn (KHÔNG biến 'giá trên trời'→'thiên "
            "giá', 'tốc độ tay'→'thủ sức'). Chỉ dùng Hán-Việt khi nó VỐN tự nhiên hơn hoặc là "
            "thuật ngữ/tên riêng/văn phong trang trọng phù hợp; (3) gộp lối nói vòng thành cụm "
            "gọn. Súc tích nghĩa "
            "là BỎ CHỮ THỪA, KHÔNG PHẢI bỏ nội dung. KHÔNG bỏ từ MANG NGHĨA: "
            "giữ tính từ/danh từ bổ nghĩa (vd 'giày cao GÓT' phải giữ 'gót', không rút thành "
            "'giày cao'); giữ HỌ/TÊN riêng trong xưng hô (vd '卫老公' → 'chồng Vệ'/'anh Vệ', "
            "KHÔNG rút thành 'chồng' làm mất họ). Giữ trọn ý nghĩa, không bịa thêm, "
            "không cắt ý; ngắn gọn nhưng trôi chảy tự nhiên, KHÔNG cộc lốc.\n"
            "- KHÔNG ÉP ĐỘ DÀI THEO CÂU GỐC: tiếng gốc "
            "(Trung/Nhật/Hàn) cô đọng hơn tiếng "
            "Việt về bản chất — một câu 6 chữ Hán thường cần 9-12 âm tiết tiếng Việt mới "
            "diễn đạt đủ (do phải thêm đại từ xưng hô, hư từ, giới từ). Vì vậy TUYỆT ĐỐI "
            "KHÔNG lấy số âm tiết câu gốc làm đích: ép bằng câu gốc là "
            "điều BẤT KHẢ THI và "
            "chỉ dẫn tới CẮT NGHĨA. Hãy lấy 'max_chars' làm mốc, không lấy câu gốc.\n"
            "- DÙNG ĐỦ NGÂN SÁCH: 'max_chars' là NGÂN SÁCH được phép "
            "dùng, KHÔNG phải chỉ "
            "tiêu phải tiết kiệm. Khi bản dịch đã nằm trong ngân sách thì DỪNG rút gọn — "
            "rút thêm nữa chỉ làm mất chi tiết mà không được lợi gì. "
            "Một bản dịch chỉ dùng "
            "một nửa ngân sách thường là dấu hiệu ĐÃ BỎ MẤT nội dung: hãy đọc lại và bổ "
            "sung phần còn thiếu.\n"
            "- GIỚI HẠN ĐỘ DÀI (max_chars): mỗi dòng có trường 'max_chars' là số ký tự "
            "mà giọng đọc kịp phát âm trong thời lượng cảnh đó (tính từ tốc độ phát âm "
            "THỰC ĐO của engine TTS). Hãy ĐẾM ký tự bản dịch và cố viết gọn trong giới "
            "hạn — CHỈ bằng cách bỏ từ đệm/thừa, rút cụm vòng vo, dùng từ cô đọng hơn.\n"
            "- NHƯNG NGHĨA LUÔN THẮNG GIỚI HẠN (TUYỆT ĐỐI): 'max_chars' là MỤC TIÊU, "
            "KHÔNG phải mệnh lệnh được phép hy sinh nội dung. TUYỆT ĐỐI KHÔNG: bỏ thông "
            "tin, bỏ chủ ngữ/tân ngữ làm câu tối nghĩa, bỏ từ phủ định ('không', 'chưa', "
            "'đừng'), đảo người nói/người nghe, hay biến câu thành mảnh cụt vô nghĩa. "
            "Nếu KHÔNG THỂ vừa giới hạn mà vẫn giữ TRỌN nghĩa, hãy GIỮ NGHĨA và chấp "
            "nhận vượt 'max_chars' — phần mềm sẽ tự nén giọng, hoàn toàn ổn. Người xem "
            "thà nghe giọng hơi gấp mà HIỂU ĐÚNG, còn hơn nghe rõ mà nghĩa sai.\n"
            "- VÍ DỤ PHẢN DIỆN (những lỗi CẤM tuyệt đối, lấy từ ca thật):\n"
            "  * \"cháu đã nói về chú thế nào,\" → \"Chú?\" — SAI: mất sạch nghĩa.\n"
            "  * \"Không phải chú không cho cháu cơ hội.\" → \"Cho cơ hội,\" — SAI: mất "
            "phủ định, đảo ngược ý.\n"
            "  * \"cháu đã nói về chú thế nào,\" → \"chú nói về cháu thế nào,\" — SAI: "
            "ĐẢO NGƯỢC ai nói về ai (cháu là người nói, chú là người được nói tới).\n"
            "  * \"trong lòng nghĩ về chú ra sao,\" → \"Chú nghĩ gì?\" — SAI: đảo chủ "
            "thể (CHÁU nghĩ về chú, không phải chú nghĩ) + mất nghĩa.\n"
            "  * \"Trở thành\" → \"Thành\" — SAI: cụt lủn, vô nghĩa.\n"
            "  Cách rút gọn ĐÚNG: \"Không phải chú không cho cháu cơ hội.\" → \"Chú vẫn "
            "cho cháu cơ hội.\" (ngắn hơn, GIỮ TRỌN nghĩa).\n"
            "- GIỮ NGUYÊN HƯỚNG HÀNH ĐỘNG (ai làm gì với ai) — LỖI TÁI PHẠM NHIỀU LẦN: "
            "TUYỆT ĐỐI không hoán đổi CHỦ NGỮ với TÂN NGỮ. Trước khi chốt mỗi dòng, tự "
            "hỏi: 'Ai là người thực hiện? Ai là người nhận?' — bản dịch phải giữ ĐÚNG "
            "hướng đó như bản gốc. Đảo hướng là lỗi NẶNG hơn cả câu dài.\n"
            "  Cạm bẫy điển hình với tiếng Trung — câu KHÔNG có chủ ngữ tường minh: "
            "'内心是怎么想我的' nghĩa là 'trong lòng (NGƯỜI NGHE) nghĩ về TÔI ra sao'. "
            "Dịch thành 'Chú nghĩ gì?' là SAI HOÀN TOÀN: đã biến người nghe thành chủ "
            "thể suy nghĩ và làm biến mất TÔI (tân ngữ). Khi câu gốc lược chủ ngữ, hãy "
            "dùng 'speaker'/'addressee' và các dòng lân cận để khôi phục ĐÚNG vai — "
            "KHÔNG được đoán bừa rồi đảo ngược.\n"
            "  Tuyệt đối KHÔNG trả trường 'max_chars' trong kết quả.\n"
        )

        # [v3.23.225] Chỉ thị xử lý cờ needs_expansion — vòng TỰ SỬA giữa các giai đoạn.
        # Log thực tế: cùng 5 dòng bị cắt nghĩa trôi qua cả 3 giai đoạn mà không giai đoạn
        # nào sửa, dù GĐ tinh chỉnh/bản địa hoá đã có bản gốc trong tay.
        expansion = (
            "- BỔ SUNG DÒNG BỊ DỊCH THIẾU (needs_expansion): dòng nào có "
            "'needs_expansion': true nghĩa là hệ thống đã ĐO ĐƯỢC bản dịch hiện tại NGẮN "
            "BẤT THƯỜNG so với bản gốc (nhiều khả năng khâu trước đã làm RỚT nội dung). "
            "Với các dòng đó, nhiệm vụ số MỘT của bạn KHÔNG phải là gọt câu chữ, mà là:\n"
            "  (1) Đọc kỹ trường 'original' (bản gốc);\n"
            "  (2) Đối chiếu xem bản dịch hiện tại đã BỎ SÓT gì — thường là: tân ngữ, tên "
            "riêng, trạng ngữ nơi chốn/thời gian, từ phủ định, hoặc cả một vế câu;\n"
            "  (3) VIẾT LẠI dòng đó cho ĐỦ NỘI DUNG bản gốc, kể cả khi phải dài hơn "
            "'max_chars' — thà dài mà đúng còn hơn ngắn mà mất nghĩa;\n"
            "  (4) Kiểm tra HƯỚNG hành động: ai làm gì với ai, đúng như bản gốc.\n"
            "  Ví dụ: gốc '作为王建强的侄女' dịch thành 'Là cháu chú,' đã RỚT tên riêng -> "
            "phải bổ sung: 'Là cháu gái của Vương Kiến Cường,'.\n"
        )

        # Chỉ dẫn xưng hô — tránh đoán bừa vai vế/giới tính (lỗi kinh điển khi dịch
        # phim Trung: huynh/đệ/tỷ/muội, anh/chị/em lộn xộn giữa các câu).
        pronoun = (
            "- XƯNG HÔ (rất quan trọng): chỉ dùng đại từ chỉ rõ giới tính/vai vế (anh, "
            "chị, em, cô, chú, ông, bà, huynh, đệ, tỷ, muội…) khi BIẾT CHẮC từ Bảng "
            "nhân vật, trường 'speaker'/'addressee', hoặc bối cảnh hiển nhiên. Khi chưa "
            "chắc giới tính, dùng đại từ TRUNG TÍNH nhất quán (ta-ngươi cho cổ trang, "
            "tôi-bạn cho hiện đại) — thà trung tính còn hơn gọi sai giới. Giữ xưng hô "
            "NHẤT QUÁN cho cùng cặp nhân vật xuyên suốt.\n"
            "- THÁI ĐỘ/CẢM XÚC: nếu một dòng có trường 'cue' (bối cảnh + thái độ/cảm xúc "
            "do AI nghe-nhìn video ghi chú), HÃY dùng nó để chọn đúng GIỌNG VĂN và sắc "
            "thái (giận dữ, trìu mến, sợ hãi, mỉa mai, trang trọng, suồng sã…) và xưng "
            "hô phù hợp. Trường 'cue' chỉ là gợi ý ngữ cảnh — TUYỆT ĐỐI KHÔNG đưa nội "
            "dung 'cue' vào bản dịch.\n"
        )

        # [v3.23.125] VÍ DỤ NGUYÊN TẮC (few-shot): minh hoạ bằng cặp trước→sau giúp model
        # bám đúng phong cách hơn là chỉ mô tả. Ví dụ về phía câu ĐÍCH (tiếng Việt) nên
        # áp dụng được bất kể tiếng gốc là Trung/Hàn/Nhật.
        examples = (
            "- VÍ DỤ MINH HOẠ (cách viết câu ĐÍCH; chỉ để học PHONG CÁCH, đừng chép):\n"
            "  1) Gọn, bỏ chữ thừa: \"Ta đã sớm biết rằng ngươi nhất định sẽ đến nơi "
            "này\" → \"Ta biết ngươi sẽ đến.\"\n"
            "  2) KHÔNG chèn chú thích: \"Hắn là Tiêu Viêm (nam chính)\" → \"Hắn là "
            "Tiêu Viêm.\"\n"
            "  3) Xưng hô nhất quán: cùng hai người thì giữ MỘT cặp xưng hô xuyên suốt, "
            "không lúc \"anh-em\" lúc \"ngươi-ta\".\n"
            "  4) Tự nhiên, không sượng: \"Ngươi rốt cuộc đang làm cái gì vậy đó hả\" → "
            "\"Rốt cuộc ngươi đang làm gì?\"\n"
            "  5) Thành ngữ dịch THOÁT nghĩa, không dịch chữ vô nghĩa.\n"
        )

        # [v3.23.427] Rào chắn CHỐNG SÓT CHỮ NGUỒN — dữ liệu thật: đôi khi 1 chữ Hán lọt qua
        # (literal #132 sót '夫'; style #114 viết lại nhưng để '送饭' chưa dịch). Áp cho mọi
        # giai đoạn để không chữ Hán/Nhật/Hàn nào lọt vào bản dịch tiếng Việt.
        no_source_script = (
            "- DỊCH HẾT sang ngôn ngữ đích: TUYỆT ĐỐI không để sót chữ Hán/Nhật/Hàn (CJK) nào "
            "trong bản dịch. Khi biên tập/viết lại câu, rà đã dịch trọn MỌI chữ, không bỏ quên "
            "cụm nào ở giữa hay cuối câu (vd '送饭' phải thành 'đưa cơm', không giữ chữ Hán).\n"
            "- KHÔNG THAY NGƯỢC: nếu bản hiện có ĐÃ dịch đúng (kể cả từ "
            "Hán-Việt như 'tinh', 'phấn đồ cường'), TUYỆT ĐỐI không đổi ngược lại "
            "thành chữ Hán gốc ('tinh'→'星', 'phấn đồ cường'→'奋图强' là SAI "
            "NGHIÊM TRỌNG). Từ Hán-Việt viết bằng chữ Latinh là ĐÚNG, giữ nguyên.\n"
            "- DỊCH TRỌN TỪ GHÉP: đặc biệt chú ý chữ ĐUÔI/GHÉP hay bị "
            "bỏ sót — dịch cả cụm, KHÔNG để trailing CJK. Ví dụ SAI→ĐÚNG: 'hỗn账'→"
            "'hỗn trướng'; 'Phi Phi姐'→'chị Phi Phi'; 'thuần dương之气'→'khí "
            "thuần dương'; 'tà祟'→'tà túy/tà ma'; '欣赏'→'ngưỡng mộ; "
            "'更棘手者'→'càng nan giải/gai góc hơn'; '耍赖'→'giở trò/ăn vạ' "
            "(dịch TRỌN cụm, không để sót '棘手'/'赖').\n"
            "- THÀNH NGỮ 4 CHỮ (成语) & câu THƠ chữ Hán: dịch TRỌN sang Hán-Việt, "
            "không để sót chữ đuôi (vd '心浮气躁'→'tâm phù khí táo', '天堑'→'thiên "
            "hào', không để 'tâm浮气躁'/'thiên堑').\n"
        )

        # [v3.23.448] Rule TAG CẢM XÚC (tuỳ chọn, chỉ khi bật + có video ngữ cảnh). Rỗng khi tắt
        # → không đổi prompt. Neo vào NGỮ CẢNH VIDEO (hình ảnh + âm thanh) để tránh bịa.
        emotion_cues = (
            "- TAG CẢM XÚC (chỉ khi có video ngữ cảnh): khi HÌNH ẢNH hoặc ÂM THANH trong "
            "video cho thấy RÕ RÀNG nhân vật đang cười / thở dài / hắng giọng, hãy chèn tag "
            "tương ứng [cười] / [thở dài] / [hắng giọng] vào ĐÚNG vị trí trong câu (nơi âm "
            "thanh đó xảy ra). CHỈ dùng đúng 3 tag này; CHỈ khi THẤY/NGHE rõ trong video — "
            "TUYỆT ĐỐI không suy đoán từ nội dung lời thoại đơn thuần. Nếu bản trước đã có "
            "tag thì GIỮ NGUYÊN. Tag chỉ để giọng đọc biểu cảm, KHÔNG hiển thị cho người "
            "xem (phần mềm tự lọc khỏi phụ đề).\n"
        ) if getattr(self, "_emotion_cues", False) else ""
        # [v3.23.612] Tuỳ chọn xưng hô thuần Việt (rỗng khi tắt → prompt không đổi).
        native_address = (
            NATIVE_ADDRESS_TRANSLATION_RULE
            if getattr(self, "_native_address", False) else ""
        )
        # [v3.23.620] Quy trình xác định vai cho tiếng Trung lược chủ ngữ — lỗi còn lại
        # nhiều nhất sau khi bật thinking (Dao Dao nói '师兄我…' bị dịch thành 'ta…').
        role_procedure = (
            "- XÁC ĐỊNH VAI TRƯỚC KHI DỊCH MỖI DÒNG: (1) ai đang nói, nói với ai (dựa vào "
            "speaker/addressee, lời gọi trong câu và các dòng lân cận); (2) 我/我们 = NGƯỜI "
            "NÓI, 你/你们 = NGƯỜI NGHE, 他/她 = người thứ ba; (3) danh xưng đứng đầu câu như "
            "'师兄，我…' là người nói GỌI người nghe — người nói tự xưng theo vai của mình "
            "với người đó (sư muội nói với sư huynh: 'Sư huynh, muội…'). (4) Câu mệnh lệnh "
            "'你走吧' là người nói bảo NGƯỜI NGHE đi. Sai vai là lỗi nặng nhất.\n"
            "- Biệt danh/tên gọi thân mật (vd '瑶瑶') giữ đúng dạng gọi thân mật đã ghi "
            "trong Roster (Dao Dao), KHÔNG thay bằng họ tên đầy đủ.\n"
            "- Thán từ (呸, 哼, 哎…) dịch bằng thán từ tiếng Việt tự nhiên ('Xì!', 'Hừ!', "
            "'Ôi…'), không phiên âm Hán-Việt.\n"
        )

        if stage.kind is TranslationStageKind.PREPROCESS:
            timing = (
                "Được phép hiệu chuẩn nhẹ mốc thời gian nếu phát hiện lệch rõ ràng."
                if stage.allow_retime
                else "Giữ nguyên tuyệt đối mốc thời gian; không cần trả start_s/end_s."
            )
            return (
                "## VAI TRÒ\n"
                "Bạn là chuyên gia hiệu đính phụ đề gốc trước khi dịch.\n\n"
                "## BỐI CẢNH\n"
                f"- Ngôn ngữ gốc dự kiến: {context.source_lang or 'chưa rõ'}\n"
                f"- Tóm tắt phim: {overview}\n"
                f"- Bảng nhân vật (Roster) để chuẩn hoá danh xưng: {characters}\n\n"
                "## NHIỆM VỤ\n"
                "Hiệu đính từng dòng phụ đề — GIỮ NGUYÊN ngôn ngữ gốc, GIỮ NGUYÊN Ý NGHĨA "
                "và SẮC THÁI. Cụ thể:\n"
                "1. Sửa CHÍNH TẢ & lỗi OCR — ưu tiên hàng đầu là DẤU tiếng Việt: dấu THANH "
                "(huyền/sắc/hỏi/ngã/nặng) và dấu MŨ/MÓC (â/ê/ô/ơ/ư) bị sai hoặc thiếu "
                "(vd 'nước mặt'→'nước mắt', 'điều tôt'→'điều tốt', 'căn bênh'→'căn bệnh', "
                "'hanh'→'hạnh'); cùng lỗi nhầm phụ âm/nguyên âm OCR ('phiên toàn'→'phiên "
                "tòa', 'tự tư'→'tự tử', 'lên'→'nên').\n"
                "2. Sửa DẤU CÂU: thêm dấu kết câu còn thiếu; TÁCH 2 câu bị DÍNH bằng dấu "
                "câu đúng (vd 'Cẩn thận Sắp được rồi!'→'Cẩn thận! Sắp được rồi!'); cân đối "
                "NGOẶC ĐƠN và NGOẶC KÉP (mở phải có đóng); bỏ khoảng trắng thừa, khoảng "
                "trắng trước dấu câu, và ký tự OCR lạ.\n"
                "3. Chuẩn hoá VIẾT HOA đầu câu, tên riêng, địa danh. NHƯNG GIỮ NGUYÊN các "
                "dòng CỐ Ý VIẾT HOA TOÀN BỘ — biển hiệu, tiêu đề, chữ trên màn hình (thường "
                "trong ngoặc, vd '(ĐỊA NGỤC SÁT SINH)', '(RỪNG GƯƠM)'); và TUYỆT ĐỐI KHÔNG "
                "tạo chữ hoa GIỮA TỪ (không biến 'phạt' thành 'pHạt').\n"
                "4. Chuẩn hoá DANH XƯNG/TÊN RIÊNG cho NHẤT QUÁN theo ĐÚNG cách viết trong "
                "Bảng nhân vật ở trên (vd luôn 'Ja-hong', không lúc 'Ja-hong' lúc 'Ja-Hong').\n"
                "5. GIỮ NGUYÊN VẸN thẻ định dạng inline ('<i>', '<b>', thẻ ASS '{...}', "
                "'\\N') — không dịch, không xoá, không đổi vị trí thẻ.\n"
                "6. Hội thoại 2 người trên cùng dòng: giữ mỗi lượt thoại có gạch đầu dòng "
                "'- ' và nằm trên MỘT DÒNG RIÊNG (dùng ký tự xuống dòng).\n\n"
                "## RÀNG BUỘC BẤT BIẾN (vi phạm là hỏng bản gốc)\n"
                f"- {timing}\n"
                "- Số dòng trả về = số dòng đầu vào; giữ nguyên line_no; KHÔNG gộp/tách "
                "các dòng phụ đề (mỗi line_no vào–ra độc lập).\n"
                "- GIỮ NGUYÊN cấu trúc xuống dòng BÊN TRONG mỗi dòng: dùng ký tự xuống "
                "dòng THẬT, TUYỆT ĐỐI KHÔNG xuất chuỗi '\\n' hay '\\N' dạng chữ, và KHÔNG "
                "tự ý gộp 2 dòng hiển thị thành 1 nếu bản gốc đang xuống dòng.\n"
                "- Dòng KHÔNG PHẢI lời thoại (credit/nhóm dịch, chú thích trên màn hình): "
                "giữ NGUYÊN VĂN, không dịch, không 'sửa' như lời thoại.\n"
                "- KHÔNG kiểm duyệt, KHÔNG làm nhẹ, KHÔNG thay từ đồng nghĩa: giữ đúng "
                "văn phong và sắc thái gốc kể cả từ mạnh/thô.\n"
                "- KHÔNG thêm nội dung mới, KHÔNG chú thích, KHÔNG đổi số liệu/tên riêng "
                "trừ khi CHẮC CHẮN là lỗi OCR.\n"
                "- Nếu một dòng đã đúng, TRẢ LẠI Y NGUYÊN (không viết lại cho 'mượt').\n"
                # [v3.23.502] PREPROCESS trước đây THIẾU hẳn quy tắc căn dòng, dù đây là khâu
                # chuẩn hoá dòng NGUỒN — lệch ở đây kéo lệch toàn bộ các giai đoạn sau.
                f"{line_align}\n"
                "## ĐỊNH DẠNG\nChỉ xuất JSON hợp lệ cho lô hiện tại."
            )

        if stage.kind is TranslationStageKind.LITERAL:
            # [v3.23.620] Luôn yêu cầu điền speaker: đo trên phim mẫu, bảo model "để trống
            # speaker" khi tắt tag làm nó bỏ luôn bước xác định ai nói với ai → đảo xưng hô
            # (你→'anh', 师兄→'sư muội'). Tắt tag chỉ ẩn nhãn khỏi phụ đề hiển thị.
            speaker_clause = (
                "điền trường speaker = tên người nói của dòng (theo Roster) nếu xác định được"
                if context.enable_tags
                else "điền trường speaker = tên người nói của dòng (theo Roster) nếu xác định "
                "được — chỉ để phần mềm và chính bạn suy luận xưng hô, KHÔNG hiển thị cho "
                "người xem"
            )
            desc_clause = (
                "trường description CHỈ ghi tiếng động/âm thanh không lời mà dòng gốc ghi "
                "rõ (vd '(敲门声)' → 'tiếng gõ cửa'); TUYỆT ĐỐI không ghi cảm xúc, giọng điệu, "
                "hành động, suy nghĩ hay 'nội tâm' — mặc định để trống"
                if context.include_desc
                else "để trống description (người dùng KHÔNG cần mô tả tiếng động)"
            )
            return (
                "## VAI TRÒ\n"
                "Bạn là biên dịch viên phụ đề chuyên dịch thô sát nghĩa.\n\n"
                "## BỐI CẢNH\n"
                f"- Ngôn ngữ đích: {context.target_lang}\n"
                f"- Tóm tắt phim: {overview}\n"
                f"- Bảng nhân vật (Roster): {characters}{glossary_instr}{names_instr}\n\n"
                "## NHIỆM VỤ\n"
                f"Dịch sát nghĩa từng câu sang ngôn ngữ đích; {speaker_clause}; {desc_clause}.\n\n"
                f"## QUY TẮC\n{role_procedure}{concise}{pronoun}{native_address}"
                f"{no_source_script}{emotion_cues}{examples}"
                f"{video_rule}{line_align}"
                "- Chỉ dựa vào nội dung được cung cấp (text + bối cảnh trên), không thêm "
                "thông tin ngoài.\n"
                "- Nếu một câu không thể dịch (chỉ là tên riêng, tiếng động, ký hiệu), hãy "
                "giữ nguyên hoặc phiên âm tự nhiên; TUYỆT ĐỐI KHÔNG chèn chú thích kiểu "
                "'[không dịch được]', '(vô nghĩa)' hay bất kỳ giải thích nào vào bản dịch.\n\n"
                "## ĐỊNH DẠNG\nChỉ xuất JSON hợp lệ cho lô hiện tại."
            )

        if stage.kind is TranslationStageKind.STYLE:
            desc_clause_style = (
                "" if context.include_desc else "- Để trống description.\n"
            )
            task = (
                "Từ bản dịch thô, biên tập lại câu chữ cho trôi chảy thoát ý hợp thể loại phim mà không sai nghĩa gốc."
                if has_prior
                else "Dịch trực tiếp từ bản gốc sang ngôn ngữ đích, thể hiện trôi chảy thoát ý đúng phong cách/thể loại phim."
            )
            return (
                "## VAI TRÒ\n"
                "Bạn là biên kịch phụ đề chuyên tinh chỉnh văn phong trôi chảy thoát ý theo thể loại phim.\n\n"
                "## BỐI CẢNH\n"
                f"- Ngôn ngữ đích: {context.target_lang}\n"
                f"- Thể loại/phong cách mục tiêu: {stage.style_name}\n"
                f"- Tóm tắt phim: {overview}\n"
                f"- Bảng nhân vật (Roster): {characters}{glossary_instr}{names_instr}\n\n"
                f"## NHIỆM VỤ\n{task}\n\n"
                f"## QUY TẮC\n{desc_clause_style}{PERSON_ANCHOR_RULE_REFINE}{expansion}"
                f"{role_procedure}{concise}{pronoun}"
                f"{native_address}{no_source_script}{emotion_cues}{examples}"
                f"{video_rule}{line_align}"
                "- Chỉ dựa vào nội dung được cung cấp, không thêm thông tin ngoài.\n\n"
                "## ĐỊNH DẠNG\nChỉ xuất JSON hợp lệ cho lô hiện tại."
            )

        locale_notes = stage.locale_notes.strip() or f"Theo chuẩn sử dụng tự nhiên trôi chảy thoát ý của {context.target_lang}"
        desc_clause_loc = (
            "" if context.include_desc else "- Để trống description.\n"
        )
        task = (
            "Bản địa hoá bản dịch HIỆN CÓ cho tự nhiên, trôi chảy, thoát ý với người xem bản "
            "địa — CHỈ chỉnh những chỗ THỰC SỰ cần (đơn vị đo, xưng hô, cách gọi tên, thành "
            "ngữ, ca dao, tục ngữ, quy ước văn hoá).\n"
            "- GIỮ NGUYÊN câu đã tự nhiên và đúng nghĩa; KHÔNG viết lại chỉ để đổi từ đồng nghĩa "
            "hay đảo trật tự khi không cần.\n"
            "- TUYỆT ĐỐI không đổi số liệu, tỉ lệ và THỨ TỰ đơn vị (vd 'gạo:nước' giữ đúng "
            "chiều, không đảo thành 'nước:gạo'), tên riêng, sự kiện lịch sử, thuật ngữ ẩm thực.\n"
            "- KHÔNG làm kém đi hay lùi lại các cải thiện tự nhiên đã có ở bước trước; nếu bản "
            "hiện có đã tốt thì để nguyên."
            if has_prior
            else "Dịch trực tiếp từ bản gốc và bản địa hoá trôi chảy thoát ý ngay sang ngôn ngữ đích."
        )
        return (
            "## VAI TRÒ\n"
            "Bạn là chuyên gia bản địa hoá trôi chảy thoát ý phụ đề.\n\n"
            "## BỐI CẢNH\n"
            f"- Ngôn ngữ đích: {context.target_lang}\n"
            f"- Yêu cầu bản địa hoá: {locale_notes}\n"
            f"- Tóm tắt phim: {overview}\n"
            f"- Bảng nhân vật (Roster): {characters}{glossary_instr}{names_instr}\n\n"
            f"## NHIỆM VỤ\n{task}\n\n"
            f"## QUY TẮC\n{desc_clause_loc}{PERSON_ANCHOR_RULE_REFINE}{expansion}"
            f"{role_procedure}{concise}{pronoun}"
            f"{native_address}{no_source_script}{emotion_cues}{examples}"
            f"{video_rule}{line_align}"
            "- Chỉ dựa vào nội dung được cung cấp, không thêm thông tin ngoài.\n\n"
            "## ĐỊNH DẠNG\nChỉ xuất JSON hợp lệ cho lô hiện tại."
        )

    def _translate_single_batch(
        self,
        *,
        batch: list[TranslationLine],
        source_before: list[dict[str, Any]],
        source_after: list[dict[str, Any]],
        history_before: list[dict[str, Any]],
        start_idx: int,
        is_preprocess: bool,
        is_literal: bool = False,
        config: Any,           # [perf P1] config xây dựng một lần ở translate_stage
        model_name: str,
        cancel_cb: CancellationCallback | None,
        _depth: int = 0,
        _ctx_size: int = 20,   # giới hạn history window, truyền từ translate_stage
        video_files: list[Any] | None = None,
        video_refs: list[Any] | None = None,
        dual_payload: bool = False,
        expansion_indices: frozenset[int] = frozenset(),
        video_token_estimate: int = 0,
    ) -> list[TranslationLine]:
        """Dịch một batch, TỰ ĐỘNG CHIA ĐÔI đệ quy khi AI trả sai số dòng.

        Khi ``_call_gemini`` báo ``_BatchCountMismatchError`` (model liên tục trả
        sai số dòng — thường do batch quá lớn vượt giới hạn context của model):

        * Chia batch làm đôi → gọi đệ quy cho từng nửa.
        * Nửa đầu hoàn tất → output làm ``history_before`` cho nửa sau.
        * Đệ quy tới ``batch_size=1``. Ở kích thước 1, không thể chia tiếp → dùng
          fallback: giữ nguyên text gốc, log warning (dịch tiếp không lỗi toàn bộ).

        Args:
            _depth: Độ sâu đệ quy (giới hạn an toàn ≤ 12, tương đương batch ≤ 4096).
        """
        _MAX_DEPTH = 12
        if not batch:
            return []

        # [v3.23.222] Truyền mốc dòng KẾ TIẾP để ngân sách ``max_chars`` tính theo KHUNG
        # HIỆU DỤNG (khung gốc + gap mà TTS thật sự được dùng) — trước đây chỉ dùng khung
        # gốc nên ép model cắt nghĩa ở cả những dòng có khoảng lặng rộng phía sau.
        current_payload = [
            (
                self._line_to_dual_payload(
                    line, next_start_ms, line.index in expansion_indices
                )
                if dual_payload
                else self._line_to_payload(line, next_start_ms)
            )
            for line, next_start_ms in zip(
                batch,
                [ln.start_ms for ln in batch[1:]] + [None],
                strict=True,
            )
        ]
        self._attach_address_rules_once(current_payload, batch)
        # [v3.23.26] Theo best practice Gemini: với context lớn, đặt DỮ LIỆU TRƯỚC và
        # INSTRUCTION Ở CUỐI (neo suy luận vào dữ liệu phía trên). Trước đây instruction
        # đặt đầu → model dễ "quên" khi data dài. Câu lệnh cuối bắt đầu bằng "Dựa trên
        # các khối trên" để buộc model bám sát dữ liệu vừa đọc.
        instruction_text = (
            # [v3.23.503] Nói rõ nhiệm vụ ĐỐI CHIẾU: trường 'original' là phụ đề GỐC THÔ (trước
            # mọi khâu xử lý). Nhờ vậy mỗi giai đoạn tự phát hiện và sửa lệch do khâu TRƯỚC gây
            # ra, thay vì tin tưởng mù quáng vào đầu vào mình nhận được.
            "Dựa trên các khối dữ liệu trên: trường 'original' là phụ đề GỐC THÔ của ĐÚNG "
            "dòng đó (chưa qua xử lý), trường 'text' là bản do khâu TRƯỚC tạo ra. "
            "BẮT BUỘC đối chiếu 'text' với 'original' của CÙNG line_no: nếu 'text' đang mang "
            "nội dung của dòng khác, bị rút ngắn đánh rơi ý, hoặc sai nghĩa so với 'original', "
            "hãy SỬA LẠI theo 'original' — đừng kế thừa lỗi của khâu trước. Giữ đúng nghĩa gốc. "
            "Chỉ xử lý <current_batch>, trả JSON hợp lệ theo schema."
            if dual_payload
            else "Dựa trên các khối dữ liệu trên: chỉ xử lý <current_batch> và trả JSON "
            "hợp lệ theo schema."
        )
        prompt_blocks: list[str] = []
        if source_before:
            prompt_blocks.append(_wrap_xml_block("previous_context", source_before))
        if history_before:
            prompt_blocks.append(_wrap_xml_block("translated_history", history_before))
        prompt_blocks.append(_wrap_xml_block("current_batch", current_payload))
        if source_after:
            prompt_blocks.append(_wrap_xml_block("next_context", source_after))
        # [v3.23.500] THUẬT NGỮ THEO LÔ: cả bảng glossary (hàng trăm mục) nằm trong
        # system_instruction thì model gần như bỏ qua — đo trên bản dịch thật: 72/113 thuật ngữ
        # bị dịch khác quy định, 173 lượt, có ca mất hẳn nội dung ("奔驰" Mercedes → "tài xế
        # riêng"). Nay LỌC đúng các thuật ngữ XUẤT HIỆN trong lô này và đặt ngay cạnh dòng cần
        # dịch, nơi model chú ý nhất.
        batch_glossary = _batch_glossary_block(batch, getattr(self, "_active_glossary", ""))
        if batch_glossary:
            prompt_blocks.append(_wrap_xml_block("must_use_terms", batch_glossary))
        prompt_blocks.append(_wrap_xml_block("instruction", instruction_text))
        prompt = "\n\n".join(prompt_blocks)
        # config được truyền vào từ translate_stage (xây dựng một lần)

        # [v3.23.521] Đếm số lần ÉP DỊCH LẠI vì lệch dòng. Đo thật: có lô bị bắt 12 lần liên
        # tiếp — mỗi lần là một lượt gọi API mà kết quả vẫn lệch, tốn quota vô ích và cuối cùng
        # còn nguy cơ bỏ cuộc giữ nguyên chữ Hán (tệ hơn bản dịch lệch). Sau ngưỡng này thì NHẬN
        # kết quả và để cờ chuẩn đoán ``shifted_lines`` báo lại cho người dùng rà tay.
        alignment_retries = 0
        # [v3.23.579] Cùng trần cho lần ép dịch lại vì model trả NGUYÊN VĂN câu gốc.
        echo_retries = 0

        def _validator(data: dict[str, Any]) -> None:
            try:
                self._validate_batch(data, len(batch), start_idx)
            except _BatchValidationError as exc:
                # [v3.23.520] Kèm NỘI DUNG lô lỗi để đối chiếu sau này (câu gốc + bản model trả về).
                _record_process_event(
                    "batch_invalid",
                    batch_from=batch[0].index if batch else 0,
                    batch_to=batch[-1].index if batch else 0,
                    reason=str(exc)[:160],
                    content=_batch_content_sample(batch, data),
                )
                raise
            # [v3.23.518] Kiểm CĂN DÒNG phải nằm TRONG bộ kiểm này, vì chỉ ở đây lỗi
            # ``_BatchValidationError`` mới được vòng thử-lại/chia-nhỏ của ``_call_gemini`` bắt.
            # v3.23.516 đặt nhầm phép kiểm ở NGOÀI (sau khi hàm đã trả về) nên lỗi thoát thẳng ra
            # và làm SẬP cả tác vụ dịch.
            if not is_preprocess:
                nonlocal alignment_retries
                misaligned = (
                    self._detect_batch_misalignment_payload(batch, data)
                    if alignment_retries < MAX_ALIGNMENT_RETRIES
                    else []
                )
                if misaligned:
                    alignment_retries += 1
                    _record_process_event(
                        "batch_misaligned",
                        lines=[item["index"] for item in misaligned[:5]],
                        shift=misaligned[0]["shift"],
                        batch_from=batch[0].index if batch else 0,
                        batch_to=batch[-1].index if batch else 0,
                        content=_batch_content_sample(
                            batch, data, _misalignment_focus_lines(misaligned)
                        ),
                    )
                    raise _BatchValidationError(_misalignment_message(misaligned))
            # [v3.23.579] Dịch thô trả nguyên văn chữ Hán ở nhiều dòng (ca thật: 55/60
            # dòng) mà vẫn đủ số dòng — bắt ở đây để vòng thử-lại dịch lại lô ngay.
            if is_literal and not is_preprocess:
                nonlocal echo_retries
                echoed = (
                    self._source_echo_in_payload(batch, data)
                    if echo_retries < MAX_ALIGNMENT_RETRIES
                    else []
                )
                if echoed:
                    echo_retries += 1
                    _record_process_event(
                        "batch_source_echo",
                        lines=[echoed[0], echoed[-1]],
                        count=len(echoed),
                        batch_from=batch[0].index if batch else 0,
                        batch_to=batch[-1].index if batch else 0,
                    )
                    raise _BatchValidationError(
                        f"Model trả NGUYÊN VĂN câu gốc (chưa dịch) ở {len(echoed)}/"
                        f"{len(batch)} dòng ({echoed[0]}-{echoed[-1]}) — cần dịch lại."
                    )

        # [v3.23.19] Ước lượng token gồm cả VIDEO để quota manager điều tiết ĐÚNG.
        # Trước đây chỉ tính text → quota "mù" với video → acquire cho qua → server
        # báo 429 input_token vượt 250k. Nay cộng token video đính kèm.
        est_total = max(1, len(prompt) // 4) + max(0, video_token_estimate)

        try:
            data = self._call_gemini(
                model_name, prompt, config, _validator, cancel_cb=cancel_cb,
                video_files=video_files, video_refs=video_refs, est_tokens=est_total,
            )
            ai_items = data.get("subtitles", []) if data else []
            batch_result = self._merge_ai_items(batch, ai_items, is_preprocess)

            # [v3.23.16] Nếu model thiếu ÍT dòng → VÁ theo CỬA SỔ LIỀN KỀ (kèm ngữ
            # cảnh xung quanh) thay vì vá từng dòng cô lập. Vá dòng cô lập khiến câu đó
            # MẤT ngữ cảnh → dịch sai/lặp, và các câu kề (đã lệch do câu thiếu) không
            # được sửa. Mở rộng mỗi dòng thiếu thêm _PATCH_PADDING câu mỗi bên (lấy từ
            # chính batch), gộp cửa sổ chồng nhau, dịch lại cả vùng với ngữ cảnh đúng.
            missing_line_nos = data.get("_missing_line_nos") if data else None
            if missing_line_nos:
                missing_set = set(missing_line_nos)
                batch_result = self._patch_missing_with_context(
                    batch=batch, batch_result=batch_result, missing_set=missing_set,
                    source_before=source_before, source_after=source_after,
                    history_before=history_before, is_preprocess=is_preprocess,
                    is_literal=is_literal, config=config, model_name=model_name,
                    cancel_cb=cancel_cb, _depth=_depth, _ctx_size=_ctx_size,
                    video_files=video_files, video_refs=video_refs, dual_payload=dual_payload,
                    video_token_estimate=video_token_estimate,
                    expansion_indices=expansion_indices,
                )

            # Hậu xác thực nội dung: phát hiện "dịch không có tác dụng".
            # [Halving Infinite Loop Fix] So sánh Output của AI với BẢN GỐC
            # (``original_text`` — tiếng Trung/Anh bất biến), KHÔNG so với ``orig.text``
            # (input của giai đoạn hiện tại). Và CHỈ khâu LITERAL (dịch thô sát nghĩa)
            # mới bắt buộc đổi ngôn ngữ → mới kích hoạt halving khi giống bản gốc.
            # Khâu STYLE/LOCALIZE giữ nguyên bản nháp tốt là HỢP LỆ → chỉ log INFO,
            # KHÔNG ném exception (tránh chia đôi lô vô hạn, spam API → lỗi 429).
            if not is_preprocess and len(batch_result) == len(batch) and len(batch) >= 3:
                noop_count = 0
                for orig, out in zip(batch, batch_result):
                    reference = (orig.original_text or orig.text).strip()
                    if reference and reference == out.text.strip():
                        noop_count += 1
                if noop_count == len(batch):
                    if is_literal:
                        logger.warning(
                            "Batch %d dòng (LITERAL): 100%% giống BẢN GỐC (không dịch "
                            "được) → kích hoạt halving để thử lô nhỏ hơn.",
                            len(batch),
                        )
                        raise _BatchCountMismatchError(
                            f"Dịch không có tác dụng: toàn bộ {len(batch)} dòng giống hệt bản gốc."
                        )
                    logger.info(
                        "Batch %d dòng (%s): AI giữ nguyên 100%% — hợp lệ ở khâu tinh "
                        "chỉnh/bản địa hoá, bỏ qua halving.",
                        len(batch), "STYLE/LOCALIZE",
                    )

            return batch_result

        except _BatchCountMismatchError as exc:
            # ── Kích hoạt halving đệ quy ────────────────────────────────────
            if len(batch) == 1 or _depth >= _MAX_DEPTH:
                # Không thể chia tiếp → giữ text gốc, ghi log, tiếp tục
                _record_process_event(
                    "batch_gave_up",
                    batch_from=batch[0].index if batch else 0,
                    batch_to=batch[-1].index if batch else 0,
                    depth=_depth,
                )
                logger.warning(
                    "Batch %d dòng (depth=%d) vẫn thất bại sau %d retry "
                    "và không thể chia tiếp → giữ text gốc: %s",
                    len(batch), _depth, self._retry_count, exc,
                )
                return list(batch)  # fallback: giữ nguyên text gốc

            mid = len(batch) // 2
            first_half, second_half = batch[:mid], batch[mid:]
            logger.info(
                "Chia đôi batch %d→%d+%d (depth=%d) do model trả sai số dòng.",
                len(batch), len(first_half), len(second_half), _depth,
            )

            # Nửa đầu: source_after bổ sung source của nửa sau
            extra_src = [self._line_to_payload(l) for l in second_half]
            first_result = self._translate_single_batch(
                batch=first_half, source_before=source_before,
                source_after=extra_src + source_after,
                history_before=history_before, start_idx=start_idx,
                is_preprocess=is_preprocess, is_literal=is_literal, config=config,
                model_name=model_name,
                cancel_cb=cancel_cb, _depth=_depth + 1, _ctx_size=_ctx_size,
                # [v3.23.152] Mỗi nửa vẫn đính TOÀN BỘ video_files -> phải truyền tiếp
                # ước lượng token video, nếu không quota manager "mù" video (đúng bug
                # v3.23.19 tái xuất trong đường đệ quy) -> acquire cho qua -> 429 TPM.
                video_files=video_files, video_refs=video_refs, dual_payload=dual_payload,
                video_token_estimate=video_token_estimate,
                # [v3.23.576] Cờ needs_expansion phải theo xuống nửa lô / cửa sổ vá —
                # trước đây mất khi lô bị chia đôi, dòng dịch thiếu không được nhắc.
                expansion_indices=expansion_indices,
            )

            # Nửa sau: history_before + output nửa đầu, có CAP theo ctx_size
            extra_hist = [self._line_to_payload(l) for l in first_result]
            combined_hist = (history_before + extra_hist)[-_ctx_size:] if _ctx_size > 0 else []
            extra_src_prev = [self._line_to_payload(l) for l in first_half]
            second_result = self._translate_single_batch(
                batch=second_half,
                source_before=source_before + extra_src_prev,
                source_after=source_after,
                history_before=combined_hist,
                start_idx=start_idx + mid,
                is_preprocess=is_preprocess, is_literal=is_literal, config=config,
                model_name=model_name,
                cancel_cb=cancel_cb, _depth=_depth + 1, _ctx_size=_ctx_size,
                video_files=video_files, video_refs=video_refs, dual_payload=dual_payload,
                video_token_estimate=video_token_estimate,
                # [v3.23.576] Cờ needs_expansion phải theo xuống nửa lô / cửa sổ vá —
                # trước đây mất khi lô bị chia đôi, dòng dịch thiếu không được nhắc.
                expansion_indices=expansion_indices,
            )

            return first_result + second_result

    # ── Vá dòng thiếu kèm ngữ cảnh (cửa sổ liền kề) ───────────────────────
    _PATCH_PADDING = 6  # số câu lân cận mỗi bên đưa vào vùng vá (đủ ngữ cảnh)

    def _patch_missing_with_context(
        self,
        *,
        batch: list[TranslationLine],
        batch_result: list[TranslationLine],
        missing_set: set[int],
        source_before: list[dict[str, Any]],
        source_after: list[dict[str, Any]],
        history_before: list[dict[str, Any]],
        is_preprocess: bool,
        is_literal: bool,
        config: Any,
        model_name: str,
        cancel_cb: CancellationCallback | None,
        _depth: int,
        _ctx_size: int,
        video_files: list[Any] | None,
        video_refs: list[Any] | None = None,
        dual_payload: bool,
        video_token_estimate: int = 0,
        expansion_indices: frozenset[int] = frozenset(),
    ) -> list[TranslationLine]:
        """Vá các dòng thiếu theo CỬA SỔ LIỀN KỀ (kèm ngữ cảnh), gộp lại batch_result.

        Với mỗi cửa sổ [start, end) trong batch (đã gộp quanh các dòng thiếu), dịch lại
        cả vùng với ngữ cảnh ĐÚNG: các câu ngay trước cửa sổ (trong batch) làm
        ``source_before``, các câu ngay sau làm ``source_after`` — nối thêm ngữ cảnh
        gốc của batch ở hai biên. Nhờ vậy câu thiếu có đủ ngữ cảnh và các câu kề bị
        lệch/lặp cũng được dịch lại đồng thời.
        """
        batch_indices = [ln.index for ln in batch]
        windows = _compute_patch_windows(batch_indices, missing_set, self._PATCH_PADDING)
        if not windows:
            return batch_result
        # Cả batch nằm trong một cửa sổ → vá riêng vô nghĩa, để halving xử lý.
        if len(windows) == 1 and windows[0] == (0, len(batch)):
            return batch_result

        result_by_index = {r.index: r for r in batch_result}
        for start, end in windows:
            if cancel_cb is not None and cancel_cb():
                break
            window_batch = batch[start:end]
            if not window_batch or len(window_batch) >= len(batch):
                continue
            # Ngữ cảnh TRƯỚC cửa sổ: câu liền trước trong batch (+ ngữ cảnh batch gốc
            # nếu cửa sổ chạm đầu batch). Tương tự cho ngữ cảnh SAU.
            before_in_batch = [self._line_to_payload(l) for l in batch[:start]]
            after_in_batch = [self._line_to_payload(l) for l in batch[end:]]
            win_source_before = (source_before + before_in_batch)
            if _ctx_size > 0:
                win_source_before = win_source_before[-_ctx_size:]
            win_source_after = (after_in_batch + source_after)
            if _ctx_size > 0:
                win_source_after = win_source_after[:_ctx_size]
            missing_here = sorted(
                ln.index for ln in window_batch if ln.index in missing_set
            )
            logger.info(
                "Vá %d dòng thiếu (line_no %s) bằng cửa sổ %d câu (kèm ngữ cảnh) "
                "thay vì vá cô lập.",
                len(missing_here), missing_here[:5], len(window_batch),
            )
            patched = self._translate_single_batch(
                batch=window_batch,
                source_before=win_source_before,
                source_after=win_source_after,
                history_before=history_before,
                start_idx=window_batch[0].index - 1,
                is_preprocess=is_preprocess, is_literal=is_literal,
                config=config, model_name=model_name, cancel_cb=cancel_cb,
                _depth=_depth + 1, _ctx_size=_ctx_size,
                video_files=video_files, video_refs=video_refs, dual_payload=dual_payload,
                video_token_estimate=video_token_estimate,
                # [v3.23.576] Cờ needs_expansion phải theo xuống nửa lô / cửa sổ vá —
                # trước đây mất khi lô bị chia đôi, dòng dịch thiếu không được nhắc.
                expansion_indices=expansion_indices,
            )
            # [v3.23.131] CHỈ ghi đè đúng các dòng THIẾU. Trước đây ghi đè CẢ dòng ngữ
            # cảnh trong cửa sổ → nếu bản vá bị lệch (model tách/gộp câu khác đi) thì làm
            # HỎNG luôn những dòng vốn đã dịch đúng. Giới hạn ở dòng thiếu để bản vá tệ
            # nhất cũng chỉ ảnh hưởng chính dòng đang trống, không lan sang dòng kề.
            for p in patched:
                if p.index in missing_set:
                    result_by_index[p.index] = p
        return [result_by_index[ln.index] for ln in batch]

    # ── Phân tích ngữ cảnh toàn cục ────────────────────────────────────
    def analyze_global_context(
        self,
        source_lines: list[TranslationLine],
        target_lang: str,
        model_name: str = "gemini-3.1-flash-lite",
        cancel_cb: CancellationCallback | None = None,
        video_refs: list[Any] | None = None,
        with_visual_cues: bool = False,
        prior_context: str = "",
    ) -> SubtitleContextAnalysis:
        """Phân tích TOÀN BỘ phụ đề, trả về ngôn ngữ + nhân vật đầy đủ + tóm tắt.

        Khác với cách lấy mẫu cũ, hàm này gửi **toàn bộ** nội dung phụ đề
        (sau khi khử trùng lặp liên tiếp) để AI có đủ ngữ cảnh nhận diện:

        * Ngôn ngữ gốc chính xác ngay cả khi có nhiều ngôn ngữ trộn lẫn.
        * Nhân vật phụ / nhân vật xuất hiện ít trong phần đầu nhưng quan trọng.
        * Cốt truyện đầy đủ, kể cả cung bậc và twist ở cuối tập.

        Context window của Gemini 2.0+ (1 M token) đủ chứa phụ đề phim dài
        (~5000 dòng ≈ 30 000 token). Nếu file quá lớn (> ``_MAX_LINES_HARD_CAP``),
        sẽ bảo toàn phần đầu, giữa và cuối theo tỷ lệ cân bằng.
        """
        self._ensure_available()
        if not source_lines:
            return SubtitleContextAnalysis()

        # ── Khử trùng lặp liên tiếp và áp dụng soft-cap ────────────────────
        # Các phần mềm OCR thường tạo ra nhiều frame trùng nhau cho cùng 1 cụm
        # phụ đề. Khử trùng để tiết kiệm token mà không mất nội dung.
        _MAX_LINES_HARD_CAP = 8_000   # xấp xỉ 50 000 token — vẫn trong context window

        # [v3.23.603] Cần nhãn người nói cho TỪNG dòng thì không bỏ dòng lặp — xem
        # :func:`analysis_source_lines`.
        deduped = analysis_source_lines(source_lines, keep_repeats=with_visual_cues)

        total_unique = len(deduped)
        if total_unique > _MAX_LINES_HARD_CAP:
            # Giữ lại đầu/giữa/cuối để bao phủ cả arc tự sự
            n = _MAX_LINES_HARD_CAP
            step = total_unique / n
            deduped = [deduped[int(i * step)] for i in range(n)]
            logger.info(
                "analyze_global_context: file lớn (%d dòng duy nhất) → "
                "giữ %d dòng phân bố đều để tránh vượt context window.",
                total_unique, n,
            )

        logger.info(
            "analyze_global_context: %d dòng gốc → %d dòng sau khử trùng → gửi AI%s.",
            len(source_lines), len(deduped),
            " (giữ dòng lặp liền nhau: Visual Cues cần người nói cho từng dòng)"
            if with_visual_cues else "",
        )

        # ── Định dạng nội dung ────────────────────────────────────────────
        # Chỉ giữ index + text, không cần timing (tiết kiệm ~30% token)
        # [v3.23.622] NGOẠI LỆ: phân tích kèm Visual Cues với ĐÚNG MỘT đoạn video (mọi tập
        # phim ngắn ≤ 18 phút) phải có mốc MM:SS như bộ quét/đường nhiều đoạn đã có. Đo thật
        # (log gemini-3.1-flash-lite): lượt đầu chỉ điền người nói cho 6/95 và 5/143 dòng
        # (89/95, 138/143 để trống) rồi tốn thêm một lời gọi mang lại cả video để vá;
        # tài liệu video-understanding: "use the MM:SS format" để trỏ đúng khung hình.
        timed_cues = bool(with_visual_cues and video_refs and len(video_refs) == 1)
        if timed_cues:
            from subtitles_extractor.application.services.visual_cue_batching import (
                clip_timestamp,
            )

            clip_start = float(getattr(video_refs[0], "start_sec", 0.0) or 0.0)
            content_block = "\n".join(
                f"{ln.index}. [{clip_timestamp(ln.start_ms, clip_start)}] {ln.text}"
                for ln in deduped
            )
        else:
            content_block = "\n".join(f"{ln.index}. {ln.text}" for ln in deduped)

        # ── System instruction ────────────────────────────────────────────
        system_instr = (
            "Bạn là chuyên gia phân tích nội dung phim và phụ đề đa ngôn ngữ. "
            "Nhiệm vụ: phân tích TOÀN BỘ nội dung phụ đề và ngữ cảnh video để nghe nhìn (nếu có) dưới đây và trả JSON theo schema. "
        )
        # [v3.23.92] PHÂN TÍCH TUẦN TỰ TÍCH LUỸ: nếu đây là tập sau của phim bộ, nạp
        # ngữ cảnh đã thiết lập (nhân vật + thuật ngữ tích luỹ từ các tập TRƯỚC) để model
        # TÁI SỬ DỤNG đúng tên/cách dịch, chỉ THÊM cái mới -> tên nhất quán xuyên tập.
        if prior_context.strip():
            system_instr += (
                "\n\n=== NGỮ CẢNH PHIM BỘ ĐÃ THIẾT LẬP (từ các tập TRƯỚC) ===\n"
                "Đây là MỘT TẬP trong phim bộ. Danh sách nhân vật và thuật ngữ dưới đây "
                "ĐÃ được chốt ở các tập trước. BẮT BUỘC:\n"
                "- Với nhân vật/thuật ngữ đã có: TÁI SỬ DỤNG CHÍNH XÁC tên & cách dịch "
                "đã thiết lập (KHÔNG đổi tên, KHÔNG đặt tên mới cho cùng một nhân vật).\n"
                "- Chỉ THÊM nhân vật/thuật ngữ MỚI thực sự chưa có trong danh sách.\n"
                "- Khi cùng một người được gọi bằng chức danh khác nhau, ưu tiên ánh xạ "
                "về tên đã thiết lập.\n"
                f"{prior_context.strip()}\n"
                "=== HẾT NGỮ CẢNH ĐÃ THIẾT LẬP ===\n"
            )
        system_instr += (
            "\n\nYÊU CẦU cho từng trường:"
            "\n• source_lang: Mã ISO 639-1 của ngôn ngữ mà CHÍNH CÁC DÒNG PHỤ ĐỀ Ở TRÊN "
            "ĐANG ĐƯỢC VIẾT (ngôn ngữ của phần CHỮ trong danh sách '<số>. <nội dung>'), "
            "vd zh, ja, ko, en, vi. ĐÂY LÀ NGÔN NGỮ CỦA VĂN BẢN CẦN DỊCH — TUYỆT ĐỐI "
            "KHÔNG phải ngôn ngữ THOẠI trong audio video, cũng KHÔNG phải ngôn ngữ gốc "
            "của bộ phim. VÍ DỤ QUAN TRỌNG: phim Trung Quốc (audio tiếng Trung, tên nhân "
            "vật/thuật ngữ là Hán tự) NHƯNG các dòng phụ đề trên viết bằng tiếng Anh ⇒ "
            "source_lang = 'en' (KHÔNG phải 'zh'). Chỉ khi CHỮ phụ đề trộn nhiều ngôn ngữ "
            "mới chọn ngôn ngữ chiếm đa số KÝ TỰ trong các dòng trên."
            f"\n• characters: MẢNG đối tượng, mỗi nhân vật một phần tử với các trường: "
            f"{{\"name\": tên GỐC nguyên văn (Hán/Nhật/Hàn giữ nguyên chữ gốc), "
            f"\"aka\": phiên âm/chuyển ngữ sang {target_lang}, "
            f"\"gender\": 'nam' hoặc 'nữ', \"age\": mô tả tuổi (vd '~25', 'trung niên'), "
            f"\"role\": vai trò + quan hệ + mô tả ngắn + CÁCH XƯNG HÔ}}.\n"
            # [v3.23.524] Đo trên phim thật: cặp 春花→宋钱 dùng lẫn lộn "chị" 16 lượt, "em" 15,
            # "cậu" 12 — người xem thấy quan hệ nhân vật nhảy loạn. Chỉ thị "giữ nhất quán" chung
            # chung ở khâu dịch KHÔNG đủ, vì mỗi lô dịch độc lập nên model tự chọn lại mỗi lần.
            # Chốt sẵn ngay từ khâu phân tích, rồi theo roster đi vào MỌI giai đoạn dịch.
            f"BẮT BUỘC chốt CÁCH XƯNG HÔ trong \"role\": với các nhân vật mà người này hay nói "
            f"chuyện cùng, ghi rõ tự xưng là gì và gọi đối phương là gì, dạng "
            f"'xưng hô: với <Tên> tự xưng \'em\' gọi \'anh\''. Chọn theo tuổi tác, vai vế và "
            f"mức thân sơ, rồi GIỮ NGUYÊN suốt phim — tiếng Việt không có đại từ trung tính như "
            f"'you', đổi cách gọi giữa chừng làm người xem hiểu sai quan hệ.\n"
            # [v3.23.550] Chuẩn đoán thật: roster ghi "với Tống Tiền tự xưng 'tôi/Mộ Vân/chị' gọi
            # 'cậu/Tống thần y/Tống Tiền'" — BA phương án cho một cặp. Khâu dịch được phép chọn
            # bất kỳ nên mỗi lô chọn một kiểu, và bảng "đã chốt" thành ra không chốt gì cả.
            f"MỖI CẶP chỉ ghi ĐÚNG MỘT đại từ tự xưng và ĐÚNG MỘT đại từ gọi đối phương — KHÔNG "
            f"dùng '/' để liệt kê nhiều phương án. Danh xưng/biệt hiệu (vd 'Tống thần y', 'Liễu "
            f"tiểu thư') là cách GỌI TÊN, ghi riêng trong ngoặc, không thay cho đại từ: "
            f"'với Tống Tiền tự xưng \'chị\' gọi \'cậu\' (danh xưng: Tống thần y)'. Nếu quan hệ "
            f"THẬT SỰ đổi theo diễn biến thì ghi rõ ĐIỀU KIỆN cho từng giai đoạn: 'trước khi thua "
            f"xưng \'ta\' gọi \'mày\'; sau khi thua xưng \'tôi\' gọi \'Tống tiên sinh\''.\n"
            # [v3.23.623] Ca thật: nhân vật là "Dao Dao" nhưng mệnh đề ghi "với Diêu Dao" → quy tắc
            # không gắn được vào cặp nào, các lưới xưng hô im lặng với cả cuộc thoại của cặp đó.
            # [v3.23.634] Đại từ theo thể loại + đúng một phương án + không bịa tên/trộn chữ Hán.
            f"{ANALYSIS_ADDRESS_REGISTER_RULE}"
            f"Sau chữ 'với' PHẢI viết đúng tên (phiên âm hoặc tên gốc) của MỘT nhân vật khác "
            f"CÓ TRONG chính bảng này, y hệt cách bảng viết — không đặt tên mới, không biến thể "
            f"chính tả, không mô tả thay tên.\n"
            f"Ví dụ (phụ đề tiếng Trung → {target_lang}): "
            f"{{\"name\": \"林昆\", \"aka\": \"Lâm Côn\", \"gender\": \"nam\", \"age\": \"~25\", "
            f"\"role\": \"nhân vật chính: con trai tổng chủ, sở hữu hệ thống\"}}\n"
            # [v3.23.490] LIỆT KÊ ĐỦ: người dùng phản ánh thiếu nhân vật phụ có tên thật sự
            # xuất hiện (vd 王梅/Vương Mai). Model hay chỉ kể tuyến chính → phải nói rõ.
            # [v3.23.494] Chuẩn đoán thật cho thấy roster bị lẫn HỆ THỐNG/THƯƠNG HIỆU/ỨNG DỤNG
            # (宠妻系统, Alipay, 支付宝) và cùng một người bị tách thành nhiều mục (胖子 vs 三胖子)
            # → bảng quy đổi tên gửi cho khâu dịch bị nhiễu, gọi tên không nhất quán.
            "CHỈ LIỆT KÊ CON NGƯỜI. TUYỆT ĐỐI KHÔNG đưa vào đây: hệ thống/thần khí/pháp bảo, "
            "ứng dụng, thương hiệu, công ty, tổ chức, địa điểm, vật phẩm — những thứ đó thuộc "
            "về glossary.\n"
            "GỘP BÍ DANH: một người có nhiều cách gọi (tên thật, biệt danh, cách gọi thân mật, "
            "chức danh) thì chỉ tạo MỘT phần tử; ghi các cách gọi vào \"name\", ngăn bằng ' / ' "
            "theo thứ tự phổ biến giảm dần (vd \"胖子 / 三胖子\"). KHÔNG tạo hai phần tử cho cùng "
            "một người.\n"
            # [v3.23.620] Đo trên phim mẫu: aka '师父 (Sư phụ của Diệp Nhiên)' lọt NGUYÊN VĂN vào
            # bản dịch ('Sư phụ của Diệp Nhiên đã bế tử quan'), còn '瑶瑶' bị dịch thành họ tên
            # 'Sở Dao' vì aka chỉ có một cách đọc. aka là chữ được CHÉP vào lời thoại.
            "\"aka\" là ĐÚNG chữ sẽ chép vào lời thoại khi nhắc người đó: ghi cách đọc cho TỪNG "
            "bí danh trong \"name\", cùng thứ tự, ngăn bằng ' / ' (vd name \"楚瑶 / 瑶瑶\" → aka "
            "\"Sở Dao / Dao Dao\"). Nhân vật KHÔNG tên thì aka là danh xưng NGẮN như lời thoại gọi "
            "('师父' → 'Sư phụ', '门下弟子' → 'Đệ tử'), TUYỆT ĐỐI không viết cụm mô tả kiểu 'Sư "
            "phụ của X', 'Đệ tử cùng môn' — quan hệ ghi vào \"role\".\n"
            "LIỆT KÊ ĐẦY ĐỦ — KHÔNG chỉ tuyến chính: bao gồm CẢ nhân vật phụ, nhân vật chỉ "
            "xuất hiện vài cảnh, và nhân vật chỉ được NHẮC TÊN trong lời thoại (nhân vật vắng "
            "mặt, người đã mất, người ở xa). Bất kỳ ai có TÊN RIÊNG trong phụ đề đều phải có "
            "mặt. Nhân vật không tên nhưng có vai rõ (bác sĩ, cảnh sát, quản gia…) thì ghi "
            "nhãn vai trò làm tên. Thà thừa còn hơn thiếu.\n"
            "BẮT BUỘC ghi GIỚI TÍNH cho TỪNG nhân vật: hãy NGHE GIỌNG NÓI trong "
            "audio của video (cao độ, âm sắc) kết hợp hình ảnh để xác định chính xác — "
            "giới tính quyết định cách xưng hô (anh/em/cô/chú/ông/bà) khi dịch, sai giới "
            "tính là sai cả bản dịch. Nêu thêm QUAN HỆ vai vế nếu rõ (vd 'anh trai của X', "
            "'cấp trên của Y').\n"
            f"Tên đã là Latin (Anh, Pháp…) thì giữ nguyên ở \"name\", \"aka\" để trống; nêu "
            f"chức danh trong \"role\" (vd 'Tiến sĩ', 'Giám đốc NASA').\n"
            f"BẮT BUỘC: luôn ghi \"aka\" ({target_lang}) cho mọi tên CJK (Hán, Nhật, Hàn)."
            f"\n• overview: Viết HOÀN TOÀN bằng {target_lang} — "
            f"TUYỆT ĐỐI không dùng chữ ngôn ngữ gốc trong overview. "
            f"Khi nhắc tên nhân vật: LUÔN dùng phiên âm {target_lang} "
            f"(vd: 'Lâm Côn' thay vì '林昆', 'Diệp Thiên' thay vì '叶天'). "
            "Tóm tắt ĐẦY ĐỦ bao gồm: "
            "(1) Thế giới quan và bối cảnh; "
            "(2) Tuyến nhân vật chính và mối quan hệ; "
            "(3) Diễn biến cốt truyện chính (bắt đầu → phát triển → cao trào → kết thúc tập); "
            "(4) Các tình tiết và twist quan trọng. "
            "Độ dài: không giới hạn từ. Ưu tiên thông tin đầy đủ và súc tích."
            # [v3.23.23] Glossary: chốt cách dịch thuật ngữ + viết tắt để NHẤT QUÁN.
            f"\n• glossary: MẢNG đối tượng giúp DỊCH NHẤT QUÁN toàn phim. Mỗi phần tử: "
            f"{{\"term\": thuật ngữ GỐC nguyên văn, \"translation\": bản dịch {target_lang} chuẩn}}. "
            # [v3.23.490] Người dùng phản ánh glossary quá ít (9 mục cho 5540 dòng). Model chỉ
            # bắt vài thuật ngữ "đặc thù" → phải liệt kê RÕ các nhóm cần thu thập.
            f"\nHÃY THU THẬP RỘNG — mọi cụm LẶP LẠI hoặc dễ dịch lệch, gồm:"
            f"\n  - ĐỊA DANH & CƠ SỞ: thành phố, quận, đường, bệnh viện, khách sạn, công ty, "
            f"trường học, quán ăn, khu chung cư (vd '福安私立医院', '中山酒店')."
            f"\n  - TỔ CHỨC & CHỨC DANH: tập đoàn, phòng ban, chức vụ, cách xưng hô công việc "
            f"(vd '总裁 => tổng tài', '林总 => Tổng giám đốc Lâm')."
            f"\n  - CHUYÊN NGÀNH: y khoa, pháp lý, tài chính, kỹ thuật, võ thuật/tu luyện "
            f"(vd '骨髓移植 => ghép tuỷ', '白血病 => bệnh bạch cầu')."
            f"\n  - TIỀN TỆ & ĐƠN VỊ ĐO: đơn vị tiền, cách đọc số lớn, đơn vị đo lường "
            f"(vd '万 => vạn (10 nghìn)', '亿 => trăm triệu')."
            f"\n  - THÀNH NGỮ, TỤC NGỮ, DANH NGÔN, thơ, khẩu hiệu, lời bài hát được trích."
            f"\n  - BIỆT DANH & TIẾNG LÓNG mạng, cách gọi thân mật/miệt thị "
            f"(vd '老登', '卸货'), món ăn, thương hiệu, vật phẩm đặc thù."
            f"\n  - TỪ VIẾT TẮT: GIỮ NGUYÊN dạng viết tắt, chú thích lần đầu "
            f"(vd 'FBI => FBI (Cục Điều tra Liên bang Mỹ)'). KHÔNG phiên âm các viết tắt "
            f"đã thông dụng quốc tế."
            f"\n  - TỈ LỆ/ĐỊNH LƯỢNG dễ nhầm chiều: ghi RÕ thứ tự, KHÔNG đảo "
            f"(vd '米水比例 => tỉ lệ gạo:nước (gạo trước, nước sau)')."
            f"\nSỐ LƯỢNG: phim dài vài nghìn dòng thường có HÀNG CHỤC mục — đừng chỉ liệt kê "
            f"vài mục. Chỉ bỏ qua từ ngữ thông thường không ảnh hưởng tính nhất quán."
            # [v3.23.505] Đo thực tế: khi glossary phình lên 348 mục thì tỉ lệ vi phạm TĂNG, vì
            # phần lớn mục mới là TỪ PHỔ THÔNG (骨髓 = tủy, 恶心 = kinh tởm, 大老板 = sếp lớn) —
            # ép dùng một bản dịch cứng cho chúng vừa vô nghĩa vừa làm loãng các mục thật sự
            # quan trọng. Nêu rõ tiêu chí LOẠI TRỪ và cách ghi bản dịch.
            f"\n\nTIÊU CHÍ CHỌN (quan trọng — chất lượng hơn số lượng):"
            f"\n  - CHỈ đưa vào những cụm mà dịch KHÔNG NHẤT QUÁN sẽ gây khó hiểu: tên riêng, "
            f"thương hiệu, địa điểm, tổ chức, chức danh gắn với nhân vật, thuật ngữ do phim tự "
            f"đặt ra, thành ngữ/tục ngữ, tiếng lóng đặc thù, đơn vị tiền tệ."
            f"\n  - KHÔNG đưa vào TỪ PHỔ THÔNG mà người dịch nào cũng dịch giống nhau "
            f"(vd 'bệnh viện', 'buồn nôn', 'sếp lớn', 'tủy'). Chúng làm loãng bảng và ép bản dịch "
            f"trở nên cứng nhắc."
            f"\n  - Với tiếng lóng/thành ngữ, chọn bản dịch TỰ NHIÊN trong {target_lang} chứ đừng "
            f"phiên âm máy móc; nếu cần giải thích thì đặt trong ngoặc SAU bản dịch, còn phần "
            f"trước ngoặc phải là cụm dùng được thẳng trong lời thoại."
            + GLOSSARY_WORDING_RULE.format(target_lang=target_lang)
        )
        if getattr(self, "_native_address", False):
            system_instr += NATIVE_ADDRESS_ANALYSIS_RULE

        # [v3.23.37] Khi bật phân tích hình ảnh: LẤY LUÔN Visual Cues trong cùng request
        # phân tích (đã gửi video) → tiết kiệm quota, không cần gọi riêng. Thêm chỉ thị
        # 'cues' và dùng schema kết hợp.
        if with_visual_cues:
            system_instr += (
                f"\n\u2022 cues: Mảng gợi ý nghe-nhìn cho từng dòng phụ đề mà BẠN "
                f"XÁC ĐỊNH ĐƯỢC từ video. HÃY TẬN DỤNG TỐI ĐA cả HÌNH (gương "
                f"mặt, biểu cảm, cử chỉ, khẩu hình, trang phục, bối cảnh) LẪN TIẾNG "
                f"(âm sắc, giới tính, độ tuổi, ngữ điệu giọng nói) để: "
                f"(1) NHẬN DIỆN & GHI NHỚ danh tính người nói/người nghe NHẤT QUÁN "
                f"xuyên suốt phim — cùng một gương mặt/giọng nói phải gán CÙNG một "
                f"tên. Khi khớp được với Bảng nhân vật (Roster) ở trên, PHẢI dùng "
                f"đúng tên trong Roster; người chưa có thì đặt nhãn vai vế nhất quán. "
                f"Khi người nói KHÔNG xuất hiện trong khung hình (thuyết minh, lồng "
                f"tiếng ngoài hình, nói qua điện thoại/loa), HÃY dựa vào GIỌNG NÓI để "
                f"nhận diện và ghi chú kênh thoại (vd 'qua điện thoại', 'thuyết minh'). "
                f"(2) GHI bối cảnh + THÁI ĐỘ/CẢM XÚC của người nói (vd 'giận dữ "
                f"quát trong đại điện', 'thì thầm e sợ', 'trêu đùa thân mật') để "
                f"khâu dịch chọn đúng giọng văn và xưng hô. "
                f"Mỗi phần tử: {{\"id\": số dòng, \"spk\": người nói, \"to\": người "
                f"nghe, \"cue\": bối cảnh+thái độ/cảm xúc ngắn gọn}}. "
                # [v3.23.490] Người dùng phản ánh tên trong cues LÚC tiếng đích LÚC tiếng gốc,
                # lộn xộn → khâu dịch không khớp được với Roster. Chốt MỘT quy tắc: luôn dùng
                # TÊN GỐC, đúng bằng trường "name" của Roster, để đối chiếu máy móc chính xác.
                f"QUY TẮC TÊN (BẮT BUỘC, không được lẫn lộn): \"spk\" và \"to\" LUÔN ghi "
                f"TÊN GỐC nguyên văn, TRÙNG KHỚP TUYỆT ĐỐI với trường \"name\" trong "
                f"characters (vd '林逸', KHÔNG ghi 'Lâm Dật', không ghi lẫn hai thứ tiếng, "
                f"không thêm chức danh). Nhân vật chưa có trong characters thì dùng nhãn vai "
                f"trò ngắn và GIỮ NGUYÊN nhãn đó ở mọi dòng. Riêng \"cue\" (mô tả bối cảnh) "
                f"viết bằng {target_lang}. "
                # [v3.23.600] Không để trống người nói: lồng tiếng cần mọi dòng có giọng.
                f"{_speaker_fallback_rule()} "
                # [v3.23.596] "CHỈ thêm dòng bạn TỰ TIN" khiến model bỏ hẳn dòng thay vì
                # để trống nhãn — mỗi đoạn nay có danh sách dòng kèm mốc giờ để đi hết.
                f"Nếu chỉ xem một đoạn video, chỉ trả cues cho các dòng THUỘC đoạn đó "
                f"(mỗi dòng trong danh sách phụ đề kèm mốc [MM:SS] tính từ đầu video)."
            )
        analysis_schema = (
            _CONTEXT_WITH_CUES_SCHEMA if with_visual_cues else _CONTEXT_ANALYSIS_SCHEMA
        )

        # ── Prompt ────────────────────────────────────────────────────────
        prompt = _wrap_xml_block(
            "subtitle_content",
            f"Tổng số phụ đề (sau khử trùng): {len(deduped)} / {len(source_lines)} dòng gốc.\n\n"
            f"{content_block}",
        )
        if timed_cues:
            # Chỉ thị ĐỨNG CUỐI sau khối dữ liệu dài (prompting-strategies: "Place your specific
            # instructions or questions at the very end ... Based on the information above").
            prompt += (
                "\n\nDựa trên video và danh sách phụ đề ở trên: số trong [MM:SS] là lúc dòng "
                "đó hiện trên hình, tính từ ĐẦU video. Trả characters, overview, glossary và "
                f"cues. \"cues\" phải có ĐÚNG một phần tử cho MỖI id trong {len(deduped)} dòng, "
                "theo thứ tự — với từng id hãy xem khung hình và nghe giọng quanh mốc đó "
                "(±2 giây) rồi ghi spk, to, cue. Không bỏ dòng nào, không để trống spk."
            )

        # [Thinking ưu tiên BẬT] Thinking dynamic giúp phiên âm tên CJK + suy luận
        # ngữ cảnh chính xác hơn. Các model mới (kể cả nhỏ) đa số đã hỗ trợ; nếu model
        # THỰC SỰ không hỗ trợ, _call_gemini tự bỏ thinking và thử lại (fallback êm).
        config = self._build_config(
            0.25, analysis_schema, system_instr,
            enable_thinking=True,
            thinking_budget=-1,
            model_name=model_name,
            thinking_level=getattr(self, "_analysis_thinking_level", "medium"),
            media_resolution=getattr(self, "_analysis_media_resolution", "medium"),
        )

        def _noop_validator(data: dict[str, Any]) -> None:
            pass  # schema đơn giản — không cần validate nghiêm ngặt

        # [v3.23.475] Dùng bản HEAL: xoay key do quota (429) làm handle video của key cũ bị
        # 403 → tải lại bằng key hiện tại, tránh mất ngữ cảnh HÌNH ở các đoạn CUỐI phim.
        video_handles = self._resolve_handles_with_heal(video_refs, [])
        base_est = max(1, len(prompt) // 4)

        # [v3.23.14] Gemini chỉ nhận TỐI ƯU 1 video/request; gửi nhiều đoạn cùng lúc
        # gây 400 INVALID_ARGUMENT và dễ vượt giới hạn token. → Khi có >1 đoạn, dùng
        # MAP-REDUCE: phân tích TỪNG đoạn riêng (mỗi request 1 video + toàn bộ text),
        # rồi GỘP các phân tích cục bộ thành một phân tích tổng mạch lạc. Kết quả tương
        # đương gửi tất cả cùng lúc nhưng không vượt quota / không 400.
        if len(video_handles) > 1:
            map_reduce_result = self._analyze_context_map_reduce(
                prompt=prompt, system_instr=system_instr, config=config,
                video_handles=video_handles, video_refs=video_refs or [],
                target_lang=target_lang, model_name=model_name,
                base_est=base_est, cancel_cb=cancel_cb,
                with_visual_cues=with_visual_cues,
                segment_lines=deduped,
            )
            # Đối chiếu nhãn ngôn ngữ với ký tự thật của phụ đề (xem _reconcile_source_lang).
            return replace(
                map_reduce_result,
                source_lang=self._reconcile_source_lang(
                    map_reduce_result.source_lang, source_lines
                ),
            )

        # Trường hợp 0 hoặc 1 video: gửi trực tiếp như cũ.
        est = base_est
        if video_refs:
            est += sum(
                int(max(0.0, getattr(r, "end_sec", 0.0) - getattr(r, "start_sec", 0.0))
                    * getattr(self, "_analysis_video_tps", 100))
                for r in video_refs
            )
            logger.info("Phân tích ngữ cảnh kèm %d đoạn video.", len(video_handles))

        data = self._call_gemini(
            model_name, prompt, config, _noop_validator, cancel_cb=cancel_cb,
            video_files=video_handles, est_tokens=est,
            video_refs=list(video_refs) if video_refs else None,
            reject_truncated=True,
        )
        cues_json = self._extract_cues_json(data) if with_visual_cues else ""
        return SubtitleContextAnalysis(
            source_lang=self._reconcile_source_lang(
                str(data.get("source_lang") or "").strip(), source_lines
            ),
            characters=_normalize_roster(data.get("characters")),
            overview=str(data.get("overview") or "").strip(),
            glossary=_normalize_glossary(data.get("glossary")),
            visual_cues=cues_json,
        )

    @staticmethod
    def _reconcile_source_lang(
        ai_source_lang: str, source_lines: list[TranslationLine]
    ) -> str:
        """Đối chiếu ``source_lang`` do AI trả với KÝ TỰ THẬT của phụ đề (hàm thuần).

        AI đôi khi nhầm ngôn ngữ THOẠI của phim / ngôn ngữ gốc bộ phim với ngôn ngữ
        mà CHỮ phụ đề đang viết (vd phim Trung nhưng .srt tiếng Anh → AI trả 'zh').
        Nhãn CJK sai kéo theo bộ lọc 'dòng rác CJK' và các gợi ý prompt sai lệch.

        Chỉ CAN THIỆP ở trường hợp CHẮC CHẮN mâu thuẫn: AI nói CJK nhưng văn bản gần
        như KHÔNG có ký tự CJK. Khi đó ghi WARNING rõ ràng để người dùng kiểm tra —
        KHÔNG tự đoán mã Latin cụ thể (en/fr/…) vì không thể suy luận tất định.

        Args:
            ai_source_lang: Mã ngôn ngữ AI trả (có thể rỗng).
            source_lines:   Các dòng phụ đề gốc để đo tỉ lệ ký tự CJK.

        Returns:
            ``source_lang`` giữ nguyên (chỉ cảnh báo, không sửa để tránh đoán sai).
        """
        lang = (ai_source_lang or "").strip()
        if not lang or not any(
            lang.lower().startswith(p) for p in _CJK_LANG_PREFIXES
        ):
            return lang
        non_empty = [ln for ln in source_lines if (ln.text or "").strip()]
        if not non_empty:
            return lang
        cjk_lines = sum(1 for ln in non_empty if _CJK_CHAR_RE.search(ln.text))
        cjk_ratio = cjk_lines / len(non_empty)
        if cjk_ratio < _CJK_TEXT_MIN_RATIO:
            logger.warning(
                "Phân tích ngữ cảnh trả source_lang='%s' (nhóm CJK) nhưng chỉ %.0f%% "
                "dòng phụ đề có ký tự CJK → nhiều khả năng AI nhầm NGÔN NGỮ THOẠI/gốc "
                "của phim với NGÔN NGỮ CHỮ phụ đề. Hãy đặt 'Ngôn ngữ gốc' đúng với "
                "ngôn ngữ CHỮ trong phụ đề (vd 'en' nếu phụ đề tiếng Anh).",
                lang, 100.0 * cjk_ratio,
            )
        return lang

    @staticmethod
    def _extract_cues_json(data: dict[str, Any]) -> str:
        """[v3.23.37] Chuyển mảng 'cues' trong kết quả phân tích → chuỗi JSON rút gọn.

        Lọc phần tử hợp lệ (id > 0 và có ít nhất speaker hoặc addressee). Trả '' nếu
        không có cue nào, để không ghi đè dữ liệu cũ một cách vô ích.
        """
        raw_cues = data.get("cues")
        if not isinstance(raw_cues, list):
            return ""
        cleaned: list[dict[str, Any]] = []
        for item in raw_cues:
            if not isinstance(item, dict):
                continue
            try:
                line_no = int(item.get("id", 0))
            except (TypeError, ValueError):
                continue
            spk = str(item.get("spk", "") or "").strip()
            to = str(item.get("to", "") or "").strip()
            if line_no > 0 and (spk or to):
                entry: dict[str, Any] = {"id": line_no}
                if spk:
                    entry["spk"] = spk
                if to:
                    entry["to"] = to
                cleaned.append(entry)
        if not cleaned:
            return ""
        cleaned.sort(key=lambda c: c["id"])
        return json.dumps(cleaned, ensure_ascii=False, separators=(",", ":"))

    def _analyze_context_map_reduce(
        self,
        prompt: str,
        system_instr: str,
        config: Any,
        video_handles: list[Any],
        video_refs: list[Any],
        target_lang: str,
        model_name: str,
        base_est: int,
        cancel_cb: CancellationCallback | None,
        with_visual_cues: bool = False,
        segment_lines: list[TranslationLine] | None = None,
    ) -> SubtitleContextAnalysis:
        """[v3.23.23] Phân tích ngữ cảnh TUẦN TỰ TÍCH LUỸ qua từng đoạn video.

        [v3.23.596] ``segment_lines``: dòng phụ đề (có mốc giờ) để mỗi đoạn nhận danh sách
        dòng của chính nó khi lấy Visual Cues — xem :func:`_segment_cue_lines_directive`.

        Thay cho map-reduce (N+1 request: N đoạn + 1 gộp), cách này chỉ tốn N request
        và KHÔNG cần bước gộp riêng: kết quả phân tích đoạn k được đưa vào làm NGỮ CẢNH
        cho đoạn k+1, nên đoạn cuối đã cho ra bản phân tích hợp nhất toàn phim. Vừa tiết
        kiệm quota (tránh 429 ở bước gộp), vừa chất lượng hơn (đoạn sau biết nhân vật/
        cốt truyện tích luỹ từ các đoạn trước thay vì gộp mù hai bản rời).
        """
        def _noop(_data: dict[str, Any]) -> None:
            pass

        accumulated: SubtitleContextAnalysis | None = None
        # [v3.23.37] Gộp Visual Cues từ mọi đoạn (mỗi đoạn trả cues cho dòng của nó).
        merged_cues: dict[int, dict[str, Any]] = {}
        n = len(video_handles)
        # [v3.23.587] Key đã resolve handle của từng đoạn — xem ``_refresh_segment_handle``.
        handle_keys = [self._current_key_id()] * n
        # [v3.23.564] Tiến độ theo đoạn: nạp phần đã xong ở lần chạy trước (nếu có).
        store = getattr(self, "_analysis_checkpoints", None)
        # [v3.23.593] Tiến độ dùng chung cho mọi model (xem analysis_checkpoint).
        checkpoint_key = self._analysis_progress_key(prompt, video_refs, with_visual_cues)
        done_segments: set[int] = set()
        # [v3.23.566] Diễn biến chi tiết theo đoạn — ghép theo THỜI GIAN ở cuối (xem
        # overview_composer): hết cảnh nối các bản tóm tắt toàn phim lặp nhau.
        segment_summaries: dict[int, str] = {}
        segment_models: dict[int, str] = {}
        if store is not None:
            saved = self._load_analysis_progress(
                store, checkpoint_key, model_name, prompt, video_refs, with_visual_cues
            )
            if saved is not None and saved.done_segments:
                done_segments = {i for i in saved.done_segments if 0 <= i < n}
                accumulated = saved.accumulated
                merged_cues = dict(saved.cues)
                segment_summaries = dict(saved.segment_summaries)
                segment_models = {
                    i: model for i, model in saved.segment_models.items()
                    if i in done_segments
                }
                logger.info(
                    "Tiếp tục phân tích từ lần trước: %d/%d đoạn đã xong%s, chỉ gửi %d "
                    "đoạn còn thiếu bằng '%s'.",
                    len(done_segments), n,
                    describe_segment_models(segment_models, model_name),
                    n - len(done_segments), model_name,
                )
        # [v3.23.566] Tiến độ lưu bởi bản cũ (có đoạn đã xong nhưng chưa có diễn biến theo
        # đoạn): giữ cách gộp tóm tắt cũ cho trọn phiên — ghép theo thời gian chỉ với vài
        # đoạn mới sẽ đánh mất chi tiết của các đoạn đã phân tích trước đó.
        legacy_progress = bool(done_segments) and not segment_summaries
        quota_exhausted: QuotaExhaustedError | None = None
        # [v3.23.585] Model hạn mức ngày ÍT mà server quá tải kéo dài: dừng gửi các đoạn
        # còn lại — mỗi lượt 503 vẫn bị tính, gửi tiếp chỉ đốt hạn mức. Nhật ký thật: lần
        # phân tích đầu đốt 60 request của 3 key trong 93 phút, xong 1/17 đoạn.
        overload_stop: SubtitleTranslationError | None = None

        def _save_progress() -> None:
            if store is not None:
                store.save(
                    checkpoint_key,
                    AnalysisProgress(
                        done_segments=set(done_segments),
                        accumulated=accumulated,
                        cues=merged_cues,
                        segment_summaries=segment_summaries,
                        segment_models=segment_models,
                    ),
                )

        def _run_segment(idx: int, *, is_retry: bool = False) -> bool:
            """Phân tích MỘT đoạn, hợp nhất vào ngữ cảnh tích luỹ. True nếu thành công."""
            nonlocal accumulated
            self._refresh_segment_handle(idx, video_handles, video_refs, handle_keys)
            handle = video_handles[idx]
            ref = video_refs[idx] if idx < len(video_refs) else None
            # Danh sách riêng của lời gọi: nếu key bị xoay NGAY trong lời gọi thì ref mới
            # được ghi vào đây, rồi chép ngược ra ``video_refs`` ở khối ``finally``.
            segment_refs = [ref] if ref is not None else None
            span = ""
            seg_est = base_est
            if ref is not None:
                start = getattr(ref, "start_sec", 0.0)
                end = getattr(ref, "end_sec", 0.0)
                span = f" (đoạn {idx + 1}/{n}, {start:.0f}-{end:.0f}s)"
                seg_est += int(max(0.0, end - start) * getattr(self, "_analysis_video_tps", 100))
            is_last = idx == n - 1
            logger.info(
                "Phân tích ngữ cảnh TUẦN TỰ đoạn %d/%d%s (kèm ngữ cảnh tích luỹ%s).",
                idx + 1, n, " — THỬ LẠI cuối phiên" if is_retry else "",
                "" if accumulated is None else " từ đoạn trước",
            )

            # Đưa kết quả tích luỹ (nếu có) vào prompt làm ngữ cảnh cho đoạn hiện tại.
            ctx_block = ""
            if accumulated is not None:
                previous_detail = segment_summaries.get(idx - 1, "")
                ctx_block = (
                    "\n\nNGỮ CẢNH TÍCH LUỸ từ các đoạn TRƯỚC (hãy CẬP NHẬT, BỔ SUNG, "
                    "KHÔNG bỏ sót thông tin cũ — hợp nhất với quan sát đoạn này):\n"
                    f"• Nhân vật:\n{accumulated.characters}\n"
                    f"• Tóm tắt:\n{accumulated.overview}\n"
                    f"• Thuật ngữ:\n{accumulated.glossary}"
                    + (
                        f"\n• Diễn biến đoạn NGAY TRƯỚC:\n{previous_detail}"
                        if previous_detail
                        else ""
                    )
                )
            if is_last:
                seg_prompt = (
                    f"{prompt}{ctx_block}\n\n(Đây là đoạn CUỐI{span}. Hãy trả về bản "
                    "phân tích HỢP NHẤT HOÀN CHỈNH cho TOÀN PHIM: gộp ngữ cảnh tích luỹ "
                    "ở trên với quan sát đoạn này thành nhân vật đầy đủ, tóm tắt mạch lạc "
                    "toàn phim, và bảng thuật ngữ nhất quán.)"
                )
            else:
                seg_prompt = (
                    f"{prompt}{ctx_block}\n\n(Bạn đang xem MỘT đoạn video{span} của toàn "
                    "phim. Hãy phân tích, đồng thời HỢP NHẤT với ngữ cảnh tích luỹ ở trên "
                    "thành bản cập nhật đầy đủ tới thời điểm này.)"
                )
            seg_prompt += _segment_fields_directive(target_lang)
            if with_visual_cues and segment_lines and ref is not None:
                seg_prompt += _segment_cue_lines_directive(
                    segment_lines,
                    float(getattr(ref, "start_sec", 0.0)),
                    float(getattr(ref, "end_sec", 0.0)),
                )
            try:
                data = self._call_gemini(
                    model_name, seg_prompt, config, _noop, cancel_cb=cancel_cb,
                    video_files=[handle], est_tokens=seg_est,
                    video_refs=segment_refs, reject_truncated=True,
                )
                current = SubtitleContextAnalysis(
                    source_lang=str(data.get("source_lang") or "").strip(),
                    characters=_normalize_roster(data.get("characters")),
                    overview=str(data.get("overview") or "").strip(),
                    glossary=_normalize_glossary(data.get("glossary")),
                )
                # Tích luỹ: ưu tiên kết quả mới (đã gồm ngữ cảnh cũ); giữ trường cũ nếu
                # đoạn này trả rỗng (model thi thoảng bỏ trống 1 trường).
                detail = str(data.get("segment_summary") or "").strip()
                if detail and not legacy_progress:
                    # [v3.23.566] Model đã trả diễn biến riêng của đoạn → 'overview' là
                    # bản toàn cảnh CẬP NHẬT (thay thế), không nối vào bản cũ.
                    segment_summaries[idx] = detail
                    accumulated = self._merge_sequential_analysis(
                        accumulated, current, replace_overview=True
                    )
                else:
                    accumulated = self._merge_sequential_analysis(accumulated, current)
                # Gộp cues của đoạn này (id duy nhất; đoạn sau ghi đè nếu trùng id).
                if with_visual_cues:
                    seg_cues_json = self._extract_cues_json(data)
                    if seg_cues_json:
                        for c in json.loads(seg_cues_json):
                            merged_cues[int(c["id"])] = c
                done_segments.add(idx)
                segment_models[idx] = model_name
                _save_progress()
                return True
            except QuotaExhaustedError as exc:
                # [v3.23.564] Hết hạn mức NGÀY ở MỌI key: trước đây lỗi này thoát khỏi
                # vòng lặp và VỨT toàn bộ các đoạn đã phân tích (ca thật: 20/22 đoạn của
                # gemini-3.8-flash). Nay dừng gửi tiếp, giữ phần đã có.
                nonlocal quota_exhausted
                quota_exhausted = exc
                logger.warning(
                    "Hết hạn mức ngày khi phân tích đoạn %d — dừng gửi tiếp, GIỮ %d đoạn "
                    "đã xong: %s", idx + 1, len(done_segments), exc,
                )
                return False
            except SubtitleTranslationError as exc:
                # [v3.23.477] Ghi nhận lỗi VĨNH VIỄN (vd 400 INVALID_ARGUMENT) để KHÔNG retry vô
                # ích ở lượt cuối phiên (chỉ 503/504/timeout mới đáng thử lại).
                if not self._is_retryable(exc):
                    permanent_failed.add(idx)
                if (
                    self._is_overload_error(exc)
                    and self._overload_attempt_cap(model_name) is not None
                ):
                    nonlocal overload_stop
                    overload_stop = exc
                    logger.warning(
                        "Gemini QUÁ TẢI kéo dài ở đoạn %d — DỪNG gửi các đoạn còn lại để "
                        "không đốt hạn mức ngày của '%s' (mỗi lượt 503 vẫn bị tính). GIỮ "
                        "%d đoạn đã xong; chạy lại phân tích sau ít phút, hoặc đổi "
                        "model, sẽ chỉ gửi các đoạn còn thiếu.",
                        idx + 1, model_name, len(done_segments),
                    )
                    return False
                logger.warning(
                    "Phân tích đoạn %d thất bại%s, giữ ngữ cảnh tích luỹ hiện có: %s",
                    idx + 1, " (cả lần thử lại)" if is_retry else "", exc,
                )
                return False
            finally:
                if segment_refs and segment_refs[0] is not ref:
                    video_refs[idx] = segment_refs[0]

        permanent_failed: set[int] = set()
        failed_segments: list[int] = []
        for idx in range(n):
            if cancel_cb is not None and cancel_cb():
                raise TranslationCancelledError("Người dùng đã huỷ khi phân tích ngữ cảnh.")
            if idx in done_segments:
                continue
            if quota_exhausted is not None or overload_stop is not None:
                failed_segments.append(idx)
                continue
            if not _run_segment(idx):
                failed_segments.append(idx)

        # [v3.23.155] SECOND-CHANCE: 503/504 là quá tải TẠM THỜI phía Gemini — sau khi
        # đi hết các đoạn còn lại (vài phút), thử lại MỘT lượt các đoạn đã hỏng. Lúc này
        # ngữ cảnh tích luỹ đã GIÀU HƠN (gồm các đoạn sau) nên bản hợp nhất khi thử lại
        # thành công còn tốt hơn; đặc biệt cứu được đoạn MỞ PHIM (giới thiệu nhân vật).
        retryable_failed = [i for i in failed_segments if i not in permanent_failed]
        if permanent_failed:
            logger.warning(
                "%d đoạn lỗi VĨNH VIỄN (không retry: vd sai tham số) — bỏ qua lượt thử lại: %s",
                len(permanent_failed),
                ", ".join(str(i + 1) for i in sorted(permanent_failed)),
            )
        if (
            retryable_failed
            and quota_exhausted is None
            and overload_stop is None
            and not (cancel_cb is not None and cancel_cb())
        ):
            logger.info(
                "Thử lại %d đoạn phân tích thất bại (%s) sau khi hoàn tất các đoạn khác.",
                len(retryable_failed), ", ".join(str(i + 1) for i in retryable_failed),
            )
            for idx in retryable_failed:
                if cancel_cb is not None and cancel_cb():
                    raise TranslationCancelledError(
                        "Người dùng đã huỷ khi phân tích ngữ cảnh."
                    )
                if quota_exhausted is not None or overload_stop is not None:
                    break
                _run_segment(idx, is_retry=True)

        missing = [i for i in range(n) if i not in done_segments]
        if store is not None and not missing:
            if getattr(self, "_hold_analysis_progress", False):
                # [v3.23.643] Bước hoàn thiện người nói / tự kiểm tra còn chạy sau đây (tới hàng chục
                # phút, gửi lại cả video): huỷ hay đóng app giữa chừng không được làm mất 24 đoạn đã
                # phân tích. Use case xoá sau khi mọi bước xong (``release_analysis_progress``).
                self._held_analysis_progress.append((store, checkpoint_key))
            else:
                store.clear(checkpoint_key)
        if missing and accumulated is not None:
            logger.warning(
                "Phân tích ngữ cảnh THIẾU %d/%d đoạn (%s) — đã lưu tiến độ; chạy lại "
                "phân tích (cùng model hoặc model khác) sẽ chỉ gửi các đoạn này.",
                len(missing), n, ", ".join(str(i + 1) for i in missing),
            )
        if quota_exhausted is not None and accumulated is None:
            raise quota_exhausted
        if overload_stop is not None and accumulated is None:
            # Chưa đoạn nào xong: KHÔNG thử nốt lượt "chỉ văn bản" bên dưới — model đang
            # quá tải thì lượt đó cũng chỉ đốt thêm hạn mức.
            raise ModelOverloadedError(
                f"Gemini đang quá tải (503) với model '{model_name}' — đã dừng phân tích "
                "để giữ hạn mức ngày. Hãy thử lại sau ít phút hoặc đổi model phân tích.",
                model_name=model_name,
            ) from overload_stop

        if accumulated is not None and segment_summaries:
            import dataclasses

            from subtitles_extractor.application.services.overview_composer import (
                SegmentSummary,
                compose_overview,
            )

            accumulated = dataclasses.replace(
                accumulated,
                overview=compose_overview(
                    accumulated.overview,
                    [
                        SegmentSummary(
                            float(getattr(video_refs[i], "start_sec", 0.0)),
                            float(getattr(video_refs[i], "end_sec", 0.0)),
                            text,
                        )
                        for i, text in segment_summaries.items()
                        if i < len(video_refs)
                    ],
                ),
            )
        if accumulated is not None:
            if with_visual_cues and merged_cues:
                import dataclasses
                cues_list = sorted(merged_cues.values(), key=lambda c: c["id"])
                cues_json = json.dumps(cues_list, ensure_ascii=False, separators=(",", ":"))
                logger.info("Gộp %d Visual Cues từ %d đoạn phân tích.", len(cues_list), n)
                accumulated = dataclasses.replace(accumulated, visual_cues=cues_json)
            if missing:
                # [v3.23.585] Kết quả thiếu đoạn phải TỰ nói ra điều đó: giao diện từng
                # báo "Phân tích hoàn tất" cho cả lượt dừng giữa chừng.
                stop_reason = ""
                if quota_exhausted is not None:
                    stop_reason = ANALYSIS_STOP_DAILY_QUOTA
                elif overload_stop is not None:
                    stop_reason = ANALYSIS_STOP_OVERLOAD
                accumulated = replace(
                    accumulated,
                    missing_segments=tuple(index + 1 for index in missing),
                    total_segments=n,
                    stop_reason=stop_reason,
                )
            return accumulated

        # Mọi đoạn đều lỗi → thử lần cuối CHỈ với text (không video).
        data = self._call_gemini(
            model_name, prompt, config, _noop, cancel_cb=cancel_cb,
            video_files=None, est_tokens=base_est, reject_truncated=True,
        )
        return SubtitleContextAnalysis(
            source_lang=str(data.get("source_lang") or "").strip(),
            characters=_normalize_roster(data.get("characters")),
            overview=str(data.get("overview") or "").strip(),
            glossary=_normalize_glossary(data.get("glossary")),
        )

    def _analysis_progress_key(
        self, prompt: str, video_refs: list[Any], with_visual_cues: bool
    ) -> str:
        """[v3.23.593] Khoá tiến độ phân tích DÙNG CHUNG cho mọi model.

        Như :meth:`_analysis_checkpoint_key` nhưng không gồm tên model: đổi model khi
        Gemini quá tải (đúng lời khuyên ứng dụng đưa ra) vẫn chạy tiếp các đoạn còn thiếu.
        """
        return self._analysis_checkpoint_key(
            SHARED_PROGRESS_MODEL, prompt, video_refs, with_visual_cues
        )

    def _load_analysis_progress(
        self,
        store: AnalysisCheckpointStore,
        progress_key: str,
        model_name: str,
        prompt: str,
        video_refs: list[Any],
        with_visual_cues: bool,
    ) -> AnalysisProgress | None:
        """[v3.23.593] Đọc tiến độ dùng chung; chưa có thì nhận tiến độ bản cũ của model.

        Tiến độ bản cũ (khoá gồm tên model) được chép sang khoá dùng chung rồi xoá, mỗi
        đoạn ghi là do ``model_name`` phân tích.
        """
        saved = store.load(progress_key)
        if saved is not None:
            return saved
        legacy_key = self._analysis_checkpoint_key(
            model_name, prompt, video_refs, with_visual_cues
        )
        legacy = store.load(legacy_key)
        if legacy is None:
            return None
        legacy.segment_models = dict.fromkeys(legacy.done_segments, model_name)
        store.save(progress_key, legacy)
        store.clear(legacy_key)
        return legacy

    def hold_analysis_progress(self) -> None:
        """[v3.23.643] Giữ tiến độ phân tích trên đĩa tới khi :meth:`release_analysis_progress`.

        Đếm lồng nhau (hai lượt chồng nhau trên cùng một bộ dịch không xoá tiến độ của nhau).
        """
        self._hold_analysis_depth = getattr(self, "_hold_analysis_depth", 0) + 1
        self._hold_analysis_progress = True
        if not hasattr(self, "_held_analysis_progress"):
            self._held_analysis_progress: list[tuple[AnalysisCheckpointStore, str]] = []

    def release_analysis_progress(self, *, clear: bool) -> None:
        """[v3.23.643] Hết giữ; ``clear=True`` (mọi bước đã xong) thì xoá tiến độ đã giữ.

        Chỉ lượt NGOÀI CÙNG (đếm về 0) mới thật sự xoá / thả.
        """
        depth = max(0, getattr(self, "_hold_analysis_depth", 0) - 1)
        self._hold_analysis_depth = depth
        if depth > 0:
            return
        held = getattr(self, "_held_analysis_progress", [])
        self._hold_analysis_progress = False
        self._held_analysis_progress = []
        if not clear:
            return
        for store, key in held:
            try:
                store.clear(key)
            except OSError as exc:
                logger.warning("Không xoá được tiến độ phân tích %s: %s", key, exc)

    def _analysis_checkpoint_key(
        self,
        model_name: str,
        prompt: str,
        video_refs: list[Any],
        with_visual_cues: bool,
    ) -> str:
        """[v3.23.564] Khoá tiến độ phân tích: đổi bất kỳ yếu tố nào thì làm lại từ đầu.

        [v3.23.593] Chỉ còn dùng để nhận tiến độ lưu bởi bản cũ và (với tên model
        :data:`SHARED_PROGRESS_MODEL`) làm khoá dùng chung — xem
        :meth:`_analysis_progress_key`.

        Gồm model, phụ đề (prompt), cài đặt video/thinking và ranh giới các đoạn (theo
        thời gian — tên file cloud đổi khi tải lại nhưng đoạn thì không). CỐ Ý KHÔNG gồm
        chỉ thị hệ thống: nó chứa ngữ cảnh phim bộ, vốn đổi sau MỖI lần dịch một tập — nếu
        đưa vào khoá thì dịch xen giữa hai lần phân tích sẽ làm mất tiến độ.
        """
        ranges = ";".join(
            f"{float(getattr(ref, 'start_sec', 0.0)):.2f}-"
            f"{float(getattr(ref, 'end_sec', 0.0)):.2f}"
            for ref in video_refs
        )
        return analysis_checkpoint_key(
            model_name,
            prompt,
            ranges,
            str(bool(with_visual_cues)),
            str(getattr(self, "_analysis_media_resolution", "")),
            str(getattr(self, "_analysis_thinking_level", "")),
            str(getattr(self, "_analysis_video_fps", "")),
            # [v3.23.612] Chỉ thêm khi bật để khoá của các lượt cũ không đổi.
            *(("native_address",) if getattr(self, "_native_address", False) else ()),
        )

    @staticmethod
    def _merge_sequential_analysis(
        prev: SubtitleContextAnalysis | None,
        current: SubtitleContextAnalysis,
        replace_overview: bool = False,
    ) -> SubtitleContextAnalysis:
        """[v3.23.489] HỢP NHẤT TÍCH LUỸ kết quả phân tích tuần tự (trước đây GHI ĐÈ).

        Lỗi cũ: ``current.overview or prev.overview`` khiến đoạn CUỐI ghi đè toàn bộ tóm tắt →
        sau 23 đoạn chỉ còn tóm tắt của đoạn cuối (~1000 ký tự cho phim 2.6 giờ), mất sạch chi
        tiết các đoạn trước dù đã tốn quota phân tích đủ 23 đoạn.

        Nay:
        * ``overview``: NỐI TIẾP theo thứ tự thời gian (bỏ qua nếu bản mới đã bao hàm bản cũ —
          tức model tự mở rộng đúng), có trần độ dài để không phình prompt.
        * ``characters``/``glossary``: HỢP NHẤT theo tên/thuật ngữ, bản ghi sau bổ sung cho bản
          ghi trước (giữ mục cũ, thêm mục mới, ưu tiên mô tả dài hơn).

        Args:
            prev: Ngữ cảnh tích luỹ từ các đoạn trước (None nếu đây là đoạn đầu).
            current: Kết quả phân tích của đoạn hiện tại.

        Returns:
            Ngữ cảnh đã tích luỹ đầy đủ.
        """
        if prev is None:
            return current
        overview = (
            current.overview or prev.overview
            if replace_overview
            else _merge_overview_text(prev.overview, current.overview)
        )
        return SubtitleContextAnalysis(
            source_lang=current.source_lang or prev.source_lang,
            characters=_merge_roster_text(prev.characters, current.characters),
            overview=overview,
            glossary=_merge_glossary_text(prev.glossary, current.glossary),
        )

    @staticmethod
    def _parse_visual_cue_items(
        items: list[dict[str, Any]], max_line_no: int = 0
    ) -> list[VisualCue]:
        """[JSON Minification] Giải nén item rút gọn (id/spk/to/cue) → :class:`VisualCue`.

        Khoá JSON được nén để AI nhả được hàng ngàn dòng mà không chạm trần 8192
        token output: ``id``→line_no, ``spk``→speaker, ``to``→addressee, ``cue``→scene.
        Hàm thuần, dễ test.
        """
        cues: list[VisualCue] = []
        # [v3.23.638] ``"cues": null`` / số / chuỗi (model lệch schema) không được làm hỏng cả lượt.
        if not isinstance(items, (list, tuple)):
            return cues
        for item in items:
            if not isinstance(item, dict):
                continue
            line_no = _safe_int(item.get("id", item.get("line_no", 0)))
            # [v3.23.496] Model có thể trả chỉ số VƯỢT số dòng thật (dữ liệu thật: cue id=5541
            # trong khi phụ đề chỉ có 5540 dòng — lệch 1 ở dòng cuối). Cue trỏ dòng không tồn tại
            # là vô dụng khi dịch → loại ngay tại nguồn thay vì để lọt xuống khâu sau.
            if max_line_no > 0 and line_no > max_line_no:
                continue
            if line_no <= 0:
                continue
            cues.append(
                VisualCue(
                    line_no=line_no,
                    speaker=_clean_speaker_label(item.get("spk", item.get("speaker", ""))),
                    addressee=_clean_speaker_label(item.get("to", item.get("addressee", ""))),
                    scene=str(item.get("cue", item.get("scene", "")) or "").strip(),
                )
            )
        return cues

    def analyze_visual_cues(
        self,
        source_lines: list[TranslationLine],
        target_lang: str,
        model_name: str = "gemini-3.1-flash-lite",
        video_refs: list[Any] | None = None,
        batch_size: int = 150,
        sleep_between_s: float = 4.0,
        cancel_cb: CancellationCallback | None = None,
        progress_cb: StageProgressCallback | None = None,
        characters: str = "",
        known_cues: list[VisualCue] | None = None,
        context_lines: list[TranslationLine] | None = None,
    ) -> list[VisualCue]:
        """[Vision Director] Quét video trả lời Ai nói / Nói với ai / Bối cảnh-thái độ.

        Khác :meth:`analyze_global_context` (chỉ tóm tắt cốt truyện), hàm này phân
        tích TỪNG DÒNG phụ đề dựa trên video để sinh :class:`VisualCue`. Áp dụng:

        * **Micro-Batching**: xé thành lô ``batch_size`` dòng (mặc định 150), chèn
          ``time.sleep(sleep_between_s)`` giữa các lô để chống lỗi 429 (rate limit).
        * **JSON Minification**: schema dùng khoá rút gọn (id/spk/to/cue) để không
          chạm trần 8192 token output khi trả hàng ngàn dòng.
        * **Thinking Bypass**: tắt thinking (model nhỏ không hỗ trợ).

        Args:
            source_lines: Toàn bộ dòng phụ đề gốc.
            target_lang: Ngôn ngữ đích (để ghi chú vai vế bằng ngôn ngữ người dùng).
            model_name: Model Gemini có thị giác.
            video_refs: Các đoạn video đã tải lên làm ngữ cảnh.
            batch_size: Số dòng mỗi lô (chống chạm trần output).
            sleep_between_s: Thời gian nghỉ giữa các lô (chống 429).
            cancel_cb: Cho phép huỷ giữa chừng.
            progress_cb: Báo tiến độ ∈ [0, 1].
            characters: Bảng nhân vật (tuỳ chọn) — có thì ``spk``/``to`` phải dùng đúng
                tên trong bảng để lưới xưng hô tra được quy tắc.
            known_cues: [v3.23.581] Nhãn đã có của các dòng KHÔNG cần quét — dòng nằm
                xen kẽ trong vùng của lô được gửi kèm làm mạch hội thoại.
            context_lines: [v3.23.581] Toàn bộ dòng phụ đề (để lấy chữ của các dòng đó).

        Returns:
            Danh sách :class:`VisualCue` theo thứ tự dòng (chỉ các dòng được yêu cầu).
        """
        self._ensure_available()
        if not source_lines:
            return []

        system_instr = _visual_cues_system_instruction(target_lang, characters)
        # [Thinking ưu tiên BẬT] giúp suy luận "ai nói với ai" chính xác hơn; nếu
        # model không hỗ trợ, _call_gemini tự bỏ thinking và thử lại.
        config = self._build_config(
            0.2, _VISUAL_CUES_SCHEMA, system_instr,
            enable_thinking=True, thinking_budget=-1,
            model_name=model_name,
            thinking_level=getattr(self, "_analysis_thinking_level", "medium"),
            media_resolution=getattr(self, "_analysis_media_resolution", "medium"),
        )
        # [v3.23.475] Dùng bản HEAL: xoay key do quota (429) làm handle video của key cũ bị
        # 403 → tải lại bằng key hiện tại, tránh mất ngữ cảnh HÌNH ở các đoạn CUỐI phim.
        video_handles = self._resolve_handles_with_heal(video_refs, [])

        def _noop_validator(_data: dict[str, Any]) -> None:
            pass

        from subtitles_extractor.application.services.visual_cue_batching import (
            batch_lines_by_video_chunk,
            known_context_items,
            scan_line_items,
        )

        all_cues: list[VisualCue] = []
        # [v3.23.559] Trần id là chỉ số dòng LỚN NHẤT, không phải số dòng: khi quét bổ
        # sung chỉ các dòng còn thiếu (vd 2856 dòng rải tới dòng 3902), lấy ``len`` sẽ
        # vứt oan mọi cue của nửa cuối phim.
        max_line_no = max(line.index for line in source_lines)
        # [v3.23.560] Cắt lô tại ranh giới đoạn video → mỗi lô kèm đúng MỘT đoạn thay vì
        # 2-4 đoạn (xem visual_cue_batching).
        chunk_bounds = [
            (float(getattr(r, "start_sec", 0.0)), float(getattr(r, "end_sec", 0.0)))
            for r in (video_refs or [])
        ]
        batches = batch_lines_by_video_chunk(source_lines, chunk_bounds, batch_size)
        total_batches = len(batches)
        for batch_idx, batch in enumerate(batches):
            if cancel_cb is not None and cancel_cb():
                raise TranslationCancelledError("Người dùng đã huỷ phân tích Visual Cues.")
            # Chọn đoạn video phủ đúng khoảng thời gian của lô (tiết kiệm token).
            batch_handles = video_handles
            covering: list[Any] = []
            if video_refs:
                # Theo thời điểm BẮT ĐẦU (cùng quy tắc chia lô): câu cuối lô kéo dài qua
                # mép đoạn không được kéo thêm cả đoạn kế tiếp vào.
                t_start = min(ln.start_ms for ln in batch) / 1000.0
                t_end = max(ln.start_ms for ln in batch) / 1000.0 + 0.001
                covering = [
                    r for r in video_refs
                    if getattr(r, "start_sec", 0.0) < t_end
                    and getattr(r, "end_sec", 0.0) > t_start
                ]
                # [v3.23.561] Heal theo lô + KHÔNG còn rơi về "gửi kèm cả 22 đoạn" khi
                # đoạn của lô không resolve được (~3 triệu token cho một lô 150 dòng).
                batch_handles = self._handles_for_batch(covering, video_refs, batch_idx)
            # [v3.23.581] Mốc thời gian của từng dòng TÍNH TỪ ĐẦU đoạn video kèm theo —
            # trước đây model phải tự dò dòng nào hiện lúc nào trong đoạn 6 phút.
            clip_start = (
                float(getattr(covering[0], "start_sec", 0.0))
                if len(covering) == 1
                else None
            )
            blocks = [
                _wrap_xml_block("instruction", "Phân tích lô hiện tại, trả JSON rút gọn."),
            ]
            if known_cues and context_lines:
                known = known_context_items(batch, context_lines, known_cues, clip_start)
                if known:
                    blocks.append(_wrap_xml_block("known", known))
            blocks.append(_wrap_xml_block("lines", scan_line_items(batch, clip_start)))
            prompt = "\n\n".join(blocks)

            # [v3.23.561] Truyền ``video_refs`` + ước lượng token CÓ video. Trước đây
            # thiếu cả hai: (1) 429 tạm thời bị coi là lô "không đính video" nên xoay key
            # tự do → handle video thành của key cũ → 403; (2) 403 không tự tải lại được
            # vì điều kiện heal cần ``video_refs``; (3) quota chỉ tính token chữ nên không
            # điều tiết TPM, lô thứ hai đã dính 429. Nhật ký thật: chết ở lô 2.
            video_tokens = sum(
                int(
                    max(0.0, getattr(r, "end_sec", 0.0) - getattr(r, "start_sec", 0.0))
                    * getattr(self, "_analysis_video_tps", 300)
                )
                for r in covering
            )
            data = self._call_gemini(
                model_name, prompt, config, _noop_validator,
                cancel_cb=cancel_cb, video_files=batch_handles,
                est_tokens=max(1, len(prompt) // 4) + video_tokens,
                video_refs=list(covering), reject_truncated=True,
            )
            items = data.get("cues", []) if data else []
            # [v3.23.581] Chỉ nhận nhãn của dòng được yêu cầu — model có thể trả lại cả
            # các dòng tham khảo trong khối <known>.
            batch_ids = {ln.index for ln in batch}
            all_cues.extend(
                cue
                for cue in self._parse_visual_cue_items(items, max_line_no=max_line_no)
                if cue.line_no in batch_ids
            )

            if progress_cb is not None:
                progress_cb((batch_idx + 1) / total_batches)
            # [Chống Rate Limit] Nghỉ giữa các lô, trừ lô cuối.
            # [v3.23.560] _interruptible_sleep trả True khi BỊ HUỶ. Trước đây điều kiện bị
            # đảo (``not``) nên MỌI lần nghỉ bình thường đều ném "đã huỷ" sau lô đầu —
            # quét cues nhiều lô chưa bao giờ chạy hết, và làm đổ cả lượt dịch.
            is_last_batch = batch_idx >= total_batches - 1
            if (
                sleep_between_s > 0
                and not is_last_batch
                and self._interruptible_sleep(sleep_between_s, cancel_cb)
            ):
                raise TranslationCancelledError("Đã huỷ trong lúc nghỉ giữa các lô.")
        return all_cues

    def analyze_text_cues(
        self,
        source_lines: list[TranslationLine],
        target_lang: str,
        model_name: str = "gemini-3.1-flash-lite",
        characters: str = "",
        batch_size: int = TEXT_CUES_BATCH_SIZE,
        cancel_cb: CancellationCallback | None = None,
        progress_cb: StageProgressCallback | None = None,
        sleep_between_s: float = 2.0,
        voice_labels: dict[int, str] | None = None,
        voice_source: str = "",
        source_lang: str = "",
    ) -> list[VisualCue]:
        """[v3.23.623] Gán ai nói / nói với ai cho từng dòng CHỈ từ chữ (không cần video).

        Một lời gọi cho cả tập ngắn (≤ ``TEXT_CUES_FULL_EPISODE_LINES`` dòng): prompt đặt TOÀN
        BỘ phụ đề trước, danh sách dòng cần gán sau, rồi một câu chốt (prompting-strategies:
        dữ liệu trước, câu hỏi cuối). Phim dài hơn được chia lô với cửa sổ phụ đề quanh lô.

        Args:
            source_lines: Toàn bộ dòng phụ đề gốc.
            target_lang: Ngôn ngữ ghi ``cue``.
            model_name: Model dùng để suy luận (có thinking càng tốt).
            characters: Bảng nhân vật để nhãn trùng tên gốc trong bảng.
            batch_size: Số dòng mỗi lô.
            cancel_cb: Huỷ giữa chừng.
            progress_cb: Tiến độ ∈ [0, 1].
            sleep_between_s: Nghỉ giữa các lô (chống 429).
            voice_labels: ``{số dòng: nhãn giọng}`` từ bộ nhận dạng giọng nói (tuỳ chọn). Có
                thì mỗi dòng trong bản chép mang nhãn ``[v:N]`` và prompt dặn gán mỗi nhãn cho
                MỘT nhân vật — tín hiệu ÂM THANH độc lập với việc đoán qua chữ.
            voice_source: Đường dẫn video cạnh phụ đề; có (và chưa có ``voice_labels``) thì tự
                tách giọng bằng :meth:`label_voices`. Lỗi → bỏ qua, gán chỉ bằng chữ.
            source_lang: Mã ngôn ngữ gốc (gợi ý cho bộ nhận dạng giọng nói).

        Returns:
            Danh sách :class:`VisualCue` theo thứ tự dòng.
        """
        self._ensure_available()
        lines = analysis_source_lines(source_lines, keep_repeats=True)
        if not lines:
            return []
        if voice_labels is None and voice_source:
            voice_labels = self.label_voices(voice_source, lines, source_lang, cancel_cb)
        voices = {k: v for k, v in (voice_labels or {}).items() if v}
        config = self._build_config(
            0.2, _TEXT_CUES_SCHEMA,
            _text_cues_system_instruction(target_lang, characters, bool(voices)),
            enable_thinking=True, thinking_budget=-1, model_name=model_name,
            thinking_level=getattr(self, "_analysis_thinking_level", "medium"),
        )

        def _noop_validator(_data: dict[str, Any]) -> None:
            pass

        max_line_no = max(line.index for line in lines)
        positions = {line.index: pos for pos, line in enumerate(lines)}
        batches = [lines[i:i + batch_size] for i in range(0, len(lines), batch_size)]
        all_cues: list[VisualCue] = []
        for batch_idx, batch in enumerate(batches):
            if cancel_cb is not None and cancel_cb():
                raise TranslationCancelledError("Người dùng đã huỷ việc gán người nói.")
            if len(lines) <= TEXT_CUES_FULL_EPISODE_LINES:
                window = lines
            else:
                first = positions[batch[0].index] - TEXT_CUES_WINDOW_LINES
                last = positions[batch[-1].index] + TEXT_CUES_WINDOW_LINES + 1
                window = lines[max(0, first):last]
            def _row(ln: TranslationLine) -> str:
                tag = voices.get(ln.index)
                return f"{ln.index}|[v:{tag}]|{ln.text}" if tag else f"{ln.index}|{ln.text}"

            transcript = "\n".join(_row(ln) for ln in window)
            ids = [ln.index for ln in batch]
            prompt = "\n\n".join([
                _wrap_xml_block("episode_transcript", transcript),
                _wrap_xml_block("lines_to_label", [
                    ({"id": ln.index, "voice": voices[ln.index], "text": ln.text}
                     if ln.index in voices else {"id": ln.index, "text": ln.text})
                    for ln in batch]),
                "Dựa trên toàn bộ tập phim ở trên: trả spk, to, cue cho MỖI id trong "
                f"<lines_to_label> ({len(ids)} dòng: {ids[0]}…{ids[-1]}), theo thứ tự, "
                "không bỏ dòng nào.",
            ])
            data = self._call_gemini(
                model_name, prompt, config, _noop_validator, cancel_cb=cancel_cb,
                est_tokens=max(1, len(prompt) // 4), reject_truncated=True,
            )
            items = data.get("cues", []) if data else []
            wanted = set(ids)
            all_cues.extend(
                cue for cue in self._parse_visual_cue_items(items, max_line_no=max_line_no)
                if cue.line_no in wanted
            )
            if progress_cb is not None:
                progress_cb((batch_idx + 1) / len(batches))
            if (
                sleep_between_s > 0 and batch_idx < len(batches) - 1
                and self._interruptible_sleep(sleep_between_s, cancel_cb)
            ):
                raise TranslationCancelledError("Đã huỷ trong lúc nghỉ giữa các lô.")
        return all_cues

    def audit_analysis(
        self,
        source_lines: list[TranslationLine],
        target_lang: str,
        model_name: str,
        characters: str,
        glossary: str,
        overview: str,
        labels: dict[int, tuple[str, str]],
        defects_json: str,
        focus_ids: list[int],
        cancel_cb: CancellationCallback | None = None,
    ) -> dict[str, Any]:
        """[v3.23.624] Hỏi model BẢN VÁ cho phân tích cũ (không phân tích lại từ đầu).

        Args:
            source_lines: Phụ đề nguồn.
            target_lang: Ngôn ngữ tóm tắt.
            model_name: Model kiểm định.
            characters / glossary / overview: Phân tích hiện tại (đã gồm chỉnh sửa của người dùng).
            labels: ``{số dòng: (người nói, người nghe)}`` hiện có.
            defects_json: Khiếm khuyết tìm được (xem ``defects_to_prompt``).
            focus_ids: Dòng nghi lỗi; tập dài chỉ gửi vùng quanh các dòng này.
            cancel_cb: Huỷ giữa chừng.

        Returns:
            Bản vá (dict theo ``_AUDIT_PATCH_SCHEMA``); rỗng khi model không trả gì.
        """
        self._ensure_available()
        lines = analysis_source_lines(source_lines, keep_repeats=True)
        if len(lines) > AUDIT_FULL_EPISODE_LINES and focus_ids:
            positions = {ln.index: pos for pos, ln in enumerate(lines)}
            keep: set[int] = set()
            for line_id in focus_ids:
                pos = positions.get(line_id)
                if pos is not None:
                    keep.update(range(max(0, pos - AUDIT_WINDOW_LINES),
                                      min(len(lines), pos + AUDIT_WINDOW_LINES + 1)))
            if len(keep) > AUDIT_MAX_PROMPT_LINES:
                # [v3.23.631] Trần cứng: dù người gọi quên chia lô, prompt không phủ cả phim —
                # giữ các dòng GẦN dòng nghi lỗi nhất.
                focus_pos = sorted(
                    positions[i] for i in focus_ids if i in positions
                )
                keep = set(sorted(
                    keep, key=lambda p: (min(abs(p - f) for f in focus_pos), p)
                )[:AUDIT_MAX_PROMPT_LINES])
            lines = [ln for pos, ln in enumerate(lines) if pos in keep] or lines
        elif len(lines) > AUDIT_NO_FOCUS_MAX_LINES:
            # [v3.23.637] Khiếm khuyết cấp bảng (vd thiếu mệnh đề xưng hô) không có dòng cụ thể:
            # phim dài chỉ gửi vài đoạn LIỀN MẠCH rải đều (đủ thấy ai nói với ai), không cả phim.
            chunk = AUDIT_NO_FOCUS_MAX_LINES // AUDIT_NO_FOCUS_CHUNKS
            starts = [
                int(i * (len(lines) - chunk) / max(1, AUDIT_NO_FOCUS_CHUNKS - 1))
                for i in range(AUDIT_NO_FOCUS_CHUNKS)
            ]
            lines = [ln for st in dict.fromkeys(starts) for ln in lines[st:st + chunk]]

        def _row(ln: TranslationLine) -> str:
            spk, to = labels.get(ln.index, ("", ""))
            tag = f"[{spk}→{to}]" if (spk or to) else "[?]"
            return f"{ln.index}|{tag}|{ln.text}"

        prompt = "\n\n".join([
            _wrap_xml_block("phu_de_goc", "\n".join(_row(ln) for ln in lines)),
            _wrap_xml_block("bang_nhan_vat_hien_tai", characters or "(rỗng)"),
            _wrap_xml_block("bang_thuat_ngu_hien_tai", glossary or "(rỗng)"),
            _wrap_xml_block("tom_tat_hien_tai", overview or "(rỗng)"),
            _wrap_xml_block("khiem_khuyet", defects_json),
            "Đối chiếu bản phân tích hiện tại với phụ đề gốc và trả BẢN VÁ cho các khiếm khuyết "
            "đã nêu (cùng mọi lỗi rõ ràng khác bạn thấy). Mục đúng thì không nhắc tới.",
        ])
        config = self._build_config(
            0.2, _AUDIT_PATCH_SCHEMA, _audit_system_instruction(target_lang),
            enable_thinking=True, thinking_budget=-1, model_name=model_name,
            # [v3.23.631] Mức riêng cho kiểm định (không theo cài đặt phân tích "high": 16-30 nghìn
            # token suy nghĩ + 504 ở mốc 300 giây trong log thật); "medium" là điểm tối ưu đo được.
            thinking_level=AUDIT_THINKING_LEVEL,
        )
        data = self._call_gemini(
            model_name, prompt, config, lambda _d: None, cancel_cb=cancel_cb,
            est_tokens=max(1, len(prompt) // 4), reject_truncated=True,
        )
        return data if isinstance(data, dict) else {}

    def shorten_for_speech(
        self,
        items: list[dict[str, Any]],
        target_lang: str,
        model_name: str,
        cancel_cb: CancellationCallback | None = None,
        sleep_between_s: float = 1.0,
    ) -> dict[int, str]:
        """[v3.23.629] Viết lại các câu dịch NGẮN HƠN để TTS đọc kịp khung (không dịch lại).

        Args:
            items: Mỗi phần tử ``{"id", "nguon", "ban_dich", "toi_da_am_tiet", "nguoi_noi",
                "nguoi_nghe", "truoc", "sau"}`` (các khoá tuỳ chọn có thể vắng).
            target_lang: Ngôn ngữ của bản dịch.
            model_name: Model Gemini.
            cancel_cb: Huỷ giữa chừng.
            sleep_between_s: Nghỉ giữa các lô (chống 429).

        Returns:
            ``{id: câu mới}`` cho các id model trả về (use case kiểm hợp lệ từng câu).
        """
        self._ensure_available()
        if not items:
            return {}
        config = self._build_config(
            0.3, _SHORTEN_SCHEMA, _shorten_system_instruction(target_lang),
            enable_thinking=True, thinking_budget=-1, model_name=model_name,
            thinking_level="low",
        )
        wanted_all: dict[int, str] = {}
        batches = [items[i:i + SHORTEN_BATCH_SIZE] for i in range(0, len(items), SHORTEN_BATCH_SIZE)]
        for batch_idx, batch in enumerate(batches):
            if cancel_cb is not None and cancel_cb():
                raise TranslationCancelledError("Người dùng đã huỷ việc rút gọn câu.")
            prompt = "\n\n".join([
                _wrap_xml_block("cac_dong_can_rut_gon", batch),
                "Viết lại NGẮN HƠN từng dòng trong <cac_dong_can_rut_gon> theo đúng giới hạn "
                f"'toi_da_am_tiet' ({len(batch)} dòng, id: {[b['id'] for b in batch]}).",
            ])
            data = self._call_gemini(
                model_name, prompt, config, lambda _d: None, cancel_cb=cancel_cb,
                est_tokens=max(1, len(prompt) // 4), reject_truncated=True,
            )
            ids = {int(b["id"]) for b in batch}
            for row in (data or {}).get("lines", []) if isinstance(data, dict) else []:
                try:
                    line_id = int(row.get("id"))
                except (TypeError, ValueError, AttributeError):
                    continue
                text = str(row.get("text") or "").strip()
                if line_id in ids and text:
                    wanted_all[line_id] = text
            if (
                sleep_between_s > 0 and batch_idx < len(batches) - 1
                and self._interruptible_sleep(sleep_between_s, cancel_cb)
            ):
                raise TranslationCancelledError("Đã huỷ trong lúc nghỉ giữa các lô.")
        return wanted_all

    def label_voices(
        self,
        video_path: str,
        source_lines: list[TranslationLine],
        source_lang: str = "",
        cancel_cb: CancellationCallback | None = None,
    ) -> dict[int, str]:
        """[v3.23.623] Nhãn giọng nói (cùng nhãn = cùng người) cho từng dòng, từ ÂM THANH video.

        Tách riêng âm thanh lời thoại, nhờ ``gemini-3.5-transcribe`` tách giọng, ghép từ nói
        vào dòng theo mốc thời gian. Kết quả tách giọng được nhớ theo (đường dẫn, kích thước,
        ngày sửa) nên dịch lại / dịch tiếp không tốn lời gọi. Mọi lỗi (thiếu ffmpeg, video
        không có tiếng, hết hạn mức, âm thanh quá 28 phút) → ``{}``: tính năng tuỳ chọn, luồng
        chính không bị ảnh hưởng.

        Args:
            video_path: Video có tiếng cạnh phụ đề.
            source_lines: Dòng phụ đề (mốc mili-giây là mốc trên video này).
            source_lang: Mã ngôn ngữ gốc để gợi ý bộ nhận dạng.
            cancel_cb: Huỷ giữa chừng.

        Returns:
            ``{số dòng: nhãn giọng}``; dòng không có từ nói chồng lấn thì vắng mặt.
        """
        import tempfile
        from pathlib import Path

        from subtitles_extractor.infrastructure.translation import gemini_diarization as diar
        from subtitles_extractor.infrastructure.translation.gemini_video_context import (
            GeminiVideoContextProvider,
        )

        path = Path(str(video_path or ""))
        if not path.is_file() or not source_lines:
            return {}
        words = diar.load_cached_words(path)
        if words is None:
            try:
                duration = GeminiVideoContextProvider._read_duration_sec(path)
            except Exception as exc:  # noqa: BLE001 — không đọc được thời lượng: bỏ qua tính năng
                logger.warning("Không đọc được thời lượng video để tách giọng: %s", exc)
                return {}
            if duration > diar.MAX_DIARIZED_AUDIO_SEC:
                logger.info(
                    "Video %.0f phút vượt giới hạn tách giọng (%d phút) — gán người nói bằng "
                    "chữ.", duration / 60, diar.MAX_DIARIZED_AUDIO_SEC // 60,
                )
                return {}
            with tempfile.TemporaryDirectory(prefix="se_voice_") as tmp:
                audio = Path(tmp) / "voices.m4a"
                if not diar.extract_dialogue_audio(path, audio):
                    return {}
                words = self._diarize_with_keys(audio, source_lang, cancel_cb)
            if not words:
                return {}
            diar.save_cached_words(path, words)
        labels = diar.voice_labels_from_words(words, source_lines)
        logger.info(
            "Tách giọng: %d cụm giọng, %d/%d dòng có nhãn giọng.",
            len({w.speaker for w in words}), len(labels), len(source_lines),
        )
        return labels

    def _diarize_with_keys(
        self, audio: Any, source_lang: str, cancel_cb: CancellationCallback | None
    ) -> list[Any]:
        """Gọi tách giọng lần lượt theo từng API key (xoay key khi hết hạn mức)."""
        from subtitles_extractor.infrastructure.translation import gemini_diarization as diar

        self._ensure_available()
        timeout_ms = int(max(self._request_timeout_s, 120.0) * 1000)
        keys = list(self._api_keys) or [self._api_key]
        for key in keys:
            if cancel_cb is not None and cancel_cb():
                raise TranslationCancelledError("Người dùng đã huỷ việc tách giọng.")
            try:
                client = self._genai_module.Client(
                    api_key=key, http_options={"timeout": timeout_ms})
                return diar.diarize_audio(
                    client, self._types_module, audio, diar.language_codes_for(source_lang))
            except Exception as exc:  # noqa: BLE001 — SDK ném nhiều loại; phân loại bên dưới
                logger.warning("Tách giọng thất bại với một key: %s", self._concise_error(exc))
                if not (self._is_rate_limit_error(exc) or self._is_daily_quota_error(exc)):
                    break
        return []

    def list_available_models(self) -> list[str]:
        """Lấy danh sách model từ Gemini API với filter linh hoạt.

        Tương thích nhiều phiên bản SDK:
        * SDK cũ:  ``model.supported_generation_methods`` = ``["generateContent", ...]``
        * SDK mới: attribute có thể là ``supported_actions`` hoặc không có.

        Fallback: nếu không tìm thấy methods, bao gồm model có "gemini" trong tên
        (đây là tất cả những gì người dùng cần cho dịch phụ đề).
        """
        if not self.is_available():
            return []
        try:
            client = self._get_client()
            result: list[str] = []
            all_count = 0

            for m in client.models.list():
                all_count += 1
                raw_name: str = getattr(m, "name", "") or ""
                if not raw_name:
                    continue
                model_id = (
                    raw_name.removeprefix("models/")
                    if raw_name.startswith("models/")
                    else raw_name
                )

                # Thử nhiều tên attribute khác nhau (tương thích SDK mọi phiên bản)
                methods: list[str] = (
                    getattr(m, "supported_generation_methods", None)
                    or getattr(m, "supported_actions", None)
                    or []
                )
                methods_lower = [str(x).lower() for x in (methods or [])]

                supports_generate = any(
                    "generatecontent" in x or "generate_content" in x
                    for x in methods_lower
                )
                # Fallback: mọi model có "gemini" đều có thể dùng cho dịch
                is_gemini_model = "gemini" in model_id.lower()

                if supports_generate or is_gemini_model:
                    if self._is_free_tier_text_model(model_id):
                        result.append(model_id)
                        logger.debug(
                            "Lấy model: %s | methods=%s", model_id,
                            methods_lower or "(không có)",
                        )
                    else:
                        logger.debug(
                            "Bỏ qua model không free/không phải text-out: %s", model_id,
                        )
                else:
                    logger.debug(
                        "Bỏ qua model: %s | methods=%s", model_id, methods_lower or "(không có)"
                    )

            logger.info(
                "list_available_models: tổng %d model từ API → %d model free tier được chọn.",
                all_count, len(result),
            )
            if self._model_health is not None:
                self._model_health.set_known_models(result)
            return sorted(result)

        except Exception as exc:
            logger.warning("Không lấy được danh sách model Gemini: %s", exc)
            return []

    @staticmethod
    def _is_free_tier_text_model(model_id: str) -> bool:
        """[v3.23.36] Lọc model dùng được cho dịch ở BẢN MIỄN PHÍ (text-out).

        Loại bỏ: model *-pro (2.5-pro, 3.1-pro… quota free = 0/0), các bản đời cũ
        (gemini-2.0-*, gemini-2-*, 1.0/1.5 — đã ngừng free), và model KHÔNG sinh văn
        bản (embedding, aqa, imagen, veo, tts, image, vision-only, learnlm…).
        Chỉ giữ dòng *-flash và *-flash-lite thế hệ 2.5 trở lên.
        """
        name = model_id.lower()
        # Loại model không sinh text.
        non_text = ("embedding", "aqa", "imagen", "veo", "-tts", "image", "learnlm",
                    "gemma", "-exp", "thinking-exp")
        if any(tok in name for tok in non_text):
            return False
        # Loại *-pro (free = 0/0 theo bảng quota).
        if "pro" in name:
            return False
        # Loại đời cũ không còn free (2.0, 2-flash không có '.', 1.0, 1.5).
        if "gemini-2.0" in name or "gemini-1." in name:
            return False
        if name.startswith("gemini-2-flash") or name == "gemini-2-flash":
            return False
        # Giữ các dòng flash / flash-lite thế hệ 2.5+ và 3.x.
        return "flash" in name

    # [v3.23.222] Ngân sách ký tự nay tính bằng MÔ HÌNH VẬT LÝ đo được ở tầng TTS
    # (``timing_math.readable_char_budget``), thay cho giả định "tốc độ đọc là hằng số
    # 16 ký tự/giây" — vốn sai bản chất vì mỗi câu có chi phí cố định ~0.3s (lấy hơi,
    # đuôi âm). Xem docstring của ``readable_char_budget`` để biết bằng chứng đo đạc.
    def _length_hint(self, line: TranslationLine, next_start_ms: int | None = None) -> int:
        """Số ký tự tối đa NÊN dùng để TTS đọc kịp; 0 nếu không rõ mốc thời gian.

        Args:
            line: Dòng phụ đề cần dịch.
            next_start_ms: Mốc bắt đầu dòng KẾ TIẾP (ms); None nếu là dòng cuối hoặc
                không biết. Có giá trị này thì ngân sách tính theo KHUNG HIỆU DỤNG (khung
                gốc + phần gap tới câu sau mà TTS thật sự được dùng) -> không ép model
                cắt nghĩa ở những dòng vốn có khoảng lặng rộng phía sau.

        Returns:
            Ngân sách ký tự (>= ``MIN_CHAR_BUDGET``), hoặc 0 nếu mốc thời gian không hợp lệ.
        """
        if line.end_ms - line.start_ms <= 0:
            return 0
        if next_start_ms is None:
            # KHÔNG biết dòng kế tiếp (dòng cuối batch / payload ngữ cảnh): chỉ dùng khung
            # GỐC. Không được mượn ``max_gap_use_s`` như câu cuối phim — sẽ phồng ngân sách
            # sai (đo: khung 0.64s -> 46 ký tự thay vì 6) và làm model dịch dài quá khung.
            usable_s = (line.end_ms - line.start_ms) / 1000.0
        else:
            usable_s = effective_available_seconds(
                line.start_ms / 1000.0,
                line.end_ms / 1000.0,
                next_start_ms / 1000.0,
            )
        # [v3.23.238] Ngân sách tính theo ÂM TIẾT (đúng vật lý của tiếng Việt đơn âm tiết)
        # rồi quy sang ký tự để gửi model. Trước đây tính thẳng theo ký tự — sai đơn vị,
        # vì số ký tự mỗi âm tiết dao động 2-6 (đo: mô hình ký tự chỉ đạt R²=0.07, mô hình
        # âm tiết biên dưới đạt R²=0.89).
        #
        # Sàn ``MIN_CHAR_BUDGET`` vẫn giữ: dòng có khung dưới SÀN VẬT LÝ của engine (vd
        # 0.20s) thì ép model cắt ngắn cũng VÔ ÍCH — audio vẫn tràn. Cắt nghĩa mà không
        # được gì là mất trắng.
        tts_budget = syllable_budget_to_chars(readable_syllable_budget(usable_s))
        # [v3.23.439] Dịch CPS-aware: khi bật (>0), siết thêm theo tốc độ MẮT ĐỌC. Lấy MIN của
        # ngân sách TTS và ngân sách đọc → phụ đề vừa nói kịp vừa đọc kịp. Giữ SÀN
        # MIN_CHAR_BUDGET (dòng khung siêu ngắn ép cắt cũng vô ích). Cơ chế bảo toàn nghĩa
        # (rule 'KHÔNG bỏ từ mang nghĩa') vẫn giữ nguyên — chỉ hạ trần độ dài, không ép bỏ ý.
        reading_budget = reading_char_budget(usable_s, getattr(self, "_reading_cps_budget", 0.0))
        budget = tts_budget
        if reading_budget > 0:
            budget = min(tts_budget, reading_budget)
        return max(MIN_CHAR_BUDGET, budget)

    def line_char_budgets(self, lines: list[TranslationLine]) -> dict[int, int]:
        """[v3.23.605] Ngân sách ``max_chars`` của từng dòng — đúng con số gửi cho model.

        Mốc dòng kế tiếp lấy theo thứ tự thời gian của cả danh sách (khung hiệu dụng), như
        mọi dòng không đứng cuối lô khi dịch.

        Args:
            lines: Dòng phụ đề có mốc thời gian.

        Returns:
            ``{index: max_chars}``; dòng có mốc thời gian không hợp lệ thì vắng mặt.
        """
        ordered = sorted(lines, key=lambda line: (line.start_ms, line.index))
        next_starts: list[int | None] = [line.start_ms for line in ordered[1:]] + [None]
        budgets: dict[int, int] = {}
        for line, next_start_ms in zip(ordered, next_starts, strict=True):
            hint = self._length_hint(line, next_start_ms)
            if hint:
                budgets[line.index] = hint
        return budgets

    def _line_to_payload(
        self, line: TranslationLine, next_start_ms: int | None = None
    ) -> dict[str, Any]:
        data: dict[str, Any] = {"line_no": line.index, "text": line.text}
        if line.speaker:
            data["speaker"] = line.speaker
        # [v3.23.88] Gửi kèm addressee + cue (nghe-nhìn) để KHÂU LITERAL chọn đúng xưng
        # hô/giọng văn ngay từ đầu — trước đây prompt có nhắc 'addressee'/'cue' nhưng
        # payload lại bỏ, gây bất nhất.
        if line.addressee:
            data["addressee"] = line.addressee
        if line.scene:
            data["cue"] = line.scene
        if line.description:
            data["description"] = line.description
        hint = self._length_hint(line, next_start_ms)
        if hint:
            data["max_chars"] = hint
        return data

    def _attach_address_rules_once(
        self, payload: list[dict[str, Any]], batch: list[TranslationLine]
    ) -> None:
        """[v3.23.551] Gắn quy tắc xưng hô vào dòng ĐẦU TIÊN của mỗi cặp (nói → nghe) trong lô.

        Chỉ gắn một lần mỗi cặp mỗi lô: đo trên phim thật, gắn mọi dòng thêm ~11% ký tự prompt
        (đang có lỗi quota token/phút), còn gắn lần đầu mỗi cặp chỉ tốn ~2% — model vẫn thấy quy
        tắc ngay cạnh các dòng của cặp đó. Dòng ngữ cảnh (trước/sau/lịch sử) KHÔNG được gắn.
        """
        from subtitles_extractor.application.services.monologue_guard import (
            MONOLOGUE_ADDRESSEE,
            MONOLOGUE_RULE,
            is_self_addressed,
        )

        seen: set[tuple[str, str]] = set()
        pair_rules: dict[tuple[str, str], str] = {}
        for data, line in zip(payload, batch, strict=False):
            if not line.speaker or not line.addressee:
                continue
            if is_self_addressed(line.speaker, line.addressee):
                # [v3.23.577] Độc thoại: "addressee = chính mình" khiến model mang cách
                # xưng của câu thoại trước ("Nhưng em có truyền thừa…" khi đang một mình).
                # Ghi rõ không có người nghe, kèm quy tắc tự xưng — một lần mỗi lô.
                data["addressee"] = MONOLOGUE_ADDRESSEE
                if (line.speaker, MONOLOGUE_ADDRESSEE) not in seen:
                    data["address"] = MONOLOGUE_RULE
                    seen.add((line.speaker, MONOLOGUE_ADDRESSEE))
                continue
            key = (line.speaker, line.addressee)
            # [v3.23.623] Phép gán 我/你 cho MỌI dòng của cặp (quy tắc đầy đủ vẫn chỉ ở dòng đầu).
            if key not in pair_rules:
                probe: dict[str, Any] = {}
                self._attach_address_rule(probe, line)
                pair_rules[key] = probe.get("address", "")
            rule = pair_rules[key]
            if rule:
                gloss = address_pronoun_gloss(
                    rule,
                    getattr(line, "original_text", "") or getattr(line, "text", "") or "")
                if gloss:
                    data["pron"] = gloss
            if key in seen:
                continue
            if rule:
                data["address"] = rule
                seen.add(key)

    def _attach_address_rule(self, data: dict[str, Any], line: TranslationLine) -> None:
        """[v3.23.551] Gắn quy tắc xưng hô đã tra sẵn cho cặp (người nói → người nghe) của dòng.

        Vì sao: khâu dịch nhận ``speaker``/``addressee`` nhưng phải TỰ TRA quy tắc trong bảng nhân
        vật dài. Chuẩn đoán thật: roster quy định ``张依贝`` với Tống Tiền "tự xưng 'em' gọi 'anh'"
        mà dòng 601-655 vẫn ra "tôi"/"cậu" — CÙNG MỘT CẢNH. Đặt quy tắc NGAY TRÊN dòng cần dùng
        (dữ liệu tại chỗ) thay vì trông model tự tra — cùng nguyên tắc "schema hơn chỉ thị".
        """
        if not line.speaker or not line.addressee:
            return
        characters = getattr(self, "_active_characters", "") or ""
        if not characters:
            return
        cached_key = getattr(self, "_address_rules_source", None)
        if cached_key != characters:
            from subtitles_extractor.application.services.roster_format import address_rule_map

            try:
                self._address_rules = address_rule_map(characters)
            except (ValueError, TypeError, AttributeError):
                self._address_rules = {}
            self._address_rules_source = characters
        from subtitles_extractor.application.services.roster_format import lookup_address_rule

        rule = lookup_address_rule(
            getattr(self, "_address_rules", {}), line.speaker, line.addressee
        )
        if rule:
            data["address"] = rule

    def _line_to_dual_payload(
        self,
        line: TranslationLine,
        next_start_ms: int | None = None,
        needs_expansion: bool = False,
    ) -> dict[str, Any]:
        """[Anti-Chinese Whispers + Silent Context Injection] Payload TRUYỀN KÉP.

        Gửi cho AI CẢ bản nháp (``text``) LẪN bản gốc (``original``) cùng các gợi ý
        hình ảnh (người nói, nói-với-ai). Nhờ vậy khâu tinh chỉnh/bản địa hoá đối
        chiếu được với bản gốc (không tam sao thất bản) và "thấy" bối cảnh qua chữ
        viết mà không tốn token video.
        """
        data: dict[str, Any] = {"line_no": line.index, "text": line.text}
        if line.original_text and line.original_text != line.text:
            data["original"] = line.original_text
        if line.speaker:
            data["speaker"] = line.speaker
        if line.addressee:
            data["addressee"] = line.addressee
        if line.scene:
            data["cue"] = line.scene
        if line.description:
            data["description"] = line.description
        hint = self._length_hint(line, next_start_ms)
        if hint:
            data["max_chars"] = hint
        # [v3.23.225] Cờ TỰ SỬA: dòng bị lưới ``under_translation_guard`` nghi DỊCH THIẾU
        # ở khâu TRƯỚC. Giai đoạn này DÙ SAO CŨNG gọi API và DÙ SAO CŨNG có 'original'
        # trong payload kép -> tận dụng để BỔ SUNG phần bị rớt, không tốn thêm lượt gọi.
        if needs_expansion:
            data["needs_expansion"] = True
        return data

    def _context_window(
        self,
        lines: list[TranslationLine], start_idx: int, batch_len: int, ctx_size: int
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if ctx_size <= 0:
            return [], []
        before = lines[max(0, start_idx - ctx_size):start_idx]
        after_start = start_idx + batch_len
        after = lines[after_start:after_start + ctx_size]
        to_payload = self._line_to_payload
        return [to_payload(item) for item in before], [to_payload(item) for item in after]

    def _validate_batch(self, payload: dict[str, Any], expected_count: int, start_idx: int) -> None:
        # [v3.23.9] Validate theo TẬP line_no, không theo vị trí.
        # [v3.23.11] Phân biệt thiếu ÍT (vá cửa sổ) vs thiếu NHIỀU (halving).
        # [v3.23.18] Một số model yếu (flash-lite) ĐÁNH SỐ LẠI line_no từ 1 cho mỗi
        # batch, hoặc trả dư dòng. Nếu SỐ LƯỢNG item khớp xấp xỉ expected (±0), ta
        # CHẤP NHẬN và để _merge_ai_items ghép THEO VỊ TRÍ (model giữ đúng thứ tự,
        # chỉ sai cách đánh số). Tránh halving vô hạn khi model không tuân thủ line_no.
        items = payload.get("subtitles", []) if payload else []
        actual_count = len(items)

        expected_line_nos = {start_idx + offset + 1 for offset in range(expected_count)}

        # [v3.23.18] Model trả DƯ nhiều dòng (vd 95 cho batch 25) = sai cấu trúc thật
        # (model phớt lờ batch, trả cả phụ đề/ngữ cảnh).
        # [v3.23.512] NHƯNG phần dư thường chỉ là các dòng NGỮ CẢNH bị dịch kèm, và chúng có
        # line_no NẰM NGOÀI khoảng của lô. Log thật cho thấy khi lô bị chia còn 1 dòng, model trả
        # 26 dòng (1 dòng cần + 25 dòng ngữ cảnh) → bị coi là hỏng, thử lại 5 lần, chia nhỏ tiếp,
        # cuối cùng BỎ CUỘC giữ nguyên văn bản gốc — dù dòng cần dịch NẰM SẴN trong kết quả.
        # Nay lọc đúng các dòng thuộc lô; chỉ báo hỏng khi lọc xong vẫn KHÔNG đủ.
        if actual_count > expected_count + 2:
            in_range = [
                item
                for item in items
                if _safe_int(item.get("line_no", 0)) in expected_line_nos
                and str(item.get("text", "")).strip()
            ]
            covered = {_safe_int(item.get("line_no", 0)) for item in in_range}
            # [v3.23.576] Thiếu ÍT dòng của lô (thường là dòng rác OCR model để trống, vd
            # "L上上") thì phần còn lại vẫn dùng được: lọc rồi để nhánh "thiếu" phía dưới
            # VÁ đúng các dòng đó. Nhật ký thật: lô 517-529 trả 60 mục, đủ 12/13 dòng, chỉ
            # dòng 523 rỗng → bị coi là sai cấu trúc, gọi lại 5 lần rồi chia đôi lô.
            few_missing = len(expected_line_nos - covered) <= _patch_threshold(
                expected_count
            )
            if covered == expected_line_nos or (in_range and few_missing):
                logger.info(
                    "Model trả dư %d dòng (kỳ vọng %d), có %d/%d dòng của lô → lọc bỏ "
                    "phần ngữ cảnh thừa, chỉ vá dòng còn thiếu (nếu có).",
                    actual_count, expected_count, len(covered), expected_count,
                )
                payload["subtitles"] = in_range
                items = in_range
                actual_count = len(items)
            else:
                raise _BatchValidationError(
                    f"Model trả DƯ {actual_count} dòng (kỳ vọng {expected_count}) — "
                    "sai cấu trúc, cần chia nhỏ."
                )

        def _is_blank(item: dict[str, Any]) -> bool:
            return not str(item.get("text", "")).strip()

        # [v3.23.80] CHỈ coi là "đã có" khi line_no hợp lệ VÀ text KHÔNG rỗng. Dòng có
        # line_no nhưng text rỗng = chưa dịch thật → đưa vào tập "thiếu" để VÁ lại kèm
        # ngữ cảnh, thay vì âm thầm giữ nguyên văn bản gốc (dòng chưa dịch).
        actual_line_nos = {
            _safe_int(item.get("line_no"), -1)
            for item in items
            if _safe_int(item.get("line_no"), -1) > 0 and not _is_blank(item)
        }
        missing = expected_line_nos - actual_line_nos

        if not missing:
            return

        # [v3.23.128] Phân biệt 2 tình huống để TRÁNH dồn lệch (lỗi nặng nhất):
        #  (a) Model dùng ĐÚNG line_no, chỉ thiếu vài dòng (đầu/giữa/cuối) → VÁ theo
        #      line_no: an toàn tuyệt đối, không xê dịch dòng nào, bất kể batch lớn/nhỏ.
        #  (b) Model ĐÁNH SỐ LỆCH (line_no không thuộc tập kỳ vọng, vd lại từ 1):
        #        - ĐỦ số lượng  → ghép theo VỊ TRÍ (an toàn vì cùng thứ tự).
        #        - THIẾU số lượng → KHÔNG ghép vị trí (không biết thiếu ở đâu → dòng sau
        #          dồn lên thế chỗ dòng trước → sai phụ đề hàng loạt). Chia đôi để model
        #          trả lại chuẩn hơn.
        uses_correct_linenos = (
            bool(actual_line_nos) and actual_line_nos <= expected_line_nos
        )
        missing_sorted = sorted(missing)
        if uses_correct_linenos:
            # Thiếu ÍT (≤ max(2, 15%)) → vá theo line_no; thiếu NHIỀU → chia đôi
            # (dịch lại đáng tin hơn vá lắt nhắt quá nhiều).
            patch_threshold = _patch_threshold(expected_count)
            if len(missing) <= patch_threshold:
                raise _BatchPartialError(payload, missing_sorted)
            raise _BatchValidationError(
                f"Model thiếu {len(missing)} dòng (vd line_no {missing_sorted[:5]}); "
                f"kỳ vọng {expected_count} dòng, nhận {actual_count}."
            )

        # Model đánh số LỆCH.
        if actual_count == expected_count:
            logger.info(
                "Model trả đủ %d dòng, line_no lệch (từ %d) — ghép theo VỊ TRÍ.",
                actual_count, start_idx + 1,
            )
            self._renumber_items_by_position(items, start_idx)
            patch_needed = {
                start_idx + i + 1 for i, item in enumerate(items) if _is_blank(item)
            }
            if patch_needed:
                raise _BatchPartialError(payload, sorted(patch_needed))
            return

        # Lệch + thiếu → chia đôi (TUYỆT ĐỐI không dồn vị trí).
        raise _BatchValidationError(
            f"Model thiếu {len(missing)} dòng (vd line_no {missing_sorted[:5]}); "
            f"kỳ vọng {expected_count} dòng, nhận {actual_count}."
        )

    @staticmethod
    def _renumber_items_by_position(items: list[dict[str, Any]], start_idx: int) -> None:
        """Gán lại line_no theo VỊ TRÍ khi model đánh số lệch nhưng đủ & đúng thứ tự.

        Sửa tại chỗ: item thứ i nhận line_no = start_idx + i + 1. Nhờ đó
        _merge_ai_items (ghép theo line_no) ghép đúng câu gốc.
        """
        for offset, item in enumerate(items):
            item["line_no"] = start_idx + offset + 1

    # ── Điểm vào chính của port ──────────────────────────────────────────
    @staticmethod
    def _noise_line_indices(
        source_lines: list[TranslationLine], context: TranslationContext
    ) -> set[int]:
        """Index các dòng "rác" cần GIỮ NGUYÊN (không gửi model dịch).

        Với nguồn CJK, dòng KHÔNG chứa bất kỳ ký tự CJK nào (vd OCR đọc nhầm khung hình ra
        chuỗi Latin lộn xộn "akas") gần như chắc chắn là nhiễu OCR. Gửi chúng cho model dễ bị
        GỘP/BỎ -> lệch dòng dây chuyền. Giữ nguyên văn để vừa chống lệch vừa trung thực với OCR.

        Chỉ kích hoạt khi ngôn ngữ nguồn thuộc nhóm CJK; ngôn ngữ Latin không áp dụng (mọi dòng
        đều không có ký tự CJK -> sẽ lọc nhầm).

        [v3.23.133] BỔ SUNG (mọi ngôn ngữ): dòng chỉ gồm ký hiệu/nhạc (♪), số, dấu câu —
        KHÔNG có chữ nào — cũng giữ nguyên (không cần dịch, tránh model trả rỗng).
        """
        src_lang = (context.source_lang or "").strip().lower()
        # [v3.23.133] MỌI ngôn ngữ: dòng KHÔNG chứa CHỮ nào (chỉ ký hiệu nhạc ♪, dấu câu,
        # số) thì giữ NGUYÊN — không gửi model. Model (nhất là bản lite) hay trả RỖNG cho
        # các dòng "♪ ♪" → kích hoạt "nuốt chữ" + vá thừa (tốn thời gian, đôi khi lệch).
        # \w trong Python 3 khớp cả chữ Unicode (Latin có dấu, CJK…) nên [^\W\d_] = "một
        # chữ bất kỳ"; không tìm thấy chữ nào ⇒ dòng chỉ ký hiệu/số ⇒ không cần dịch.
        noise = {
            ln.index
            for ln in source_lines
            if (ln.text or "").strip() and re.search(r"[^\W\d_]", ln.text) is None
        }
        if any(src_lang.startswith(p) for p in _CJK_LANG_PREFIXES):
            cjk_noise = {
                ln.index
                for ln in source_lines
                if (ln.text or "").strip() and not _CJK_CHAR_RE.search(ln.text)
            }
            # [v3.23.350] CHỐNG TỰ-BẮN-NHẦM: nếu nhánh CJK coi (gần) TOÀN BỘ dòng là
            # rác thì nhãn ngôn ngữ nguồn gần như chắc SAI (vd đặt 'zh' nhưng phụ đề
            # thực tế là tiếng Anh). Trước đây cứ áp máy móc → translate_stage lọc sạch
            # input rồi ``return list(full_input_lines)`` → TRẢ NGUYÊN BẢN GỐC trong
            # 0.0s, KHÔNG gọi API, KHÔNG báo lỗi ("dịch mà y hệt bản gốc"). Khắc phục:
            # chỉ áp nhánh CJK khi nó KHÔNG nuốt quá ``_CJK_NOISE_MAX_RATIO`` số dòng
            # có nội dung; ngược lại BỎ QUA nhánh này (để nội dung được dịch bình thường)
            # và cảnh báo rõ nguyên nhân để người dùng sửa nhãn ngôn ngữ.
            non_empty_count = sum(
                1 for ln in source_lines if (ln.text or "").strip()
            )
            if (
                non_empty_count > 0
                and len(cjk_noise) / non_empty_count >= _CJK_NOISE_MAX_RATIO
            ):
                logger.warning(
                    "Ngôn ngữ nguồn đặt '%s' (nhóm CJK) nhưng %d/%d dòng KHÔNG có ký "
                    "tự CJK → nhiều khả năng SAI NHÃN ngôn ngữ (phụ đề thực tế không "
                    "phải CJK). BỎ QUA bộ lọc 'dòng rác CJK' để vẫn dịch bình thường; "
                    "hãy kiểm tra lại 'Ngôn ngữ gốc' cho đúng với nội dung phụ đề.",
                    src_lang, len(cjk_noise), non_empty_count,
                )
            else:
                noise |= cjk_noise
        return noise

    def _batch_context(
        self,
        source_lines: list[TranslationLine],
        start: int,
        batch_len: int,
        ctx_size: int,
        prior_output: list[TranslationLine],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        """[v3.23.149] Gói ngữ cảnh cho MỘT batch: nguồn trước/sau + lịch sử đã dịch.

        Dùng chung cho đường tuần tự lẫn song song (DRY). ``prior_output`` là danh
        sách dòng ĐÃ dịch dùng làm lịch sử: tuần tự truyền output động; song song
        truyền phần output đã CHỐT của các đợt trước (neo lịch sử theo đợt).
        """
        source_before, source_after = self._context_window(
            source_lines, start, batch_len, ctx_size
        )
        history_before = (
            [self._line_to_payload(line) for line in prior_output[-ctx_size:]]
            if ctx_size > 0 else []
        )
        return source_before, source_after, history_before

    def _effective_parallel(self, requested: int, model_name: str) -> int:
        """[v3.23.149] Mức song song HIỆU DỤNG = min(yêu cầu, RPM limit của model).

        Dự đoán trước khi mở luồng: mở nhiều luồng hơn RPM/phút của model là vô ích
        (quota manager sẽ bắt chờ) — chủ động hạ để không tốn tài nguyên vô nghĩa.
        """
        requested = max(1, min(int(requested or 1), 4))
        if requested <= 1 or self._quota_manager is None:
            return requested
        try:
            rpm_limit = int(
                self._quota_manager.get_remaining(
                    model_name, key_id=self._current_key_id()
                ).get("rpm_limit", requested)
            )
        except (AttributeError, TypeError, ValueError):
            return requested
        return max(1, min(requested, rpm_limit))

    def _run_batches_parallel(
        self,
        *,
        parallel: int,
        input_lines: list[TranslationLine],
        source_lines: list[TranslationLine],
        batch_size: int,
        ctx_size: int,
        total_batches: int,
        stage: TranslationStageConfig,
        config: Any,
        is_literal: bool,
        is_preprocess: bool,
        use_dual_payload: bool,
        progress_cb: StageProgressCallback | None,
        cancel_cb: CancellationCallback | None,
        expansion_indices: frozenset[int] = frozenset(),
        resume: BatchResume | None = None,
        batch_done_cb: BatchDoneCallback | None = None,
    ) -> list[TranslationLine]:
        """[v3.23.149] Dịch các batch SONG SONG theo ĐỢT (wave) có neo lịch sử.

        - Mỗi đợt gồm ``parallel`` batch chạy đồng thời; ``history_before`` của MỌI
          batch trong đợt neo vào đuôi output các đợt ĐÃ CHỐT (giữ mạch xưng hô/giọng
          điệu; ngữ cảnh nguồn trước/sau vẫn đầy đủ theo từng batch).
        - Quota an toàn khi song song: mỗi request tự ``acquire`` (đặt chỗ RPM/TPM,
          thread-safe) nên tổng lưu lượng luôn được điều tiết đúng; cooldown 429 áp
          cho mọi luồng cùng key; xoay key được ``_state_lock`` bảo vệ.
        - Kết quả ghép ĐÚNG THỨ TỰ batch (không phụ thuộc thứ tự hoàn thành).
        - [v3.23.584] Lô đã dịch ở lần chạy dở (``resume``) dùng lại, không gửi; lô dịch
          xong được báo qua ``batch_done_cb`` — kể cả khi một lô khác trong đợt thất bại.

        Raises:
            TranslationCancelledError: người dùng huỷ giữa chừng.
            SubtitleTranslationError: một batch thất bại sau mọi retry.
        """
        starts = list(range(0, len(input_lines), batch_size))
        output: list[TranslationLine] = []
        done_batches = 0
        logger.info(
            "Dịch SONG SONG giai đoạn '%s': %d batch, %d luồng/đợt (neo lịch sử theo đợt).",
            stage.kind.value, total_batches, parallel,
        )
        with ThreadPoolExecutor(
            max_workers=parallel, thread_name_prefix="dich-song-song"
        ) as pool:
            for wave_begin in range(0, len(starts), parallel):
                if cancel_cb is not None and cancel_cb():
                    raise TranslationCancelledError("Người dùng đã huỷ tiến trình dịch.")
                wave = starts[wave_begin:wave_begin + parallel]
                anchor_output = list(output)  # lịch sử = các đợt đã chốt
                future_to_start = {}
                wave_results: dict[int, list[TranslationLine]] = {}
                for start in wave:
                    current_batch = input_lines[start:start + batch_size]
                    resumed = resume.lines_for(current_batch) if resume else None
                    if resumed is not None:
                        wave_results[start] = resumed
                        done_batches += 1
                        if progress_cb is not None:
                            progress_cb(done_batches / total_batches)
                        continue
                    source_before, source_after, history_before = self._batch_context(
                        source_lines, start, len(current_batch), ctx_size, anchor_output
                    )
                    future = pool.submit(
                        self._translate_single_batch,
                        batch=current_batch,
                        source_before=source_before,
                        source_after=source_after,
                        history_before=history_before,
                        start_idx=start,
                        is_literal=is_literal,
                        is_preprocess=is_preprocess,
                        config=config,
                            model_name=stage.model_name,
                        cancel_cb=cancel_cb,
                        _ctx_size=ctx_size,
                        video_files=None,
                        dual_payload=use_dual_payload,
                        expansion_indices=expansion_indices,
                        video_token_estimate=0,
                    )
                    future_to_start[future] = start
                first_error: BaseException | None = None
                translated: list[int] = []
                for future in as_completed(future_to_start):
                    error = future.exception()
                    if error is not None:
                        if first_error is None:
                            first_error = error
                        continue
                    wave_results[future_to_start[future]] = future.result()
                    translated.append(future_to_start[future])
                    done_batches += 1
                    if progress_cb is not None:
                        progress_cb(done_batches / total_batches)
                if batch_done_cb is not None:
                    for start in sorted(translated):
                        batch_done_cb(
                            resume.to_original(wave_results[start])
                            if resume is not None
                            else list(wave_results[start])
                        )
                if first_error is not None:
                    raise first_error
                for start in wave:
                    output.extend(wave_results[start])
        return output

    def translate_stage(
        self,
        *,
        stage: TranslationStageConfig,
        context: TranslationContext,
        source_lines: list[TranslationLine],
        input_lines: list[TranslationLine],
        has_prior_translation: bool,
        progress_cb: StageProgressCallback | None = None,
        cancel_cb: CancellationCallback | None = None,
        video_refs: list[Any] | None = None,
        attach_video: bool = False,
        source_was_preprocessed: bool = False,
        resume_lines: list[TranslationLine] | None = None,
        batch_done_cb: BatchDoneCallback | None = None,
    ) -> list[TranslationLine]:
        self._ensure_available()
        if not input_lines:
            return []

        is_preprocess = stage.kind is TranslationStageKind.PREPROCESS
        is_literal = stage.kind is TranslationStageKind.LITERAL
        # [Anti-Chinese Whispers] Seed bản gốc bất biến ở giai đoạn dịch ĐẦU TIÊN
        # (khi text vẫn là ngôn ngữ gốc). Các stage sau giữ nguyên để đối chiếu.
        # [v3.23.503] Gieo NGAY TỪ PREPROCESS: trước đây bỏ qua khâu này nên ``original_text``
        # thực chất lưu bản ĐÃ HIỆU ĐÍNH, không phải bản gốc thô — nếu PREPROCESS làm lệch dòng
        # thì mọi giai đoạn sau đối chiếu với dữ liệu đã sai và không thể phát hiện.
        if not has_prior_translation:
            input_lines = [
                ln if ln.original_text else replace(ln, original_text=ln.text)
                for ln in input_lines
            ]
        # [Dual Payload] Bật truyền kép cho khâu có bản nháp từ giai đoạn trước: gửi kèm bản gốc
        # để AI đối chiếu, chống làm sai lệch nghĩa.
        # [v3.23.503] Thêm LITERAL: nó nhận đầu vào là KẾT QUẢ của PREPROCESS, nên nếu không kèm
        # bản gốc thô thì lệch dòng do PREPROCESS gây ra sẽ trôi qua mà không khâu nào bắt được.
        # [v3.23.504] LITERAL cần cờ RIÊNG: ``has_prior_translation`` chỉ bật sau một giai đoạn
        # DỊCH, nên khi LITERAL chạy nó luôn False — điều kiện cũ khiến LITERAL không bao giờ
        # nhận payload kép, kể cả khi PREPROCESS đã sửa dòng nguồn. Dùng ``source_was_preprocessed``
        # để chỉ bật đúng lúc CÓ khâu hiệu đính phía trước; chạy thẳng LITERAL thì không đính kèm
        # (bản gốc trùng y hệt đầu vào, đính thêm chỉ tốn token).
        use_dual_payload = (
            stage.kind is TranslationStageKind.LITERAL and source_was_preprocessed
        ) or (has_prior_translation and stage.kind in (
            TranslationStageKind.STYLE,
            TranslationStageKind.LOCALIZE,
        ))
        # [v3.23.225] VÒNG TỰ SỬA (chi phí 0): dòng bị nghi DỊCH THIẾU ở khâu TRƯỚC được
        # đánh cờ ngay trong payload của khâu NÀY. Log thực tế cho thấy cùng 5 dòng hỏng
        # trôi qua cả ba giai đoạn mà không giai đoạn nào sửa — trong khi GĐ tinh chỉnh và
        # bản địa hoá DÙ SAO CŨNG gọi API và DÙ SAO CŨNG có bản gốc trong payload kép.
        # Tính từ chính ``input_lines`` (= output khâu trước) nên KHÔNG cần lưu state.
        expansion_indices: frozenset[int] = frozenset()
        if use_dual_payload:
            expansion_indices = frozenset(
                s.index
                for s in find_under_translated(
                    [
                        (ln.index, ln.original_text, ln.text)
                        for ln in input_lines
                        if ln.original_text and ln.text
                    ]
                )
            )
            if expansion_indices:
                logger.info(
                    "Giai đoạn '%s': yêu cầu AI BỔ SUNG nội dung cho %d dòng bị nghi "
                    "dịch thiếu ở khâu trước (không tốn thêm lượt gọi).",
                    stage.kind.value,
                    len(expansion_indices),
                )

        # [v3.23.107] BẢO VỆ DÒNG RÁC: nguồn CJK mà dòng KHÔNG có ký tự CJK (OCR đọc
        # nhầm ra "akas", "aks") rất dễ khiến model GỘP/BỎ -> lệch dòng dây chuyền (sai timestamp
        # cho hàng loạt câu sau). Không gửi chúng cho model; giữ NGUYÊN VĂN rồi trộn lại theo
        # index. Vừa chống lệch, vừa đúng tinh thần "OCR trung thực".
        full_input_lines = input_lines
        noise_indices = self._noise_line_indices(source_lines, context)
        if noise_indices:
            input_lines = [ln for ln in input_lines if ln.index not in noise_indices]
            source_lines = [ln for ln in source_lines if ln.index not in noise_indices]
            if not input_lines:
                return list(full_input_lines)
        # [v3.23.133] Lõi validate/renumber giả định line_no LIÊN TỤC [1..N] theo vị trí.
        # Index có LỖ HỔNG (vd 1,2,4,5 sau khi lọc dòng nhiễu) khiến model trả đúng index
        # thực vẫn bị tưởng "lệch số" rồi đánh số lại theo vị trí → DỒN LỆCH bản dịch.
        # Khắc phục: đánh số lại LIÊN TỤC [1..N] để gửi model, rồi ÁNH XẠ NGƯỢC.
        # [v3.23.593] Áp cho MỌI tập dòng không liên tục từ 1, không riêng ca lọc dòng
        # nhiễu: các lượt dịch lại một nhóm dòng (trọng tài, dọn chữ Hán sót…) gửi tập con
        # như 5,6,7,723,724,725 — lượt thật v3.23.592: dòng 5 nhận bản dịch của dòng 724,
        # bốn dòng còn lại trả rỗng, lưới dòng lẻ không phân xử được dòng nào.
        reindex_map = self._contiguous_reindex_map(input_lines, source_lines)
        if reindex_map:
            back = {original: pos for pos, original in reindex_map.items()}
            expansion_indices = frozenset(
                back[index] for index in expansion_indices if index in back
            )
            input_lines = [
                replace(ln, index=pos + 1) for pos, ln in enumerate(input_lines)
            ]
            source_lines = [
                replace(ln, index=pos + 1) for pos, ln in enumerate(source_lines)
            ]

        # [v3.23.584] Lô đã dịch ở lần chạy DỞ của chính giai đoạn này → dùng lại.
        resume = BatchResume.build(resume_lines, reindex_map)
        response_schema = self._response_schema_for(stage, context)
        system_instruction = self._system_instruction(
            stage, context, has_prior_translation,
            with_video=bool(attach_video and video_refs),
        )
        batch_size = max(1, stage.batch_size)
        # [v3.23.131] TỰ GIỚI HẠN lô cho model "lite": các model nhẹ (vd
        # gemini-3.1-flash-lite) trả kém ổn định khi lô quá lớn — hay rớt/đánh số lệch
        # dòng, sinh nhiều lần vá/chia đôi (chậm) và đôi khi lệch nội dung. Trần ~60 dòng
        # giúp ổn định rõ rệt mà vẫn nhanh. Không đụng tới model đầy đủ (flash/pro).
        capped_batch = effective_batch_size(stage.model_name, batch_size)
        if capped_batch != batch_size:
            logger.info(
                "Model '%s' là bản lite → giảm lô %d -> %d cho ổn định (chống rớt dòng).",
                stage.model_name, batch_size, capped_batch,
            )
            batch_size = capped_batch
        ctx_size = max(0, stage.context_size)
        total = len(input_lines)
        total_batches = max(1, (total + batch_size - 1) // batch_size)
        # Tạo config MỘT LẦN rồi truyền vào _translate_single_batch
        # (thay vì mỗi batch tự build lại — tiết kiệm CPU và nhất quán thinking settings)
        # [v3.23.509] Dựng config qua closure để mỗi LÔ có thể gắn schema ràng buộc riêng
        # (số phần tử = số dòng của lô, khoảng line_no của lô) mà không lặp danh sách tham số.
        def _config_for(schema_for_call: dict[str, Any]) -> Any:
            return self._build_config(
                stage.temperature,
                schema_for_call,
                system_instruction,
                enable_thinking=stage.enable_thinking,
                thinking_budget=stage.thinking_budget,
                model_name=stage.model_name,
                thinking_level=getattr(stage, "thinking_level", "low"),
                # [v3.23.127] Khi đính video cho từng batch: dùng độ phân giải THẤP để tiết
                # kiệm token (chi tiết hình ảnh đã được trích sẵn thành 'cue' dạng văn bản ở
                # khâu phân tích). Không đính video thì tham số này vô hại.
                media_resolution="low" if (attach_video and video_refs) else None,
            )

        _prebuilt_config = _config_for(response_schema)

        output: list[TranslationLine] = []
        # [v3.23.149] SONG SONG THEO ĐỢT khi người dùng bật (>1) và KHÔNG đính video
        # (batch đính video nặng token — TPM sẽ bắt chờ, song song vô ích; giữ tuần tự
        # cho ổn định). Mức hiệu dụng tự hạ theo RPM limit của model (dự đoán trước).
        _parallel = self._effective_parallel(
            getattr(self, "_parallel_batches", 1), stage.model_name
        )
        if _parallel > 1 and not (attach_video and video_refs) and total_batches > 1:
            output = self._run_batches_parallel(
                parallel=_parallel,
                input_lines=input_lines,
                source_lines=source_lines,
                batch_size=batch_size,
                ctx_size=ctx_size,
                total_batches=total_batches,
                stage=stage,
                config=_prebuilt_config,
                is_literal=is_literal,
                is_preprocess=is_preprocess,
                use_dual_payload=use_dual_payload,
                expansion_indices=expansion_indices,
                progress_cb=progress_cb,
                cancel_cb=cancel_cb,
                resume=resume,
                batch_done_cb=batch_done_cb,
            )
            # Trộn lại dòng rác + ánh xạ ngược index (chung với đường tuần tự bên dưới).
            if noise_indices or reindex_map:
                output = [
                    replace(ln, index=reindex_map.get(ln.index, ln.index)) for ln in output
                ]
                translated_by_index = {ln.index: ln for ln in output}
                return [translated_by_index.get(ln.index, ln) for ln in full_input_lines]
            return output

        reused_batches = 0
        for batch_idx, start in enumerate(range(0, total, batch_size)):
            if cancel_cb is not None and cancel_cb():
                raise TranslationCancelledError("Người dùng đã huỷ tiến trình dịch.")

            current_batch = input_lines[start:start + batch_size]
            resumed = resume.lines_for(current_batch) if resume else None
            if resumed is not None:
                output.extend(resumed)
                reused_batches += 1
                if progress_cb is not None:
                    progress_cb((batch_idx + 1) / total_batches)
                continue
            source_before, source_after, history_before = self._batch_context(
                source_lines, start, len(current_batch), ctx_size, output
            )

            # Chọn đoạn video phủ đúng khoảng thời gian của batch này (nếu bật đính
            # video cho giai đoạn). Phim cắt nhiều đoạn → mỗi batch chỉ gửi (các) đoạn
            # giao với khoảng [đầu, cuối] của batch, tránh gửi thừa token.
            batch_video_handles: list[Any] | None = None
            batch_video_tokens = 0
            # [v3.23.499] KHỞI TẠO TRƯỚC: các đoạn video phủ khoảng thời gian của batch. Trước đây
            # chỉ gán bên trong nhánh "có đính video", nhưng lại được dùng ở lời gọi phía dưới →
            # nhánh KHÔNG đính video (vd vòng dọn CJK sót dịch lại từ nguồn) văng
            # UnboundLocalError, làm hỏng cả tác vụ dịch.
            covering: list[Any] = []
            if attach_video and video_refs:
                t_start = min(ln.start_ms for ln in current_batch) / 1000.0
                t_end = max(ln.end_ms for ln in current_batch) / 1000.0
                covering = [
                    r for r in video_refs
                    if getattr(r, "start_sec", 0.0) < t_end
                    and getattr(r, "end_sec", 0.0) > t_start
                ]
                # [v3.23.561] Heal theo lô. Trước đây handle nào 403 (file thuộc key cũ sau
                # khi xoay key) bị BỎ QUA LẶNG LẼ → lô gửi đi KHÔNG có video mà không lỗi nào
                # kích hoạt tải lại. Nhật ký thật: 261 lần bỏ qua, cả 3 giai đoạn dịch chạy
                # không video dù người dùng bật đính video.
                batch_video_handles = self._handles_for_batch(
                    covering, video_refs, batch_idx
                )
                # [v3.23.648] Chỉ gửi khoảng của đoạn mà lô này thật sự cần (± lề) thay vì cả đoạn ~6 phút.
                batch_video_handles, seconds_sent = self._clip_batch_handles(
                    batch_video_handles, covering, t_start, t_end
                )
                # [v3.23.19] Ước lượng token video của các đoạn phủ (thời lượng × 100)
                # để quota điều tiết đúng. Cảnh báo nếu một batch đính video vượt ngưỡng
                # an toàn TPM (250k free tier) — đính cả đoạn dài rất tốn token.
                batch_video_tokens = int(seconds_sent * _VIDEO_TOKENS_PER_SEC)
                if batch_video_tokens > 200_000:
                    logger.warning(
                        "Batch %d đính video ~%d token (gần/vượt trần TPM 250k) — dễ gây "
                        "429. Cân nhắc TẮT đính video cho giai đoạn này; phân tích ngữ "
                        "cảnh toàn cục thường đã đủ để dịch chính xác.",
                        batch_idx + 1, batch_video_tokens,
                    )

            batch_result = self._translate_single_batch(
                batch=current_batch,
                source_before=source_before,
                source_after=source_after,
                history_before=history_before,
                start_idx=start,
                is_literal=is_literal,
                is_preprocess=is_preprocess,
                # [v3.23.509] Schema RÀNG BUỘC theo lô: số phần tử phải bằng số dòng gửi đi và
                # line_no phải nằm trong khoảng của lô → chặn gộp/bỏ/lệch dòng ngay ở tầng schema.
                config=_config_for(
                    _batch_constrained_schema(response_schema, current_batch)
                ),
                model_name=stage.model_name,
                cancel_cb=cancel_cb,
                _ctx_size=ctx_size,
                video_files=batch_video_handles,
                # [v3.23.483] TRUYỀN video_refs để heal 403 tự tải lại đoạn khi xoay key. Trước
                # đây thiếu → điều kiện re-upload (video_needs_refresh AND video_refs) không thoả
                # → handle cũ bị dùng lại → "Dịch phụ đề thất bại: 403" (app_log) DÙ đã set
                # reupload_cb (v482). Đây là mảnh ghép cuối để heal 403 ở khâu dịch hoạt động.
                video_refs=list(covering),
                dual_payload=use_dual_payload,
                expansion_indices=expansion_indices,
                video_token_estimate=batch_video_tokens,
            )
            output.extend(batch_result)
            if batch_done_cb is not None:
                batch_done_cb(resume.to_original(batch_result))

            if progress_cb is not None:
                progress_cb((batch_idx + 1) / total_batches)

        if reused_batches:
            logger.info(
                "Giai đoạn '%s': dùng lại %d/%d lô đã dịch ở lần chạy dở — không gọi lại "
                "API cho các lô đó.",
                stage.kind.value, reused_batches, total_batches,
            )

        # [v3.23.107] Trộn lại dòng rác (nguyên văn) đúng vị trí theo index.
        if noise_indices or reindex_map:
            # [v3.23.133] Ánh xạ NGƯỢC index liên tục → index gốc trước khi trộn lại.
            output = [
                replace(ln, index=reindex_map.get(ln.index, ln.index)) for ln in output
            ]
            translated_by_index = {ln.index: ln for ln in output}
            output = [translated_by_index.get(ln.index, ln) for ln in full_input_lines]
        output = self._revert_stage_regressions(
            full_input_lines, output, use_dual_payload
        )
        self._warn_under_translated(output, is_preprocess)
        return output

    @staticmethod
    def _contiguous_reindex_map(
        input_lines: list[TranslationLine], source_lines: list[TranslationLine]
    ) -> dict[int, int]:
        """[v3.23.593] ``vị trí 1..N → index gốc`` khi tập dòng KHÔNG liên tục từ 1.

        Args:
            input_lines: Dòng gửi dịch (đã bỏ dòng nhiễu).
            source_lines: Câu gốc song song với ``input_lines``.

        Returns:
            Rỗng nếu index đã là 1..N (không cần đánh số lại), hoặc nếu hai danh sách
            không song song (khi đó đánh số lại theo vị trí sẽ ghép sai câu gốc).
        """
        indices = [ln.index for ln in input_lines]
        if indices == list(range(1, len(indices) + 1)):
            return {}
        if indices != [ln.index for ln in source_lines]:
            logger.warning(
                "Dòng gửi dịch và câu gốc không cùng tập số dòng (%d/%d) — không đánh số "
                "lại.", len(input_lines), len(source_lines),
            )
            return {}
        return {pos + 1: index for pos, index in enumerate(indices)}

    @staticmethod
    def _revert_stage_regressions(
        input_lines: list[TranslationLine],
        output: list[TranslationLine],
        use_dual_payload: bool,
    ) -> list[TranslationLine]:
        """[v3.23.226] HOÀN NGUYÊN các dòng bị giai đoạn này làm rớt nội dung.

        Nguyên tắc: giai đoạn tinh chỉnh/bản địa hoá chỉ được GIỮ hoặc CẢI THIỆN, TUYỆT
        ĐỐI không được làm TỆ ĐI. Log v225 cho thấy mỗi giai đoạn vừa sửa được dòng hỏng
        cũ, vừa làm rớt nội dung ở một dòng KHÁC đang tốt (GĐ3 sửa 82/71/80 nhưng hỏng 84;
        GĐ4 sửa 84 nhưng hỏng LẠI 82) -> đuổi bắt vòng tròn, không bao giờ về 0.

        Bản trước tuy có thể kém trau chuốt hơn, nhưng KHÔNG mất nội dung — và mất nội
        dung là lỗi nặng hơn nhiều so với câu chữ chưa mượt.

        Args:
            input_lines: Đầu vào của giai đoạn (bản dịch của khâu trước).
            output: Đầu ra của giai đoạn.
            use_dual_payload: Chỉ áp dụng cho giai đoạn CÓ bản gốc để đối chiếu.

        Returns:
            Danh sách đã hoàn nguyên các dòng bị làm hỏng (giữ nguyên phần còn lại).
        """
        if not use_dual_payload:
            return output
        text_truoc = {ln.index: ln.text for ln in input_lines}
        items = [
            (ln.index, ln.original_text, text_truoc[ln.index], ln.text)
            for ln in output
            if ln.original_text and ln.text and ln.index in text_truoc
        ]
        if not items:
            return output
        hong = find_regressions(items)
        if not hong:
            return output
        logger.warning(
            "Giai đoạn này LÀM RỚT nội dung ở %d dòng đang tốt -> HOÀN NGUYÊN về bản "
            "trước (thà kém mượt còn hơn mất nghĩa): %s",
            len(hong),
            sorted(hong),
        )
        # [v3.23.588] Vào file chuẩn đoán — trước đây chỉ có trong nhật ký ứng dụng.
        _record_process_event("content_drops_reverted", lines=sorted(hong))
        return [
            replace(ln, text=text_truoc[ln.index]) if ln.index in hong else ln
            for ln in output
        ]

    @staticmethod
    def _warn_under_translated(
        output: list[TranslationLine], is_preprocess: bool
    ) -> None:
        """[v3.23.224] Cảnh báo các dòng bị DỊCH THIẾU nội dung (so với mặt bằng job).

        Chuỗi v222-v223 cho thấy model có thể rút gọn tới mức MẤT NGHĨA mà không một chỉ
        số kỹ thuật nào báo động (TTS thậm chí còn đẹp hơn). Lưới này so mỗi dòng với
        TRUNG VỊ của chính bộ phim -> không cần hằng số đoán mò về tỉ lệ độ dài Việt/CJK.
        Chỉ CẢNH BÁO, không tự sửa (một số dòng ngắn là đúng: "Ừm.", "Hả?").

        Args:
            output: Kết quả dịch của giai đoạn.
            is_preprocess: True nếu là khâu tiền xử lý (chưa dịch — bỏ qua).
        """
        if is_preprocess:
            return
        pairs = [
            (ln.index, ln.original_text, ln.text)
            for ln in output
            if ln.original_text and ln.text
        ]
        if pairs:
            log_under_translated(pairs)

    @staticmethod
    def _repair_escaped_newlines(text: str) -> str:
        """Khôi phục xuống dòng bị model xuất dạng LITERAL (hàm thuần).

        LLM thỉnh thoảng trả ký tự xuống dòng dưới dạng chuỗi ``\\n`` (gạch chéo + 'n')
        trong trường JSON thay vì ký tự xuống dòng THẬT — khiến phụ đề hiển thị lộ ra
        ``\\n`` trên màn hình (đã gặp thực tế ở lô cuối phim). Phụ đề không bao giờ có
        nội dung hợp lệ là gạch-chéo-n nên quy đổi về xuống dòng thật là an toàn. Giữ
        NGUYÊN ``\\N`` (chữ HOA — mã ngắt dòng cứng của ASS) để không phá tag ASS.

        Args:
            text: Văn bản model trả về (có thể chứa ``\\r\\n``/``\\n`` dạng literal).

        Returns:
            Văn bản với mọi literal ``\\r\\n``/``\\n``/``\\r`` đổi thành ``\n`` thật.
        """
        if "\\" not in text:
            return text
        # Thứ tự: \r\n trước để không sinh 2 lần xuống dòng; chỉ 'n' thường, chừa '\N'.
        return (
            text.replace("\\r\\n", "\n")
            .replace("\\n", "\n")
            .replace("\\r", "\n")
        )

    def _detect_batch_misalignment_payload(
        self, batch: list[TranslationLine], payload: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """[v3.23.518] Dò lệch dòng ngay trên payload THÔ của model (trước khi ghép).

        Chạy ở đây để lỗi phát sinh nằm gọn trong vòng thử-lại của ``_call_gemini``.

        Args:
            batch: Các dòng nguồn của lô.
            payload: JSON model trả về (khoá ``subtitles``).

        Returns:
            Danh sách dòng lệch (rỗng nếu không đủ căn cứ).
        """
        try:
            texts = self._payload_texts(batch, payload)
            result = [
                replace(line, text=texts.get(line.index, line.text or "")) for line in batch
            ]
            return self._detect_batch_misalignment(batch, result)
        except Exception:  # noqa: BLE001 — kiểm chất lượng không được làm hỏng khâu dịch
            logger.debug("Bỏ qua kiểm căn dòng trên payload.", exc_info=True)
            return []

    @staticmethod
    def _payload_texts(
        batch: list[TranslationLine], payload: dict[str, Any]
    ) -> dict[int, str]:
        """Bản dịch trong payload THÔ theo số dòng (model đánh số lại → theo VỊ TRÍ).

        Args:
            batch: Các dòng nguồn của lô.
            payload: JSON model trả về (khoá ``subtitles``).

        Returns:
            ``số dòng → văn bản model trả`` (dòng model không trả thì không có mặt).
        """
        batch_line_nos = {line.index for line in batch}
        texts: dict[int, str] = {}
        for offset, item in enumerate(payload.get("subtitles") or []):
            if not isinstance(item, dict):
                continue
            line_no = _safe_int(item.get("line_no"), -1)
            if line_no <= 0 or line_no not in batch_line_nos:
                line_no = batch[offset].index if offset < len(batch) else -1
            if line_no > 0:
                texts[line_no] = str(item.get("text", "") or "")
        return texts

    def _source_echo_in_payload(
        self, batch: list[TranslationLine], payload: dict[str, Any]
    ) -> list[int]:
        """[v3.23.579] Dòng dịch thô mà model trả NGUYÊN VĂN câu gốc, nếu đủ nhiều.

        Xem :mod:`subtitles_extractor.application.services.source_echo_guard`.

        Args:
            batch: Các dòng nguồn của lô.
            payload: JSON model trả về.

        Returns:
            Các dòng nguyên văn; rỗng nếu dưới ngưỡng (một dòng lẻ là dòng bỏ trống).
        """
        from subtitles_extractor.application.services.source_echo_guard import (
            is_echo_batch,
            source_echo_lines,
        )

        sources = {line.index: (line.original_text or line.text or "") for line in batch}
        echoed = source_echo_lines(sources, self._payload_texts(batch, payload))
        return echoed if is_echo_batch(len(echoed), len(batch)) else []

    def _detect_batch_misalignment(
        self, batch: list[TranslationLine], result: list[TranslationLine]
    ) -> list[dict[str, Any]]:
        """[v3.23.516] Dò lệch dòng trong MỘT lô bằng mốc neo tên nhân vật + thuật ngữ.

        Chỉ kết luận khi có TỪ HAI dòng trở lên bị cờ: một dòng đơn lẻ thường chỉ là đảo trật tự
        hợp lệ khi câu bị tách đôi, còn lệch thật gần như luôn kéo theo cả cụm. Bất kỳ lỗi nào khi
        dò cũng được nuốt — kiểm chất lượng không được phép làm hỏng khâu dịch.

        Args:
            batch: Các dòng nguồn của lô.
            result: Kết quả đã ghép của lô.

        Returns:
            Danh sách dòng lệch (rỗng nếu không đủ căn cứ).
        """
        try:
            from subtitles_extractor.application.services.alignment_check import (
                MIN_FLAGGED_LINES,
                build_anchor_pairs,
                corroborated_findings,
                detect_length_shift,
                drop_unreliable_anchors,
                find_low_correlation_blocks,
                find_misaligned_lines,
            )

            sources = {line.index: (line.original_text or line.text or "") for line in batch}
            translations = {line.index: (line.text or "") for line in result}

            # Tín hiệu 1 — MỐC NEO (chính xác cao, nhưng chỉ phủ ~13% dòng có tên/thuật ngữ).
            # [v3.23.549] Lọc neo không đáng tin + đòi dòng bổ trợ CÙNG độ lệch: trên phim thật
            # 5/5 dòng bị cờ là dương tính giả — mỗi cái kéo theo một lượt dịch lại vô ích.
            anchors = build_anchor_pairs(
                getattr(self, "_active_characters", "") or "",
                getattr(self, "_active_glossary", "") or "",
            )
            anchors = drop_unreliable_anchors(anchors, sources, translations)
            if anchors:
                by_anchor = corroborated_findings(
                    find_misaligned_lines(sources, translations, anchors)
                )
                if len(by_anchor) >= MIN_FLAGGED_LINES:
                    return by_anchor

            # [v3.23.549] Tín hiệu 1b — KHỐI tương quan thấp trên cửa sổ trượt. Tín hiệu 2 bên
            # dưới tính MỘT hệ số cho cả lô nên khối lệch nằm giữa lô bị pha loãng; ca thật 18
            # dòng sai liền (model gộp hai dòng ngắn) lọt qua cả ba lớp kiểm cũ.
            blocks = find_low_correlation_blocks(sources, translations)
            # [v3.23.573] Tinh chỉnh/bản địa hoá: khối mà bản trả về vẫn căn đúng
            # từng dòng với bản của giai đoạn trước là báo nhầm — không dịch lại.
            blocks = [
                block
                for block in blocks
                if not self._block_aligned_with_prior(batch, result, block)
            ]
            if blocks:
                worst = blocks[0]
                return [
                    {
                        "index": worst["from_index"],
                        "belongs_to_index": worst["from_index"],
                        "shift": 0,
                        "block_lines": worst["lines"],
                        "min_correlation": worst["min_correlation"],
                    },
                    {
                        "index": worst["to_index"],
                        "belongs_to_index": worst["to_index"],
                        "shift": 0,
                    },
                ]

            # [v3.23.517] Tín hiệu 2 — TƯƠNG QUAN ĐỘ DÀI, phủ được MỌI dòng kể cả lô không có tên
            # riêng nào. Câu gốc dài kéo theo bản dịch dài, nên khi căn ĐÚNG dòng thì hai dãy độ
            # dài tương quan mạnh; cả khối bị đẩy đi thì tương quan tại 0 tụt và đỉnh chuyển sang
            # đúng khoảng lệch. Đo thật: lô bình thường 0.71-0.92, lô lệch tụt còn 0.40.
            shift = detect_length_shift(sources, translations)
            if shift:
                ordered = sorted(sources)
                return [
                    {
                        "index": index,
                        "belongs_to_index": index + shift,
                        "shift": shift,
                    }
                    for index in ordered[:2]
                ]
            return []
        except Exception:  # noqa: BLE001 — kiểm chất lượng không được làm hỏng khâu dịch
            logger.debug("Bỏ qua kiểm căn dòng cho lô (lỗi nội bộ).", exc_info=True)
            return []

    @staticmethod
    def _block_aligned_with_prior(
        batch: list[TranslationLine],
        result: list[TranslationLine],
        block: dict[str, Any],
    ) -> bool:
        """[v3.23.573] Khối tương quan độ dài thấp nhưng căn ĐÚNG từng dòng với bản trước.

        Tương quan độ dài sụp ở đoạn câu ngắn mà tiếng Việt đảo trật tự ("卖给我" 3 chữ
        ra "miếng ngọc bội này cho em không?"): đoạn 371-390 của phim thật bị báo lệch ở
        10/23 lượt dịch dù căn đúng, mỗi lần tốn một lượt gọi dịch lại cả lô. Ở giai
        đoạn có bản trước (tinh chỉnh, bản địa hoá), so thẳng từng dòng với bản đó là bằng
        chứng chắc hơn — xem ``stage_shift_guard.aligned_with_previous``. Dịch thô chưa có
        bản trước nên giữ nguyên cách cũ.

        Args:
            batch: Các dòng đầu vào của lô (``text`` là bản của giai đoạn trước).
            result: Bản model vừa trả.
            block: Khối nghi lệch từ ``find_low_correlation_blocks``.

        Returns:
            True nếu khối căn đúng với bản trước (bỏ qua cờ); False nếu không đủ căn cứ.
        """
        from subtitles_extractor.application.services.stage_shift_guard import (
            aligned_with_previous,
        )

        previous = {
            line.index: line.text
            for line in batch
            if line.original_text
            and line.text
            and line.text != line.original_text
            and not _CJK_CHAR_RE.search(line.text)
        }
        if not previous:
            return False
        current = {line.index: line.text or "" for line in result}
        indices = [
            line.index
            for line in batch
            if block["from_index"] <= line.index <= block["to_index"]
        ]
        if not aligned_with_previous(previous, current, indices):
            return False
        logger.info(
            "Bỏ qua cờ lệch khối dòng %d-%d (tương quan độ dài %s): bản trả về căn đúng "
            "từng dòng với bản của giai đoạn trước.",
            block["from_index"], block["to_index"], block["min_correlation"],
        )
        _record_process_event(
            "batch_misaligned_dismissed",
            lines=[block["from_index"], block["to_index"]],
            min_correlation=block["min_correlation"],
        )
        return True

    @staticmethod
    def _merge_ai_items(
        current_batch: list[TranslationLine], ai_items: list[dict[str, Any]], is_preprocess: bool
    ) -> list[TranslationLine]:
        # [v3.23.9] Ghép theo line_no (ID) thay vì theo VỊ TRÍ. Nếu model xáo trộn
        # thứ tự dòng nhưng giữ đúng line_no, ta vẫn ghép bản dịch vào ĐÚNG câu gốc
        # (chống lỗi "nội dung bị xáo trộn"). Lập map line_no → item; fallback vị trí
        # cho item thiếu line_no (model cũ/không tuân thủ).
        items_by_line_no: dict[int, dict[str, Any]] = {}
        unmapped: list[dict[str, Any]] = []
        for item in ai_items:
            line_no = _safe_int(item.get("line_no"), -1)
            if line_no > 0 and line_no not in items_by_line_no:
                items_by_line_no[line_no] = item
            else:
                unmapped.append(item)

        # [v3.23.513] Model ĐÁNH SỐ LẠI line_no từ 1 cho mỗi lô (hành vi đã biết của các model
        # "lite"): số hợp lệ nhưng KHÔNG dòng nào thuộc lô. Trước đây map theo line_no trượt sạch,
        # phần dự phòng theo vị trí lại không chạy (vì item "có" line_no), nên MỌI dòng của lô giữ
        # nguyên văn bản gốc — tức còn nguyên chữ Hán — kèm log "AI trả rỗng cho dòng N".
        # Nghịch lý: line_no RÁC (âm/chữ) lại ghép đúng theo vị trí, chỉ số hợp lệ-nhưng-lệch mới hỏng.
        # Chỉ ghép theo vị trí khi KHÔNG có giao nhau và SỐ LƯỢNG khớp — cùng thứ tự nên an toàn.
        batch_line_nos = {line.index for line in current_batch}
        if (
            items_by_line_no
            and not (batch_line_nos & set(items_by_line_no))
            and len(ai_items) == len(current_batch)
        ):
            logger.info(
                "Model đánh số line_no lại từ đầu (không khớp lô) nhưng ĐỦ %d dòng → ghép theo "
                "vị trí thay vì bỏ qua.",
                len(current_batch),
            )
            unmapped = list(ai_items)
            items_by_line_no = {}

        merged: list[TranslationLine] = []
        # [v3.23.80] Item thiếu line_no được tiêu thụ theo THỨ TỰ (FIFO) cho các câu gốc
        # chưa khớp — thay cho ``unmapped[offset]`` (offset = vị trí trong batch) vốn lệch
        # khi batch TRỘN (có item kèm line_no lẫn item thiếu), làm mất bản dịch câu giữa.
        unmapped_iter = iter(unmapped)
        for original in current_batch:
            # Ưu tiên khớp theo index (line_no) của câu gốc; nếu không có thì lấy
            # item kế tiếp từ phần unmapped (giữ tương thích model không trả line_no).
            ai_item = items_by_line_no.get(original.index)
            if ai_item is None:
                ai_item = next(unmapped_iter, {})
            raw_text = ai_item.get("text", "")
            candidate = str(raw_text).strip() if raw_text is not None else ""
            # [v3.23.354] Sửa xuống dòng bị model xuất dạng literal '\n' → xuống dòng thật.
            candidate = GeminiSubtitleTranslator._repair_escaped_newlines(candidate)
            if not candidate:
                logger.warning(
                    "AI trả rỗng cho dòng %s — giữ nguyên văn bản gốc (chống nuốt chữ).",
                    original.index,
                )
                new_text = original.text
            else:
                new_text = candidate
            if is_preprocess:
                merged.append(
                    TranslationLine(
                        index=original.index,
                        start_ms=original.start_ms,
                        end_ms=original.end_ms,
                        text=new_text,
                        speaker=original.speaker,
                        description=original.description,
                        original_text=original.original_text,
                        addressee=original.addressee,
                        # [v3.23.153] GIỮ scene (cue bối cảnh/cảm xúc từ Visual Cues).
                        # Trước đây bị bỏ rơi -> mất vĩnh viễn ngay sau giai đoạn đầu ->
                        # STYLE/LOCALIZE không còn nhận 'cue' trong payload (mù bối cảnh).
                        scene=original.scene,
                    )
                )
            else:
                merged.append(
                    TranslationLine(
                        index=original.index,
                        start_ms=original.start_ms,
                        end_ms=original.end_ms,
                        text=new_text,
                        speaker=str(ai_item.get("speaker", original.speaker) or original.speaker),
                        description=str(ai_item.get("description", original.description) or original.description),
                        original_text=original.original_text,
                        addressee=original.addressee,
                        scene=original.scene,  # [v3.23.153] giữ cue cho các giai đoạn sau
                    )
                )
        return merged


__all__ = [
    "GeminiSubtitleTranslator",
    "HealedRefKey",
    "analysis_source_lines",
    "effective_batch_size",
    "healed_ref_key",
    "substitute_healed_refs",
]
