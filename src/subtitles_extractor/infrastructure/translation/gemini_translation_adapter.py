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

import json
import logging
import random
import re
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from typing import Any

from subtitles_extractor.application.services.under_translation_guard import (
    find_regressions,
    find_under_translated,
    log_under_translated,
)
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    CancellationCallback,
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
from subtitles_extractor.infrastructure.translation.key_rotation_policy import (
    should_switch_for_viability,
)
from subtitles_extractor.infrastructure.tts.timing_math import (
    MIN_CHAR_BUDGET,
    effective_available_seconds,
    readable_syllable_budget,
    syllable_budget_to_chars,
)

logger = logging.getLogger(__name__)

# [v3.23.143] Thư viện sửa JSON hỏng từ LLM (thuần Python, zero phụ thuộc bắt buộc). Import
# TUỲ CHỌN: nếu môi trường chưa cài, tự lùi về bộ vá cắt cụt nội bộ (không sập).
try:
    import json_repair as _json_repair
except ImportError:  # pragma: no cover - môi trường chưa cài optional dep
    _json_repair = None

# [v3.23.121] Bắt QuotaExhaustedError để XOAY sang API key khác khi key hiện tại hết quota.
try:
    from subtitles_extractor.infrastructure.translation.gemini_quota_manager import (
        QuotaExhaustedError,
    )
except ImportError:  # pragma: no cover - phòng trường hợp tách module
    class QuotaExhaustedError(RuntimeError):  # type: ignore[no-redef]
        ...

# ── JSON Schema ràng buộc đầu ra của Gemini ──────────────────────────────────
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

_RETRYABLE_HTTP_CODES = frozenset({408, 429, 500, 502, 503, 504})
_RETRYABLE_MARKERS = (
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
)


# [v3.23.131] Trần kích thước lô cho model "lite" (nhẹ) — lô lớn hơn gây rớt/lệch dòng.
_LITE_MODEL_BATCH_CAP = 60


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
    video ở các lần thử sau) thay vì để một lô làm HỎNG CẢ JOB dịch như trước."""


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
_CONTEXT_ANALYSIS_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "source_lang": {"type": "STRING"},
        "characters":  {"type": "STRING"},
        "overview":    {"type": "STRING"},
        # [v3.23.23] Bảng thuật ngữ + viết tắt để dịch NHẤT QUÁN xuyên suốt. Mỗi dòng:
        # 'Thuật ngữ gốc => bản dịch chuẩn' hoặc 'FBI => FBI (Cục Điều tra Liên bang)'.
        "glossary":    {"type": "STRING"},
    },
    "required": ["source_lang", "characters", "overview"],
}

# [v3.23.37] Schema phân tích ngữ cảnh KÈM Visual Cues — dùng khi bật phân tích hình
# ảnh, để LẤY LUÔN "ai nói/nói với ai" trong CÙNG request phân tích tuần tự (đã gửi
# video lên rồi) → tiết kiệm quota, KHÔNG cần gọi analyze_visual_cues riêng. Khoá 'cues'
# rút gọn (id/spk/to) để không chạm trần token output.
_CONTEXT_WITH_CUES_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "source_lang": {"type": "STRING"},
        "characters":  {"type": "STRING"},
        "overview":    {"type": "STRING"},
        "glossary":    {"type": "STRING"},
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
    "required": ["source_lang", "characters", "overview"],
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

# Số dòng giữ lại khi phân tích context (đã chuyển vào hàm analyze_global_context)
# _CONTEXT_ANALYSIS_MAX_SAMPLES đã bỏ — nay dùng toàn bộ + hard-cap 8000.


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
    ) -> None:
        # [v3.23.121] Hỗ trợ NHIỀU API key: tách theo dòng/phẩy, bỏ trùng & rỗng.
        # Khi một key chạm quota ngày, tự XOAY sang key kế tiếp còn hạn mức.
        self._api_keys: list[str] = self._parse_keys(api_key)
        self._key_index: int = 0
        self._api_key = self._api_keys[0] if self._api_keys else ""
        self._retry_count = max(1, retry_count)
        self._request_timeout_s = max(5.0, request_timeout_s)
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
        self._video_reupload_cb: Callable[[str], list[Any]] | None = None
        self._load_sdk()

    def set_video_reupload_callback(
        self, cb: Callable[[str], list[Any]] | None
    ) -> None:
        """Đăng ký callback tải lại video theo key hiện tại (chống 403 khi xoay key)."""
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

    def _rpd_remaining_for_key(self, model_name: str, key: str) -> int:
        """[v3.23.146] Số request/ngày CÒN LẠI của một key cho model (an toàn với lỗi)."""
        if self._quota_manager is None:
            return 1_000_000  # không quản quota -> coi như vô hạn
        try:
            remaining = self._quota_manager.get_remaining(
                model_name, key_id=self._fingerprint(key)
            )
        except (AttributeError, TypeError):
            return 1_000_000  # quota_manager cũ -> coi như còn
        return int(remaining.get("rpd_remaining", 1))

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
        return best_idx

    def has_any_daily_quota(self, model_name: str) -> bool:
        """[v3.23.146] True nếu CÓ ÍT NHẤT MỘT key còn quota ngày cho model.

        Cho tầng worker DỰ ĐOÁN trước khi làm việc tốn kém (nén+upload video ~2 phút): nếu
        MỌI key đã cạn quota ngày thì fail NHANH với thông báo rõ, không phí công rồi mới 429.
        """
        if self._quota_manager is None or not self._api_keys:
            return True
        return any(
            self._rpd_remaining_for_key(model_name, key) > 0 for key in self._api_keys
        )

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
            api_key (chuỗi) sẽ được dùng cho phiên sau khi đã chọn.
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
        logger.info(
            "Xoay API key #%d → #%d (còn %d req/ngày) vì key hiện tại đã hết quota '%s'.",
            self._key_index + 1, best_idx + 1,
            self._rpd_remaining_for_key(model_name, self._api_keys[best_idx]), model_name,
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
        if self._client is None:
            self._ensure_available()
            # [v3.7 fix] Đặt timeout HTTP (ms) để 1 request treo không làm đứng
            # worker vĩnh viễn. google-genai nhận http_options dạng dict {timeout: ms}.
            timeout_ms = int(self._request_timeout_s * 1000)
            try:
                self._client = self._genai_module.Client(
                    api_key=self._api_key,
                    http_options={"timeout": timeout_ms},
                )
            except TypeError:
                # SDK cũ không nhận http_options — fallback không timeout.
                logger.warning("SDK google-genai không hỗ trợ http_options.timeout; bỏ qua timeout.")
                self._client = self._genai_module.Client(api_key=self._api_key)
        return self._client

    # ── Cấu hình sinh nội dung ───────────────────────────────────────────
    # [v3.23.52] Nhận diện họ model Gemini 3.x (dùng thinking_level thay thinking_budget).
    @staticmethod
    def _is_gemini_3_family(model_name: str) -> bool:
        """True nếu là Gemini 3.x (3.6-flash, 3.5-flash-lite, 3.5-flash, 3.1-flash-lite, 3-pro…).

        Model 3.x dùng ``thinking_level`` (không dùng ``thinking_budget``) và BỎ QUA
        ``temperature/top_p/top_k`` (đã deprecated — xem changelog Gemini API 6/2026).
        """
        name = (model_name or "").lower()
        return "gemini-3" in name or "gemini-3." in name

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
        if not self._is_gemini_3_family(model_name):
            kwargs["temperature"] = max(0.0, min(1.0, temperature))
        if system_instruction:
            kwargs["system_instruction"] = system_instruction

        if media_resolution:
            resolved = self._resolve_media_resolution(media_resolution)
            if resolved is not None:
                kwargs["media_resolution"] = resolved

        if enable_thinking and thinking_budget != 0:
            try:
                if self._is_gemini_3_family(model_name):
                    # Gemini 3.x: thinking_level (KHÔNG kèm thinking_budget).
                    level = (thinking_level or "low").lower()
                    if level not in ("low", "medium", "high"):
                        level = "low"
                    try:
                        kwargs["thinking_config"] = types.ThinkingConfig(
                            thinking_level=level
                        )
                    except (AttributeError, TypeError):
                        # SDK cũ chưa có thinking_level → thử thinking_budget fallback.
                        kwargs["thinking_config"] = types.ThinkingConfig()
                else:
                    # Gemini 2.5: thinking_budget (-1 dynamic = không truyền).
                    thinking_kwargs: dict[str, Any] = {}
                    if thinking_budget > 0:
                        thinking_kwargs["thinking_budget"] = thinking_budget
                    kwargs["thinking_config"] = types.ThinkingConfig(**thinking_kwargs)
            except (AttributeError, TypeError):
                logger.debug("SDK này chưa hỗ trợ ThinkingConfig, bỏ qua thinking.")

        return types.GenerateContentConfig(**kwargs)

    def _strip_thinking_from_config(self, config: Any) -> Any:
        """Tạo bản sao config nhưng BỎ ``thinking_config`` (dùng khi fallback).

        Trả về config mới để retry với model không hỗ trợ thinking, giữ nguyên các
        thiết lập khác (schema, system_instruction, temperature).
        """
        types = self._types_module
        kwargs: dict[str, Any] = {}
        for attr in ("temperature", "response_mime_type", "response_schema", "system_instruction"):
            value = getattr(config, attr, None)
            if value is not None:
                kwargs[attr] = value
        try:
            return types.GenerateContentConfig(**kwargs)
        except (AttributeError, TypeError):
            return config

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
    @staticmethod
    def _is_retryable(error: Exception) -> bool:
        # [v3.23.142] JSON hỏng từ model thường NHẤT THỜI (một lần gen lỗi cú pháp). Cho
        # retry để gen lại thay vì bỏ cả đoạn/lô. Lớp cứu cắt cụt đã chạy trước đó.
        if isinstance(error, json.JSONDecodeError):
            return True
        if _safe_int(getattr(error, "code", None), 0) in _RETRYABLE_HTTP_CODES:
            return True
        if _safe_int(getattr(error, "status_code", None), 0) in _RETRYABLE_HTTP_CODES:
            return True
        error_text = str(error).lower()
        return any(marker in error_text for marker in _RETRYABLE_MARKERS)

    @staticmethod
    def _is_daily_quota_error(error: Exception) -> bool:
        """Nhận diện 429 do HẾT QUOTA NGÀY (RPD) — để xoay key thay vì chờ hồi."""
        text = str(error).lower()
        if "perdayperproject" in text or "requests_per_day" in text:
            return True
        if "free_tier_requests" in text and "exceeded" in text:
            return True
        # 429 kèm dấu hiệu giới hạn 'per day' / 'mỗi ngày'.
        is_429 = _safe_int(getattr(error, "code", None), 0) == 429 or "429" in text
        return is_429 and ("per day" in text or "/ngày" in text or "perday" in text)

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

    def _resolve_handles_with_heal(
        self, video_refs: list[Any] | None, fallback_handles: list[Any]
    ) -> list[Any]:
        """Resolve handle video bằng key HIỆN TẠI; nếu file thuộc key cũ (403) thì tải lại.

        Buộc resolve lại (xoá cache handle vì handle gắn theo key). Nếu thiếu handle so
        với số refs (file của key cũ không truy cập được) và có callback tải lại → tải
        lại đoạn bằng key hiện tại rồi resolve lại. Cập nhật ``video_refs`` TẠI CHỖ để
        các lần gọi sau dùng refs mới.
        """
        if not video_refs:
            return list(fallback_handles or [])
        self._video_handle_cache.clear()
        handles = self._resolve_video_handles(video_refs)
        if len(handles) >= len(video_refs):
            return handles
        if self._video_reupload_cb is None:
            return handles
        logger.info(
            "Handle video thuộc key cũ (không truy cập được) → tải lại bằng key hiện tại."
        )
        try:
            fresh_all = self._video_reupload_cb(self._api_key)
        except Exception as exc:
            logger.warning("Tải lại video bằng key mới thất bại: %s", exc)
            return handles
        if not fresh_all:
            return handles
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
    ) -> dict[str, Any]:
        """Gọi Gemini với retry cho lỗi tạm thời và lỗi sai cấu trúc batch.

        Nếu ``video_files`` được cung cấp, chúng được đính kèm TRƯỚC prompt để mô
        hình "xem" video làm ngữ cảnh. Nếu có ``self._quota_manager``, mỗi lần gọi sẽ
        đặt chỗ token trước (tránh vượt rate-limit) và đối soát token thực sau đó.
        """
        # [v3.7] Bắt sớm model rỗng để tránh retry loop vô nghĩa.
        if not (model_name or "").strip():
            raise SubtitleTranslationError("Tên model Gemini không được để trống.")

        # ── PHÂN LUỒNG 2 KÊNH theo đúng thiết kế Gemini API ──────────────────
        # KÊNH 1 (config.system_instruction): vai trò + quy tắc bền vững — đã nằm trong
        #   `config` (dựng ở `_build_config`), KHÔNG ghép vào prompt.
        # KÊNH 2 (contents): DỮ LIỆU của lượt này — video (nếu có) + prompt (các khối XML
        #   dữ liệu). Hai kênh tách bạch, không trộn lẫn.
        contents = [*(video_files or []), prompt]
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
        for attempt in range(self._retry_count):
            if cancel_cb is not None and cancel_cb():
                raise TranslationCancelledError("Người dùng đã huỷ tiến trình dịch.")
            # [v3.23.132] Sau khi XOAY KEY, handle video (do key cũ tải lên) sẽ bị 403 →
            # resolve lại bằng key hiện tại, tải lại đoạn nếu cần, rồi dựng lại contents.
            if video_needs_refresh and video_refs:
                current_handles = self._resolve_handles_with_heal(
                    video_refs, current_handles
                )
                contents = [*current_handles, prompt]
                video_needs_refresh = False
            reservation = None
            if self._quota_manager is not None:
                while True:
                    try:
                        reservation = self._quota_manager.acquire(
                            model_name, est_tokens, cancel_cb=cancel_cb,
                            key_id=self._current_key_id(),
                        )
                        break
                    except InterruptedError as exc:
                        raise TranslationCancelledError(
                            "Người dùng đã huỷ khi đang chờ quota."
                        ) from exc
                    except QuotaExhaustedError:
                        # Key hiện tại hết quota ngày → thử xoay sang key khác còn hạn.
                        if self._rotate_to_available_key(model_name):
                            video_needs_refresh = True
                            continue
                        raise  # mọi key đều hết → báo lên trên
            try:
                consumed = False
                client = self._get_client()
                response = client.models.generate_content(
                    model=model_name, contents=contents, config=config
                )
                # [v3.23.123] Generate ĐÃ trả về → server ĐÃ tính quota (kể cả khi text bị
                # safety filter). Đánh dấu để KHÔNG hoàn (release) bộ đếm ngày ở nhánh lỗi.
                consumed = True
                # Đối soát token thực tế (nếu có) để cửa sổ quota chính xác.
                if reservation is not None and self._quota_manager is not None:
                    actual = self._extract_token_count(response, est_tokens)
                    self._quota_manager.reconcile(reservation, actual)
                # [v3.7] response.text có thể None khi bị safety filter hoặc finish_reason=OTHER.
                raw_text = getattr(response, "text", None)
                if raw_text is None:
                    finish = getattr(response, "candidates", [{}])
                    raise _EmptyResponseError(
                        f"Gemini không trả về text (có thể bị safety filter hoặc lỗi model). "
                        f"candidates={finish}"
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
            except _EmptyResponseError as exc:
                last_error = exc
                # [v3.23.358] candidates=None ĐÃ tính quota ở server (consumed=True) nên
                # KHÔNG release. Trước đây lỗi này bị raise NGAY → một lô rỗng/bị chặn giết
                # CẢ JOB. Nay: nếu đang đính video (tác nhân thường gặp gây rỗng/safety),
                # BỎ video rồi thử lại NGAY; nếu vẫn rỗng thì retry backoff; hết retry mới
                # bỏ cuộc để tầng trên (halving/keep-original) xử lý.
                if current_handles and not video_dropped_for_empty:
                    video_dropped_for_empty = True
                    current_handles = []
                    contents = [prompt]
                    logger.warning(
                        "Gemini trả rỗng (candidates=None) khi đính video — thử lại KHÔNG "
                        "đính video (video có thể kích hoạt safety filter)."
                    )
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
                # [v3.23.123] Generate THẤT BẠI (503/429/mạng…) → server KHÔNG trừ quota.
                # Hoàn trả chỗ đã đặt (RPM + token + bộ đếm NGÀY) để không báo hết quota oan.
                if not consumed and reservation is not None and self._quota_manager is not None:
                    self._quota_manager.release(reservation)
                    reservation = None
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
                if (
                    video_refs
                    and self._video_reupload_cb is not None
                    and not video_needs_refresh
                    and self._is_file_permission_error(exc)
                    and attempt < self._retry_count - 1
                ):
                    logger.info(
                        "File video bị 403 (thuộc key cũ) → tải lại bằng key hiện tại."
                    )
                    video_needs_refresh = True
                    continue
                if self._is_retryable(exc) and attempt < self._retry_count - 1:
                    # [v3.23.121] Nếu là HẾT QUOTA NGÀY và còn API key khác: đánh dấu key
                    # hiện tại đã cạn rồi XOAY sang key kế — thử lại NGAY, khỏi chờ hồi.
                    _daily = self._is_daily_quota_error(exc)
                    if _daily and self._quota_manager is not None:
                        self._quota_manager.mark_daily_exhausted(
                            model_name, key_id=self._current_key_id()
                        )
                        if self._rotate_to_available_key(model_name):
                            video_needs_refresh = True
                            logger.info(
                                "Hết quota ngày ở key hiện tại → đã xoay sang key khác, "
                                "thử lại ngay với '%s'.", model_name,
                            )
                            continue
                    # [v3.23.9] Ưu tiên retryDelay server gửi kèm 429 (chờ ĐÚNG bằng
                    # thời gian quota hồi); chỉ dùng backoff mù khi server không nói.
                    server_delay = self._server_retry_delay(exc)
                    delay = server_delay if server_delay is not None else self._retry_delay(attempt)
                    # [v3.23.141] Ghi cooldown vào quota manager để các request/đoạn KẾ TIẾP
                    # tới cùng (key,model) cũng CHỜ, thay vì lao vào rồi 429 nối tiếp.
                    if server_delay is not None and self._quota_manager is not None:
                        self._quota_manager.note_rate_limited(
                            model_name, server_delay, key_id=self._current_key_id()
                        )
                    # [v3.23.147] 429 TẠM THỜI (TPM/RPM) trên lời gọi KHÔNG đính video:
                    # xoay ngay sang key còn quota + không cooldown (miễn phí vì không có
                    # file gắn key) -> tiếp tục tức thời thay vì chờ retryDelay.
                    if (
                        not _daily
                        and not video_refs
                        and self._is_rate_limit_error(exc)
                        and self._rotate_for_temporary_429(model_name)
                    ):
                        continue
                    logger.warning(
                        "Lỗi tạm thời từ Gemini (lần %d/%d): %s. Chờ %.1fs rồi thử lại%s...",
                        attempt + 1,
                        self._retry_count,
                        exc,
                        delay,
                        " (theo retryDelay server)" if server_delay is not None else "",
                    )
                    if self._interruptible_sleep(delay, cancel_cb):
                        raise TranslationCancelledError("Người dùng đã huỷ tiến trình dịch.") from exc
                    continue
                raise SubtitleTranslationError(f"Gọi Gemini thất bại: {exc}") from exc

        raise SubtitleTranslationError(f"Gọi Gemini thất bại sau {self._retry_count} lần: {last_error}")

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

    def _resolve_video_handles(self, video_refs: list[Any] | None) -> list[Any]:
        """Đổi danh sách RemoteVideoRef → handle file Gemini để đính vào contents.

        Có cache theo ``remote_name`` để không gọi lại ``files.get`` nhiều lần.
        Bỏ qua (cảnh báo) đoạn nào không resolve được thay vì làm hỏng cả lần dịch.
        """
        if not video_refs:
            return []
        client = self._get_client()
        handles: list[Any] = []
        for ref in video_refs:
            name = getattr(ref, "remote_name", None)
            if not name:
                continue
            handle = self._video_handle_cache.get(name)
            if handle is None:
                try:
                    handle = client.files.get(name=name)
                    self._video_handle_cache[name] = handle
                except Exception as exc:
                    logger.warning("Không lấy được handle video '%s': %s", name, exc)
                    continue
            handles.append(handle)
        return handles

    # ── Dựng system instruction theo giai đoạn ───────────────────────────
    @staticmethod
    def _canonical_names_directive(characters: str) -> str:
        """Trích danh sách TÊN CHUẨN từ roster + chỉ thị dùng nhất quán.

        Mỗi dòng roster ``Tên (alias)`` — tự nhận diện phần CJK để lấy phần KHÔNG-CJK
        (Latinh/tiếng đích) làm tên chuẩn. Trả về chuỗi chỉ thị (kèm danh sách tên) chèn
        vào system prompt, hoặc rỗng nếu roster không có tên dùng được.
        """
        if not characters:
            return ""
        cjk = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u30ff\uac00-\ud7af]")

        def _has_cjk(text: str) -> bool:
            return bool(cjk.search(text))

        names: list[str] = []
        seen: set[str] = set()
        for raw in characters.splitlines():
            line = raw.strip()
            if not line:
                continue
            match = re.match(r"^(.+?)\s*[(（]\s*([^)）]+?)\s*[)）]", line)  # noqa: RUF001
            if match:
                left, right = match.group(1).strip(), match.group(2).strip()
                if _has_cjk(left) and not _has_cjk(right):
                    canonical = right
                elif _has_cjk(right) and not _has_cjk(left):
                    canonical = left
                else:
                    canonical = left
            else:
                canonical = line.split(":", 1)[0].strip()
            if canonical and not _has_cjk(canonical) and canonical.lower() not in seen:
                seen.add(canonical.lower())
                names.append(canonical)
        if not names:
            return ""
        name_list = "; ".join(names[:40])  # cap tránh prompt quá dài
        return (
            "\n- TÊN NHÂN VẬT (BẮT BUỘC NHẤT QUÁN): mỗi khi nhắc đến nhân vật trong "
            "Roster (kể cả nhãn 'speaker' và trong lời thoại), PHẢI dùng ĐÚNG tên tiếng "
            "đích đã cho, viết GIỐNG HỆT (hoa/thường, dấu); KHÔNG tự tạo cách phiên âm "
            "khác, KHÔNG đổi qua lại giữa các biến thể tên cho CÙNG một người xuyên suốt "
            "phim bộ. Nhân vật chưa có trong Roster: chọn MỘT cách dịch tên rồi giữ "
            f"nguyên. DANH SÁCH TÊN CHUẨN: {name_list}."
        )

    def _system_instruction(
        self, stage: TranslationStageConfig, context: TranslationContext, has_prior: bool
    ) -> str:
        overview = context.overview or "chưa có tóm tắt"
        characters = context.characters or "chưa có"
        glossary = (context.glossary or "").strip()
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
        )
        line_align = line_align + pronoun_rule

        # Chỉ dẫn súc tích — bản dịch sẽ qua TTS đọc theo mốc thời gian phụ đề.
        # Câu quá dài buộc TTS đọc nhanh/nén tốc độ → khó nghe, mất đồng bộ.
        concise = (
            "- SÚC TÍCH (nhưng KHÔNG được đánh đổi bằng nghĩa): bản dịch sẽ được TTS đọc "
            "khít vào thời lượng phụ đề. Câu dịch quá dài khiến TTS đọc gấp gáp. Vì vậy "
            "hãy: (1) bỏ từ đệm/thừa ('thì', 'là', 'mà', 'rằng', 'đã', 'các', 'những', "
            "'một cách', 'của việc' khi không cần); (2) ưu tiên từ "
            "ngắn, dùng từ Hán-Việt "
            "cô đọng khi hợp văn cảnh; (3) gộp lối nói vòng thành cụm "
            "gọn. Súc tích nghĩa "
            "là BỎ CHỮ THỪA, KHÔNG PHẢI bỏ nội dung. Giữ trọn ý nghĩa, không bịa thêm, "
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
                "- Nếu một dòng đã đúng, TRẢ LẠI Y NGUYÊN (không viết lại cho 'mượt').\n\n"
                "## ĐỊNH DẠNG\nChỉ xuất JSON hợp lệ cho lô hiện tại."
            )

        if stage.kind is TranslationStageKind.LITERAL:
            speaker_clause = (
                "gán nhãn người nói (speaker) nếu xác định được"
                if context.enable_tags
                else "để trống trường speaker (người dùng KHÔNG cần nhãn người nói)"
            )
            desc_clause = (
                "ghi ngắn gọn tiếng động/âm thanh mô tả nếu có vào trường description"
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
                f"## QUY TẮC\n{concise}{pronoun}{examples}"
                f"{line_align}"
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
                f"## QUY TẮC\n{desc_clause_style}{expansion}{concise}{pronoun}{examples}"
                f"{line_align}"
                "- Chỉ dựa vào nội dung được cung cấp, không thêm thông tin ngoài.\n\n"
                "## ĐỊNH DẠNG\nChỉ xuất JSON hợp lệ cho lô hiện tại."
            )

        locale_notes = stage.locale_notes.strip() or f"Theo chuẩn sử dụng tự nhiên trôi chảy thoát ý của {context.target_lang}"
        desc_clause_loc = (
            "" if context.include_desc else "- Để trống description.\n"
        )
        task = (
            "Bản địa hoá bản dịch hiện có cho tự nhiên trôi chảy thoát ý với người xem bản địa "
            "(đơn vị đo, xưng hô, cách gọi tên, thành ngữ, ca dao, tục ngữ, quy ước văn hoá)."
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
            f"## QUY TẮC\n{desc_clause_loc}{expansion}{concise}{pronoun}{examples}"
            f"{line_align}"
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
        # [v3.23.26] Theo best practice Gemini: với context lớn, đặt DỮ LIỆU TRƯỚC và
        # INSTRUCTION Ở CUỐI (neo suy luận vào dữ liệu phía trên). Trước đây instruction
        # đặt đầu → model dễ "quên" khi data dài. Câu lệnh cuối bắt đầu bằng "Dựa trên
        # các khối trên" để buộc model bám sát dữ liệu vừa đọc.
        instruction_text = (
            "Dựa trên các khối dữ liệu trên: đối chiếu trường 'original' (bản gốc) khi "
            "tinh chỉnh trường 'text', giữ đúng nghĩa gốc. Chỉ xử lý <current_batch>, "
            "trả JSON hợp lệ theo schema."
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
        prompt_blocks.append(_wrap_xml_block("instruction", instruction_text))
        prompt = "\n\n".join(prompt_blocks)
        # config được truyền vào từ translate_stage (xây dựng một lần)

        def _validator(data: dict[str, Any]) -> None:
            self._validate_batch(data, len(batch), start_idx)

        # [v3.23.19] Ước lượng token gồm cả VIDEO để quota manager điều tiết ĐÚNG.
        # Trước đây chỉ tính text → quota "mù" với video → acquire cho qua → server
        # báo 429 input_token vượt 250k. Nay cộng token video đính kèm.
        est_total = max(1, len(prompt) // 4) + max(0, video_token_estimate)

        try:
            data = self._call_gemini(
                model_name, prompt, config, _validator, cancel_cb=cancel_cb,
                video_files=video_files, est_tokens=est_total,
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
                    video_files=video_files, dual_payload=dual_payload,
                    video_token_estimate=video_token_estimate,
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
                video_files=video_files, dual_payload=dual_payload,
                video_token_estimate=video_token_estimate,
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
                video_files=video_files, dual_payload=dual_payload,
                video_token_estimate=video_token_estimate,
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
        dual_payload: bool,
        video_token_estimate: int = 0,
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
                video_files=video_files, dual_payload=dual_payload,
                video_token_estimate=video_token_estimate,
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

        deduped: list[TranslationLine] = []
        prev_text = ""
        for ln in source_lines:
            t = ln.text.strip()
            if t and t != prev_text:
                deduped.append(ln)
                prev_text = t

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
            "analyze_global_context: %d dòng gốc → %d dòng sau khử trùng → gửi AI.",
            len(source_lines), len(deduped),
        )

        # ── Định dạng nội dung ────────────────────────────────────────────
        # Chỉ giữ index + text, không cần timing (tiết kiệm ~30% token)
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
            f"\n• characters: Liệt kê TẤT CẢ nhân vật có tên xuất hiện. "
            f"Định dạng SONG NGỮ cho TỪNG nhân vật trên mỗi dòng:\n"
            f"  'Tên gốc (phiên âm/chuyển ngữ sang {target_lang}) [nam/nữ, ~tuổi]: vai trò: mô tả ngắn'\n"
            f"Ví dụ (phụ đề tiếng Trung → {target_lang}):\n"
            f"  '林昆 (Lâm Côn) [nam, ~25]: nhân vật chính: con trai tổng chủ, sở hữu hệ thống'\n"
            f"  '叶天 (Diệp Thiên) [nam, trung niên]: phản diện: nhân vật nguyên tác, tính cách độc ác'\n"
            "BẮT BUỘC ghi GIỚI TÍNH [nam/nữ] cho TỪNG nhân vật: hãy NGHE GIỌNG NÓI trong "
            "audio của video (cao độ, âm sắc) kết hợp hình ảnh để xác định chính xác — "
            "giới tính quyết định cách xưng hô (anh/em/cô/chú/ông/bà) khi dịch, sai giới "
            "tính là sai cả bản dịch. Nêu thêm QUAN HỆ vai vế nếu rõ (vd 'anh trai của X', "
            "'cấp trên của Y').\n"
            f"Nếu tên đã là latin (tiếng Anh, Pháp...) thì giữ nguyên nhưng theo định dạng (chức danh/chức vị + tên đầy đủ. Ví dụ: Tiến sĩ Jonh Smith, Giám đốc NASA Jared Isaacman,...), không cần phiên âm.\n"
            f"BẮT BUỘC: luôn ghi phiên âm {target_lang} cho tên CJK (Hán, Nhật, Hàn)."
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
            f"\n• glossary: Bảng thuật ngữ & từ viết tắt giúp DỊCH NHẤT QUÁN toàn phim. "
            f"Mỗi mục một dòng, 'định dạng gốc' => 'bản dịch {target_lang} chuẩn'. Gồm:"
            f"\n  - Thuật ngữ chuyên môn/đặc thù (khoa học, kỹ thuật, tổ chức, địa danh, "
            f"chiêu thức…) cần dịch GIỐNG NHAU mọi nơi."
            f"\n  - Từ viết tắt phổ biến: GIỮ NGUYÊN dạng viết tắt, kèm chú thích lần đầu. "
            f"Ví dụ: 'FBI => FBI (Cục Điều tra Liên bang Mỹ)', 'NASA => NASA', "
            f"'IBM => IBM', 'DNA => DNA (phân tử di truyền)'. "
            f"KHÔNG phiên âm/dịch nghĩa các viết tắt đã thông dụng quốc tế."
            f"\n  Nếu phim không có thuật ngữ đặc thù, để chuỗi rỗng."
        )

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
                f"Dùng phiên âm {target_lang} cho tên CJK; tên Latin giữ nguyên. TUYỆT "
                f"ĐỐI KHÔNG đoán bừa giới tính/danh tính: nếu không chắc, để trống. "
                f"CHỈ thêm dòng bạn TỰ TIN. Nếu chỉ xem một đoạn video, chỉ trả cues "
                f"cho các dòng THUỘC đoạn đó."
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

        video_handles = self._resolve_video_handles(video_refs)
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
        )
        cues_json = self._extract_cues_json(data) if with_visual_cues else ""
        return SubtitleContextAnalysis(
            source_lang=self._reconcile_source_lang(
                str(data.get("source_lang") or "").strip(), source_lines
            ),
            characters=str(data.get("characters") or "").strip(),
            overview=str(data.get("overview") or "").strip(),
            glossary=str(data.get("glossary") or "").strip(),
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
    ) -> SubtitleContextAnalysis:
        """[v3.23.23] Phân tích ngữ cảnh TUẦN TỰ TÍCH LUỸ qua từng đoạn video.

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
        def _run_segment(idx: int, *, is_retry: bool = False) -> bool:
            """Phân tích MỘT đoạn, hợp nhất vào ngữ cảnh tích luỹ. True nếu thành công."""
            nonlocal accumulated
            handle = video_handles[idx]
            ref = video_refs[idx] if idx < len(video_refs) else None
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
                ctx_block = (
                    "\n\nNGỮ CẢNH TÍCH LUỸ từ các đoạn TRƯỚC (hãy CẬP NHẬT, BỔ SUNG, "
                    "KHÔNG bỏ sót thông tin cũ — hợp nhất với quan sát đoạn này):\n"
                    f"• Nhân vật:\n{accumulated.characters}\n"
                    f"• Tóm tắt:\n{accumulated.overview}\n"
                    f"• Thuật ngữ:\n{accumulated.glossary}"
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
            try:
                data = self._call_gemini(
                    model_name, seg_prompt, config, _noop, cancel_cb=cancel_cb,
                    video_files=[handle], est_tokens=seg_est,
                    video_refs=[ref] if ref is not None else None,
                )
                current = SubtitleContextAnalysis(
                    source_lang=str(data.get("source_lang") or "").strip(),
                    characters=str(data.get("characters") or "").strip(),
                    overview=str(data.get("overview") or "").strip(),
                    glossary=str(data.get("glossary") or "").strip(),
                )
                # Tích luỹ: ưu tiên kết quả mới (đã gồm ngữ cảnh cũ); giữ trường cũ nếu
                # đoạn này trả rỗng (model thi thoảng bỏ trống 1 trường).
                accumulated = self._merge_sequential_analysis(accumulated, current)
                # Gộp cues của đoạn này (id duy nhất; đoạn sau ghi đè nếu trùng id).
                if with_visual_cues:
                    seg_cues_json = self._extract_cues_json(data)
                    if seg_cues_json:
                        for c in json.loads(seg_cues_json):
                            merged_cues[int(c["id"])] = c
                return True
            except SubtitleTranslationError as exc:
                logger.warning(
                    "Phân tích đoạn %d thất bại%s, giữ ngữ cảnh tích luỹ hiện có: %s",
                    idx + 1, " (cả lần thử lại)" if is_retry else "", exc,
                )
                return False

        failed_segments: list[int] = []
        for idx in range(n):
            if cancel_cb is not None and cancel_cb():
                raise TranslationCancelledError("Người dùng đã huỷ khi phân tích ngữ cảnh.")
            if not _run_segment(idx):
                failed_segments.append(idx)

        # [v3.23.155] SECOND-CHANCE: 503/504 là quá tải TẠM THỜI phía Gemini — sau khi
        # đi hết các đoạn còn lại (vài phút), thử lại MỘT lượt các đoạn đã hỏng. Lúc này
        # ngữ cảnh tích luỹ đã GIÀU HƠN (gồm các đoạn sau) nên bản hợp nhất khi thử lại
        # thành công còn tốt hơn; đặc biệt cứu được đoạn MỞ PHIM (giới thiệu nhân vật).
        if failed_segments and not (cancel_cb is not None and cancel_cb()):
            logger.info(
                "Thử lại %d đoạn phân tích thất bại (%s) sau khi hoàn tất các đoạn khác.",
                len(failed_segments), ", ".join(str(i + 1) for i in failed_segments),
            )
            for idx in failed_segments:
                if cancel_cb is not None and cancel_cb():
                    raise TranslationCancelledError(
                        "Người dùng đã huỷ khi phân tích ngữ cảnh."
                    )
                _run_segment(idx, is_retry=True)

        if accumulated is not None:
            if with_visual_cues and merged_cues:
                import dataclasses
                cues_list = sorted(merged_cues.values(), key=lambda c: c["id"])
                cues_json = json.dumps(cues_list, ensure_ascii=False, separators=(",", ":"))
                logger.info("Gộp %d Visual Cues từ %d đoạn phân tích.", len(cues_list), n)
                accumulated = dataclasses.replace(accumulated, visual_cues=cues_json)
            return accumulated

        # Mọi đoạn đều lỗi → thử lần cuối CHỈ với text (không video).
        data = self._call_gemini(
            model_name, prompt, config, _noop, cancel_cb=cancel_cb,
            video_files=None, est_tokens=base_est,
        )
        return SubtitleContextAnalysis(
            source_lang=str(data.get("source_lang") or "").strip(),
            characters=str(data.get("characters") or "").strip(),
            overview=str(data.get("overview") or "").strip(),
            glossary=str(data.get("glossary") or "").strip(),
        )

    @staticmethod
    def _merge_sequential_analysis(
        prev: SubtitleContextAnalysis | None, current: SubtitleContextAnalysis
    ) -> SubtitleContextAnalysis:
        """Hợp nhất kết quả tuần tự: ưu tiên bản mới, giữ trường cũ nếu mới rỗng."""
        if prev is None:
            return current
        return SubtitleContextAnalysis(
            source_lang=current.source_lang or prev.source_lang,
            characters=current.characters or prev.characters,
            overview=current.overview or prev.overview,
            glossary=current.glossary or prev.glossary,
        )

    @staticmethod
    def _parse_visual_cue_items(items: list[dict[str, Any]]) -> list[VisualCue]:
        """[JSON Minification] Giải nén item rút gọn (id/spk/to/cue) → :class:`VisualCue`.

        Khoá JSON được nén để AI nhả được hàng ngàn dòng mà không chạm trần 8192
        token output: ``id``→line_no, ``spk``→speaker, ``to``→addressee, ``cue``→scene.
        Hàm thuần, dễ test.
        """
        cues: list[VisualCue] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            line_no = _safe_int(item.get("id", item.get("line_no", 0)))
            if line_no <= 0:
                continue
            cues.append(
                VisualCue(
                    line_no=line_no,
                    speaker=str(item.get("spk", item.get("speaker", "")) or "").strip(),
                    addressee=str(item.get("to", item.get("addressee", "")) or "").strip(),
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

        Returns:
            Danh sách :class:`VisualCue` theo thứ tự dòng.
        """
        self._ensure_available()
        if not source_lines:
            return []

        system_instr = (
            "Bạn là GIÁM SÁT VIDEO (Vision Director). Dựa trên video kèm theo và "
            "danh sách phụ đề, với MỖI dòng hãy xác định ba điều: ai đang nói (spk), "
            "nói với ai (to), và bối cảnh + thái độ/cảm xúc ngắn gọn (cue). "
            f"Ghi vai vế bằng {target_lang} (vd 'Nam đệ tử', 'Nữ tỳ', 'Sư phụ'). "
            "TUYỆT ĐỐI KHÔNG đoán bừa giới tính: nếu không rõ, để trống thay vì đoán. "
            "Trả JSON rút gọn theo schema, mỗi phần tử gồm id (số dòng), spk, to, cue. "
            "BẮT BUỘC giữ nguyên id, đủ mọi dòng trong lô, không gộp/bỏ dòng."
        )
        # [Thinking ưu tiên BẬT] giúp suy luận "ai nói với ai" chính xác hơn; nếu
        # model không hỗ trợ, _call_gemini tự bỏ thinking và thử lại.
        config = self._build_config(
            0.2, _VISUAL_CUES_SCHEMA, system_instr,
            enable_thinking=True, thinking_budget=-1,
            model_name=model_name,
            thinking_level=getattr(self, "_analysis_thinking_level", "medium"),
            media_resolution=getattr(self, "_analysis_media_resolution", "medium"),
        )
        video_handles = self._resolve_video_handles(video_refs)

        def _noop_validator(_data: dict[str, Any]) -> None:
            pass

        all_cues: list[VisualCue] = []
        total = len(source_lines)
        total_batches = max(1, (total + batch_size - 1) // batch_size)
        for batch_idx, start in enumerate(range(0, total, batch_size)):
            if cancel_cb is not None and cancel_cb():
                raise TranslationCancelledError("Người dùng đã huỷ phân tích Visual Cues.")
            batch = source_lines[start:start + batch_size]
            payload = [{"id": ln.index, "text": ln.text} for ln in batch]
            prompt = "\n\n".join([
                _wrap_xml_block("instruction", "Phân tích lô hiện tại, trả JSON rút gọn."),
                _wrap_xml_block("lines", payload),
            ])
            # Chọn đoạn video phủ đúng khoảng thời gian của lô (tiết kiệm token).
            batch_handles = video_handles
            if video_refs:
                t_start = min(ln.start_ms for ln in batch) / 1000.0
                t_end = max(ln.end_ms for ln in batch) / 1000.0
                covering = [
                    r for r in video_refs
                    if getattr(r, "start_sec", 0.0) < t_end
                    and getattr(r, "end_sec", 0.0) > t_start
                ]
                batch_handles = self._resolve_video_handles(covering) or video_handles

            data = self._call_gemini(
                model_name, prompt, config, _noop_validator,
                cancel_cb=cancel_cb, video_files=batch_handles,
            )
            items = data.get("cues", []) if data else []
            all_cues.extend(self._parse_visual_cue_items(items))

            if progress_cb is not None:
                progress_cb((batch_idx + 1) / total_batches)
            # [Chống Rate Limit] Nghỉ giữa các lô, trừ lô cuối.
            if sleep_between_s > 0 and batch_idx < total_batches - 1:
                if not self._interruptible_sleep(sleep_between_s, cancel_cb):
                    raise TranslationCancelledError("Đã huỷ trong lúc nghỉ giữa các lô.")
        return all_cues

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
    @classmethod
    def _length_hint(cls, line: TranslationLine, next_start_ms: int | None = None) -> int:
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
        return max(
            MIN_CHAR_BUDGET,
            syllable_budget_to_chars(readable_syllable_budget(usable_s)),
        )

    @classmethod
    def _line_to_payload(
        cls, line: TranslationLine, next_start_ms: int | None = None
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
        hint = cls._length_hint(line, next_start_ms)
        if hint:
            data["max_chars"] = hint
        return data

    @classmethod
    def _line_to_dual_payload(
        cls,
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
        hint = cls._length_hint(line, next_start_ms)
        if hint:
            data["max_chars"] = hint
        # [v3.23.225] Cờ TỰ SỬA: dòng bị lưới ``under_translation_guard`` nghi DỊCH THIẾU
        # ở khâu TRƯỚC. Giai đoạn này DÙ SAO CŨNG gọi API và DÙ SAO CŨNG có 'original'
        # trong payload kép -> tận dụng để BỔ SUNG phần bị rớt, không tốn thêm lượt gọi.
        if needs_expansion:
            data["needs_expansion"] = True
        return data

    @staticmethod
    def _context_window(
        lines: list[TranslationLine], start_idx: int, batch_len: int, ctx_size: int
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if ctx_size <= 0:
            return [], []
        before = lines[max(0, start_idx - ctx_size):start_idx]
        after_start = start_idx + batch_len
        after = lines[after_start:after_start + ctx_size]
        to_payload = GeminiSubtitleTranslator._line_to_payload
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

        # [v3.23.18] Model trả DƯ nhiều dòng (vd 95 cho batch 25) = sai cấu trúc thật
        # (model phớt lờ batch, trả cả phụ đề/ngữ cảnh). Bắt SỚM trước khi xét line_no,
        # vì tập line_no dư có thể "chứa" đủ tập mong đợi gây chấp nhận nhầm.
        if actual_count > expected_count + 2:
            raise _BatchValidationError(
                f"Model trả DƯ {actual_count} dòng (kỳ vọng {expected_count}) — "
                "sai cấu trúc, cần chia nhỏ."
            )

        expected_line_nos = {start_idx + offset + 1 for offset in range(expected_count)}

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
            patch_threshold = max(2, expected_count * 15 // 100)
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

    @staticmethod
    def _batch_context(
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
        source_before, source_after = GeminiSubtitleTranslator._context_window(
            source_lines, start, batch_len, ctx_size
        )
        history_before = (
            [GeminiSubtitleTranslator._line_to_payload(line) for line in prior_output[-ctx_size:]]
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
    ) -> list[TranslationLine]:
        """[v3.23.149] Dịch các batch SONG SONG theo ĐỢT (wave) có neo lịch sử.

        - Mỗi đợt gồm ``parallel`` batch chạy đồng thời; ``history_before`` của MỌI
          batch trong đợt neo vào đuôi output các đợt ĐÃ CHỐT (giữ mạch xưng hô/giọng
          điệu; ngữ cảnh nguồn trước/sau vẫn đầy đủ theo từng batch).
        - Quota an toàn khi song song: mỗi request tự ``acquire`` (đặt chỗ RPM/TPM,
          thread-safe) nên tổng lưu lượng luôn được điều tiết đúng; cooldown 429 áp
          cho mọi luồng cùng key; xoay key được ``_state_lock`` bảo vệ.
        - Kết quả ghép ĐÚNG THỨ TỰ batch (không phụ thuộc thứ tự hoàn thành).

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
                for start in wave:
                    current_batch = input_lines[start:start + batch_size]
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
                wave_results: dict[int, list[TranslationLine]] = {}
                first_error: BaseException | None = None
                for future in as_completed(future_to_start):
                    error = future.exception()
                    if error is not None:
                        if first_error is None:
                            first_error = error
                        continue
                    wave_results[future_to_start[future]] = future.result()
                    done_batches += 1
                    if progress_cb is not None:
                        progress_cb(done_batches / total_batches)
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
    ) -> list[TranslationLine]:
        self._ensure_available()
        if not input_lines:
            return []

        is_preprocess = stage.kind is TranslationStageKind.PREPROCESS
        is_literal = stage.kind is TranslationStageKind.LITERAL
        # [Anti-Chinese Whispers] Seed bản gốc bất biến ở giai đoạn dịch ĐẦU TIÊN
        # (khi text vẫn là ngôn ngữ gốc). Các stage sau giữ nguyên để đối chiếu.
        if not has_prior_translation and not is_preprocess:
            input_lines = [
                ln if ln.original_text else replace(ln, original_text=ln.text)
                for ln in input_lines
            ]
        # [Dual Payload] Bật truyền kép cho khâu tinh chỉnh/bản địa hoá (có bản nháp):
        # gửi kèm bản gốc để AI đối chiếu, chống làm sai lệch nghĩa.
        use_dual_payload = has_prior_translation and stage.kind in (
            TranslationStageKind.STYLE,
            TranslationStageKind.LOCALIZE,
        )
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
        reindex_map: dict[int, int] = {}
        if noise_indices:
            input_lines = [ln for ln in input_lines if ln.index not in noise_indices]
            source_lines = [ln for ln in source_lines if ln.index not in noise_indices]
            if not input_lines:
                return list(full_input_lines)
            # [v3.23.133] Sau khi LỌC dòng nhiễu, index còn LỖ HỔNG (vd 1,2,4,5). Lõi
            # validate/renumber giả định line_no LIÊN TỤC theo vị trí → lỗ hổng khiến
            # model trả đúng index thực vẫn bị tưởng "lệch số" rồi đánh số lại theo vị
            # trí → DỒN LỆCH bản dịch. Khắc phục: đánh số lại LIÊN TỤC [1..N] để gửi
            # model, rồi ÁNH XẠ NGƯỢC về index gốc sau khi dịch. input_lines và
            # source_lines song song (cùng tập index, cùng thứ tự) nên nhất quán.
            reindex_map = {pos + 1: ln.index for pos, ln in enumerate(input_lines)}
            input_lines = [
                replace(ln, index=pos + 1) for pos, ln in enumerate(input_lines)
            ]
            source_lines = [
                replace(ln, index=pos + 1) for pos, ln in enumerate(source_lines)
            ]

        response_schema = self._response_schema_for(stage, context)
        system_instruction = self._system_instruction(stage, context, has_prior_translation)
        batch_size = max(1, stage.batch_size)
        # [v3.23.131] TỰ GIỚI HẠN lô cho model "lite": các model nhẹ (vd
        # gemini-3.1-flash-lite) trả kém ổn định khi lô quá lớn — hay rớt/đánh số lệch
        # dòng, sinh nhiều lần vá/chia đôi (chậm) và đôi khi lệch nội dung. Trần ~60 dòng
        # giúp ổn định rõ rệt mà vẫn nhanh. Không đụng tới model đầy đủ (flash/pro).
        model_lower = (stage.model_name or "").lower()
        if "lite" in model_lower and batch_size > _LITE_MODEL_BATCH_CAP:
            logger.info(
                "Model '%s' là bản lite → giảm lô %d -> %d cho ổn định (chống rớt dòng).",
                stage.model_name, batch_size, _LITE_MODEL_BATCH_CAP,
            )
            batch_size = _LITE_MODEL_BATCH_CAP
        ctx_size = max(0, stage.context_size)
        total = len(input_lines)
        total_batches = max(1, (total + batch_size - 1) // batch_size)
        # Tạo config MỘT LẦN rồi truyền vào _translate_single_batch
        # (thay vì mỗi batch tự build lại — tiết kiệm CPU và nhất quán thinking settings)
        _prebuilt_config = self._build_config(
            stage.temperature,
            response_schema,
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
            )
            # Trộn lại dòng rác + ánh xạ ngược index (chung với đường tuần tự bên dưới).
            if noise_indices:
                output = [
                    replace(ln, index=reindex_map.get(ln.index, ln.index)) for ln in output
                ]
                translated_by_index = {ln.index: ln for ln in output}
                return [translated_by_index.get(ln.index, ln) for ln in full_input_lines]
            return output

        for batch_idx, start in enumerate(range(0, total, batch_size)):
            if cancel_cb is not None and cancel_cb():
                raise TranslationCancelledError("Người dùng đã huỷ tiến trình dịch.")

            current_batch = input_lines[start:start + batch_size]
            source_before, source_after, history_before = self._batch_context(
                source_lines, start, len(current_batch), ctx_size, output
            )

            # Chọn đoạn video phủ đúng khoảng thời gian của batch này (nếu bật đính
            # video cho giai đoạn). Phim cắt nhiều đoạn → mỗi batch chỉ gửi (các) đoạn
            # giao với khoảng [đầu, cuối] của batch, tránh gửi thừa token.
            batch_video_handles: list[Any] | None = None
            batch_video_tokens = 0
            if attach_video and video_refs:
                t_start = min(ln.start_ms for ln in current_batch) / 1000.0
                t_end = max(ln.end_ms for ln in current_batch) / 1000.0
                covering = [
                    r for r in video_refs
                    if getattr(r, "start_sec", 0.0) < t_end
                    and getattr(r, "end_sec", 0.0) > t_start
                ]
                batch_video_handles = self._resolve_video_handles(covering)
                # [v3.23.19] Ước lượng token video của các đoạn phủ (thời lượng × 100)
                # để quota điều tiết đúng. Cảnh báo nếu một batch đính video vượt ngưỡng
                # an toàn TPM (250k free tier) — đính cả đoạn dài rất tốn token.
                batch_video_tokens = sum(
                    int(max(0.0, getattr(r, "end_sec", 0.0) - getattr(r, "start_sec", 0.0))
                        * _VIDEO_TOKENS_PER_SEC)
                    for r in covering
                )
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
                config=_prebuilt_config,
                model_name=stage.model_name,
                cancel_cb=cancel_cb,
                _ctx_size=ctx_size,
                video_files=batch_video_handles,
                dual_payload=use_dual_payload,
                expansion_indices=expansion_indices,
                video_token_estimate=batch_video_tokens,
            )
            output.extend(batch_result)

            if progress_cb is not None:
                progress_cb((batch_idx + 1) / total_batches)

        # [v3.23.107] Trộn lại dòng rác (nguyên văn) đúng vị trí theo index.
        if noise_indices:
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


__all__ = ["GeminiSubtitleTranslator"]
