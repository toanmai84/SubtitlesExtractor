"""[v3.23.77] Dựng "gói chẩn đoán dịch" — toàn bộ dữ liệu liên quan tới một lần dịch.

Mục đích: người dùng xuất một file JSON duy nhất (phụ đề gốc, kết quả dịch cuối, từng giai
đoạn dịch, phân tích ngữ cảnh: nhân vật/tóm tắt/glossary/visual cues, ngữ cảnh phim bộ,
tệp video cloud…) để gửi đi phân tích, khắc phục lỗi và cải thiện CHẤT LƯỢNG dịch.

Hàm ``build_diagnostics_bundle`` là HÀM THUẦN: nhận dữ liệu đã thu thập, trả dict
JSON-serializable; không I/O, không phụ thuộc Qt — dễ kiểm thử.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from itertools import pairwise
from typing import Any, Protocol


class _EventLike(Protocol):
    index: int
    text: str

    @property
    def start_sec(self) -> float: ...

    @property
    def end_sec(self) -> float: ...


_SCHEMA = "subtitles_extractor.translation_diagnostics"
_SCHEMA_VERSION = 3

# [v3.23.83] Calibrate cờ length_anomaly dựa trên dữ liệu thật (zh->vi): tiếng Trung cực
# ngắn gọn (1 Hán tự ~ 1 từ) nên so độ dài thô với tiếng Việt phồng giả tạo. Khắc phục:
#   1. Bỏ tag người nói "[Tên:]" (chỉ có ở bản dịch) trước khi đo.
#   2. Quy đổi độ dài nguồn CJK theo hệ số giãn (1 ký tự CJK ~ vài ký tự Latinh/Việt).
#   3. Bỏ qua nguồn quá ngắn (tỉ lệ không đáng tin với câu 1-2 ký tự).
_SPEAKER_TAG_RE = re.compile(r"^\s*\[[^\]]*\]\s*")
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u30ff\uac00-\ud7af]")
_CJK_EXPANSION = 3.0  # 1 ký tự CJK quy đổi ~ 3 ký tự Latinh/Việt
# [v3.23.156] 8 -> 12: câu cực ngắn ("Yes, I am." -> "Có.") có tỉ lệ vô nghĩa.
_MIN_EFFECTIVE_LEN = 12.0  # bỏ qua nguồn hiệu dụng ngắn hơn ngưỡng này
# [v3.23.156] 0.5 -> 0.35: bản dịch NGẮN có chủ đích là tính năng (quy tắc "súc tích
# cho TTS" trong prompt); dữ liệu thật (The Hot Spot) cho thấy ratio 0.4-0.5 vẫn là
# bản dịch tốt. Chỉ cảnh báo khi mất quá 2/3 nội dung (thật sự khả nghi nuốt ý).
_LENGTH_RATIO_LOW = 0.35
_LENGTH_RATIO_HIGH = 2.5
# Token coi là "không dịch được": số/dấu, hoặc từ viết Hoa chữ đầu (tên riêng).
_PROPER_TOKEN_RE = re.compile(r"^(?:[A-Z][\w'-]*|[0-9]+[\w.]*)$")
_WORD_SPLIT_RE = re.compile(r"[\s,.!?;:…\-—\"'()\[\]]+")


def _strip_speaker_tag(text: str) -> str:
    """Bỏ tiền tố tag người nói ``[Tên:]`` (chỉ xuất hiện ở bản dịch)."""
    return _SPEAKER_TAG_RE.sub("", text).strip()


def _effective_source_length(text: str) -> float:
    """Độ dài nguồn 'hiệu dụng': ký tự CJK nhân hệ số giãn, ký tự khác đếm 1.

    Nhờ đó so độ dài giữa nguồn CJK (rất đặc) và bản dịch Latinh/Việt công bằng hơn.

    [v3.23.156] Nguồn SONG NGỮ (hardsub 2 dòng CÙNG NGHĨA: một dòng CJK + một dòng
    Latin — vd "∮须得别处觅芳草∮" + "∮ Gotta find my baby ∮"): cộng cả hai làm độ
    dài nguồn PHỒNG ĐÔI -> bản dịch (chỉ dịch MỘT nghĩa) bị flag ngắn oan. Khi phát
    hiện đồng thời dòng CJK-dominant và dòng non-CJK, lấy MAX theo dòng thay vì tổng.
    """

    def _one_line(line: str) -> tuple[float, bool]:
        cjk = sum(1 for ch in line if _CJK_RE.match(ch))
        effective = cjk * _CJK_EXPANSION + (len(line) - cjk)
        stripped = line.strip()
        is_cjk_dominant = bool(stripped) and cjk >= max(1, len(stripped) // 2)
        return effective, is_cjk_dominant

    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) >= 2:
        measured = [_one_line(ln) for ln in lines]
        has_cjk_line = any(dom for _, dom in measured)
        has_latin_line = any(not dom for _, dom in measured)
        if has_cjk_line and has_latin_line:
            return max(eff for eff, _ in measured)
    effective, _ = _one_line(text)
    return effective


def _is_untranslatable_source(source_text: str) -> bool:
    """[v3.23.156] True nếu nguồn 'không có gì để dịch' (tên riêng/số/dấu câu).

    Dữ liệu thật (The Hot Spot): 25/25 cờ identical_to_source đều là các dòng như
    "Harry.", "Gloria...", "357...", "Madox. Harry Madox." — dịch GIỮ NGUYÊN là ĐÚNG.
    Tiêu chí: <= 4 token và MỌI token đều là tên riêng (Hoa chữ đầu) hoặc số.
    """
    tokens = [t for t in _WORD_SPLIT_RE.split(source_text) if t]
    if not tokens:
        return True  # toàn dấu câu/ký hiệu
    if len(tokens) > 4:
        return False
    return all(_PROPER_TOKEN_RE.match(token) for token in tokens)


def detect_quality_flags(
    source_events: Iterable[_EventLike],
    translated_events: Iterable[_EventLike],
) -> dict[str, Any]:
    """Phát hiện (heuristic) các dòng dịch KHẢ NGHI để soi nhanh chất lượng.

    Cảnh báo: đây là HEURISTIC — có thể dương tính giả (tên riêng/số/"OK" vốn giữ
    nguyên, hoặc tỉ lệ độ dài lệch do khác biệt ngôn ngữ). Chỉ để khoanh vùng.

    Args:
        source_events: Phụ đề gốc.
        translated_events: Phụ đề sau dịch.

    Returns:
        Dict gồm các danh sách chỉ số dòng khả nghi và bộ đếm tổng quan.
    """
    source_by_index = {int(e.index): e.text for e in source_events}
    translated_list = list(translated_events)

    identical_to_source: list[int] = []
    empty_translation: list[int] = []
    length_anomaly: list[dict[str, Any]] = []

    for event in translated_list:
        idx = int(event.index)
        source_text = source_by_index.get(idx)
        if source_text is None:
            continue
        src = source_text.strip()
        dst_full = event.text.strip()
        if not src:
            continue
        if not dst_full:
            empty_translation.append(idx)
            continue
        # Bỏ tag người nói khỏi bản dịch trước khi đối chiếu/đo độ dài.
        dst = _strip_speaker_tag(dst_full)
        if dst and dst == src and not _is_untranslatable_source(src):
            identical_to_source.append(idx)
        # Cờ lệch độ dài: CJK-aware + bỏ qua nguồn quá ngắn (tránh dương tính giả).
        eff_src = _effective_source_length(src)
        if dst and eff_src >= _MIN_EFFECTIVE_LEN:
            ratio = len(dst) / eff_src
            if ratio < _LENGTH_RATIO_LOW or ratio > _LENGTH_RATIO_HIGH:
                length_anomaly.append({"index": idx, "length_ratio": round(ratio, 2)})

    # [v3.23.139] DÒNG DỊCH TRÙNG LIỀN KỀ: dấu hiệu model bị "lệch-gộp ngữ nghĩa" — khi
    # một câu bị tách qua nhiều dòng, model dồn cả câu vào dòng đầu rồi KÉO nội dung dòng
    # sau lên, làm hai dòng liền kề có bản dịch TRÙNG nhau dù NGUỒN KHÁC nhau. Cờ này giúp
    # khoanh vùng lỗi căn dòng mà cờ đếm/độ dài không bắt được (số dòng vẫn khớp).
    duplicate_adjacent: list[dict[str, Any]] = []
    trans_sorted = sorted(
        ((int(e.index), e.text) for e in translated_list), key=lambda pair: pair[0]
    )
    for (idx1, text1), (idx2, text2) in pairwise(trans_sorted):
        dst1 = _strip_speaker_tag((text1 or "").strip())
        dst2 = _strip_speaker_tag((text2 or "").strip())
        if not dst1 or dst1 != dst2:
            continue
        src1 = (source_by_index.get(idx1) or "").strip()
        src2 = (source_by_index.get(idx2) or "").strip()
        if src1 != src2:  # nguồn khác nhau mà dịch giống hệt -> nghi lệch dòng
            duplicate_adjacent.append({"indices": [idx1, idx2]})

    return {
        "note": (
            "Heuristic — length_anomaly đã CJK-aware (quy đổi độ dài nguồn CJK) và "
            "bỏ tag người nói; vẫn có thể dương tính giả (tên riêng/số giữ nguyên). "
            "duplicate_adjacent: 2 dòng liền kề dịch trùng dù nguồn khác."
        ),
        "identical_to_source_indices": identical_to_source,
        "empty_translation_indices": empty_translation,
        "length_anomaly": length_anomaly,
        "duplicate_adjacent": duplicate_adjacent,
        "counts": {
            "identical_to_source": len(identical_to_source),
            "empty_translation": len(empty_translation),
            "length_anomaly": len(length_anomaly),
            "duplicate_adjacent": len(duplicate_adjacent),
        },
    }


def _event_to_dict(event: _EventLike) -> dict[str, Any]:
    confidence = getattr(event, "confidence", None)
    conf_value: float | None
    try:
        conf_value = round(float(confidence), 4) if confidence is not None else None
    except (TypeError, ValueError):
        conf_value = None
    return {
        "index": int(event.index),
        "start_sec": round(float(event.start_sec), 3),
        "end_sec": round(float(event.end_sec), 3),
        "text": event.text,
        "confidence": conf_value,
    }


def _parse_stage_lines(lines_json: str) -> Any:
    """Giải mã ``lines_json`` (chuỗi JSON) thành đối tượng; lỗi → trả raw + cờ lỗi."""
    if not lines_json:
        return []
    try:
        return json.loads(lines_json)
    except (json.JSONDecodeError, TypeError):
        return {"_parse_error": True, "raw": lines_json}


def build_diagnostics_bundle(
    *,
    app_version: str,
    exported_at: str,
    video_path: str,
    source_events: Iterable[_EventLike],
    translated_events: Iterable[_EventLike],
    session: Any | None,
    series_context: Any | None = None,
) -> dict[str, Any]:
    """Gom toàn bộ dữ liệu một lần dịch thành dict JSON-serializable.

    Args:
        app_version: Phiên bản ứng dụng (để biết bản nào sinh dữ liệu).
        exported_at: Mốc thời gian xuất (ISO-8601).
        video_path: Đường dẫn video nguồn (str).
        source_events: Phụ đề GỐC (trước dịch).
        translated_events: Phụ đề KẾT QUẢ (sau dịch, bản hiện hành).
        session: ``TranslationSession`` (phân tích + giai đoạn) hoặc ``None``.
        series_context: Ngữ cảnh phim bộ (glossary/roster/tóm tắt chung) hoặc ``None``.

    Returns:
        Dict mô tả đầy đủ quá trình dịch, an toàn để ``json.dump``.
    """
    from pathlib import PurePath

    source_events = list(source_events)
    translated_events = list(translated_events)
    source_list = [_event_to_dict(e) for e in source_events]
    translated_list = [_event_to_dict(e) for e in translated_events]
    quality_flags = detect_quality_flags(source_events, translated_events)

    analysis: dict[str, Any] = {}
    stages: list[dict[str, Any]] = []
    cloud_files: list[dict[str, Any]] = []
    if session is not None:
        analysis = {
            "source_lang": getattr(session, "analysis_source_lang", ""),
            "target_lang": getattr(session, "analysis_target_lang", ""),
            "characters": getattr(session, "analysis_characters", ""),
            "overview": getattr(session, "analysis_overview", ""),
            "glossary": getattr(session, "analysis_glossary", ""),
            "visual_cues": getattr(session, "analysis_visual_cues", ""),
            "input_hash": getattr(session, "analysis_input_hash", ""),
        }
        for stage in getattr(session, "stages", ()) or ():
            stages.append(
                {
                    "stage_id": getattr(stage, "stage_id", ""),
                    "input_hash": getattr(stage, "input_hash", ""),
                    "completed_at": getattr(stage, "completed_at", ""),
                    "lines": _parse_stage_lines(getattr(stage, "lines_json", "")),
                }
            )
        for cloud in getattr(session, "cloud_files", ()) or ():
            cloud_files.append(
                {
                    "remote_name": getattr(cloud, "remote_name", ""),
                    "start_sec": getattr(cloud, "start_sec", 0.0),
                    "end_sec": getattr(cloud, "end_sec", 0.0),
                    "uploaded_at": getattr(cloud, "uploaded_at", ""),
                }
            )

    series: dict[str, Any] = {}
    if series_context is not None:
        series = {
            "glossary": getattr(series_context, "glossary", ""),
            "characters": getattr(series_context, "characters", ""),
            "overview": getattr(series_context, "overview", ""),
        }

    return {
        "schema": _SCHEMA,
        "schema_version": _SCHEMA_VERSION,
        "app_version": app_version,
        "exported_at": exported_at,
        "video": {
            "path": video_path,
            "filename": PurePath(video_path).name if video_path else "",
        },
        "languages": {
            "source": analysis.get("source_lang", ""),
            "target": analysis.get("target_lang", ""),
        },
        "counts": {
            "source_events": len(source_list),
            "translated_events": len(translated_list),
            "stages": len(stages),
            "cloud_files": len(cloud_files),
        },
        "analysis": analysis,
        "series_context": series,
        "quality_flags": quality_flags,
        "cloud_files": cloud_files,
        "source_events": source_list,
        "final_translation": translated_list,
        "stages": stages,
    }
