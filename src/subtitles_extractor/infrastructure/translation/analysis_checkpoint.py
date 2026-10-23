"""[v3.23.564] Lưu TIẾN ĐỘ phân tích ngữ cảnh theo từng đoạn video để chạy tiếp được.

Nhật ký thật v3.23.563: phân tích bằng ``gemini-3.8-flash`` (model mạnh nhất, hạn mức 20
request/ngày) chạy 50 phút, xong 20/22 đoạn; tới lượt thử lại cuối phiên thì hết hạn mức
ngày → ``QuotaExhaustedError`` thoát ra ngoài vòng lặp và **toàn bộ** kết quả bị vứt. Phim
cần 22 request, hạn mức chỉ 20: không lưu tiến độ thì model mạnh nhất KHÔNG BAO GIỜ phân
tích trọn phim được, dù người dùng chạy lại bao nhiêu ngày.

Mô-đun này lưu sau MỖI đoạn thành công: các đoạn đã xong, ngữ cảnh tích luỹ và Visual Cues
đã gộp. Lần chạy sau (cùng phụ đề, cùng cài đặt, cùng cách cắt đoạn) chỉ gửi các đoạn còn
thiếu.

[v3.23.593] Tiến độ KHÔNG còn gắn với model: khi Gemini quá tải, ứng dụng khuyên "bấm Phân
tích lại sau ít phút, hoặc đổi model phân tích — các đoạn đã xong được giữ", nhưng khoá
tiến độ lại gồm cả tên model nên đổi model là làm lại từ đoạn 1. Nhật ký thật v3.23.592
(phim 17 đoạn, bốn ngày quá tải): người dùng làm đúng lời khuyên, đổi qua lại bốn model;
37 request thành công cho một bản phân tích cần 17 — 20 request (54%) bỏ đi vì phải làm
lại các đoạn đã xong ở model khác. Mỗi đoạn nay ghi kèm model đã phân tích nó.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from subtitles_extractor.domain.ports.subtitle_translator_port import (
    SubtitleContextAnalysis,
)

logger = logging.getLogger(__name__)

#: Tiến độ cũ hơn mốc này bị bỏ qua — phụ đề/ý đồ người dùng có thể đã đổi.
CHECKPOINT_MAX_AGE_S = 7 * 24 * 3600

_SCHEMA = "subtitles_extractor.analysis_checkpoint/1"

#: [v3.23.593] Đứng chỗ tên model trong khoá tiến độ dùng chung cho mọi model.
SHARED_PROGRESS_MODEL = "*"


@dataclass
class AnalysisProgress:
    """Tiến độ phân tích tuần tự đã lưu.

    Attributes:
        done_segments: Chỉ số (0-based) các đoạn đã phân tích thành công.
        accumulated: Ngữ cảnh tích luỹ sau các đoạn đã xong (None nếu chưa có).
        cues: Visual Cues đã gộp, khoá theo số dòng.
        segment_summaries: [v3.23.566] Diễn biến chi tiết, khoá theo chỉ số đoạn.
        segment_models: [v3.23.593] Model đã phân tích từng đoạn, khoá theo chỉ số đoạn
            (rỗng với tiến độ lưu bởi bản cũ).
    """

    done_segments: set[int] = field(default_factory=set)
    accumulated: SubtitleContextAnalysis | None = None
    cues: dict[int, dict[str, Any]] = field(default_factory=dict)
    segment_summaries: dict[int, str] = field(default_factory=dict)
    segment_models: dict[int, str] = field(default_factory=dict)


def analysis_checkpoint_key(*parts: str) -> str:
    """Khoá tiến độ: băm mọi yếu tố làm thay đổi kết quả phân tích (hàm thuần).

    Args:
        *parts: Model, prompt, system instruction, cài đặt, ranh giới các đoạn…

    Returns:
        Chuỗi hex 24 ký tự.
    """
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()
    return digest[:24]


def describe_segment_models(segment_models: dict[int, str], current_model: str) -> str:
    """Ghi chú đoạn nào do model KHÁC phân tích, để đưa vào nhật ký (hàm thuần).

    Args:
        segment_models: ``chỉ số đoạn (0-based) → model``.
        current_model: Model của lần chạy này.

    Returns:
        Vd ``" (đoạn 1-10: gemini-3.8-flash)"``; rỗng khi mọi đoạn đều do
        ``current_model`` phân tích hoặc không rõ model.
    """
    groups: list[tuple[int, int, str]] = []
    for index in sorted(segment_models):
        model = segment_models[index]
        if groups and groups[-1][2] == model and groups[-1][1] == index - 1:
            groups[-1] = (groups[-1][0], index, model)
        else:
            groups.append((index, index, model))
    if all(model == current_model for _first, _last, model in groups):
        return ""
    parts = [
        f"đoạn {first + 1}" + (f"-{last + 1}" if last > first else "") + f": {model}"
        for first, last, model in groups
    ]
    return f" ({'; '.join(parts)})"


def _int_keyed_texts(raw: Any) -> dict[int, str]:
    """``{"3": "…"}`` đã lưu → ``{3: "…"}``; bỏ khoá không phải số và giá trị rỗng."""
    if not isinstance(raw, dict):
        return {}
    return {
        int(key): str(text)
        for key, text in raw.items()
        if str(key).isdigit() and str(text).strip()
    }


class AnalysisCheckpointStore:
    """Đọc/ghi tiến độ phân tích ra file JSON (ghi nguyên tử).

    Args:
        directory: Thư mục chứa file tiến độ (tự tạo khi ghi).
        max_age_s: Tuổi tối đa của tiến độ còn được dùng lại.
    """

    def __init__(self, directory: Path, max_age_s: float = CHECKPOINT_MAX_AGE_S) -> None:
        self._directory = Path(directory)
        self._max_age_s = max_age_s

    def _path(self, key: str) -> Path:
        return self._directory / f"analysis_{key}.json"

    def load(self, key: str) -> AnalysisProgress | None:
        """Đọc tiến độ đã lưu; None nếu không có, quá hạn hoặc hỏng.

        Args:
            key: Khoá từ :func:`analysis_checkpoint_key`.

        Returns:
            Tiến độ, hoặc None.
        """
        path = self._path(key)
        try:
            if time.time() - path.stat().st_mtime > self._max_age_s:
                return None
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as exc:   # [v3.23.638] gồm UnicodeDecodeError
            logger.warning("Không đọc được tiến độ phân tích đã lưu (%s): %s", path, exc)
            return None
        if not isinstance(raw, dict) or raw.get("schema") != _SCHEMA:
            return None
        accumulated_raw = raw.get("accumulated")
        accumulated = (
            SubtitleContextAnalysis(**{
                name: str(accumulated_raw.get(name, "") or "")
                for name in ("source_lang", "characters", "overview", "glossary")
            })
            if isinstance(accumulated_raw, dict)
            else None
        )
        cues: dict[int, dict[str, Any]] = {}
        for item in raw.get("cues", []) or []:
            if isinstance(item, dict) and str(item.get("id", "")).isdigit():
                cues[int(item["id"])] = item
        done = {int(i) for i in raw.get("done_segments", []) if str(i).isdigit()}
        summaries = _int_keyed_texts(raw.get("segment_summaries", {}))
        return AnalysisProgress(
            done_segments=done,
            accumulated=accumulated,
            cues=cues,
            segment_summaries=summaries,
            segment_models=_int_keyed_texts(raw.get("segment_models", {})),
        )

    def save(self, key: str, progress: AnalysisProgress) -> None:
        """Ghi tiến độ (ghi file tạm rồi đổi tên để không bao giờ để lại file dở).

        Lỗi ghi chỉ được cảnh báo — lưu tiến độ là việc phụ, không được làm hỏng
        phân tích.

        Args:
            key: Khoá tiến độ.
            progress: Tiến độ cần lưu.
        """
        payload = {
            "schema": _SCHEMA,
            "done_segments": sorted(progress.done_segments),
            "accumulated": asdict(progress.accumulated) if progress.accumulated else None,
            "cues": [progress.cues[i] for i in sorted(progress.cues)],
            "segment_summaries": {
                str(i): progress.segment_summaries[i]
                for i in sorted(progress.segment_summaries)
            },
            "segment_models": {
                str(i): progress.segment_models[i]
                for i in sorted(progress.segment_models)
            },
        }
        path = self._path(key)
        temp = path.with_suffix(".tmp")
        try:
            self._directory.mkdir(parents=True, exist_ok=True)
            temp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            os.replace(temp, path)
        except OSError as exc:
            logger.warning("Không lưu được tiến độ phân tích (%s): %s", path, exc)

    def clear(self, key: str) -> None:
        """Xoá tiến độ khi phân tích đã xong trọn vẹn."""
        try:
            self._path(key).unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Không xoá được tiến độ phân tích: %s", exc)


__all__ = [
    "CHECKPOINT_MAX_AGE_S",
    "SHARED_PROGRESS_MODEL",
    "AnalysisCheckpointStore",
    "AnalysisProgress",
    "analysis_checkpoint_key",
    "describe_segment_models",
]
