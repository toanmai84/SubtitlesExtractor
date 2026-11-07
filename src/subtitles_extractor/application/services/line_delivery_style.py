"""[v3.23.688] Chỉ dẫn cách đọc từng dòng lồng tiếng, lấy từ nhãn ``cue`` của phân tích ngữ cảnh (hàm thuần).

Khâu phân tích đã ghi cho nhiều câu một nhãn ngắn về cách nói ("tức giận", "mỉa mai", "nghĩ thầm", "愤怒", "自言自语"…;
1670 nhãn trên các lượt phân tích phim mẫu) nhưng trước bản này nhãn chỉ dùng để gán người nói. Gemini TTS 3.8 nhận chỉ
dẫn cách đọc ở trường riêng (``speech_metadata.style``) — không bị đọc thành tiếng — nên nhãn được chuyển thẳng sang đó.

Nhãn chỉ HÀNH ĐỘNG NÓI (đáp, gọi, hỏi, nói…) không mang cảm xúc → bỏ, để model đọc tự nhiên như cũ.
"""

from __future__ import annotations

import re

from subtitles_extractor.application.services.dubbing_cast import DubbingSource, align_to_source
from subtitles_extractor.application.services.visual_cue_serializer import deserialize_visual_cues

#: Nhãn không mang cách đọc (hành động nói / vai kể) — so khớp sau khi chuẩn hoá chữ thường.
NEUTRAL_CUES = frozenset({
    "đáp", "đáp lại", "trả lời", "hỏi", "gọi", "nói", "kể", "dẫn chuyện", "thuyết minh", "bình thường",
    "说", "问", "答", "回答", "叫", "旁白", "正常",
})
#: Nhãn dài hơn mức này là mô tả cảnh, không phải chỉ dẫn giọng đọc.
MAX_STYLE_CHARS = 40

_SPACES_RE = re.compile(r"\s+")


def delivery_style(cue: str) -> str:
    """Chỉ dẫn cách đọc từ một nhãn cue; rỗng nếu nhãn trung tính / quá dài / rỗng."""
    label = _SPACES_RE.sub(" ", (cue or "").strip())
    if not label or len(label) > MAX_STYLE_CHARS:
        return ""
    if label.lower().strip(" .,:;!?") in NEUTRAL_CUES:
        return ""
    return label


def line_delivery_styles(events: list[object], source: DubbingSource) -> tuple[str, ...]:
    """Chỉ dẫn cách đọc theo vị trí từng dòng TTS (ghép câu gốc như khâu gán người nói).

    Args:
        events: Dòng TTS (có ``start_sec``, ``end_sec``).
        source: Ngữ cảnh phân vai (Visual Cues + mốc câu gốc).

    Returns:
        Tuple cùng độ dài ``events``; rỗng nếu không dòng nào có chỉ dẫn.
    """
    cues = {cue.line_no: cue.scene for cue in deserialize_visual_cues(source.visual_cues) if cue.scene}
    if not cues or not events:
        return ()
    positions = align_to_source(events, source.source_spans)
    styles = tuple(
        delivery_style(cues.get(position + 1, "")) if position is not None else ""
        for position in positions
    )
    return styles if any(styles) else ()


__all__ = ["MAX_STYLE_CHARS", "NEUTRAL_CUES", "delivery_style", "line_delivery_styles"]
