"""[v3.23.623] Gán sẵn đại từ 我/你 cho TỪNG dòng từ quy tắc xưng hô của cặp (hàm thuần).

Vì sao: quy tắc xưng hô ("tự xưng 'muội' gọi 'sư huynh'") chỉ được gắn vào dòng ĐẦU TIÊN của
mỗi cặp trong lô, còn lưới đại từ (``address_rule_guard``) bỏ qua mọi dòng chứa 你/您. Kết quả
đo (3 tập mẫu, chấm mù): 14/15 lỗi nặng của model nhỏ là đảo vai ở đúng những dòng đó —
"师兄我就先走了" (sư muội nói) thành "Sư huynh đi trước đây", "你走吧" thành "Huynh đi đi". Model
phải tự tìm quy tắc cách 20-50 dòng rồi ánh xạ 我/你 cho đúng người; tài liệu long-context cảnh
báo độ chính xác giảm khi nhiều "kim" rải rác. Nay mỗi dòng có 我/你 mang sẵn phép gán.
"""

from __future__ import annotations

import re

__all__ = ["address_pronoun_gloss"]

_RULE_RE = re.compile(
    r"tự xưng\s*['‘’\"“”]([^'‘’\"“”]+)['‘’\"“”]\s*(?:,|và)?\s*gọi\s*['‘’\"“”]([^'‘’\"“”]+)['‘’\"“”]"
)
#: [v3.23.636] Hai phương án nối bằng "hoặc" (``'đệ tử' hoặc 'con'``): lấy phương án ĐẦU.
_ALT_QUOTED_RE = re.compile(r"(['‘“\"])([^'’”\"]+?)\1\s*hoặc\s*(['‘“\"])[^'’”\"]+?\3")
#: Quy tắc có điều kiện giai đoạn / nhiều phương án: không chắc → không gán, để model tự chọn.
_CONDITIONAL_RE = re.compile(r"trước khi|sau khi|\bnếu\b|\blúc\b|khi\s+\S+\s+(?:xưng|gọi)|giai đoạn")
#: [v3.23.636] Tự xưng bằng DANH XƯNG (không phải đại từ): model hay "sửa" thành 'ta'.
SELF_TITLES = frozenset({"đệ tử", "đồ nhi", "đồ đệ", "thuộc hạ", "tiểu nhân", "vãn bối", "tại hạ"})
_FIRST_PERSON_RE = re.compile(r"[我俺咱]")
_SECOND_PERSON_RE = re.compile(r"[你您妳]")


def first_alternative(rule: str) -> str:
    """[v3.23.636] Thu quy tắc ``tự xưng 'đệ tử' hoặc 'con' gọi 'sư thúc'`` về phương án đầu.

    Chấm mù tập 12 (phim mẫu): bảng nhân vật cho HAI cách tự xưng ("đệ tử" hoặc "con") nên
    pron bị bỏ trống 11/18 dòng của cặp và bản dịch tự xưng lúc "con", lúc "đệ tử", lúc "ta"
    (lỗi nặng "xưng hô đổi giữa các dòng"). Phương án đầu là cách model phân tích ưu tiên;
    hai cách đều đúng nghĩa nên chọn một và giữ NHẤT QUÁN cả tập. Chỉ xử lý "hoặc" (lỗi
    lọt qua prompt phân tích, vốn cấm liệt kê); phương án ``'tôi/chị'`` ngăn bằng "/" vẫn để
    model tự chọn vì khác loại đại từ (v3.23.550).
    """
    return _ALT_QUOTED_RE.sub(r"\1\2\1", rule)


def address_pronoun_gloss(rule: str, source_text: str) -> str:
    """Phép gán ``"我=muội; 你=sư huynh"`` cho một dòng, hoặc rỗng nếu không chắc.

    Args:
        rule: Quy tắc xưng hô đã tra cho cặp (người nói → người nghe) của dòng.
        source_text: Câu gốc tiếng Trung của dòng.

    Returns:
        Chuỗi gán cho đúng các đại từ có trong dòng. Rỗng khi quy tắc có điều kiện, có
        nhiều cặp xưng/gọi, dùng ``/`` liệt kê phương án, hoặc dòng không chứa 我/你.
        Phương án nối bằng "hoặc" được thu về phương án đầu (:func:`first_alternative`).
    """
    if not rule or not source_text or _CONDITIONAL_RE.search(rule):
        return ""
    rule = first_alternative(rule)
    matches = _RULE_RE.findall(rule)
    if len(matches) != 1:
        return ""
    self_term, call_term = (m.strip() for m in matches[0])
    if not self_term or not call_term or "/" in self_term or "/" in call_term:
        return ""
    parts: list[str] = []
    # Chỉ số nhiều (我们/你们) thì cách xưng số ít sai → bỏ qua đại từ đó.
    singular_first = _FIRST_PERSON_RE.search(source_text.replace("我们", "").replace("咱们", ""))
    singular_second = _SECOND_PERSON_RE.search(
        source_text.replace("你们", "").replace("您们", ""))
    if singular_first:
        if self_term.lower() in SELF_TITLES:
            # [v3.23.636] Đo thật tập 5/9/12 (3 lượt × 2 cách): ghi rõ "viết nguyên danh xưng,
            # KHÔNG đổi thành 'ta'" nâng tỉ lệ bám pron của giai đoạn dịch thô 88% → 95%.
            parts.append(
                f'我="{self_term}" (tự xưng bằng danh xưng: viết nguyên "{self_term}", '
                f'KHÔNG đổi thành "ta")'
            )
        else:
            parts.append(f"我={self_term}")
    if singular_second:
        parts.append(f"你={call_term}")
    return "; ".join(parts)
