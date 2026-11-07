"""[v3.23.679] Kiểm tra PHÁT ÂM câu lồng tiếng: AI nghe lại từng câu, so với lời cần đọc (hàm thuần).

Trước đây bộ chấm này chỉ có trong công cụ đo (``bm/vn/judge_vn.py``): VieNeu sau bản 3.23.663 còn 20/296 câu đọc
sai lời (CER > 0,15 — đọc nhầm từ, nuốt chữ, đọc thừa). Người dùng phải nghe dò từng câu mới thấy.

Ngưỡng:
* ``WRONG_CER = 0,15``: như bộ chấm đo (nhiễu của chính bộ chấm ~0,06 — câu đúng chấm ra 0,01–0,04).
* Bỏ qua câu ≤ 2 âm tiết ("Hừ", "Sư thúc!"): bộ chép lời hay nghe sai câu cực ngắn ("Hừ" → "hư"), báo nhầm nhiều hơn
  bắt đúng.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

#: Tỉ lệ sai chữ (sau chuẩn hoá) từ mức này coi là "đọc sai".
WRONG_CER = 0.15
#: Câu ít âm tiết hơn mức này không kiểm.
MIN_SYLLABLES = 3

_NON_WORD = re.compile(r"[^\w]+", re.UNICODE)
_DIGITS = re.compile(r"\d+")
_ONES = ("không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín")


def _read_below_100(n: int) -> str:
    tens, unit = divmod(n, 10)
    if tens == 0:
        return _ONES[unit]
    head = "mười" if tens == 1 else f"{_ONES[tens]} mươi"
    if unit == 0:
        return head
    tail = {1: "mốt" if tens > 1 else "một", 5: "lăm"}.get(unit, _ONES[unit])
    return f"{head} {tail}"


def read_number_vi(n: int) -> str:
    """Đọc số nguyên 0–9999 bằng chữ ("2" → "hai", "105" → "một trăm lẻ năm"); số lớn hơn giữ nguyên chữ số.

    [v3.23.679] Đo thật: AI chép "chưa được 2 năm" cho câu "Chưa được hai năm" → CER 0,21 bị báo đọc sai.
    """
    if n < 0 or n > 9999:
        return str(n)
    if n < 100:
        return _read_below_100(n)
    parts: list[str] = []
    thousands, rest = divmod(n, 1000)
    if thousands:
        parts.append(f"{_ONES[thousands]} nghìn")
    hundreds, rest = divmod(rest, 100)
    if hundreds or (thousands and rest):
        parts.append(f"{_ONES[hundreds]} trăm")
    if rest:
        parts.append(("lẻ " + _ONES[rest]) if rest < 10 and parts else _read_below_100(rest))
    return " ".join(parts)


#: [v3.23.679] Phụ âm đầu đọc giống nhau ở giọng miền Nam (giọng đọc VieNeu/Edge nhiều giọng Nam): bộ chép lời không
#: phân biệt được → so sau khi gộp. Đo thật 237 câu: 4/12 câu "đọc sai" là "Dao Dao" chép thành "giao giao".
_DIALECT_FOLD: tuple[tuple[str, str], ...] = (("gi", "d"), ("r", "d"), ("x", "s"), ("tr", "ch"))
_WORD_SPLIT = re.compile(r"(\W+)", re.UNICODE)


def _fold_dialect(text: str) -> str:
    out: list[str] = []
    for word in _WORD_SPLIT.split(text):
        for head, repl in _DIALECT_FOLD:
            if word.startswith(head) and len(word) > len(head):
                word = repl + word[len(head):]
                break
        out.append(word)
    return "".join(out)


def normalize_spoken(text: str) -> str:
    """Chữ thường, đọc số thành chữ, gộp phụ âm đầu đồng âm, bỏ dấu câu/khoảng trắng/tag người nói (giữ dấu thanh)."""
    text = re.sub(r"\[[^\]]*\]", " ", text or "")       # "[Lâm:]" → bỏ
    text = _DIGITS.sub(lambda m: f" {read_number_vi(int(m.group()))} ", text)
    text = _fold_dialect(unicodedata.normalize("NFC", text).lower())
    return _NON_WORD.sub("", text).replace("_", "")


def char_error_rate(reference: str, heard: str) -> float:
    """Khoảng cách Levenshtein theo ký tự / độ dài lời cần đọc (đã chuẩn hoá). Lời rỗng: 0 nếu nghe rỗng, 1 nếu không."""
    ref, hyp = normalize_spoken(reference), normalize_spoken(heard)
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, 1):
        cur = [i]
        for j, hc in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return prev[-1] / len(ref)


def syllable_count(text: str) -> int:
    return len([w for w in re.split(r"\s+", re.sub(r"\[[^\]]*\]", " ", text or "")) if _NON_WORD.sub("", w)])


@dataclass(frozen=True, slots=True)
class LineToCheck:
    event_index: int
    text: str
    audio_path: str


@dataclass(frozen=True, slots=True)
class PronunciationVerdict:
    event_index: int
    expected: str
    heard: str
    cer: float

    @property
    def wrong(self) -> bool:
        return self.cer > WRONG_CER


def lines_to_check(results: object, only: set[int] | None = None) -> list[LineToCheck]:
    """Các dòng đáng kiểm: đã có audio riêng, không bị bỏ, đủ âm tiết. ``only``: giới hạn theo event_index."""
    from pathlib import Path

    out: list[LineToCheck] = []
    for r in results or []:  # type: ignore[attr-defined]
        index = int(getattr(r, "event_index", -1))
        if only is not None and index not in only:
            continue
        path = str(getattr(r, "segment_audio_path", "") or "")
        text = str(getattr(r, "text", "") or "")
        if getattr(r, "was_skipped", False) or not path or not Path(path).is_file():
            continue
        if syllable_count(text) < MIN_SYLLABLES:
            continue
        out.append(LineToCheck(index, text, path))
    return out


def judge_prompt(count: int) -> str:
    return (
        "Bạn là kỹ thuật viên kiểm âm lồng tiếng phim. Có "
        f"{count} đoạn âm thanh đánh số 0..{count - 1}, mỗi đoạn là MỘT câu tiếng Việt do máy đọc.\n"
        "Với từng đoạn hãy chép lại CHÍNH XÁC mọi lời nghe được (kể cả từ bị lặp, đọc thừa, đọc sai), KHÔNG sửa theo "
        "ngữ cảnh, KHÔNG đoán lời đúng.\n"
        'Trả DUY NHẤT JSON {"items":[{"i":0,"heard":"..."}]}'
    )


def parse_heard(payload: object, count: int) -> dict[int, str]:
    """Đọc JSON trả về → ``{chỉ số đoạn: lời nghe được}`` (bỏ mục hỏng/ngoài khoảng)."""
    items = payload.get("items") if isinstance(payload, dict) else payload
    out: dict[int, str] = {}
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        try:
            index = int(str(item.get("i")))
        except (TypeError, ValueError):
            continue
        if 0 <= index < count:
            out[index] = str(item.get("heard") or "")
    return out


def summarize(verdicts: list[PronunciationVerdict]) -> tuple[int, int, float]:
    """``(số câu đã kiểm, số câu đọc sai, CER trung bình)``."""
    if not verdicts:
        return 0, 0, 0.0
    wrong = sum(1 for v in verdicts if v.wrong)
    return len(verdicts), wrong, sum(v.cer for v in verdicts) / len(verdicts)
