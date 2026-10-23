"""[v3.23.441] Ngắt dòng phụ đề dài thành 2 dòng CÂN ĐỐI tại điểm ngắt tự nhiên.

Phụ đề CJK→Việt thường ra 1 dòng dài (đo 19 tập: 9% dòng >42 ký tự, 0% có ngắt dòng) → player
tự wrap tùy tiện (ngắt giữa cụm từ, 2 dòng lệch nhau). Chuẩn ngành: tối đa ~42 ký tự/dòng, tách
2 dòng cân đối, ưu tiên ngắt SAU dấu câu / TRƯỚC liên từ, KHÔNG ngắt giữa từ.

Thuần tính toán, không đụng nghĩa (chỉ chèn ``\\n``). An toàn cho mọi định dạng xuất.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

#: Số ký tự tối đa mỗi dòng phụ đề (chuẩn ngành ~42 cho màn hình 16:9).
DEFAULT_MAX_CHARS_PER_LINE: int = 42

#: Liên từ/từ nối tiếng Việt — NÊN ngắt dòng TRƯỚC chúng (đọc mạch lạc hơn).
_VI_CONJUNCTIONS: frozenset[str] = frozenset({
    "và", "nhưng", "mà", "thì", "để", "rằng", "nên", "vì", "nếu", "khi", "hay",
    "hoặc", "còn", "với", "cho", "của", "là", "sẽ", "đã", "cũng", "vẫn",
})

#: [v3.23.621] Lượng từ/chỉ định đi liền danh từ sau — ngắt NGAY SAU chúng làm treo chữ
#: ("Ta đã nhẫn nhịn các / ngươi…").
_VI_BOUND_TO_NEXT: frozenset[str] = frozenset({
    "các", "những", "mọi", "mỗi", "từng", "mấy", "vài", "một", "hai", "ba", "bốn", "năm",
    "sáu", "bảy", "tám", "chín", "mười", "trăm", "nghìn", "ngàn", "vạn", "triệu", "cái",
    "chiếc", "con", "vị", "tên", "đám", "lũ", "bọn", "rất", "quá", "hơi", "khá", "thật",
    "không", "chưa", "chẳng", "đừng", "chớ",
})

#: [v3.23.621] Trợ từ cuối câu — không được đứng ĐẦU dòng 2 ("… nhẫn nhịn hai mươi năm / rồi.").
_VI_SENTENCE_FINAL: frozenset[str] = frozenset({
    "rồi", "nhé", "nha", "à", "ạ", "ư", "chứ", "nhỉ", "đâu", "vậy", "thôi", "sao", "hả",
    "đấy", "đó", "kìa", "nữa", "luôn", "đi",
})

#: Phạt tương đương dời điểm ngắt xa điểm giữa bấy nhiêu ký tự.
_SPLIT_COMPOUND_PENALTY = 14.0
_BOUND_WORD_PENALTY = 10.0
_FINAL_PARTICLE_PENALTY = 10.0

_COMPOUNDS_FILE = Path(__file__).resolve().parents[2] / "data" / "vi_compound_words.txt"


@lru_cache(maxsize=1)
def _vi_compounds() -> frozenset[tuple[str, str]]:
    """[v3.23.621] Cặp âm tiết của từ ghép thông dụng (không ngắt dòng giữa chúng).

    Ngắt dòng cũ chỉ tìm khoảng trắng gần giữa câu — tiếng Việt viết cách từng âm tiết nên
    hay cắt đôi từ ghép: "tranh vị / trí", "sức / mạnh", "chuyện tình / cảm" (đo trên phim
    mẫu: 5/300 dòng dịch). Thiếu tệp thì bỏ qua luật này, không lỗi.
    """
    pairs: set[tuple[str, str]] = set()
    try:
        lines = _COMPOUNDS_FILE.read_text(encoding="utf-8").splitlines()
    except OSError:
        return frozenset()
    for raw in lines:
        word = raw.strip().lower()
        if not word or word.startswith("#"):
            continue
        syllables = word.split()
        for left, right in zip(syllables, syllables[1:]):
            pairs.add((left, right))
    return frozenset(pairs)


def _word_before(text: str, space_pos: int) -> str:
    return text[:space_pos].rsplit(" ", 1)[-1].strip(",.!?;:…\"'“”()").lower()


#: Dấu câu mà ngắt dòng NGAY SAU sẽ tự nhiên nhất.
_BREAK_AFTER_PUNCT: frozenset[str] = frozenset({",", ".", "!", "?", ";", ":", "…"})


def _candidate_score(
    text: str, space_pos: int, ideal: float, max_chars: int
) -> float | None:
    """Điểm cho một vị trí khoảng trắng làm điểm ngắt (càng CAO càng tốt); None nếu không hợp lệ.

    Ưu tiên: gần giữa câu (cân đối 2 dòng) + ngắt sau dấu câu + ngắt trước liên từ; loại vị trí
    làm một trong hai dòng vượt ``max_chars``.
    """
    left_len = space_pos  # ký tự trước khoảng trắng
    right_len = len(text) - space_pos - 1
    if left_len <= 0 or right_len <= 0:
        return None
    if left_len > max_chars or right_len > max_chars:
        return None  # tách xong vẫn tràn → vị trí này vô dụng

    # Nền: gần giữa càng tốt (khoảng cách tới điểm lý tưởng, đảo dấu).
    score = -abs(space_pos - ideal)

    prev_char = text[space_pos - 1]
    if prev_char in _BREAK_AFTER_PUNCT:
        score += 25.0  # ngắt ngay sau dấu câu: rất tự nhiên

    next_word = text[space_pos + 1:].split(" ", 1)[0].strip(",.!?;:…").lower()
    if next_word in _VI_CONJUNCTIONS:
        score += 12.0  # ngắt trước liên từ: mạch lạc

    if prev_char not in _BREAK_AFTER_PUNCT:
        prev_word = _word_before(text, space_pos)
        if (prev_word, next_word) in _vi_compounds():
            score -= _SPLIT_COMPOUND_PENALTY  # cắt đôi từ ghép
        if prev_word in _VI_BOUND_TO_NEXT:
            score -= _BOUND_WORD_PENALTY  # lượng từ/phủ định treo cuối dòng
        if next_word in _VI_SENTENCE_FINAL:
            score -= _FINAL_PARTICLE_PENALTY  # trợ từ cuối câu lên đầu dòng

    return score


def wrap_subtitle_line(text: str, max_chars: int = DEFAULT_MAX_CHARS_PER_LINE) -> str:
    """Ngắt một dòng phụ đề dài thành 2 dòng cân đối; giữ nguyên nếu đã ngắn/đã có ngắt dòng.

    Args:
        text: Nội dung dòng (một dòng).
        max_chars: Ngưỡng ký tự tối đa mỗi dòng.

    Returns:
        Text đã chèn ``\\n`` tại điểm ngắt tốt nhất, hoặc nguyên bản nếu không cần/không thể ngắt
        (không có khoảng trắng hợp lệ — vd một từ dài).
    """
    stripped = text.strip()
    if "\n" in text or len(stripped) <= max_chars:
        return text

    ideal = len(stripped) / 2.0
    best_pos: int | None = None
    best_score = float("-inf")
    for pos, char in enumerate(stripped):
        if char != " ":
            continue
        score = _candidate_score(stripped, pos, ideal, max_chars)
        if score is not None and score > best_score:
            best_score = score
            best_pos = pos

    if best_pos is None:
        # Không có điểm ngắt giữ được cả hai dòng ≤ max_chars → nới: chọn khoảng trắng gần giữa
        # nhất bất kể độ dài (vẫn hơn để player ngắt giữa từ).
        for pos, char in enumerate(stripped):
            if char == " ":
                dist = abs(pos - ideal)
                if best_pos is None or dist < abs(best_pos - ideal):
                    best_pos = pos
        if best_pos is None:
            return text  # một từ duy nhất, không thể ngắt

    return stripped[:best_pos] + "\n" + stripped[best_pos + 1:]


__all__ = ["wrap_subtitle_line", "DEFAULT_MAX_CHARS_PER_LINE"]
