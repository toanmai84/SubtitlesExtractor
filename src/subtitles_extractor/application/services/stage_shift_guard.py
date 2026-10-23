"""[v3.23.569] Giai đoạn tinh chỉnh ĐẨY LỆCH DÒNG so với bản trước — dò và phân xử.

Chuẩn đoán thật v3.23.568: dịch thô (literal) căn đúng dòng 3395-3421, giai đoạn tinh
chỉnh phong cách (style) trả về bản mà mỗi dòng mang câu của dòng thứ HAI kế tiếp
("又大" ra "lại vừa đắt đỏ.", "好嘞" ra "Chẳng phải anh"). Bộ dò lệch khối theo tương
quan độ dài (v3.23.549) im lặng: các câu ngắn dài xấp xỉ nhau nên lệch hai dòng vẫn
tương quan 0,6-0,9. Bản cuối còn 9 dòng lệch (3395-3403) và mất hẳn một câu.

Tín hiệu mới: so bản tinh chỉnh với CHÍNH BẢN TRƯỚC NÓ (cùng tiếng Việt). Căn đúng thì
dòng i giống dòng i của bản trước nhất; bị đẩy lệch thì cả một dải dòng liên tiếp giống
dòng i+k. Đo trên 60+ bộ chuẩn đoán thật: bắt được mọi dải lệch đã soi tay.

Nhưng lệch so với bản trước chưa nói ai đúng — nhiều khi chính bản dịch thô lệch và bản
tinh chỉnh đã SỬA lại (soi tay: 5/10 ca). Tương quan độ dài với câu gốc phân xử sai ở dải
ngắn, nên dùng TRỌNG TÀI: một bản dịch mới từ câu gốc cho riêng vùng tranh chấp; phía nào
khớp trọng tài theo từng dòng thì thắng.

[v3.23.589] Thêm dải VIẾT LẠI: cả một dải dòng không còn giống bản trước VÀ cũng không
giống dòng lân cận nào (giai đoạn tinh chỉnh bịa lời theo tiếng ồn đám đông trong video).
Bộ dò lệch ở trên không thấy vì nó cần độ giống ≥ 0,6 với một dòng kề. Xem
:func:`find_stage_rewrites`; phân xử vẫn bằng cùng một trọng tài.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher

#: Độ giống tối thiểu với dòng lân cận để coi là "mang câu của dòng khác".
SHIFT_MIN_SIMILARITY = 0.6
#: Phải giống dòng lân cận hơn giống chính dòng mình ít nhất ngần này.
SHIFT_MIN_GAIN = 0.3
#: Số dòng lệch cùng hướng tối thiểu để thành một dải (một dòng lẻ thường là đảo câu).
SHIFT_MIN_RUN = 3
#: [v3.23.573] Dải phải có ít nhất ngần này dòng bằng chứng MẠNH; các dòng kề chỉ có
#: bằng chứng yếu (xem :data:`SHIFT_WEAK_MIN_SIMILARITY`) được nối thêm cho đủ
#: :data:`SHIFT_MIN_RUN`.
SHIFT_MIN_STRICT_LINES = 2
#: [v3.23.573] Bằng chứng YẾU: giống dòng lân cận ≥ mức này và hơn giống chính dòng mình.
#: Ca thật v3.23.572 dòng 3618-3621: model cắt lại câu ("cho tới chị Xuân Hoa," giống
#: dòng sau 0,75 nhưng giống chính nó 0,5) nên chỉ hai dòng đạt bằng chứng mạnh.
SHIFT_WEAK_MIN_SIMILARITY = 0.5
#: Dò lệch tối đa ngần này dòng.
SHIFT_MAX_OFFSET = 3
#: Dòng quá ngắn ("Được.") giống bừa nhiều dòng — không dùng làm bằng chứng.
SHIFT_MIN_WORDS = 2
#: Hai dòng lệch cách nhau không quá ngần này vị trí vẫn thuộc cùng một dải.
SHIFT_MAX_GAP = 2
#: Hai DẢI cùng độ lệch cách nhau không quá ngần này vị trí là một vùng lệch đứt quãng
#: (ca thật: 3395-3401 và 3405-3418 cùng lệch +2, đứt vì dòng 3402 trùng câu tình cờ và
#: dòng 3403 quá ngắn để làm bằng chứng) — gộp lại để lùi luôn cả khe giữa.
SHIFT_MERGE_GAP = 4
#: [v3.23.573] Bộ dò lệch KHỐI theo tương quan độ dài báo lệch ở giai đoạn tinh chỉnh/bản
#: địa hoá, nhưng bản trả về căn đúng từng dòng với bản trước ở ≥ tỉ lệ này số dòng có
#: bằng chứng → không phải lệch. Đo trên 20 lần báo lệch thật ở hai giai đoạn này: 7 lần
#: báo nhầm căn đúng 95-100% số dòng, 13 lần lệch thật chỉ 0-33%.
PRIOR_ALIGNED_SHARE = 0.8
#: Cần ít nhất ngần này dòng có bằng chứng mới kết luận được là căn đúng.
PRIOR_ALIGNED_MIN_LINES = 5
#: Chênh độ giống với trọng tài để một dòng được tính là "thắng".
ARBITRATION_MARGIN = 0.15
#: Số dòng thắng tối thiểu để có phán quyết.
ARBITRATION_MIN_WINS = 2
#: Bản trước phải thắng GẤP ngần này lần mới được lùi về. Kiểm chéo trên 26 ca tranh chấp
#: thật (mỗi ca xử bằng 18-19 bản dịch độc lập của chính phim đó): tỉ lệ 2 lùi nhầm 4,3%,
#: tỉ lệ 3 chỉ 2,6% mà vẫn bắt 97% ca cần lùi.
PREVIOUS_WIN_RATIO = 3
#: Bản mới thắng gấp ngần này lần thì coi là bản mới căn đúng (giữ nguyên).
CURRENT_WIN_RATIO = 2

#: [v3.23.589] Dòng bị VIẾT LẠI: giống chính dòng đó của bản trước DƯỚI mức này. Tinh
#: chỉnh bình thường (đổi xưng hô, gọt câu) giữ độ giống 0,5-1,0; đo trên 106 giai đoạn
#: tinh chỉnh thật, mức 0,2 cho 23 dải ngoài những dải bộ dò lệch đã bắt.
REWRITE_MAX_SIMILARITY = 0.2
#: [v3.23.589] Số dòng viết lại tối thiểu để thành một dải. Dòng viết lại LẺ có cả sửa lẫn
#: hỏng (đo: 540 sửa / 377 hỏng) và trọng tài theo từng dòng sai khoảng 10% — không xử.
REWRITE_MIN_RUN = 3
#: [v3.23.589] Hai dòng viết lại cách nhau không quá ngần này vị trí vẫn thuộc một dải.
#: Rộng hơn :data:`SHIFT_MAX_GAP` một dòng: lời bịa hay chen giữa hai dòng còn nguyên
#: (ca thật v3.23.584 dòng 68, 71, 73). Đo với khe 2 → 3: số dải cần lùi bắt được 5 → 8,
#: số dải lùi nhầm không tăng (0,9 → 0,8); khe 4 không bắt thêm dải nào.
REWRITE_MAX_GAP = 3

VERDICT_PREVIOUS = "previous"
VERDICT_CURRENT = "current"
VERDICT_UNDECIDED = "undecided"

_PUNCTUATION_RE = re.compile(r"[^\w\s]")


@dataclass(frozen=True)
class StageShift:
    """Một dải dòng mà bản mới mang câu của dòng lệch ``offset`` so với bản trước.

    Attributes:
        from_index: Dòng đầu dải.
        to_index: Dòng cuối dải.
        offset: Dòng i của bản mới giống dòng i+offset của bản trước; 0 là dải VIẾT LẠI
            (không còn giống dòng nào của bản trước).
        lines: Số dòng bị cờ trong dải.
    """

    from_index: int
    to_index: int
    offset: int
    lines: int

    @property
    def is_rewrite(self) -> bool:
        """[v3.23.589] Dải viết lại chứ không phải đẩy lệch."""
        return self.offset == 0


def _words(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFC", (text or "").lower())
    return _PUNCTUATION_RE.sub(" ", normalized).split()


def text_similarity(first: str, second: str) -> float:
    """Độ giống theo TỪ của hai câu tiếng Việt, 0..1 (hàm thuần)."""
    left, right = _words(first), _words(second)
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def shared_word_count(first: str, second: str) -> int:
    """[v3.23.592] Số TỪ khớp theo thứ tự giữa hai câu (hàm thuần).

    Bổ sung cho :func:`text_similarity` ở câu ngắn: "Làm gì thế?" và "Thế ạ?" chỉ chung
    một từ mà độ giống đã là 0,4.
    """
    left, right = _words(first), _words(second)
    if not left or not right:
        return 0
    blocks = SequenceMatcher(None, left, right).get_matching_blocks()
    return sum(block.size for block in blocks)


@dataclass(frozen=True)
class _LineMatch:
    """Độ giống của một dòng bản mới với CHÍNH dòng đó và với dòng lân cận giống nhất."""

    own: float
    best: float
    offset: int

    @property
    def strict(self) -> bool:
        """Bằng chứng mạnh: giống dòng lân cận RÕ hơn hẳn giống chính dòng mình."""
        gain = self.best - self.own
        return self.best >= SHIFT_MIN_SIMILARITY and gain >= SHIFT_MIN_GAIN

    @property
    def weak(self) -> bool:
        """Bằng chứng yếu: vẫn giống dòng lân cận hơn, nhưng câu đã bị cắt lại."""
        return self.best >= SHIFT_WEAK_MIN_SIMILARITY and self.best > self.own


def _line_matches(
    previous: dict[int, str], current: dict[int, str]
) -> dict[int, _LineMatch]:
    """Dòng đủ dài để làm bằng chứng → độ giống với chính nó và với dòng lân cận."""
    ordered = sorted(previous)
    matches: dict[int, _LineMatch] = {}
    for position, index in enumerate(ordered):
        text = current.get(index, "")
        if len(_words(text)) < SHIFT_MIN_WORDS:
            continue
        own = text_similarity(text, previous[index])
        best_score, best_offset = 0.0, 0
        for offset in range(-SHIFT_MAX_OFFSET, SHIFT_MAX_OFFSET + 1):
            neighbour = position + offset
            if offset == 0 or not 0 <= neighbour < len(ordered):
                continue
            score = text_similarity(text, previous[ordered[neighbour]])
            if score > best_score:
                best_score, best_offset = score, offset
        matches[index] = _LineMatch(own, best_score, best_offset)
    return matches


def _is_weak_at(matches: dict[int, _LineMatch], index: int, offset: int) -> bool:
    match = matches.get(index)
    return match is not None and match.offset == offset and match.weak


def _extend_with_weak_lines(
    run: list[int], ordered: list[int], matches: dict[int, _LineMatch], offset: int
) -> list[int]:
    """[v3.23.573] Nối các dòng chỉ có bằng chứng YẾU cùng độ lệch vào dải.

    [v3.23.574] Tính cả dòng yếu nằm TRONG khe giữa hai dòng mạnh — ca thật phim
    "灭门惨案" dòng 706-708: 706 và 708 mạnh, 707 bị cắt lại câu ("còn kịp không thể so
    sánh"), trước đây dải chỉ được đếm hai dòng nên lọt.
    """
    position = {index: pos for pos, index in enumerate(ordered)}
    first, last = position[run[0]], position[run[-1]]
    extended = [
        ordered[cursor]
        for cursor in range(first, last + 1)
        if ordered[cursor] in run or _is_weak_at(matches, ordered[cursor], offset)
    ]
    for step, edge in ((-1, 0), (1, -1)):
        cursor = position[extended[edge]] + step
        while 0 <= cursor < len(ordered):
            if not _is_weak_at(matches, ordered[cursor], offset):
                break
            if step < 0:
                extended.insert(0, ordered[cursor])
            else:
                extended.append(ordered[cursor])
            cursor += step
    return extended


def find_stage_shifts(
    previous: dict[int, str], current: dict[int, str]
) -> list[StageShift]:
    """Các dải dòng bản mới bị đẩy lệch so với bản trước (hàm thuần).

    [v3.23.573] Dải có ít nhất :data:`SHIFT_MIN_STRICT_LINES` dòng bằng chứng mạnh được
    nối thêm các dòng kề có bằng chứng yếu cùng độ lệch — lệch kiểu CẮT LẠI CÂU (dòng i
    mang phần đầu dòng i+1) chỉ để lại hai dòng giống hẳn dòng kề.

    Args:
        previous: ``index → bản dịch`` của giai đoạn trước (đầu vào của giai đoạn này).
        current: ``index → bản dịch`` giai đoạn vừa chạy.

    Returns:
        Các dải lệch, sắp theo dòng đầu.
    """
    ordered = sorted(previous)
    position = {index: pos for pos, index in enumerate(ordered)}
    matches = _line_matches(previous, current)
    flagged = {index: match.offset for index, match in matches.items() if match.strict}
    shifts: list[StageShift] = []
    run: list[int] = []

    def _flush() -> None:
        if len(run) < SHIFT_MIN_STRICT_LINES:
            return
        offset = flagged[run[0]]
        lines = _extend_with_weak_lines(run, ordered, matches, offset)
        if len(lines) >= SHIFT_MIN_RUN:
            shifts.append(StageShift(lines[0], lines[-1], offset, len(lines)))

    for index in ordered:
        if index not in flagged:
            continue
        if (
            run
            and flagged[index] == flagged[run[-1]]
            and position[index] - position[run[-1]] <= SHIFT_MAX_GAP
        ):
            run.append(index)
            continue
        _flush()
        run = [index]
    _flush()
    return _merge_close_shifts(shifts, position)


def _is_rewritten(before: str, after: str) -> bool:
    """Cả hai câu đủ dài để làm bằng chứng mà gần như không còn từ chung."""
    if len(_words(before)) < SHIFT_MIN_WORDS or len(_words(after)) < SHIFT_MIN_WORDS:
        return False
    return text_similarity(before, after) < REWRITE_MAX_SIMILARITY


def find_stage_rewrites(
    previous: dict[int, str],
    current: dict[int, str],
    shifts: list[StageShift] | None = None,
) -> list[StageShift]:
    """[v3.23.589] Các dải dòng bản mới VIẾT LẠI khác hẳn bản trước (hàm thuần).

    Ca thật v3.23.588, phim "灭门惨案II" dòng 68-77 (cảnh dân làng xếp hàng): bản dịch thô
    đúng ("不要生氣" ra "Đừng giận."), giai đoạn phong cách đổi thành lời đám đông ("Mở
    cửa!", "Nhanh lên!"). Không dòng nào giống dòng kề của bản trước nên
    :func:`find_stage_shifts` im lặng; bản cuối sai 7 dòng. Bản thô đúng cả dải này ở
    cả 14 lượt dịch phim; bản cuối có dòng bị viết lại ở 8 lượt.

    Viết lại chưa nói ai đúng: kiểm chéo 23 dải thật thì 12 dải là giai đoạn này SỬA một
    bản trước lệch dòng, 8 dải làm hỏng bản trước đúng, 3 dải không rõ. Vì vậy hàm chỉ
    DÒ; phân xử vẫn là :func:`judge_shift`.

    Args:
        previous: ``index → bản dịch`` của giai đoạn trước.
        current: ``index → bản dịch`` giai đoạn vừa chạy.
        shifts: Các dải lệch đã dò được; dải viết lại chồng lên chúng bị bỏ (đã có phân
            xử riêng).

    Returns:
        Các dải viết lại (``offset`` bằng 0), sắp theo dòng đầu.
    """
    ordered = sorted(previous)
    rewritten = [
        position
        for position, index in enumerate(ordered)
        if _is_rewritten(previous[index], current.get(index, ""))
    ]
    groups: list[list[int]] = []
    for position in rewritten:
        if groups and position - groups[-1][-1] <= REWRITE_MAX_GAP:
            groups[-1].append(position)
        else:
            groups.append([position])
    found: list[StageShift] = []
    for group in groups:
        if len(group) < REWRITE_MIN_RUN:
            continue
        first, last = ordered[group[0]], ordered[group[-1]]
        if any(s.from_index <= last and s.to_index >= first for s in shifts or []):
            continue
        found.append(StageShift(first, last, 0, len(group)))
    return found


def aligned_with_previous(
    previous: dict[int, str], current: dict[int, str], indices: list[int]
) -> bool:
    """[v3.23.573] Bản mới có căn ĐÚNG từng dòng với bản trước ở các dòng đã cho (thuần).

    Một dòng căn đúng khi giống chính dòng đó của bản trước ít nhất bằng giống bất kỳ dòng
    lân cận nào. Dòng quá ngắn (xem :data:`SHIFT_MIN_WORDS`) không làm bằng chứng.

    Args:
        previous: ``index → bản dịch`` của giai đoạn trước.
        current: ``index → bản dịch`` vừa nhận.
        indices: Các dòng cần xét.

    Returns:
        True khi đủ :data:`PRIOR_ALIGNED_MIN_LINES` dòng bằng chứng và ít nhất
        :data:`PRIOR_ALIGNED_SHARE` trong số đó căn đúng.
    """
    matches = _line_matches(previous, current)
    evidence = [matches[index] for index in indices if index in matches]
    if len(evidence) < PRIOR_ALIGNED_MIN_LINES:
        return False
    aligned = sum(1 for match in evidence if match.own >= match.best)
    return aligned / len(evidence) >= PRIOR_ALIGNED_SHARE


def _merge_close_shifts(
    shifts: list[StageShift], position: dict[int, int]
) -> list[StageShift]:
    merged: list[StageShift] = []
    for shift in shifts:
        last = merged[-1] if merged else None
        if (
            last is not None
            and last.offset == shift.offset
            and position[shift.from_index] - position[last.to_index] <= SHIFT_MERGE_GAP
        ):
            merged[-1] = StageShift(
                last.from_index, shift.to_index, last.offset, last.lines + shift.lines
            )
            continue
        merged.append(shift)
    return merged


def judge_shift(
    indices: list[int],
    previous: dict[int, str],
    current: dict[int, str],
    reference: dict[int, str],
) -> str:
    """Phân xử bản nào căn đúng dòng, dựa trên bản dịch trọng tài từ câu gốc (hàm thuần).

    Mỗi dòng trong dải được tính cho phía giống trọng tài hơn (chênh ≥
    :data:`ARBITRATION_MARGIN`). Bản trước phải thắng ≥ :data:`ARBITRATION_MIN_WINS` dòng
    VÀ gấp :data:`PREVIOUS_WIN_RATIO` lần bản mới — lùi nhầm là làm hỏng một bản đúng.

    Args:
        indices: Các dòng của dải tranh chấp.
        previous: Bản trước.
        current: Bản mới.
        reference: Bản dịch trọng tài (dịch mới từ câu gốc).

    Returns:
        ``"previous"``, ``"current"`` hoặc ``"undecided"``.
    """
    previous_wins = current_wins = 0
    for index in indices:
        judge = reference.get(index, "")
        if not judge.strip():
            continue
        gap = text_similarity(judge, previous.get(index, "")) - text_similarity(
            judge, current.get(index, "")
        )
        if gap >= ARBITRATION_MARGIN:
            previous_wins += 1
        elif gap <= -ARBITRATION_MARGIN:
            current_wins += 1
    if previous_wins >= max(ARBITRATION_MIN_WINS, PREVIOUS_WIN_RATIO * current_wins):
        return VERDICT_PREVIOUS
    if current_wins >= max(ARBITRATION_MIN_WINS, CURRENT_WIN_RATIO * previous_wins):
        return VERDICT_CURRENT
    return VERDICT_UNDECIDED


def shift_span(ordered: list[int], shift: StageShift, padding: int) -> list[int]:
    """Các dòng của dải kèm ``padding`` dòng mỗi bên (dòng biên hay bị gộp/lặp câu)."""
    start = ordered.index(shift.from_index)
    end = ordered.index(shift.to_index)
    return ordered[max(0, start - padding) : end + padding + 1]


__all__ = [
    "ARBITRATION_MARGIN",
    "PRIOR_ALIGNED_MIN_LINES",
    "PRIOR_ALIGNED_SHARE",
    "REWRITE_MAX_GAP",
    "REWRITE_MAX_SIMILARITY",
    "REWRITE_MIN_RUN",
    "SHIFT_MAX_GAP",
    "SHIFT_MIN_RUN",
    "SHIFT_MIN_STRICT_LINES",
    "SHIFT_WEAK_MIN_SIMILARITY",
    "VERDICT_CURRENT",
    "VERDICT_PREVIOUS",
    "VERDICT_UNDECIDED",
    "StageShift",
    "aligned_with_previous",
    "find_stage_rewrites",
    "find_stage_shifts",
    "judge_shift",
    "shared_word_count",
    "shift_span",
    "text_similarity",
]
