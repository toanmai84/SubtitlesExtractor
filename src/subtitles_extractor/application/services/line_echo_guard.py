"""[v3.23.576] Dòng dịch NÓI LẶP nội dung của dòng kề — dò, lùi và dịch lại.

Chuẩn đoán thật (phim "傻子落水…", 3902 dòng, v3.23.575): khi một câu gốc bị tách qua
hai dòng mà trật tự tiếng Việt phải đảo, model dồn TRỌN câu vào một dòng rồi VẪN dịch
dòng kế bên — người xem (và giọng thuyết minh) nghe cùng một ý hai lần:

* 2820 "各位各位" → "Mọi người, mọi người bình tĩnh nào," rồi 2821 "大家稍安勿躁"
  → "Mọi người bình tĩnh nào,";
* 2728 "来参加婚礼呢?" → "loại đồ nhà quê này đến dự đám cưới thế?" rồi 2729
  "这种山炮" → "Loại đồ nhà quê này";
* lần dịch trước: 3526 "而是因为" → "mà là vì em thật sự rất thích anh." rồi 3527
  → "em thật sự rất thích anh.".

Có khi dòng còn MẤT luôn nghĩa của chính nó: 2880 "本想着和徐家攀点关系" ra y hệt câu
của dòng 2881. Cờ ``duplicate_adjacent`` chỉ bắt hai dòng GIỐNG HỆT nên bỏ sót cả loại.

Tín hiệu: bản dịch dòng i CHỨA gần trọn bản dịch của 1-4 dòng kề (trước hoặc sau) — mà
câu gốc dòng i lại KHÔNG chứa câu gốc các dòng đó (chứa thì lặp là đúng: "退钱退钱").
Đo trên 108 bộ chuẩn đoán thật (bản cuối): 295 dòng, 2-5 dòng mỗi lần dịch phim 3902
dòng; soát tay mọi ca của các lần dịch gần nhất đều là lỗi thật. Ca có câu gốc chung từ
34% trở lên gần như toàn là lời thoại lặp THẬT nên bị loại; phim nguồn chữ Latin so theo
từ (bỏ được 40 báo nhầm ở phim "Along with the Gods").

Một phần lớn do CHÍNH giai đoạn tinh chỉnh gây ra (bản trước còn tách câu đúng dòng):
"Ông nội, cháu gái" / "biết sai rồi" bị tinh chỉnh thành "Ông nội, cháu gái biết sai
rồi." / "Biết sai rồi.". Những dòng đó được lùi về bản trước — không tốn lượt gọi. Mô
phỏng trên 107 lần dịch có đủ ba giai đoạn: dòng nói lặp ở bản bản địa hoá 346 → 248
(lùi 111 dòng), dòng trùng hệt dòng kề cũng giảm 140 → 119. Phần còn lại được khâu dọn
cuối dịch lại riêng từ nguồn (xem ``TranslateSubtitlesUseCase``).

Toàn bộ là hàm thuần: không I/O.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from itertools import pairwise

from subtitles_extractor.application.services.translation_diagnostics import (
    source_content_key,
)

#: Tổng số từ tối thiểu của (các) dòng kề để làm bằng chứng — "Được." giống bừa.
ECHO_MIN_WORDS = 3
#: Tỉ lệ từ của dòng kề nằm trong dòng i để coi là dòng i nói lặp.
ECHO_MIN_CONTAINMENT = 0.8
#: Câu gốc dòng i chứa từ ngần này câu gốc dòng kề trở lên thì lặp là ĐÚNG (thoại lặp).
ECHO_MAX_SOURCE_OVERLAP = 0.34
#: Xét tối đa ngần này dòng kề liên tiếp mỗi phía (câu bị tách tối đa 4-5 dòng phụ đề).
ECHO_MAX_NEIGHBORS = 4

_PUNCTUATION_RE = re.compile(r"[^\w\s]")
_CJK_RE = re.compile("[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


@dataclass(frozen=True, slots=True)
class LineEcho:
    """Dòng ``index`` nói lặp nội dung của các dòng ``neighbors``.

    Attributes:
        index: Dòng chứa nội dung lặp.
        neighbors: Các dòng kề liên tiếp bị lặp lại (tăng dần).
        containment: Tỉ lệ từ của các dòng kề nằm trong dòng ``index``.
    """

    index: int
    neighbors: tuple[int, ...]
    containment: float

    @property
    def lines(self) -> tuple[int, ...]:
        """Cả nhóm dòng liên quan (dòng lặp + các dòng kề), tăng dần."""
        return tuple(sorted({self.index, *self.neighbors}))


def translation_words(text: str) -> list[str]:
    """Các từ của một câu dịch: chữ thường, bỏ dấu câu (hàm thuần)."""
    normalized = unicodedata.normalize("NFC", (text or "").lower())
    return _PUNCTUATION_RE.sub(" ", normalized).split()


def source_tokens(text: str) -> list[str]:
    """Chữ Hán của câu gốc CJK; từ của câu gốc chữ Latin (hàm thuần)."""
    characters = _CJK_RE.findall(text or "")
    return characters if characters else translation_words(text)


_words = translation_words
_source_tokens = source_tokens


def contained_share(container: list[str], part: list[str]) -> float:
    """Tỉ lệ phần tử của ``part`` khớp theo thứ tự vào ``container`` (0..1, hàm thuần)."""
    if not part or not container:
        return 0.0
    matcher = SequenceMatcher(None, container, part, autojunk=False)
    return sum(block.size for block in matcher.get_matching_blocks()) / len(part)


def _echo_toward(
    position: int,
    ordered: list[int],
    sources: dict[int, str],
    translations: dict[int, str],
    direction: int,
) -> LineEcho | None:
    """Nhóm dòng kề DÀI NHẤT về một phía mà dòng ở ``position`` nói lặp lại."""
    index = ordered[position]
    own_words = _words(translations.get(index, ""))
    own_vocabulary = set(own_words)
    own_source = _source_tokens(sources.get(index, ""))
    found: LineEcho | None = None
    for count in range(1, ECHO_MAX_NEIGHBORS + 1):
        positions = [position + direction * step for step in range(1, count + 1)]
        if not all(0 <= pos < len(ordered) for pos in positions):
            break
        neighbors = tuple(sorted(ordered[pos] for pos in positions))
        neighbor_words = [
            word for n in neighbors for word in _words(translations.get(n, ""))
        ]
        if len(neighbor_words) < ECHO_MIN_WORDS:
            continue
        # Cận trên rẻ trước khi so khớp thứ tự: từ không có mặt thì không thể khớp.
        present = sum(1 for word in neighbor_words if word in own_vocabulary)
        share = (
            contained_share(own_words, neighbor_words)
            if present >= ECHO_MIN_CONTAINMENT * len(neighbor_words)
            else 0.0
        )
        if share < ECHO_MIN_CONTAINMENT:
            if found is not None:
                break  # nhóm dài hơn không còn nằm trọn trong dòng này
            continue
        neighbor_source = [
            token for n in neighbors for token in _source_tokens(sources.get(n, ""))
        ]
        if contained_share(own_source, neighbor_source) >= ECHO_MAX_SOURCE_OVERLAP:
            break
        found = LineEcho(index, neighbors, round(share, 2))
    return found


def _echo_at(
    position: int,
    ordered: list[int],
    sources: dict[int, str],
    translations: dict[int, str],
) -> LineEcho | None:
    """Dòng ở vị trí ``position`` có nói lặp dòng kề không (xét phía sau trước)."""
    if not _words(translations.get(ordered[position], "")):
        return None
    return _echo_toward(position, ordered, sources, translations, 1) or _echo_toward(
        position, ordered, sources, translations, -1
    )


def _echoes_near(
    position: int,
    ordered: list[int],
    sources: dict[int, str],
    translations: dict[int, str],
) -> int:
    """Số dòng nói lặp trong vùng ±:data:`ECHO_MAX_NEIGHBORS` quanh ``position``."""
    low = max(0, position - ECHO_MAX_NEIGHBORS)
    high = min(len(ordered), position + ECHO_MAX_NEIGHBORS + 1)
    return sum(
        1
        for near in range(low, high)
        if _echo_at(near, ordered, sources, translations) is not None
    )


def find_line_echoes(
    sources: dict[int, str], translations: dict[int, str]
) -> list[LineEcho]:
    """Mọi dòng dịch nói lặp nội dung dòng kề mà câu gốc không lặp (hàm thuần).

    Args:
        sources: ``số dòng → câu gốc``.
        translations: ``số dòng → bản dịch``.

    Returns:
        Các dòng lặp, tăng dần theo số dòng.
    """
    ordered = sorted(translations)
    echoes = (
        _echo_at(position, ordered, sources, translations)
        for position in range(len(ordered))
    )
    return [echo for echo in echoes if echo is not None]


def _safe_to_restore(
    before: str,
    now: str,
    position: int,
    ordered: list[int],
    translations: dict[int, str],
) -> bool:
    """Bản trước không được mang lại lỗi khác: chữ Hán chưa dịch, hay trùng dòng kề.

    Ca thật (bản cũ, trước khi có lưới lệch dòng): bản trước lệch một dòng nên câu cũ của
    dòng đó GIỐNG HỆT dòng kề hiện tại — lùi về là đổi lỗi lặp lấy lỗi trùng dòng.
    """
    if _CJK_RE.search(before) and not _CJK_RE.search(now):
        return False
    words = _words(before)
    for step in (-2, -1, 1, 2):
        near = position + step
        if not 0 <= near < len(ordered):
            continue
        if _words(translations.get(ordered[near], "")) == words:
            return False
    return True


def find_echo_reverts(
    sources: dict[int, str], previous: dict[int, str], current: dict[int, str]
) -> dict[int, str]:
    """Dòng mà CHÍNH giai đoạn này làm nói lặp → bản trước của dòng đó (hàm thuần).

    Chỉ lùi khi: bản trước của dòng không lặp; giai đoạn này đã SỬA chính dòng đó (lặp vì
    dòng kề đổi thì lùi dòng này không giải quyết gì); bản trước không mang lại chữ Hán
    chưa dịch hay câu trùng hệt dòng kề; và đặt cạnh các dòng kề MỚI, vùng lân cận bớt
    dòng nói lặp.

    Args:
        sources: ``số dòng → câu gốc``.
        previous: Bản dịch đầu vào của giai đoạn.
        current: Bản dịch đầu ra của giai đoạn.

    Returns:
        ``số dòng → bản trước`` cho các dòng cần lùi.
    """
    echoed_before = {echo.index for echo in find_line_echoes(sources, previous)}
    working = dict(current)
    ordered = sorted(working)
    position_of = {index: position for position, index in enumerate(ordered)}
    reverts: dict[int, str] = {}
    for echo in find_line_echoes(sources, current):
        before = previous.get(echo.index, "")
        if echo.index in echoed_before or not before.strip():
            continue
        if before == current.get(echo.index):
            continue
        position = position_of[echo.index]
        now = current.get(echo.index, "")
        if not _safe_to_restore(before, now, position, ordered, working):
            continue
        trial = {**working, echo.index: before}
        # Lùi dòng này có thể làm DÒNG KỀ (đã được giai đoạn này viết lại) thành nói lặp —
        # chỉ nhận khi cả vùng lân cận bớt lặp.
        if _echoes_near(position, ordered, sources, trial) >= _echoes_near(
            position, ordered, sources, working
        ):
            continue
        working[echo.index] = before
        reverts[echo.index] = before
    return reverts


def _identical_pair(
    first: int, second: int, sources: dict[int, str], translations: dict[int, str]
) -> bool:
    """[v3.23.510] Hai dòng liền kề dịch GIỐNG HỆT nhau dù câu gốc khác nội dung."""
    text = (translations.get(first) or "").strip()
    if not text or text != (translations.get(second) or "").strip():
        return False
    # [v3.23.557] Câu gốc chỉ khác bảng mã dấu câu thì bản dịch giống hệt là ĐÚNG.
    # [v3.23.577] Cũng ĐÚNG khi chỉ khác trợ từ: "怎么是你" / "怎么是你啊" cùng ra "Sao
    # lại là chú em thế?" — ca thật v3.23.576 bị dịch lại oan một lượt gọi.
    return source_content_key(sources.get(first, "")) != source_content_key(
        sources.get(second, "")
    )


def _merge_overlapping(groups: list[set[int]]) -> list[tuple[int, ...]]:
    """Gộp các nhóm dòng có chung dòng, trả nhóm đã sắp tăng dần."""
    merged: list[set[int]] = []
    for group in sorted(groups, key=min):
        if merged and merged[-1] & group:
            merged[-1] |= group
        else:
            merged.append(set(group))
    return [tuple(sorted(group)) for group in merged]


def repeated_line_groups(
    sources: dict[int, str], translations: dict[int, str]
) -> list[tuple[int, ...]]:
    """Các nhóm dòng liền kề nói trùng nhau: giống hệt, hoặc một dòng nói lặp dòng kề.

    Args:
        sources: ``số dòng → câu gốc``.
        translations: ``số dòng → bản dịch``.

    Returns:
        Các nhóm dòng (tăng dần, không chồng nhau), mỗi nhóm cần dịch lại cùng nhau.
    """
    groups = [
        {first, second}
        for first, second in pairwise(sorted(translations))
        if _identical_pair(first, second, sources, translations)
    ]
    groups.extend(set(echo.lines) for echo in find_line_echoes(sources, translations))
    return _merge_overlapping(groups)


def group_still_repeats(
    group: tuple[int, ...], sources: dict[int, str], translations: dict[int, str]
) -> bool:
    """Nhóm dòng (sau khi dịch lại) còn dòng nào trùng hay nói lặp không (hàm thuần)."""
    ordered = sorted(translations)
    positions = [ordered.index(index) for index in group if index in translations]
    if not positions:
        return False
    low = max(0, min(positions) - ECHO_MAX_NEIGHBORS)
    high = min(len(ordered), max(positions) + ECHO_MAX_NEIGHBORS + 1)
    window = {index: translations[index] for index in ordered[low:high]}
    members = set(group)
    return any(members & set(found) for found in repeated_line_groups(sources, window))


__all__ = [
    "ECHO_MAX_NEIGHBORS",
    "ECHO_MAX_SOURCE_OVERLAP",
    "ECHO_MIN_CONTAINMENT",
    "ECHO_MIN_WORDS",
    "LineEcho",
    "contained_share",
    "find_echo_reverts",
    "find_line_echoes",
    "group_still_repeats",
    "repeated_line_groups",
    "source_tokens",
    "translation_words",
]
