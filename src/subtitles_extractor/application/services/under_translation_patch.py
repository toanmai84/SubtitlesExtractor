"""[v3.23.632] Vá RIÊNG các dòng dịch THIẾU còn sót ở bản cuối (hàm thuần, không I/O).

Nhật ký thật của phim 86 tập (3649 dòng, 4 giai đoạn): lưới ``under_translation_guard`` chỉ
CẢNH BÁO. Cơ chế "bổ sung" của giai đoạn sau (``needs_expansion``) không cứu được những dòng
này — cùng một dòng bị báo lại ở TỪNG giai đoạn rồi nằm nguyên trong bản cuối:

* 1734 ``师父师姐`` → ``Sư tỷ...`` (mất "sư phụ"): báo ở giai đoạn 2, 3 và 4 (16/10);
* 3148 ``就把这个臭小子`` → báo ở cả ba giai đoạn;
* 1190 ``最得意的弟子`` → ``của vi sư.``, 1396, 917 … báo lại ở mọi giai đoạn (17/10).

Cách vá: dịch LẠI riêng các dòng còn bị báo ở bản cuối, kèm vài dòng kề làm ngữ cảnh (dòng phụ
đề là MẢNH câu: dịch lẻ từng mảnh chính là nguyên nhân mất nghĩa). Mỗi bản dịch lại được nhận
hay bỏ RIÊNG — chỉ nhận khi nó thật sự tốt hơn bản cũ, nếu không giữ nguyên (không bao giờ làm
tệ thêm):

* không rỗng, không còn chữ Hán, khác bản cũ;
* dài hơn bản cũ (đếm chữ cái) — mục tiêu là khắc phục dịch THIẾU;
* thay vào rồi thì dòng KHÔNG còn bị lưới dịch thiếu báo;
* không biến thành dòng nói lặp dòng kề (``line_echo_guard``) hay trùng dòng kề.
"""

from __future__ import annotations

import re

from subtitles_extractor.application.services.line_echo_guard import group_still_repeats
from subtitles_extractor.application.services.under_translation_guard import (
    UnderTranslatedLine,
    find_under_translated,
)

#: Số dòng kề mỗi phía kèm vào lượt dịch lại để có ngữ cảnh.
PATCH_WINDOW_RADIUS = 2
#: Vá tối đa ngần này dòng mỗi lượt (nặng nhất trước) — chặn chi phí khi phim lỗi hàng loạt.
MAX_PATCH_LINES = 40

_CJK_RE = re.compile("[㐀-䶿一-鿿豈-﫿぀-ヿ가-힯]")
_LETTER_RE = re.compile(r"[^\W\d_]")
_PUNCTUATION_RE = re.compile(r"[^\w\s]")


def _letters(text: str) -> int:
    return len(_LETTER_RE.findall(text))


def _normalized(text: str) -> str:
    return " ".join(_PUNCTUATION_RE.sub(" ", text.lower()).split())


def patch_targets(
    pairs: list[tuple[int, str, str]], limit: int = MAX_PATCH_LINES,
) -> list[UnderTranslatedLine]:
    """Các dòng cần vá: dòng còn bị lưới dịch thiếu báo, nặng nhất trước, tối đa ``limit``."""
    return find_under_translated(pairs)[:limit]


def patch_window(
    targets: list[UnderTranslatedLine], ordered_indices: list[int],
    radius: int = PATCH_WINDOW_RADIUS,
) -> list[int]:
    """Số dòng gửi đi dịch lại: các dòng cần vá cộng ``radius`` dòng kề mỗi phía (tăng dần).

    Args:
        targets: Dòng cần vá.
        ordered_indices: Số dòng của cả phim, tăng dần (để biết ai là dòng kề).
        radius: Số dòng kề mỗi phía.
    """
    position = {index: i for i, index in enumerate(ordered_indices)}
    wanted: set[int] = set()
    for target in targets:
        at = position.get(target.index)
        if at is None:
            continue
        wanted.update(ordered_indices[max(0, at - radius): at + radius + 1])
    return sorted(wanted)


def accept_patches(
    pairs: list[tuple[int, str, str]],
    targets: list[UnderTranslatedLine],
    retried: dict[int, str],
) -> dict[int, str]:
    """Chọn các bản dịch lại đáng nhận: ``{số dòng: bản dịch mới}``.

    Duyệt dòng nặng nhất trước; mỗi dòng được xét trên bản đã gồm các bản vá nhận trước đó, nên
    hai dòng kề nhau không thể cùng nhận những bản trùng nhau.

    Args:
        pairs: ``(số dòng, câu gốc, bản dịch hiện tại)`` của cả phim.
        targets: Dòng cần vá (kết quả :func:`patch_targets`).
        retried: ``số dòng → bản dịch lại`` (có thể thiếu dòng).
    """
    sources = {index: source for index, source, _ in pairs}
    texts = {index: text for index, _, text in pairs}
    accepted: dict[int, str] = {}
    for target in targets:
        candidate = (retried.get(target.index) or "").strip()
        old = texts.get(target.index, "")
        if not candidate or _CJK_RE.search(candidate):
            continue
        if _normalized(candidate) == _normalized(old) or _letters(candidate) <= _letters(old):
            continue
        trial = {**texts, target.index: candidate}
        if any(
            index != target.index and _normalized(trial[index]) == _normalized(candidate)
            for index in _neighbours(trial, target.index)
        ):
            continue
        if group_still_repeats((target.index,), sources, trial):
            continue
        trial_pairs = [(i, sources[i], trial[i]) for i in sorted(trial)]
        if any(s.index == target.index for s in find_under_translated(trial_pairs)):
            continue
        texts = trial
        accepted[target.index] = candidate
    return accepted


def _neighbours(texts: dict[int, str], index: int) -> list[int]:
    ordered = sorted(texts)
    at = ordered.index(index)
    return ordered[max(0, at - 1): at] + ordered[at + 1: at + 2]


__all__ = [
    "MAX_PATCH_LINES", "PATCH_WINDOW_RADIUS", "accept_patches", "patch_targets", "patch_window",
]
