"""[v3.23.583] Giai đoạn tinh chỉnh làm HỎNG xưng hô mà giai đoạn trước đã dùng đúng.

Chuẩn đoán thật v3.23.582 (phim "he thong sung the", 5540 dòng): cảnh Tống Hân Sảng nói
với bạn thân Trương Mẫn (bảng nhân vật: tự xưng 'mình' gọi 'cậu'), bản dịch thô dùng
"tớ … cậu" ở mọi dòng. Giai đoạn tinh chỉnh phong cách đổi sang "chị … em" ở 6 dòng
(3010, 3015, 3042, 3044, 3045, 3058: "Nếu Tiểu Dật biết bệnh tình của **chị**",
"**chị** cầu xin **em** đấy") trong khi các dòng khác của cùng cảnh vẫn "tớ/cậu" — hai
người bạn ngang tuổi bỗng thành chị em giữa một cuộc nói chuyện.

Lưới xưng hô hiện có (:mod:`address_rule_guard`) không cứu được: 'chị'/'em' cũng là danh
từ chỉ người nên nó không dám thay, và quy tắc 'mình' thì cả cảnh không dòng nào dùng.

Ở đây có bằng chứng mà lưới kia không có: CHÍNH bản dịch trước đã dùng đúng. Đo trên
45 lượt dịch thật (3 phim): giữa bản dịch thô và bản cuối có 302 dòng (một vai) mà đại từ
đổi từ cách roster cho phép sang cách roster không cho phép, ngược lại 449 dòng được
sửa đúng — giai đoạn tinh chỉnh vừa sửa vừa phá.

Cách sửa: chỉ thay ĐÚNG đại từ giai đoạn mới đưa vào bằng đại từ của bản trước; phần
còn lại của câu (văn phong mới) giữ nguyên. Các chốt an toàn, rút từ soát tay:

1. Dòng có quy tắc xưng hô của cặp nhân vật (theo Visual Cues). Vai lấy từ câu gốc:
   chỉ 我 → tự xưng, chỉ 你 → gọi người nghe, có cả hai → mỗi bản đúng hai đại từ.
   Số nhiều (我们, 你们, 咱俩) bỏ qua.
2. Mọi đại từ của bản trước đều đúng roster; mọi đại từ mới đều sai roster.
3. Đại từ mới đứng ĐÚNG CHỖ đại từ cũ (cùng từ đứng trước hoặc sau): "你放心" bản trước
   "Chị xem," bản mới "Em đảm bảo," — 'em' giờ là người nói, thay vào sẽ sai nghĩa.
4. Đặt lại được MỌI đại từ sai, không thì thôi — "em mau đưa tớ đi" (một nửa) còn tệ hơn.
5. Không đụng đại từ đứng sau từ thân tộc ("mẹ anh", "anh em") hay tên riêng giữa câu,
   và bỏ qua dòng mà Visual Cues đổi người nói giữa câu.

Kết quả trên cùng 45 lượt: 279 dòng được khôi phục. Soát tay 74 dòng (mọi dòng của phim
này, mọi dòng hai vai, 30 dòng ngẫu nhiên): sai 1 — nhãn Visual Cues gán ngược người nói
("叫你兰姐吧").

Toàn bộ module là HÀM THUẦN: không I/O, không phụ thuộc Qt/AI.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from subtitles_extractor.application.services.address_rule_guard import (
    KNOWN_PRONOUNS,
    allowed_self_forms,
    follows_proper_noun,
    unstable_cue_lines,
)
from subtitles_extractor.application.services.roster_format import (
    address_rule_map,
    is_classifier_epithet,
    lookup_address_rule,
)
from subtitles_extractor.application.services.translation_diagnostics import (
    ADDRESS_WORD_RE,
    PLURAL_SOURCE_RE,
    SELF_SOURCE_RE,
    YOU_SOURCE_RE,
    address_words,
    cue_lines_by_pair,
    is_kin_compound_at,
    split_scenes,
)

__all__ = [
    "PEER_SELF_FORMS",
    "AddressRegression",
    "allowed_address_forms",
    "find_address_regressions",
]

ROLE_SELF = "self"
ROLE_ADDRESS = "address"
ROLE_BOTH = "both"

#: Hai cách tự xưng NGANG HÀNG giữa bạn bè — cùng đi với 'cậu'. Roster ghi 'mình' mà bản
#: dịch dùng 'tớ' (hay ngược lại) vẫn đúng vai; chỉ 'chị'/'em'… mới là đổi quan hệ.
PEER_SELF_FORMS = frozenset({"mình", "tớ"})

_ADDRESS_FORM_RE = re.compile(r"gọi\s+'(?P<address>[^']*)'", re.IGNORECASE)

#: 'mình' là đại từ tự xưng TRỪ khi là "một mình", "tự mình", "chính mình", "bản thân
#: mình", "chúng mình"… (ADDRESS_WORD_RE không bắt 'mình' vì các nghĩa đó).
_MINH_RE = re.compile(
    r"(?i)(?<![\w])(?<!một )(?<!tự )(?<!chính )(?<!thân )(?<!chúng )(?<!bọn )"
    r"(?<!tụi )(?<!cả )mình(?![\w])(?!\s+mẩy)"
)

#: Từ thân tộc: đại từ đứng ngay sau nó là SỞ HỮU hoặc tạo cụm khác nghĩa ("anh em").
_KIN_HEAD_RE = (
    r"anh|chị|em|con|cháu|mẹ|má|bố|ba|cha|vợ|chồng|ông|bà|cô|chú|bác|dì|cậu|mợ|thím"
)

@dataclass(frozen=True, slots=True)
class AddressRegression:
    """Một dòng mà giai đoạn mới đổi đại từ đúng (bản trước) sang đại từ sai.

    Attributes:
        index: Chỉ số dòng.
        role: ``"self"`` (tự xưng), ``"address"`` (gọi người nghe) hoặc ``"both"``.
        speaker: Người nói theo Visual Cues.
        addressee: Người nghe theo Visual Cues.
        swaps: Các cặp ``(đại từ giai đoạn mới đưa vào, đại từ bản trước được khôi
            phục)``.
        text_before: Bản dịch của giai đoạn mới.
        text_after: Bản dịch sau khi khôi phục đại từ.
    """

    index: int
    role: str
    speaker: str
    addressee: str
    swaps: tuple[tuple[str, str], ...]
    text_before: str
    text_after: str


def allowed_address_forms(rule: str) -> set[str]:
    """MỌI đại từ GỌI người nghe roster cho phép ở cặp này (kể cả quy tắc có điều kiện).

    Args:
        rule: Nguyên văn quy tắc, vd ``"tự xưng 'chị' gọi 'chú em'"``.

    Returns:
        Tập đại từ viết thường (từ đầu của mỗi phương án, nếu là đại từ); rỗng nếu quy
        tắc chỉ gọi bằng tên/danh xưng ('Lâm tiên sinh').

    [v3.23.603] Lời chửi mở đầu bằng loại từ ('con khốn', 'con đàn bà dâm đãng') không
    cho phép đại từ "con". Chuẩn đoán thật 灭门惨案II dòng 911 (``洪志權 → 秦可秀: xưng
    'tao' gọi 'con khốn'``): bản thô "Để tao cho con khốn này biết thế nào là lợi hại",
    bản tinh chỉnh "Để tao cho mày nếm mùi lợi hại" — lưới coi "con" của "con khốn" là
    đại từ đúng quy tắc rồi "khôi phục": "Để tao cho **con** nếm mùi lợi hại" (giọng cha
    mẹ nói với con). Cùng cách đọc với :func:`roster_format.is_classifier_epithet`.
    """
    forms: set[str] = set()
    for match in _ADDRESS_FORM_RE.finditer(rule or ""):
        for option in (match.group("address") or "").split("/"):
            words = option.strip().lower().split()
            if (
                words
                and words[0] in KNOWN_PRONOUNS
                and not is_classifier_epithet(words)
            ):
                forms.add(words[0])
    return forms


def _allowed(rule: str, role: str) -> set[str]:
    if role == ROLE_ADDRESS:
        return allowed_address_forms(rule)
    forms = allowed_self_forms(rule)
    return forms | PEER_SELF_FORMS if forms & PEER_SELF_FORMS else forms


def _role(source: str) -> str:
    """Vai của đại từ trong bản dịch theo ngôi nhân xưng của câu gốc.

    Returns:
        ``"self"``, ``"address"``, ``"both"`` (có cả 我 lẫn 你), hoặc rỗng khi câu gốc
        không có ngôi nào / có số nhiều (我们, 你们, 咱俩).
    """
    if PLURAL_SOURCE_RE.search(source):
        return ""
    has_self = bool(SELF_SOURCE_RE.search(source))
    has_you = bool(YOU_SOURCE_RE.search(source))
    if has_self and not has_you:
        return ROLE_SELF
    if has_you and not has_self:
        return ROLE_ADDRESS
    return ROLE_BOTH if has_self and has_you else ""


def _pronouns(text: str) -> set[str]:
    """Đại từ xưng hô trong câu dịch (viết thường), bỏ từ thân tộc dùng như danh từ."""
    found = set(address_words(text or "", ADDRESS_WORD_RE))
    if _MINH_RE.search(text or ""):
        found.add("mình")
    return found


def _token_matches(text: str, word: str) -> list[re.Match[str]]:
    """Các chỗ ``word`` đứng làm ĐẠI TỪ xưng hô (không tính "cậu ấy", "chị dâu"…)."""
    if word == "mình":
        return list(_MINH_RE.finditer(text))
    return [
        match
        for match in ADDRESS_WORD_RE.finditer(text)
        if match.group(0).lower() == word and not is_kin_compound_at(text, match)
    ]


def _neighbours(text: str, match: re.Match[str]) -> tuple[str, str]:
    """Từ đứng ngay trước và ngay sau một đại từ (viết thường; rỗng nếu không có)."""
    before = re.search(r"(\w+)\W*$", text[: match.start()])
    after = re.match(r"\W*(\w+)", text[match.end():])
    return (
        before.group(1).lower() if before else "",
        after.group(1).lower() if after else "",
    )


def _same_slot(previous_text: str, restored: str, text: str, replaced: str) -> bool:
    """Đại từ mới đứng ĐÚNG CHỖ của đại từ cũ: cùng từ đứng sau, hoặc cùng từ đứng trước.

    Giai đoạn tinh chỉnh có thể viết lại hẳn câu và đổi VAI của đại từ: "你放心" bản trước
    "Chị xem," (gọi người nghe), bản mới "Em đảm bảo," ('em' giờ là người nói tự xưng).
    Thay 'em' bằng 'chị' ở đó sẽ thành "Chị đảm bảo" — sai nghĩa.
    """
    old = _token_matches(previous_text, restored)
    new = _token_matches(text, replaced)
    if len(old) != 1 or len(new) != 1:
        return False
    old_before, old_after = _neighbours(previous_text, old[0])
    new_before, new_after = _neighbours(text, new[0])
    if not any((old_before, old_after, new_before, new_after)):
        # Cả hai bản chỉ có đúng tiếng gọi/tự xưng ("你" → "Anh..." / "Em...").
        return True
    return bool(old_after and old_after == new_after) or bool(
        old_before and old_before == new_before
    )


def _after_kin_word(text: str, start: int) -> bool:
    """Đại từ đứng ngay sau một từ thân tộc ("mẹ anh", "anh em") — dễ ra cụm khác nghĩa.

    "anh tôi" → "anh em" (anh em ruột), "mẹ tôi" → "mẹ con": không đụng tới.
    """
    return re.search(rf"(?i)(?<![\w])(?:{_KIN_HEAD_RE})\s+$", text[:start]) is not None


def _restore(text: str, replaced: str, restored: str) -> str | None:
    """Thay chỗ ``replaced`` làm đại từ bằng ``restored``; ``None`` nếu không an toàn."""
    matches = _token_matches(text, replaced)
    if len(matches) != 1:
        return None
    match = matches[0]
    if follows_proper_noun(text, replaced):
        return None
    if _after_kin_word(text, match.start()):
        return None
    word = match.group(0)
    new_word = restored.capitalize() if word[:1].isupper() else restored
    return text[: match.start()] + new_word + text[match.end():]


def _line_swaps(
    previous_text: str, text: str, allowed: set[str], pronoun_count: int
) -> list[tuple[str, str]]:
    """Các cặp ``(đại từ mới sai, đại từ cũ đúng)`` đứng cùng chỗ trong một dòng.

    Args:
        previous_text: Bản dịch của giai đoạn trước.
        text: Bản dịch của giai đoạn vừa chạy.
        allowed: Đại từ roster cho phép ở vai đang xét.
        pronoun_count: Số đại từ mỗi bản phải có (1: một vai; 2: cả tự xưng lẫn gọi).

    Returns:
        Danh sách cặp cần thay; rỗng nếu không đủ chắc.
    """
    before = _pronouns(previous_text)
    after = _pronouns(text)
    if len(before) != pronoun_count or len(after) != pronoun_count or before == after:
        return []
    if not before <= allowed:
        return []
    swaps: list[tuple[str, str]] = []
    for replaced in sorted(after - before):
        if replaced in allowed:
            return []
        slots = [
            restored
            for restored in sorted(before - after)
            if _same_slot(previous_text, restored, text, replaced)
        ]
        if len(slots) != 1:
            # Khôi phục một nửa ("em mau đưa tớ đi") còn tệ hơn để nguyên: câu lai hai
            # lối xưng hô. Chỉ sửa khi đặt lại được MỌI đại từ sai.
            return []
        swaps.append((replaced, slots[0]))
    return swaps


def _apply_swaps(text: str, swaps: list[tuple[str, str]]) -> str | None:
    """Thay lần lượt; ``None`` nếu một chỗ thay không an toàn (không sửa nửa vời)."""
    fixed: str | None = text
    for replaced, restored in swaps:
        fixed = _restore(fixed, replaced, restored) if fixed is not None else None
    return fixed


def find_address_regressions(
    sources: dict[int, str],
    previous: dict[int, str],
    current: dict[int, str],
    visual_cues: str,
    characters: str,
) -> list[AddressRegression]:
    """Các dòng mà giai đoạn mới làm hỏng xưng hô bản trước đã dùng đúng (hàm thuần).

    Dòng chỉ có MỘT vai (câu gốc chỉ có 我 hoặc chỉ có 你) so theo đại từ roster cho vai
    đó. Dòng có cả hai (``我求你了`` — "tớ cầu xin cậu") thì mỗi bản phải có đúng hai đại
    từ, bản trước cả hai đều đúng roster, và mỗi đại từ mới sai được đặt lại đúng chỗ
    của đại từ cũ.

    Args:
        sources: ``index → câu gốc``.
        previous: ``index → bản dịch của giai đoạn trước``.
        current: ``index → bản dịch của giai đoạn vừa chạy``.
        visual_cues: JSON Visual Cues (``id``/``spk``/``to``).
        characters: Bảng nhân vật đã chuẩn hoá (chứa quy tắc xưng hô).

    Returns:
        Danh sách dòng cần khôi phục đại từ, sắp theo chỉ số.
    """
    rules = address_rule_map(characters)
    if not rules or not previous or not current:
        return []
    unstable = unstable_cue_lines(current, visual_cues)
    regressions: list[AddressRegression] = []
    for (speaker, addressee), indices in cue_lines_by_pair(
        visual_cues, set(current)
    ).items():
        rule = lookup_address_rule(rules, speaker, addressee)
        if not rule:
            continue
        for index in indices:
            role = _role(sources.get(index, ""))
            if not role or index in unstable:
                continue
            if role == ROLE_BOTH:
                allowed = _allowed(rule, ROLE_SELF) | _allowed(rule, ROLE_ADDRESS)
                count = 2
            else:
                allowed, count = _allowed(rule, role), 1
            text = current.get(index, "")
            swaps = _line_swaps(previous.get(index, ""), text, allowed, count)
            fixed = _apply_swaps(text, swaps) if swaps else None
            if fixed is None or fixed == text:
                continue
            regressions.append(
                AddressRegression(
                    index=index,
                    role=role,
                    speaker=speaker,
                    addressee=addressee,
                    swaps=tuple(swaps),
                    text_before=text,
                    text_after=fixed,
                )
            )
    kept = _consistent_with_scene(regressions, current, visual_cues)
    return sorted(kept, key=lambda item: item.index)


def _consistent_with_scene(
    regressions: list[AddressRegression],
    current: dict[int, str],
    visual_cues: str,
) -> list[AddressRegression]:
    """[v3.23.604] Bỏ lượt khôi phục làm dòng LỆCH khỏi cả cảnh của cặp nhân vật.

    Giai đoạn mạnh hơn có thể đổi xưng hô CẢ CẢNH có chủ ý. Chuẩn đoán thật 11/10
    (灭门惨案II, bản địa hoá bằng gemini-3.8-flash): cảnh 694-703 Chu Chính Quân tặng
    vòng cho Tần Khả Tú (lúc chồng cô còn sống) được viết lại thành "tôi … cô"; lưới chỉ
    khôi phục được dòng 699 về "rất hợp với em mà" — một dòng "em" giữa các dòng "cô".

    Bỏ lượt khôi phục khi đại từ được trả lại KHÔNG có ở dòng nào khác của cặp trong cảnh
    (sau khi đã áp các lượt khôi phục khác), còn đại từ giai đoạn mới đưa vào thì CÓ.

    Đo trên dữ liệu thật:

    * 153 dòng đã khôi phục (mọi file chuẩn đoán có nhật ký diễn biến): 4-5 dòng thuộc
      loại này (699; 450 "chú" giữa bốn dòng "cậu"; 786 "tôi" giữa các dòng "tao"; 1143
      "tôi" trong cảnh "anh/em").
    * Chạy lại lưới trên 124 cặp giai đoạn đã lưu: bỏ 21/279 lượt. Soát tay: đa số giữ
      đúng đại từ cả cảnh đang dùng — đứa trẻ "Bỏ cháu ra!" không bị đổi thành "tôi",
      lão Mạc "Cả nhà mày mới chết sạch!" không thành "ông"; hai ca còn bàn cãi được.
    """
    if not regressions:
        return regressions
    resolved = dict(current)
    for item in regressions:
        resolved[item.index] = item.text_after
    scene_of: dict[int, list[int]] = {}
    for indices in cue_lines_by_pair(visual_cues, set(current)).values():
        for scene in split_scenes(indices):
            for index in scene:
                scene_of[index] = scene
    kept: list[AddressRegression] = []
    for item in regressions:
        # Xét MỌI dòng của cặp trong cảnh, không tách vai: đại từ được trả lại không xuất
        # hiện ở bất kỳ dòng nào của cảnh thì dòng này mới thành "dòng lạ". Đã thử tách
        # vai: lưới bỏ thêm 6 lượt, trong đó có lượt đúng (dòng 293 "quấy rầy em" →
        # "chị", khi "chị" chỉ có ở tiếng gọi của dòng khác).
        others: set[str] = set()
        for index in scene_of.get(item.index, [item.index]):
            if index != item.index:
                others |= _pronouns(resolved.get(index, ""))
        if any(new in others and old not in others for new, old in item.swaps):
            continue
        kept.append(item)
    return kept
