"""[v3.23.554] Ép ĐẠI TỪ TỰ XƯNG về đúng quy tắc đã ghi trong bảng nhân vật.

Bối cảnh đo được (chuẩn đoán thật v3.23.553, phim 3902 dòng, roster 119 quy tắc):

* **51 dòng** dùng đại từ tự xưng KHÁC quy tắc roster đã ghi rõ (``宋钱`` với Mẹ Tống
  phải xưng 'con' nhưng ra "để bảo vệ bạn gái cũ của **tôi**"; ``宋钱`` với Từ Tuấn Hào
  phải xưng 'tao' nhưng ra "**tôi** nhất định sẽ tặng…").
* **50/51 sai ngay từ giai đoạn DỊCH THÔ** — model không tra bảng, chứ không phải giai
  đoạn tinh chỉnh phá. Từ v3.23.551 quy tắc đã được gắn thẳng vào lô dịch mà vẫn sai →
  dạy bằng CHỈ THỊ đã tới hạn, phải ÉP BẰNG MÃ ("schema hơn chỉ thị").

Sửa máy móc đại từ là việc NGUY HIỂM trong tiếng Việt vì đại từ trùng danh từ thân tộc
("sinh **con**", "**anh** tôi", "nhà họ Liễu **ta**"), nên module này cố ý sửa ÍT mà
CHẮC. Năm chốt an toàn, tất cả rút từ dữ liệu thật:

1. Quy tắc phải **không điều kiện**: "trước xưng 'tôi'… sau xưng 'em'" là quy tắc theo
   giai đoạn — bỏ qua.
2. Quy tắc phải nêu **đúng một** đại từ tự xưng.
3. Câu gốc phải có ngôi **nhất số ít** (我/俺/咱) và **không** có ngôi hai (你/您).
4. Bản dịch phải có **đúng một** đại từ, nằm trong :data:`REPLACEABLE_PRONOUNS` — những
   từ không bao giờ là danh từ thường — và không đứng ngay sau danh từ thân tộc hay tên
   riêng ("mẹ tôi", "họ Liễu ta") vì thay vào đó sẽ đổi nghĩa.
5. Trong cùng **một cảnh**, quy tắc phải đang được áp ít nhất
   :data:`MIN_RULE_SHARE_IN_SCENE`. Cảnh mà gần như mọi dòng đều lệch là phim ĐỔI GIỌNG
   có chủ ý (An Tuyết Mạn từ "tao/mày" sang "tôi" khi quỳ xin lỗi) — máy không được đè.

Đo trên chính lần chạy đó: 9 dòng được sửa, soi tay đúng cả 9; 35 dòng còn lại được báo
ra chuẩn đoán để người dùng tự quyết.

[v3.23.555] Thêm CHẾ ĐỘ 2 — chuẩn hoá theo ĐA SỐ TRONG CẢNH. Ca trôi lớn nhất của lần
chạy v3.23.554 (``张依贝→宋钱``: 'em' 12 dòng / 'tôi' 4 dòng trong CÙNG một cảnh) không
sửa được bằng chế độ 1 vì roster cho hai phương án ('anh/tôi'). Nhưng roster cho phép
hai cách KHÔNG có nghĩa là được trộn trong một cảnh — nên lấy đa số của chính cảnh đó
làm chuẩn, với hàng rào: đa số phải nằm trong các phương án roster cho phép, và sau mỗi
dòng thiểu số phải còn dòng theo đa số (thiểu số nằm trọn ở cuối cảnh là ĐỔI GIỌNG).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from subtitles_extractor.application.services.roster_format import (
    address_rule_map,
    lookup_address_rule,
)
from subtitles_extractor.application.services.translation_diagnostics import (
    ADDRESS_WORD_RE,
    PLURAL_SOURCE_RE,
    SELF_SOURCE_RE,
    VIETNAMESE_PRONOUNS,
    YOU_SOURCE_RE,
    cue_lines_by_pair,
    split_scenes,
)

__all__ = [
    "KNOWN_PRONOUNS",
    "MIN_RULE_SHARE_IN_SCENE",
    "OUTDATED_RULE_MIN_LINES",
    "OUTDATED_RULE_MIN_SHARE",
    "REPLACEABLE_PRONOUNS",
    "SCENE_MAJORITY_MIN_LINES",
    "SCENE_MAJORITY_SHARE",
    "AddressRuleFix",
    "AddressRuleViolation",
    "OutdatedAddressRule",
    "allowed_self_forms",
    "apply_address_fixes",
    "find_address_rule_fixes",
    "find_address_rule_violations",
    "find_outdated_rules",
    "follows_proper_noun",
    "possessive_swap_is_safe",
    "rule_returns_after",
    "unstable_cue_lines",
]

logger = logging.getLogger(__name__)

#: Đại từ tự xưng KHÔNG BAO GIỜ đóng vai danh từ thường → thay máy móc là an toàn. Các từ
#: thân tộc ('em', 'anh', 'chị', 'con'…) cố ý ĐỨNG NGOÀI: "sinh con", "anh tôi" sẽ hỏng.
REPLACEABLE_PRONOUNS = frozenset({"tôi", "tao", "ta", "tớ", "mình"})

#: Đại từ/danh xưng nhận dạng được trong bản dịch (gồm danh từ thân tộc thay đại từ).
KNOWN_PRONOUNS = frozenset(VIETNAMESE_PRONOUNS) | {
    "mình", "mẹ", "má", "bố", "ba", "cha", "lão",
}

#: Số từ đứng trước đại từ biến nó thành SỐ NHIỀU ("hai ta", "cả nhà ta") — không thay.
_QUANTIFIER_HEAD = "hai|ba|cả|mấy|dăm|vài"

#: Danh từ chỉ người: đại từ đứng ngay sau là SỞ HỮU ("mẹ tôi") — thay vào sẽ đổi nghĩa.
_KINSHIP_HEAD = (
    "anh|chị|em|con|cháu|mẹ|má|bố|ba|cha|vợ|chồng|ông|bà|cô|chú|bác|dì|cậu|mợ|thím|"
    "bạn|nhà|thằng|đứa|người|gã|lão"
)

#: [v3.23.562] Đại từ tự xưng CHẮC CHẮN số ít — đổi ở vị trí sở hữu không đổi nghĩa.
_SINGULAR_SELF_PRONOUNS = frozenset({"tôi", "tao", "tớ"})

#: [v3.23.562] Danh từ đứng trước + đại từ mới tạo cụm từ KHÁC nghĩa ("anh em" = anh em
#: ruột, "mẹ con" = mẹ và con, "anh chị" = các anh chị) — không đổi ở vị trí sở hữu.
_AMBIGUOUS_PAIRS = frozenset({
    ("anh", "em"), ("chị", "em"), ("anh", "chị"), ("chị", "anh"),
    ("mẹ", "con"), ("má", "con"), ("bố", "con"), ("ba", "con"), ("cha", "con"),
    ("ông", "cháu"), ("bà", "cháu"), ("cô", "cháu"), ("chú", "cháu"), ("bác", "cháu"),
    ("dì", "cháu"), ("cậu", "cháu"), ("mợ", "cháu"), ("thím", "cháu"),
    ("vợ", "chồng"), ("chồng", "vợ"), ("ông", "bà"), ("bà", "ông"),
    ("cô", "chú"), ("chú", "cô"), ("em", "anh"), ("em", "chị"),
})

#: Từ báo hiệu quy tắc CÓ ĐIỀU KIỆN theo giai đoạn — nhiều cách xưng đều hợp lệ.
_CONDITIONAL_RULE_RE = re.compile(
    r"\b(?:sau khi|trước khi|khi|lúc|từ khi|về sau|ban đầu|trước|sau|nếu|đôi khi)\b",
    re.IGNORECASE,
)

#: Trích phần tự xưng trong quy tắc "tự xưng 'x' gọi 'y'".
_SELF_FORM_RE = re.compile(r"(?:tự\s+)?xưng\s+'(?P<self>[^']*)'", re.IGNORECASE)

#: Ký tự kết thúc câu: từ viết hoa ngay sau đó là ĐẦU CÂU, không phải tên riêng.
_SENTENCE_END_RE = re.compile("[.!?\u2026:;\"'\u201c\u201d\\-\u2013\u2014]\\s*$")

#: Tỉ lệ dòng trong cảnh theo đúng quy tắc; dưới mức này coi là phim đổi giọng có chủ ý.
MIN_RULE_SHARE_IN_SCENE = 0.25

#: [v3.23.557] Khi CẢ PHIM (không riêng một cảnh) dùng một cách xưng khác hẳn quy tắc
#: roster, lỗi nằm ở BẢNG NHÂN VẬT chứ không ở bản dịch. Ca thật v3.23.556: roster bắt
#: 沈明月 xưng 'tôi' với 柳慕云 nhưng cả 6/6 dòng đều dịch 'chị' (hợp lý vì vai chị em),
#: và roster bắt 向城 xưng 'ta' trong khi 5/5 dòng dùng 'tôi'. Máy không đè bản dịch —
#: chỉ báo để người dùng sửa roster cho lần phân tích sau. Ngưỡng 5 dòng: đo trên 8 bộ
#: chuẩn đoán thật thì bắt đúng cả hai ca trên mà không sinh báo thừa ở lần chạy nào.
OUTDATED_RULE_MIN_LINES = 5
OUTDATED_RULE_MIN_SHARE = 0.8

#: [v3.23.555] Hậu tố biến đại từ thành DANH TỪ ghép: "anh trai", "em gái", "con rể",
#: "chị dâu", "anh hai". Ca thật v3.23.554 dòng 3298 "anh trai đi dự" bị báo lệch oan.
_COMPOUND_TAIL = (
    "trai|gái|dâu|rể|hai|ba|cả|út|họ|chồng|vợ|ruột|nuôi|kết nghĩa|ơi|ấy|này|đó"
)

#: [v3.23.555] CHẾ ĐỘ 2 — chuẩn hoá theo ĐA SỐ TRONG CẢNH. Ca lớn nhất của v3.23.554
#: (``张依贝→宋钱``: 'em' 12 dòng / 'tôi' 4 dòng trong CÙNG một cảnh) không sửa được bằng
#: chế độ quy tắc vì roster cho hai phương án. Nhưng dù roster cho phép hai cách, trong
#: MỘT CẢNH chỉ được dùng MỘT cách — nên lấy chính bộ phim làm chuẩn.
SCENE_MAJORITY_SHARE = 0.65
SCENE_MAJORITY_MIN_LINES = 6


@dataclass(frozen=True, slots=True)
class AddressRuleFix:
    """Một dòng sẽ được ép đại từ tự xưng về đúng quy tắc."""

    index: int
    speaker: str
    addressee: str
    pronoun_found: str
    pronoun_expected: str
    text_before: str
    text_after: str


@dataclass(frozen=True, slots=True)
class OutdatedAddressRule:
    """[v3.23.557] Quy tắc xưng hô trong bảng nhân vật mà CẢ PHIM dùng khác."""

    speaker: str
    addressee: str
    expected: str
    observed: str
    lines_following: int
    lines_observed: int
    first_index: int


@dataclass(frozen=True, slots=True)
class AddressRuleViolation:
    """Một dòng lệch quy tắc xưng hô; ``fixable`` cho biết có tự sửa được không."""

    index: int
    speaker: str
    addressee: str
    pronoun_found: str
    pronoun_expected: str
    fixable: bool
    reason: str


def unconditional_self_form(rule: str) -> str:
    """Đại từ tự xưng DUY NHẤT của một quy tắc không điều kiện (hàm thuần).

    Args:
        rule: Nguyên văn quy tắc, vd ``"tự xưng 'tao' gọi 'mày'"``.

    Returns:
        Đại từ viết thường, hoặc chuỗi rỗng nếu quy tắc có điều kiện, nêu nhiều phương án,
        hoặc phần tự xưng không phải đại từ (danh xưng như 'Tống thần y').
    """
    if not rule or _CONDITIONAL_RULE_RE.search(rule):
        return ""
    match = _SELF_FORM_RE.search(rule)
    if not match:
        return ""
    options = [
        part.strip() for part in (match.group("self") or "").split("/") if part.strip()
    ]
    if len(options) != 1:
        return ""
    head = options[0].split()[0].lower()
    return head if head in KNOWN_PRONOUNS else ""


def allowed_self_forms(rule: str) -> set[str]:
    """[v3.23.555] MỌI đại từ tự xưng roster cho phép ở cặp này, kể cả quy tắc có
    điều kiện giai đoạn.

    Quy tắc "trước xưng 'tôi' … sau xưng 'em'" cho phép cả hai — dùng làm HÀNG RÀO
    cho chế độ đa số: đa số của một cảnh chỉ được coi là chuẩn nếu nằm trong tập
    này. Nhờ vậy tránh ca thật nguy hiểm (v3.23.548): gã côn đồ quát "câm miệng hết
    cho **tao**" suýt bị đổi thành "tôi" vì đa số dòng quanh đó dịch nhầm là "tôi".

    Args:
        rule: Nguyên văn quy tắc xưng hô của cặp nhân vật.

    Returns:
        Tập đại từ (viết thường) roster cho phép; rỗng nếu không có quy tắc.
    """
    forms: set[str] = set()
    for match in _SELF_FORM_RE.finditer(rule or ""):
        for option in (match.group("self") or "").split("/"):
            words = option.strip().split()
            if words and words[0].lower() in KNOWN_PRONOUNS:
                forms.add(words[0].lower())
    return forms


def _line_uses(text: str, form: str) -> bool:
    """Câu dịch có dùng cách xưng ``form`` không (kể cả từ thân tộc ngoài danh sách).

    [v3.23.555] ``ADDRESS_WORD_RE`` chỉ bắt đại từ chuẩn nên 'mẹ', 'bố' không được
    nhận — ca thật v3.23.554 dòng 559 "Mẹ thực sự đứng lên được rồi con ơi!" bị báo
    lệch oan dù người mẹ ĐANG xưng đúng là 'mẹ'.
    """
    return bool(re.search(rf"(?i)(?<![\w]){re.escape(form)}(?![\w])", text or ""))


def _is_compound_noun(text: str, pronoun: str) -> bool:
    """Đại từ đi kèm hậu tố tạo DANH TỪ ghép ("anh trai", "em gái") chứ không xưng hô."""
    return bool(
        re.search(rf"(?i)\b{re.escape(pronoun)}\s+(?:{_COMPOUND_TAIL})\b", text or "")
    )


def _pronouns_in(text: str) -> set[str]:
    """Tập đại từ xưng hô xuất hiện trong câu dịch (viết thường)."""
    return {word.lower() for word in ADDRESS_WORD_RE.findall(text or "")}


def is_possessive_position(text: str, pronoun: str) -> bool:
    """Đại từ đứng ngay sau danh từ chỉ người hoặc tên riêng → SỞ HỮU, không được thay.

    Cũng chặn số từ đứng trước ("hai ta" = số nhiều). Từ viết hoa ở ĐẦU CÂU không tính là
    tên riêng: "Nay tôi đã khỏi" phải sửa được, còn "nhà họ Liễu ta" thì không.
    """
    heads = f"{_KINSHIP_HEAD}|{_QUANTIFIER_HEAD}"
    if re.search(rf"(?i)\b(?:{heads})\s+{re.escape(pronoun)}\b", text):
        return True
    return follows_proper_noun(text, pronoun)


def follows_proper_noun(text: str, pronoun: str) -> bool:
    """Đại từ đứng ngay sau một từ VIẾT HOA không ở đầu câu (tên riêng: "nhà họ Liễu ta").

    [v3.23.562] Trước đây nhận "viết hoa" bằng lớp ký tự ``[A-ZĐÀ-Ỹ]`` — nhưng dải
    ``À-Ỹ`` của Unicode chứa cả chữ THƯỜNG có dấu (é, ó, ở, đ…). Hệ quả: "còn ép tôi",
    "đánh tôi", "ở tôi" đều bị coi là "sau tên riêng" và lưới xưng hô bỏ qua oan.
    """
    for match in re.finditer(rf"\b(\w+)\s+{re.escape(pronoun)}\b", text):
        word = match.group(1)
        if not word[:1].isupper():
            continue
        prefix = text[: match.start()]
        if prefix.strip() and not _SENTENCE_END_RE.search(prefix):
            return True
    return False


def possessive_swap_is_safe(text: str, found: str, expected: str) -> bool:
    """[v3.23.562] Đổi đại từ tự xưng ở vị trí SỞ HỮU có giữ nguyên nghĩa không.

    Ở dòng tự xưng (câu gốc có 我, không có 你), "mẹ tôi"/"nhà tao" chỉ có thể là của
    NGƯỜI NÓI — đổi theo cách xưng của họ vẫn đúng nghĩa: "mẹ tôi" → "mẹ em". Chuẩn đoán
    thật v3.23.562: 9/24 dòng lệch quy tắc bị bỏ qua chỉ vì chốt sở hữu. Chỉ KHÔNG an toàn
    khi:

    * đại từ gốc không chắc là số ít ("ta", "mình" — "nhà ta", "hai ta" là số nhiều);
    * đứng sau số từ hoặc tên riêng;
    * danh từ đứng trước ghép với đại từ mới thành cụm từ khác nghĩa ("anh tôi" → "anh
      em" = anh em ruột, "mẹ tôi" → "mẹ con", "chị tôi" → "chị chị").

    Args:
        text: Câu dịch.
        found: Đại từ đang dùng.
        expected: Đại từ theo quy tắc.

    Returns:
        True nếu đổi được mà không đổi nghĩa.
    """
    if found not in _SINGULAR_SELF_PRONOUNS:
        return False
    if re.search(rf"(?i)\b(?:{_QUANTIFIER_HEAD})\s+{re.escape(found)}\b", text):
        return False
    if follows_proper_noun(text, found):
        return False
    for match in re.finditer(rf"(?i)\b({_KINSHIP_HEAD})\s+{re.escape(found)}\b", text):
        head = match.group(1).lower()
        if head == expected or (head, expected) in _AMBIGUOUS_PAIRS:
            return False
    return True


def _replace_pronoun(text: str, found: str, expected: str) -> str:
    """Thay đại từ trong câu, GIỮ cách viết hoa của từ gốc (hàm thuần)."""

    def _swap(match: re.Match[str]) -> str:
        word = match.group(0)
        return expected.capitalize() if word[:1].isupper() else expected

    return re.sub(rf"(?i)(?<![\w]){re.escape(found)}(?![\w])", _swap, text)


def _rule_share_in_scene(
    scene: list[int], translations: dict[int, str], expected: str
) -> float:
    """Tỉ lệ dòng trong cảnh dùng ĐÚNG đại từ quy định (trên số dòng có đại từ)."""
    following = 0
    deviating = 0
    for index in scene:
        pronouns = _pronouns_in(translations.get(index, ""))
        if not pronouns:
            continue
        if expected in pronouns:
            following += 1
        else:
            deviating += 1
    total = following + deviating
    return following / total if total else 0.0


def rule_returns_after(
    scenes: list[list[int]],
    position: int,
    sources: dict[int, str],
    translations: dict[int, str],
    expected: str,
) -> bool:
    """[v3.23.555] Sau cảnh này, phim có QUAY LẠI dùng đúng cách xưng của roster không.

    Dùng để phân biệt hai lý do khiến CẢ CẢNH lệch quy tắc:

    * **Model dịch ẩu một cảnh** — ca thật v3.23.554 dòng 2803-2951: cảnh đối đầu, Tống
      Tiền phải xưng 'tao' với kẻ thù nhưng cả 6 dòng ra 'tôi'; các cảnh SAU vẫn 'tao'.
      → cảnh này là ngoại lệ, phải sửa.
    * **Phim đổi quan hệ** — ca thật dòng 3815: Tống Tiền chuyển sang xưng 'anh' với Hứa
      Đào Hoa và KHÔNG bao giờ quay lại 'em'. → roster đã lỗi thời, máy không được đè.

    Args:
        scenes: Các cảnh của cặp nhân vật (đã tách theo khoảng cách dòng).
        position: Vị trí cảnh đang xét trong ``scenes``.
        sources: ``index → câu gốc``.
        translations: ``index → bản dịch``.
        expected: Cách xưng roster quy định.

    Returns:
        ``True`` khi các cảnh sau dùng đúng quy tắc NHIỀU HƠN số lần lệch.
    """
    later = [index for scene in scenes[position + 1:] for index in scene]
    if not later:
        return False
    stats = _scene_self_pronouns(later, sources, translations)
    following = len(stats.get(expected, []))
    deviating = sum(len(v) for key, v in stats.items() if key != expected)
    return following > deviating


def _candidates(
    sources: dict[int, str],
    translations: dict[int, str],
    by_pair: dict[tuple[str, str], list[int]],
    characters: str,
) -> list[tuple[int, str, str, str, str]]:
    """Dòng lệch quy tắc: ``(index, người_nói, người_nghe, đại_từ_gặp, đại_từ_đúng)``."""
    rules = address_rule_map(characters)
    if not rules:
        return []
    found: list[tuple[int, str, str, str, str]] = []
    for (speaker, addressee), indices in by_pair.items():
        expected = unconditional_self_form(
            lookup_address_rule(rules, speaker, addressee)
        )
        if not expected:
            continue
        for index in indices:
            source = sources.get(index, "")
            if (
                PLURAL_SOURCE_RE.search(source)
                or not SELF_SOURCE_RE.search(source)
                or YOU_SOURCE_RE.search(source)
            ):
                continue
            text = translations.get(index, "")
            pronouns = _pronouns_in(text)
            if len(pronouns) != 1 or expected in pronouns or _line_uses(text, expected):
                continue
            pronoun = next(iter(pronouns))
            if _is_compound_noun(text, pronoun):
                continue
            found.append((index, speaker, addressee, pronoun, expected))
    return found


#: [v3.23.565] Dấu kết câu. Dòng dịch KHÔNG kết thúc bằng các dấu này (kể cả kết bằng dấu
#: phẩy hay "…") nghĩa là câu còn chảy sang dòng sau — cùng một người đang nói.
_SENTENCE_FINAL_RE = re.compile(
    "(?:(?<!\\.)\\.|[!?\u3002\uff01\uff1f])[\"'\u201d\u2019)\u300d]*\\s*$"
)


def unstable_cue_lines(translations: dict[int, str], visual_cues: str) -> set[int]:
    """[v3.23.565] Dòng mà Visual Cues đổi NGƯỜI NÓI giữa chừng một câu (hàm thuần).

    Chuẩn đoán thật v3.23.564: câu "之前打伤我哥 / 打断我妈的腿 / 还逼我高中辍学" (Tống
    Dao kể tội nhà họ Từ, ba dòng nối nhau bằng dấu phẩy) có cue ghi dòng đầu là 宋瑶
    nhưng hai dòng sau là 安雪漫. Lưới xưng hô tin cue, ép "mẹ tôi" → "mẹ em" theo quy tắc
    của An Tuyết Mạn — cùng một câu thành "anh tôi … mẹ em … ép em". Trên cả phim có 124
    câu nhiều dòng mà cue đổi người nói giữa câu (vd dòng 54 "ngươi không những hết ngốc,"
    giữa lời Hồng Liên Nữ Đế lại ghi Thiết Trụ nói).

    Một "câu" là chuỗi dòng liền nhau mà dòng trước KHÔNG kết thúc bằng dấu kết câu. Câu
    có từ hai người nói trở lên theo cue → mọi dòng của nó bị coi là cue không đáng tin.

    Args:
        translations: ``index → bản dịch`` (dấu câu của bản dịch là nguồn tách câu).
        visual_cues: JSON Visual Cues (``id``/``spk``).

    Returns:
        Tập chỉ số dòng có cue không đáng tin để ép sửa.
    """
    speakers = _speakers_by_line(visual_cues)
    if not speakers:
        return set()
    unstable: set[int] = set()
    group: list[int] = []

    def _close(members: list[int]) -> None:
        if len(members) > 1 and len({speakers[i] for i in members if i in speakers}) > 1:
            unstable.update(members)

    for index in sorted(translations):
        if group and (
            index != group[-1] + 1
            or _SENTENCE_FINAL_RE.search((translations.get(group[-1]) or "").strip())
        ):
            _close(group)
            group = []
        group.append(index)
    _close(group)
    return unstable


def _speakers_by_line(visual_cues: str) -> dict[int, str]:
    """``số dòng → người nói`` từ JSON Visual Cues (bỏ qua bản ghi hỏng)."""
    import json

    try:
        parsed = json.loads(visual_cues) if (visual_cues or "").strip() else []
    except (json.JSONDecodeError, TypeError):
        return {}
    if isinstance(parsed, dict):
        parsed = parsed.get("cues", [])
    speakers: dict[int, str] = {}
    for item in parsed if isinstance(parsed, list) else []:
        if not isinstance(item, dict):
            continue
        speaker = str(item.get("spk") or "").strip()
        try:
            line_no = int(item.get("id"))
        except (TypeError, ValueError):
            continue
        if speaker:
            speakers[line_no] = speaker
    return speakers


def _skip_reason(text: str, pronoun: str, expected: str) -> str:
    """Lý do KHÔNG tự sửa được một dòng lệch (chuỗi rỗng nghĩa là sửa được)."""
    if pronoun not in REPLACEABLE_PRONOUNS:
        return f"'{pronoun}' cũng là danh từ chỉ người — thay máy móc dễ hỏng nghĩa"
    if is_possessive_position(text, pronoun) and not possessive_swap_is_safe(
        text, pronoun, expected
    ):
        return "đại từ ở vị trí sở hữu"
    return ""


def _scene_self_pronouns(
    scene: list[int], sources: dict[int, str], translations: dict[int, str]
) -> dict[str, list[int]]:
    """Các dòng TỰ XƯNG trong một cảnh, gom theo đại từ (hàm thuần).

    Chỉ nhận dòng mà câu GỐC có ngôi nhất số ít và không có ngôi hai — để chắc chắn đại từ
    trong bản dịch là cách nhân vật TỰ XƯNG, không phải cách gọi đối phương.
    """
    by_pronoun: dict[str, list[int]] = {}
    for index in scene:
        source = sources.get(index, "")
        if (
            PLURAL_SOURCE_RE.search(source)
            or not SELF_SOURCE_RE.search(source)
            or YOU_SOURCE_RE.search(source)
        ):
            continue
        text = translations.get(index, "")
        pronouns = _pronouns_in(text)
        if len(pronouns) != 1:
            continue
        pronoun = next(iter(pronouns))
        if _is_compound_noun(text, pronoun):
            continue
        by_pronoun.setdefault(pronoun, []).append(index)
    return by_pronoun


def _majority_fixes_in_scene(
    scene: list[int],
    sources: dict[int, str],
    translations: dict[int, str],
    speaker: str,
    addressee: str,
    allowed: set[str],
) -> list[AddressRuleFix]:
    """[v3.23.555] Chuẩn hoá thiểu số về ĐA SỐ của cảnh (hàm thuần).

    Điều kiện: cảnh đủ dài, một cách xưng chiếm đa số áp đảo, đa số đó nằm trong các
    phương án roster cho phép, và sau mỗi dòng thiểu số còn ít nhất một dòng theo đa số —
    nếu thiểu số nằm trọn ở CUỐI cảnh thì đó là ĐỔI GIỌNG có chủ ý, không phải trôi.
    """
    by_pronoun = _scene_self_pronouns(scene, sources, translations)
    total = sum(len(v) for v in by_pronoun.values())
    if len(by_pronoun) < 2 or total < SCENE_MAJORITY_MIN_LINES:
        return []
    majority = max(by_pronoun, key=lambda key: len(by_pronoun[key]))
    if len(by_pronoun[majority]) / total < SCENE_MAJORITY_SHARE:
        return []
    if majority not in allowed:
        return []
    fixes: list[AddressRuleFix] = []
    for pronoun, indices in by_pronoun.items():
        if pronoun == majority or pronoun not in REPLACEABLE_PRONOUNS:
            continue
        for index in indices:
            text = translations.get(index, "")
            if is_possessive_position(text, pronoun) and not possessive_swap_is_safe(
                text, pronoun, majority
            ):
                continue
            if not any(other > index for other in by_pronoun[majority]):
                continue
            replaced = _replace_pronoun(text, pronoun, majority)
            if replaced == text:
                continue
            fixes.append(
                AddressRuleFix(
                    index=index,
                    speaker=speaker,
                    addressee=addressee,
                    pronoun_found=pronoun,
                    pronoun_expected=majority,
                    text_before=text,
                    text_after=replaced,
                )
            )
    return fixes


def find_address_rule_fixes(
    sources: dict[int, str],
    translations: dict[int, str],
    visual_cues: str,
    characters: str,
) -> list[AddressRuleFix]:
    """Các dòng CHẮC CHẮN sửa được về đúng quy tắc xưng hô của roster (hàm thuần).

    Args:
        sources: ``index → câu gốc``.
        translations: ``index → bản dịch hiện tại``.
        visual_cues: JSON Visual Cues (``id``/``spk``/``to``).
        characters: Bảng nhân vật đã chuẩn hoá.

    Returns:
        Danh sách sửa đổi, sắp theo chỉ số dòng. Rỗng khi thiếu dữ liệu hoặc mọi ca đều
        vướng chốt an toàn (xem docstring module).
    """
    by_pair = cue_lines_by_pair(visual_cues, set(translations))
    scenes_by_pair = {key: split_scenes(value) for key, value in by_pair.items()}
    rules = address_rule_map(characters)
    # [v3.23.565] Cue đổi người nói giữa câu → không đủ tin để ép sửa (xem hàm).
    unstable = unstable_cue_lines(translations, visual_cues)
    fixes: list[AddressRuleFix] = []
    # CHẾ ĐỘ 1 — ép theo quy tắc roster KHÔNG điều kiện.
    for index, speaker, addressee, found, expected in _candidates(
        sources, translations, by_pair, characters
    ):
        text = translations.get(index, "")
        if index in unstable or _skip_reason(text, found, expected):
            continue
        scenes = scenes_by_pair.get((speaker, addressee), [])
        position = next(
            (pos for pos, sc in enumerate(scenes) if index in sc), -1
        )
        scene = scenes[position] if position >= 0 else [index]
        if _rule_share_in_scene(
            scene, translations, expected
        ) < MIN_RULE_SHARE_IN_SCENE and not (
            position >= 0
            and rule_returns_after(scenes, position, sources, translations, expected)
        ):
            continue
        replaced = _replace_pronoun(text, found, expected)
        if replaced == text:
            continue
        fixes.append(
            AddressRuleFix(
                index=index,
                speaker=speaker,
                addressee=addressee,
                pronoun_found=found,
                pronoun_expected=expected,
                text_before=text,
                text_after=replaced,
            )
        )
    # CHẾ ĐỘ 2 — chuẩn hoá thiểu số về đa số của cảnh (khi roster cho nhiều phương án).
    already = {fix.index for fix in fixes}
    for (speaker, addressee), scenes in scenes_by_pair.items():
        allowed = allowed_self_forms(lookup_address_rule(rules, speaker, addressee))
        if not allowed:
            continue
        for scene in scenes:
            fixes.extend(
                fix
                for fix in _majority_fixes_in_scene(
                    scene, sources, translations, speaker, addressee, allowed
                )
                if fix.index not in already and fix.index not in unstable
            )
    return sorted(fixes, key=lambda fix: fix.index)


def find_outdated_rules(
    sources: dict[int, str],
    translations: dict[int, str],
    visual_cues: str,
    characters: str,
) -> list[OutdatedAddressRule]:
    """Các quy tắc xưng hô mà CẢ PHIM dùng khác hẳn — lỗi ở roster, không ở bản dịch.

    Điều kiện: cặp nhân vật có quy tắc KHÔNG điều kiện, đủ
    :data:`OUTDATED_RULE_MIN_LINES` lượt tự xưng trong cả phim, và MỘT cách xưng khác
    chiếm từ :data:`OUTDATED_RULE_MIN_SHARE` trở lên.

    Args:
        sources: ``index → câu gốc``.
        translations: ``index → bản dịch``.
        visual_cues: JSON Visual Cues.
        characters: Bảng nhân vật đã chuẩn hoá.

    Returns:
        Danh sách quy tắc nên sửa, kèm cách xưng phim đang dùng.
    """
    rules = address_rule_map(characters)
    if not rules:
        return []
    by_pair = cue_lines_by_pair(visual_cues, set(translations))
    outdated: list[OutdatedAddressRule] = []
    for (speaker, addressee), indices in by_pair.items():
        expected = unconditional_self_form(
            lookup_address_rule(rules, speaker, addressee)
        )
        if not expected:
            continue
        stats = _scene_self_pronouns(sorted(indices), sources, translations)
        total = sum(len(value) for value in stats.values())
        if total < OUTDATED_RULE_MIN_LINES:
            continue
        deviating = {
            pronoun: lines for pronoun, lines in stats.items() if pronoun != expected
        }
        if not deviating:
            continue
        observed = max(deviating, key=lambda key: len(deviating[key]))
        if len(deviating[observed]) / total < OUTDATED_RULE_MIN_SHARE:
            continue
        outdated.append(
            OutdatedAddressRule(
                speaker=speaker,
                addressee=addressee,
                expected=expected,
                observed=observed,
                lines_following=len(stats.get(expected, [])),
                lines_observed=len(deviating[observed]),
                first_index=min(deviating[observed]),
            )
        )
    return sorted(outdated, key=lambda item: item.first_index)


def find_address_rule_violations(
    sources: dict[int, str],
    translations: dict[int, str],
    visual_cues: str,
    characters: str,
) -> list[AddressRuleViolation]:
    """Mọi dòng lệch quy tắc xưng hô, kèm lý do không tự sửa được (hàm thuần).

    Dùng cho chuẩn đoán: phần máy không dám sửa vẫn phải hiện ra để người dùng tự quyết.
    """
    by_pair = cue_lines_by_pair(visual_cues, set(translations))
    fixable = {
        fix.index
        for fix in find_address_rule_fixes(
            sources, translations, visual_cues, characters
        )
    }
    # Cặp nào đã được báo ở mục "quy tắc roster lỗi thời" thì không liệt kê lại từng dòng:
    # lỗi nằm ở bảng nhân vật, sửa một chỗ là xong, không phải soi tay hàng chục dòng.
    outdated_pairs = {
        (rule.speaker, rule.addressee)
        for rule in find_outdated_rules(sources, translations, visual_cues, characters)
    }
    unstable = unstable_cue_lines(translations, visual_cues)
    violations: list[AddressRuleViolation] = []
    for index, speaker, addressee, found, expected in _candidates(
        sources, translations, by_pair, characters
    ):
        if (speaker, addressee) in outdated_pairs:
            continue
        reason = _skip_reason(translations.get(index, ""), found, expected)
        if not reason and index in unstable:
            reason = "cue đổi người nói giữa câu — nghi gán nhầm người nói"
        if index in fixable:
            reason = "tự sửa được"
        elif not reason:
            reason = "cả cảnh dùng cách khác — nhiều khả năng đổi giọng có chủ ý"
        violations.append(
            AddressRuleViolation(
                index=index,
                speaker=speaker,
                addressee=addressee,
                pronoun_found=found,
                pronoun_expected=expected,
                fixable=index in fixable,
                reason=reason,
            )
        )
    return sorted(violations, key=lambda item: item.index)


def apply_address_fixes(
    translations: dict[int, str], fixes: list[AddressRuleFix]
) -> dict[int, str]:
    """Áp các sửa đổi lên bản dịch, trả về BẢN SAO (hàm thuần)."""
    if not fixes:
        return translations
    patched = dict(translations)
    for fix in fixes:
        patched[fix.index] = fix.text_after
    return patched
