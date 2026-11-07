"""[v3.23.670] Độ phủ bảng nhân vật theo CHÍNH phụ đề (hàm thuần, không gọi AI).

Phân tích không video lập bảng nhân vật chỉ từ chữ, và hay sót người: đo tập 1-6 phim mẫu —
"hệ thống" (có thoại hàng chục dòng) bị bỏ, nhóm đệ tử chế giễu không có dòng nào, người chỉ
được nhắc ("Tần sư huynh", sư tỷ bế quan) mất hẳn. Bước tự kiểm tra cũ chỉ phát hiện thiếu
người qua NHÃN người nói — không video thì chưa có nhãn, nên không bắt được gì.

Mô-đun này tìm trong phụ đề các cách gọi chỉ một NGƯỜI cụ thể (tên đứng riêng thành lời gọi,
"họ + chức danh", chức danh chỉ một người thứ ba như 师父/师叔/长老/前辈, thực thể có thoại như
系统/统子) rồi đối chiếu với phần tên của bảng nhân vật. Cách gọi chưa khớp ai được trả về kèm
số dòng để bước tự kiểm tra hỏi model: người mới hay bí danh của người đã có.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

#: Chức danh / cách gọi chỉ MỘT người cụ thể (thường là người thứ ba), kèm cách đọc Hán-Việt để
#: khớp với phần phiên âm của bảng nhân vật. KHÔNG gồm xưng hô ngang hàng (师兄/师姐/师妹/师弟,
#: 哥/姐…) — đó thường là cách hai nhân vật đã có gọi nhau, gắn cờ sẽ chỉ gây nhiễu.
TITLE_READINGS: dict[str, tuple[str, ...]] = {
    # (Không gồm 师傅: phim hiện đại dùng gọi lịch sự tài xế/thợ — không chỉ một người.)
    "师父": ("sư phụ",), "师尊": ("sư tôn",), "师叔": ("sư thúc",),
    "师伯": ("sư bá",), "师祖": ("sư tổ",), "师娘": ("sư nương",), "长老": ("trưởng lão",),
    "宗主": ("tông chủ",), "掌门": ("chưởng môn",), "掌教": ("chưởng giáo",),
    "前辈": ("tiền bối",), "老祖": ("lão tổ",), "圣女": ("thánh nữ",), "圣子": ("thánh tử",),
    "系统": ("hệ thống",), "统子": ("thống tử", "hệ thống"), "魔修": ("ma tu",),
    # Cách gọi NHÂN VẬT CHÍNH khi tập không nhắc tên (tập 4 phim mẫu: bảng thiếu hẳn nhân vật
    # chính, chỉ còn "với 宿主" trong vai trò của hệ thống).
    "宿主": ("ký chủ", "túc chủ"), "逆徒": ("nghịch đồ",), "徒儿": ("đồ nhi",),
    "城主": ("thành chủ",), "族长": ("tộc trưởng",), "皇上": ("hoàng thượng",),
    "陛下": ("bệ hạ",), "王爷": ("vương gia",), "殿下": ("điện hạ",), "老爷": ("lão gia",),
    "夫人": ("phu nhân",), "少爷": ("thiếu gia",), "太子": ("thái tử",), "公主": ("công chúa",),
}
#: [v3.23.670] Bản PHỒN THỂ của các chức danh (phụ đề Đài Loan/Hồng Kông) — cùng cách đọc.
_TRADITIONAL = {
    "師父": "师父", "師尊": "师尊", "師叔": "师叔", "師伯": "师伯", "師祖": "师祖", "師娘": "师娘",
    "長老": "长老", "掌門": "掌门", "前輩": "前辈", "聖女": "圣女", "聖子": "圣子", "系統": "系统",
    "統子": "统子", "族長": "族长", "王爺": "王爷", "老爺": "老爷", "少爺": "少爷", "逆徒": "逆徒",
}
TITLE_READINGS.update({trad: TITLE_READINGS[simp] for trad, simp in _TRADITIONAL.items()})
#: Chức danh đứng sau một họ ("秦师兄", "南宫师叔", "楚师姐") → người cụ thể theo HỌ.
_SURNAME_TITLE_RE = re.compile(
    # (Không gồm 总/哥/姐: "人总要长大" → "人总", "哥哥", "帅哥" đều là từ thường.)
    r"([一-鿿]{1,2})(师兄|师姐|师妹|师弟|师叔|师伯|长老|仙子|公子|小姐"
    r"|師兄|師姐|師妹|師弟|師叔|師伯|長老)"
)
#: Họ kép: "南宫师叔" là họ 南宫; còn lại họ là MỘT chữ đứng ngay trước chức danh ("今日和秦师兄"
#: → 秦, không phải "和秦").
_COMPOUND_SURNAMES = frozenset({
    "南宫", "欧阳", "上官", "司马", "慕容", "诸葛", "东方", "西门", "独孤", "令狐", "公孙",
    "皇甫", "轩辕", "夏侯", "宇文", "长孙", "司徒", "百里", "端木", "尉迟", "澹台", "呼延",
})
_CJK_RE = re.compile(r"[一-鿿]")
#: Chữ kana: phụ đề tiếng Nhật — chức danh Hán (先生, 殿下…) ở đó là từ thường, không dò.
_KANA_RE = re.compile(r"[\u3040-\u30ff]")
#: Họ phổ biến: tên đứng riêng một dòng chỉ tính khi mở đầu bằng họ (hoặc là tên lặp kiểu "瑶瑶")
#: — "住手", "放心", "老婆" đứng riêng một dòng rất thường gặp mà không phải tên.
_COMMON_SURNAMES = frozenset(
    "王李张刘陈杨黄赵吴周徐孙马朱胡郭何高林罗郑梁谢宋唐许韩冯邓曹彭曾肖田董袁潘于蒋蔡余杜叶程"
    "苏魏吕丁任沈姚卢姜崔钟谭陆汪范金石廖贾夏韦付方白邹孟熊秦邱江尹薛闫段雷侯龙史陶黎贺顾毛郝"
    "龚邵万钱严覃武戴莫孔向汤楚萧柳慕宁凌云沐秋燕温洛楚陌君墨"
)
#: Từ LẶP không phải tên (gọi người thân, thán từ, trạng từ).
_REDUPLICATED_WORDS = frozenset({
    "爸爸", "妈妈", "哥哥", "姐姐", "弟弟", "妹妹", "爷爷", "奶奶", "宝宝", "叔叔", "伯伯",
    "舅舅", "姥姥", "婆婆", "公公", "谢谢", "等等", "哈哈", "嗯嗯", "啊啊", "好好", "慢慢",
    "看看", "试试", "刚刚", "常常", "天天", "渐渐", "走走", "坐坐", "快快", "乖乖", "嘿嘿",
    "呵呵", "嘻嘻", "呜呜", "哇哇", "拜拜", "姑姑", "嫂嫂", "宝贝",
})
_PUNCT_RE = re.compile(r"[\s，。！？、…“”‘’\"'!?,.~～—\-:：;；()（）]+")
#: Từ đứng riêng một dòng mà KHÔNG phải tên người (thán từ, đại từ, từ thường gặp).
_NOT_NAMES = frozenset({
    "什么", "怎么", "为什么", "哈哈", "哈哈哈", "哎呀", "哎呦", "好吧", "好的", "是的", "不是",
    "没有", "可以", "不行", "住口", "够了", "就是", "谢谢", "对不起", "老天", "老天爷", "师兄",
    "师姐", "师妹", "师弟", "你们", "我们", "他们", "大家", "等等", "等下", "等会儿", "废话",
    "当然", "明白", "卑鄙", "无耻", "混蛋", "放肆", "退婚", "燕子", "啊啊", "嗯嗯", "放开",
    "救命", "快跑", "小心", "闭嘴", "滚开", "拜见", "参见", "恭喜", "多谢", "师父", "师叔",
    # [v3.23.676] Từ thường mở đầu bằng chữ trùng HỌ phổ biến (白/马/江/高/万/方/金/石/龙/云/秋).
    "白痴", "马上", "江湖", "高手", "高兴", "万一", "万岁", "方便", "方法", "金子", "石头",
    "云端", "秋天", "江山", "白天", "马车", "金丹", "方才",
})
#: [v3.23.676] Từ dài hơn CHỨA chức danh nhưng không chỉ chức danh đó ("老爷爷" = ông cụ, không phải "lão gia").
_TITLE_FALSE_CONTAINERS: dict[str, tuple[str, ...]] = {"老爷": ("老爷爷",), "夫人": ("夫人们",)}
#: Chữ hầu như không có trong TÊN người (đại từ, trợ từ, từ thường) — loại dòng như "我的灵力".
_COMMON_CHARS = frozenset("我你他她它们的了吗呢吧是在不没有很都也就还这那么什怎谁哪个些说做要会能想看走来去好")
#: Đuôi của từ KHÔNG phải tên người dù đứng riêng một dòng (cảnh giới, công pháp, nơi chốn…):
#: "合体境", "筑基期", "太虚诀", "青云宗".
_NON_NAME_SUFFIXES = frozenset("境期阶级层功诀术法丹药剑阵宗派门城山峰殿阁府村镇国界")


@dataclass(frozen=True)
class UnlistedMention:
    """Một cách gọi chỉ người trong phụ đề chưa khớp ai trong bảng nhân vật."""

    term: str
    count: int
    line_ids: tuple[int, ...]


def _heads(characters: str) -> list[str]:
    """Phần TÊN của từng dòng bảng nhân vật (trước '—' / ':'), viết thường."""
    out = []
    for raw in (characters or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        head = re.split(r"\s[—–-]\s|—|:", line, maxsplit=1)[0]
        out.append(head.lower())
    return out


def _clean(text: str) -> str:
    return _PUNCT_RE.sub("", text or "")


#: Số cách gọi chưa khớp tối đa trả về (giữ lời gọi tự kiểm tra nhỏ).
MAX_UNLISTED_MENTIONS = 8


def find_unlisted_mentions(
    characters: str, source_lines: Sequence[Any], *, min_count: int = 2,
    limit: int = MAX_UNLISTED_MENTIONS, glossary: str = "",
) -> list[UnlistedMention]:
    """Cách gọi chỉ người xuất hiện ≥ ``min_count`` lần mà phần tên của bảng nhân vật chưa có.

    Args:
        characters: Bảng nhân vật.
        source_lines: Dòng phụ đề (có ``index`` và ``text``).
        min_count: Số lần tối thiểu (1 lần lẻ thường là nhắc qua).
        limit: Số cách gọi tối đa trả về.
        glossary: Bảng thuật ngữ — cách gọi đã được xếp vào thuật ngữ (vd "系统" là tên một vật
            sở hữu, không nói) không bị báo lại.

    Returns:
        Danh sách theo số lần giảm dần.
    """
    heads = _heads(characters)
    gloss_terms = _glossary_terms(glossary)
    found: Counter[str] = Counter()
    where: dict[str, list[int]] = {}

    def add(term: str, line_no: int) -> None:
        found[term] += 1
        where.setdefault(term, []).append(line_no)

    for line in source_lines:
        text = _clean(str(getattr(line, "original_text", None) or getattr(line, "text", "") or ""))
        if not text or not _CJK_RE.search(text) or _KANA_RE.search(text):
            continue
        line_no = int(getattr(line, "index", 0) or 0)
        rest = text
        for match in _SURNAME_TITLE_RE.finditer(text):
            surname = _surname(match.group(1))
            # Chỉ HỌ thật: "找师叔" (đi tìm sư thúc), "这师兄" không phải người theo họ.
            if surname not in _COMPOUND_SURNAMES and surname not in _COMMON_SURNAMES:
                continue
            add(surname + match.group(2), line_no)
            # "南宫师叔" là MỘT người theo họ — không đếm thêm "师叔" trơn cho cùng chỗ đó.
            rest = rest.replace(surname + match.group(2), "|")
        for title in TITLE_READINGS:
            scan = rest
            for container in _TITLE_FALSE_CONTAINERS.get(title, ()):
                scan = scan.replace(container, "|")
            if title in scan:
                add(title, line_no)
        # Tên đứng riêng thành cả một dòng (lời gọi: "瑶瑶", "叶然"); bỏ trợ từ cuối ("师妹啊").
        bare = text.lstrip("哎啊哦嗯唉喂呀诶呦哟嘿哼咦").rstrip("啊呀吧呢嘛啦哦哟喂")
        if (
            2 <= len(bare) <= 3 and all(_CJK_RE.match(ch) for ch in bare)
            and bare not in _NOT_NAMES
            and not any(title in bare for title in TITLE_READINGS)
            and not _SURNAME_TITLE_RE.search(bare)
            and not any(ch in _COMMON_CHARS for ch in bare)
            and bare[-1] not in _NON_NAME_SUFFIXES
            and (
                bare[0] in _COMMON_SURNAMES or bare[:2] in _COMPOUND_SURNAMES
                or (len(bare) == 2 and bare[0] == bare[1] and bare not in _REDUPLICATED_WORDS)
            )
        ):
            add(bare, line_no)

    out: list[UnlistedMention] = []
    for term, count in found.most_common():
        if count < min_count or _covered(term, heads) or any(term in g for g in gloss_terms):
            continue
        out.append(UnlistedMention(term, count, tuple(where[term][:12])))
        if len(out) >= limit:
            break
    return out


def _glossary_terms(glossary: str) -> list[str]:
    """Phần thuật ngữ GỐC (trước '=') của từng dòng bảng thuật ngữ."""
    out = []
    for raw in (glossary or "").splitlines():
        term = re.split(r"\s*(?:=>|=|→|:)\s*", raw.strip(), maxsplit=1)[0].strip()
        if term:
            out.append(term)
    return out


def _surname(prefix: str) -> str:
    """Họ trong 1-2 chữ đứng trước chức danh: họ kép nếu có, không thì chữ cuối."""
    return prefix if prefix in _COMPOUND_SURNAMES else prefix[-1]


def _covered(term: str, heads: list[str]) -> bool:
    """Cách gọi đã có trong phần tên của bảng (chữ gốc, cách đọc, hoặc họ của "họ + chức danh")."""
    if any(term in head for head in heads):
        return True
    readings = TITLE_READINGS.get(term, ())
    if any(reading in head for reading in readings for head in heads):
        return True
    match = _SURNAME_TITLE_RE.fullmatch(term)
    if match:
        surname = _surname(match.group(1))
        return any(surname in head for head in heads)
    return False


__all__ = ["MAX_UNLISTED_MENTIONS", "TITLE_READINGS", "UnlistedMention", "find_unlisted_mentions"]
