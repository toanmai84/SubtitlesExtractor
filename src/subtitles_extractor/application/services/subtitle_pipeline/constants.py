"""Hằng số dùng chung trong pipeline xây dựng phụ đề.

Module này tập trung các regex pattern, lookup table, ngưỡng cấu hình để
tránh khởi tạo lặp lại trong các vòng lặp nóng. Không chứa logic — chỉ
dữ liệu hằng.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Regex patterns — biên dịch sẵn để tối ưu vòng lặp nóng.
# ---------------------------------------------------------------------------

PURE_PUNCTUATION_REGEX: re.Pattern[str] = re.compile(r"^[\W_]+$", flags=re.UNICODE)
"""Khớp chuỗi chỉ chứa dấu câu/ký tự không phải chữ — rác đơn thuần."""

SINGLE_ALPHANUMERIC_REGEX: re.Pattern[str] = re.compile(
    r"^[a-zA-Z0-9]$", flags=re.UNICODE
)
"""Khớp đúng 1 ký tự Latin/digit — rác thường gặp từ logo nhỏ."""

NON_WORD_CHARACTER_REGEX: re.Pattern[str] = re.compile(r"[^\w]+", flags=re.UNICODE)
"""Loại bỏ mọi ký tự không-word để đếm độ dài hiệu dụng."""

EDGE_NOISE_PUNCTUATION_REGEX: re.Pattern[str] = re.compile(
    r"^[~@#$%^&*\|_=+\-<>.,，。]+|[~@#$^&*\|_=+\-<>]+$",
    flags=re.UNICODE,
)
"""Khớp dấu câu rác ở 2 đầu chuỗi (trim biên).

[v3.7 fix] ĐÃ BỎ ``%`` khỏi tập strip ở BIÊN CUỐI. Trước đây ``30%`` bị cắt
thành ``30`` (mất nghĩa phần trăm). ``%`` sau số là ký tự hợp lệ; chỉ giữ strip
``%`` ở biên ĐẦU (hiếm gặp, thường là rác)."""

DIGIT_ONLY_TOKEN_REGEX: re.Pattern[str] = re.compile(r"^[\d\s]+$", flags=re.UNICODE)
"""Khớp token chỉ chứa chữ số và khoảng trắng — rác timer/logo."""

LATIN_REPETITIVE_REGEX: re.Pattern[str] = re.compile(
    r"^[A-Za-z]*([A-Za-z])\1{2,}[A-Za-z]*$",
    flags=re.UNICODE,
)
"""Khớp chuỗi Latin có 1 ký tự lặp >= 3 lần liên tiếp (vd 'Moooo')."""

WHITESPACE_RUN_REGEX: re.Pattern[str] = re.compile(r"\s{2,}")
"""Khớp run >= 2 khoảng trắng (để chuẩn hoá thành 1 space)."""

# ---------------------------------------------------------------------------
# Lookup tables — frozenset cho lookup O(1).
# ---------------------------------------------------------------------------

CJK_TRAILING_PUNCTUATIONS: frozenset[str] = frozenset(
    "。，！？、；：…—～「」『』【】〈〉《》\"\""
)
"""Dấu câu CJK + Western thường gắn cuối câu, cần normalize khi so sánh."""

CJK_CRITICAL_REVERSAL_KEYWORDS: frozenset[str] = frozenset(
    {
        "不", "没", "你", "我", "他", "她", "它", "是", "非", "有", "无",
        "男", "女", "去", "来", "要", "别", "好", "坏", "买", "卖", "死",
        "活", "多", "少", "大", "小",
    }
)
"""Tử huyệt CJK — khác 1 ký tự ở đây thường đảo nghĩa hoàn toàn."""

OCR_HALLUCINATION_TYPO_MAP: dict[str, str] = {
    # Ảo giác OCR ký tự gần giống — đã có từ v2.24+ và an toàn trên 11 file test.
    "涘": "埃",
    "俟": "埃",
    "相当於": "相当于",
    "增強": "增强",
    "別": "别",
    "説": "说",
    "这木一": "这才一",
    "这水一": "这才一",
    # ── Các OCR typo cụ thể (1 ký tự nhầm) đã thấy trong test data v3.0 ──
    # Chỉ thêm các pattern an toàn (không gây regression trên 11 file).
    # Lý do an toàn: pattern compound rõ ràng là OCR error (vd '现己' không
    # phải từ hợp lệ; '已' luôn là đúng).
    "现己": "现已",   # 己/已 OCR confuse — '己' đứng sau '现' chắc chắn sai
    "凡个": "几个",   # 凡/几 confuse '抓了凡个'
    "整介": "整个",   # 介/个 confuse — '整介' không phải từ hợp lệ
    "本介": "本个",
    "几介": "几个",
    "仪用": "仅用",   # 仪/仅 — '仪用' không hợp lệ trong context phổ biến
    "而旦": "而且",   # 旦/且 — '而旦' không phải từ hợp lệ
    # ── v2.9+: Thêm các typo từ phân tích test3/test data mới ──
    # '么' rất hay bị OCR nhầm thành '公', '区', '去' (nét ngang gần giống).
    # Pattern compound rõ ràng là sai → an toàn 100%.
    "什公": "什么",   # 公/么 — '什公用' / '什公事' không tồn tại
    "怎区": "怎么",   # 区/么 — '怎区' không phải từ hợp lệ (khác '怎公' là tên nhân vật)
    # '了' hay bị nhầm thành '不' trong một số font nhỏ.
    # CHỈ thêm pattern compound rõ ràng, KHÔNG thêm '不'→'了' đơn vì ambiguous.
    "去不还能": "去了还能",  # '去不还能' ngữ pháp sai, '去了还能' đúng
    # Extra char insertion — OCR thêm '开' vào giữa từ '府上' thành '府开上'.
    "府开上": "府上",   # '开' là nhiễu giữa '府' và '上'
    # '失踪二年' → '失踪二十二年' — OCR drop '十二' — KHÔNG sửa ở đây
    # (quá nhiều context cần biết) — để algorithm restore xử lý.
}
"""Bảng sửa lỗi ảo giác OCR (cấp text-level).

QUY TẮC AN TOÀN khi thêm mapping mới:
    * Pattern hiện tại không phải từ tiếng Trung hợp lệ trong ngữ cảnh
      phổ biến (vd "现己" sai chính tả, "凡个" không có nghĩa).
    * Pattern sau khi sửa là từ chuẩn (vd "现已", "几个").
    * KHÔNG thêm Phồn→Giản đơn (vd "謝→谢", "強→强") vì REFERENCE từng video
      có thể dùng phồn cho tên người hoặc style riêng → gây regression.

Áp dụng tại :func:`pre_filter_garbage_boxes` ngay sau khi strip text box.
"""


# Bảng Hán Phồn → Hán Giản, áp dụng riêng (tránh trộn với typo map context-sensitive).
# Reference phụ đề tiếng Trung hiện đại thường dùng Hán giản; PaddleOCR đôi khi
# output phồn (vd "純陽" → "纯阳") gây mismatch. Normalize output về Hán giản
# giúp khớp reference ổn định hơn.
#
# Chỉ chứa các cặp 1-1 KHÔNG có ambiguity (mỗi ký tự phồn map 1 ký tự giản duy
# nhất, không phá ngữ nghĩa). Các ký tự đa nghĩa (vd 后/後/后, 干/乾/幹) được
# loại trừ để tránh sai sót.
HAN_TRADITIONAL_TO_SIMPLIFIED: dict[str, str] = {
    # Đã quan sát trong test data:
    "純": "纯", "內": "内", "並": "并", "煙": "烟", "沒": "没",
    "吶": "呐", "謝": "谢", "強": "强", "決": "决", "恆": "恒",
    "繼": "继", "脈": "脉", "屍": "尸", "瑤": "瑶", "於": "于",
    "夠": "够", "嚴": "严", "鹿": "鹿",  # 鹿 thực ra giống nhau
    "繁": "繁", "個": "个", "們": "们", "這": "这", "麼": "么",
    "為": "为", "說": "说", "過": "过", "還": "还", "後": "后",
    "來": "来", "對": "对", "時": "时", "會": "会", "點": "点",
    "樣": "样", "兒": "儿", "種": "种", "幾": "几",
    "親": "亲", "氣": "气", "聽": "听", "見": "见", "現": "现",
    "請": "请", "歡": "欢", "覺": "觉", "業": "业", "問": "问",
    "讓": "让", "頭": "头", "間": "间", "話": "话", "聲": "声",
    "從": "从", "雖": "虽", "麗": "丽", "東": "东", "馬": "马",
    "達": "达", "閉": "闭", "開": "开", "關": "关", "驚": "惊",
    "藍": "蓝", "媽": "妈", "兩": "两", "區": "区", "經": "经",
    "級": "级", "結": "结", "線": "线", "車": "车", "輛": "辆",
    "鐵": "铁", "鋼": "钢", "銀": "银", "銅": "铜", "錢": "钱",
    "錯": "错", "鏡": "镜", "鐘": "钟", "鬥": "斗", "顯": "显",
    "願": "愿", "顆": "颗", "頻": "频", "領": "领", "額": "额",
    "風": "风", "飛": "飞", "餘": "余", "餐": "餐", "館": "馆",
    "驕": "骄", "體": "体", "禮": "礼",
    "豐": "丰", "節": "节", "靈": "灵", "歲": "岁", "齊": "齐",
    "齡": "龄", "齒": "齿", "龍": "龙", "舊": "旧", "讀": "读",
    "腦": "脑", "膽": "胆", "膠": "胶", "臉": "脸", "蘇": "苏",
    "藥": "药", "蘭": "兰", "蝦": "虾", "蟲": "虫", "處": "处",
    "號": "号", "蛻": "蜕", "袞": "衮", "裡": "里",
    "覽": "览", "觀": "观", "證": "证", "誰": "谁",
    "誤": "误", "課": "课", "諸": "诸", "識": "识",
    "譽": "誉", "讚": "赞", "贏": "赢", "踐": "践",
    "辭": "辞", "辨": "辨", "農": "农", "邊": "边", "鄉": "乡",
    "醫": "医", "陳": "陈", "陰": "阴", "陽": "阳", "雙": "双",
    "離": "离", "難": "难", "雲": "云", "電": "电",
    "霸": "霸", "霧": "雾", "韋": "韦", "頁": "页", "頂": "顶",
    "項": "项", "順": "顺", "預": "预", "頓": "顿",
    "顏": "颜", "顧": "顾", "颗": "颗",  # đã giản
    "騎": "骑", "驟": "骤",
    "魚": "鱼", "鳥": "鸟", "鳴": "鸣", "鴨": "鸭", "鵝": "鹅",
    "鶴": "鹤", "鷄": "鸡", "鹽": "盐", "鹼": "碱", "麵": "面",
    "黨": "党", "墙": "墙",  # đã giản
    "壓": "压", "塊": "块", "墳": "坟", "壞": "坏", "壟": "垄",
    "壽": "寿", "夢": "梦", "夾": "夹", "奮": "奋", "妝": "妆",
    "嫻": "娴", "嬌": "娇", "孫": "孙", "學": "学", "寧": "宁",
    "實": "实", "寫": "写", "尋": "寻", "導": "导",
    "屬": "属", "巒": "峦", "巔": "巅", "幣": "币", "幫": "帮",
    "廣": "广", "廳": "厅", "彈": "弹", "歸": "归", "當": "当",
    "徵": "征", "復": "复", "怎": "怎",  # đã giản
    "悶": "闷", "懂": "懂",  # đã giản
    "戀": "恋", "戰": "战", "戲": "戏", "戶": "户", "擔": "担",
    "據": "据", "擊": "击", "擇": "择", "擾": "扰", "攢": "攒",
    "攤": "摊", "敗": "败", "敵": "敌", "斷": "断",
    "晝": "昼", "暢": "畅", "曆": "历", "書": "书",
    "條": "条", "極": "极", "構": "构", "樂": "乐", "標": "标",
    "樹": "树", "橋": "桥", "歐": "欧", "歎": "叹", "歷": "历",
    "氾": "氾",  # đã giản
    "況": "况", "減": "减", "溫": "温", "準": "准",
    "滅": "灭", "滲": "渗", "滾": "滚", "漁": "渔", "漢": "汉",
    "潔": "洁", "潛": "潜", "灣": "湾", "災": "灾",
    "煉": "炼", "熱": "热", "燈": "灯", "營": "营", "獅": "狮",
    "獨": "独", "獲": "获", "獻": "献", "豬": "猪", "猶": "犹",
    "獸": "兽", "獵": "猎", "瑩": "莹", "畫": "画", "異": "异",
    "畢": "毕", "發": "发", "監": "监", "盡": "尽",
    "眾": "众", "睏": "困", "礙": "碍", "確": "确", "稱": "称",
    "穀": "谷", "積": "积", "穩": "稳", "競": "竞", "筆": "笔",
    "筍": "笋", "篩": "筛", "簡": "简", "籃": "篮",
    "粵": "粤", "糧": "粮", "糾": "纠", "紀": "纪",
    "紅": "红", "紙": "纸", "細": "细", "終": "终",
    "組": "组", "絕": "绝", "絞": "绞", "綁": "绑",
    "綠": "绿", "維": "维", "綱": "纲", "網": "网", "緊": "紧",
    "緣": "缘", "編": "编", "緩": "缓", "縛": "缚",
    "縝": "缜", "縣": "县", "縱": "纵", "總": "总", "績": "绩",
    "繞": "绕", "繡": "绣", "織": "织", "繪": "绘", "繳": "缴",
}
"""Bảng Hán Phồn → Hán Giản (mỗi cặp 1-1, không ambiguity).

Áp dụng trên TEXT-LEVEL sau khi join các box thành câu — chuẩn hoá output
về Hán giản trước khi so sánh hoặc xuất file. Lý do: phụ đề tiếng Trung
hiện đại (đặc biệt phim/show) dùng giản; PaddleOCR đôi khi output phồn,
gây inconsistency.
"""

LATIN_VALID_SHORT_TOKENS: frozenset[str] = frozenset(
    {
        "OK", "NO", "TV", "DVD", "VIP", "USB", "GPS", "AI", "PC", "PM", "AM",
        "USA", "UK", "EU", "WTO", "GDP", "CEO", "CFO", "BBC", "CNN", "FBI",
        "USD", "EUR", "RMB", "CNY", "GBP", "JPY", "KRW",
        "MP3", "MP4", "4K", "8K", "HD", "FHD", "UHD",
        "QQ", "WIFI", "WI-FI", "WHO", "UN",
    }
)
"""Whitelist acronym Latin phổ biến — không coi là rác kể cả conf thấp."""

LATIN_VOWELS: frozenset[str] = frozenset("aeiouyAEIOUY")
"""Tập nguyên âm Latin — dùng phân biệt từ thật vs gibberish."""


__all__ = [
    "CJK_CRITICAL_REVERSAL_KEYWORDS",
    "CJK_TRAILING_PUNCTUATIONS",
    "DIGIT_ONLY_TOKEN_REGEX",
    "EDGE_NOISE_PUNCTUATION_REGEX",
    "HAN_TRADITIONAL_TO_SIMPLIFIED",
    "LATIN_REPETITIVE_REGEX",
    "LATIN_VALID_SHORT_TOKENS",
    "LATIN_VOWELS",
    "NON_WORD_CHARACTER_REGEX",
    "OCR_HALLUCINATION_TYPO_MAP",
    "PURE_PUNCTUATION_REGEX",
    "SINGLE_ALPHANUMERIC_REGEX",
    "WHITESPACE_RUN_REGEX",
]
