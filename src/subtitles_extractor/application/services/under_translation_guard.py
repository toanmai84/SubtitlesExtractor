"""[v3.23.224] Phát hiện dòng bị DỊCH THIẾU (cắt mất nội dung) — thống kê TỰ HIỆU CHỈNH.

Bối cảnh: chuỗi v222-v223 cho thấy model dịch có thể rút gọn tới mức **mất nghĩa** mà mọi
chỉ số kỹ thuật đều không báo động (TTS thậm chí đẹp hơn). Ca thật đã ghi nhận:

* "cháu đã nói về chú thế nào," -> "Chú?" (mất sạch nghĩa)
* "Không phải chú không cho cháu cơ hội." -> "Cho cơ hội," (mất phủ định kép)
* "đương nhiên sẽ nhận lại" -> "sẽ chịu,"

Vấn đề đo lường: **không có hằng số phổ quát** cho "một dòng dịch nên dài bao nhiêu" —
tỉ lệ độ dài Việt/CJK phụ thuộc ngôn ngữ nguồn, thể loại, văn phong. Đoán một hằng số là
đoán mò.

Giải pháp: **lấy chính bộ phim làm chuẩn**. Trong cùng một job, mọi dòng đều đi qua cùng
cặp ngôn ngữ và cùng văn phong, nên tỉ lệ ``len(dịch) / len(gốc)`` có một mức TRUNG VỊ ổn
định. Dòng nào có tỉ lệ thấp bất thường **so với chính bộ phim đó** là dòng đáng ngờ —
không cần biết trước hằng số nào.

Module này chỉ CẢNH BÁO (log), không tự sửa: quyết định cuối cùng thuộc về người dùng, và
một số dòng ngắn là hợp lệ thật ("Ừm.", "Hả?").
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from statistics import median

__all__ = [
    "MIN_SHORTFALL_LETTERS",
    "UNDER_TRANSLATION_RATIO",
    "UNDER_TRANSLATION_RATIO_LATIN",
    "UnderTranslatedLine",
    "find_regressions",
    "find_under_translated",
    "log_under_translated",
]

logger = logging.getLogger(__name__)

# Regex ký tự CJK (Hán/Nhật/Hàn) — dùng để đoán họ ngôn ngữ NGUỒN của job.
_CJK_CHAR_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u30ff\uac00-\ud7af]")

# Dòng đáng ngờ khi tỉ lệ độ dài của nó thấp hơn ngưỡng này SO VỚI TRUNG VỊ của chính
# job. Ngưỡng là BỘI SỐ của trung vị TỪNG phim nên tự thích ứng theo cặp ngôn ngữ.
#
# [v3.23.353] NGƯỠNG THEO HỌ NGÔN NGỮ NGUỒN. Log chạy thật lộ ra: một NGƯỠNG PHẲNG không
# tách được hai nhóm cùng nằm ở ~40-48% trung vị:
#   • Nguồn CJK: câu bị mất phủ định/mất một phần nghĩa vẫn ~40% (VD '不是叔叔不给你机会'
#     → 'Cho cơ hội,' ≈41%) → PHẢI cảnh báo.
#   • Nguồn Latin (Anh): câu mệnh lệnh/thành ngữ nén hợp lệ cũng ~40-48% ('Please, take a
#     seat.'→'Mời ngồi.' ≈43%, 'Speak up.'→'Nói đi.' ≈42%) → KHÔNG nên cảnh báo.
# Tiếng Việt bỏ mạo từ/trợ động từ nên nén câu Latin mạnh hơn nhiều so với CJK → dùng
# ngưỡng CHẶT HƠN cho nguồn Latin. Hiệu chỉnh trên log thật (890 dòng Anh→Việt): 0.35 loại
# sạch 7 dương-tính-giả mà vẫn giữ 3 ca mất nghĩa thật (13-25%); nguồn CJK giữ 0.50 để
# không bỏ sót ca mất phủ định (~41%).
UNDER_TRANSLATION_RATIO = 0.50           # nguồn CJK (Hán/Nhật/Hàn)
UNDER_TRANSLATION_RATIO_LATIN = 0.35     # nguồn Latin — nén sang Việt mạnh hơn

# Dưới ngưỡng này thì mẫu quá nhỏ để trung vị có ý nghĩa thống kê -> không cảnh báo.
_MIN_SAMPLE = 12

# Dòng gốc quá ngắn (thán từ, tên riêng) dao động mạnh -> loại khỏi mẫu lẫn cảnh báo.
# [v3.23.553] Với nguồn CJK, ngưỡng áp lên SỐ CHỮ HÁN (không tính dấu câu).
_MIN_SOURCE_CHARS = 4

# [v3.23.553] Với nguồn CJK chỉ cảnh báo khi bản dịch THIẾU ÍT NHẤT chừng này chữ cái so
# với mức kỳ vọng. Lý do (đo trên 5 lần chạy thật, phim 3902 dòng): tỉ lệ của dòng ngắn bị
# LƯỢNG TỬ HOÁ — thiếu 1 chữ cái là đủ lật cờ giữa các lần chạy dù chất lượng không đổi
# ('Xem ra,', 'nể mặt em đi,'...). Biên 2 xoá sạch nhóm nhiễu này; biên 3 thì bỏ sót ca
# mất nội dung thật ('这就是神仙姐姐的' → 'đây chính là...', '所以我才跟着大家' →
# 'Nên cháu hùa,'), nên chọn 2.
MIN_SHORTFALL_LETTERS = 2

# [v3.23.553] Gốc CJK có ≤ chừng này chữ Hán KHÁC NHAU là từ tượng thanh/lặp
# ('哈哈哈哈哈哈哈') — không có nội dung từ vựng để "dịch thiếu", nên loại khỏi mẫu.
_MIN_DISTINCT_SOURCE_CHARS = 2

# Chữ cái (mọi hệ chữ, kể cả chữ Việt có dấu) — KHÔNG gồm chữ số, dấu câu, khoảng trắng.
_LETTER_RE = re.compile(r"[^\W\d_]")

# [v3.23.578] Số viết bằng chữ Hán ("三十四点五万", "百分之两百") dịch ra CHỮ SỐ
# ("345.000", "200%"), mà bản dịch chỉ đếm CHỮ CÁI — nên dòng số bị coi là "dịch
# thiếu" với tỉ lệ 7-14% trung vị. Nhật ký thật (phim 5540 dòng): 14/29 cờ ở bản dịch
# thô là dòng số, và cờ đó đẩy vào ``needs_expansion`` khiến giai đoạn sau "bổ sung" bằng
# câu của dòng kề ("3.450.000 Tệ" → "3.450.000 Tệ đã vào tài khoản." trong khi dòng sau
# là "đã vào tài khoản."). Chỉ bỏ CỤM ≥ 2 chữ số có đơn vị (十百千万亿点) — "十分",
# "一个" giữ nguyên — và CHỈ khi bản dịch có chữ số: "一点" → "một chút", "万一" → "lỡ
# như" vẫn dịch ra chữ cái nên phải được đếm (bỏ vô điều kiện làm lệch trung vị cả ở
# phim không có dòng số nào, đo trên 34 lần chạy thật).
_NUMERAL_RUN_RE = re.compile(
    "(?:百分之)?"
    "(?=[零〇一二三四五六七八九十百千万萬亿億两兩点點]*[十百千万萬亿億点點])"
    "[零〇一二三四五六七八九十百千万萬亿億两兩点點]{2,}"
)
_DIGIT_RE = re.compile(r"\d")


@dataclass(frozen=True, slots=True)
class UnderTranslatedLine:
    """Một dòng bị nghi dịch thiếu nội dung."""

    index: int
    source_text: str
    translated_text: str
    ratio: float
    median_ratio: float

    @property
    def severity(self) -> float:
        """Mức bất thường: tỉ lệ của dòng so với trung vị job (càng nhỏ càng nặng)."""
        return self.ratio / self.median_ratio if self.median_ratio > 0 else 1.0


def _is_cjk_job(pairs: list[tuple[int, str, str]]) -> bool:
    """Job có nguồn CJK khi ≥ 20% dòng nguồn (khác rỗng) chứa ký tự CJK (hàm thuần)."""
    sources = [src for _, src, _ in pairs if src.strip()]
    if not sources:
        return True
    cjk_lines = sum(1 for src in sources if _CJK_CHAR_RE.search(src))
    return cjk_lines / len(sources) >= 0.20


def _measure(text: str, is_source: bool, is_cjk_job: bool) -> int:
    """Độ dài dùng để so tỉ lệ (hàm thuần).

    [v3.23.553] Nguồn CJK: đếm CHỮ HÁN ở gốc và CHỮ CÁI ở bản dịch — dấu câu không mang
    nội dung nhưng trước đây chiếm tới 1/4 độ dài dòng ngắn ('怎么样?' = 4 thay vì 3), làm
    tỉ lệ nhảy theo cách chấm câu chứ không theo nội dung. Nguồn Latin giữ nguyên cách đo
    cũ vì ngưỡng 0,35 đã được hiệu chỉnh riêng theo cách đo đó trên dữ liệu thật.
    """
    if not is_cjk_job:
        return len(text.strip())
    if is_source:
        return len(_CJK_CHAR_RE.findall(text))
    return len(_LETTER_RE.findall(text))


def _measure_source(source: str, translation: str, is_cjk_job: bool) -> int:
    """Độ dài câu gốc, bỏ cụm số chữ Hán đã được dịch ra CHỮ SỐ (hàm thuần).

    Xem ``_NUMERAL_RUN_RE``: chỉ bỏ khi bản dịch có chữ số, vì chữ số không được đếm ở
    phía bản dịch.
    """
    if is_cjk_job and _DIGIT_RE.search(translation):
        source = _NUMERAL_RUN_RE.sub("", source)
    return _measure(source, True, is_cjk_job)


@dataclass(frozen=True, slots=True)
class _Measured:
    """Một dòng đã đo, dùng nội bộ để tránh đo lại."""

    index: int
    source_text: str
    translated_text: str
    source_len: int
    translated_len: int

    @property
    def ratio(self) -> float:
        return self.translated_len / self.source_len


def _measure_pairs(
    pairs: list[tuple[int, str, str]], is_cjk_job: bool
) -> list[_Measured]:
    """Đo các cặp và loại dòng gốc quá ngắn (hàm thuần)."""
    measured: list[_Measured] = []
    for idx, src, dst in pairs:
        source_len = _measure_source(src, dst, is_cjk_job)
        if source_len < _MIN_SOURCE_CHARS:
            continue
        if is_cjk_job and _is_repetitive(src):
            continue
        measured.append(
            _Measured(
                index=idx,
                source_text=src.strip(),
                translated_text=dst.strip(),
                source_len=source_len,
                translated_len=_measure(dst, False, is_cjk_job),
            )
        )
    return measured


def _is_repetitive(source_text: str) -> bool:
    """Gốc CJK chỉ gồm vài chữ Hán lặp lại — tiếng cười, tiếng kêu (hàm thuần)."""
    distinct = set(_CJK_CHAR_RE.findall(source_text))
    return len(distinct) <= _MIN_DISTINCT_SOURCE_CHARS


def _is_suspect(
    line: _Measured, median_ratio: float, threshold: float, is_cjk_job: bool
) -> bool:
    """Dòng có ngắn bất thường so với mức kỳ vọng của phim không (hàm thuần)."""
    expected_len = median_ratio * threshold * line.source_len
    margin = MIN_SHORTFALL_LETTERS if is_cjk_job else 0
    return line.translated_len < expected_len - margin


def _absorbed_by_neighbours(
    line: _Measured,
    pairs_by_index: dict[int, tuple[str, str]],
    median_ratio: float,
    is_cjk_job: bool,
) -> bool:
    """Phần thiếu của dòng đã nằm ở dòng kề ±1 chưa (hàm thuần).

    [v3.23.553] Dòng phụ đề là MẢNH câu; tiếng Việt đảo trật tự nên nội dung hay dời sang
    dòng kề. Ca thật v3.23.552: '这就是神仙姐姐的' → 'đây chính là...' còn dòng sau
    '元神碎片' → 'mảnh nguyên thần của Thần Tiên Tỷ Tỷ?' — không mất gì. Cờ oan loại này
    nguy hiểm hơn bỏ sót: nó đẩy vào ``needs_expansion`` khiến model "bổ sung" và LẶP nội
    dung giữa hai dòng. Cùng chính sách ±1 dòng với lưới tên riêng. Chỉ áp cho nguồn CJK;
    nguồn Latin giữ nguyên hành vi đã hiệu chỉnh.
    """
    if not is_cjk_job:
        return False
    deficit = median_ratio * line.source_len - line.translated_len
    surplus = 0.0
    for neighbour_index in (line.index - 1, line.index + 1):
        neighbour = pairs_by_index.get(neighbour_index)
        if neighbour is None:
            continue
        neighbour_source, neighbour_translation = neighbour
        expected = median_ratio * _measure_source(
            neighbour_source, neighbour_translation, is_cjk_job
        )
        actual = _measure(neighbour_translation, False, is_cjk_job)
        surplus += max(0.0, actual - expected)
    return surplus >= deficit


def _resolve_threshold(is_cjk_job: bool, threshold: float | None) -> float:
    """Chọn ngưỡng theo HỌ NGÔN NGỮ NGUỒN nếu caller không chỉ định (hàm thuần).

    Nguồn Latin (Anh…) nén sang tiếng Việt mạnh hơn CJK nên cần ngưỡng chặt hơn để
    tránh dương-tính-giả. Nếu caller truyền ``threshold`` tường minh thì tôn trọng.
    """
    if threshold is not None:
        return threshold
    return UNDER_TRANSLATION_RATIO if is_cjk_job else UNDER_TRANSLATION_RATIO_LATIN


def find_under_translated(
    pairs: list[tuple[int, str, str]],
    threshold: float | None = None,
) -> list[UnderTranslatedLine]:
    """Tìm các dòng có bản dịch ngắn BẤT THƯỜNG so với mặt bằng của chính job (hàm thuần).

    Args:
        pairs: Danh sách ``(line_no, văn_bản_gốc, bản_dịch)``.
        threshold: Tỉ lệ so với TRUNG VỊ của job để coi là đáng ngờ. ``None`` = TỰ CHỌN
            theo họ ngôn ngữ nguồn (CJK→0.50, Latin→0.35). Truyền số để ép cứng.

    Returns:
        Danh sách dòng đáng ngờ, sắp theo mức nặng (nặng nhất trước). Rỗng nếu mẫu quá
        nhỏ để thống kê có ý nghĩa.
    """
    is_cjk_job = _is_cjk_job(pairs)
    measured = _measure_pairs(pairs, is_cjk_job)
    if len(measured) < _MIN_SAMPLE:
        return []
    median_ratio = median(line.ratio for line in measured)
    if median_ratio <= 0:
        return []
    effective_threshold = _resolve_threshold(is_cjk_job, threshold)
    pairs_by_index = {idx: (src, dst) for idx, src, dst in pairs}
    suspects = [
        UnderTranslatedLine(
            index=line.index,
            source_text=line.source_text,
            translated_text=line.translated_text,
            ratio=line.ratio,
            median_ratio=median_ratio,
        )
        for line in measured
        if _is_suspect(line, median_ratio, effective_threshold, is_cjk_job)
        and not _absorbed_by_neighbours(line, pairs_by_index, median_ratio, is_cjk_job)
    ]
    return sorted(suspects, key=lambda s: s.severity)


def log_under_translated(pairs: list[tuple[int, str, str]]) -> list[UnderTranslatedLine]:
    """Chạy :func:`find_under_translated` và GHI LOG cảnh báo (không tự sửa).

    Không tự sửa vì hai lẽ: (1) một số dòng ngắn là ĐÚNG ("Ừm.", "Hả?"); (2) dịch lại tự
    động có thể sinh vòng lặp tốn kém mà vẫn không chắc đúng. Người dùng cần được BÁO để
    tự kiểm.

    Args:
        pairs: Danh sách ``(line_no, văn_bản_gốc, bản_dịch)``.

    Returns:
        Danh sách dòng đáng ngờ (đã log).
    """
    suspects = find_under_translated(pairs)
    if not suspects:
        return []
    logger.warning(
        "Nghi DỊCH THIẾU nội dung ở %d dòng (bản dịch ngắn bất thường so với mặt bằng "
        "chung của phim). Nên rà lại các dòng này:",
        len(suspects),
    )
    for s in suspects[:10]:
        logger.warning(
            "  dòng %d: chỉ dài %.0f%% mức bình thường | gốc: '%s' -> dịch: '%s'",
            s.index,
            s.severity * 100,
            s.source_text[:40],
            s.translated_text[:40],
        )
    return suspects


def find_regressions(
    items: list[tuple[int, str, str, str]],
    threshold: float | None = None,
) -> set[int]:
    """[v3.23.226] Tìm các dòng bị giai đoạn dịch KẾ TIẾP làm HỎNG (hàm thuần).

    Bằng chứng từ log chạy thật (v225): vòng tự sửa có tác dụng — mọi dòng được gắn cờ
    ``needs_expansion`` đều được bổ sung — NHƯNG mỗi giai đoạn lại làm rớt nội dung ở một
    dòng KHÁC vốn đang tốt::

        GĐ2 -> hỏng: 82, 71, 80
        GĐ3 -> sửa xong 82/71/80, nhưng LÀM HỎNG 84
        GĐ4 -> sửa xong 84, nhưng LÀM HỎNG LẠI 82

    Đây là "tam sao thất bản": ta đuổi bắt vòng tròn, mỗi lượt tinh chỉnh vừa sửa lỗi cũ
    vừa gieo lỗi mới. Dạy model "hãy bổ sung" là chưa đủ — phải CHẶN nó phá dòng đang tốt.

    Nguyên tắc: **giai đoạn sau chỉ được GIỮ hoặc CẢI THIỆN, không được làm TỆ ĐI.** Dòng
    nào ĐANG BÌNH THƯỜNG ở đầu vào mà thành ĐÁNG NGỜ ở đầu ra thì tầng gọi hoàn nguyên nó
    về bản trước — bản trước tuy có thể kém trau chuốt, nhưng KHÔNG mất nội dung, và mất
    nội dung là lỗi nặng hơn nhiều.

    Args:
        items: Danh sách ``(line_no, văn_bản_gốc, bản_dịch_TRƯỚC, bản_dịch_SAU)``.
        threshold: Tỉ lệ so với trung vị để coi là đáng ngờ (dùng chung với lưới chính).

    Returns:
        Tập ``line_no`` mà giai đoạn vừa rồi đã làm hỏng (nên hoàn nguyên).
    """
    before_pairs = [(idx, src, before) for idx, src, before, _ in items]
    after_by_index = {idx: after for idx, _, _, after in items}
    # Hoàn nguyên MỘT dòng của một cặp đã đảo trật tự sẽ gây lặp hoặc mất nội dung, nên
    # dòng thiếu mà dòng kề (bản SAU) đã gánh phần thiếu thì không tính là hồi quy.
    after_pairs_by_index = {idx: (src, after) for idx, src, _, after in items}
    is_cjk_job = _is_cjk_job(before_pairs)
    measured_before = _measure_pairs(before_pairs, is_cjk_job)
    if len(measured_before) < _MIN_SAMPLE:
        return set()
    # [v3.23.553] MỘT trung vị chung, lấy từ bản TRƯỚC. Trước đây mỗi bản có trung vị
    # riêng: khi giai đoạn sau viết dài hơn ở phần lớn phim, trung vị của bản SAU tăng và
    # những dòng KHÔNG HỀ ĐỔI chữ nào bị coi là "vừa hỏng" — log thật v3.23.552: 26/26
    # dòng bị hoàn nguyên ở giai đoạn Bản địa hoá đều giống hệt bản Tinh chỉnh.
    median_ratio = median(line.ratio for line in measured_before)
    if median_ratio <= 0:
        return set()
    effective_threshold = _resolve_threshold(is_cjk_job, threshold)
    regressions: set[int] = set()
    for line in measured_before:
        if _is_suspect(line, median_ratio, effective_threshold, is_cjk_job):
            continue
        after_text = after_by_index.get(line.index, "")
        after_len = _measure(after_text, False, is_cjk_job)
        # Chỉ tính khi dòng THỰC SỰ bị rút ngắn — dòng giữ nguyên hoặc dài ra không thể
        # là "giai đoạn này làm mất nội dung".
        if after_len >= line.translated_len:
            continue
        after_line = _Measured(
            index=line.index,
            source_text=line.source_text,
            translated_text=after_text.strip(),
            source_len=line.source_len,
            translated_len=after_len,
        )
        if _is_suspect(
            after_line, median_ratio, effective_threshold, is_cjk_job
        ) and not _absorbed_by_neighbours(
            after_line, after_pairs_by_index, median_ratio, is_cjk_job
        ):
            regressions.add(line.index)
    return regressions
