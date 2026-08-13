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
UNDER_TRANSLATION_RATIO_LATIN = 0.35     # nguồn Latin (Anh, Pháp…) — nén sang Việt mạnh hơn

# Dưới ngưỡng này thì mẫu quá nhỏ để trung vị có ý nghĩa thống kê -> không cảnh báo.
_MIN_SAMPLE = 12

# Dòng gốc quá ngắn (thán từ, tên riêng) dao động mạnh -> loại khỏi mẫu lẫn cảnh báo.
_MIN_SOURCE_CHARS = 4


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


def _resolve_threshold(
    usable: list[tuple[int, str, str]], threshold: float | None
) -> float:
    """Chọn ngưỡng theo HỌ NGÔN NGỮ NGUỒN của job nếu caller không chỉ định (hàm thuần).

    Nguồn Latin (Anh…) nén sang tiếng Việt mạnh hơn CJK nên cần ngưỡng chặt hơn để
    tránh dương-tính-giả. Quyết định theo TỈ LỆ dòng nguồn có ký tự CJK trong mẫu:
    < 20% coi là job Latin. Nếu caller truyền ``threshold`` tường minh thì tôn trọng.

    Args:
        usable:    Các cặp ``(idx, src, dst)`` đã lọc.
        threshold: Ngưỡng do caller chỉ định, hoặc ``None`` để tự dò.

    Returns:
        Ngưỡng bội-số-trung-vị sẽ dùng.
    """
    if threshold is not None:
        return threshold
    if not usable:
        return UNDER_TRANSLATION_RATIO
    cjk_lines = sum(1 for _, src, _ in usable if _CJK_CHAR_RE.search(src))
    is_cjk_job = cjk_lines / len(usable) >= 0.20
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
    usable = [
        (idx, src, dst)
        for idx, src, dst in pairs
        if len(src.strip()) >= _MIN_SOURCE_CHARS and src.strip()
    ]
    if len(usable) < _MIN_SAMPLE:
        return []

    effective_threshold = _resolve_threshold(usable, threshold)
    ratios = {idx: len(dst.strip()) / len(src.strip()) for idx, src, dst in usable}
    median_ratio = median(ratios.values())
    if median_ratio <= 0:
        return []

    suspects = [
        UnderTranslatedLine(
            index=idx,
            source_text=src.strip(),
            translated_text=dst.strip(),
            ratio=ratios[idx],
            median_ratio=median_ratio,
        )
        for idx, src, dst in usable
        if ratios[idx] < median_ratio * effective_threshold
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
    dang_ngo_truoc = {
        s.index
        for s in find_under_translated(
            [(idx, src, before) for idx, src, before, _ in items], threshold
        )
    }
    dang_ngo_sau = {
        s.index
        for s in find_under_translated(
            [(idx, src, after) for idx, src, _, after in items], threshold
        )
    }
    return dang_ngo_sau - dang_ngo_truoc
