"""[v3.23.590] Hai dòng CÁCH XA nhau mang cùng một bản dịch dù câu gốc khác hẳn.

Chuẩn đoán thật v3.23.589 (phim "灭门惨案II"): dòng 1141 "一二三" (đếm "một, hai, ba")
ra "Một chiếc thuyền nan lướt nhẹ trôi" — nguyên văn bản dịch của dòng 1165 "一葉輕舟去",
cách đó 24 dòng. Sai ngay từ bản dịch thô, đi qua mọi giai đoạn vào bản cuối; không cờ
chuẩn đoán nào báo (``duplicate_adjacent`` chỉ xét hai dòng LIỀN KỀ). Cùng lỗi ở lượt
v3.23.580 của phim này.

Đo trên bản cuối của 54 lượt dịch thật (ba phim), cặp dòng cách nhau 2-60 dòng, bản dịch
giống hệt từ ba từ trở lên, câu gốc chung dưới một nửa số chữ: 87 cặp ở 36 lượt. Soát tay:
nhiều cặp là dòng mang lời của dòng khác — "宋" (tiếng gọi dở "Tống…") ra "Liễu đại tiểu
thư" ở 9/32 lượt, "柳慕云" ra "chị dâu Đào Hoa,", "好主意啊" ra "Còn ngây ra đó làm gì," —
phần còn lại là hai câu gốc đồng nghĩa ("彻底好了" / "已经痊愈了" cùng ra "đã khỏi hẳn
rồi.") mà bản dịch giống nhau là ĐÚNG.

Chỉ nhìn hai dòng thì không biết dòng nào sai, cũng không biết có dòng nào sai không. Nên
module chỉ DÒ; khâu dọn cuối dịch lại riêng các dòng đó từ câu gốc (không kèm video — lời
lạc thường do model nghe tiếng trong video) rồi gọi :func:`retranslations_to_take`: chỉ
nhận bản dịch lại của dòng nào ra KHÁC HẲN bản cũ. Hai câu đồng nghĩa dịch lại vẫn ra như
cũ nên giữ nguyên bản đã trau chuốt. Mô phỏng (lấy bản thô của lượt khác làm "bản dịch
lại"): dòng sai được thay ở 77-100% số lần thử, dòng đúng của cặp 0-26%.

Toàn bộ là hàm thuần: không I/O.
"""

from __future__ import annotations

from subtitles_extractor.application.services.line_echo_guard import (
    source_tokens,
    translation_words,
)
from subtitles_extractor.application.services.stage_shift_guard import text_similarity

#: Bản dịch ngắn hơn ngần này từ ("Được rồi.", "Vào đi.") trùng nhau là chuyện thường.
DISTANT_MIN_WORDS = 3
#: Hai dòng liền kề thuộc về ``line_echo_guard``; ở đây xét từ cách một dòng trở lên.
DISTANT_MIN_GAP = 2
#: Xa hơn ngần này dòng thì không còn cùng lô dịch — trùng là do thoại lặp lại.
DISTANT_MAX_GAP = 60
#: Câu gốc ngắn hơn chia sẻ từ ngần này số chữ với câu kia thì bản dịch trùng là ĐÚNG
#: ("柳大小姐" / "那个柳大小姐").
DISTANT_MAX_SOURCE_OVERLAP = 0.5
#: Bản dịch lại giống bản cũ từ mức này trở lên thì coi như model xác nhận bản cũ.
RETRANSLATION_MAX_SIMILARITY = 0.5


def _same_source(first: str, second: str) -> bool:
    """Hai câu gốc có chung phần lớn nội dung (xét trên câu ngắn hơn)."""
    left, right = set(source_tokens(first)), set(source_tokens(second))
    shorter = min(len(left), len(right))
    if not shorter:
        return left == right
    return len(left & right) / shorter >= DISTANT_MAX_SOURCE_OVERLAP


def find_distant_duplicates(
    sources: dict[int, str], translations: dict[int, str]
) -> list[tuple[int, int]]:
    """Các cặp dòng không liền kề có CÙNG bản dịch dù câu gốc khác nhau (hàm thuần).

    Mỗi dòng chỉ ghép với lần xuất hiện KẾ TIẾP của cùng bản dịch, nên ba dòng trùng
    nhau cho hai cặp nối đuôi.

    Args:
        sources: ``số dòng → câu gốc``.
        translations: ``số dòng → bản dịch``.

    Returns:
        Các cặp ``(dòng trước, dòng sau)``, sắp theo dòng trước.
    """
    latest: dict[tuple[str, ...], int] = {}
    pairs: list[tuple[int, int]] = []
    for index in sorted(translations):
        words = tuple(translation_words(translations.get(index, "")))
        if len(words) < DISTANT_MIN_WORDS:
            continue
        earlier = latest.get(words)
        latest[words] = index
        if earlier is None:
            continue
        # Khoảng cách tính theo SỐ DÒNG phụ đề, để một tập dòng rời rạc (dịch lại vài
        # dòng được chọn) vẫn được xét như trên cả phim.
        if not DISTANT_MIN_GAP <= index - earlier <= DISTANT_MAX_GAP:
            continue
        if _same_source(sources.get(earlier, ""), sources.get(index, "")):
            continue
        pairs.append((earlier, index))
    return pairs


def retranslations_to_take(
    pairs: list[tuple[int, int]],
    translations: dict[int, str],
    retranslated: dict[int, str],
) -> dict[int, str]:
    """Bản dịch lại nên nhận cho các dòng của những cặp trùng xa (hàm thuần).

    Nhận dòng nào mà bản dịch lại KHÁC HẲN bản cũ (xem
    :data:`RETRANSLATION_MAX_SIMILARITY`): model chỉ nhìn câu gốc của dòng đó mà nói
    điều khác, tức bản cũ là lời của dòng khác. Dòng dịch lại ra gần như cũ được giữ
    nguyên bản đã qua tinh chỉnh.

    Args:
        pairs: Kết quả :func:`find_distant_duplicates`.
        translations: ``số dòng → bản dịch`` hiện tại.
        retranslated: ``số dòng → bản dịch lại`` (thiếu dòng nào thì bỏ qua dòng đó).

    Returns:
        ``số dòng → bản dịch mới`` cho các dòng nên thay; cặp mà sau khi thay vẫn trùng
        hệt nhau thì không thay dòng nào của cặp đó.
    """
    taken: dict[int, str] = {}
    for pair in pairs:
        changed = {
            index: retranslated[index].strip()
            for index in pair
            if retranslated.get(index, "").strip()
            and text_similarity(retranslated[index], translations.get(index, ""))
            < RETRANSLATION_MAX_SIMILARITY
        }
        first, second = (changed.get(i, translations.get(i, "")) for i in pair)
        if translation_words(first) == translation_words(second):
            continue
        taken.update(changed)
    return taken


__all__ = [
    "DISTANT_MAX_GAP",
    "DISTANT_MAX_SOURCE_OVERLAP",
    "DISTANT_MIN_GAP",
    "DISTANT_MIN_WORDS",
    "RETRANSLATION_MAX_SIMILARITY",
    "find_distant_duplicates",
    "retranslations_to_take",
]
