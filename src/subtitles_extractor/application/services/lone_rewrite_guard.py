"""[v3.23.592] Dòng LẺ bị giai đoạn tinh chỉnh viết lại khác hẳn bản trước.

Lưới dải viết lại (v3.23.589) chỉ bắt được từ ba dòng gần nhau trở lên. Chuẩn đoán thật
v3.23.591 (phim "灭门惨案II", nói tiếng Quảng Đông, phụ đề theo văn viết):

    dòng 68  不要生氣   bản thô "Đừng giận chứ."  →  bản địa hoá "Mau mở cửa!"

Hai dòng kề (67, 69) không đổi một chữ. "Mau mở cửa!" là lời đám đông NGHE được trong
video ở thời điểm đó — quy tắc "video chỉ là ngữ cảnh" thêm vào prompt ở v3.23.591 không
ngăn được model. Dòng này sai ở bản cuối của 9/17 lượt dịch phim này, trong khi bản thô
đúng ở cả 17 lượt.

Chỉ nhìn hai bản thì không biết bản nào đúng: nhiều khi chính bản trước sai và giai đoạn
tinh chỉnh đã sửa lại. Nên module chỉ DÒ ứng viên; khâu dịch gom mọi ứng viên của một
giai đoạn vào một lượt dịch trọng tài từ câu gốc (không kèm video), rồi
:func:`rewrites_to_revert` chỉ lùi dòng nào mà trọng tài đứng hẳn về phía bản trước.

Bốn chốt giữ cho việc lùi không gây hại, mỗi chốt có ca thật trong bộ test:

* Không đem phân xử dòng mà bản trước còn SÓT chữ gốc chưa dịch — bản mới ở đó là một
  lần sửa (6% số ứng viên của kho).
* Không đem phân xử dòng chỉ CHIA LẠI câu với dòng kề: câu cũ của nó đã sang dòng kề,
  hoặc câu mới của nó lấy từ dòng kề trong khi dòng kề cũng đổi. Lùi riêng một dòng của
  cặp đó thì một vế bị nói hai lần, vế kia biến mất. Hai câu coi là cùng một vế khi giống
  nhau từ 0,4 VÀ chung ít nhất hai từ (câu ngắn chung một từ đã đạt 0,4).
* Trọng tài phải giống bản trước hơn bản mới một khoảng rõ ràng; lưỡng lự thì giữ bản
  mới.
* Trọng tài phải thật sự ĐỒNG Ý với bản trước (giống từ 0,5): ở câu ngắn, chung một đại
  từ đã đủ tạo ra khoảng cách ở chốt trên.

Đo trên 56 lượt dịch thật của ba phim, 112 lần chuyển giai đoạn (lấy bản thô của một lượt
KHÁC làm trọng tài, các lượt còn lại làm chuẩn; lặp với bốn hạt giống ngẫu nhiên):

* 335-349 dòng bị viết hỏng, lùi được 241-260 (khoảng 73%);
* 381-397 dòng được viết lại ĐÚNG hơn, lùi nhầm 23-35 (khoảng 7%);
* 82-109 trên khoảng 520 dòng chỉ đổi cách nói bị đưa về câu chữ cũ (không mất nghĩa);
* xét cả cụm ba dòng quanh dòng bị lùi (để thấy câu bị cắt vế): 22-35 trên 355-400 lần
  lùi làm cụm đó xa các bản thô khác hơn. Trước khi có chốt chia lại câu là 54.

Tức mỗi lượt dịch bớt khoảng 4-5 dòng sai nghĩa, mất khoảng nửa dòng được sửa đúng. Trung
bình 11 ứng viên mỗi giai đoạn (nhiều nhất 61) — thường gói trong một request chữ.

Cách đo này còn bi quan ở một điểm: trọng tài mô phỏng là bản thô CÓ kèm video của lượt
khác nên mang cùng thói dịch theo lời nghe được (dòng 10 của lượt thật: hai trong chín bản
thô cùng sai "Cười chết mất" cho câu "搔你胳肢窩"); trọng tài thật không kèm video.

Toàn bộ là hàm thuần: không I/O.
"""

from __future__ import annotations

from subtitles_extractor.application.services.cjk_utils import contains_cjk
from subtitles_extractor.application.services.stage_shift_guard import (
    REWRITE_MAX_SIMILARITY,
    shared_word_count,
    text_similarity,
)

#: Hai câu giống nhau từ mức này trở lên thì coi là CÙNG một vế câu đã chuyển chỗ giữa hai
#: dòng kề (giai đoạn tinh chỉnh chia lại câu) — không phải viết lại.
MOVED_MIN_SIMILARITY = 0.4
#: … và phải chung ít nhất ngần này TỪ: câu ngắn chỉ chung một từ ("Làm gì thế?" và
#: "Thế ạ?") chưa phải là cùng một vế câu.
MOVED_MIN_SHARED_WORDS = 2
#: Trọng tài phải giống bản trước hơn bản mới ít nhất ngần này mới lùi.
REFEREE_MIN_MARGIN = 0.3
#: … và phải thật sự ĐỒNG Ý với bản trước, không chỉ "ít khác hơn": ở câu ngắn, chung một
#: đại từ đã đủ tạo khoảng cách trên ("Chị yên tâm," với "Chị xem," là 0,4).
REFEREE_MIN_AGREEMENT = 0.5
#: Trần số dòng đem phân xử mỗi giai đoạn — quá ngần này là cả giai đoạn có vấn đề, không
#: phải vài dòng lẻ.
MAX_LONE_REWRITES_PER_STAGE = 60


def _rewritten(before: str, after: str) -> bool:
    """Bản mới khác HẲN bản trước (cả hai đều có chữ)."""
    if not before.strip() or not after.strip() or before.strip() == after.strip():
        return False
    return text_similarity(before, after) < REWRITE_MAX_SIMILARITY


def _finished_leftover(before: str, after: str) -> bool:
    """Bản trước còn sót chữ gốc chưa dịch, bản mới đã dịch nốt — một lần SỬA."""
    return contains_cjk(before) and not contains_cjk(after)


def _same_clause(first: str, second: str) -> bool:
    """Hai câu mang CÙNG một vế câu (giống đủ nhiều và chung đủ số từ)."""
    return (
        text_similarity(first, second) >= MOVED_MIN_SIMILARITY
        and shared_word_count(first, second) >= MOVED_MIN_SHARED_WORDS
    )


def _resplit_with_neighbour(
    index: int, near: int, previous: dict[int, str], current: dict[int, str]
) -> bool:
    """Dòng ``index`` và dòng kề ``near`` chỉ CHIA LẠI một câu cho nhau (hàm thuần).

    Hai chiều: câu cũ của dòng đã sang dòng kề (lùi dòng này thì vế đó bị nói hai lần);
    hoặc câu mới của dòng lấy từ câu cũ của dòng kề trong khi dòng kề cũng đổi (lùi dòng
    này thì vế đó biến mất). Dòng kề KHÔNG đổi mà dòng này chép lại câu của nó thì là viết
    hỏng thật — vẫn đem phân xử.
    """
    near_before, near_after = previous.get(near, ""), current.get(near, "")
    if _same_clause(previous.get(index, ""), near_after):
        return True
    return near_before.strip() != near_after.strip() and _same_clause(
        current[index], near_before
    )


def find_lone_rewrites(previous: dict[int, str], current: dict[int, str]) -> list[int]:
    """Các dòng bị viết lại khác hẳn mà KHÔNG phải chia lại câu với dòng kề (hàm thuần).

    Dòng kề tính theo THỨ TỰ các dòng của ``current`` (dòng đầu/cuối chỉ có một dòng kề).
    Bỏ qua dòng mà bản trước còn sót chữ gốc chưa dịch: bản mới ở đó là một lần sửa.

    Args:
        previous: ``số dòng → bản trước`` (đầu vào của giai đoạn).
        current: ``số dòng → bản mới`` (đầu ra của giai đoạn).

    Returns:
        Số dòng các ứng viên, tăng dần.
    """
    ordered = sorted(current)
    candidates: list[int] = []
    for position, index in enumerate(ordered):
        before = previous.get(index, "")
        if not _rewritten(before, current[index]):
            continue
        if _finished_leftover(before, current[index]):
            continue
        neighbours = [
            ordered[near]
            for near in (position - 1, position + 1)
            if 0 <= near < len(ordered)
        ]
        if not any(
            _resplit_with_neighbour(index, near, previous, current)
            for near in neighbours
        ):
            candidates.append(index)
    return candidates


def rewrites_to_revert(
    candidates: list[int],
    previous: dict[int, str],
    current: dict[int, str],
    reference: dict[int, str],
) -> list[int]:
    """Ứng viên nào nên lùi về bản trước theo bản dịch trọng tài (hàm thuần).

    Lùi khi trọng tài giống bản trước từ :data:`REFEREE_MIN_AGREEMENT` trở lên VÀ giống
    bản trước hơn bản mới từ :data:`REFEREE_MIN_MARGIN` trở lên. Thiếu bản trọng tài cho
    dòng nào thì giữ bản mới của dòng đó.

    Args:
        candidates: Kết quả :func:`find_lone_rewrites`.
        previous: ``số dòng → bản trước``.
        current: ``số dòng → bản mới``.
        reference: ``số dòng → bản dịch trọng tài`` từ câu gốc.

    Returns:
        Số dòng nên lùi, theo thứ tự của ``candidates``.
    """
    reverted: list[int] = []
    for index in candidates:
        referee = reference.get(index, "")
        if not referee.strip():
            continue
        toward_previous = text_similarity(referee, previous.get(index, ""))
        toward_current = text_similarity(referee, current.get(index, ""))
        if (
            toward_previous >= REFEREE_MIN_AGREEMENT
            and toward_previous - toward_current >= REFEREE_MIN_MARGIN
        ):
            reverted.append(index)
    return reverted


def reference_lines_needed(candidates: list[int], ordered: list[int]) -> list[int]:
    """Các dòng cần gửi trọng tài: ứng viên kèm một dòng kề mỗi bên làm ngữ cảnh.

    Args:
        candidates: Số dòng ứng viên.
        ordered: Mọi số dòng của bản dịch, tăng dần.

    Returns:
        Số dòng cần dịch trọng tài, tăng dần, không trùng.
    """
    position = {index: at for at, index in enumerate(ordered)}
    wanted: set[int] = set()
    for index in candidates:
        at = position.get(index)
        if at is None:
            continue
        for near in (at - 1, at, at + 1):
            if 0 <= near < len(ordered):
                wanted.add(ordered[near])
    return sorted(wanted)


__all__ = [
    "MAX_LONE_REWRITES_PER_STAGE",
    "MOVED_MIN_SHARED_WORDS",
    "MOVED_MIN_SIMILARITY",
    "REFEREE_MIN_AGREEMENT",
    "REFEREE_MIN_MARGIN",
    "find_lone_rewrites",
    "reference_lines_needed",
    "rewrites_to_revert",
]
