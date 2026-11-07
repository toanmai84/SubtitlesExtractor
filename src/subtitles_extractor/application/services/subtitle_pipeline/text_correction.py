"""Sửa lỗi text OCR cấp ký tự và khôi phục ký tự bị drop.

Bao gồm:
    * :func:`correct_hallucination_typos` — sửa lỗi ảo giác OCR phổ biến
      (mapping cố định, vd 涘→埃).
    * :func:`clean_edge_noise` — trim dấu câu rác ở 2 đầu chuỗi.
    * :func:`accumulate_confidence_bucket` — cộng dồn evidence theo
      ngưỡng high/medium cho các restorer.
    * :func:`apply_yi_suffix_restore` — phục hồi ký tự CJK bị OCR drop
      (đặc biệt là chữ '一' đầu/giữa câu và kết tử '了/吗/呢/吧' cuối câu).
    * :func:`restore_dropped_space` — phục hồi space bị drop trong cấu
      trúc `vai trò + lời nói` của phụ đề CJK.

Tất cả các hàm là **pure function** (không state) — dễ unit test.
"""

from __future__ import annotations

import contextlib
import contextvars
from collections.abc import Iterator

from loguru import logger

from subtitles_extractor.application.services.cjk_utils import is_predominantly_cjk
from subtitles_extractor.application.services.subtitle_pipeline.constants import (
    EDGE_NOISE_PUNCTUATION_REGEX,
    HAN_TRADITIONAL_TO_SIMPLIFIED,
    OCR_HALLUCINATION_TYPO_MAP,
)


def normalize_traditional_to_simplified(text: str) -> str:
    """Chuẩn hoá Hán Phồn → Hán Giản trong ``text``.

    Reference phụ đề Trung Quốc hiện đại thường dùng Hán giản; PaddleOCR
    đôi khi output phồn (vd "純陽" → "纯阳") gây mismatch. Hàm này chạy
    char-by-char qua bảng :data:`HAN_TRADITIONAL_TO_SIMPLIFIED` để
    chuẩn hoá toàn bộ ký tự phồn về giản.

    Args:
        text: Chuỗi gốc (có thể chứa cả phồn và giản).

    Returns:
        Chuỗi đã chuẩn hoá hoàn toàn về Hán giản. Các ký tự không nằm
        trong bảng mapping được giữ nguyên.
    """
    if not text:
        return text
    # Char-by-char translate là cách nhanh nhất cho mapping single char.
    # `str.translate` đòi hỏi ord(char) → str, nên ta dùng dict comprehension
    # 1 lần cache ở module-level cũng được, nhưng do chuỗi thường ngắn,
    # join generator đơn giản và đủ nhanh.
    return "".join(HAN_TRADITIONAL_TO_SIMPLIFIED.get(ch, ch) for ch in text)


def correct_hallucination_typos(source_text: str) -> str:
    """Sửa các lỗi ảo giác OCR phổ biến trước khi phân tích nội dung.

    Args:
        source_text: Chuỗi text cần sửa.

    Returns:
        Chuỗi đã sửa lỗi theo bảng :data:`OCR_HALLUCINATION_TYPO_MAP`.
    """
    for wrong_pattern, correct_pattern in OCR_HALLUCINATION_TYPO_MAP.items():
        if wrong_pattern in source_text:
            source_text = source_text.replace(wrong_pattern, correct_pattern)
    return restore_bound_words(source_text)


#: [v3.23.683] Từ hai chữ "dính liền" (连绵词): chữ được đánh dấu hầu như CHỈ xuất hiện trong từ đó, nên khi OCR đọc đúng
#: chữ ấy thì chữ kề bên buộc phải là chữ còn lại. Phim nối 86 tập: "觊觎我的女人" đọc thành "凯觎…" ở MỌI khung (tin cậy
#: 0,96, 2 câu) — bầu chọn giữa các khung không sửa được. Mỗi mục: (từ, chữ trái dính, chữ phải dính). Chỉ đưa vào chữ
#: thật sự dính (vd 吩 dính nhưng 咐 còn trong 嘱咐 → chỉ chữ trái).
BOUND_WORDS: tuple[tuple[str, bool, bool], ...] = (
    ("觊觎", True, True), ("踌躇", True, True), ("忐忑", True, True), ("尴尬", True, True),
    ("蜻蜓", True, True), ("蝙蝠", True, True), ("蟋蟀", True, True), ("蚯蚓", True, True),
    ("蜈蚣", True, True), ("蝌蚪", True, True), ("鹦鹉", True, True),
    ("琵琶", True, True), ("枇杷", True, True), ("琥珀", True, False), ("玳瑁", True, True),
    ("囫囵", True, True), ("踉跄", True, True), ("蹒跚", True, True), ("徘徊", True, True),
    ("窈窕", True, True), ("旖旎", True, True), ("婀娜", True, False), ("憧憬", True, True),
    ("缥缈", True, False), ("氤氲", True, True), ("匍匐", True, True), ("蹉跎", True, True),
    ("傀儡", True, True), ("魍魉", True, True), ("魑魅", True, False), ("饕餮", True, True),
    ("麒麟", True, False), ("蟾蜍", False, True), ("吩咐", True, False), ("彷徨", False, True),
    ("鸳鸯", True, True), ("玫瑰", True, False), ("葡萄", False, True),
    ("嘈杂", True, False), ("蹂躏", True, True), ("涟漪", False, True), ("龌龊", True, True),
)


def _bound_partners() -> tuple[dict[str, str], dict[str, str]]:
    right_of: dict[str, str] = {}
    left_of: dict[str, str] = {}
    for word, left_bound, right_bound in BOUND_WORDS:
        if left_bound:
            right_of[word[0]] = word[1]
        if right_bound:
            left_of[word[1]] = word[0]
    return right_of, left_of


_RIGHT_OF, _LEFT_OF = _bound_partners()
_BOUND_CHARS = frozenset(ch for word, _left, _right in BOUND_WORDS for ch in word)


def restore_bound_words(text: str) -> str:
    """[v3.23.683] Sửa chữ kề một chữ "dính liền" bị OCR đọc nhầm hình ("凯觎" → "觊觎").

    Chỉ thay khi chữ kề là chữ Hán KHÁC (không phải dấu câu / Latin / khoảng trắng — câu có thể bị ngắt dòng), bản
    thân nó không phải chữ dính của từ nào, và chữ còn lại của từ không đứng ở phía bên kia (đảo thứ tự "尬尴").
    """
    if not text or not any(ch in _BOUND_CHARS for ch in text):
        return text
    chars = list(text)
    for i, ch in enumerate(text):
        prev = text[i - 1] if i > 0 else ""
        nxt = text[i + 1] if i + 1 < len(text) else ""
        # Chữ kề được đọc từ câu GỐC (không từ bản đang sửa) để một lần sửa không kéo theo lần khác.
        partner = _RIGHT_OF.get(ch)
        if partner and nxt and prev != partner and nxt != partner and _replaceable(nxt):
            chars[i + 1] = partner
        partner = _LEFT_OF.get(ch)
        if partner and prev and nxt != partner and prev != partner and _replaceable(prev):
            chars[i - 1] = partner
    return "".join(chars)


def _replaceable(ch: str) -> bool:
    """Chữ kề được phép thay: chữ Hán, không phải chữ dính của từ nào (tránh hai luật giành nhau / đảo thứ tự)."""
    return is_single_cjk_char(ch) and ch not in _BOUND_CHARS


#: [v3.23.682] Phim coi là giản thể khi dòng có chữ phồn (trong bảng) ≤ tỉ lệ này so với dòng có chữ giản tương ứng.
MINORITY_TRADITIONAL_RATIO = 0.02
#: Cần ít nhất chừng này dòng giản thể mới đủ căn cứ.
MINORITY_TRADITIONAL_MIN_SIMPLIFIED_LINES = 20


def unify_minority_traditional(texts: list[str]) -> dict[int, str]:
    """[v3.23.682] Phim giản thể mà OCR đọc lẻ tẻ vài chữ ra dạng phồn → đổi về giản. Trả ``{vị trí: chữ mới}``.

    Phim nối 86 tập (3661 câu): 3 câu chứa chữ phồn ("最強的五人", "楚瑶解決了", "更何況") so với 1951 câu chứa chữ
    giản tương ứng; ảnh trong phim cả 3 đều là giản thể. Khác lần thử v3.7 (đổi vô điều kiện, hỏng phim dùng lẫn
    phồn/giản): chỉ đổi khi phồn thể là THIỂU SỐ rõ rệt của CHÍNH phim đó; phim phồn thể / dùng lẫn thì giữ nguyên.
    """
    traditional = {k for k, v in HAN_TRADITIONAL_TO_SIMPLIFIED.items() if k != v}
    simplified = {v for k, v in HAN_TRADITIONAL_TO_SIMPLIFIED.items() if k != v}
    trad_lines = [i for i, text in enumerate(texts) if any(ch in traditional for ch in text)]
    simp_lines = sum(1 for text in texts if any(ch in simplified for ch in text))
    if not trad_lines or simp_lines < MINORITY_TRADITIONAL_MIN_SIMPLIFIED_LINES:
        return {}
    if len(trad_lines) > MINORITY_TRADITIONAL_RATIO * simp_lines:
        return {}
    return {i: normalize_traditional_to_simplified(texts[i]) for i in trad_lines}


def clean_edge_noise(text_input: str) -> str:
    """Trim dấu câu rác ở 2 đầu chuỗi.

    [v3.7 NOTE] ĐÃ THỬ wire ``normalize_traditional_to_simplified`` vào đây để
    đồng nhất output về giản-thể. Tuy nhiên đo lường trên test4 cho thấy phụ đề
    "chuẩn" cũng là sản phẩm OCR và dùng LẪN phồn/giản (113 dòng chứa 煙/脈/強/
    輩...). Builder và chuẩn thường ra CÙNG dạng phồn cho cùng frame → normalize
    builder→giản phá vỡ các cặp vốn đã khớp ⇒ net-negative. Vì vậy KHÔNG auto
    normalize ở đây; hàm ``normalize_traditional_to_simplified`` vẫn export để
    dùng có chủ đích (vd video giản-thể thuần) nếu cần.

    Args:
        text_input: Chuỗi đầu vào (có thể chứa dấu câu rác ở biên).

    Returns:
        Chuỗi đã trim biên + strip whitespace.
    """
    return EDGE_NOISE_PUNCTUATION_REGEX.sub("", text_input).strip()


def _max_adjacent_repeat(text: str) -> int:
    """Số lần lặp liên tiếp lớn nhất của một ký tự (vd '一一' → 2, '来来来' → 3)."""
    if not text:
        return 0
    best = run = 1
    for idx in range(1, len(text)):
        run = run + 1 if text[idx] == text[idx - 1] else 1
        if run > best:
            best = run
    return best


def is_single_cjk_char(character: str) -> bool:
    """Kiểm tra ``character`` có phải đúng 1 ký tự CJK (Hán/Hiragana/Katakana).

    Args:
        character: Chuỗi cần kiểm tra.

    Returns:
        ``True`` nếu chuỗi dài 1 ký tự và thuộc dải Unicode CJK/Kana.
    """
    if len(character) != 1:
        return False
    code_point = ord(character)
    return (
        0x4E00 <= code_point <= 0x9FFF
        or 0x3400 <= code_point <= 0x4DBF
        or 0x3040 <= code_point <= 0x30FF
    )


def accumulate_confidence_bucket(
    existing_bucket: tuple[int, int] | None,
    new_confidence: float,
) -> tuple[int, int]:
    """Cộng dồn evidence theo ngưỡng confidence vào bucket (high, medium).

    Args:
        existing_bucket: Cặp ``(high_count, med_count)`` hiện tại, hoặc None.
        new_confidence: Confidence của candidate mới.

    Returns:
        Cặp ``(high_count, med_count)`` đã cộng dồn:
            * high: confidence >= 0.92.
            * medium: 0.85 <= confidence < 0.92.
            * (confidence < 0.85 không được tính.)
    """
    high_count, medium_count = existing_bucket if existing_bucket else (0, 0)
    if new_confidence >= 0.92:
        high_count += 1
    elif new_confidence >= 0.85:
        medium_count += 1
    return (high_count, medium_count)


_MIN_NON_YI_SUPPORT_RATIO = 0.2


def apply_yi_suffix_restore(
    voted_text: str,
    texts_with_confidences: list[tuple[str, float]],
    geometry: dict[str, tuple[float, float]] | None = None,
) -> str:
    """Phục hồi ký tự CJK bị OCR drop ở đầu/giữa/cuối câu.

    Phát hiện thực nghiệm trên fulltest (3603 câu): PaddleOCR rất hay drop:
        * Ký tự '一' bất kỳ vị trí nào (nét ngang nhỏ dễ nhầm nhiễu).
        * Kết tử cuối câu '了', '个', '吗', '呢', '吧' (font nhỏ ở cuối thường mờ).
        * Ký tự đầu câu bất kỳ CJK — đặc biệt '二', '不', '没', '个', ... (v2.9+).

    Ba pattern được nhận biết:

    * **prefix_prepend**: ``candidate.endswith(voted_text)`` → prepend 1 CJK char
      ở đầu. Bắt ``voted='连灵力'`` candidates ``'个连灵力'``,
      ``voted='师姐'`` candidates ``'二师姐'``, ``voted='愧是美人录里'``
      candidates ``'不愧是美人录里'``, v.v.
    * **suffix_append**: ``candidate.startswith(voted_text)`` → append 1 CJK char
      ở cuối. Bắt drop kết tử ``了/吗/呢/吧/儿`` cuối câu.
    * **yi_insert**: ``'一'`` được chèn vào VỊ TRÍ GIỮA (position > 0) của câu.
      Position 0 đã được xử lý bởi ``prefix_prepend`` nên không cần duplicate.

    Args:
        voted_text: Text đã được ROVER vote.
        texts_with_confidences: Toàn bộ candidate ``(text, conf)`` trong group.

    Returns:
        Text với ký tự được khôi phục nếu thỏa heuristic, ngược lại
        trả ``voted_text`` gốc.
    """
    expected_full_length = len(voted_text) + 1
    candidate_evidence: dict[tuple[str, str, int], tuple[int, int]] = {}

    for candidate_text, candidate_conf in texts_with_confidences:
        if not candidate_text or len(candidate_text) != expected_full_length:
            continue

        # ── Pattern 1: prefix_prepend — voted_text là SUFFIX của candidate ──
        # Bắt drop ký tự đầu câu bất kỳ CJK: '二师姐'→'师姐', '不愧'→'愧',
        # '个连灵力'→'连灵力', '一直盯着'→'直盯着' v.v.
        # Ưu tiên kiểm tra TRƯỚC yi_insert để tránh double-count cho '一' pos-0.
        if candidate_text.endswith(voted_text):
            prepended_char = candidate_text[0]
            if is_single_cjk_char(prepended_char):
                key = ("prefix_prepend", prepended_char, 0)
                candidate_evidence[key] = accumulate_confidence_bucket(
                    candidate_evidence.get(key), candidate_conf
                )
                continue

        # ── Pattern 2: suffix_append — voted_text là PREFIX của candidate ──
        # Bắt drop kết tử cuối câu: '了', '吗', '呢', '吧', '儿', '一'...
        if candidate_text.startswith(voted_text):
            appended_char = candidate_text[-1]
            if is_single_cjk_char(appended_char):
                key = ("suffix_append", appended_char, len(voted_text))
                candidate_evidence[key] = accumulate_confidence_bucket(
                    candidate_evidence.get(key), candidate_conf
                )
                continue

        # ── Pattern 3: yi_insert GIỮA câu (position > 0) ──
        # Chỉ xử lý '一' xuất hiện tại vị trí > 0.  Position-0 đã được
        # xử lý bởi prefix_prepend ở trên để tránh threshold double-count.
        if "一" in candidate_text:
            for yi_idx in range(1, len(candidate_text)):  # bắt đầu từ 1
                if candidate_text[yi_idx] != "一":
                    continue
                stripped = candidate_text[:yi_idx] + candidate_text[yi_idx + 1 :]
                if stripped == voted_text:
                    key = ("yi_insert", "一", yi_idx)
                    candidate_evidence[key] = accumulate_confidence_bucket(
                        candidate_evidence.get(key), candidate_conf
                    )
                    break

    if not candidate_evidence:
        return voted_text

    best_key, (best_high, best_med) = max(
        candidate_evidence.items(),
        key=lambda item: item[1][0] * 2 + item[1][1],
    )

    operation_type, inserted_char, position = best_key

    if operation_type == "prefix_prepend":
        # '一' đầu câu drop rất phổ biến (举两得→一举两得) → 1 high đủ.
        # Ký tự khác ('公','心','二','和'...) ở pos-0 thường là ARTIFACT rìa trái
        # (logo/watermark OCR) hơn là drop thật → cần evidence mạnh hơn (2 high
        # hoặc 4 med) để giảm chèn bậy. Đo trên test4 (cặp ground-truth đáng tin).
        if inserted_char == "一":
            is_strong_evidence = best_high >= 1 or best_med >= 2
        else:
            is_strong_evidence = best_high >= 2 or best_med >= 4
    elif operation_type == "yi_insert":
        # '一' giữa câu: cần evidence chắc hơn vì ít phổ biến hơn đầu câu.
        is_strong_evidence = best_high >= 1 or best_med >= 2
    elif operation_type == "suffix_append":
        # '一' cuối câu (vd '第一', '统一') — khá phổ biến.
        # Các kết tử khác ('了', '儿', ...) cần evidence mạnh hơn.
        if inserted_char == "一":
            is_strong_evidence = best_high >= 1 or best_med >= 2
        else:
            is_strong_evidence = best_high >= 2 or best_med >= 4
    else:
        is_strong_evidence = False

    if not is_strong_evidence:
        return voted_text

    # [v3.23.652] Ngưỡng tuyệt đối (2 high) quá lỏng với nhóm dài: '终于被我逮到了吧' (31 khung) có 4 khung đọc lẫn chữ
    # cảnh sau → '…吧天'. Chữ khác '一' phải xuất hiện ở ≥ 20% số khung của nhóm (drop thật '不愧…' 2/10, '…了' 8/24).
    # '一' giữ ngưỡng cũ: drop '一' thật vẫn hay chỉ 1/11–1/31 khung đọc được.
    if inserted_char != "一":
        support = best_high + best_med
        if support < _MIN_NON_YI_SUPPORT_RATIO * len(texts_with_confidences):
            return voted_text

    if operation_type == "suffix_append":
        restored = voted_text + inserted_char
    elif operation_type == "yi_insert":
        # Yi-insert có thể ở vị trí bất kỳ giữa câu.
        restored = voted_text[:position] + inserted_char + voted_text[position:]
    elif operation_type == "prefix_prepend":
        # Luôn prepend ở đầu câu.
        restored = inserted_char + voted_text
    else:
        return voted_text

    # [v3.23.657] B3: phụ đề căn giữa — ký tự thật làm hộp chữ CÂN quanh tâm; ký tự "khôi phục" từ vệt nhiễu cạnh chữ
    # làm hộp lệch tâm (ca thật tập 8: '一小小教训…' lệch 18 px, bản không '一' trùng tâm ROI). Bác khi bản có ký tự lệch
    # tâm hơn bản không có quá 1/4 bề rộng một chữ.
    if geometry and not _centering_supports(restored, voted_text, geometry):
        logger.debug("Char-Restorer: bác '{}' → '{}' (hộp chữ lệch tâm phụ đề).", voted_text, restored)
        return voted_text

    # [GUARD] Chặn over-correction tạo ký tự CJK lặp liền kề bất thường mà voted gốc
    # KHÔNG có. Ví dụ '不惜一切' + insert '一' → '不惜一一切' (cặp '一一' do OCR
    # double-detect rìa, không phải drop thật). Mẫu lặp 3 lần ('来来来') hợp lệ nên
    # chỉ chặn khi phép chèn LÀM TĂNG số ký tự lặp liền kề so với voted gốc.
    if _max_adjacent_repeat(restored) > _max_adjacent_repeat(voted_text):
        return voted_text

    logger.debug(
        "Char-Restorer: '{}' → '{}' (op={}, char={}, pos={}, high={}, med={})",
        voted_text,
        restored,
        operation_type,
        inserted_char,
        position,
        best_high,
        best_med,
    )
    return restored


def restore_dropped_space(
    voted_text: str,
    texts_with_confidences: list[tuple[str, float]],
) -> str:
    """Phục hồi space bị OCR drop trong câu phụ đề CJK.

    Pattern phổ biến: phụ đề CJK có cấu trúc ``'vai trò + lời nói'`` như
    ``'阿姨 来不及了'``, ``'妈 我先走了'``. PaddleOCR đôi khi detect space,
    đôi khi không. ROVER vote thường chọn version không space.

    Heuristic an toàn:
        * ``voted_text`` predominantly CJK, không có space sẵn.
        * ``voted_text`` >= 5 ký tự (tránh case ngắn 2-4 chars).
        * Trong group có candidate text bằng ``voted_text`` khi bỏ space.
        * Candidate có space đó có >= 2 frames conf >= 0.92 (high) HOẶC
          >= 4 frames conf >= 0.85 (med).

    Args:
        voted_text: Text đã được ROVER vote (thường không có space).
        texts_with_confidences: Toàn bộ candidate ``(text, conf)`` trong group.

    Returns:
        Text với space được restore tại các vị trí evidence-strong,
        ngược lại trả ``voted_text`` gốc.
    """
    if not voted_text or " " in voted_text:
        return voted_text
    if len(voted_text) < 5:
        return voted_text
    if not is_predominantly_cjk(voted_text):
        return voted_text

    space_positions_evidence: dict[tuple[int, ...], tuple[int, int]] = {}

    for candidate_text, candidate_conf in texts_with_confidences:
        if not candidate_text or " " not in candidate_text:
            continue
        candidate_stripped = candidate_text.replace(" ", "")
        if candidate_stripped != voted_text:
            continue
        space_positions: list[int] = []
        characters_seen = 0
        for char in candidate_text:
            if char == " ":
                space_positions.append(characters_seen)
            else:
                characters_seen += 1
        position_key = tuple(space_positions)
        if not position_key:
            continue
        space_positions_evidence[position_key] = accumulate_confidence_bucket(
            space_positions_evidence.get(position_key), candidate_conf
        )

    if not space_positions_evidence:
        return voted_text

    best_positions, (best_high, best_med) = max(
        space_positions_evidence.items(),
        key=lambda item: item[1][0] * 2 + item[1][1],
    )

    is_strong_evidence = best_high >= 2 or best_med >= 4
    if not is_strong_evidence:
        return voted_text

    restored_chars: list[str] = []
    for char_idx, char in enumerate(voted_text):
        if char_idx in best_positions:
            restored_chars.append(" ")
        restored_chars.append(char)
    restored_text = "".join(restored_chars)

    logger.debug(
        "Space-Restorer: '{}' → '{}' (positions={}, high={}, med={})",
        voted_text, restored_text, best_positions, best_high, best_med,
    )
    return restored_text


def restore_dropped_yi_prefix(
    voted_text: str,
    texts_with_confidences: list[tuple[str, float]],
    geometry: dict[str, tuple[float, float]] | None = None,
) -> str:
    """Orchestrator: chạy Yi-Restorer rồi Space-Restorer (đúng thứ tự).

    Args:
        voted_text: Text đã được ROVER vote.
        texts_with_confidences: Toàn bộ candidate trong group.

    Returns:
        Text đã được khôi phục ký tự + space (nếu có evidence).
    """
    if not voted_text or len(voted_text) < 2:
        return voted_text

    if not is_predominantly_cjk(voted_text):
        return voted_text

    yi_restored_text = apply_yi_suffix_restore(voted_text, texts_with_confidences, geometry)
    if yi_restored_text != voted_text:
        collector = _RESTORED_TEXTS.get()
        if collector is not None:
            collector.add(yi_restored_text)
            collector.add(clean_edge_noise(restore_dropped_space(yi_restored_text, texts_with_confidences)))
    return restore_dropped_space(yi_restored_text, texts_with_confidences)


#: [v3.23.657] Tâm phụ đề (toạ độ trong ROI) khi phụ đề CĂN GIỮA; ``None`` = không kiểm căn giữa.
_CENTER_X: contextvars.ContextVar[float | None] = contextvars.ContextVar("subtitle_center_x", default=None)
#: Bản có ký tự khôi phục được lệch tâm hơn bản không có tối đa chừng này bề rộng một chữ.
_MAX_EXTRA_OFFSET_CHARS = 0.25


def current_centering_reference() -> float | None:
    """[v3.23.657] Tâm phụ đề đang áp (để tầng bỏ phiếu biết có cần tính hình học hộp chữ không)."""
    return _CENTER_X.get()


@contextlib.contextmanager
def centering_reference(center_x: float | None) -> Iterator[None]:
    """[v3.23.657] Đặt tâm phụ đề căn giữa cho khối ``with`` (``None`` = tắt kiểm căn giữa)."""
    token = _CENTER_X.set(center_x)
    try:
        yield
    finally:
        _CENTER_X.reset(token)


def _centering_supports(restored: str, voted: str, geometry: dict[str, tuple[float, float]]) -> bool:
    center = _CENTER_X.get()
    with_char, without_char = geometry.get(restored), geometry.get(voted)
    if center is None or with_char is None or without_char is None or not voted:
        return True
    char_width = without_char[1] / max(1, len(voted))
    extra_offset = abs(with_char[0] - center) - abs(without_char[0] - center)
    return extra_offset <= _MAX_EXTRA_OFFSET_CHARS * char_width


#: [v3.23.655] Câu có ký tự được khôi phục trong lượt dựng phụ đề đang chạy (``None`` = không thu thập).
_RESTORED_TEXTS: contextvars.ContextVar[set[str] | None] = contextvars.ContextVar("restored_texts", default=None)


@contextlib.contextmanager
def collect_restored_texts() -> Iterator[set[str]]:
    """[v3.23.655] Thu các câu được Yi-Restorer thêm ký tự trong khối ``with`` (để đánh dấu "Nghi ngờ OCR")."""
    found: set[str] = set()
    token = _RESTORED_TEXTS.set(found)
    try:
        yield found
    finally:
        _RESTORED_TEXTS.reset(token)


__all__ = [
    "accumulate_confidence_bucket",
    "centering_reference",
    "collect_restored_texts",
    "current_centering_reference",
    "apply_yi_suffix_restore",
    "clean_edge_noise",
    "correct_hallucination_typos",
    "restore_bound_words",
    "is_single_cjk_char",
    "normalize_traditional_to_simplified",
    "unify_minority_traditional",
    "restore_dropped_space",
    "restore_dropped_yi_prefix",
]
