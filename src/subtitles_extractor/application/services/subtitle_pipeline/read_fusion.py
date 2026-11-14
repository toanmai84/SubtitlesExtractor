"""[v3.23.711] Gộp các LẦN ĐỌC của cùng một câu trước khi dựng phụ đề.

Từ 3.23.710 mỗi câu được OCR vài lần (khung xác nhận + đọc lại định kỳ). Các lần đọc không phải lúc nào cũng giống
nhau, và khâu dựng từng coi mỗi lần đọc khác đi là một câu khác. Lượt chạy thật phim nối 86 tập bằng bản 3.23.710
(3663 câu đáp án, 3660 khớp) còn đúng các lỗi kiểu này:

* lần đọc lại trả RỖNG dù chữ còn trên màn hình → câu bị cắt đuôi ("你" kết thúc sớm 0,68 s, "哎" sớm 0,36 s), mất
  đầu ("啊" trễ 0,52 s) hoặc mất hẳn ("你" 1379 s: ba lần đọc là "你", rỗng, "你");
* lần đọc lại rơi chữ "一" → "一晚上" thành ba câu "一晚上" / "晚上" / "一晚上";
* hai lần đọc khác nhau một chữ gần giống hệt ("白" / "自") → một câu thành hai.

Hai bước, đều là hàm thuần trên danh sách khung:

1. :func:`fuse_same_pixel_reads` — bộ lọc khung đã so điểm ảnh và ghi ``continues_previous`` cho khung có CÙNG điểm
   ảnh chữ với khung trước. Một chuỗi khung như vậy là một câu duy nhất trên màn hình: lần đọc rỗng nhận chữ của
   lần đọc có chữ, lần đọc thiếu "一" nhận bản đủ "一".
2. :func:`unify_confusable_variants` — hai đoạn liền nhau (không khung trống xen giữa) chỉ hơn kém chữ "一", hoặc
   cùng số chữ và chỉ khác nhau ở các cặp chữ dễ lẫn về hình (:data:`CONFUSABLE_PAIRS`), là một câu; bản được giữ
   chọn theo ngữ liệu của chính phim và được báo lại để gắn cờ "Nghi ngờ OCR".

Dữ liệu OCR thô của bản cũ không có ``continues_previous`` nên bước 1 không làm gì (dựng như trước).
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Sequence

from subtitles_extractor.domain.entities.ocr_frame_result import OcrFrameResult, OcrTextBox

__all__ = [
    "CONFUSABLE_PAIRS", "FILL_MIN_CONFIDENCE", "fresh_read_flags", "fuse_same_pixel_reads", "measure_char_pitch", "measure_text_height",
    "tighten_inflated_boxes",
    "snap_inflated_boxes_to_band",
    "measure_text_band", "unify_confusable_variants",
]

#: Lần đọc từ mức tin cậy này mới được tin; lần đọc rỗng / dưới mức nhận chữ của nó (rác không bị nhân ra cả chuỗi).
FILL_MIN_CONFIDENCE: float = 0.80

#: Hai bản đọc hơn kém chữ "一": chênh lệch độ lệch tâm từ mức này (theo chiều cao chữ) mới dùng để phân xử.
YI_CENTER_MARGIN_HEIGHTS: float = 0.12

#: [v3.23.712] Phần bề rộng hộp không tỉ lệ với số chữ (lề hai đầu), tính theo chữ: bề rộng ≈ bước chữ × (số chữ + 0,35).
BOX_MARGIN_CHARS: float = 0.35
#: Hai bản đọc hơn kém chữ "一": tổng sai lệch hình học (lệch tâm + lệch bề rộng, theo chữ) phải chênh từ mức này.
YI_GEOMETRY_MARGIN_CHARS: float = 0.25
#: Trong một chuỗi cùng điểm ảnh, hai bản đọc cùng số chữ khác nhau tối đa ngần này vị trí là cùng một câu.
CHAIN_MAX_SUBSTITUTIONS: int = 2
#: Chênh lệch độ tin cậy từ mức này mới dùng để chọn giữa hai bản đọc lẫn chữ (dưới mức đó: bản nhiều khung hơn).
SUBSTITUTION_CONFIDENCE_MARGIN: float = 0.02
#: Hộp chữ của một lần đọc cao hơn ngần này lần hộp gọn nhất của cùng câu trong chuỗi → hộp phình (dính nền sáng).
INFLATED_BOX_RATIO: float = 1.25

#: [v3.23.712] Lần đọc rỗng / hỏng chỉ nhận chữ của lần đọc có hộp nằm đúng chỗ, đúng cỡ phụ đề; không biết hình học
#: của phim thì bản thắng phải có ít nhất chừng này khung. Rác tin cậy cao ("à", "4", "R" — vật trong cảnh) đọc được
#: một lần thì không bị nhân ra cả chuỗi khung cùng điểm ảnh.
SUBTITLE_CENTER_TOLERANCE_HEIGHTS: float = 0.50
SUBTITLE_HEIGHT_RANGE: tuple[float, float] = (0.60, 1.50)
SPREAD_MIN_FRAMES_WITHOUT_GEOMETRY: int = 3

#: Cặp chữ Hán gần giống hệt về hình (khác một nét) mà mô hình nhận dạng hay đọc lẫn trên cùng một ảnh.
CONFUSABLE_PAIRS: frozenset[frozenset[str]] = frozenset(
    frozenset(pair) for pair in (
        "白自", "己已", "已巳", "己巳", "未末", "土士", "人入", "日曰", "干千", "于干", "天夫", "大太", "太犬", "大犬",
        "王玉", "王主", "玉主", "刀力", "问间", "侯候", "徒徙", "戊戌", "戌戍", "戊戍", "鸟乌", "免兔", "拔拨", "折拆",
        "贝见", "治冶", "今令", "师帅", "历厉", "体休", "竟竞", "析折", "未朱", "田由", "甲申", "目自", "日目",
    )
)


def _frame_text(frame: OcrFrameResult) -> str:
    return "".join((box.text or "").strip() for box in frame.text_boxes)


def _frame_confidence(frame: OcrFrameResult) -> float:
    boxes = frame.text_boxes
    return sum(float(box.confidence) for box in boxes) / len(boxes) if boxes else 0.0


def _read_signature(frame: OcrFrameResult) -> tuple[tuple[str, float, tuple[tuple[float, ...], ...]], ...]:
    return tuple(
        (box.text or "", float(box.confidence), tuple(tuple(map(float, point)) for point in (box.polygon or ())))
        for box in frame.text_boxes
    )


def fresh_read_flags(frames: Sequence[OcrFrameResult]) -> list[bool]:
    """[v3.23.726] ``True`` = khung là một LẦN ĐỌC OCR thật; ``False`` = khung chép kết quả của khung ngay trước.

    Khung trùng điểm ảnh (``continues_previous``) không được OCR lại mà chép nguyên hộp chữ của khung tham chiếu; chỉ
    các khung đọc lại theo lịch (khoảng 0,44 s một lần) mới là lần đọc mới. Đếm theo khung vì thế nhân một lần đọc
    lên theo thời gian ảnh đứng yên: lượt chạy thật 3.23.725 (phim nối 86 tập) đọc "做这等腌臜之事" 3 lần và
    "腌攒" 2 lần, nhưng hai lần đọc "腌攒" rơi đầu hai đợt chép 11 khung → 21 khung "攒" / 15 khung "臜" → câu sai;
    "哼" 2 lần / "亨" 1 lần nhưng 11 khung "亨" / 4 khung "哼" → câu sai. Khung là bản chép khi mang
    ``continues_previous`` VÀ hộp chữ trùng khít khung ngay trước (chữ, độ tin cậy, toạ độ). Dữ liệu OCR thô bản cũ không
    có ``continues_previous`` → mọi khung là lần đọc (như trước).
    """
    flags: list[bool] = []
    previous: tuple[tuple[str, float, tuple[tuple[float, ...], ...]], ...] | None = None
    for frame in frames:
        signature = _read_signature(frame)
        flags.append(not (frame.continues_previous and previous is not None and signature == previous))
        previous = signature
    return flags


def _box_center(frame: OcrFrameResult) -> float | None:
    xs = [point[0] for box in frame.text_boxes for point in (box.polygon or ())]
    return (min(xs) + max(xs)) / 2.0 if xs else None


def _box_height(frame: OcrFrameResult) -> float:
    ys = [point[1] for box in frame.text_boxes for point in (box.polygon or ())]
    return float(max(ys) - min(ys)) if ys else 0.0


def _box_width(frame: OcrFrameResult) -> float:
    xs = [point[0] for box in frame.text_boxes for point in (box.polygon or ())]
    return float(max(xs) - min(xs)) if xs else 0.0


def measure_char_pitch(frames: Sequence[OcrFrameResult]) -> float | None:
    """[v3.23.712] Bước chữ của phụ đề (px mỗi chữ), đo từ các khung một hộp ≥ 4 chữ tin cậy ≥ 0,90; thiếu dữ liệu → None."""
    ratios = sorted(
        _box_width(frame) / (len(_frame_text(frame)) + BOX_MARGIN_CHARS)
        for frame in frames
        if len(frame.text_boxes) == 1 and len(_frame_text(frame)) >= 4 and _frame_confidence(frame) >= 0.90
        and _box_width(frame) > 0
    )
    if len(ratios) < 50:
        return None
    return ratios[len(ratios) // 2]


def measure_text_height(frames: Sequence[OcrFrameResult]) -> float | None:
    """[v3.23.712] Chiều cao hộp chữ điển hình của phụ đề (khung một hộp ≥ 3 chữ tin cậy ≥ 0,90); thiếu dữ liệu → None."""
    heights = sorted(
        _box_height(frame) for frame in frames
        if len(frame.text_boxes) == 1 and len(_frame_text(frame)) >= 3 and _frame_confidence(frame) >= 0.90
    )
    if len(heights) < 50:
        return None
    return heights[len(heights) // 2] or None


def _looks_like_subtitle(frame: OcrFrameResult, center_x: float | None, text_height: float | None) -> bool:
    """Hộp chữ nằm đúng chỗ và đúng cỡ phụ đề của phim (tâm lệch ≤ 0,5 lần chiều cao chữ, cao 0,6–1,5 lần)."""
    center = _box_center(frame)
    if center_x is None or not text_height or center is None:
        return False
    low, high = SUBTITLE_HEIGHT_RANGE
    return (
        abs(center - center_x) <= SUBTITLE_CENTER_TOLERANCE_HEIGHTS * text_height
        and low * text_height <= _box_height(frame) <= high * text_height
    )


def _pick_yi_variant(
    longer: Sequence[OcrFrameResult], shorter: Sequence[OcrFrameResult], center_x: float | None,
    char_pitch: float | None = None,
) -> bool:
    """``True`` nếu bản ĐỦ chữ "一" là bản đúng (``longer`` / ``shorter``: các khung của từng bản đọc).

    Phân xử bằng hình học hộp chữ của phụ đề căn giữa, cộng hai sai lệch (đơn vị: chữ), lấy khung tốt nhất mỗi bản:

    * LỆCH TÂM — rơi "一" đầu câu làm hộp lệch nửa chữ ("晚上" lệch 26 px, "一晚上" 0 px); đọc THỪA "一" do dính vật
      trong cảnh làm hộp phình một bên (tập 08: "一小小教训你们一下吧" 2/32 khung, lệch 18 px);
    * LỆCH BỀ RỘNG so với số chữ đọc được (cần ``char_pitch``) — rơi "一" GIỮA câu không làm lệch tâm: "其实我" +
      "直都在" (hai hộp, tổng rộng 419 px = 7 chữ) đọc ra 6 chữ, trong khi "其实我一直都在" khớp đủ 7.

    Hai bản sai lệch gần như nhau (hoặc không biết tâm) thì bản có nhiều khung hơn; hoà thì bản đủ chữ.
    """
    if center_x is not None:
        unit = char_pitch if char_pitch else max((_box_height(frame) for frame in (*longer, *shorter)), default=0.0)

        def mismatch(frames: Sequence[OcrFrameResult]) -> float | None:
            best: float | None = None
            for frame in frames:
                center = _box_center(frame)
                if center is None or unit <= 0:
                    continue
                error = abs(center - center_x) / unit
                if char_pitch:
                    error += abs(_box_width(frame) / char_pitch - BOX_MARGIN_CHARS - len(_frame_text(frame)))
                best = error if best is None else min(best, error)
            return best

        long_error, short_error = mismatch(longer), mismatch(shorter)
        if long_error is not None and short_error is not None:
            margin = YI_GEOMETRY_MARGIN_CHARS if char_pitch else YI_CENTER_MARGIN_HEIGHTS
            if abs(long_error - short_error) >= margin:
                return long_error < short_error
    return len(longer) >= len(shorter)


def _only_yi_inserted(longer: str, shorter: str) -> bool:
    """``longer`` = ``shorter`` chèn thêm toàn chữ "一"."""
    if len(longer) <= len(shorter):
        return False
    index = 0
    for char in longer:
        if index < len(shorter) and char == shorter[index]:
            index += 1
        elif char != "一":
            return False
    return index == len(shorter)


def fuse_same_pixel_reads(
    frames: Sequence[OcrFrameResult], center_x: float | None = None, char_pitch: float | None = None,
    text_height: float | None = None,
) -> tuple[list[OcrFrameResult], int, set[str]]:
    """Gộp các lần đọc trong từng chuỗi khung cùng điểm ảnh chữ.

    Trả ``(khung, số khung được sửa, các câu được chọn giữa hai bản đọc lẫn chữ)``. Trong một chuỗi (một hình):

    * khung RỖNG hoặc đọc với độ tin cậy dưới :data:`FILL_MIN_CONFIDENCE` nhận hộp chữ của lần đọc tốt nhất — khi
      lần đọc đó nằm đúng chỗ, đúng cỡ phụ đề của phim (``center_x``, ``text_height``);
    * các bản đọc chỉ hơn kém nhau chữ "一" → bản đúng theo hình học (:func:`_pick_yi_variant`);
    * [v3.23.712] các bản đọc CÙNG SỐ CHỮ khác nhau ≤ :data:`CHAIN_MAX_SUBSTITUTIONS` vị trí và XEN KẼ nhau trên cùng
      điểm ảnh ("癞蛤蟆" → "痴蛤蟆" → "癞蛤蟆", "吗" / "鸣", "狠狠" / "狠狼") → một câu (:func:`_pick_substitution`);
    * [v3.23.712] cùng chữ nhưng hộp chữ PHÌNH (cao > :data:`INFLATED_BOX_RATIO` lần hộp gọn nhất, do dính nền
      sáng) nhận hộp gọn — hộp phình từng bị bộ lọc hình dạng loại, làm mất câu "打" và cắt mốc "不", "爽".

    Chuỗi có các bản đọc tin cậy cao khác nhau kiểu khác thì để nguyên.
    """
    result = list(frames)
    fixed = 0
    conflicts: set[str] = set()
    corpus: list[str] | None = None
    start = 0
    count = len(result)
    while start < count:
        end = start + 1
        while end < count and result[end].continues_previous:
            end += 1
        if end - start >= 2:
            texts = {_frame_text(frame) for frame in result[start:end]}
            if len(texts) > 2 and corpus is None:       # chỉ dựng ngữ liệu khi có chuỗi cần phân xử
                corpus = sorted({text for text in map(_frame_text, result) if len(text) >= 2})
            fixed += _fuse_chain(
                result, start, end, center_x, char_pitch, text_height, corpus or [], conflicts,
            )
        start = end
    return result, fixed, conflicts


def _substitution_count(left: str, right: str) -> int | None:
    """Số vị trí khác nhau giữa hai chuỗi CÙNG độ dài; khác độ dài → ``None``."""
    if len(left) != len(right):
        return None
    return sum(1 for a, b in zip(left, right, strict=True) if a != b)


def _pick_substitution(
    left: str, right: str, left_frames: Sequence[OcrFrameResult], right_frames: Sequence[OcrFrameResult],
    corpus: Sequence[str], left_reads: int | None = None, right_reads: int | None = None,
) -> str:
    """Chọn giữa hai bản đọc cùng số chữ của CÙNG một hình (``left`` là bản đang giữ).

    Thứ tự: (1) ngữ liệu của chính phim (:func:`_context_evidence`); (2) độ tin cậy cao nhất của mỗi bản khi chênh
    ≥ :data:`SUBSTITUTION_CONFIDENCE_MARGIN` — đo thật: "吗" 1,00 / "鸣" 0,96; "狠" 1,00 / "狼" 0,97; "癞" 0,99 /
    "痴" 0,93; (3) bản có nhiều LẦN ĐỌC hơn (:func:`fresh_read_flags` — [v3.23.726] trước đây đếm khung, tính cả khung
    chép); (4) bản đang giữ.
    """
    left_evidence, right_evidence = _context_evidence(left, right, corpus), _context_evidence(right, left, corpus)
    if left_evidence != right_evidence:
        return left if left_evidence > right_evidence else right
    left_conf = max(map(_frame_confidence, left_frames), default=0.0)
    right_conf = max(map(_frame_confidence, right_frames), default=0.0)
    if abs(left_conf - right_conf) >= SUBSTITUTION_CONFIDENCE_MARGIN:
        return left if left_conf > right_conf else right
    left_count = len(left_frames) if left_reads is None else left_reads
    right_count = len(right_frames) if right_reads is None else right_reads
    return right if right_count > left_count else left


def _fuse_chain(
    frames: list[OcrFrameResult], start: int, end: int, center_x: float | None, char_pitch: float | None,
    text_height: float | None, corpus: Sequence[str], conflicts: set[str],
) -> int:
    by_text: dict[str, list[OcrFrameResult]] = {}
    reads: dict[str, int] = {}         # [v3.23.726] số LẦN ĐỌC thật của mỗi bản (không tính khung chép)
    runs: list[str] = []               # dãy bản đọc tin cậy theo thời gian, đã nén các khung liền nhau cùng chữ
    fresh = fresh_read_flags(frames[start:end])
    for offset, frame in enumerate(frames[start:end]):
        text = _frame_text(frame)
        if text and _frame_confidence(frame) >= FILL_MIN_CONFIDENCE and (not runs or runs[-1] != text):
            runs.append(text)
        # Lần đọc rỗng hoặc tin cậy thấp là lần đọc HỎNG của cùng hình (tập 04: "啊" 1,00 → "国sd" 0,51 → "啊" 1,00
        # trên cùng điểm ảnh — câu từng bị cắt mất 0,9 s đầu): không tham gia chọn chữ, và nhận chữ của lần đọc tốt.
        if text and _frame_confidence(frame) >= FILL_MIN_CONFIDENCE:
            by_text.setdefault(text, []).append(frame)
            reads[text] = reads.get(text, 0) + int(fresh[offset])
    if not by_text:
        return _unify_weak_alternation(frames, start, end, fresh, center_x, text_height, corpus, conflicts)
    ordered = sorted(by_text, key=lambda text: (reads[text], len(by_text[text])), reverse=True)
    winner = ordered[0]
    disputed = False
    for text in ordered[1:]:
        differing = _substitution_count(winner, text)
        if differing is not None and differing <= CHAIN_MAX_SUBSTITUTIONS:
            # Chỉ khi các bản đọc XEN KẼ (A → B → A): chữ quay lại bản cũ chứng tỏ hình không đổi. Một chiều A → B có
            # thể là hai câu thật khác nhau một chữ mà bộ lọc khung không thấy đổi ("任务一" → "任务二") → để nguyên.
            if len(runs) == len(set(runs)):
                # [v3.23.713] Không gộp, nhưng hai bản đọc trên CÙNG điểm ảnh vẫn là tranh chấp: gắn cờ cả hai để vòng
                # soát đọc lại nhiều khung phân xử (tập 47: "腌攒" ×3 lần đọc rồi "腌臜" ×1, một chiều → trước đây
                # câu sai lọt không cờ). Câu thật khác một chữ thì đọc lại sẽ xác nhận nguyên văn — không mất gì.
                conflicts.update((winner, text))
                return 0
            winner = _pick_substitution(
                winner, text, by_text[winner], by_text[text], corpus, reads[winner], reads[text],
            )
            disputed = True
        elif center_x is not None and _only_yi_inserted(winner, text):
            if not _pick_yi_variant(by_text[winner], by_text[text], center_x, char_pitch):
                winner = text
        elif center_x is not None and _only_yi_inserted(text, winner):
            if _pick_yi_variant(by_text[text], by_text[winner], center_x, char_pitch):
                winner = text
        else:
            return 0                   # bản đọc khác nhau kiểu khác / không biết tâm: không đủ chắc để gộp ở mức khung
    candidates = by_text[winner]
    top_confidence = max(map(_frame_confidence, candidates))
    # Hộp GỌN nhất trong các lần đọc tin cậy cao nhất của bản thắng: hộp phình là hộp dính thêm nền sáng.
    best = min(
        (frame for frame in candidates if _frame_confidence(frame) >= top_confidence - 0.03),
        key=lambda frame: (_box_height(frame), -_frame_confidence(frame)),
    )
    best_height = _box_height(best)
    # Lần đọc rỗng / hỏng chỉ nhận chữ khi bản thắng trông đúng là phụ đề (hình học), hoặc đã đọc được đủ nhiều khung.
    if center_x is not None and text_height:
        can_spread = _looks_like_subtitle(best, center_x, text_height)
    else:
        can_spread = len(candidates) >= SPREAD_MIN_FRAMES_WITHOUT_GEOMETRY
    fixed = 0
    for index in range(start, end):
        frame = frames[index]
        strong = bool(_frame_text(frame)) and _frame_confidence(frame) >= FILL_MIN_CONFIDENCE
        if (
            strong and _frame_text(frame) == winner
            and (best_height <= 0 or _box_height(frame) <= INFLATED_BOX_RATIO * best_height)
        ):
            continue
        if not strong and not can_spread:
            continue
        frames[index] = dataclasses.replace(frame, text_boxes=list(best.text_boxes))
        fixed += 1
    if disputed and fixed:
        conflicts.add(winner)
    return fixed


#: [v3.23.726] Lần đọc yếu (dưới :data:`FILL_MIN_CONFIDENCE`) từ mức này mới được xét khi chuỗi không có lần đọc mạnh.
WEAK_READ_MIN_CONFIDENCE: float = 0.40


def _unify_weak_alternation(
    frames: list[OcrFrameResult], start: int, end: int, fresh: Sequence[bool], center_x: float | None,
    text_height: float | None, corpus: Sequence[str], conflicts: set[str],
) -> int:
    """[v3.23.726] Chuỗi cùng điểm ảnh CHỈ có lần đọc yếu, hai bản đọc cùng số chữ XEN KẼ nhau (A → B → A) → một câu.

    Lượt chạy thật 3.23.725: "哼" 0,57 → "亨" 0,51 → "哼" 0,63 trên cùng điểm ảnh; cả ba dưới 0,80 nên bước gộp bỏ qua,
    hai đoạn "哼" chỉ 2 khung bị loại vì ngắn, còn lại câu "亨" (đáp án "哼"). Chữ quay lại bản cũ trên cùng một hình
    chứng tỏ đây là một câu; bản giữ chọn theo ngữ liệu phim → số LẦN ĐỌC thật → độ tin cậy cao nhất. Chỉ đổi chữ của
    các khung có chữ (lần đọc yếu không lan sang khung rỗng), chỉ khi hộp chữ đúng chỗ / đúng cỡ phụ đề; câu được gắn
    cờ nghi ngờ để soát lại.
    """
    texts: dict[str, list[int]] = {}
    runs: list[str] = []
    for index in range(start, end):
        frame = frames[index]
        text = _frame_text(frame)
        if not text:
            continue
        if _frame_confidence(frame) < WEAK_READ_MIN_CONFIDENCE:
            return 0
        texts.setdefault(text, []).append(index)
        if not runs or runs[-1] != text:
            runs.append(text)
    if len(texts) != 2 or len(runs) == len(set(runs)):
        return 0                                   # không phải hai bản đọc xen kẽ: để nguyên
    left, right = runs[0], next(text for text in runs if text != runs[0])
    differing = _substitution_count(left, right)
    if differing is None or differing > CHAIN_MAX_SUBSTITUTIONS:
        return 0
    if center_x is None or not text_height or not all(
        _looks_like_subtitle(frames[index], center_x, text_height) for index in texts[left] + texts[right]
    ):
        return 0
    left_frames = [frames[index] for index in texts[left]]
    right_frames = [frames[index] for index in texts[right]]
    left_reads = sum(1 for index in texts[left] if fresh[index - start])
    right_reads = sum(1 for index in texts[right] if fresh[index - start])
    left_evidence, right_evidence = _context_evidence(left, right, corpus), _context_evidence(right, left, corpus)
    if left_evidence != right_evidence:
        winner = left if left_evidence > right_evidence else right
    elif left_reads != right_reads:
        winner = left if left_reads > right_reads else right
    else:
        left_conf = max(map(_frame_confidence, left_frames))
        right_conf = max(map(_frame_confidence, right_frames))
        winner = left if left_conf >= right_conf else right
    loser = right if winner == left else left
    best = max((frames[index] for index in texts[winner]), key=_frame_confidence)
    for index in texts[loser]:
        frames[index] = dataclasses.replace(frames[index], text_boxes=list(best.text_boxes))
    conflicts.add(winner)
    return len(texts[loser])


#: [v3.23.714] Hai khung liền nhau cách nhau quá chừng này (giây) thì không coi là cùng một lượt hiện chữ.
ADJACENT_READ_MAX_GAP_SEC: float = 0.25


def tighten_inflated_boxes(frames: Sequence[OcrFrameResult]) -> tuple[list[OcrFrameResult], int]:
    """[v3.23.714] Khung LIỀN NHAU đọc cùng một chữ: hộp phình (cao / rộng hơn :data:`INFLATED_BOX_RATIO` lần hộp gọn
    nhất của lượt) nhận hộp gọn — kể cả khi điểm ảnh chữ ĐỔI giữa hai khung (nền sáng tràn vào làm bộ lọc khung báo
    "chữ đổi", nên :func:`fuse_same_pixel_reads` không coi là cùng chuỗi).

    Lượt chạy thật của chủ dự án (3.23.712, khoảng gộp 0,1 s, dung sai căn giữa nhỏ): "喜" 369,24–369,32 s hộp cao
    143 px thay vì 76 → bị lọc căn lề loại, câu kết thúc sớm 0,16 s; "你" 5834,28–5834,44 s (cao 99 / 149 px) → bắt
    đầu trễ 0,20 s; "天驹山" 7247,96–7248,12 s (hộp rộng 351 px thay vì 188) → kết thúc sớm 0,28 s. Chỉ đổi HÌNH HỌC,
    chữ giữ nguyên; chỉ xét khung có đúng một hộp, đọc tin cậy (≥ :data:`FILL_MIN_CONFIDENCE`).
    """
    result = list(frames)
    count, fixed, start = len(result), 0, 0

    def usable(frame: OcrFrameResult) -> bool:
        return len(frame.text_boxes) == 1 and bool(_frame_text(frame)) and _frame_confidence(frame) >= FILL_MIN_CONFIDENCE

    while start < count:
        if not usable(result[start]):
            start += 1
            continue
        text, end = _frame_text(result[start]), start + 1
        while (
            end < count and usable(result[end]) and _frame_text(result[end]) == text
            and result[end].timestamp_sec - result[end - 1].timestamp_sec <= ADJACENT_READ_MAX_GAP_SEC
        ):
            end += 1
        if end - start >= 2:
            run = result[start:end]
            sized = [frame for frame in run if _box_height(frame) > 0 and _box_width(frame) > 0]
            if sized:
                tight = min(sized, key=lambda frame: (_box_height(frame) * _box_width(frame), -_frame_confidence(frame)))
                tight_height, tight_width = _box_height(tight), _box_width(tight)
                for index in range(start, end):
                    frame = result[index]
                    if (
                        _box_height(frame) > INFLATED_BOX_RATIO * tight_height
                        or _box_width(frame) > INFLATED_BOX_RATIO * tight_width
                    ):
                        result[index] = dataclasses.replace(frame, text_boxes=list(tight.text_boxes))
                        fixed += 1
        start = end
    return result, fixed


#: [v3.23.715] Hộp phình không có hộp gọn để mượn: chỉ đưa về dải phụ đề khi tâm lệch ≤ chừng này lần chiều cao chữ…
BAND_CENTER_TOLERANCE_HEIGHTS: float = 0.15
#: … bề rộng hộp trừ số chữ (tính theo bước chữ) nằm trong khoảng này (đo thật: phụ đề 0,32–0,72; nhiễu ≥ 1,2)…
BAND_WIDTH_EXCESS_CHARS: tuple[float, float] = (-0.5, 1.0)
#: … và hộp trùm lên dải phụ đề (mỗi mép hụt không quá chừng này lần chiều cao chữ).
BAND_COVER_TOLERANCE_HEIGHTS: float = 0.15
_HAN_PATTERN = re.compile(r"[\u3400-\u9fff]")


def measure_text_band(frames: Sequence[OcrFrameResult]) -> tuple[float, float] | None:
    """[v3.23.715] Dải dọc điển hình ``(mép trên, mép dưới)`` của hộp phụ đề (khung một hộp ≥ 3 chữ tin cậy ≥ 0,90)."""
    tops: list[float] = []
    bottoms: list[float] = []
    for frame in frames:
        if len(frame.text_boxes) != 1 or len(_frame_text(frame)) < 3 or _frame_confidence(frame) < 0.90:
            continue
        polygon = frame.text_boxes[0].polygon
        if not polygon:
            continue
        ys = [float(point[1]) for point in polygon]
        tops.append(min(ys))
        bottoms.append(max(ys))
    if len(tops) < 50:
        return None
    tops.sort()
    bottoms.sort()
    top, bottom = tops[len(tops) // 2], bottoms[len(bottoms) // 2]
    return (top, bottom) if bottom > top else None


def snap_inflated_boxes_to_band(
    frames: Sequence[OcrFrameResult],
    center_x: float | None,
    char_pitch: float | None,
    text_height: float | None,
    band: tuple[float, float] | None,
) -> tuple[list[OcrFrameResult], int]:
    """[v3.23.715] Hộp chữ phình theo chiều DỌC mà cả lượt hiện chữ không có hộp gọn nào → đưa về dải phụ đề của phim.

    Lượt chạy thật 3.23.713 (tiền xử lý không viền): câu một chữ "是" 2654,64–2655,36 s bị khoanh cao HẾT vùng chữ
    (0–136 px, dải phụ đề 65–133) ở cả 18 khung, "你" 4666,36–4666,64 s cũng vậy → tâm dọc lệch, bộ lọc lệch dọc giữa
    các khung loại cả câu; "切" mất 2 khung cuối, "你" 3269,96 s bị coi là hộp trôi. :func:`tighten_inflated_boxes`
    không cứu được vì không khung nào có hộp gọn. Đo trên 4 bộ dữ liệu trọn phim: mọi hộp phình là phụ đề thật đều
    nằm đúng tâm (lệch ≤ 4 px) và rộng đúng số chữ (dư 0,32–0,72 chữ); nhiễu thì lệch tâm ≥ 40 px hoặc rộng dư ≥ 1,2
    chữ, hoặc không phải chữ Hán. Chỉ đổi mép trên / dưới của hộp; chữ, độ tin cậy và toạ độ ngang giữ nguyên.
    """
    result = list(frames)
    if center_x is None or not char_pitch or not text_height or band is None:
        return result, 0
    band_top, band_bottom = band
    cover = BAND_COVER_TOLERANCE_HEIGHTS * text_height
    low, high = BAND_WIDTH_EXCESS_CHARS
    fixed = 0
    for index, frame in enumerate(result):
        if len(frame.text_boxes) != 1:
            continue
        box = frame.text_boxes[0]
        text = _frame_text(frame)
        if not box.polygon or not text or float(box.confidence) < FILL_MIN_CONFIDENCE or not _HAN_PATTERN.search(text):
            continue
        xs = [float(point[0]) for point in box.polygon]
        ys = [float(point[1]) for point in box.polygon]
        left, right, top, bottom = min(xs), max(xs), min(ys), max(ys)
        if bottom - top <= INFLATED_BOX_RATIO * text_height:
            continue
        if abs((left + right) / 2.0 - center_x) > BAND_CENTER_TOLERANCE_HEIGHTS * text_height:
            continue
        if not low <= (right - left) / char_pitch - len(text) <= high:
            continue
        if top > band_top + cover or bottom < band_bottom - cover:
            continue
        kind = type(box.polygon[0][0])
        polygon = [
            (kind(left), kind(band_top)), (kind(right), kind(band_top)),
            (kind(right), kind(band_bottom)), (kind(left), kind(band_bottom)),
        ]
        result[index] = dataclasses.replace(
            frame, text_boxes=[OcrTextBox(text=box.text, confidence=box.confidence, polygon=polygon)],
        )
        fixed += 1
    return result, fixed


def _confusable_only(left: str, right: str) -> bool:
    if len(left) != len(right) or left == right:
        return False
    return all(a == b or frozenset((a, b)) in CONFUSABLE_PAIRS for a, b in zip(left, right, strict=True))


def _build_canonical_map() -> dict[str, str]:
    """Mỗi chữ → chữ đại diện của nhóm dễ lẫn hình chứa nó (白, 自, 目, 日, 曰 chung một nhóm…)."""
    parent: dict[str, str] = {}

    def find(char: str) -> str:
        while parent.setdefault(char, char) != char:
            parent[char] = parent[parent[char]]
            char = parent[char]
        return char

    for pair in CONFUSABLE_PAIRS:
        first, second = sorted(pair)
        parent[find(second)] = find(first)
    return {char: find(char) for char in list(parent)}


_CANONICAL: dict[str, str] = _build_canonical_map()


def _canonical(text: str) -> str:
    return "".join(_CANONICAL.get(char, char) for char in text)


def _context_evidence(text: str, other: str, corpus: Sequence[str]) -> tuple[int, int]:
    """Bằng chứng từ ngữ liệu của phim cho ``text`` (so với bản đọc kia ``other`` cùng số chữ).

    ``(độ dài cụm chữ dài nhất chứa chỗ khác nhau mà một CÂU KHÁC cũng có, số câu khác có cặp chữ liền kề quanh chỗ
    khác nhau)``. "Câu khác" không gồm các bản đọc lẫn của chính câu này: cùng dạng đại diện, hoặc cùng số chữ và
    chỉ khác ≤ :data:`CHAIN_MAX_SUBSTITUTIONS` vị trí.
    """
    own = _canonical(text)
    others = [
        line for line in corpus
        if _canonical(line) != own and (_substitution_count(line, text) or 99) > CHAIN_MAX_SUBSTITUTIONS
    ]
    longest = 0
    pair_hits = 0
    for index, (a, b) in enumerate(zip(text, other, strict=True)):
        if a == b:
            continue
        for begin in range(index + 1):
            for stop in range(max(index + 1, begin + 2), len(text) + 1):
                if stop - begin <= longest:
                    continue
                piece = text[begin:stop]
                if any(piece in line for line in others):
                    longest = stop - begin
        for pair in (text[index - 1:index + 1] if index else "", text[index:index + 2]):
            if len(pair) == 2:
                pair_hits += sum(1 for line in others if pair in line)
    return longest, pair_hits


def _pick_confusable(left: str, right: str, left_frames: int, right_frames: int, corpus: Sequence[str]) -> str:
    """Chọn giữa hai bản đọc chỉ khác nhau ở cặp chữ dễ lẫn hình (hai đoạn liền nhau, không có bằng chứng điểm ảnh).

    Độ tin cậy của mô hình không phân xử được (đo thật: "清清白白" 0,95 và bản sai "清清自白" 0,96). Dùng chính ngữ
    liệu của phim (:func:`_context_evidence`: "白日宣淫" 4 chữ thắng "想自" 2 chữ; "清白" có ở câu khác); hoà thì
    bản có nhiều khung hơn, rồi bản xuất hiện trước.
    """
    left_evidence, right_evidence = _context_evidence(left, right, corpus), _context_evidence(right, left, corpus)
    if left_evidence != right_evidence:
        return left if left_evidence > right_evidence else right
    return right if right_frames > left_frames else left


def unify_confusable_variants(
    frames: Sequence[OcrFrameResult], center_x: float | None = None, char_pitch: float | None = None,
) -> tuple[list[OcrFrameResult], set[str]]:
    """Hai đoạn khung LIỀN NHAU chỉ khác nhau ở cặp chữ dễ lẫn hình, hoặc chỉ hơn kém chữ "一" → một câu.

    Trả ``(khung, các câu đã chọn)``.

    Chỉ xét khung một hộp chữ. Cặp chữ dễ lẫn: chọn theo ngữ liệu của chính phim (:func:`_pick_confusable`); hộp
    chữ (vị trí) của từng khung giữ nguyên, chỉ thay chữ. Các câu trả về cần gắn cờ nghi ngờ vì hai bản đọc mâu thuẫn.
    """
    result = list(frames)
    merged_starts: set[int] = set()
    # đoạn = [bắt đầu, kết thúc) các khung liền nhau cùng chữ
    segments: list[tuple[int, int, str]] = []
    index = 0
    count = len(result)
    while index < count:
        frame = result[index]
        text = _frame_text(frame) if len(frame.text_boxes) == 1 else ""
        stop = index + 1
        while stop < count and len(result[stop].text_boxes) == len(frame.text_boxes) and (
            (_frame_text(result[stop]) if len(result[stop].text_boxes) == 1 else "") == text
        ):
            stop += 1
        segments.append((index, stop, text))
        index = stop
    corpus = sorted({text for _, _, text in segments if len(text) >= 2})
    fresh = fresh_read_flags(result)
    position = 0
    while position + 1 < len(segments):
        left_start, left_stop, left_text = segments[position]
        right_start, right_stop, right_text = segments[position + 1]
        if not (left_text and right_text):
            position += 1
            continue
        if center_x is not None and (
            _only_yi_inserted(left_text, right_text) or _only_yi_inserted(right_text, left_text)
        ):
            # Hơn kém chữ "一" ("一晚上" → "晚上" → "一晚上", không khung trống xen giữa): bản có hộp chữ đúng tâm.
            left_frames, right_frames = result[left_start:left_stop], result[right_start:right_stop]
            if len(left_text) > len(right_text):
                winner = left_text if _pick_yi_variant(left_frames, right_frames, center_x, char_pitch) else right_text
            else:
                winner = right_text if _pick_yi_variant(right_frames, left_frames, center_x, char_pitch) else left_text
        elif _confusable_only(left_text, right_text):
            # [v3.23.726] Hoà ngữ liệu thì bản nhiều LẦN ĐỌC hơn thắng (khung chép không phải bằng chứng mới).
            winner = _pick_confusable(
                left_text, right_text, sum(fresh[left_start:left_stop]), sum(fresh[right_start:right_stop]), corpus,
            )
        else:
            position += 1
            continue
        loser_start, loser_stop = (left_start, left_stop) if winner == right_text else (right_start, right_stop)
        for k in range(loser_start, loser_stop):
            box = result[k].text_boxes[0]
            result[k] = dataclasses.replace(
                result[k], text_boxes=[OcrTextBox(text=winner, confidence=box.confidence, polygon=box.polygon)],
            )
        segments[position:position + 2] = [(left_start, right_stop, winner)]
        merged_starts.discard(right_start)
        merged_starts.add(left_start)
        position = max(0, position - 1)          # đoạn vừa gộp có thể lại sát một bản lẫn khác (A-B-A)
    # Chỉ báo chữ CUỐI CÙNG của các đoạn đã gộp (bản thắng tạm ở bước giữa không còn trong kết quả).
    chosen = {text for start, _, text in segments if start in merged_starts}
    return result, chosen
