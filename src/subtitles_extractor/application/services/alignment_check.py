"""[v3.23.516] Kiểm CĂN DÒNG ngay tại từng lô dịch, để thử lại khi model trả kết quả lệch.

Vì sao cần: các lớp kiểm sẵn có chỉ xét CẤU TRÚC (đủ số dòng, ``line_no`` hợp lệ) nên lệch KHỐI lọt
hoàn toàn — mọi dòng vẫn đủ, vẫn dài bình thường, chỉ có điều nội dung nằm sai chỗ. Dữ liệu thật có
vùng dòng 257-262 lệch nguyên khối 3 dòng mà không cờ nào báo, và cũng không có cơ hội thử lại vì
lô đã được chấp nhận.

Cách kiểm TẤT ĐỊNH: dùng bảng nhân vật và bảng thuật ngữ đã duyệt làm MỐC NEO. Một dòng bị coi là
lệch khi bản dịch của nó KHÔNG mang mốc neo nào của nguồn chính nó, NHƯNG lại mang mốc neo của nguồn
một dòng lân cận trong cùng lô.

Ngưỡng thận trọng: chỉ kết luận lô lệch khi có TỪ HAI dòng trở lên bị cờ. Lý do là một dòng đơn lẻ
thường chỉ là ĐẢO TRẬT TỰ hợp lệ — khi câu bị tách đôi, tiếng Việt hay đặt từ khoá sang dòng kế
(``宋欣爽的单日返现`` / ``次数已达上限`` → "Tống Hân Sảng đã đạt giới hạn" / "số lần hoàn tiền trong
ngày"). Lệch thật gần như luôn kéo theo cả cụm.

Toàn bộ hàm ở đây là HÀM THUẦN: không I/O, không phụ thuộc Qt/AI.
"""

from __future__ import annotations

import re
from typing import Any

MAX_NEIGHBOUR_DISTANCE = 3
"""Chỉ đối chiếu với các dòng lân cận trong khoảng này — lệch khối thực tế đều nằm trong đó."""

MIN_FLAGGED_LINES = 2
"""Số dòng bị cờ tối thiểu để kết luận LÔ lệch (một dòng đơn lẻ thường là đảo trật tự hợp lệ)."""

MIN_ANCHOR_TARGET_CHARS = 3
"""Bản dịch mốc neo phải đủ dài mới dùng được — chuỗi 1-2 ký tự khớp bừa quá dễ."""


SENTENCE_END_MARKS = "。！？?!…～~"
"""Dấu kết câu — dùng để nhận ra hai dòng phụ đề vốn là MỘT câu bị tách đôi."""


def is_split_sentence(sources: dict[int, str], first_index: int, second_index: int) -> bool:
    """[v3.23.522] True nếu hai dòng liền kề là MỘT câu bị tách đôi.

    Vì sao quan trọng: khi câu bị tách đôi, tiếng Việt đảo trật tự nên từ khoá có thể rơi sang
    dòng kế mà bản dịch vẫn ĐÚNG. Ví dụ thật: ``我早就跟贺强`` / ``睡过了`` dịch thành "tôi sớm đã
    ngủ với" / "Hạ Cường rồi." — tên nhân vật chuyển dòng là điều tự nhiên, không phải lệch.

    Đo trên dữ liệu thật: 32/40 dòng bị cờ (80%) thuộc đúng dạng này — tức phần lớn cảnh báo là
    DƯƠNG TÍNH GIẢ nếu không loại trừ.

    Args:
        sources: Bản đồ chỉ số → câu gốc.
        first_index: Chỉ số dòng thứ nhất.
        second_index: Chỉ số dòng thứ hai.

    Returns:
        True khi hai dòng kề nhau và dòng TRƯỚC không kết thúc bằng dấu kết câu.
    """
    if abs(first_index - second_index) != 1:
        return False
    previous = sources.get(min(first_index, second_index), "").strip()
    return bool(previous) and not previous.endswith(tuple(SENTENCE_END_MARKS))


MAX_ANCHOR_FREQUENCY = 0.02
"""Mốc neo xuất hiện ở quá tỉ lệ này số dòng thì BỎ — quá phổ biến nên khớp nhầm.

[v3.23.523] Tên nhân vật chính là mốc neo TỒI: nó có mặt ở hàng trăm dòng, nên "dòng bên cạnh cũng
chứa tên đó" chẳng chứng minh được gì. Dữ liệu thật: dòng 2578 gốc ``我如果告尉了小逸`` dịch
"Nếu chị mà nói cho Lâm Dật biết" — HOÀN TOÀN ĐÚNG, nhưng bị cờ vì biệt danh ``小逸`` không có trong
roster còn "Lâm Dật" thì xuất hiện khắp nơi."""


def drop_frequent_anchors(
    anchors: list[tuple[str, str]],
    translations: dict[int, str],
    max_frequency: float = MAX_ANCHOR_FREQUENCY,
) -> list[tuple[str, str]]:
    """Bỏ các mốc neo xuất hiện quá dày trong bản dịch.

    Args:
        anchors: Danh sách mốc neo ``(từ gốc, bản dịch)``.
        translations: Bản dịch theo chỉ số, dùng để đếm tần suất.
        max_frequency: Tỉ lệ số dòng tối đa mà một mốc neo được phép xuất hiện.

    Returns:
        Danh sách mốc neo đủ hiếm để dùng làm bằng chứng căn dòng.
    """
    if not translations:
        return anchors
    total = len(translations)
    limit = max(3, int(total * max_frequency))
    frequency: dict[str, int] = {}
    for text in translations.values():
        lowered = text.lower()
        for _, target in anchors:
            if target in lowered:
                frequency[target] = frequency.get(target, 0) + 1
    return [pair for pair in anchors if frequency.get(pair[1], 0) <= limit]


def build_anchor_pairs(characters: str, glossary: str) -> list[tuple[str, str]]:
    """Dựng danh sách mốc neo ``(từ gốc, bản dịch)`` từ bảng nhân vật + bảng thuật ngữ.

    Args:
        characters: Bảng nhân vật (quy cách cố định).
        glossary: Bảng thuật ngữ, mỗi dòng ``gốc = dịch``.

    Returns:
        Danh sách cặp đã chuẩn hoá chữ thường ở vế dịch, bỏ trùng.
    """
    from subtitles_extractor.application.services.roster_format import (
        parse_glossary,
        parse_roster,
    )

    anchors: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add(source_term: str, target_text: str) -> None:
        term = source_term.strip()
        target = re.sub(r"\([^)]*\)", "", target_text).strip().lower()
        if len(term) < 2 or len(target) < MIN_ANCHOR_TARGET_CHARS:
            return
        key = (term, target)
        if key not in seen:
            seen.add(key)
            anchors.append(key)

    for entry in parse_glossary(glossary):
        first_option = entry.translation.split("/")[0]
        for alias in re.split(r"\s*[/|]\s*", entry.term):
            add(alias, first_option)
    for person in parse_roster(characters):
        if not person.aka.strip():
            continue
        for alias in re.split(r"\s*[/|]\s*", person.name):
            add(alias, person.aka)
    return anchors


def _has_anchor_of(source_text: str, translated_text: str, anchors: list[tuple[str, str]]) -> bool:
    """True nếu ``translated_text`` mang ít nhất một mốc neo của ``source_text``."""
    if not source_text or not translated_text:
        return False
    lowered = translated_text.lower()
    return any(term in source_text and target in lowered for term, target in anchors)


def drop_unreliable_anchors(
    anchors: list[tuple[str, str]],
    sources: dict[int, str],
    translations: dict[int, str],
) -> list[tuple[str, str]]:
    """[v3.23.549] Bỏ mốc neo có vế DỊCH xuất hiện nhiều hơn vế GỐC.

    Vì sao cần: mốc neo chỉ là bằng chứng căn dòng khi hai vế tương ứng MỘT-MỘT. Nếu vế tiếng
    Việt xuất hiện ở nhiều dòng hơn số dòng chứa từ gốc, phần dư đến từ chỗ khác — hô ngữ model
    thêm vào, hoặc một từ gốc ĐỒNG NGHĨA mà bảng thuật ngữ không có. Khi đó "dòng bên cạnh cũng
    chứa chuỗi này" không chứng minh được gì.

    Đo trên phim thật (3902 dòng): bộ lọc này một mình khử 4/5 dương tính giả.

    * ``捕鱼`` → "bắt cá": gốc 4 dòng, dịch 7 dòng. Dòng ``那你的渔具呢?`` dịch "dụng cụ đánh
      **bắt cá**" — ``渔具`` sinh ra cụm đó, không phải lệch dòng.
    * ``医生`` → "bác sĩ": gốc 11, dịch 16. Model thêm hô ngữ "Bác sĩ, ..." ở dòng không hề có
      ``医生``.

    Args:
        anchors: Danh sách mốc neo ``(từ gốc, bản dịch)``.
        sources: Câu gốc theo chỉ số.
        translations: Bản dịch theo chỉ số.

    Returns:
        Các mốc neo mà vế dịch KHÔNG phổ biến hơn vế gốc.
    """
    if not sources or not translations:
        return anchors
    lowered = [text.lower() for text in translations.values()]
    reliable: list[tuple[str, str]] = []
    for term, target in anchors:
        source_hits = sum(1 for text in sources.values() if term in text)
        if source_hits <= 0:
            continue
        target_hits = sum(1 for text in lowered if target in text)
        if target_hits <= source_hits:
            reliable.append((term, target))
    return reliable


MIN_CORROBORATING_LINES = 2
"""Số dòng CÙNG độ lệch cần có trong một cụm để tin là lệch thật."""

CORROBORATION_SPAN = 4
"""Hai dòng cùng độ lệch phải cách nhau không quá ngần này mới coi là cùng một cụm."""


def corroborated_findings(
    findings: list[dict[str, Any]],
    span: int = CORROBORATION_SPAN,
    min_lines: int = MIN_CORROBORATING_LINES,
) -> list[dict[str, Any]]:
    """[v3.23.549] Chỉ giữ dòng lệch có dòng khác CÙNG ĐỘ LỆCH ở gần bên cạnh.

    Lệch dòng thật do model gộp/bỏ dòng nên luôn kéo theo cả CỤM dòng liên tiếp cùng một độ
    lệch. Một dòng đơn lẻ với độ lệch riêng là dấu hiệu của trùng nghĩa ngẫu nhiên.

    Đo trên phim thật: 5 dòng bị cờ có 5 độ lệch KHÁC NHAU (-1, +2, -3, +3, +2) và không dòng
    nào kề dòng nào — cả 5 đều là dương tính giả, và bộ lọc này khử sạch.

    Ngoại lệ: HOÁN ĐỔI TƯƠNG HỖ (dòng A mang nội dung B, dòng B mang nội dung A) cũng là bằng
    chứng đủ mạnh dù hai độ lệch ngược dấu — hai mốc neo độc lập chỉ thẳng vào nhau thì không thể
    là trùng hợp. Không dương tính giả nào trong dữ liệu thật có dạng này.

    Args:
        findings: Kết quả từ :func:`find_misaligned_lines`.
        span: Khoảng cách tối đa giữa hai dòng cùng cụm.
        min_lines: Số dòng tối thiểu trong cụm.

    Returns:
        Các dòng lệch có bằng chứng bổ trợ, sắp theo chỉ số.
    """
    pointer = {int(item["index"]): int(item["belongs_to_index"]) for item in findings}
    swapped = {
        index for index, target in pointer.items() if pointer.get(target) == index
    }

    by_shift: dict[int, list[dict[str, Any]]] = {}
    for item in findings:
        by_shift.setdefault(int(item["shift"]), []).append(item)

    kept: list[dict[str, Any]] = [item for item in findings if item["index"] in swapped]
    for group in by_shift.values():
        group.sort(key=lambda item: item["index"])
        run = [group[0]]
        for item in group[1:]:
            if item["index"] - run[-1]["index"] <= span:
                run.append(item)
                continue
            if len(run) >= min_lines:
                kept.extend(run)
            run = [item]
        if len(run) >= min_lines:
            kept.extend(run)
    unique = {int(item["index"]): item for item in kept}
    return [unique[index] for index in sorted(unique)]


def find_misaligned_lines(
    sources: dict[int, str],
    translations: dict[int, str],
    anchors: list[tuple[str, str]],
    max_distance: int = MAX_NEIGHBOUR_DISTANCE,
) -> list[dict[str, Any]]:
    """Tìm các dòng có bản dịch mang nội dung của dòng LÂN CẬN thay vì của chính nó.

    Args:
        sources: Bản đồ chỉ số → câu gốc (trong phạm vi đang xét).
        translations: Bản đồ chỉ số → bản dịch tương ứng.
        anchors: Mốc neo từ :func:`build_anchor_pairs`.
        max_distance: Khoảng lân cận tối đa cần đối chiếu.

    Returns:
        Danh sách ``{"index", "belongs_to_index", "shift"}`` cho từng dòng nghi lệch.
    """
    if not anchors:
        return []
    anchors = drop_frequent_anchors(anchors, translations)
    if not anchors:
        return []
    ordered = sorted(sources)
    findings: list[dict[str, Any]] = []
    for position, index in enumerate(ordered):
        own_source = sources.get(index, "")
        own_translation = translations.get(index, "")
        if _has_anchor_of(own_source, own_translation, anchors):
            continue  # bản dịch khớp nguồn của chính nó → bình thường
        for distance in range(1, max_distance + 1):
            neighbours = (position - distance, position + distance)
            matched = next(
                (
                    ordered[candidate]
                    for candidate in neighbours
                    if 0 <= candidate < len(ordered)
                    and _has_anchor_of(sources[ordered[candidate]], own_translation, anchors)
                ),
                None,
            )
            if matched is not None:
                # Câu tách đôi + lệch đúng 1 dòng → đảo trật tự hợp lệ, KHÔNG phải lệch.
                if is_split_sentence(sources, index, matched):
                    break
                findings.append({
                    "index": index,
                    "belongs_to_index": matched,
                    "shift": matched - index,
                })
                break
    return findings


def is_batch_misaligned(
    sources: dict[int, str],
    translations: dict[int, str],
    anchors: list[tuple[str, str]],
    min_flagged: int = MIN_FLAGGED_LINES,
) -> list[dict[str, Any]]:
    """Kết luận LÔ có lệch dòng hay không (dùng để quyết định thử lại).

    Trả về danh sách dòng lệch khi đủ ngưỡng, ngược lại trả danh sách RỖNG — tức không đủ căn cứ
    để bắt model dịch lại. Thà bỏ sót một dòng lệch còn hơn bắt dịch lại hàng loạt lô vốn đã đúng.

    Args:
        sources: Câu gốc của lô.
        translations: Bản dịch tương ứng.
        anchors: Mốc neo.
        min_flagged: Ngưỡng số dòng bị cờ.

    Returns:
        Danh sách dòng lệch (rỗng nếu chưa đủ ngưỡng).
    """
    findings = find_misaligned_lines(sources, translations, anchors)
    return findings if len(findings) >= min_flagged else []


MIN_LINES_FOR_CORRELATION = 10
"""Dưới ngần này dòng thì hệ số tương quan quá nhiễu để kết luận."""

LOW_CORRELATION_THRESHOLD = 0.55
"""Tương quan độ dài tại độ lệch 0 thấp hơn mức này là dấu hiệu lô đang lệch.

Đo trên phim thật (222 lô): trung vị 0.84, phân vị 5% là 0.63 — nên 0.55 nằm dưới hẳn vùng bình
thường, chỉ chạm tới các lô thực sự bất thường."""

MIN_CORRELATION_GAIN = 0.05
"""Độ lệch khác phải tương quan CAO HƠN mức tại 0 ít nhất ngần này mới coi là bằng chứng lệch."""


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    """Hệ số tương quan Pearson; None khi không đủ dữ liệu hoặc một dãy không biến thiên."""
    if len(xs) < MIN_LINES_FOR_CORRELATION or len(xs) != len(ys):
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    dev_x = sum((x - mean_x) ** 2 for x in xs) ** 0.5
    dev_y = sum((y - mean_y) ** 2 for y in ys) ** 0.5
    if not dev_x or not dev_y:
        return None
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    return covariance / (dev_x * dev_y)


def detect_length_shift(
    sources: dict[int, str],
    translations: dict[int, str],
    max_distance: int = MAX_NEIGHBOUR_DISTANCE,
) -> int | None:
    """[v3.23.517] Dò lệch khối bằng TƯƠNG QUAN ĐỘ DÀI — phủ được MỌI dòng.

    Vì sao cần thêm cách này: mốc neo tên/thuật ngữ chỉ phủ khoảng 13% số dòng, nên vùng lệch mà
    không chứa tên riêng nào sẽ lọt. Độ dài câu thì dòng nào cũng có: câu gốc dài kéo theo bản dịch
    dài, nên dãy độ dài của nguồn và của bản dịch phải tương quan mạnh khi căn ĐÚNG dòng. Nếu cả
    khối bị đẩy đi k dòng, tương quan tại 0 tụt xuống và đỉnh chuyển sang k.

    Đo trên phim thật: các lô bình thường có tương quan tại 0 khoảng 0.71-0.92, còn lô lệch tụt
    xuống 0.40 và đỉnh nằm đúng ở khoảng lệch thật.

    Args:
        sources: Câu gốc theo chỉ số.
        translations: Bản dịch theo chỉ số (có thể chứa cả dòng ngoài lô để dò lệch).
        max_distance: Khoảng lệch tối đa cần dò.

    Returns:
        Khoảng lệch phát hiện được (khác 0), hoặc None nếu không đủ bằng chứng.
    """
    ordered = sorted(sources)
    source_lengths = [float(len(sources[index])) for index in ordered]
    baseline = _pearson(
        source_lengths, [float(len(translations.get(index, ""))) for index in ordered]
    )
    if baseline is None or baseline >= LOW_CORRELATION_THRESHOLD:
        return None
    best_shift, best_score = 0, baseline
    for shift in range(-max_distance, max_distance + 1):
        if shift == 0:
            continue
        score = _pearson(
            source_lengths,
            [float(len(translations.get(index + shift, ""))) for index in ordered],
        )
        if score is not None and score > best_score:
            best_shift, best_score = shift, score
    if best_shift and best_score >= baseline + MIN_CORRELATION_GAIN:
        return best_shift
    return None


BLOCK_WINDOW_LINES = 10
"""Số dòng mỗi cửa sổ trượt khi dò lệch theo KHỐI."""

BLOCK_LOW_CORRELATION = 0.20
"""Tương quan độ dài dưới mức này là bất thường.

Đo trên phim thật (3902 dòng, 3893 cửa sổ): trung vị 0.88, phân vị 5% là 0.59. Khối lệch thật
tụt xuống -0.80. Ngưỡng 0.20 nằm giữa hai vùng, rất xa cả hai."""

BLOCK_MIN_RUN = 6
"""Số cửa sổ liên tiếp dưới ngưỡng cần có để kết luận — chặn nhiễu cục bộ.

Đo trên phim thật: khối lệch thật kéo 17 cửa sổ liên tiếp, còn các dao động lành tính chỉ 1-4
cửa sổ. Ngưỡng 6 tách sạch hai nhóm: 1 khối đúng, 0 khối sai trên toàn bộ 3902 dòng."""

BLOCK_MAX_GAP = 2
"""Hai cửa sổ dưới ngưỡng cách nhau không quá ngần này thì vẫn coi là cùng một khối."""


def find_low_correlation_blocks(
    sources: dict[int, str],
    translations: dict[int, str],
    window: int = BLOCK_WINDOW_LINES,
    threshold: float = BLOCK_LOW_CORRELATION,
    min_run: int = BLOCK_MIN_RUN,
) -> list[dict[str, Any]]:
    """[v3.23.549] Dò lệch dòng theo KHỐI bằng tương quan độ dài trên CỬA SỔ TRƯỢT.

    Vì sao cần thêm cách này: :func:`detect_length_shift` tính MỘT hệ số cho cả lô, nên một khối
    lệch 18 dòng nằm trong lô 50 dòng bị pha loãng và không bao giờ chạm ngưỡng. Nó cũng đòi cả
    lô cùng trôi một độ lệch, trong khi lỗi thật thường là model GỘP hai dòng ngắn rồi đẩy lệch
    từ đó tới khi tự đồng bộ lại.

    Ca thật (phim 3902 dòng): dòng 2683 ``说够了赶紧给我滚开`` bị gộp luôn nội dung dòng 2684
    ``挡我道了`` thành "Nói đủ rồi thì cút ngay cho tao, chắn đường rồi." — 18 dòng sau đó mang bản
    dịch của dòng KẾ TIẾP. Mọi cờ hiện có đều im lặng; riêng tương quan tụt từ +0.88 xuống -0.60.

    Mốc neo tên/thuật ngữ KHÔNG cứu được ca này vì khối đó không chứa mốc neo đáng tin nào — đây
    đúng là khoảng trống mà độ dài câu lấp được, vì dòng nào cũng có độ dài.

    Args:
        sources: Câu gốc theo chỉ số.
        translations: Bản dịch theo chỉ số.
        window: Số dòng mỗi cửa sổ.
        threshold: Ngưỡng tương quan coi là bất thường.
        min_run: Số cửa sổ liên tiếp tối thiểu.

    Returns:
        Danh sách ``{"from_index", "to_index", "lines", "min_correlation", "windows"}``, mỗi mục
        là một khối nghi lệch, sắp theo mức nghiêm trọng giảm dần.
    """
    ordered = sorted(sources)
    if len(ordered) < window + min_run:
        return []
    source_lengths = [float(len(sources[index])) for index in ordered]
    target_lengths = [float(len(translations.get(index, ""))) for index in ordered]

    low_positions: list[tuple[int, float]] = []
    for start in range(len(ordered) - window + 1):
        score = _pearson(
            source_lengths[start : start + window], target_lengths[start : start + window]
        )
        if score is not None and score < threshold:
            low_positions.append((start, score))

    blocks: list[dict[str, Any]] = []
    run: list[tuple[int, float]] = []

    def _flush() -> None:
        if len(run) < min_run:
            return
        first_index = ordered[run[0][0]]
        last_index = ordered[min(run[-1][0] + window - 1, len(ordered) - 1)]
        blocks.append({
            "from_index": first_index,
            "to_index": last_index,
            "lines": last_index - first_index + 1,
            "min_correlation": round(min(score for _, score in run), 3),
            "windows": len(run),
        })

    for position, score in low_positions:
        if run and position - run[-1][0] <= BLOCK_MAX_GAP:
            run.append((position, score))
            continue
        _flush()
        run = [(position, score)]
    _flush()

    blocks.sort(key=lambda item: (item["windows"], -item["min_correlation"]), reverse=True)
    return blocks
