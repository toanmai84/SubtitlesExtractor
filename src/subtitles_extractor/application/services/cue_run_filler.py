"""[v3.23.572] Điền nhãn Visual Cues cho dòng NÓI TIẾP khi model chỉ ghi đầu lượt lời.

Chuẩn đoán thật v3.23.571 (phân tích bằng gemini-3.8-flash): cues chỉ phủ 2908/3902
dòng (74,5%). Không phải rải rác — 14/22 đoạn video phủ 98-100%, còn 8 đoạn chỉ 19-43%.
Ở 8 đoạn đó model chỉ ghi cue ở dòng ĐẦU mỗi lượt lời: 82-100% cặp cue liền nhau đổi
người nói (đoạn đầy đủ chỉ 26-36%). Dòng nói tiếp bị bỏ trống nên 25% số dòng dịch mà
không có người nói/người nghe, cũng không có quy tắc xưng hô. Đo trên bản dịch cuối,
lấy nhãn mà ≥ 3 lượt phân tích độc lập cùng thống nhất làm chuẩn: dòng không nhãn có
3,4% dòng sai quy tắc mà lưới xưng hô bắt được, dòng có nhãn chỉ 0,3% — gấp 11 lần.

Cue ghi ở đầu lượt lời thì các dòng sau nó, tới cue kế tiếp, là cùng lượt lời. So với
từng lượt phân tích đầy đủ khác của cùng phim: lấy cue ĐỨNG TRƯỚC khớp người nói 62-73%
— ngang mức hai lượt phân tích độc lập khớp nhau (58-73%) — còn lấy cue ĐỨNG SAU chỉ
22-29%. Ba chốt chặn, cùng đo trên dữ liệu thật:

* Chỉ điền ở vùng THƯA (dưới 70% số dòng trong ±20 dòng có cue): vùng dày mà thiếu vài
  dòng là model cố ý để trống. Cách nhận vùng này bắt đủ 975/975 dòng thiếu thuộc 8 đoạn
  thưa, chỉ lẫn 6 dòng của đoạn dày.
* Dừng lượt lời ở khoảng lặng > 2 giây: dòng sau khoảng lặng chỉ khớp chuẩn 14/24.
* Chỉ điền khi cues phủ ít nhất 60% cả phim. Năm lượt phân tích phủ 71-83% (đa số đoạn
  đủ, vài đoạn chỉ ghi đầu lượt lời) điền đúng 81-87%; ba lượt chỉ phủ 22-27% (cả đoạn
  dài không có cue nào) chỉ đúng 67-70% — khi đó khâu dịch QUÉT BỔ SUNG, không điền.

Kết quả trên phim này: điền 891 dòng, cặp người nói → người nghe khớp chuẩn 87% (241/277;
cue do model tự ghi: 96%). Lượt phân tích đã đầy đủ (99,8%) không bị điền dòng nào.
"""

from __future__ import annotations

from dataclasses import dataclass

from subtitles_extractor.domain.ports.subtitle_translator_port import VisualCue

#: Nửa độ rộng cửa sổ (số dòng mỗi bên) để đo mật độ cue quanh một dòng.
SPARSE_WINDOW_LINES = 20
#: Mật độ cue dưới ngưỡng này thì vùng đó là kiểu "chỉ ghi đầu lượt lời".
SPARSE_COVERAGE_BELOW = 0.7
#: Khoảng lặng dài hơn mức này (mili-giây) kết thúc lượt lời.
RUN_BREAK_SILENCE_MS = 2000
#: Cues phủ dưới mức này cả phim thì không phải kiểu "ghi đầu lượt lời" — không điền.
MIN_COVERAGE_FOR_RUN_FILL = 0.6


@dataclass(frozen=True, slots=True)
class LineSpan:
    """Vị trí và thời gian của một dòng phụ đề.

    Attributes:
        line_no: Số dòng (trùng ``id`` của Visual Cues).
        start_ms: Mốc bắt đầu (mili-giây).
        end_ms: Mốc kết thúc (mili-giây).
    """

    line_no: int
    start_ms: int
    end_ms: int


def _sparse_flags(covered: list[bool]) -> list[bool]:
    """Mỗi vị trí có nằm trong vùng cue thưa không (cửa sổ trượt, tổng tiền tố)."""
    prefix = [0]
    for is_covered in covered:
        prefix.append(prefix[-1] + int(is_covered))
    flags: list[bool] = []
    for position in range(len(covered)):
        low = max(0, position - SPARSE_WINDOW_LINES)
        high = min(len(covered), position + SPARSE_WINDOW_LINES + 1)
        density = (prefix[high] - prefix[low]) / (high - low)
        flags.append(density < SPARSE_COVERAGE_BELOW)
    return flags


def fill_cue_runs(
    cues: list[VisualCue], spans: list[LineSpan]
) -> tuple[list[VisualCue], list[int]]:
    """Điền nhãn người nói/nghe cho dòng nói tiếp trong vùng cue thưa (hàm thuần).

    Args:
        cues: Visual Cues như model trả.
        spans: MỌI dòng phụ đề (kể cả dòng không có cue), theo thứ tự thời gian.

    Returns:
        (cues đã điền, các dòng được điền). Không điền gì thì trả đúng danh sách cũ.
    """
    by_line = {cue.line_no: cue for cue in cues}
    ordered = sorted(spans, key=lambda span: span.line_no)
    covered = [span.line_no in by_line for span in ordered]
    if not ordered or sum(covered) / len(ordered) < MIN_COVERAGE_FOR_RUN_FILL:
        return cues, []
    sparse = _sparse_flags(covered)
    added: list[VisualCue] = []
    run_head: VisualCue | None = None
    previous_end: int | None = None
    for position, span in enumerate(ordered):
        silence = 0 if previous_end is None else span.start_ms - previous_end
        previous_end = span.end_ms
        if silence > RUN_BREAK_SILENCE_MS:
            run_head = None
        cue = by_line.get(span.line_no)
        if cue is not None:
            run_head = cue if cue.speaker else None
            continue
        if run_head is None or not sparse[position]:
            continue
        added.append(
            VisualCue(
                line_no=span.line_no,
                speaker=run_head.speaker,
                addressee=run_head.addressee,
                scene="",
            )
        )
    if not added:
        return cues, []
    merged = sorted([*cues, *added], key=lambda cue: cue.line_no)
    return merged, [cue.line_no for cue in added]


__all__ = [
    "MIN_COVERAGE_FOR_RUN_FILL",
    "RUN_BREAK_SILENCE_MS",
    "SPARSE_COVERAGE_BELOW",
    "SPARSE_WINDOW_LINES",
    "LineSpan",
    "fill_cue_runs",
]
