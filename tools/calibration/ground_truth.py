"""Đọc ground-truth SRT và ghép cặp tự động seraw↔SRT bằng độ trùng n-gram.

Việc ghép cặp đúng là *điều kiện tiên quyết* của hiệu chuẩn: nếu ghép sai cặp,
mọi chỉ số đo được đều vô nghĩa (bài học từ phiên trước — cặp giả định sai khiến
CER giả tạo cao tới 1.6).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from tools.calibration.exceptions import (
    GroundTruthNotFoundError,
    PairingAmbiguousError,
)

_SRT_TIME_PATTERN = re.compile(
    r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)"
)
_CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")


@dataclass(frozen=True, slots=True)
class GroundTruthCue:
    """Một câu phụ đề ground-truth.

    Attributes:
        start_sec: Mốc bắt đầu (giây).
        end_sec: Mốc kết thúc (giây).
        text: Nội dung văn bản đã gộp dòng.
    """

    start_sec: float
    end_sec: float
    text: str


def parse_srt(srt_path: Path) -> list[GroundTruthCue]:
    """Phân tích file SRT thành danh sách :class:`GroundTruthCue`.

    Args:
        srt_path: Đường dẫn file ``.srt`` (UTF-8).

    Returns:
        Danh sách câu phụ đề theo thứ tự thời gian.

    Raises:
        GroundTruthNotFoundError: Khi file không tồn tại hoặc không đọc được.
    """
    if not srt_path.is_file():
        raise GroundTruthNotFoundError(f"Không tìm thấy SRT: {srt_path}")
    try:
        raw_text = srt_path.read_text(encoding="utf-8", errors="ignore")
    except OSError as os_error:
        raise GroundTruthNotFoundError(f"Không đọc được SRT: {srt_path}") from os_error

    cues: list[GroundTruthCue] = []
    for block in re.split(r"\n\s*\n", raw_text.replace("\r\n", "\n").strip()):
        time_match = _SRT_TIME_PATTERN.search(block)
        if time_match is None:
            continue
        parts = list(map(int, time_match.groups()))
        start = parts[0] * 3600 + parts[1] * 60 + parts[2] + parts[3] / 1000.0
        end = parts[4] * 3600 + parts[5] * 60 + parts[6] + parts[7] / 1000.0
        content_lines = [
            line
            for line in block.split("\n")
            if "-->" not in line and not line.strip().isdigit() and line.strip()
        ]
        text = "".join(content_lines).strip()
        if text:
            cues.append(GroundTruthCue(start_sec=start, end_sec=end, text=text))
    return cues


def _cjk_characters(text: str) -> str:
    """Chỉ giữ lại ký tự chữ Hán trong chuỗi."""
    return "".join(_CJK_PATTERN.findall(text))


def _character_ngrams(text: str, gram_size: int = 3) -> set[str]:
    """Tập n-gram ký tự của chuỗi."""
    return {text[i : i + gram_size] for i in range(len(text) - gram_size + 1)}


def _srt_ngrams(srt_path: Path, gram_size: int = 3) -> set[str]:
    cues = parse_srt(srt_path)
    return _character_ngrams(_cjk_characters("".join(cue.text for cue in cues)), gram_size)


def _seraw_ngrams(
    seraw_path: Path,
    gram_size: int = 3,
    confidence_floor: float = 0.6,
    max_boxes: int = 60_000,
) -> set[str]:
    with seraw_path.open(encoding="utf-8") as handle:
        document = json.load(handle)
    collected: list[str] = []
    box_count = 0
    for frame in document.get("frames", []):
        for box in frame.get("boxes", []):
            if box.get("c", 0.0) >= confidence_floor:
                collected.append(_cjk_characters(box.get("t", "")))
                box_count += 1
        if box_count >= max_boxes:
            break
    return _character_ngrams("".join(collected), gram_size)


@dataclass(frozen=True, slots=True)
class PairingResult:
    """Kết quả ghép cặp một seraw với SRT khớp nhất.

    Attributes:
        seraw_path: File OCR thô.
        srt_path: File SRT khớp nhất.
        overlap_ratio: Tỷ lệ n-gram seraw nằm trong SRT (∈ [0, 1]).
        runner_up_ratio: Tỷ lệ của ứng viên xếp nhì (đo độ tách bạch).
    """

    seraw_path: Path
    srt_path: Path
    overlap_ratio: float
    runner_up_ratio: float

    @property
    def is_confident(self) -> bool:
        """Cặp đáng tin khi độ trùng cao và bỏ xa ứng viên nhì."""
        return self.overlap_ratio >= 0.40 and self.overlap_ratio >= 2.0 * self.runner_up_ratio


def auto_pair_seraw_to_srt(
    seraw_paths: list[Path],
    srt_paths: list[Path],
    gram_size: int = 3,
) -> list[PairingResult]:
    """Ghép mỗi seraw với SRT có độ trùng n-gram cao nhất.

    Args:
        seraw_paths: Danh sách file ``*_seraw.json``.
        srt_paths: Danh sách file ``*.srt`` ground-truth.
        gram_size: Cỡ n-gram ký tự dùng để so trùng.

    Returns:
        Danh sách :class:`PairingResult`, mỗi seraw một kết quả.

    Raises:
        PairingAmbiguousError: Khi không có SRT nào để ghép.
    """
    if not srt_paths:
        raise PairingAmbiguousError("Không có file SRT ground-truth để ghép cặp.")

    srt_ngram_cache = {path: _srt_ngrams(path, gram_size) for path in srt_paths}
    results: list[PairingResult] = []

    for seraw_path in seraw_paths:
        seraw_ngrams = _seraw_ngrams(seraw_path, gram_size)
        if not seraw_ngrams:
            logger.warning("seraw rỗng n-gram, bỏ qua: {}", seraw_path.name)
            continue
        scored = sorted(
            (
                (len(seraw_ngrams & srt_ngrams) / len(seraw_ngrams), srt_path)
                for srt_path, srt_ngrams in srt_ngram_cache.items()
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        best_ratio, best_path = scored[0]
        runner_up = scored[1][0] if len(scored) > 1 else 0.0
        results.append(
            PairingResult(
                seraw_path=seraw_path,
                srt_path=best_path,
                overlap_ratio=best_ratio,
                runner_up_ratio=runner_up,
            )
        )
        logger.info(
            "Ghép cặp {} → {} ({:.1%}, nhì {:.1%}){}",
            seraw_path.name,
            best_path.name,
            best_ratio,
            runner_up,
            "" if results[-1].is_confident else "  [KHÔNG đủ tin cậy]",
        )
    return results
