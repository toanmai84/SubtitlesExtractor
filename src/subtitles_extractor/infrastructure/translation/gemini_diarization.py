"""[v3.23.623] Tách giọng nói (diarization) bằng ``gemini-3.5-transcribe`` → nhãn giọng cho từng dòng.

Lỗi dịch nặng nhất còn lại là ĐẢO VAI (ai nói với ai). Chỉ đọc chữ thì cảnh mơ hồ bị đọc sai
một cách HỆ THỐNG (bỏ phiếu nhiều lượt không sửa được: 5 lượt cùng ra ~91% ở tập 2). Âm thanh
là tín hiệu độc lập với ngữ nghĩa. Đo thật trên tập mẫu (đáp án gán tay theo dòng):

* ``gemini-3.5-transcribe`` tách giọng: tập 2 cụm ``spk:0`` = Diệp Nhiên 77/77 dòng, ``spk:2``
  = Nam Cung 28/28, ``spk:3`` = hệ thống 24/24 (cả các dòng 3-9 mà đọc chữ cứ gán nhầm).
  8-11 giây, 4-6k token vào, MIỄN PHÍ ở hạng miễn phí.
* Đưa nhãn giọng cho model văn bản rẻ (3.5-flash-lite) chỉ việc gán MỖI nhãn cho một nhân vật:
  độ đúng người nói **97-99% (tập 1) và 99-100% (tập 2)**, ổn định giữa các lượt, vượt cả
  phân tích kèm video của model Flash đầy đủ.

Mô-đun gồm hàm thuần (đọc phản hồi, ghép từ → dòng) và phần gọi API/ffmpeg tách riêng để kiểm thử.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import logging
import re
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DIARIZATION_MODEL = "gemini-3.5-transcribe"
#: Tài liệu transcribe.md: bật diarization/mốc từ thì âm thanh tối đa 30 phút.
MAX_DIARIZED_AUDIO_SEC = 28 * 60
#: Nới mốc dòng ± chừng này giây khi ghép từ nói vào dòng (lời nói lệch phụ đề cứng chút ít).
VOICE_PAD_S = 0.3

_LANGUAGE_CODES = {
    "zh": "cmn-Hans-CN", "zh-cn": "cmn-Hans-CN", "ja": "ja-JP", "ko": "ko-KR", "en": "en-US",
    "vi": "vi-VN", "th": "th-TH", "es": "es-ES", "fr": "fr-FR", "de": "de-DE",
}


@dataclass(frozen=True, slots=True)
class VoiceWord:
    """Một từ do bộ nhận dạng giọng nói tách ra.

    Attributes:
        speaker: Nhãn cụm giọng (đã rút gọn, vd ``"0"``).
        start: Giây bắt đầu (tính từ đầu âm thanh).
        end: Giây kết thúc.
    """

    speaker: str
    start: float
    end: float


def language_codes_for(source_lang: str) -> list[str]:
    """Mã ngôn ngữ BCP-47 gợi ý cho bộ nhận dạng (rỗng = tự nhận diện)."""
    code = _LANGUAGE_CODES.get((source_lang or "").strip().lower())
    return [code] if code else []


def _short_label(label: str) -> str:
    """``"spk:2"``/``"spk_2"`` → ``"2"``; không có số thì giữ nguyên chữ."""
    text = (label or "").strip()
    match = re.search(r"(\d+)\s*$", text)
    return match.group(1) if match else text


def parse_offset(value: Any) -> float | None:
    """Mốc dạng ``"1.700s"`` (hoặc số) → giây; ``None`` nếu không đọc được."""
    if value in (None, ""):
        return None
    try:
        return float(str(value).strip().rstrip("sS"))
    except ValueError:
        return None


def words_from_response(response: Any) -> list[VoiceWord]:
    """Đọc mọi từ kèm nhãn giọng từ phản hồi ``generate_content`` của model transcribe."""
    words: list[VoiceWord] = []
    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            transcription = getattr(part, "audio_transcription", None)
            if not transcription:
                continue
            speaker = _short_label(getattr(transcription, "speaker_label", "") or "")
            for item in getattr(transcription, "words", None) or []:
                start = parse_offset(getattr(item, "start_offset", None))
                end = parse_offset(getattr(item, "end_offset", None))
                if speaker and start is not None:
                    words.append(VoiceWord(speaker, start, end if end is not None else start))
    words.sort(key=lambda w: w.start)
    return words


def voice_labels_from_words(
    words: Sequence[VoiceWord], lines: Sequence[Any], pad_s: float = VOICE_PAD_S
) -> dict[int, str]:
    """Nhãn giọng của từng dòng: cụm có TỔNG thời lượng chồng lấn lớn nhất (hàm thuần).

    Args:
        words: Từ đã tách giọng (đã sắp theo ``start``).
        lines: Dòng phụ đề có ``index``, ``start_ms``, ``end_ms``.
        pad_s: Nới mốc dòng hai phía.

    Returns:
        ``{số dòng: nhãn}``; dòng không có từ nào chồng lấn thì KHÔNG có mặt.
    """
    if not words:
        return {}
    starts = [w.start for w in words]
    longest = max((w.end - w.start for w in words), default=0.0)
    labels: dict[int, str] = {}
    for line in lines:
        lo = line.start_ms / 1000.0 - pad_s
        hi = line.end_ms / 1000.0 + pad_s
        first = bisect.bisect_left(starts, lo - longest)
        last = bisect.bisect_right(starts, hi)
        totals: dict[str, float] = {}
        for word in words[first:last]:
            overlap = min(hi, word.end) - max(lo, word.start)
            if overlap > 0:
                totals[word.speaker] = totals.get(word.speaker, 0.0) + overlap
        if totals:
            labels[line.index] = max(totals, key=totals.__getitem__)
    return labels


def _cache_path(video_path: Path) -> Path:
    stat = video_path.stat()
    digest = hashlib.sha1(
        f"{video_path.resolve()}|{stat.st_size}|{int(stat.st_mtime)}".encode()
    ).hexdigest()[:20]
    return Path(tempfile.gettempdir()) / "subtitles_extractor_voice" / f"{digest}.json"


def load_cached_words(video_path: Path) -> list[VoiceWord] | None:
    """Từ đã tách giọng của video này (lần trước), hoặc ``None``."""
    try:
        raw = json.loads(_cache_path(video_path).read_text(encoding="utf-8"))
        return [VoiceWord(str(w[0]), float(w[1]), float(w[2])) for w in raw]
    except (OSError, ValueError, TypeError, IndexError):
        return None


def save_cached_words(video_path: Path, words: Sequence[VoiceWord]) -> None:
    """Lưu từ đã tách giọng (lỗi ghi bị bỏ qua — chỉ là bộ nhớ đệm)."""
    try:
        path = _cache_path(video_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps([[w.speaker, w.start, w.end] for w in words]), encoding="utf-8")
    except OSError as exc:
        logger.debug("Không lưu được bộ nhớ đệm tách giọng: %s", exc)


def extract_dialogue_audio(video_path: Path, out_path: Path, timeout_s: float = 300.0) -> bool:
    """Cắt riêng âm thanh lời thoại (mono 16 kHz AAC) từ video. ``False`` nếu không được."""
    from subtitles_extractor.infrastructure.media import find_ffmpeg
    from subtitles_extractor.infrastructure.process.hidden_process import run_ffmpeg_hardened
    from subtitles_extractor.infrastructure.translation.context_audio import (
        context_audio_args,
        select_context_audio,
    )
    from subtitles_extractor.infrastructure.translation.gemini_video_context import (
        GeminiVideoContextProvider,
    )

    try:
        ffmpeg = find_ffmpeg()
        streams = GeminiVideoContextProvider._probe_audio_streams(video_path)
        audio = select_context_audio(streams)
        if audio is None:
            return False
        command = [
            ffmpeg, "-nostdin", "-v", "error", "-y", "-i", str(video_path), "-vn",
            *context_audio_args(audio), str(out_path),
        ]
        run_ffmpeg_hardened(command, timeout=timeout_s, check=True)
    except Exception as exc:  # noqa: BLE001 — ffmpeg/PyAV lỗi bất kỳ: tính năng tuỳ chọn
        logger.warning("Không tách được âm thanh để nhận dạng giọng: %s", exc)
        return False
    return out_path.is_file() and out_path.stat().st_size > 1024


def diarize_audio(
    client: Any, types: Any, audio_path: Path, language_codes: list[str],
    sleep: Callable[[float], None] = time.sleep,
) -> list[VoiceWord]:
    """Tải âm thanh lên, gọi model transcribe có diarization, trả danh sách từ."""
    uploaded = client.files.upload(file=str(audio_path))
    try:
        for _ in range(60):
            state = getattr(getattr(uploaded, "state", None), "name", str(uploaded.state))
            if state != "PROCESSING":
                break
            sleep(2.0)
            uploaded = client.files.get(name=uploaded.name)
        response = client.models.generate_content(
            model=DIARIZATION_MODEL,
            contents=[uploaded],
            config=types.GenerateContentConfig(
                audio_transcription_config=types.AudioTranscriptionConfig(
                    language_codes=language_codes, diarization=True, word_timestamp=True,
                )
            ),
        )
        return words_from_response(response)
    finally:
        try:
            client.files.delete(name=uploaded.name)
        except Exception as exc:  # noqa: BLE001 — dọn dẹp tuỳ chọn, tệp tự hết hạn sau 48 giờ
            logger.debug("Không xoá được tệp âm thanh tạm: %s", exc)


__all__ = [
    "DIARIZATION_MODEL",
    "MAX_DIARIZED_AUDIO_SEC",
    "VoiceWord",
    "diarize_audio",
    "extract_dialogue_audio",
    "language_codes_for",
    "load_cached_words",
    "parse_offset",
    "save_cached_words",
    "voice_labels_from_words",
    "words_from_response",
]
