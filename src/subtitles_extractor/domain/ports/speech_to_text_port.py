"""Hợp đồng nhận dạng giọng nói thành phụ đề (Speech-to-Text).

Adapter (vd WhisperX) nhận một tệp audio/video, phiên âm thành các câu phụ đề có
mốc thời gian. Đầu ra là ``list[SubtitleEvent]`` — đồng nhất với mọi nguồn khác
(hardsub OCR, phụ đề nhúng) để tái dùng editor/export/translate/TTS.

Engine STT thường là phụ thuộc nặng (model GPU), nên port tách ``is_available()``
để UI biết có dùng được không trước khi gọi (giống các TTS adapter).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol, runtime_checkable

from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent

# ``(current, total, message)`` — tiến độ phiên âm cho UI.
TranscriptionProgressCallback = Callable[[int, int, str], None]


@dataclass(frozen=True, slots=True)
class TranscriptionConfig:
    """Cấu hình phiên âm.

    Attributes:
        language:       Mã ngôn ngữ ISO ("vi", "en", "zh"…) hoặc "" để tự nhận diện.
        model_size:     Kích thước model ("tiny", "base", "small", "medium",
                        "large-v3"…). Lớn hơn = chính xác hơn nhưng nặng hơn.
        device:         "cuda" hoặc "cpu".
        compute_type:   Kiểu tính toán WhisperX ("float16", "int8"…).
        batch_size:     Số đoạn xử lý song song.
        enable_align:   Bật alignment cấp từ (timestamp chính xác hơn).
        enable_diarize: Bật phân tách người nói (cần token HuggingFace).
        hf_token:       Token HuggingFace cho diarization (rỗng nếu không dùng).
    """

    language: str = ""
    model_size: str = "small"
    device: str = "cuda"
    compute_type: str = "float16"
    batch_size: int = 16
    enable_align: bool = True
    align_device: str = "cpu"  # [v3.23.6] align trên CPU tránh xung đột cuDNN GPU
    enable_diarize: bool = False
    hf_token: str = ""
    # [v3.22.6] Tách câu: WhisperX (NLTK Punkt) KHÔNG tách được câu CJK → segment
    # dài 20-30s nhồi nhiều câu. Ta tự tách theo word-timestamps:
    #   * gặp dấu câu kết thúc (。！？.!?…) → ngắt câu;
    #   * khoảng lặng giữa 2 từ ≥ split_gap_sec → ngắt câu;
    #   * câu vượt max_chars_per_cue ký tự → ngắt mềm tại dấu phẩy gần nhất.
    enable_sentence_split: bool = True
    split_gap_sec: float = 0.3
    target_chars_per_cue: int = 4
    max_chars_per_cue: int = 8
    max_cue_duration_sec: float = 4.0
    use_word_segmentation: bool = True  # [v3.23] jieba phân từ CJK khi tách câu
    filter_hallucinations: bool = True  # [v3.23] lọc câu ảo giác của Whisper


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    """Kết quả phiên âm.

    Attributes:
        events:            Danh sách câu phụ đề có mốc thời gian.
        detected_language: Ngôn ngữ engine nhận diện được (nếu auto).
        raw_segments:      Dữ liệu THÔ từ WhisperX (segments + words + timestamp)
                           để xuất ra file phục vụ hiệu chuẩn thuật toán tách câu.
    """

    events: list[SubtitleEvent] = field(default_factory=list)
    detected_language: str = ""
    raw_segments: list[dict] = field(default_factory=list)


@runtime_checkable
class SpeechToTextPort(Protocol):
    """Phiên âm giọng nói trong audio/video thành phụ đề."""

    def is_available(self) -> bool:
        """Trả ``True`` nếu engine (và phụ thuộc) đã sẵn sàng để dùng."""
        ...

    def get_engine_name(self) -> str:
        """Tên engine để hiển thị (vd ``"WhisperX"``)."""
        ...

    def transcribe(
        self,
        media_path: Path,
        config: TranscriptionConfig,
        progress_callback: TranscriptionProgressCallback | None = None,
        cancellation_check: Callable[[], bool] | None = None,
    ) -> TranscriptionResult:
        """Phiên âm media thành phụ đề.

        Args:
            media_path:         Đường dẫn audio hoặc video.
            config:             Cấu hình phiên âm.
            progress_callback:  ``(current, total, message)`` cho UI.
            cancellation_check: Trả ``True`` để dừng sớm.

        Returns:
            Kết quả phiên âm (events + ngôn ngữ nhận diện).

        Raises:
            SpeechToTextError: Khi engine không sẵn sàng hoặc lỗi phiên âm.
        """
        ...


__all__ = [
    "TranscriptionConfig",
    "TranscriptionResult",
    "TranscriptionProgressCallback",
    "SpeechToTextPort",
]
