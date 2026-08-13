"""Port cho tính năng Text-to-Speech phụ đề."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


class TTSCancelledError(Exception): ...
class TTSUnavailableError(Exception): ...
class TTSGenerationError(Exception): ...


@dataclass(frozen=True)
class TTSRequest:
    """Yêu cầu TTS toàn bộ bộ phụ đề.

    Ánh xạ fields theo engine:
    * Edge TTS:   language=locale, speaker=voice name.
    * Gemini TTS: language=model name, speaker=voice name.
    * VieNeu-TTS: language='vi-VN', speaker=<preset> hoặc ref_audio_path (cloning).
    """
    events: list
    language: str = ""
    speaker: str = ""
    base_speed: float = 1.0
    max_speed: float = 3.0
    device: str = "auto"
    normalize: bool = True
    voice_clarity: bool = True      # Lọc bass rumble (<85Hz) + làm rõ giọng nói.
    high_quality: bool = True       # Ưu tiên chất lượng: nén CHỈ qua Edge API rate
                                    # (server-side, giữ pitch, chất lượng cao), KHÔNG
                                    # time-stretch client-side (tránh artifact rung/méo).
    target_lufs: float = -16.0      # Chuẩn loudness EBU R128/ITU-R BS.1770 cho master.
                                    # -16 LUFS: web/streaming; -23 LUFS: phim/TV phát sóng.
                                    # 0 = tắt LUFS (chỉ dùng RMS per-segment + soft-limit).
    gap_threshold_s: float = 0.3
    retry_count: int = 3
    retry_delay_s: float = 1.0
    # ── Xử lý Thời gian & Làm sạch ──────────────────────────────────────────
    clean_tags: bool = True         # Xoá HTML/ASS tags trước khi đọc
    dialog_pause_ms: int = 300      # Khoảng lặng (ms) khi dòng có dấu '-'
    max_overlap_ms: int = 500       # Cho phép audio lấn sang câu sau tối đa (ms)
    skip_overlap_ms: int = 0         # Bỏ qua nếu overlap > N ms; 0 = tắt (không bao giờ bỏ qua)
    double_pass: bool = True        # Bật thuật toán Double Pass để tính Speed chính xác
    # ── Elastic timing: dời câu sau (trong dung sai) thay vì nén tốc độ ───────
    elastic_timing: bool = True     # Tăng tốc là cuối cùng — ưu tiên dời câu sau
    max_drift_s: float = 2.5
    lead_in_s: float = 0.3          # "Ăn gian" đầu cụm: cụm quá tải được bắt đầu sớm
                                    # tối đa bao nhiêu giây vào khoảng lặng TRƯỚC nó
                                    # (không đè câu trước) → thêm thời gian, giảm nén.
    # ── Chiến lược phân cụm & co giãn (tầm nhìn vĩ mô) ───────────────────────
    anchor_gap_s: float = 0.7       # Khoảng lặng gốc ≥ ngưỡng này = neo đồng bộ cứng.
    max_segment_s: float = 10.0     # Cụm dài hơn ngưỡng này được chia tại khoảng lặng
                                    # lớn nhất bên trong (neo mềm) → nén cục bộ hơn.
    comfort_speed_ratio: float = 1.25  # Nén tới base×ratio "êm tai" → ưu tiên giữ đồng
                                       # bộ (drift 0) trước khi dời mốc thời gian.
    min_pause_ratio: float = 0.35   # Khoảng nghỉ hội thoại rút tối đa còn tỉ lệ này khi
                                    # quá tải (rút nghỉ KHÔNG méo giọng → ưu tiên dùng).
    max_intra_gap_s: float = 0.5    # Khoảng dừng TỐI ĐA giữa 2 câu liền trong cùng cụm.
                                    # Khi câu trước đọc xong sớm mà mốc gốc câu sau ở xa,
                                    # kéo câu sau lên để khoảng dừng không vượt ngưỡng này
                                    # → lời thoại liền mạch, tránh dừng lâu thiếu tự nhiên.
    allow_audio_overlap: bool = True  # Cho phép chồng tiếng TTS: câu sau neo đúng mốc
                                      # gốc (giữ lipsync) kể cả khi câu trước chưa dứt.
                                      # Phụ đề .srt xuất kèm LUÔN được cắt không chồng lấn.
    min_stretch_ratio: float = 0.75   # Giãn câu tối đa: đọc chậm nhất tới base×ratio để
                                      # lấp khoảng trống tới câu sau (giữ câu sau đúng mốc).
    timing_strategy: str = "lipsync"  # Chiến lược căn thời gian:
                                      #  "lipsync"  – bám CẢ mốc đầu & cuối từng câu, vượt
                                      #               max mới mượn khoảng lặng (đồng bộ nhất).
                                      #  "balanced" – bám mốc đầu, mượn khoảng lặng tới câu
                                      #               kế để đọc thong thả (cân bằng).
                                      #  "smooth"   – dồn mốc, KHÔNG chồng tiếng, giọng rõ
                                      #               từng câu, chấp nhận lệch mốc (mượt).        # Dung sai dời thời gian tích luỹ tối đa (giây)
    edge_concurrency: int = 16      # Số request Edge TTS song song (1-64). Cao = nhanh nhưng dễ rate-limit
    last_line_max_extend_s: float = 0.0  # Câu cuối kéo dài tối đa (giây) vào khoảng lặng cuối phim.
                                         # 0 = KHÔNG kéo dài (giữ end gốc) — tránh phát tiếng khi màn hình đã đen.
    # ── Bỏ đọc mô tả/ký hiệu khi TTS (SRT xuất kèm VẪN giữ nguyên) ───────────
    skip_paren: bool = True         # Bỏ nội dung trong ( )
    skip_square: bool = True        # Bỏ nội dung trong [ ]
    skip_curly: bool = True         # Bỏ nội dung trong { }
    skip_music_pair: bool = True    # Bỏ nội dung trong cặp ♪ ... ♪ (lời nhạc)
    skip_music_line: bool = True    # Bỏ đọc cả dòng bắt đầu bằng ♪ (dòng nhạc)
    # ── Voice cloning (VieNeu-TTS): nhân bản giọng từ audio tham chiếu ──────
    ref_audio_path: str = ""
    ref_text: str = ""
    # ── Định dạng & chất lượng file xuất ─────────────────────────────────────
    output_format: str = "wav"      # "wav" | "flac" | "mp3" | "ogg" | "opus" | "m4a"
                                    # wav/flac = lossless (debug chuẩn nhất); flac nhỏ
                                    # ~50% wav mà không mất chất lượng. mp3/ogg/opus/m4a
                                    # = nén có mất (nhỏ hơn nhiều, dễ gửi) — encode TRỰC
                                    # TIẾP từ master float32 nên không thêm khâu nhiễu.
    output_bitrate_kbps: int = 320  # Bitrate cho định dạng có mất (mp3/ogg/opus/m4a).
    wav_subtype: str = "PCM_16"     # Kiểu mẫu WAV/FLAC: PCM_16 | PCM_24 | FLOAT.
    # ── Gemini Native Audio Dialog ───────────────────────────────────────────
    style_prompt: str = ""
    affective_dialog: bool = True
    # [v3.23.246] Nhiệt độ lấy mẫu cho Gemini TTS. None = dùng mặc định của model. Hạ thấp
    # (vd 0.7) làm giọng ỔN ĐỊNH hơn, giảm "ngân dài ngẫu nhiên" (nguồn gốc hallucination
    # đo được ở v244); nhưng quá thấp có thể làm giọng bớt biểu cảm. Người dùng tự chỉnh.
    gemini_temperature: float | None = None
    # [v3.23.207] Thời lượng media gốc (giây) — file audio xuất ra sẽ dài ĐÚNG bằng
    # video để mux không lệch (trước: master = end câu cuối + đệm -> lệch vài giây).
    media_duration_s: float | None = None


@dataclass(frozen=True)
class TTSSegmentResult:
    """Kết quả TTS của một dòng phụ đề."""
    event_index: int
    start_sec: float
    end_sec: float
    text: str
    # [v3.23.77] Văn bản dành cho FILE PHỤ ĐỀ — GIỮ tag người nói "[Tên:]" (khác với
    # ``text`` vốn đã bỏ tag để đọc audio). Rỗng → bên xuất SRT fallback về ``text``.
    subtitle_text: str = ""
    audio_duration_s: float = 0.0
    speed_used: float = 1.0
    was_truncated: bool = False
    was_skipped: bool = False
    error_msg: str = ""
    # Timing thực tế sau elastic scheduling (-1 = không đổi so với gốc)
    adjusted_start_sec: float = -1.0
    adjusted_end_sec: float = -1.0
    # ── Trường DEBUG chi tiết (mặc định 0/—; điền ở khâu trộn để soi quá trình) ──
    pass1_dur_s: float = 0.0        # Độ dài giọng Pass 1 (đọc ở tốc độ cơ bản)
    pass2_dur_s: float = 0.0        # Độ dài giọng Pass 2 (nếu tái tạo để tăng tốc)
    used_pass2: bool = False        # Có dùng Pass 2 (Edge API đổi tốc độ) không
    scheduled_speed: float = 0.0    # Tốc độ scheduler dự kiến trước khi render
    api_speed: float = 0.0          # Tốc độ thực gửi Edge API (đã kẹp 0.5–3.0)
    window_strict_s: float = 0.0    # Khung gốc của câu (End − Start)
    window_ext_s: float = 0.0       # Khung mở rộng (tới mốc câu kế − khe hở)
    safe_window_s: float = 0.0      # Khung an toàn = ext + dung sai thị giác
    overlap_s: float = 0.0          # Lấn vượt khung an toàn (>0 mới là chồng thật)
    stretch_method: str = ""        # "" | "OLA" | "WSOLA" | "librosa"
    stretch_ratio: float = 1.0      # Tỉ lệ nén ở Max Squeeze (1.0 = không nén thêm)
    cut_amount_ms: float = 0.0      # Độ dài đuôi bị cắt (ms) nếu was_truncated
    pause_ms: float = 0.0           # Khoảng nghỉ hội thoại chèn vào (ms)
    is_dialog: bool = False         # Câu hội thoại (mở đầu bằng gạch ngang)
    ducked_prev_s: float = 0.0      # Độ dài đã dìm (ducking) lên câu liền trước (s)


TTSProgressCallback = Callable[[float, str], None]
TTSCancellationCallback = Callable[[], bool]


class SubtitleTTSPort(ABC):
    @abstractmethod
    def is_available(self) -> bool: ...
    @abstractmethod
    def get_engine_name(self) -> str: ...
    @abstractmethod
    def list_languages(self) -> list[str]: ...
    @abstractmethod
    def list_speakers(self, language: str) -> list[str]: ...
    @abstractmethod
    def generate(
        self, request: TTSRequest, output_path: Path,
        progress_cb: TTSProgressCallback | None = None,
        cancel_cb: TTSCancellationCallback | None = None,
    ) -> list[TTSSegmentResult]: ...


__all__ = [
    "TTSRequest", "TTSSegmentResult", "SubtitleTTSPort",
    "TTSCancelledError", "TTSUnavailableError", "TTSGenerationError",
    "TTSProgressCallback", "TTSCancellationCallback",
]
