"""Adapter Gemini TTS — TTS chất lượng cao dùng Google Gemini API.

Hỗ trợ 2 chế độ:
- **Standard TTS** (``generate_content``): stable, 1 API call/dòng.
- **Native Audio Dialog** (Live API WebSocket):
  tiếng nói tự nhiên hơn với prosody và cảm xúc thực sự.

Fix enableAffectiveDialog (github.com/googleapis/python-genai/issues/865):
  ``enable_affective_dialog`` chỉ hoạt động với:
  * ``api_version="v1alpha"`` trong client
  * Model ``gemini-2.5-flash-preview-native-audio-dialog``
  Các model khác (gemini-live-2.5-flash-native-audio, 3.1-flash-live-preview)
  KHÔNG hỗ trợ → tự động bỏ cờ này khi chọn model không tương thích.

Output: PCM-16 mono 24 000 Hz.
Cài đặt: pip install google-genai soundfile
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from subtitles_extractor.domain.ports.subtitle_tts_port import (
    SubtitleTTSPort,
    TTSCancellationCallback,
    TTSCancelledError,
    TTSProgressCallback,
    TTSRequest,
    TTSSegmentResult,
    TTSUnavailableError,
)
from subtitles_extractor.infrastructure.tts.audio_utils import (
    cap_nhat_ban_tot_nhat,
    is_effectively_silent,
    shorter_take,
    trim_edge_silence,
)
from subtitles_extractor.infrastructure.tts.edge_tts_adapter import (
    _preprocess_tts_text,
    _skip_from_request,
)
from subtitles_extractor.infrastructure.tts.text_prep import (
    dem_am_tiet,
    has_speakable_content,
    wrap_transcript_for_tts,
)
from subtitles_extractor.infrastructure.tts.timing_math import (
    GEMINI_MIN_BASE_S,
    GEMINI_MIN_PER_SYLLABLE_S,
    QUALITY_STRETCH_CAP,
    effective_available_seconds,
    exceeds_window_even_compressed,
    fit_limit_samples,
    is_abnormally_long_vs_floor,
    lead_in_seconds,
    master_length_samples,
    total_speed_ratio,
    window_below_engine_floor,
)

logger = logging.getLogger(__name__)

# ── Model catalog ─────────────────────────────────────────────────────────────

# Standard TTS: dùng generate_content API
_STANDARD_TTS_MODELS = [
    "gemini-2.5-flash-preview-tts",    # nhanh, rẻ — khuyến nghị cho subtitle dài
    "gemini-2.5-pro-preview-tts",      # chất lượng cao nhất
]

# Native Audio Dialog: dùng Live API (WebSocket) — giọng tự nhiên, cảm xúc
# Chú ý: enable_affective_dialog chỉ hoạt động với model đầu tiên + api_version=v1alpha
_NATIVE_AUDIO_MODELS = [
    "gemini-2.5-flash-native-audio-latest",              # ★ Recommended tracking alias
    "gemini-2.5-flash-preview-native-audio-dialog",      # Hỗ trợ affective dialog (v1alpha)
    "gemini-live-2.5-flash-native-audio",                # Stable pinned
    "gemini-2.5-flash-native-audio-preview-12-2025",     # Preview 12-2025
]

# Models hỗ trợ enable_affective_dialog (cần api_version=v1alpha)
_AFFECTIVE_DIALOG_SUPPORTED_MODELS = {
    "gemini-2.5-flash-preview-native-audio-dialog",
}

GEMINI_TTS_MODELS = _STANDARD_TTS_MODELS + _NATIVE_AUDIO_MODELS

# 30 voices (Gemini TTS, tháng 5/2026)
# [v3.23.249] 30 giọng CHÍNH THỨC theo tài liệu Google (GoogleCloudPlatform/generative-ai
# notebook). Phiên bản trước có 16/30 tên BỊA (Perseus, Electra, Polaris, Vega, Rigel,
# Deneb, Alula, Altair, Tethys, Dione, Ankaa, Izar, Albireo, Acamar, Aljanah, Seginus) —
# chọn phải sẽ lỗi API. Đồng thời thiếu 16 giọng thật (Zephyr, Algieba, Despina...). Nay
# đồng bộ đúng danh sách chính thức, sắp xếp theo bảng chữ cái.
GEMINI_TTS_VOICES = [
    "Achernar", "Achird", "Algenib", "Algieba", "Alnilam",
    "Aoede", "Autonoe", "Callirrhoe", "Charon", "Despina",
    "Enceladus", "Erinome", "Fenrir", "Gacrux", "Iapetus",
    "Kore", "Laomedeia", "Leda", "Orus", "Puck",
    "Pulcherrima", "Rasalgethi", "Sadachbia", "Sadaltager", "Schedar",
    "Sulafat", "Umbriel", "Vindemiatrix", "Zephyr", "Zubenelgenubi",
]

_SAMPLE_RATE = 24_000
# [v3.23.251] Ngôn ngữ đầu ra TTS. Tài liệu Live API khuyến nghị đặt language_code rõ ràng
# để giữ nhất quán — "without this definition, Gemini might alter the conversation language
# depending on the provided context". App đọc phụ đề tiếng Việt nên cố định vi-VN, tránh
# model đổi giọng/ngôn ngữ theo ngữ cảnh (vd gặp tên riêng CJK trong bản dịch).
_TTS_LANGUAGE_CODE = "vi-VN"
_NATIVE_CONCURRENCY = 10  # Sessions chạy song song (RPM/RPD Unlimited)
_TURN_TIMEOUT_S = 45      # Timeout tối đa cho mỗi dòng trong session


class GeminiTTSAdapter(SubtitleTTSPort):
    """TTS dùng Gemini API — Standard TTS + Native Audio Dialog."""

    def __init__(self, api_key: str) -> None:
        # [v3.23.130] Nếu nhận chuỗi NHIỀU key (mỗi dòng/phẩy một key), chỉ lấy key ĐẦU
        # tiên — tránh ký tự xuống dòng lọt vào header HTTP (LocalProtocolError).
        first = ""
        for piece in (api_key or "").replace(",", "\n").splitlines():
            if piece.strip():
                first = piece.strip()
                break
        self._api_key = first or (api_key or "").strip()

    def is_available(self) -> bool:
        """Kiểm tra PACKAGE đã cài chưa (không phụ thuộc API key).

        API key được kiểm tra riêng khi generate — màn hình chọn engine chỉ
        cần biết thư viện google-genai đã cài hay chưa.
        """
        try:
            from google import genai  # noqa: F401
            return True
        except ImportError:
            return False

    def get_engine_name(self) -> str:
        return "Gemini TTS"

    def list_languages(self) -> list[str]:
        return GEMINI_TTS_MODELS

    def list_speakers(self, language: str) -> list[str]:
        return GEMINI_TTS_VOICES

    # ── Client factory ────────────────────────────────────────────────────────

    def _make_client(self, *, use_v1alpha: bool = False):
        """Tạo Gemini client. Native Audio affective dialog yêu cầu v1alpha.

        [Native Audio Fallback] Tính năng Lồng tiếng Cảm xúc cần ``api_version=v1alpha``
        truyền qua ``http_options``. Thư viện ``google-genai`` bản CŨ chưa hỗ trợ tham
        số này → ``TypeError: unexpected keyword argument 'http_options'``. Bắt lỗi và
        lùi về Client mặc định để tương thích ngược, thay vì làm sập app.
        """
        from google import genai

        if use_v1alpha:
            try:
                return genai.Client(
                    api_key=self._api_key,
                    http_options={"api_version": "v1alpha"},
                )
            except TypeError:
                logger.warning(
                    "SDK google-genai bản cũ không hỗ trợ http_options (v1alpha) → "
                    "fallback Client mặc định; tính năng lồng tiếng cảm xúc có thể bị hạn chế."
                )
        return genai.Client(api_key=self._api_key)

    # ── Public generate ───────────────────────────────────────────────────────

    def generate(
        self,
        request: TTSRequest,
        output_path: Path,
        progress_cb: TTSProgressCallback | None = None,
        cancel_cb: TTSCancellationCallback | None = None,
    ) -> list[TTSSegmentResult]:
        if not self.is_available():
            raise TTSUnavailableError(
                "Gemini TTS chưa cài. Chạy: pip install google-genai soundfile"
            )
        if not self._api_key:
            raise TTSUnavailableError(
                "Gemini TTS cần API Key. Nhập API Key vào ô cấu hình Gemini."
            )

        model = request.language or _STANDARD_TTS_MODELS[0]
        voice = request.speaker or "Aoede"
        is_native = model in _NATIVE_AUDIO_MODELS
        sr = _SAMPLE_RATE

        mode_str = "Native Audio Dialog" if is_native else "Standard TTS"
        logger.info(
            "Gemini TTS [%s]: model=%s voice=%s events=%d",
            mode_str, model, voice, len(request.events),
        )
        if progress_cb:
            progress_cb(0.0, f"Khởi tạo Gemini TTS ({model})…")

        valid = [e for e in request.events if e.text.strip()]
        if not valid:
            return []

        last_end = max(e.end_sec for e in valid)
        # [v3.23.207] File xuất dài ĐÚNG thời lượng video (đồng bộ VieNeu) — mux không lệch.
        master = np.zeros(
            master_length_samples(
                last_end, sr, getattr(request, "media_duration_s", None), tail_pad_s=1.0
            ),
            dtype=np.float32,
        )

        if is_native:
            results = self._generate_native_audio(
                model=model, voice=voice, request=request,
                master=master, sr=sr,
                progress_cb=progress_cb, cancel_cb=cancel_cb,
            )
        else:
            try:
                client = self._make_client()
            except Exception as exc:
                raise TTSUnavailableError(f"Không khởi tạo Gemini client: {exc}") from exc
            results = self._generate_standard_tts(
                client=client, model=model, voice=voice,
                request=request, master=master, sr=sr,
                progress_cb=progress_cb, cancel_cb=cancel_cb,
            )

        # [4.1/4.2] Master phòng thu + xuất đa định dạng dùng module chung (đồng nhất
        # với Edge): làm rõ giọng → chuẩn LUFS → soft-limit → true-peak limiter.
        from subtitles_extractor.infrastructure.tts import audio_mastering as _am

        if request.normalize:
            if progress_cb:
                progress_cb(0.95, "Master âm thanh (LUFS + true-peak + làm rõ giọng)…")
            master = _am.master_finalize(
                master, sr,
                target_lufs=getattr(request, "target_lufs", -16.0),
                apply_clarity=getattr(request, "voice_clarity", True),
            )
        else:
            # [v3.23.209] Nén êm thay clip cứng (đồng bộ VieNeu): master cộng dồn chồng
            # tiếng vượt 1.0 -> clip phẳng gây méo ở đoạn chồng.
            master = _am.soft_limit(master)

        fmt = getattr(request, "output_format", "wav")
        if progress_cb:
            progress_cb(0.97, f"Đang ghi file {fmt.upper()}…")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        written = _am.write_audio(
            master, sr, output_path,
            fmt=fmt,
            subtype=getattr(request, "wav_subtype", "PCM_16"),
            bitrate_kbps=getattr(request, "output_bitrate_kbps", 320),
        )

        n_ok = sum(1 for r in results if not r.was_skipped)
        logger.info("Gemini TTS [%s] xong: %d/%d OK → %s",
                    mode_str, n_ok, len(results), written)
        # Giải phóng master lớn ngay (chống phình RAM khi render liên tục).
        del master
        import gc
        gc.collect()
        if progress_cb:
            progress_cb(1.0, "Hoàn tất!")
        return results

    # ── Standard TTS ──────────────────────────────────────────────────────────

    def _generate_standard_tts(
        self, *, client, model, voice, request, master, sr, progress_cb, cancel_cb
    ) -> list[TTSSegmentResult]:
        results = []
        n = len(request.events)
        # [v3.23.228] BUG PARITY: Gemini vốn KHÔNG dùng "ăn gian đầu" (lead-in) — cài đặt
        # người dùng bật ở UI bị phớt lờ, trong khi Edge và VieNeu đều tôn trọng. Hệ quả:
        # cùng một phụ đề, Gemini có khung hẹp hơn tới 0.25s -> nén giọng nhiều hơn và
        # chồng tiếng nhiều hơn hai engine kia. Nay theo dõi mốc kết thúc audio câu trước
        # để chỉ ăn gian vào KHOẢNG LẶNG THẬT (không đè lên câu trước).
        prev_audio_end = 0.0
        for idx, event in enumerate(request.events):
            if cancel_cb and cancel_cb():
                raise TTSCancelledError("Người dùng đã huỷ TTS.")
            if progress_cb:
                ratio = 0.03 + 0.94 * (idx / max(1, n))
                progress_cb(ratio, f"Dòng {idx + 1}/{n}: {event.text[:30]}…")
            # [v3.23.199] Gap-aware (đồng bộ VieNeu v194): tận dụng khoảng lặng tới
            # câu sau -> giảm tỉ lệ nén -> giọng ít gấp/méo (đo thực: nén mạnh -55%).
            next_start = (
                request.events[idx + 1].start_sec
                if idx + 1 < len(request.events)
                else getattr(request, "media_duration_s", None)
            )
            result = self._process_standard(
                event=event, idx=idx, master=master, next_start_sec=next_start,
                client=client, model=model, voice=voice,
                sr=sr, request=request, cancel_cb=cancel_cb,
                prev_audio_end_sec=prev_audio_end,
            )
            results.append(result)
            if not result.was_skipped and result.audio_duration_s > 0.0:
                prev_audio_end = (
                    result.adjusted_start_sec + result.audio_duration_s
                )
        return results


    @staticmethod
    def _time_stretch(audio: np.ndarray, sr: int, ratio: float) -> np.ndarray:
        """Co/giãn audio theo tỉ lệ mà KHÔNG đổi cao độ (giữ pitch).

        Dùng module dùng chung ``time_stretch_preserve_pitch`` (WSOLA/OLA, librosa nếu
        có). Trước đây khi thiếu librosa, hàm rơi về ``scipy.signal.resample`` làm méo
        cao độ → "giọng sóc chuột". Nay mọi nhánh đều giữ nguyên cao độ.

        Args:
            audio: mảng float32 mono.
            sr:    sample rate.
            ratio: > 1 = nhanh hơn (rút ngắn), < 1 = chậm hơn.

        Returns:
            Audio đã co giãn, cao độ giữ nguyên.
        """
        from subtitles_extractor.infrastructure.tts.time_stretch import (
            time_stretch_preserve_pitch,
        )

        return time_stretch_preserve_pitch(audio, sr, ratio)

    def _process_standard(
        self, *, event, idx, master, client, model, voice, sr, request, cancel_cb,
        next_start_sec: float | None = None,
        prev_audio_end_sec: float = 0.0,
    ) -> TTSSegmentResult:
        from google.genai import types
        # [v3.23.154] Chuẩn "văn bản âm thanh thật" (như Edge Pass 1): áp SKIP
        # ngoặc/nhạc theo cấu hình + BỎ tag người nói. Trước đây thiếu cả hai ->
        # Gemini TTS đọc TO "[Nam:]", "(cười)", nhạc ký hiệu; is_dialog cũng sai.
        text, is_dialog = _preprocess_tts_text(
            event.text, request.clean_tags, _skip_from_request(request),
            strip_speaker_tag=True,
        )
        start_sec, end_sec = event.start_sec, event.end_sec
        available = end_sec - start_sec
        # [v3.23.202] Cửa sổ ngắn không phải lý do bỏ khi cho phép chồng tiếng (đồng
        # bộ VieNeu): audio tràn tự nhiên -> giữ trọn câu thoại ngắn.
        window_too_short = (
            available < request.gap_threshold_s
            and not getattr(request, "allow_audio_overlap", True)
        )
        # [v3.23.211] Chỉ dấu câu/ký hiệu -> bỏ NGAY (đồng bộ VieNeu/Edge), tránh
        # retry vô ích 10 lần gọi API (tốn quota) rồi báo sai nguyên nhân.
        unspeakable = bool(text) and not has_speakable_content(text)
        if not text or unspeakable or window_too_short:
            return TTSSegmentResult(
                event_index=idx, start_sec=start_sec, end_sec=end_sec,
                text=text, was_skipped=True,
                error_msg=(
                    "Không có nội dung đọc được (chỉ dấu câu/ký hiệu)."
                    if unspeakable else "Cửa sổ quá ngắn / rỗng."
                ),
                adjusted_start_sec=start_sec, adjusted_end_sec=end_sec,
            )
        # [v3.23.251] language_code giữ nhất quán ngôn ngữ (xem hằng số). Bọc an toàn
        # cho SDK cũ không nhận field.
        try:
            std_speech_config = types.SpeechConfig(
                language_code=_TTS_LANGUAGE_CODE,
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
                ),
            )
        except (TypeError, ValueError):
            std_speech_config = types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
                )
            )
        config = types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=std_speech_config,
            # [v3.23.246] Nhiệt độ tuỳ chỉnh (None -> mặc định model). Hạ thấp giảm
            # hallucination "ngân dài ngẫu nhiên".
            temperature=getattr(request, "gemini_temperature", None),
        )
        audio = self._standard_with_retry(
            client, model, text, config, request, cancel_cb,
            # [v3.23.234] Khung THẬT của câu -> vòng retry phân biệt được "audio bất
            # thường so với nhịp đọc engine" với "audio không vừa khung phụ đề".
            available_s=effective_available_seconds(start_sec, end_sec, next_start_sec),
        )
        if audio is None:
            return TTSSegmentResult(
                event_index=idx, start_sec=start_sec, end_sec=end_sec,
                text=text, was_skipped=True,
                error_msg=f"Thất bại sau {request.retry_count} lần.",
                adjusted_start_sec=start_sec, adjusted_end_sec=end_sec,
            )
        # [v3.23.210] Cắt im lặng biên TRƯỚC pause/stretch (đồng bộ VieNeu v204): tiếng
        # vang đúng mốc phụ đề, im lặng đầu không đẩy tiếng tràn sang câu sau.
        audio = trim_edge_silence(audio, sr, adaptive=True)  # [v3.23.241] ngưỡng tự dò

        # [v3.23.217] Pause hội thoại KHÔNG bị nén và KHÔNG tính vào phần cần nén
        # (đồng bộ VieNeu): tính tỉ lệ trên riêng GIỌNG, khung dành cho giọng đã trừ
        # pause; pause chèn SAU khi nén -> giữ đúng độ dài người dùng đặt.
        pause_s = (
            request.dialog_pause_ms / 1000.0
            if (is_dialog and request.dialog_pause_ms > 0)
            else 0.0
        )
        voice_dur = len(audio) / sr
        speed_used = request.base_speed
        was_truncated = False

        # Time-stretch nếu audio dài hơn window (Gemini không có speed param)
        # [v3.23.199] Khung HIỆU DỤNG = khung gốc + gap tới câu sau (đồng bộ VieNeu
        # v194) -> giảm mạnh tỉ lệ nén cần thiết -> giọng tự nhiên hơn.
        base_available = effective_available_seconds(
            start_sec, end_sec, next_start_sec
        )
        # [v3.23.228] ĂN GIAN ĐẦU (lead-in) — đồng bộ VieNeu v219. Chỉ dời sớm những câu
        # THẬT SỰ cần khung rộng hơn (``needed_lead_s``); câu vốn đã vừa khung giữ NGUYÊN
        # mốc phụ đề, tránh đúng bug v218 (dời sớm cả loạt -> tiếng lệch trước khẩu hình).
        needed_window = voice_dur / max(1.0, float(request.base_speed)) + pause_s
        lead_s = lead_in_seconds(
            start_sec,
            prev_audio_end_sec,
            max_lead_s=float(getattr(request, "lead_in_s", 0.0)),
            needed_lead_s=max(0.0, needed_window - base_available),
        )
        play_start_sec = start_sec - lead_s
        effective_available = base_available + lead_s
        # [v3.23.215] Tốc độ TỔNG đúng ngữ nghĩa (đồng bộ VieNeu): audio luôn đọc TỐI
        # THIỂU ở base_speed (trước đây base chỉ là nhãn), nén thêm nếu chưa vừa khung;
        # chặn bởi max_speed + trần chất lượng 2.0.
        available_for_voice = max(0.05, effective_available - pause_s)
        capped_ratio = total_speed_ratio(
            voice_dur, available_for_voice, request.base_speed, request.max_speed
        )
        if capped_ratio > 1.0:
            logger.debug(
                "Gemini TTS time-stretch dòng %d: %.2fx (%.2fs → %.2fs)",
                idx, capped_ratio, voice_dur, available_for_voice,
            )
            stretched = self._time_stretch(audio, sr, capped_ratio)
            if len(stretched) > 0:
                audio = stretched
                speed_used = capped_ratio
        if pause_s > 0.0:
            audio = np.concatenate(
                (np.zeros(int(pause_s * sr), dtype=np.float32), audio)
            )
        audio_dur = len(audio) / sr

        overlap_s = audio_dur - effective_available
        if request.skip_overlap_ms > 0 and overlap_s > (request.skip_overlap_ms / 1000.0):
            return TTSSegmentResult(
                event_index=idx, start_sec=start_sec, end_sec=end_sec,
                text=text, was_skipped=True,
                error_msg=f"Overlap {overlap_s*1000:.0f}ms > skip {request.skip_overlap_ms}ms",
                adjusted_start_sec=start_sec, adjusted_end_sec=end_sec,
            )
        # [v3.23.198] Tôn trọng "Cho phép chồng tiếng" (đồng bộ fix VieNeu v197): bật
        # -> KHÔNG cắt, audio tràn tự nhiên vào master cộng dồn -> giữ trọn thoại.
        max_s = fit_limit_samples(
            effective_available, request.max_overlap_ms,
            getattr(request, "allow_audio_overlap", True), sr,
        )
        if max_s is not None and len(audio) > max_s:
            audio = audio[:max_s]; was_truncated = True
        ss = max(0, int(play_start_sec * sr)); es = min(ss + len(audio), len(master))
        if ss < len(master):
            master[ss:es] += audio[:es - ss]
        return TTSSegmentResult(
            event_index=idx, start_sec=start_sec, end_sec=end_sec,
            text=text, audio_duration_s=len(audio) / sr,
            speed_used=speed_used, was_truncated=was_truncated,
            # [v3.23.203] Chồng thật (đồng bộ Edge/VieNeu) -> UI đếm đúng.
            overlap_s=max(0.0, (len(audio) / sr) - effective_available),
            # [v3.23.228] adjusted = mốc PHÁT THẬT (đã trừ lead-in) — đồng bộ VieNeu:
            # SRT xuất ra khớp tiếng, và UI đếm "Dời mốc" đúng. Trước đây Gemini luôn ghi
            # mốc gốc vì không có lead-in.
            adjusted_start_sec=play_start_sec, adjusted_end_sec=end_sec,
        )

    def _standard_with_retry(
        self, client, model, text, config, request, cancel_cb, available_s: float = 0.0
    ):
        # [v3.23.221] Lưới "audio dài bất thường" (đồng bộ VieNeu): model neural thi
        # thoảng ngân dài một âm tiết -> nén kịch trần + đè câu sau. Giữ ứng viên ngắn
        # nhất thay vì vứt (audio dài VẪN là thoại hợp lệ).
        char_count = len(text.strip())
        overlong_best: np.ndarray | None = None
        # [v3.23.235] Dừng sớm khi lấy mẫu lại không còn ra bản ngắn hơn — xem
        # ``audio_utils.RESAMPLE_PATIENCE``.
        best_duration_s = float("inf")
        no_improve_streak = 0
        # [v3.23.227] Phân biệt retry do LỖI (cần backoff tăng dần vì có thể là 429/mạng)
        # với retry do CHẤT LƯỢNG AUDIO (call ĐÃ thành công, chỉ là model lấy mẫu tồi).
        # Khác VieNeu (offline, retry ngay không rủi ro), Gemini là API có rate limit ->
        # KHÔNG bỏ chờ hẳn, nhưng dùng delay CƠ BẢN thay vì nhân tăng dần: lỗi chất lượng
        # không phải tín hiệu quá tải, nhân delay chỉ làm chậm mà không giảm rủi ro nào.
        quality_retry = False
        for attempt in range(max(1, request.retry_count)):
            if cancel_cb and cancel_cb():
                raise TTSCancelledError("Huỷ trong retry Standard TTS.")
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=wrap_transcript_for_tts(text),
                    config=config,
                )
                # [v3.23.258] SDK google-genai 2.x TỰ decode base64 -> ``inline_data.data``
                # đã là BYTES audio thô (PCM 16-bit), KHÔNG phải chuỗi base64. Trước đây gọi
                # ``base64.b64decode`` LẦN NỮA -> double-decode -> audio rác/lỗi. Nay dùng
                # trực tiếp, đồng bộ với đường native audio (dòng ``chunks.append(idata.data)``).
                # Nội soi SDK 2.12.1: Blob.data là Optional[bytes], parser tự b64decode.
                raw = response.candidates[0].content.parts[0].inline_data.data
                audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                # [v3.23.210] Audio TOÀN IM LẶNG (có độ dài nhưng không tiếng) coi như
                # thất bại -> retry (đồng bộ VieNeu v205: lưới len>0 để lọt dạng lỗi
                # này -> mất thoại mà vẫn báo OK).
                if len(audio) > 0 and not is_effectively_silent(audio):
                    duration_s = len(audio) / _SAMPLE_RATE
                    ly_do = self._ly_do_lay_mau_lai(
                        duration_s, char_count, available_s, request,
                        syllable_count=dem_am_tiet(text),
                    )
                    if ly_do is None:
                        return audio
                    overlong_best = shorter_take(overlong_best, audio)
                    quality_retry = True
                    best_duration_s, no_improve_streak, nen_dung = cap_nhat_ban_tot_nhat(
                        best_duration_s, duration_s, no_improve_streak
                    )
                    if nen_dung:
                        logger.info(
                            "Khung câu quá hẹp so với nhịp đọc engine — %d lượt lấy mẫu "
                            "lại liên tiếp không ngắn hơn %.2fs. Dừng, dùng bản ngắn "
                            "nhất: '%s…'",
                            no_improve_streak, best_duration_s, text[:25],
                        )
                        break
                    logger.warning(
                        "Standard TTS lần %d/%d: %s (%.2fs cho %d ký tự, khung %.2fs) "
                        "— lấy mẫu lại: '%s…'",
                        attempt + 1, request.retry_count, ly_do, duration_s,
                        char_count, available_s, text[:25],
                    )
                else:
                    quality_retry = True
                    logger.warning(
                        "Standard TTS lần %d/%d: audio %s cho '%s…'",
                        attempt + 1, request.retry_count,
                        "rỗng" if len(audio) == 0 else "toàn im lặng", text[:25],
                    )
            except Exception as exc:
                quality_retry = False  # lỗi thật (429/mạng/SDK) -> backoff tăng dần
                logger.warning("Standard TTS lần %d/%d: %s", attempt + 1, request.retry_count, exc)
            if attempt < request.retry_count - 1:
                delay = (
                    request.retry_delay_s
                    if quality_retry
                    else request.retry_delay_s * (attempt + 1)
                )
                elapsed = 0.0
                while elapsed < delay:
                    if cancel_cb and cancel_cb():
                        raise TTSCancelledError("Huỷ khi chờ retry.")
                    time.sleep(0.1); elapsed += 0.1
        if overlong_best is not None:
            # Thà đọc lê thê còn hơn MẤT THOẠI.
            return overlong_best
        return None

    # ── Native Audio Dialog (Live API) ────────────────────────────────────────

    def _generate_native_audio(
        self, *, model, voice, request, master, sr, progress_cb, cancel_cb
    ) -> list[TTSSegmentResult]:
        """Xử lý Native Audio song song — RPM=Unlimited, RPD=Unlimited.

        Dùng asyncio.gather + Semaphore để chạy _NATIVE_CONCURRENCY sessions
        đồng thời. Mỗi session chỉ xử lý 1 dòng (không bao giờ 1011 Deadline).
        Tốc độ tăng ~_NATIVE_CONCURRENCY lần so với sequential.
        """
        n = len(request.events)
        results: list[TTSSegmentResult | None] = [None] * n
        use_affective = (
            request.affective_dialog and model in _AFFECTIVE_DIALOG_SUPPORTED_MODELS
        )

        valid_indices: list[int] = []
        for idx, event in enumerate(request.events):
            txt = event.text.strip()
            available = event.end_sec - event.start_sec
            window_too_short = (
                available < request.gap_threshold_s
                and not getattr(request, "allow_audio_overlap", True)
            )
            unspeakable = bool(txt) and not has_speakable_content(txt)
            if not txt or unspeakable or window_too_short:
                results[idx] = TTSSegmentResult(
                    event_index=idx, start_sec=event.start_sec, end_sec=event.end_sec,
                    text=txt, was_skipped=True,
                    error_msg=(
                        "Không có nội dung đọc được (chỉ dấu câu/ký hiệu)."
                        if unspeakable else "Cửa sổ quá ngắn / rỗng."
                    ),
                    adjusted_start_sec=event.start_sec, adjusted_end_sec=event.end_sec,
                )
            else:
                valid_indices.append(idx)

        n_valid = len(valid_indices)
        if n_valid == 0:
            return [r for r in results if r is not None]  # type: ignore

        logger.info(
            "Native Audio concurrent: %d dòng, concurrency=%d (~%.0fx nhanh hơn)",
            n_valid, _NATIVE_CONCURRENCY,
            min(_NATIVE_CONCURRENCY, n_valid),
        )

        # Chạy toàn bộ items song song trong 1 event loop
        coro = self._async_generate_concurrent(
            model=model, voice=voice, valid_indices=valid_indices,
            request=request, use_affective=use_affective,
            cancel_cb=cancel_cb, progress_cb=progress_cb, n_valid=n_valid,
        )
        try:
            audio_map: dict[int, np.ndarray | None] = asyncio.run(coro)
        except RuntimeError as exc:
            if "event loop" in str(exc).lower():
                loop = asyncio.new_event_loop()
                try:
                    audio_map = loop.run_until_complete(
                        self._async_generate_concurrent(
                            model=model, voice=voice, valid_indices=valid_indices,
                            request=request, use_affective=use_affective,
                            cancel_cb=cancel_cb, progress_cb=progress_cb, n_valid=n_valid,
                        )
                    )
                finally:
                    loop.close()
            else:
                raise

        # Ghi audio vào master WAV (sequential, thread-safe)
        # [v3.23.228] Ăn gian đầu — đồng bộ nhánh Standard và VieNeu.
        prev_audio_end = 0.0
        for idx in valid_indices:
            event = request.events[idx]
            audio_or_none = audio_map.get(idx)
            # [v3.23.154] Cùng chuẩn với văn bản ĐÃ tổng hợp -> pause khớp audio.
            text_clean, is_dialog = _preprocess_tts_text(
                event.text, request.clean_tags, _skip_from_request(request),
                strip_speaker_tag=True,
            )
            start_sec, end_sec = event.start_sec, event.end_sec
            available = end_sec - start_sec

            if audio_or_none is None:
                results[idx] = TTSSegmentResult(
                    event_index=idx, start_sec=start_sec, end_sec=end_sec,
                    text=text_clean, was_skipped=True,
                    error_msg="Thất bại sau tất cả lần retry.",
                    adjusted_start_sec=start_sec, adjusted_end_sec=end_sec,
                )
                continue

            audio = trim_edge_silence(audio_or_none, sr, adaptive=True)  # [v3.23.241]
            # [v3.23.217] Pause chèn SAU nén (đồng bộ VieNeu) — không bị bóp, không làm
            # giọng bị nén oan.
            pause_s = (
                request.dialog_pause_ms / 1000.0
                if (is_dialog and request.dialog_pause_ms > 0)
                else 0.0
            )
            voice_dur = len(audio) / sr
            speed_used = request.base_speed
            was_truncated = False

            # [v3.23.199] Gap-aware (đồng bộ VieNeu v194): khung hiệu dụng gồm gap tới
            # câu sau -> giảm tỉ lệ nén -> giọng tự nhiên hơn.
            next_start = (
                request.events[idx + 1].start_sec
                if idx + 1 < len(request.events)
                else getattr(request, "media_duration_s", None)
            )
            base_available = effective_available_seconds(
                start_sec, end_sec, next_start
            )
            # [v3.23.228] Chỉ dời sớm câu THẬT SỰ cần khung rộng hơn (bug v218: dời cả
            # loạt -> tiếng lệch trước khẩu hình).
            needed_window = voice_dur / max(1.0, float(request.base_speed)) + pause_s
            lead_s = lead_in_seconds(
                start_sec,
                prev_audio_end,
                max_lead_s=float(getattr(request, "lead_in_s", 0.0)),
                needed_lead_s=max(0.0, needed_window - base_available),
            )
            play_start_sec = start_sec - lead_s
            effective_available = base_available + lead_s
            available_for_voice = max(0.05, effective_available - pause_s)
            ratio = total_speed_ratio(
                voice_dur, available_for_voice, request.base_speed, request.max_speed
            )
            if ratio > 1.0:
                stretched = self._time_stretch(audio, sr, ratio)
                if len(stretched) > 0:
                    audio, speed_used = stretched, ratio
            if pause_s > 0.0:
                audio = np.concatenate(
                    (np.zeros(int(pause_s * sr), dtype=np.float32), audio)
                )
            audio_dur = len(audio) / sr

            overlap_s = audio_dur - effective_available
            if request.skip_overlap_ms > 0 and overlap_s > (request.skip_overlap_ms / 1000.0):
                results[idx] = TTSSegmentResult(
                    event_index=idx, start_sec=start_sec, end_sec=end_sec,
                    text=text_clean, was_skipped=True,
                    error_msg=f"Overlap {overlap_s*1000:.0f}ms",
                    adjusted_start_sec=start_sec, adjusted_end_sec=end_sec,
                )
                continue

            max_s = fit_limit_samples(
                effective_available, request.max_overlap_ms,
                getattr(request, "allow_audio_overlap", True), sr,
            )
            if max_s is not None and len(audio) > max_s:
                audio = audio[:max_s]; was_truncated = True

            ss = max(0, int(play_start_sec * sr)); es = min(ss + len(audio), len(master))
            if ss < len(master):
                master[ss:es] += audio[:es - ss]
            results[idx] = TTSSegmentResult(
                event_index=idx, start_sec=start_sec, end_sec=end_sec,
                text=text_clean, audio_duration_s=len(audio) / sr,
                speed_used=speed_used, was_truncated=was_truncated,
                overlap_s=max(0.0, (len(audio) / sr) - effective_available),
                # [v3.23.228] Mốc PHÁT THẬT (đã trừ lead-in) -> SRT khớp tiếng.
                adjusted_start_sec=play_start_sec, adjusted_end_sec=end_sec,
            )
            prev_audio_end = play_start_sec + len(audio) / sr

        for i, r in enumerate(results):
            if r is None:
                ev = request.events[i]
                results[i] = TTSSegmentResult(
                    event_index=i, start_sec=ev.start_sec, end_sec=ev.end_sec,
                    text=ev.text, was_skipped=True, error_msg="Không xử lý được.",
                    adjusted_start_sec=ev.start_sec, adjusted_end_sec=ev.end_sec,
                )
        return results  # type: ignore[return-value]

    async def _async_generate_concurrent(
        self, *, model, voice, valid_indices: list[int], request,
        use_affective: bool, cancel_cb, progress_cb, n_valid: int,
    ) -> dict[int, np.ndarray | None]:
        """asyncio.gather + Semaphore — chạy _NATIVE_CONCURRENCY sessions song song."""
        semaphore = asyncio.Semaphore(_NATIVE_CONCURRENCY)
        completed = 0
        audio_map: dict[int, np.ndarray | None] = {}
        lock = asyncio.Lock()

        async def process_one(idx: int) -> None:
            nonlocal completed
            event = request.events[idx]
            # [v3.23.154] Áp skip + bỏ tag người nói trước khi tổng hợp.
            text_clean, _ = _preprocess_tts_text(
                event.text, request.clean_tags, _skip_from_request(request),
                strip_speaker_tag=True,
            )

            async with semaphore:
                if cancel_cb and cancel_cb():
                    audio_map[idx] = None
                    return
                # [v3.23.234] Khung THẬT của câu -> vòng retry biết audio có vừa hay
                # không (khác hẳn "audio có bất thường so với nhịp đọc engine không").
                nxt = (
                    request.events[idx + 1].start_sec
                    if idx + 1 < len(request.events)
                    else None
                )
                audio = await self._async_native_single_with_retry(
                    model=model, voice=voice, text=text_clean,
                    request=request, use_affective=use_affective,
                    available_s=effective_available_seconds(
                        event.start_sec, event.end_sec, nxt
                    ),
                )

            async with lock:
                audio_map[idx] = audio
                completed += 1
                if progress_cb:
                    ratio = 0.03 + 0.90 * (completed / max(1, n_valid))
                    progress_cb(
                        ratio,
                        f"Đã xong {completed}/{n_valid} "
                        f"(x{min(_NATIVE_CONCURRENCY, n_valid)} song song)…"
                    )

        await asyncio.gather(*[process_one(idx) for idx in valid_indices], return_exceptions=True)
        return audio_map

    @staticmethod
    def _ly_do_lay_mau_lai(
        duration_s: float,
        char_count: int,
        available_s: float,
        request: Any,
        syllable_count: int = 0,
    ) -> str | None:
        """[v3.23.234] Có nên lấy mẫu lại câu này không, và VÌ SAO? (hàm thuần)

        Gộp HAI câu hỏi vốn khác nhau — nhầm lẫn giữa chúng đã gây hồi quy ở v232:

        1. **"Audio có bất thường so với NHỊP ĐỌC của Gemini không?"** -> hallucination
           (model ngân lê thê). Dùng mô hình độ dài riêng của Gemini.
        2. **"Audio có VỪA KHUNG PHỤ ĐỀ không?"** -> tràn khung. Không liên quan gì tới
           việc engine đọc nhanh hay chậm.

        Ở v232 tôi nới ngưỡng (1) cho khớp nhịp đọc chậm của Gemini và gọi 6 câu bị bắt là
        "retry oan". Nhưng những lần retry đó KHÔNG oan: ``shorter_take`` giữ bản ngắn
        nhất trong N lần, ép model đọc gọn cho vừa khung. Bỏ chúng đi -> lấn tăng từ
        **6.75s lên 18.61s** (đo trên FLAC Gemini thật). Lưới (1) bắt được chúng chỉ là
        ĂN MAY; thứ đáng hỏi là (2).

        Args:
            duration_s: Độ dài audio vừa sinh.
            char_count: Số ký tự văn bản.
            available_s: Khung thời gian câu này thực sự có (0 = không rõ -> bỏ qua (2)).
            request: Yêu cầu TTS (lấy ``max_speed``).

        Returns:
            Chuỗi mô tả lý do, hoặc None nếu audio dùng được.
        """
        # [v3.23.262] Đo hallucination THUẦN theo âm tiết (nhất quán với VieNeu). Text
        # không có âm tiết (rỗng/toàn dấu câu) -> KHÔNG thể "ngân dài" (không có âm để ngân)
        # -> bỏ qua kiểm hallucination. Trước đây fallback sang ký tự (R²=0.07) chỉ chạy khi
        # syllable_count=0, mà ca đó vốn không nên coi là bất thường -> bỏ nhánh ký tự.
        hallucination = syllable_count > 0 and is_abnormally_long_vs_floor(
            duration_s,
            syllable_count,
            min_base_s=GEMINI_MIN_BASE_S,
            min_per_syllable_s=GEMINI_MIN_PER_SYLLABLE_S,
        )
        if hallucination:
            return "audio DÀI BẤT THƯỜNG so với nhịp đọc Gemini"
        max_ratio = min(float(getattr(request, "max_speed", 2.0)), QUALITY_STRETCH_CAP)
        # [v3.23.237] Khung DƯỚI SÀN VẬT LÝ của engine -> không bản nào vừa được, kể cả
        # bản đọc một âm tiết. Lấy mẫu lại là vô nghĩa HOÀN TOÀN (không phải "khó").
        #
        # Sàn phải tính từ mô hình BIÊN DƯỚI THEO ÂM TIẾT, không phải mô hình trung bình
        # theo ký tự: sàn là "audio NGẮN NHẤT model sinh được", mà trung bình đã lẫn cả
        # phần model ngân dài ngẫu nhiên. v236 dùng nhầm mô hình trung bình -> sàn 0.45s,
        # CAO GẤP ĐÔI thực tế (0.23s) -> hai dòng (khung 0.40s và 0.44s) bị tước quyền lấy
        # mẫu lại oan dù vẫn còn cứu được.
        #
        # Sau khi sửa, chỉ còn 1/95 dòng thật sự dưới sàn: 沦为 (khung 0.20s) — lỗi nằm ở
        # TẦNG PHỤ ĐỀ (dòng quá vụn), TTS không cứu được.
        if window_below_engine_floor(
            available_s,
            max_ratio,
            min_base_s=GEMINI_MIN_BASE_S,
            min_per_syllable_s=GEMINI_MIN_PER_SYLLABLE_S,
        ):
            return None
        if exceeds_window_even_compressed(duration_s, available_s, max_ratio):
            return "audio TRÀN KHUNG dù đã nén hết cỡ"
        return None

    async def _async_native_single_with_retry(
        self, *, model, voice, text: str, request, use_affective: bool,
        available_s: float = 0.0,
    ) -> np.ndarray | None:
        """Fully async retry — asyncio.sleep thay time.sleep để không block event loop."""
        current_affective = use_affective
        # [v3.23.221] Lưới "audio dài bất thường" — đồng bộ với Standard TTS và VieNeu.
        char_count = len(text.strip())
        overlong_best: np.ndarray | None = None
        # [v3.23.235] Dừng sớm khi lấy mẫu lại không còn ra bản ngắn hơn — xem
        # ``audio_utils.RESAMPLE_PATIENCE``.
        best_duration_s = float("inf")
        no_improve_streak = 0
        quality_retry = False  # [v3.23.227] xem chú thích ở _standard_with_retry
        for attempt in range(max(1, request.retry_count)):
            try:
                audio = await self._async_native_single(
                    model=model, voice=voice, text=text,
                    request=request, use_affective=current_affective,
                )
                if audio is not None and len(audio) > 0:
                    duration_s = len(audio) / _SAMPLE_RATE
                    ly_do = self._ly_do_lay_mau_lai(
                        duration_s, char_count, available_s, request,
                        syllable_count=dem_am_tiet(text),
                    )
                    if ly_do is None:
                        return audio
                    overlong_best = shorter_take(overlong_best, audio)
                    quality_retry = True
                    best_duration_s, no_improve_streak, nen_dung = cap_nhat_ban_tot_nhat(
                        best_duration_s, duration_s, no_improve_streak
                    )
                    if nen_dung:
                        logger.info(
                            "Khung câu quá hẹp so với nhịp đọc engine — %d lượt lấy mẫu "
                            "lại liên tiếp không ngắn hơn %.2fs. Dừng, dùng bản ngắn "
                            "nhất: '%s…'",
                            no_improve_streak, best_duration_s, text[:25],
                        )
                        break
                    logger.warning(
                        "Native Audio lần %d/%d: %s (%.2fs cho %d ký tự, khung %.2fs) "
                        "— lấy mẫu lại: '%s…'",
                        attempt + 1, request.retry_count, ly_do, duration_s,
                        char_count, available_s, text[:25],
                    )
                else:
                    # [v3.23.249] Audio RỖNG (model từ chối đọc) — hay gặp với câu cực
                    # ngắn ("Ơ.", "Hả?") ở temperature thấp. Log rõ để chẩn đoán thay vì
                    # âm thầm retry. Thử tắt affective từ lần sau: câu ngắn đôi khi bị
                    # affective làm model "diễn" quá rồi trả rỗng.
                    logger.warning(
                        "Native Audio lần %d/%d: audio RỖNG cho '%s…' — thử lại%s.",
                        attempt + 1, request.retry_count, text[:25],
                        " (tắt affective)" if current_affective else "",
                    )
                    current_affective = False
            except Exception as exc:
                exc_str = str(exc)
                if "enableAffectiveDialog" in exc_str or (
                    "affective" in exc_str.lower() and "1007" in exc_str
                ):
                    current_affective = False
                    try:
                        audio = await self._async_native_single(
                            model=model, voice=voice, text=text,
                            request=request, use_affective=False,
                        )
                        if audio is not None and len(audio) > 0:
                            return audio
                    except Exception as retry_exc:
                        # Fallback (tắt affective) cũng hỏng → ghi log để chẩn đoán,
                        # rồi để vòng lặp retry ngoài xử lý tiếp.
                        logger.warning(
                            "TTS fallback (affective=False) thất bại: %s", retry_exc
                        )
                elif "1011" in exc_str or "Deadline expired" in exc_str:
                    # [v3.23.247] 1011 Deadline là lỗi TẠM THỜI (server chậm/quá tải phản
                    # hồi), KHÔNG phải lỗi vĩnh viễn. Trước đây `return None` ngay -> bỏ
                    # luôn dòng -> MẤT TIẾNG. Nay để vòng lặp retry lần sau (rất thường
                    # thành công). Chỉ bỏ khi đã cạn lượt (xử lý sau vòng for).
                    logger.warning(
                        "1011 Deadline lần %d/%d dòng '%s…' — thử lại.",
                        attempt + 1, request.retry_count, text[:25],
                    )
                else:
                    logger.warning(
                        "Native Audio retry %d/%d: %s", attempt + 1, request.retry_count, exc
                    )
            if attempt < request.retry_count - 1:
                await asyncio.sleep(
                    request.retry_delay_s
                    if quality_retry
                    else request.retry_delay_s * (attempt + 1)
                )
                quality_retry = False
        if overlong_best is not None:
            return overlong_best
        return None

    async def _async_native_single(
        self, *, model, voice, text: str, request, use_affective: bool
    ) -> np.ndarray | None:
        """1 Live API session = 1 dòng phụ đề. Session ngắn giảm mạnh nguy cơ 1011.

        [v3.23.247] Vẫn có thể gặp 1011 Deadline khi server Gemini quá tải (lỗi 503
        UNAVAILABLE tạm thời, không phải do session dài) — trường hợp đó được retry ở
        ``_async_native_single_with_retry`` thay vì bỏ dòng.
        """
        from google.genai import types

        style = request.style_prompt.strip()
        if not style:
            # [v3.23.251] Style MẶC ĐỊNH kèm lớp bảo vệ ngôn ngữ THỨ HAI (cùng language_code
            # ở SpeechConfig). Tài liệu Live API khuyến nghị nêu rõ ngôn ngữ trong system
            # instruction để model không đổi ngôn ngữ theo ngữ cảnh (vd tên riêng CJK còn
            # sót trong bản dịch). CHỈ áp cho style mặc định — nếu người dùng tự đặt style
            # (có thể cho ngôn ngữ khác), tôn trọng lựa chọn của họ, không ép tiếng Việt.
            style = (
                "Đọc phụ đề phim một cách tự nhiên, truyền cảm, "
                "phù hợp với cảm xúc và ngữ cảnh của từng dòng. "
                "Luôn đọc bằng tiếng Việt."
            )
        # [v3.23.251] Thử đặt language_code (giữ nhất quán ngôn ngữ). SDK cũ có thể không
        # nhận field này trong SpeechConfig -> fallback không có language_code.
        try:
            speech_config = types.SpeechConfig(
                language_code=_TTS_LANGUAGE_CODE,
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
                ),
            )
        except (TypeError, ValueError):
            speech_config = types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
                )
            )
        config_kwargs: dict = {
            "response_modalities": ["AUDIO"],
            "speech_config": speech_config,
            "system_instruction": types.Content(parts=[types.Part(text=style)]),
        }
        # [v3.23.246] Nhiệt độ tuỳ chỉnh — chỉ thêm khi người dùng đặt (tránh gửi None cho
        # SDK cũ). Hạ thấp giảm hallucination "ngân dài ngẫu nhiên" (đo được ở v244).
        temperature = getattr(request, "gemini_temperature", None)
        if temperature is not None:
            config_kwargs["temperature"] = temperature
        if use_affective and model in _AFFECTIVE_DIALOG_SUPPORTED_MODELS:
            config_kwargs["enable_affective_dialog"] = True
            client = self._make_client(use_v1alpha=True)
        else:
            client = self._make_client(use_v1alpha=False)

        config = types.LiveConnectConfig(**config_kwargs)
        chunks: list[bytes] = []

        async with client.aio.live.connect(model=model, config=config) as session:
            await session.send_client_content(
                turns=types.Content(
                    parts=[types.Part(text=wrap_transcript_for_tts(text))],
                    role="user",
                ),
                turn_complete=True,
            )
            try:
                async with asyncio.timeout(_TURN_TIMEOUT_S):
                    async for response in session.receive():
                        sc = getattr(response, "server_content", None)
                        if sc:
                            mt = getattr(sc, "model_turn", None)
                            if mt:
                                for part in getattr(mt, "parts", []):
                                    idata = getattr(part, "inline_data", None)
                                    if idata and idata.data:
                                        chunks.append(idata.data)
                            if getattr(sc, "turn_complete", False):
                                break
            except TimeoutError:
                logger.warning("Timeout %ds dòng '%s…'.", _TURN_TIMEOUT_S, text[:25])

        if not chunks:
            return None
        raw = b"".join(chunks)
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        # [v3.23.210] Im lặng toàn phần = thất bại (đồng bộ VieNeu v205).
        if len(audio) == 0 or is_effectively_silent(audio):
            return None
        return audio


__all__ = [
    "GEMINI_TTS_MODELS",
    "GEMINI_TTS_VOICES",
    "_AFFECTIVE_DIALOG_SUPPORTED_MODELS",
    "_NATIVE_AUDIO_MODELS",
    "_STANDARD_TTS_MODELS",
    "GeminiTTSAdapter",
]
