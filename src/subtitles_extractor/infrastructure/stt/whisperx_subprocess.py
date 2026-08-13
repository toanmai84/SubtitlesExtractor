"""Script chạy WhisperX trong TIẾN TRÌNH CON RIÊNG BIỆT (process isolation).

LÝ DO TỒN TẠI: PaddlePaddle-GPU và PyTorch-CUDA cùng nạp cuDNN/cuBLAS DLL trong MỘT
tiến trình sẽ xung đột symbol trên Windows (vd "_CudaDeviceProperties is already
registered", "cublas64 not configured correctly", "partially initialized module
paddle"). Đây là xung đột nhị phân nổi tiếng, không sửa được ở mức Python.

GIẢI PHÁP: chạy WhisperX (torch) ở tiến trình con TÁCH BIỆT. Tiến trình chính chỉ
nạp PaddleOCR (paddle); tiến trình con chỉ nạp torch/whisperx. Hai process không
chia sẻ bộ nhớ → không xung đột DLL. GPU vẫn dùng được ở cả hai (process riêng).

GIAO TIẾP:
  * Đầu vào: đường dẫn file WAV 16kHz mono + JSON config qua argv.
  * Tiến độ: in dòng ``PROGRESS <cur> <total> <msg>`` ra stderr (process cha đọc).
  * Kết quả: ghi JSON {segments, language} ra file output chỉ định.
  * Lỗi: in ``ERROR <message>`` ra stderr + exit code khác 0.

Script này KHÔNG import gì từ ``subtitles_extractor`` để giữ độc lập tối đa; chỉ
phụ thuộc whisperx + numpy + wave (chuẩn).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import wave


def _eprint(message: str) -> None:
    """In ra stderr + flush ngay để process cha đọc kịp thời."""
    sys.stderr.write(message + "\n")
    sys.stderr.flush()


def _progress(current: int, total: int, message: str) -> None:
    _eprint(f"PROGRESS {current} {total} {message}")


def _load_wav_as_float32(audio_path: str):
    import numpy as np

    with wave.open(audio_path, "rb") as wav_file:
        frame_count = wav_file.getnframes()
        raw_bytes = wav_file.readframes(frame_count)
    samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)
    return samples / 32768.0


def _resolve_device_and_compute(requested_device: str, requested_compute: str):
    device = requested_device
    compute_type = requested_compute
    if requested_device == "cuda":
        cuda_ok = False
        try:
            import torch

            cuda_ok = bool(torch.cuda.is_available())
        except (ImportError, AssertionError, RuntimeError):
            cuda_ok = False
        if not cuda_ok:
            _eprint("WARN CUDA không khả dụng → dùng CPU.")
            device = "cpu"
    if device == "cpu" and compute_type in ("float16", "fp16", "half"):
        compute_type = "int8"
    return device, compute_type


def _load_whisperx_model(
    whisperx, model_size, device, compute_type, language, asr_options, dlog
):
    """[v3.23.364] Nạp model WhisperX bền vững qua nhiều biến thể môi trường.

    Thứ tự thử: (1) tên model + ``asr_options`` chống lặp; (2) nếu bản WhisperX cũ KHÔNG
    nhận ``asr_options`` (TypeError) → nạp mặc định; (3) nếu là 'turbo' mà tên tắt chưa
    được nhận (ValueError/OSError) → thử repo CTranslate2 chuẩn của cộng đồng. Trả về
    model đã nạp hoặc ném lỗi cuối cùng.
    """

    def _try_load(arch: str):
        try:
            return whisperx.load_model(
                arch, device, compute_type=compute_type,
                language=language, asr_options=asr_options,
            )
        except TypeError:
            # Bản WhisperX cũ: chữ ký không có asr_options → nạp mặc định.
            dlog("[WARN] load_model khong nhan asr_options; nap mac dinh.")
            return whisperx.load_model(
                arch, device, compute_type=compute_type, language=language,
            )

    try:
        return _try_load(model_size)
    except (ValueError, OSError, RuntimeError) as load_exc:
        if "turbo" in str(model_size):
            fallback_repo = "deepdml/faster-whisper-large-v3-turbo-ct2"
            dlog(f"[WARN] load '{model_size}' lỗi ({load_exc}); thử '{fallback_repo}'.")
            return _try_load(fallback_repo)
        raise


def run(audio_path: str, output_path: str, config: dict) -> int:
    # Tắt torio FFmpeg (tránh lỗi DLL lúc import torchaudio).
    os.environ.setdefault("TORIO_USE_FFMPEG", "0")
    os.environ.setdefault("TORCHAUDIO_USE_FFMPEG", "0")

    # [v3.23.7] Log file chi tiết cạnh output để chẩn đoán crash native (access
    # violation không in ra stderr kịp). Ghi & flush từng bước; nếu process chết
    # giữa chừng, dòng cuối trong log cho biết nó chết ở ĐÂU.
    debug_log_path = str(output_path) + ".worker.log"

    def _dlog(message: str) -> None:
        try:
            with open(debug_log_path, "a", encoding="utf-8") as handle:
                handle.write(message + "\n")
                handle.flush()
        except OSError:
            pass

    _dlog(f"[START] audio={audio_path}")

    try:
        import whisperx
        _dlog("[OK] import whisperx")
    except (ImportError, FileNotFoundError, OSError) as exc:
        _eprint(f"ERROR Không nạp được WhisperX: {exc}")
        _dlog(f"[ERROR] import whisperx: {exc}")
        return 2

    device, compute_type = _resolve_device_and_compute(
        config.get("device", "cuda"), config.get("compute_type", "float16")
    )
    language = config.get("language") or None

    try:
        _progress(10, 100, f"Đang nạp model WhisperX ({device})…")
        model_size = config.get("model_size", "small")
        # [v3.23.364] Tham khảo cấu hình STT chất lượng (WhisperX/faster-whisper/Whisper-
        # WebUI): WhisperX ĐÃ bật VAD + condition_on_previous_text=False (giảm ảo giác).
        # Bổ sung 2 tham số CHỐNG LẶP mà faster-whisper KHÔNG bật mặc định — trị "ghost
        # transcripts" (câu ảo giác lặp lại ở đoạn nhạc/im lặng, rất hay gặp ở phim bộ):
        #   • repetition_penalty > 1: phạt token đã sinh (giảm lặp mềm).
        #   • no_repeat_ngram_size: chặn CỨNG lặp n-gram (mẫu ảo giác phổ biến nhất).
        # Đặt bảo thủ để KHÔNG chặn nhầm thoại lặp hợp lệ ("Không, không, không!").
        asr_options = {
            "repetition_penalty": 1.15,
            "no_repeat_ngram_size": 3,
        }
        _dlog(f"[STEP] load_model model={model_size} device={device} compute={compute_type}")
        model = _load_whisperx_model(
            whisperx, model_size, device, compute_type, language, asr_options, _dlog
        )
        _dlog("[OK] load_model")
        audio = _load_wav_as_float32(audio_path)
        _dlog(f"[OK] load_audio samples={len(audio)}")

        _progress(35, 100, "Đang phiên âm…")
        _dlog("[STEP] transcribe")
        transcription = model.transcribe(audio, batch_size=int(config.get("batch_size", 16)))
        detected_language = transcription.get("language", config.get("language", ""))
        segments = transcription.get("segments", [])
        _dlog(f"[OK] transcribe segments={len(segments)} lang={detected_language}")

        # [v3.23.5] GHI kết quả (chưa align) ra file NGAY — phòng khi align gây crash
        # native (access violation 0xC0000005 từ torchaudio/wav2vec2) giết cả tiến
        # trình con. Khi đó process cha vẫn đọc được kết quả cấp câu, không mất trắng.
        def _dump(segs, lang):
            with open(output_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {"segments": _sanitize_segments(segs), "language": lang or ""},
                    handle, ensure_ascii=False,
                )

        _dump(segments, detected_language)
        _dlog("[OK] dump pre-align")

        # [v3.23.7] KHÔNG align trong CÙNG tiến trình transcribe nữa. CTranslate2
        # (faster-whisper) đã nạp cuDNN của nó vào process này; nếu align nạp thêm
        # torch+cuDNN → xung đột → access violation 0xC0000005 (kể cả align CPU, vì
        # chỉ riêng torch khởi tạo đã đụng). Align được chạy ở TIẾN TRÌNH CON RIÊNG
        # (mode=align) do adapter điều phối. Ở đây CHỈ transcribe — align & diarize
        # (đều dùng torch) chạy ở tiến trình con align riêng để tách khỏi CTranslate2.

        _progress(95, 100, "Đang ghi kết quả…")
        clean_segments = _sanitize_segments(segments)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(
                {"segments": clean_segments, "language": detected_language or ""},
                handle, ensure_ascii=False,
            )
        _progress(100, 100, f"Hoàn tất: {len(clean_segments)} segment.")
        return 0
    except (RuntimeError, ValueError, OSError, KeyError) as exc:
        _eprint(f"ERROR Lỗi WhisperX khi phiên âm: {exc}")
        return 3


def _safe_float(value, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _sanitize_segments(segments) -> list:
    clean = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        entry = {
            "start": _safe_float(segment.get("start")),
            "end": _safe_float(segment.get("end")),
            "text": str(segment.get("text", "")),
        }
        if segment.get("speaker"):
            entry["speaker"] = str(segment["speaker"])
        words = segment.get("words")
        if isinstance(words, list):
            entry["words"] = [
                {
                    "word": str(w.get("word", "")),
                    "start": _safe_float(w.get("start")),
                    "end": _safe_float(w.get("end")),
                    "score": _safe_float(w.get("score")),
                }
                for w in words
                if isinstance(w, dict)
            ]
        clean.append(entry)
    return clean


def run_align(audio_path: str, segments_path: str, output_path: str, config: dict) -> int:
    """[v3.23.7] Chế độ ALIGN RIÊNG: chạy ở tiến trình con tách biệt khỏi transcribe.

    Đọc segments (từ transcribe) + audio, chỉ chạy căn chỉnh cấp từ rồi ghi lại.
    Tiến trình này KHÔNG nạp CTranslate2/faster-whisper, chỉ torch/wav2vec2 → một
    stack cuDNN duy nhất → không xung đột → không access violation.
    """
    os.environ.setdefault("TORIO_USE_FFMPEG", "0")
    os.environ.setdefault("TORCHAUDIO_USE_FFMPEG", "0")
    debug_log_path = str(output_path) + ".align.log"

    def _dlog(message: str) -> None:
        try:
            with open(debug_log_path, "a", encoding="utf-8") as handle:
                handle.write(message + "\n")
                handle.flush()
        except OSError:
            pass

    _dlog("[ALIGN START]")
    try:
        import whisperx
        _dlog("[OK] import whisperx")
    except (ImportError, FileNotFoundError, OSError) as exc:
        _eprint(f"ERROR Không nạp được WhisperX (align): {exc}")
        return 2

    try:
        with open(segments_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        segments = payload.get("segments", [])
        detected_language = payload.get("language", config.get("language", "")) or "en"
        if not segments:
            _eprint("WARN Không có segment để align.")
            return 0

        align_device = config.get("align_device", "cpu")
        audio = _load_wav_as_float32(audio_path)

        # [v3.23.9] Align chỉ chạy khi bật. Nếu tắt align nhưng bật diarize → bỏ qua
        # align, vẫn chạy diarize bên dưới (diarize-only).
        if config.get("enable_align", True):
            _progress(70, 100, f"Đang căn chỉnh timestamp cấp từ ({align_device})…")
            _dlog(f"[STEP] load_align_model device={align_device} lang={detected_language}")
            align_model, metadata = whisperx.load_align_model(
                language_code=detected_language, device=align_device
            )
            _dlog("[OK] load_align_model")
            _dlog("[STEP] align run")
            aligned = whisperx.align(
                segments, align_model, metadata, audio, align_device,
                return_char_alignments=False,
            )
            segments = aligned.get("segments", segments)
            _dlog(f"[OK] align segments={len(segments)}")
        else:
            _dlog("[SKIP] align tắt (chỉ diarize)")

        # [v3.23.9] Diarization (pyannote, cũng dùng torch) chạy CÙNG tiến trình align
        # — tách khỏi CTranslate2 nên không xung đột.
        if config.get("enable_diarize", False) and config.get("hf_token"):
            try:
                _progress(85, 100, "Đang phân tách người nói…")
                _dlog("[STEP] diarize")
                diarize_model = whisperx.DiarizationPipeline(
                    use_auth_token=config["hf_token"], device=align_device
                )
                diarize_segments = diarize_model(audio)
                segments = whisperx.assign_word_speakers(
                    diarize_segments, {"segments": segments}
                ).get("segments", segments)
                _dlog("[OK] diarize")
            except (RuntimeError, ValueError, OSError, KeyError) as diar_exc:
                _eprint(f"WARN Bỏ qua diarization (lỗi): {diar_exc}")
                _dlog(f"[ERROR] diarize: {diar_exc}")

        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(
                {"segments": _sanitize_segments(segments), "language": detected_language},
                handle, ensure_ascii=False,
            )
        _progress(95, 100, f"Căn chỉnh xong: {len(segments)} segment.")
        return 0
    except (RuntimeError, ValueError, OSError, KeyError, json.JSONDecodeError) as exc:
        _eprint(f"ERROR Lỗi align: {exc}")
        _dlog(f"[ERROR] {exc}")
        return 3


def main() -> None:
    parser = argparse.ArgumentParser(description="WhisperX worker (process-isolated).")
    parser.add_argument("--mode", default="transcribe", choices=["transcribe", "align"])
    parser.add_argument("--audio", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", required=True, help="JSON config string.")
    parser.add_argument("--segments", help="(mode=align) file segments đầu vào.")
    args = parser.parse_args()
    config = json.loads(args.config)
    if args.mode == "align":
        sys.exit(run_align(args.audio, args.segments, args.output, config))
    sys.exit(run(args.audio, args.output, config))


if __name__ == "__main__":
    main()
