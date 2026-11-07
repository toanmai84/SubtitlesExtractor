"""[v3.23.678] Tách LỜI THOẠI khỏi NỀN (nhạc + tiếng động) bằng model MDX-Net ONNX (họ UVR).

Dùng cho chế độ xuất "Lồng tiếng sạch": giọng Việt trộn lên phần nền đã bỏ lời gốc, thay vì chỉ HẠ tiếng gốc
(ducking 12:1 vẫn còn nghe tiếng Trung lẫn dưới giọng Việt).

Đo thật tập 1 phim mẫu, 60 s đầu, model ``Kim_Vocal_2`` (tách lời; nền = gốc − lời):
* Gemini chép lời bản GỐC: 126 chữ Hán (6 câu thoại). Chép lời phần NỀN đã tách: "KHÔNG CÓ LỜI".
* CPU (container 4 nhân, cùng lúc chạy bộ test): 60,7 s cho 60 s âm thanh.

Thuật toán (giống UVR / audio-separator): STFT n_fft riêng từng model, hop 1024, cắt 3072 dải tần thấp, khung 256 bước;
mỗi khối lấy ``chunk = hop·(dim_t−1)`` mẫu, bỏ ``n_fft/2`` mẫu hai mép (vùng biên STFT kém chính xác). Đọc/ghi THEO KHỐI
nên phim 139 phút (2,9 GB nếu nạp cả float32 stereo) không cần nạp hết vào RAM.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 44100
HOP = 1024
DIM_F = 3072
DIM_T = 256


@dataclass(frozen=True, slots=True)
class SeparationModel:
    """Một model MDX-Net: tên tệp, n_fft, stem chính model trả ra, URL tải, sha256, cỡ (byte)."""

    file_name: str
    n_fft: int
    primary_is_vocals: bool
    url: str
    sha256: str
    size_bytes: int


#: Model mặc định: tách LỜI (nền = gốc − lời). 64 MB.
KIM_VOCAL_2 = SeparationModel(
    file_name="Kim_Vocal_2.onnx",
    n_fft=7680,
    primary_is_vocals=True,
    url="https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/Kim_Vocal_2.onnx",
    sha256="ce74ef3b6a6024ce44211a07be9cf8bc6d87728cc852a68ab34eb8e58cde9c8b",
    size_bytes=66759214,
)


def _window(n_fft: int) -> np.ndarray:
    return np.hanning(n_fft + 1)[:-1].astype(np.float32)  # Hann tuần hoàn (= torch.hann_window)


def stft_chunk(x: np.ndarray, n_fft: int) -> np.ndarray:
    """``x``: [B, 2, L] → [B, 4, DIM_F, frames] (thực/ảo kênh trái, thực/ảo kênh phải)."""
    win = _window(n_fft)
    pad = n_fft // 2
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad)), mode="reflect")
    n_frames = 1 + (xp.shape[-1] - n_fft) // HOP
    idx = np.arange(n_fft)[None, :] + HOP * np.arange(n_frames)[:, None]
    spec = np.fft.rfft(xp[..., idx] * win, axis=-1)            # [B, 2, T, F]
    spec = np.transpose(spec, (0, 1, 3, 2))[:, :, :DIM_F, :]     # [B, 2, DIM_F, T]
    return np.stack([spec.real[:, 0], spec.imag[:, 0], spec.real[:, 1], spec.imag[:, 1]], axis=1).astype(np.float32)


def istft_chunk(spec: np.ndarray, n_fft: int, length: int) -> np.ndarray:
    """Ngược của :func:`stft_chunk` (dải tần cao bị cắt coi như 0)."""
    batch, frames = spec.shape[0], spec.shape[-1]
    n_bins = n_fft // 2 + 1
    full = np.zeros((batch, 4, n_bins, frames), dtype=np.float32)
    full[:, :, :DIM_F, :] = spec
    cplx = np.stack([full[:, 0] + 1j * full[:, 1], full[:, 2] + 1j * full[:, 3]], axis=1)
    win = _window(n_fft)
    seg = np.fft.irfft(np.transpose(cplx, (0, 1, 3, 2)), n=n_fft, axis=-1).astype(np.float32) * win
    total = n_fft + HOP * (frames - 1)
    out = np.zeros((batch, 2, total), dtype=np.float32)
    norm = np.zeros(total, dtype=np.float32)
    win_sq = win ** 2
    for t in range(frames):
        out[:, :, t * HOP:t * HOP + n_fft] += seg[:, :, t]
        norm[t * HOP:t * HOP + n_fft] += win_sq
    pad = n_fft // 2
    result: np.ndarray = out[:, :, pad:pad + length] / np.maximum(norm[pad:pad + length], 1e-8)
    return result


def chunk_geometry(n_fft: int) -> tuple[int, int, int]:
    """``(chunk, trim, gen)``: số mẫu mỗi khối, số mẫu bỏ mỗi mép, số mẫu hữu ích mỗi khối."""
    chunk = HOP * (DIM_T - 1)
    trim = n_fft // 2
    return chunk, trim, chunk - 2 * trim


def separate_file(
    session: object,
    model: SeparationModel,
    source_wav: Path,
    background_wav: Path,
    *,
    progress: Callable[[float], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> bool:
    """Tách ``source_wav`` (stereo 44,1 kHz) → ghi phần NỀN vào ``background_wav``. Trả False nếu bị huỷ.

    ``session``: ``onnxruntime.InferenceSession`` (truyền vào để test giả lập được).
    """
    import soundfile as sf

    chunk, trim, gen = chunk_geometry(model.n_fft)
    with sf.SoundFile(str(source_wav)) as src:
        if src.samplerate != SAMPLE_RATE or src.channels != 2:
            raise ValueError(f"Cần WAV stereo 44,1 kHz, nhận {src.channels} kênh {src.samplerate} Hz")
        total = src.frames
        tmp = background_wav.with_name(f"{background_wav.stem}.part{background_wav.suffix}")
        # FLAC 16 bit: tập 45 phút ~150 MB (WAV float ~950 MB). Định dạng theo đuôi tệp đích.
        subtype = "PCM_16" if background_wav.suffix.lower() == ".flac" else "FLOAT"
        completed = True
        with sf.SoundFile(str(tmp), "w", samplerate=SAMPLE_RATE, channels=2, subtype=subtype) as dst:
            for start in range(0, total, gen):
                if cancelled is not None and cancelled():
                    completed = False
                    break
                # Khối [start − trim, start + gen + trim), ngoài biên tệp đệm 0 (như UVR đệm 0 hai đầu).
                lo = start - trim
                src.seek(max(0, lo))
                data = src.read(chunk - max(0, -lo), dtype="float32", always_2d=True).T
                block = np.zeros((2, chunk), dtype=np.float32)
                block[:, max(0, -lo):max(0, -lo) + data.shape[1]] = data
                spec = stft_chunk(block[None], model.n_fft)
                pred = session.run(None, {"input": spec})[0]  # type: ignore[attr-defined]
                primary = istft_chunk(pred, model.n_fft, chunk)[0, :, trim:trim + gen]
                mix = block[:, trim:trim + gen]
                background = mix - primary if model.primary_is_vocals else primary
                keep = min(gen, total - start)
                dst.write(background[:, :keep].T)
                if progress is not None:
                    progress(min(1.0, (start + keep) / max(1, total)))
    # Đổi tên SAU khi đóng tệp (Windows không cho đổi tên tệp đang mở).
    if completed:
        tmp.replace(background_wav)
    else:
        tmp.unlink(missing_ok=True)
    return completed


# ── Model: tải về một lần, kiểm sha256 ─────────────────────────────────────────────────────


class SeparationError(RuntimeError):
    """Không tải được model / không tách được âm thanh (thông điệp hiển thị cho người dùng)."""


def ensure_model(
    models_dir: Path,
    model: SeparationModel = KIM_VOCAL_2,
    *,
    progress: Callable[[float], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
    opener: Callable[..., Any] | None = None,
) -> Path:
    """Trả đường dẫn model (tải lần đầu ~64 MB, kiểm sha256; tệp hỏng/không khớp thì tải lại)."""
    import hashlib
    import urllib.request

    models_dir.mkdir(parents=True, exist_ok=True)
    target = models_dir / model.file_name
    if target.is_file() and target.stat().st_size == model.size_bytes:
        return target
    part = target.with_suffix(".part")
    digest = hashlib.sha256()
    done = 0
    try:
        with (opener or urllib.request.urlopen)(model.url, timeout=60) as resp, open(part, "wb") as out:
            while True:
                if cancelled is not None and cancelled():
                    raise SeparationError("Đã huỷ tải model tách giọng.")
                block = resp.read(1 << 20)
                if not block:
                    break
                out.write(block)
                digest.update(block)
                done += len(block)
                if progress is not None:
                    progress(min(1.0, done / model.size_bytes))
    except OSError as exc:
        part.unlink(missing_ok=True)
        raise SeparationError(f"Không tải được model tách giọng ({model.file_name}): {exc}") from exc
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    if digest.hexdigest() != model.sha256:
        part.unlink(missing_ok=True)
        raise SeparationError(f"Model tách giọng tải về bị hỏng (sai sha256) — hãy thử lại: {model.file_name}")
    part.replace(target)
    return target


def make_session(model_path: Path) -> object:
    """``onnxruntime.InferenceSession`` dùng GPU nếu có (CUDA / DirectML), không thì CPU đủ luồng."""
    import os

    import onnxruntime as ort

    available = set(ort.get_available_providers())
    providers = [p for p in ("CUDAExecutionProvider", "DmlExecutionProvider") if p in available]
    providers.append("CPUExecutionProvider")
    options = ort.SessionOptions()
    options.intra_op_num_threads = max(1, os.cpu_count() or 1)
    logger.info("Tách giọng: %s (%s)", model_path.name, ", ".join(providers))
    return ort.InferenceSession(str(model_path), sess_options=options, providers=providers)


# ── Phần nền của một video (có bộ nhớ đệm) ─────────────────────────────────────────────────


def background_cache_path(cache_dir: Path, video_path: Path, model: SeparationModel = KIM_VOCAL_2) -> Path:
    """Tệp nền đã tách cho ``video_path``: khoá theo đường dẫn + cỡ + giờ sửa + model (video đổi → tách lại)."""
    import hashlib

    stat = video_path.stat()
    key = f"{video_path.resolve()}|{stat.st_size}|{int(stat.st_mtime)}|{model.file_name}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"{video_path.stem[:40]}_{digest}_nen.flac"


def prepare_background(
    ffmpeg: str,
    video_path: Path,
    work_dir: Path,
    *,
    model: SeparationModel = KIM_VOCAL_2,
    progress: Callable[[float], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
    session_factory: Callable[[Path], object] = make_session,
    start_sec: float = 0.0,
    duration_sec: float = 0.0,
) -> Path | None:
    """Phần NỀN (bỏ lời thoại) của ``video_path``, tách một lần rồi dùng lại. ``None`` = bị huỷ.

    Tiến độ: 0–5 % tải model (lần đầu), 5–10 % tách âm thanh khỏi video, 10–100 % tách giọng.
    ``duration_sec`` > 0: chỉ tách đoạn ``[start_sec, start_sec + duration_sec)`` (xuất thử) — có bộ nhớ đệm riêng.
    """
    import subprocess
    import tempfile

    cache_dir = work_dir / "backgrounds"
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = background_cache_path(cache_dir, video_path, model)
    window: list[str] = []
    if duration_sec > 0:
        target = target.with_name(f"{target.stem}_{int(start_sec * 1000)}_{int(duration_sec * 1000)}{target.suffix}")
        window = ["-ss", f"{start_sec:.3f}", "-t", f"{duration_sec:.3f}"]
    if target.is_file():
        logger.info("Dùng lại phần nền đã tách: %s", target.name)
        return target

    def report(lo: float, hi: float) -> Callable[[float], None]:
        return lambda f: progress(lo + (hi - lo) * f) if progress is not None else None

    model_path = ensure_model(work_dir / "models", model, progress=report(0.0, 0.05), cancelled=cancelled)
    with tempfile.TemporaryDirectory(prefix="subext_sep_") as tmp:
        source = Path(tmp) / "goc.wav"
        from subtitles_extractor.infrastructure.process.hidden_process import no_window_kwargs

        result = subprocess.run(  # noqa: S603 — tham số cố định, không qua shell
            [ffmpeg, "-v", "error", "-y", *window, "-i", str(video_path), "-vn", "-ac", "2", "-ar", str(SAMPLE_RATE),
             "-c:a", "pcm_f32le", str(source)],
            capture_output=True, text=True, **no_window_kwargs(),
        )
        if result.returncode != 0 or not source.is_file():
            raise SeparationError(f"Không tách được âm thanh khỏi video: {(result.stderr or '').strip()[-300:]}")
        if progress is not None:
            progress(0.10)
        session = session_factory(model_path)
        ok = separate_file(session, model, source, target, progress=report(0.10, 1.0), cancelled=cancelled)
    return target if ok else None
