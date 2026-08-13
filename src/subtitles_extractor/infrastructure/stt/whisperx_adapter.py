"""Adapter Speech-to-Text dùng WhisperX.

WhisperX = Whisper + alignment cấp từ (wav2vec2) + tuỳ chọn diarization. Cho mốc
thời gian chính xác hơn Whisper thuần, rất hợp để dựng phụ đề.

Engine là phụ thuộc NẶNG (torch + model GPU) nên:
  * ``is_available()`` probe ``import whisperx`` — UI ẩn/hiện chức năng tương ứng.
  * Mọi import whisperx/torch đều LAZY (trong hàm) để app khởi động được dù chưa cài.

Adapter trích audio bằng ffmpeg (16kHz mono, ``aresample=async=1`` chống lệch
tiếng — đồng bộ với waveform widget) rồi đưa vào WhisperX.
"""

from __future__ import annotations

import logging
import os
import queue
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Callable, Final

# [v3.22.3] Tắt torio FFmpeg backend SỚM (mức module) — torio cố nạp
# libtorio_ffmpeg6.pyd khi import torchaudio; thiếu FFmpeg DLL trên Windows gây
# FileNotFoundError. Ta không cần torio (đã trích audio bằng ffmpeg.exe).
os.environ.setdefault("TORIO_USE_FFMPEG", "0")
os.environ.setdefault("TORCHAUDIO_USE_FFMPEG", "0")

from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.exceptions import SpeechToTextError
from subtitles_extractor.domain.ports.speech_to_text_port import (
    TranscriptionConfig,
    TranscriptionProgressCallback,
    TranscriptionResult,
)
from subtitles_extractor.domain.value_objects.confidence import Confidence
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval

logger = logging.getLogger(__name__)

_AUDIO_SAMPLE_RATE = 16_000


def _safe_float(value, default: float = 0.0) -> float:
    """Chuyển sang float an toàn (None/lỗi → default)."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


#: Khe hở tối thiểu giữa hai câu liền nhau (giây). Đủ nhỏ để không thấy hụt, đủ lớn để
#: trình phát không hiển thị chồng khung hình.
_MIN_CUE_GAP_SEC: Final[float] = 0.02

#: Nhịp kiểm huỷ khi tiến trình con im lặng (giây). Đủ nhỏ để bấm Huỷ thấy phản hồi
#: gần như tức thì, đủ lớn để không đốt CPU.
_CANCEL_POLL_SECONDS: Final[float] = 0.2

#: Tên thư mục môi trường Python riêng cho WhisperX, đặt cạnh thư mục dự án.
WHISPERX_ENV_DIRNAME: Final[str] = "whisperx_env"

#: Biến môi trường để chỉ định thẳng trình thông dịch (ưu tiên cao nhất).
WHISPERX_PYTHON_ENV_VAR: Final[str] = "SUBEXT_WHISPERX_PYTHON"

_MISSING_ENV_MESSAGE: Final[str] = (
    "Chưa có môi trường WhisperX.\n\n"
    "WhisperX cần cài RIÊNG, không chung với ứng dụng, vì hai lý do:\n"
    "  • Nó yêu cầu huggingface-hub < 1.0 trong khi ứng dụng đang dùng bản 1.x — "
    "cài chung sẽ hạ cấp và có thể làm hỏng VieNeu-TTS/PaddleOCR.\n"
    "  • torch nạp CUDA riêng, dễ xung đột DLL với paddle.\n\n"
    "Cách cài (chạy trong thư mục dự án):\n"
    "  python -m venv whisperx_env\n"
    "  whisperx_env\\Scripts\\python -m pip install torch torchaudio torchvision "
    "--index-url https://download.pytorch.org/whl/cu129\n"
    "  whisperx_env\\Scripts\\python -m pip install whisperx\n\n"
    "Hoặc đặt biến môi trường SUBEXT_WHISPERX_PYTHON trỏ tới python.exe có sẵn WhisperX."
)


def _ensure_worker_script(worker_script: Path) -> None:
    """Kiểm tệp worker có mặt không, báo rõ nếu thiếu.

    [v3.23.340] Thiếu tệp script khiến Python thoát **mã 2** — TRÙNG mã của lỗi sai đối
    số, nên v3.23.338 đã chẩn đoán nhầm sang "chạy nhầm trình thông dịch". Kiểm tường
    minh ở đây để phân biệt được hai nguyên nhân.

    Args:
        worker_script: Đường dẫn tệp worker.

    Raises:
        SpeechToTextError: Khi tệp không tồn tại.
    """
    if worker_script.is_file():
        return
    logger.error("Thiếu tệp worker WhisperX: %s", worker_script)
    raise SpeechToTextError(
        "Bản đóng gói thiếu tệp xử lý của WhisperX:\n"
        f"  {worker_script}\n\n"
        "Đây là lỗi đóng gói, không phải do máy bạn. Hãy build lại ứng dụng bằng "
        "phiên bản mới (tệp này đã được bổ sung vào bản đóng gói từ v3.23.340)."
    )


def resolve_whisperx_python() -> str | None:
    """Tìm trình thông dịch Python có cài WhisperX.

    [v3.23.333] Trước đây dùng thẳng ``sys.executable``, gây BA vấn đề cùng lúc:

    1. **Xung đột phụ thuộc** — whisperx ghim ``huggingface-hub<1.0.0`` còn ứng dụng
       đang chạy bản ``1.x``; cài chung sẽ hạ cấp và có thể làm hỏng VieNeu-TTS.
    2. **Xung đột DLL** — torch nạp CUDA riêng, đụng paddle (chính là lý do adapter này
       vốn đã chạy subprocess).
    3. **Hỏng ở bản đóng gói** — ``sys.executable`` khi đó là chính tệp ``.exe``, chạy
       nó sẽ mở lại ứng dụng chứ không chạy script.

    Thứ tự tìm: biến môi trường → thư mục ``whisperx_env`` cạnh dự án → ``sys.executable``
    (chỉ khi chạy từ mã nguồn VÀ thực sự có whisperx).

    Returns:
        Đường dẫn python.exe, hoặc ``None`` nếu không tìm được.
    """
    import importlib.util
    import os

    explicit = os.environ.get(WHISPERX_PYTHON_ENV_VAR, "").strip()
    if explicit and Path(explicit).is_file():
        return explicit

    # Môi trường riêng đặt cạnh gốc dự án (cùng cấp với build_env).
    project_root = Path(__file__).resolve().parents[4]
    for relative in ("Scripts/python.exe", "bin/python", "bin/python3"):
        candidate = project_root / WHISPERX_ENV_DIRNAME / relative
        if candidate.is_file():
            return str(candidate)

    # Chạy từ mã nguồn và whisperx nằm sẵn trong môi trường hiện tại.
    if not getattr(sys, "frozen", False):
        try:
            if importlib.util.find_spec("whisperx") is not None:
                return sys.executable
        except (ImportError, ValueError):
            pass
    return None


def _subprocess_flags() -> dict[str, int]:
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NO_WINDOW}  # type: ignore[attr-defined]
    return {}


class WhisperXAdapter:
    """Phiên âm giọng nói bằng WhisperX (lazy import, GPU-aware)."""

    def __init__(
        self,
        ffmpeg_binary: str = "ffmpeg",
        subprocess_timeout_sec: float = 600.0,
        use_subprocess: bool = True,
    ) -> None:
        self._ffmpeg = ffmpeg_binary
        self._timeout = subprocess_timeout_sec
        # [v3.23] Mặc định chạy WhisperX trong TIẾN TRÌNH CON để tránh xung đột
        # cuDNN/cuBLAS DLL với PaddleOCR (paddle) khi cùng nạp torch trong 1 process.
        self._use_subprocess = use_subprocess

    def is_available(self) -> bool:
        # [v3.23] KHÔNG import whisperx vào process chính (sẽ nạp torch → xung đột
        # DLL với paddle). Chỉ kiểm package CÓ CÀI hay không qua find_spec.
        # [v3.23.333] Khả dụng = TÌM ĐƯỢC môi trường có WhisperX, không phải "có
        # whisperx trong tiến trình này" — vì nó cố ý được cài ở môi trường RIÊNG.
        return resolve_whisperx_python() is not None

    def get_engine_name(self) -> str:
        return "WhisperX (Speech-to-Text)"

    def transcribe(
        self,
        media_path: Path,
        config: TranscriptionConfig,
        progress_callback: TranscriptionProgressCallback | None = None,
        cancellation_check: Callable[[], bool] | None = None,
    ) -> TranscriptionResult:
        if not media_path.exists():
            raise SpeechToTextError(f"Không tìm thấy media: {media_path}.")
        if not self.is_available():
            raise SpeechToTextError(_MISSING_ENV_MESSAGE)

        def report(current: int, total: int, message: str) -> None:
            if progress_callback is not None:
                progress_callback(current, total, message)

        def cancelled() -> bool:
            return cancellation_check is not None and cancellation_check()

        report(0, 100, "Đang trích audio từ video…")
        with tempfile.TemporaryDirectory(prefix="se_stt_") as tmp_dir:
            audio_path = Path(tmp_dir) / "audio.wav"
            self._extract_audio(media_path, audio_path)
            if cancelled():
                return TranscriptionResult()
            if self._use_subprocess:
                return self._run_whisperx_subprocess(
                    Path(tmp_dir), audio_path, config, report, cancelled
                )
            return self._run_whisperx(audio_path, config, report, cancelled)

    @staticmethod
    def _build_subprocess_env() -> dict[str, str]:
        """Tạo env cho subprocess WhisperX: khôi phục torch\\lib vào PATH (Windows).

        Tiến trình chính loại torch\\lib khỏi PATH để bảo vệ paddle; nhưng subprocess
        CẦN cuDNN của torch (cho align wav2vec2) — thiếu nó gây access violation
        0xC0000005. Suy ra đường dẫn torch\\lib từ site-packages (find_spec ở process
        chính bị chặn) và thêm vào ĐẦU PATH của subprocess.
        """
        import os

        env = dict(os.environ)
        if os.name != "nt":
            return env

        # [v3.23.338] torch nay nằm trong MÔI TRƯỜNG RIÊNG `whisperx_env`, không còn ở
        # môi trường chính. Suy `torch\\lib` từ chính trình thông dịch sẽ chạy worker;
        # lấy nhầm từ `sysconfig` của tiến trình chính sẽ không tìm thấy gì.
        python_exe = resolve_whisperx_python()
        if python_exe:
            env_root = Path(python_exe).parent.parent
            for relative in ("Lib/site-packages", "lib/site-packages"):
                torch_lib = env_root / relative / "torch" / "lib"
                if torch_lib.is_dir():
                    env["PATH"] = str(torch_lib) + os.pathsep + env.get("PATH", "")
                    break
        return env

    def _run_whisperx_subprocess(
        self,
        tmp_dir: Path,
        audio_path: Path,
        config: TranscriptionConfig,
        report: TranscriptionProgressCallback,
        cancelled: Callable[[], bool],
    ) -> TranscriptionResult:
        """[v3.23] Chạy WhisperX ở tiến trình con (cô lập khỏi paddle/torch DLL).

        Tránh xung đột cuDNN/cuBLAS giữa PaddleOCR (paddle) và WhisperX (torch) khi
        cùng nạp trong một process — lỗi nhị phân nổi tiếng trên Windows.
        """
        import json

        output_path = tmp_dir / "result.json"
        config_json = json.dumps({
            "language": config.language,
            "model_size": config.model_size,
            "device": config.device,
            "compute_type": config.compute_type,
            "batch_size": config.batch_size,
            "enable_align": config.enable_align,
            "align_device": config.align_device,
            "enable_diarize": config.enable_diarize,
            "hf_token": config.hf_token,
        })
        worker_script = Path(__file__).with_name("whisperx_subprocess.py")
        python_exe = resolve_whisperx_python()
        if python_exe is None:
            raise SpeechToTextError(_MISSING_ENV_MESSAGE)
        _ensure_worker_script(worker_script)
        command = [
            python_exe, str(worker_script),
            "--audio", str(audio_path),
            "--output", str(output_path),
            "--config", config_json,
        ]

        # [v3.23.5] Chuẩn bị môi trường cho subprocess: KHÔI PHỤC torch\lib vào PATH.
        # Tiến trình cha đã LOẠI torch\lib khỏi PATH (để bảo vệ paddle), nhưng
        # subprocess WhisperX CẦN torch\lib (cuDNN của torch cho align wav2vec2) —
        # thiếu nó gây access violation 0xC0000005. Thêm lại torch\lib cho subprocess.
        sub_env = self._build_subprocess_env()

        # Chạy & đọc tiến độ realtime từ stderr (dòng "PROGRESS cur total msg").
        try:
            process = subprocess.Popen(
                # [v3.23.345] stdout=DEVNULL, KHÔNG phải PIPE. Đây là sửa lỗi TREO
                # VĨNH VIỄN: bản trước mở ống stdout nhưng chỉ đọc stderr. WhisperX và
                # faster-whisper in rất nhiều ra stdout (nhận diện ngôn ngữ, thanh tiến
                # độ tqdm, cảnh báo). Khi bộ đệm ống đầy (~4–64 KB), tiến trình con KẸT
                # ở lệnh ghi stdout, còn tiến trình cha KẸT ở vòng đọc stderr — bế tắc
                # hoàn toàn, không timeout nào cứu vì `for line in stderr` chặn ở lệnh
                # đọc. Đã tái hiện: treo vĩnh viễn, phải giết tiến trình.
                # Giao thức chỉ dùng stderr (PROGRESS/ERROR/WARN) nên stdout là nhiễu.
                command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", errors="replace", env=sub_env,
                **_subprocess_flags(),
            )
        except OSError as exc:
            raise SpeechToTextError(f"Không khởi chạy được tiến trình WhisperX: {exc}") from exc

        error_lines: list[str] = []
        assert process.stderr is not None

        # [v3.23.346] SỬA LỖI "bấm Huỷ không ăn": trước đây kiểm huỷ NGAY TRONG vòng
        # `for line in process.stderr`, nên chỉ phát hiện khi có DÒNG MỚI về. WhisperX
        # im lặng khá lâu ở các pha nặng (nạp model lớn, xử lý một đoạn dài) — đã đo:
        # người dùng bấm Huỷ ở giây 1,5 mà tới giây 8,0 mới dừng.
        # Nay đọc stderr ở LUỒNG RIÊNG, vòng chính lấy từ hàng đợi có hạn giờ ngắn nên
        # kiểm được huỷ đều đặn kể cả khi tiến trình con hoàn toàn im lặng.
        line_queue: queue.Queue[str | None] = queue.Queue()

        def _drain_stderr() -> None:
            """Đọc stderr tới hết rồi đặt ``None`` làm dấu kết thúc."""
            try:
                for raw in process.stderr:  # type: ignore[union-attr]
                    line_queue.put(raw.rstrip("\n"))
            except (OSError, ValueError):
                pass  # ống đã đóng khi tiến trình bị giết — không phải lỗi
            finally:
                line_queue.put(None)

        reader = threading.Thread(target=_drain_stderr, daemon=True)
        reader.start()

        while True:
            try:
                item = line_queue.get(timeout=_CANCEL_POLL_SECONDS)
            except queue.Empty:
                # Không có dòng nào — vẫn phải kiểm huỷ, đây chính là điểm sửa.
                if cancelled():
                    process.terminate()
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    return TranscriptionResult()
                if process.poll() is not None and not reader.is_alive():
                    break
                continue
            if item is None:
                break
            line = item
            if cancelled():
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                return TranscriptionResult()
            if line.startswith("PROGRESS "):
                parts = line.split(" ", 3)
                if len(parts) >= 4:
                    try:
                        report(int(parts[1]), int(parts[2]), parts[3])
                    except ValueError:
                        pass
            elif line.startswith("ERROR "):
                error_lines.append(line[6:])
            elif line.startswith("WARN "):
                logger.warning("WhisperX subprocess: %s", line[5:])

        # [v3.23.345] wait() CÓ hạn giờ. Bình thường stderr đã EOF nên tiến trình con
        # sắp thoát, nhưng nếu nó kẹt trong một lời gọi CUDA thì `wait()` không hạn giờ
        # sẽ treo ứng dụng vĩnh viễn. 60 giây là rất rộng cho việc dọn dẹp.
        try:
            return_code = process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            logger.warning(
                "WhisperX không thoát sau khi đóng luồng lỗi — buộc dừng tiến trình."
            )
            process.kill()
            return_code = process.wait(timeout=10)
        if return_code != 0:
            # [v3.23.5] Nếu subprocess crash (vd align gây access violation
            # 0xC0000005 = mã 3221225477) NHƯNG đã kịp ghi kết quả pre-align ra file
            # → vẫn dùng kết quả cấp câu đó thay vì báo lỗi & mất trắng.
            if output_path.exists():
                try:
                    payload = json.loads(output_path.read_text(encoding="utf-8"))
                    if payload.get("segments"):
                        logger.warning(
                            "Tiến trình WhisperX thoát mã %s (có thể do align crash) — "
                            "dùng kết quả phiên âm cấp câu đã lưu trước đó.", return_code
                        )
                        segments = payload.get("segments", [])
                        detected_language = payload.get("language", config.language)
                        events = self._segments_to_events(segments, config)
                        return TranscriptionResult(
                            events=events,
                            detected_language=detected_language or "",
                            raw_segments=segments,
                        )
                except (json.JSONDecodeError, OSError):
                    pass
            detail = "; ".join(error_lines) or f"Tiến trình WhisperX thoát mã {return_code}."
            # [v3.23.338] Mã 2 + KHÔNG có dòng lỗi nào = dấu hiệu đặc trưng của việc
            # chạy nhầm trình thông dịch: argparse trả về 2 khi sai đối số. Ở bản đóng
            # gói, `sys.executable` là chính tệp .exe nên nó nhận tham số lạ rồi thoát 2.
            if return_code == 2 and not error_lines:
                raise SpeechToTextError(
                    "WhisperX không chạy được: tiến trình con từ chối tham số "
                    f"(mã {return_code}).\n\n"
                    "Thường do chưa có môi trường riêng, nên ứng dụng chạy nhầm chính "
                    "nó thay vì Python.\n"
                    "Hãy bấm nút “⬇️ Cài WhisperX tự động” trong nhóm Giọng nói (STT), "
                    "hoặc tạo thủ công:\n"
                    "  python -m venv whisperx_env\n"
                    "  whisperx_env\\Scripts\\python -m pip install torch torchaudio "
                    "torchvision --index-url https://download.pytorch.org/whl/cu129\n"
                    "  whisperx_env\\Scripts\\python -m pip install whisperx"
                )
            if return_code == 3221225477 or "0xC0000005" in detail:
                raise SpeechToTextError(
                    "WhisperX bị lỗi bộ nhớ (access violation) ở bước căn chỉnh cấp từ — "
                    "thường do torchaudio/wav2vec2 xung đột thư viện. Hãy TẮT 'Căn chỉnh "
                    "timestamp cấp từ' trong nhóm STT để phiên âm ổn định."
                )
            if "torio" in detail or "FFmpeg" in detail or ".dll" in detail.lower():
                raise SpeechToTextError(
                    "WhisperX thiếu thư viện FFmpeg/torchaudio. Cài lại: "
                    f"pip install torchaudio --force-reinstall. Chi tiết: {detail}"
                )
            raise SpeechToTextError(f"WhisperX lỗi: {detail}")

        if not output_path.exists():
            raise SpeechToTextError("Tiến trình WhisperX không tạo kết quả.")
        try:
            payload = json.loads(output_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise SpeechToTextError(f"Không đọc được kết quả WhisperX: {exc}") from exc

        segments = payload.get("segments", [])
        detected_language = payload.get("language", config.language)

        # [v3.23.7/9] ALIGN (& DIARIZE) ở tiến trình con riêng (tách cuDNN khỏi
        # CTranslate2). Chạy nếu bật align HOẶC diarize.
        if (config.enable_align or config.enable_diarize) and segments:
            aligned_segments = self._run_align_subprocess(
                tmp_dir, audio_path, segments, detected_language, config, report, cancelled
            )
            if aligned_segments:
                segments = aligned_segments

        events = self._segments_to_events(segments, config)
        logger.info("WhisperX (subprocess) phiên âm %d câu.", len(events))
        return TranscriptionResult(
            events=events,
            detected_language=detected_language or "",
            raw_segments=segments,
        )

    def _run_align_subprocess(
        self,
        tmp_dir: Path,
        audio_path: Path,
        segments: list,
        language: str,
        config: TranscriptionConfig,
        report: TranscriptionProgressCallback,
        cancelled: Callable[[], bool],
    ) -> list | None:
        """Chạy align ở tiến trình con riêng (CUDA ẩn → CPU thuần, không xung đột).

        Trả về segments đã align, hoặc None nếu align lỗi/crash (giữ bản chưa align).
        """
        import json
        seg_path = tmp_dir / "pre_align.json"
        out_path = tmp_dir / "aligned.json"
        seg_path.write_text(
            json.dumps({"segments": segments, "language": language}, ensure_ascii=False),
            encoding="utf-8",
        )
        config_json = json.dumps({
            "language": language,
            "align_device": config.align_device,
            "enable_align": config.enable_align,
            "enable_diarize": config.enable_diarize,
            "hf_token": config.hf_token,
        })
        worker_script = Path(__file__).with_name("whisperx_subprocess.py")
        # [v3.23.338] SỬA LỖI "thoát mã 2": trước đây dùng `sys.executable`, mà ở bản
        # ĐÓNG GÓI đó chính là tệp .exe của ứng dụng — chạy nó với `--mode align` thì
        # app không hiểu tham số và thoát mã 2 (argparse trả 2 khi sai đối số).
        # Nhánh phiên âm đã sửa ở v3.23.333 nhưng tôi BỎ SÓT nhánh căn chỉnh này.
        python_exe = resolve_whisperx_python()
        if python_exe is None:
            raise SpeechToTextError(_MISSING_ENV_MESSAGE)
        _ensure_worker_script(worker_script)
        command = [
            python_exe, str(worker_script), "--mode", "align",
            "--audio", str(audio_path), "--segments", str(seg_path),
            "--output", str(out_path), "--config", config_json,
        ]
        # [v3.23.345] Chú thích cũ nói "Ép tiến trình align dùng CPU thuần" — SAI, vì
        # đoạn dưới có điều kiện. Người đọc dễ tưởng lựa chọn "GPU (nhanh hơn)" trong
        # giao diện bị bỏ qua. Thực tế: chỉ ẩn CUDA khi người dùng CHỌN cpu, để torch
        # không khởi tạo CUDA context (thứ đụng cuDNN gây crash).
        sub_env = self._build_subprocess_env()
        if config.align_device == "cpu":
            sub_env["CUDA_VISIBLE_DEVICES"] = ""

        try:
            process = subprocess.Popen(
                # [v3.23.345] stdout=DEVNULL, KHÔNG phải PIPE. Đây là sửa lỗi TREO
                # VĨNH VIỄN: bản trước mở ống stdout nhưng chỉ đọc stderr. WhisperX và
                # faster-whisper in rất nhiều ra stdout (nhận diện ngôn ngữ, thanh tiến
                # độ tqdm, cảnh báo). Khi bộ đệm ống đầy (~4–64 KB), tiến trình con KẸT
                # ở lệnh ghi stdout, còn tiến trình cha KẸT ở vòng đọc stderr — bế tắc
                # hoàn toàn, không timeout nào cứu vì `for line in stderr` chặn ở lệnh
                # đọc. Đã tái hiện: treo vĩnh viễn, phải giết tiến trình.
                # Giao thức chỉ dùng stderr (PROGRESS/ERROR/WARN) nên stdout là nhiễu.
                command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", errors="replace", env=sub_env,
                **_subprocess_flags(),
            )
        except OSError as exc:
            logger.warning("Không khởi chạy được tiến trình align: %s", exc)
            return None

        assert process.stderr is not None
        for line in process.stderr:
            line = line.rstrip("\n")
            if cancelled():
                process.terminate()
                return None
            if line.startswith("PROGRESS "):
                parts = line.split(" ", 3)
                if len(parts) >= 4:
                    try:
                        report(int(parts[1]), int(parts[2]), parts[3])
                    except ValueError:
                        pass
            elif line.startswith(("ERROR ", "WARN ")):
                logger.warning("Align subprocess: %s", line)

        # [v3.23.345] Nhánh căn chỉnh cũng cần hạn giờ — cùng lý do như nhánh phiên âm.
        try:
            align_return_code = process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            logger.warning("Tiến trình align không thoát — buộc dừng.")
            process.kill()
            align_return_code = process.wait(timeout=10)
        if align_return_code != 0 or not out_path.exists():
            logger.warning("Align thất bại/crash — dùng timestamp cấp câu.")
            return None
        try:
            payload = json.loads(out_path.read_text(encoding="utf-8"))
            return payload.get("segments") or None
        except (json.JSONDecodeError, OSError):
            return None

    def _extract_audio(self, media_path: Path, audio_path: Path) -> None:
        # [v3.23.9] Tiền xử lý audio để CẢI THIỆN nhận diện lời nói:
        #   * highpass=80 + lowpass=8000: giữ dải tần giọng người (~80-8000Hz), bỏ
        #     ù tần thấp & nhiễu tần cao → bớt bỏ sót lời nói nhỏ.
        #   * loudnorm: chuẩn hoá âm lượng (lời thoại to/nhỏ đều nhau) → Whisper bắt
        #     được cả đoạn nói nhỏ bị nhạc nền lấn.
        #   * 16kHz mono pcm_s16le: ĐÚNG CHUẨN Whisper (model train ở 16kHz; cao hơn
        #     không giúp ích vì Whisper tự downsample). Mono vì giọng nói không cần
        #     stereo.
        audio_filter = (
            "aresample=async=1,"
            "highpass=f=80,lowpass=f=8000,"
            "loudnorm=I=-16:TP=-1.5:LRA=11"
        )
        command = [
            self._ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(media_path),
            "-af", audio_filter,
            "-ar", str(_AUDIO_SAMPLE_RATE), "-ac", "1",
            "-c:a", "pcm_s16le", "-f", "wav",
            str(audio_path),
        ]
        try:
            completed = subprocess.run(
                command, capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=self._timeout, **_subprocess_flags(),
            )
        except FileNotFoundError as exc:
            raise SpeechToTextError("Chưa cài ffmpeg (cần để trích audio).") from exc
        except subprocess.TimeoutExpired as exc:
            raise SpeechToTextError("ffmpeg quá thời gian khi trích audio.") from exc
        if completed.returncode != 0 or not audio_path.exists():
            # [v3.23.9] loudnorm/filter có thể lỗi với vài định dạng — thử lại đơn giản.
            logger.warning("ffmpeg lọc audio lỗi, thử trích đơn giản: %s",
                           completed.stderr.strip()[:150])
            simple = [
                self._ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(media_path), "-af", "aresample=async=1",
                "-ar", str(_AUDIO_SAMPLE_RATE), "-ac", "1",
                "-c:a", "pcm_s16le", "-f", "wav", str(audio_path),
            ]
            completed = subprocess.run(
                simple, capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=self._timeout, **_subprocess_flags(),
            )
            if completed.returncode != 0 or not audio_path.exists():
                raise SpeechToTextError(
                    f"ffmpeg lỗi trích audio: {completed.stderr.strip()[:200]}"
                )

    @staticmethod
    def _load_wav_as_float32(audio_path: Path):
        """Đọc WAV PCM 16-bit mono thành mảng float32 chuẩn hoá [-1, 1].

        Dùng module ``wave`` chuẩn của Python — KHÔNG phụ thuộc torio/torchaudio/
        soundfile, nên không vướng lỗi thiếu FFmpeg DLL trên Windows. WhisperX nhận
        đúng định dạng này (giống đầu ra của ``whisperx.load_audio``).
        """
        import wave

        import numpy as np

        with wave.open(str(audio_path), "rb") as wav_file:
            frame_count = wav_file.getnframes()
            raw_bytes = wav_file.readframes(frame_count)
        samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)
        return samples / 32768.0

    def _run_whisperx(
        self,
        audio_path: Path,
        config: TranscriptionConfig,
        report: TranscriptionProgressCallback,
        cancelled: Callable[[], bool],
    ) -> TranscriptionResult:
        # [v3.22.3] Vô hiệu hoá torio FFmpeg backend TRƯỚC khi import whisperx.
        # torio (backend torchaudio) cố nạp libtorio_ffmpeg6.pyd ngay lúc import;
        # trên Windows thiếu FFmpeg 6 DLL → FileNotFoundError lúc IMPORT (không phải
        # ImportError). Ta KHÔNG cần torio (đã trích WAV bằng ffmpeg.exe), nên tắt
        # nó để import suôn sẻ. Các biến môi trường này được torio/torchaudio đọc.
        import os

        os.environ.setdefault("TORIO_USE_FFMPEG", "0")
        os.environ.setdefault("TORCHAUDIO_USE_FFMPEG", "0")

        # Lazy import — chỉ nạp khi thực sự phiên âm (tránh chậm khởi động app).
        # Bắt RỘNG: torio có thể ném FileNotFoundError/OSError (DLL) ngay lúc import,
        # không chỉ ImportError. [v3.22.6] Serialize với import paddle (lock chung)
        # để tránh "partially initialized module" khi 2 thread import ML lib cùng lúc.
        from subtitles_extractor.infrastructure.heavy_import_lock import HEAVY_IMPORT_LOCK

        try:
            with HEAVY_IMPORT_LOCK:
                import whisperx
        except ImportError as exc:
            raise SpeechToTextError("Không nạp được WhisperX (chưa cài?).") from exc
        except (FileNotFoundError, OSError) as exc:
            raise SpeechToTextError(
                "WhisperX import lỗi do thiếu thư viện FFmpeg của torchaudio (torio). "
                "Khắc phục: 'pip install torchaudio --force-reinstall' hoặc đặt "
                f"FFmpeg 6.x vào PATH. Chi tiết: {exc}"
            ) from exc

        language = config.language or None
        try:
            # [v3.22.5] Tự dò CUDA: torch bản CPU (vd '2.8.0+cpu') KHÔNG chạy được
            # device='cuda' → assert "Torch not compiled with CUDA enabled". Nếu yêu
            # cầu cuda nhưng máy không có, tự rơi về cpu + compute_type tương thích.
            device, compute_type = self._resolve_device_and_compute(
                config.device, config.compute_type
            )
            report(10, 100, f"Đang nạp model WhisperX ({device})…")
            model = whisperx.load_model(
                config.model_size, device, compute_type=compute_type,
                language=language,
            )
            # [v3.22.2] KHÔNG dùng whisperx.load_audio — nó gọi torio/torchaudio →
            # cần FFmpeg DLL (libtorio_ffmpeg6.pyd) mà nhiều máy Windows thiếu
            # (FileNotFoundError module). Ta đã trích sẵn WAV 16kHz mono bằng
            # ffmpeg.exe, nên đọc thẳng thành float32 — bỏ hẳn phụ thuộc torio.
            audio = self._load_wav_as_float32(audio_path)
            if cancelled():
                return TranscriptionResult()

            report(35, 100, "Đang phiên âm…")
            transcription = model.transcribe(audio, batch_size=config.batch_size)
            detected_language = transcription.get("language", config.language)
            segments = transcription.get("segments", [])

            if config.enable_align and segments and not cancelled():
                # [v3.22.3] Align cấp từ dùng wav2vec2 (torchaudio) — có thể vẫn
                # chạm torio trên một số bản. Nếu lỗi → BỎ QUA align, vẫn trả phụ đề
                # với timestamp cấp CÂU (vẫn dùng được), thay vì sập cả tiến trình.
                try:
                    report(65, 100, "Đang căn chỉnh timestamp cấp từ…")
                    align_model, metadata = whisperx.load_align_model(
                        language_code=detected_language, device=device
                    )
                    aligned = whisperx.align(
                        segments, align_model, metadata, audio, device,
                        return_char_alignments=False,
                    )
                    segments = aligned.get("segments", segments)
                except (FileNotFoundError, OSError, RuntimeError, ValueError) as align_exc:
                    logger.warning(
                        "Bỏ qua căn chỉnh cấp từ (lỗi torchaudio/align): %s — "
                        "dùng timestamp cấp câu.", align_exc
                    )

            if config.enable_diarize and config.hf_token and not cancelled():
                report(85, 100, "Đang phân tách người nói…")
                segments = self._apply_diarization(
                    whisperx, audio, segments, config, detected_language, device
                )

            report(95, 100, "Đang dựng phụ đề…")
            events = self._segments_to_events(segments, config)
            report(100, 100, f"Hoàn tất: {len(events)} câu.")
            logger.info("WhisperX phiên âm %d câu (lang=%s).", len(events), detected_language)
            return TranscriptionResult(
                events=events,
                detected_language=detected_language or "",
                raw_segments=self._sanitize_raw_segments(segments),
            )
        except SpeechToTextError:
            raise
        except (FileNotFoundError, ImportError) as exc:
            # Thiếu DLL FFmpeg mà torio/torchaudio cần (vd libtorio_ffmpeg6.pyd).
            message = str(exc)
            if "torio" in message or "ffmpeg" in message.lower() or ".pyd" in message:
                raise SpeechToTextError(
                    "WhisperX thiếu thư viện FFmpeg cho torchaudio (torio). "
                    "Cách khắc phục: cài torch/torchaudio đúng bản có FFmpeg, hoặc "
                    "đặt FFmpeg 6.x vào PATH, hoặc cài lại bằng "
                    "'pip install torchaudio --force-reinstall'. "
                    f"Chi tiết: {message}"
                ) from exc
            raise SpeechToTextError(f"Lỗi WhisperX khi phiên âm: {exc}") from exc
        except (RuntimeError, ValueError, OSError, KeyError) as exc:
            raise SpeechToTextError(f"Lỗi WhisperX khi phiên âm: {exc}") from exc

    @staticmethod
    def _resolve_device_and_compute(
        requested_device: str, requested_compute: str
    ) -> tuple[str, str]:
        """Chọn device + compute_type khả dụng thực tế.

        Nếu yêu cầu 'cuda' nhưng torch không build kèm CUDA (vd bản '+cpu') hoặc
        không có GPU → rơi về 'cpu'. CPU không hỗ trợ float16 → đổi sang 'int8'
        (nhanh & nhẹ trên CPU). Trả về cặp (device, compute_type) an toàn.
        """
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
                logger.warning(
                    "Yêu cầu CUDA nhưng không khả dụng (torch CPU hoặc không có GPU) "
                    "→ chuyển sang CPU. Phiên âm sẽ chậm hơn."
                )
                device = "cpu"
        if device == "cpu" and compute_type in ("float16", "fp16", "half"):
            # float16 chỉ tối ưu trên GPU; CPU dùng int8 cho nhanh.
            compute_type = "int8"
        return device, compute_type

    @staticmethod
    def _apply_diarization(
        whisperx, audio, segments, config: TranscriptionConfig, language: str, device: str
    ):
        try:
            diarize_model = whisperx.DiarizationPipeline(
                use_auth_token=config.hf_token, device=device
            )
            diarize_segments = diarize_model(audio)
            return whisperx.assign_word_speakers(diarize_segments, {"segments": segments}).get(
                "segments", segments
            )
        except (RuntimeError, ValueError, OSError) as exc:
            logger.warning("Bỏ qua diarization (lỗi): %s", exc)
            return segments

    @staticmethod
    def _sanitize_raw_segments(segments) -> list[dict]:
        """Chuẩn hoá segment WhisperX thành dict JSON-serializable thuần.

        Giữ đầy đủ start/end/text/speaker + danh sách words (word/start/end/score)
        để xuất ra file phục vụ hiệu chuẩn thuật toán tách câu offline.
        """
        clean: list[dict] = []
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            entry: dict = {
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

    # Dấu kết thúc câu (CJK + Latin) và dấu ngắt mềm (phẩy).
    _SENTENCE_END_CHARS: frozenset[str] = frozenset("。！？!?…．.")
    _SOFT_BREAK_CHARS: frozenset[str] = frozenset("，,、；;：:")

    # [v3.23] Cụm "ảo giác" Whisper hay sinh ở đoạn im lặng/nhạc nền (cần lọc).
    _HALLUCINATION_PATTERNS: frozenset[str] = frozenset({
        "请订阅", "谢谢观看", "谢谢大家", "字幕由", "字幕志愿者",
        "请不吝点赞", "订阅", "转发", "打赏", "明镜与点点栏目",
        "thank you", "thanks for watching", "subscribe", "please subscribe",
        "❤", "♪",
    })

    @classmethod
    def _is_hallucination(cls, text: str) -> bool:
        """Phát hiện câu ảo giác phổ biến của Whisper (rác ở đoạn im lặng)."""
        stripped = text.strip().lower()
        if not stripped:
            return True
        for pattern in cls._HALLUCINATION_PATTERNS:
            if pattern.lower() in stripped:
                return True
        # Lặp 1 ký tự quá nhiều lần (vd "啊啊啊啊啊啊啊").
        if len(stripped) >= 4 and len(set(stripped)) == 1:
            return True
        return False

    @classmethod
    def _segments_to_events(
        cls, segments, config: TranscriptionConfig | None = None
    ) -> list[SubtitleEvent]:
        """Dựng SubtitleEvent từ segment WhisperX, TÁCH câu dài thành câu ngắn.

        Nếu segment có ``words`` (đã align) và bật tách câu → tách theo dấu câu /
        khoảng lặng / độ dài. Nếu không có words (align fail) → giữ segment nguyên
        (timestamp cấp câu). [v3.23] Lọc câu ảo giác (hallucination) của Whisper.
        """
        do_split = config is None or config.enable_sentence_split
        filter_halluc = config is None or getattr(config, "filter_hallucinations", True)
        events: list[SubtitleEvent] = []
        last_speaker: str | None = None  # [v3.23.9] theo dõi để chỉ gắn khi ĐỔI
        for segment in segments:
            words = segment.get("words") if isinstance(segment, dict) else None
            if do_split and words:
                # Neo vào mốc của CHÍNH đoạn này — nếu từ đầu tiên thiếu start thì rơi
                # về đây, chứ không phải về đầu phim.
                segment_start = _safe_segment_bound(segment, "start")
                segment_end = _safe_segment_bound(segment, "end")
                sub_cues = cls._split_words_into_cues(
                    words, config, segment_start, segment_end
                )
            else:
                sub_cues = cls._whole_segment_as_cue(segment)
            speaker = segment.get("speaker") if isinstance(segment, dict) else None
            for text, start, end in sub_cues:
                text = text.strip()
                if not text:
                    continue
                if filter_halluc and cls._is_hallucination(text):
                    logger.debug("Bỏ câu ảo giác Whisper: %s", text)
                    continue
                # [v3.23.9] Chỉ gắn [speaker] khi NGƯỜI NÓI THAY ĐỔI so với câu trước
                # (cùng người nói liên tiếp → không lặp nhãn, đỡ rối phụ đề).
                if speaker and speaker != last_speaker:
                    text = f"[{speaker}] {text}"
                if speaker:
                    last_speaker = speaker
                events.append(
                    SubtitleEvent(
                        index=len(events) + 1,
                        text=text,
                        interval=TimeInterval(start_sec=start, end_sec=max(end, start)),
                        confidence=Confidence.zero(),
                    )
                )
        # [v3.23.348] Làm sạch trước khi đánh số — xem `_sanitise_events`.
        events = cls._sanitise_events(events)
        for new_index, event in enumerate(events, start=1):
            event.index = new_index
        return events

    @staticmethod
    def _sanitise_events(events: list[SubtitleEvent]) -> list[SubtitleEvent]:
        """Sắp xếp theo thời gian và cắt phần chồng lấn giữa các câu.

        [v3.23.348] Trước đây KHÔNG có bước này, gây hai lỗi đo được:

        * **Chồng lấn.** Hai đoạn ``10.00–13.50`` và ``12.80–15.00`` (rất hay gặp khi
          bật phân tách người nói) chồng nhau 0,7 giây — trình phát hiện HAI dòng cùng
          lúc, dòng cũ chưa tắt thì dòng mới đã lên.
        * **Sai thứ tự.** Câu ở giây 20 được đánh số 1, câu ở giây 10 đánh số 2. Tệp SRT
          có mốc giảm dần; nhiều trình phát bỏ qua hoặc hiện sai.

        Cắt phần chồng lấn thay vì bỏ câu: nội dung vẫn còn, chỉ rút ngắn thời gian hiển
        thị. Nếu cắt xong mà thời lượng không còn dương thì giữ nguyên — thà chồng nhẹ
        còn hơn mất câu.

        Args:
            events: Danh sách sự kiện thô.

        Returns:
            Danh sách đã sắp xếp và hết chồng lấn.
        """
        if len(events) < 2:
            return events

        ordered = sorted(events, key=lambda e: (e.interval.start_sec, e.interval.end_sec))
        for current, following in zip(ordered, ordered[1:]):
            if current.interval.end_sec <= following.interval.start_sec:
                continue
            trimmed_end = following.interval.start_sec - _MIN_CUE_GAP_SEC
            if trimmed_end > current.interval.start_sec:
                current.interval = TimeInterval(
                    start_sec=current.interval.start_sec, end_sec=trimmed_end
                )
            else:
                logger.debug(
                    "Câu %.2f–%.2f chồng câu sau nhưng cắt sẽ mất nội dung — giữ nguyên.",
                    current.interval.start_sec, current.interval.end_sec,
                )
        return ordered

    @staticmethod
    def _whole_segment_as_cue(segment) -> list[tuple[str, float, float]]:
        text = str(segment.get("text", "")).strip()
        if not text:
            return []
        try:
            start = float(segment.get("start", 0.0))
            end = float(segment.get("end", start))
        except (TypeError, ValueError):
            return []
        return [(text, start, end)]

    @staticmethod
    def _segment_word_boundaries(text: str) -> set[int]:
        """[v3.23] Tập CHỈ SỐ ký tự là ranh giới TỪ GHÉP (rjieba nếu có).

        "第一步超神" → rjieba ["第一步","超神"] → ranh giới {3, 5}. Cắt câu tại các
        ranh giới này thay vì giữa từ ghép → phụ đề tự nhiên. Không rjieba → tập rỗng.

        Dùng rjieba (bản Rust, nhanh hơn jieba, không cảnh báo pkg_resources).
        """
        try:
            import rjieba
        except ImportError:
            return set()
        boundaries: set[int] = set()
        position = 0
        for token in rjieba.cut(text):
            position += len(token)
            boundaries.add(position)
        return boundaries

    @staticmethod
    def _is_cjk_text(text: str) -> bool:
        """Phát hiện văn bản CJK (Trung/Nhật/Hàn) — quyết định cách nối từ & độ dài.

        CJK: từ là từng ký tự liền (nối không dấu cách, câu ngắn ~5 ký tự).
        Latin: từ cách nhau bằng dấu cách (nối có dấu cách, câu dài ~30-40 ký tự).
        """
        cjk_count = sum(
            1 for ch in text
            if "\u4e00" <= ch <= "\u9fff"  # Hán
            or "\u3040" <= ch <= "\u30ff"  # Hiragana/Katakana
            or "\uac00" <= ch <= "\ud7a3"  # Hangul
        )
        letter_count = sum(1 for ch in text if ch.isalpha())
        if letter_count == 0:
            return False
        return cjk_count / letter_count > 0.3

    @staticmethod
    def _join_words(word_texts: list[str], is_cjk: bool) -> str:
        """Nối các từ thành câu — CJK nối liền, Latin nối bằng dấu cách.

        Với Latin, không thêm dấu cách TRƯỚC dấu câu/đóng ngoặc (vd "word." không
        thành "word ."). WhisperX trả word tiếng Anh KHÔNG kèm dấu cách → phải tự nối.
        """
        if is_cjk:
            return "".join(word_texts)
        parts: list[str] = []
        for token in word_texts:
            tok = token.strip()
            if not tok:
                continue
            if parts and tok[0] not in ".,!?;:)]}'\"”’":
                parts.append(" ")
            parts.append(tok)
        return "".join(parts)

    @classmethod
    def _split_words_into_cues(
        cls,
        words,
        config: TranscriptionConfig | None,
        segment_start: float = 0.0,
        segment_end: float | None = None,
    ) -> list[tuple[str, float, float]]:
        """Tách danh sách từ (có timestamp) thành câu phụ đề ngắn — CJK-aware.

        [v3.22.9] CJK: câu rất ngắn (~5 ký tự), nối liền, dùng rjieba. [v3.23.8]
        Latin (Anh...): từ cách nhau bằng dấu cách, câu dài hơn (~42 ký tự/cue),
        KHÔNG nối liền (sửa lỗi "Helpeveryoneexplore").
        """
        normalized = cls._normalize_words(words, segment_start, segment_end)
        if not normalized:
            return []

        sample_text = "".join(w[0] for w in normalized[:50])
        is_cjk = cls._is_cjk_text(sample_text)

        gap = config.split_gap_sec if config else 0.3
        max_dur = config.max_cue_duration_sec if config else 4.0
        if is_cjk:
            target = config.target_chars_per_cue if config else 4
            max_chars = config.max_chars_per_cue if config else 8
            use_jieba = getattr(config, "use_word_segmentation", True) if config else True
        else:
            # Latin: câu dài hơn nhiều — chuẩn phụ đề ~42 ký tự/dòng.
            target = 28
            max_chars = 42
            use_jieba = False  # rjieba chỉ cho CJK

        word_boundaries: set[int] = set()
        if use_jieba:
            full_text = "".join(w[0] for w in normalized)
            word_boundaries = cls._segment_word_boundaries(full_text)

        cues: list[tuple[str, float, float]] = []
        buf: list[tuple[str, float, float]] = []
        char_offset = 0

        def flush_upto(count: int) -> None:
            nonlocal buf, char_offset
            if count <= 0 or not buf:
                return
            chunk = buf[:count]
            text = cls._join_words([w[0] for w in chunk], is_cjk)
            if text.strip():
                cues.append((text, chunk[0][1], chunk[-1][2]))
            char_offset += sum(len(w[0]) for w in chunk)
            buf = buf[count:]

        def visible_len() -> int:
            # Độ dài hiển thị (gồm dấu cách cho Latin).
            return len(cls._join_words([w[0] for w in buf], is_cjk).strip())

        for i, (word_text, w_start, w_end) in enumerate(normalized):
            buf.append((word_text, w_start, w_end))
            stripped = word_text.strip()
            last_char = stripped[-1:] if stripped else ""
            total_len = visible_len()
            buf_dur = buf[-1][2] - buf[0][1]
            next_gap = normalized[i + 1][1] - w_end if i + 1 < len(normalized) else 0.0

            if last_char in cls._SENTENCE_END_CHARS:
                flush_upto(len(buf))
                continue
            if next_gap >= gap and total_len >= max(2, target // 2):
                flush_upto(len(buf))
                continue
            if last_char in cls._SOFT_BREAK_CHARS and total_len >= target:
                flush_upto(len(buf))
                continue
            if total_len >= max_chars or buf_dur >= max_dur:
                flush_upto(cls._choose_cut_point(buf, target, char_offset, word_boundaries))
                continue

        flush_upto(len(buf))
        return cues

    @staticmethod
    def _choose_cut_point(buf, target: int, char_offset: int, word_boundaries: set[int]) -> int:
        """Chọn vị trí cắt (số từ) tốt nhất.

        Cho mỗi vị trí ứng viên (sau khi đạt target, không giữa số/Latin), tính điểm:
          + ưu tiên gần ``target`` (càng gần càng tốt),
          + cộng thưởng nếu trùng ranh giới TỪ GHÉP (jieba),
          + cộng thưởng theo độ lớn khoảng lặng.
        Lấy vị trí điểm cao nhất → cân bằng giữa độ dài đẹp, ranh giới từ, và nhịp nói.
        """
        best_cut = len(buf)
        best_score = float("-inf")
        cum = 0
        for j in range(len(buf) - 1):
            cum += len(buf[j][0].strip())
            if cum < target:
                continue
            cur_c = buf[j][0].strip()[-1:]
            nxt_c = buf[j + 1][0].strip()[:1]
            if cur_c.isalnum() and cur_c.isascii() and nxt_c.isalnum() and nxt_c.isascii():
                continue
            absolute_char = char_offset + sum(len(buf[k][0]) for k in range(j + 1))
            is_boundary = absolute_char in word_boundaries
            inner_gap = buf[j + 1][1] - buf[j][2]
            # Điểm: phạt độ lệch khỏi target, thưởng ranh giới từ + gap.
            score = -abs(cum - target) * 1.0 + (2.0 if is_boundary else 0.0) + inner_gap * 3.0
            if score > best_score:
                best_score = score
                best_cut = j + 1
        return best_cut

    @staticmethod
    def _normalize_words(
        words, segment_start: float = 0.0, segment_end: float | None = None
    ) -> list[tuple[str, float, float]]:
        """Chuẩn hoá danh sách word của WhisperX về (text, start, end).

        Một số từ (vd số, ký hiệu không trong từ điển align) THIẾU start/end →
        kế thừa mốc thời gian từ từ lân cận để không mất chữ.

        [v3.23.347] SỬA LỖI LỆCH MỐC BẮT ĐẦU. Trước đây ``last_end`` khởi tạo bằng
        ``0.0``, nên khi **từ ĐẦU TIÊN** của đoạn thiếu ``start`` (rất hay gặp: số, ký
        hiệu, từ ngoài từ điển align), nó nhận mốc ``0.0`` — tức đầu phim. Đo trên ví dụ
        thật: đoạn thoại ở giây 125,4 bị đẩy về giây 0 — **lệch 125 giây**.

        Đây đúng là lý do lỗi chỉ xuất hiện "thỉnh thoảng": từ ở GIỮA đoạn kế thừa mốc
        của từ trước (hợp lý), chỉ từ ĐẦU đoạn mới rơi vào mốc 0.

        Nay neo vào ``segment_start`` — mốc mà chính WhisperX đã gán cho đoạn đó.

        Args:
            words: Danh sách word thô từ WhisperX.
            segment_start: Mốc bắt đầu của đoạn chứa các từ này, dùng làm neo khi từ
                đầu tiên thiếu ``start``.
            segment_end: Mốc kết thúc của đoạn. Dùng khi từ CUỐI thiếu ``end`` — nếu
                không, câu sẽ kết thúc ngay tại mốc bắt đầu của từ đó và **cắt cụt tiếng
                nói**. Đo được: mất tới nửa giây cuối câu.

        Returns:
            Danh sách ``(text, start, end)`` đã chuẩn hoá.
        """
        result: list[tuple[str, float, float]] = []
        missing_end_indexes: list[int] = []
        last_end = float(segment_start)
        for word in words:
            if not isinstance(word, dict):
                continue
            text = str(word.get("word", ""))
            if not text:
                continue
            start = word.get("start")
            end = word.get("end")
            end_was_missing = end is None
            try:
                start = float(start) if start is not None else last_end
                end = float(end) if end is not None else start
            except (TypeError, ValueError):
                start, end = last_end, last_end
                end_was_missing = True
            if end_was_missing:
                missing_end_indexes.append(len(result))
            result.append((text, start, end))
            last_end = end

        # Từ thiếu ``end``: ưu tiên lấy mốc bắt đầu của từ KẾ TIẾP (chặt hơn nhiều so
        # với việc để end = start). Từ CUỐI thì lấy mốc kết thúc của đoạn.
        for index in missing_end_indexes:
            text, start, _end = result[index]
            if index + 1 < len(result):
                candidate = result[index + 1][1]
            elif segment_end is not None:
                candidate = float(segment_end)
            else:
                continue
            if candidate > start:
                result[index] = (text, start, candidate)
        return result


def _safe_segment_bound(segment: object, key: str) -> float:
    """Đọc mốc ``start``/``end`` của đoạn, trả ``0.0`` nếu không đọc được.

    Args:
        segment: Đoạn thô từ WhisperX.
        key: ``"start"`` hoặc ``"end"``.

    Returns:
        Mốc thời gian (giây).
    """
    if not isinstance(segment, dict):
        return 0.0
    try:
        value = segment.get(key)
        return float(value) if value is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


__all__ = ["WhisperXAdapter"]
