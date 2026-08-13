"""Worker tổng hợp giọng VieNeu bằng GPU, chạy trong MÔI TRƯỜNG RIÊNG.

VÌ SAO cần tệp này
==================
v3.23.343 thêm nút cài ``vieneu`` vào ``whisperx_env`` — nhưng **thiếu mảnh ghép cuối**:
adapter vẫn nạp VieNeu bằng ``from vieneu import Vieneu`` NGAY TRONG tiến trình chính,
nơi ``torch`` cố ý không được đóng gói (nó xung đột CUDA với paddle). Kết quả: VieNeu
gặp ``ImportError`` rồi âm thầm lùi về ONNX/CPU, dù người dùng đã cài đủ gói.

Tệp này chạy bằng trình thông dịch của môi trường riêng — nơi CÓ torch bản CUDA.

Giao thức
---------
Nhận một tệp JSON mô tả công việc, ghi WAV ra đường dẫn yêu cầu, in JSON kết quả::

    {
      "voice_preset": "Minh Đức",       # hoặc "reference_wav": "…"
      "mode": "v3turbo",
      "items": [
        {"text": "…", "output": "…/0.wav", "max_new_frames": 350, "style": "tu_nhien"}
      ]
    }

Mã thoát: 0 = xong, 2 = tham số sai, 3 = thiếu thư viện, 4 = không có GPU, 5 = lỗi khác.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

#: Mã thoát — adapter dựa vào đây để phân biệt nguyên nhân, đừng đổi tuỳ tiện.
EXIT_OK = 0
EXIT_BAD_ARGS = 2
EXIT_LIBRARY_MISSING = 3
EXIT_NO_GPU = 4
EXIT_RUNTIME_ERROR = 5


def _eprint(message: str) -> None:
    """In ra stderr với UTF-8, để chữ có dấu không bị hỏng trên Windows."""
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError, ValueError):
        pass
    print(message, file=sys.stderr, flush=True)


def _load_engine(mode: str) -> Any:
    """Nạp engine VieNeu ở chế độ PyTorch/CUDA.

    Args:
        mode: Chế độ model (vd ``"v3turbo"``).

    Returns:
        Đối tượng engine.

    Raises:
        SystemExit: Khi thiếu thư viện hoặc không có GPU.
    """
    try:
        import torch
    except ImportError as exc:
        _eprint(f"VIENEU_GPU_NO_TORCH: {exc}")
        raise SystemExit(EXIT_LIBRARY_MISSING) from exc

    if not torch.cuda.is_available():
        _eprint(
            "VIENEU_GPU_UNAVAILABLE: torch có nhưng không thấy GPU CUDA dùng được."
        )
        raise SystemExit(EXIT_NO_GPU)

    try:
        from vieneu import Vieneu
    except ImportError as exc:
        _eprint(f"VIENEU_GPU_NO_VIENEU: {exc}")
        raise SystemExit(EXIT_LIBRARY_MISSING) from exc

    # [v3.23.364] backend="pytorch" là ĐIỂM MẤU CHỐT: để "auto" thì VieNeu vẫn có thể
    # chọn ONNX (hardcode CPUExecutionProvider → không dùng được GPU). Nhưng CHỮ KÝ SDK
    # ĐÃ THAY ĐỔI qua các phiên bản — thử lần lượt các chữ ký đã biết, nếu TypeError
    # (kwargs không được nhận) thì thử biến thể kế, kèm ghi rõ để chẩn đoán.
    signatures = (
        {"mode": mode, "backend": "pytorch", "device": "cuda"},
        {"mode": mode, "device": "cuda"},
        {"mode": mode, "backbone_device": "cuda", "codec_device": "cuda"},
        {"mode": mode},
    )
    last_error: Exception | None = None
    for kwargs in signatures:
        try:
            return Vieneu(**kwargs)
        except TypeError as exc:
            last_error = exc
            _eprint(f"VIENEU_GPU_SIG_SKIP: {kwargs} -> {exc}")
            continue
    _eprint(f"VIENEU_GPU_INIT_FAIL: mọi chữ ký đều lỗi. Cuối: {last_error}")
    raise SystemExit(EXIT_LIBRARY_MISSING)


def _resolve_voice(engine: Any, job: dict[str, Any]) -> Any:
    """Lấy dữ liệu giọng, ưu tiên tệp nhân bản nếu có.

    Giải quyết giọng NGAY TRONG worker để không phải truyền embedding qua ranh giới
    tiến trình (chúng là mảng lớn, tuần tự hoá vừa chậm vừa dễ sai).

    Args:
        engine: Engine đã nạp.
        job: Mô tả công việc.

    Returns:
        Dữ liệu giọng truyền vào ``infer``.

    Raises:
        SystemExit: Khi không giải quyết được giọng nào.
    """
    reference = job.get("reference_wav")
    if reference and Path(reference).is_file():
        return engine.encode_reference(reference)

    preset = job.get("voice_preset") or ""
    if preset:
        try:
            return engine.get_preset_voice(preset)
        except Exception as exc:  # noqa: BLE001 — tên giọng có thể đã đổi giữa các bản
            _eprint(f"VIENEU_GPU_VOICE_FALLBACK: {preset} -> {exc}")

    try:
        available = list(engine.list_preset_voices())
    except Exception as exc:  # noqa: BLE001
        _eprint(f"VIENEU_GPU_NO_VOICE: {exc}")
        raise SystemExit(EXIT_RUNTIME_ERROR) from exc
    if not available:
        _eprint("VIENEU_GPU_NO_VOICE: engine không có giọng preset nào.")
        raise SystemExit(EXIT_RUNTIME_ERROR)
    return engine.get_preset_voice(available[0])


def _supported_infer_params(engine: Any) -> frozenset[str]:
    """Tham số mà ``infer`` của bản SDK này nhận."""
    import inspect

    try:
        return frozenset(inspect.signature(engine.infer).parameters)
    except (AttributeError, TypeError, ValueError):
        return frozenset()


def _write_wav(path: str, samples: Any, sample_rate: int) -> None:
    """Ghi mảng mẫu ra WAV.

    Args:
        path: Tệp đích.
        samples: Mảng mẫu (numpy hoặc tensor torch).
        sample_rate: Tần số lấy mẫu.
    """
    import numpy as np
    import soundfile as sf

    if hasattr(samples, "detach"):        # tensor torch -> numpy
        samples = samples.detach().cpu().numpy()
    array = np.asarray(samples, dtype="float32").reshape(-1)
    sf.write(path, array, sample_rate)


def run_job(job_path: Path) -> int:
    """Thực hiện toàn bộ công việc mô tả trong tệp JSON.

    Args:
        job_path: Tệp JSON mô tả công việc.

    Returns:
        Mã thoát.
    """
    try:
        job = json.loads(job_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        _eprint(f"VIENEU_GPU_BAD_JOB: {exc}")
        return EXIT_BAD_ARGS

    items = job.get("items") or []
    if not items:
        _eprint("VIENEU_GPU_BAD_JOB: danh sách 'items' rỗng.")
        return EXIT_BAD_ARGS

    engine = _load_engine(str(job.get("mode") or "v3turbo"))
    voice = _resolve_voice(engine, job)
    supported = _supported_infer_params(engine)
    sample_rate = int(job.get("sample_rate") or 48000)

    results: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        text = str(item.get("text") or "").strip()
        output = str(item.get("output") or "")
        if not text or not output:
            results.append({"index": index, "ok": False, "error": "thiếu text/output"})
            continue

        kwargs: dict[str, Any] = {"text": text, "voice": voice}
        # Chỉ truyền tham số mà bản SDK này thực sự nhận — bản cũ không có 'style'.
        for name in ("max_new_frames", "style", "repetition_penalty", "temperature"):
            if name in item and item[name] is not None and name in supported:
                kwargs[name] = item[name]

        try:
            audio = engine.infer(**kwargs)
            _write_wav(output, audio, sample_rate)
            results.append({"index": index, "ok": True, "output": output})
        except Exception as exc:  # noqa: BLE001 — một câu lỗi không được phá cả lô
            _eprint(f"VIENEU_GPU_ITEM_ERROR[{index}]: {exc}")
            results.append({"index": index, "ok": False, "error": str(exc)})

    print(json.dumps({"results": results}, ensure_ascii=False), flush=True)
    return EXIT_OK


def serve() -> int:
    """Chế độ THƯỜNG TRÚ: nạp model một lần rồi phục vụ nhiều câu.

    [v3.23.349] SỬA LỖI "GPU chậm hơn CPU". Bản trước chạy worker MỘT LẦN CHO MỖI CÂU:
    mỗi lần phải khởi động Python, nạp torch, nạp model VieNeu (~15 giây) chỉ để tổng
    hợp một câu mất 0,1 giây. Với 55 câu là ~14 phút — chậm hơn hẳn CPU (27 giây), và
    nháy 55 cửa sổ console.

    Nay giữ tiến trình sống: nạp model MỘT lần, rồi đọc từng yêu cầu JSON một dòng từ
    stdin và trả kết quả JSON một dòng ra stdout.

    Giao thức (mỗi bên một dòng JSON)::

        vào : {"text": "…", "output": "…/1.wav", "max_new_frames": 350}
        ra  : {"ok": true, "output": "…/1.wav"}   hoặc   {"ok": false, "error": "…"}
        vào : {"quit": true}      -> thoát

    Dòng đầu tiên PHẢI là cấu hình (mode/voice_preset/reference_wav/sample_rate).

    Returns:
        Mã thoát.
    """
    first = sys.stdin.readline()
    if not first.strip():
        _eprint("VIENEU_GPU_BAD_JOB: thiếu dòng cấu hình đầu tiên.")
        return EXIT_BAD_ARGS
    try:
        config = json.loads(first)
    except ValueError as exc:
        _eprint(f"VIENEU_GPU_BAD_JOB: {exc}")
        return EXIT_BAD_ARGS

    # [v3.23.364] Bọc phần nạp model: mọi lỗi (thiếu torch/vieneu, không GPU, đổi chữ ký
    # Vieneu(), CUDA OOM…) đều BÁO LÝ DO qua stdout ({"ready": false, "error": …}) để phía
    # gọi ghi log được nguyên nhân — thay vì crash câm khiến chỉ thấy "không báo sẵn sàng".
    try:
        engine = _load_engine(str(config.get("mode") or "v3turbo"))
        voice = _resolve_voice(engine, config)
        supported = _supported_infer_params(engine)
        sample_rate = int(config.get("sample_rate") or 48000)
    except SystemExit as exc:
        reason = {
            EXIT_LIBRARY_MISSING: "thiếu thư viện (torch/vieneu) trong môi trường GPU",
            EXIT_NO_GPU: "torch không thấy GPU CUDA dùng được",
        }.get(int(exc.code) if isinstance(exc.code, int) else -1, "nạp engine thất bại")
        print(json.dumps({"ready": False, "error": reason}, ensure_ascii=False), flush=True)
        return int(exc.code) if isinstance(exc.code, int) else EXIT_LIBRARY_MISSING
    except BaseException as exc:  # noqa: BLE001 — BIÊN TIẾN TRÌNH: báo lý do rồi thoát sạch
        import traceback

        _eprint("VIENEU_GPU_LOAD_CRASH:\n" + traceback.format_exc())
        print(
            json.dumps(
                {"ready": False, "error": f"{type(exc).__name__}: {exc}"},
                ensure_ascii=False,
            ),
            flush=True,
        )
        return EXIT_LIBRARY_MISSING

    # Báo cho phía gọi biết đã sẵn sàng — nó đợi dòng này trước khi gửi câu đầu tiên.
    print(json.dumps({"ready": True}), flush=True)

    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            item = json.loads(raw)
        except ValueError as exc:
            print(json.dumps({"ok": False, "error": f"JSON hỏng: {exc}"}), flush=True)
            continue
        if item.get("quit"):
            break

        text = str(item.get("text") or "").strip()
        output = str(item.get("output") or "")
        if not text or not output:
            print(
                json.dumps({"ok": False, "error": "thiếu text/output"}), flush=True
            )
            continue

        kwargs: dict[str, Any] = {"text": text, "voice": voice}
        for name in ("max_new_frames", "style", "repetition_penalty", "temperature"):
            if name in item and item[name] is not None and name in supported:
                kwargs[name] = item[name]

        try:
            audio = engine.infer(**kwargs)
            _write_wav(output, audio, sample_rate)
            print(json.dumps({"ok": True, "output": output}), flush=True)
        except Exception as exc:  # noqa: BLE001 — một câu lỗi không được giết worker
            _eprint(f"VIENEU_GPU_ITEM_ERROR: {exc}")
            print(json.dumps({"ok": False, "error": str(exc)}), flush=True)

    return EXIT_OK


def main() -> int:
    """Điểm vào dòng lệnh."""
    parser = argparse.ArgumentParser(description="VieNeu-TTS GPU worker")
    parser.add_argument("--job", help="Tệp JSON mô tả công việc (chế độ một lượt)")
    parser.add_argument(
        "--serve", action="store_true",
        help="Chế độ thường trú: nạp model một lần, phục vụ nhiều câu qua stdin.",
    )
    args = parser.parse_args()

    if not args.serve and not args.job:
        _eprint("VIENEU_GPU_BAD_JOB: cần --job hoặc --serve.")
        return EXIT_BAD_ARGS

    try:
        if args.serve:
            return serve()
        return run_job(Path(args.job))
    except SystemExit as exc:
        return int(exc.code or EXIT_RUNTIME_ERROR)
    except Exception as exc:  # noqa: BLE001
        _eprint(f"VIENEU_GPU_FATAL: {exc}")
        return EXIT_RUNTIME_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
