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


def _shim_torch_scaling_type() -> None:
    """Tiêm các symbol FP8 CHỈ-CÓ-Ở-NIGHTLY vào ``torch.nn.functional`` nếu thiếu.

    Bối cảnh (điều tra v3.23.405–406): một gói trong chuỗi ``from vieneu import Vieneu`` (build
    cho torch NIGHTLY) import các symbol FP8 như ``ScalingType``, ``SwizzleType``,
    ``scaled_grouped_mm``, ``scaled_mm``, ``grouped_mm`` từ ``torch.nn.functional``. Các symbol
    này CHỈ có ở torch nightly; torch ỔN ĐỊNH (2.7/2.8/2.9) CHƯA có → ``ImportError`` chặn cả
    việc import vieneu, khiến TTS GPU rơi về CPU. Đã kiểm chứng: torch 2.8.0 + transformers
    4.52.4 đều KHÔNG có các symbol này (không phải lỗi phiên bản transformers).

    Giải pháp SHIM (giống ``chardet_shim`` của dự án): chèn placeholder để ``from
    torch.nn.functional import <symbol>`` thành công — TTS KHÔNG dùng FP8 nên placeholder chỉ
    cần TỒN TẠI (không bao giờ bị gọi thật). Enum giữ là lớp enum (để dùng được trong kiểu
    ``X | list[X]``); hàm là callable. Có ``__getattr__`` dự phòng theo MẪU tên FP8 để phủ luôn
    symbol nightly mới (chống whack-a-mole), nhưng KHÔNG che symbol thiếu khác (giữ AttributeError
    để feature-detection vẫn đúng).
    """
    import enum

    import torch.nn.functional as functional

    def _make_enum(symbol: str) -> type[enum.Enum]:
        # Thử symbol THẬT từ torch._C trước (torch nightly), không thì tạo enum placeholder.
        try:
            import torch._C as _torch_c

            real = getattr(_torch_c, f"_{symbol}", None)
            if real is not None:
                return real
        except ImportError:
            pass
        return enum.Enum(
            symbol, {"TensorWise": 0, "RowWise": 1, "BlockWise": 2, "Default": 3}
        )

    def _make_func(symbol: str):
        def _placeholder(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError(
                f"torch.nn.functional.{symbol} chỉ có ở torch nightly — "
                "không khả dụng trên torch ổn định (TTS không dùng FP8)."
            )

        _placeholder.__name__ = symbol
        return _placeholder

    for symbol in ("ScalingType", "SwizzleType"):
        if not hasattr(functional, symbol):
            setattr(functional, symbol, _make_enum(symbol))
    for symbol in ("grouped_mm", "scaled_mm", "scaled_grouped_mm"):
        if not hasattr(functional, symbol):
            setattr(functional, symbol, _make_func(symbol))

    # Dự phòng: symbol FP8 nightly MỚI (chưa liệt kê) — chỉ nhận diện theo mẫu tên để KHÔNG
    # che các AttributeError hợp lệ khác.
    _original_getattr = getattr(functional, "__getattr__", None)

    def _fp8_fallback(symbol: str) -> Any:
        if _original_getattr is not None:
            try:
                return _original_getattr(symbol)
            except AttributeError:
                pass
        lowered = symbol.lower()
        is_fp8 = (
            "scal" in lowered
            or "swizzle" in lowered
            or "grouped" in lowered
            or lowered.endswith("_mm")
        )
        if is_fp8:
            return _make_func(symbol)
        raise AttributeError(
            f"module 'torch.nn.functional' has no attribute {symbol!r}"
        )

    functional.__getattr__ = _fp8_fallback  # type: ignore[attr-defined]


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

    _shim_torch_scaling_type()

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


def _coerce_voice_id(value: Any) -> str:
    """[v3.23.408] Ép giá trị giọng về voice_id CHUỖI, chịu được tuple/list và chuỗi-tuple.

    ``engine.get_preset_voice`` chỉ nhận CHUỖI voice_id. Nhưng qua nhiều đường, giá trị có thể
    tới dưới dạng:
    * ``(label, voice_id)`` — tuple từ ``list_preset_voices()``.
    * ``"('label', 'voice_id')"`` — chuỗi hoá của tuple đó (bản cũ str(tuple)).
    * ``"voice_id"`` — chuỗi đúng (giữ nguyên).

    Trả về phần voice_id hợp lý nhất (phần tử cuối của tuple), hoặc chuỗi đã strip.

    Args:
        value: Giá trị giọng thô.

    Returns:
        Chuỗi voice_id (có thể rỗng nếu không suy ra được).
    """
    if isinstance(value, (tuple, list)):
        return str(value[-1]).strip() if value else ""
    text = str(value or "").strip()
    # Chuỗi hoá của tuple: "('Minh Đức — …', 'Minh Đức')" → lấy cụm trong dấu nháy cuối cùng.
    if text.startswith("(") and text.endswith(")") and "," in text:
        import ast

        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (tuple, list)) and parsed:
                return str(parsed[-1]).strip()
        except (ValueError, SyntaxError):
            pass
    return text


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
    # [v3.23.408] preset có thể là chuỗi HOẶC (do bản cũ) chuỗi tuple/tuple thật → chuẩn hoá
    # về voice_id chuỗi trước khi tra.
    preset = _coerce_voice_id(preset)
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
    # [v3.23.408] BUG: list_preset_voices() trả [(label, voice_id), ...]; trước đây truyền CẢ
    # tuple available[0] vào get_preset_voice (cần chuỗi voice_id) → "Voice '(label, id)' not
    # found". Lấy đúng voice_id (phần tử [1] của tuple, hoặc chính nó nếu đã là chuỗi).
    return engine.get_preset_voice(_coerce_voice_id(available[0]))


def _supported_infer_params(engine: Any) -> frozenset[str]:
    """Tham số mà ``infer`` của bản SDK này nhận."""
    import inspect

    try:
        return frozenset(inspect.signature(engine.infer).parameters)
    except (AttributeError, TypeError, ValueError):
        return frozenset()


def _no_surrogates(text: str) -> str:
    """[v3.23.414] Loại lone surrogate khỏi chuỗi (an toàn cho encode UTF-8).

    JSON round-trip qua stdin có thể tái tạo lone surrogate: adapter (bản cũ) escape thành
    ``\\udcXX`` (ASCII, ghi được), ``json.loads`` ở worker dựng lại lone surrogate → khi đưa vào
    model/ghi WAV/echo lỗi sẽ ném ``surrogates not allowed``. Làm sạch NGAY sau khi nhận text.
    """
    try:
        text.encode("utf-8")
        return text
    except UnicodeEncodeError:
        return text.encode("utf-8", "ignore").decode("utf-8")


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
    role_voices: dict[str, Any] = {}
    for index, item in enumerate(items):
        text = _no_surrogates(str(item.get("text") or "").strip())
        output = str(item.get("output") or "")
        if not text or not output:
            results.append({"index": index, "ok": False, "error": "thiếu text/output"})
            continue

        kwargs: dict[str, Any] = {
            "text": text, "voice": _voice_for_item(engine, item, voice, role_voices),
        }
        # Chỉ truyền tham số mà bản SDK này thực sự nhận — bản cũ không có 'style'.
        # [v3.23.663] + use_ref_codes: lần sinh lại câu ngắn bị đọc lặp tắt mã tham chiếu trong ngữ cảnh.
        for name in ("max_new_frames", "style", "repetition_penalty", "temperature", "use_ref_codes"):
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


def _voice_for_item(
    engine: Any, item: dict[str, Any], default_voice: Any, cache: dict[str, Any]
) -> Any:
    """[v3.23.593] Giọng của một câu/lô: ``voice_preset`` của vai, hoặc giọng chung.

    Lồng tiếng nhiều vai gửi kèm ID preset của vai trong từng yêu cầu. Giọng được giải
    quyết MỘT lần cho mỗi ID rồi giữ lại (``cache``) — phim có vài chục vai, mỗi vai hàng
    trăm câu.

    Args:
        engine: Engine đã nạp.
        item: Yêu cầu (câu đơn hoặc lô).
        default_voice: Giọng chung từ dòng cấu hình đầu tiên.
        cache: Bộ nhớ giọng theo ID preset (sửa tại chỗ).

    Returns:
        Dữ liệu giọng truyền vào ``infer``/``infer_batch``.
    """
    preset = _coerce_voice_id(item.get("voice_preset") or "")
    if not preset:
        return default_voice
    if preset not in cache:
        # [v3.23.594] Gọi thẳng get_preset_voice: _resolve_voice lùi về giọng preset ĐẦU
        # TIÊN khi không có giọng — vai thiếu giọng sẽ mượn giọng của vai khác. Lùi về
        # giọng chung như đường CPU (_resolve_line_voice) thì dễ nhận ra hơn.
        try:
            cache[preset] = engine.get_preset_voice(preset)
        except (KeyError, ValueError, RuntimeError, OSError, TypeError) as exc:
            _eprint(f"VIENEU_GPU_VOICE_FALLBACK: {preset} -> giọng chung ({exc})")
            cache[preset] = default_voice
    return cache[preset]


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
              (tuỳ chọn "voice_preset": "<ID>" — giọng riêng của vai khi lồng tiếng)
        ra  : {"ok": true, "output": "…/1.wav"}   hoặc   {"ok": false, "error": "…"}
        vào : {"quit": true}      -> thoát

    Dòng đầu tiên PHẢI là cấu hình (mode/voice_preset/reference_wav/sample_rate).

    Returns:
        Mã thoát.
    """
    # [v3.23.414] Bảo hiểm cấp STREAM: đọc/ghi UTF-8 với errors="replace" để lone surrogate
    # (nếu lọt qua JSON round-trip: escape "\udcXX" → json.loads tái tạo lone surrogate) KHÔNG
    # làm chết worker khi ghi reply. Kết hợp làm sạch text sau json.loads bên dưới.
    for _stream in (sys.stdin, sys.stdout):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, OSError, ValueError):
            pass

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
    role_voices: dict[str, Any] = {}

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

        # [v3.23.412] Lệnh BATCH: {"batch":[{"text","output"},...]} → infer_batch (GPU gộp
        # chunk mọi câu, chia sẻ forward → nhanh hơn nhiều). Chung MỘT giọng (đã resolve).
        item_voice = _voice_for_item(engine, item, voice, role_voices)
        batch = item.get("batch")
        if isinstance(batch, list) and batch:
            _handle_batch(engine, batch, item_voice, supported, sample_rate, item)
            continue

        text = _no_surrogates(str(item.get("text") or "").strip())
        output = str(item.get("output") or "")
        if not text or not output:
            print(
                json.dumps({"ok": False, "error": "thiếu text/output"}), flush=True
            )
            continue

        kwargs: dict[str, Any] = {"text": text, "voice": item_voice}
        # [v3.23.663] + use_ref_codes: lần sinh lại câu ngắn bị đọc lặp tắt mã tham chiếu trong ngữ cảnh.
        for name in ("max_new_frames", "style", "repetition_penalty", "temperature", "use_ref_codes"):
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


def _handle_batch(
    engine: Any,
    batch: list[dict[str, Any]],
    voice: Any,
    supported: frozenset[str],
    sample_rate: int,
    item: dict[str, Any],
) -> None:
    """[v3.23.412] Xử lý một LÔ câu bằng ``engine.infer_batch`` rồi ghi từng WAV.

    Trả về ``{"ok": True, "batch": [output,...]}`` theo ĐÚNG thứ tự đầu vào; câu nào lỗi thì
    ``output`` là chuỗi rỗng (adapter sẽ tự làm lại per-câu). Nếu SDK không có ``infer_batch``
    hoặc lỗi toàn lô → ``{"ok": False}`` để adapter lùi về per-câu.

    Args:
        engine: Instance VieNeu.
        batch: Danh sách ``{"text","output"}``.
        voice: Dữ liệu giọng đã resolve.
        supported: Tham số ``infer`` hỗ trợ (để lọc kwargs chung).
        sample_rate: SR ghi WAV.
        item: Job gốc (để lấy tham số chung như temperature).
    """
    if not hasattr(engine, "infer_batch"):
        print(json.dumps({"ok": False, "error": "SDK không có infer_batch"}), flush=True)
        return

    texts = [_no_surrogates(str(entry.get("text") or "").strip()) for entry in batch]
    outputs = [str(entry.get("output") or "") for entry in batch]

    kwargs: dict[str, Any] = {"voice": voice}
    for name in ("max_new_frames", "repetition_penalty", "temperature"):
        if name in item and item[name] is not None and name in supported:
            kwargs[name] = item[name]
    # [v3.23.420] Số chunk/forward GPU (đòn bẩy VRAM/tốc độ). Mặc định vieneu = 4 (phí VRAM);
    # adapter gửi giá trị theo VRAM (8GB→32…). Tên tham số KHÁC nhau theo mode: v3turbo dùng
    # ``batch_size``, fast/turbo dùng ``max_batch_size`` → dò signature để truyền đúng tên.
    gpu_subbatch = item.get("gpu_subbatch")
    if gpu_subbatch:
        try:
            import inspect

            batch_params = inspect.signature(engine.infer_batch).parameters
            value = max(1, int(gpu_subbatch))
            if "batch_size" in batch_params:
                kwargs["batch_size"] = value
            elif "max_batch_size" in batch_params:
                kwargs["max_batch_size"] = value
        except (TypeError, ValueError, AttributeError):
            pass

    try:
        audios = engine.infer_batch(texts, **kwargs)
    except Exception as exc:  # noqa: BLE001 — lỗi cả lô → để adapter lùi per-câu
        _eprint(f"VIENEU_GPU_BATCH_ERROR: {exc}")
        print(json.dumps({"ok": False, "error": str(exc)}), flush=True)
        return

    written: list[str] = []
    for audio, output in zip(audios, outputs):
        try:
            if output and audio is not None and getattr(audio, "size", 0) > 0:
                _write_wav(output, audio, sample_rate)
                written.append(output)
            else:
                written.append("")  # câu rỗng → adapter làm lại per-câu
        except Exception as exc:  # noqa: BLE001
            _eprint(f"VIENEU_GPU_BATCH_WRITE_ERROR: {exc}")
            written.append("")
    print(json.dumps({"ok": True, "batch": written}), flush=True)


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
