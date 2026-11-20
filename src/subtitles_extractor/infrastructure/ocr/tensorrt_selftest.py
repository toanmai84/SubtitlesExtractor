"""[v3.23.766] Tự kiểm TensorRT trong TIẾN TRÌNH RIÊNG trước khi tiến trình chính dùng nó.

Log + ``fault.log`` 19/11 14:46 (bản 3.23.763) và 15:46 (bản 3.23.765): engine TensorRT dựng xong, nhưng lần suy luận
đầu của bộ dò làm SẬP cả ứng dụng ("Windows fatal exception: access violation" trong ``predictor.run()``). Bản 3.23.765
đã chia lô vừa hồ sơ engine mà vẫn sập — và mã C++ của Paddle (``tensorrt_engine_instruction.cc``) tự kiểm hình dạng
theo hồ sơ trước khi chạy (``RuntimeDynamicShapeCheck`` → lỗi Python), nên vụ sập nằm sâu hơn, trong ``Execute`` /
``enqueueV3`` của TensorRT. Lỗi sập trong thư viện C++ KHÔNG bắt được bằng Python.

Cách duy nhất chắc chắn bảo vệ ứng dụng: chạy thử đúng cấu hình ở tiến trình con, ảnh tổng hợp cỡ dải phụ đề. Con sập /
treo / lỗi → tiến trình chính ghi lý do (mã thoát + ``fault`` của con) và chạy GPU thường; con chạy được → tiến trình
chính dùng TensorRT (engine con đã dựng nằm trong cache — không dựng lại). [v3.23.767] Bố cục kiểm trước: "luồng Paddle
riêng" (nạp paddle + dựng + OCR một luồng — bố cục ứng dụng dùng); không đạt → chạy thêm bố cục cũ "khác luồng" để chẩn
đoán. Kết quả nhớ theo bản ứng dụng + cấu hình (``selftest.json``): chỉ kiểm lại khi một trong hai thay đổi.
"""

from __future__ import annotations

import json
import logging
import multiprocessing
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

logger = logging.getLogger(__name__)

#: Lần đầu tiến trình con còn DỰNG engine (bộ dò + bộ nhận dạng, ~2 phút trên RTX 3070 Ti) — để rộng.
SELFTEST_TIMEOUT_SEC: Final[float] = 1500.0
RESULT_FILE: Final[str] = "selftest.json"
FAULT_FILE: Final[str] = "selftest_fault.log"
#: [v3.23.767] Bố cục luồng — Paddle giữ trạng thái RIÊNG TỪNG LUỒNG (kiểm bằng paddle thật trên máy đám mây:
#: ``load_inference_model`` chạy ở luồng đã nạp paddle nhưng LỖI "only supported in static graph mode" ở luồng mới, dù
#: ``in_dynamic_mode()`` như nhau — đúng lỗi tự kiểm 3.23.766 báo, log 19/11 16:32).
#: Đo tiếp: CHỈ luồng đã nạp paddle làm được — luồng khác, kể cả luồng chính Python, đều lỗi.
#: * luồng Paddle riêng: nạp paddle + dựng engine + OCR đều ở MỘT luồng cố định — bố cục ứng dụng dùng khi bật TensorRT
#:   (3.23.767);
#: * khác luồng: nạp + dựng ở luồng A, OCR ở luồng B — bố cục CŨ của ứng dụng (3.23.763–766: nạp sẵn ở luồng nền, OCR ở
#:   luồng trích xuất — các lần "access violation"); chỉ chạy để chẩn đoán khi bố cục chính không đạt.
VARIANT_DEDICATED_THREAD: Final[str] = "luồng Paddle riêng"
VARIANT_CROSS_THREAD: Final[str] = "khác luồng"
#: [v3.23.770] Phép thử THÀNH PHẦN khi bố cục chính SẬP (log 19/11 19:14, bản 3.23.769: sập ngay cả khi nạp + dựng +
#: chạy cùng một luồng → không phải lỗi luồng) — mỗi phép một tiến trình con, khoanh vùng engine / bản vá gây sập.
#: (tên, mô hình "det"|"rec", độ chính xác, biến môi trường đặt trong con)
COMPONENT_VARIANTS: Final[tuple[tuple[str, str, str, dict[str, str]], ...]] = (
    ("chỉ bộ dò", "det", "fp32", {}),
    ("chỉ bộ nhận dạng", "rec", "fp32", {}),
    ("bộ dò, hồ sơ gốc PaddleX", "det", "fp32", {"SE_TRT_DET_PROFILE": "0"}),
    ("bộ dò, không ép Int32", "det", "fp32", {"SE_TRT_SHAPE_FIX": "0"}),
    ("bộ dò FP16", "det", "fp16", {}),
)
_COMPONENTS: Final[dict[str, tuple[str, str, dict[str, str]]]] = {
    name: (kind, precision, env) for name, kind, precision, env in COMPONENT_VARIANTS}
#: [v3.23.771] Log 19/11 20:31 (bản 3.23.770): CẢ 5 phép thử thành phần đều SẬP ở ``predictor.run()`` (bộ dò, bộ nhận
#: dạng, hồ sơ gốc PaddleX, không ép Int32, FP16) → không phải bản vá / mô hình / luồng: mọi engine Paddle chạy đều sập.
#: Phép thử mới: nạp THẲNG tệp engine Paddle đã dựng (``engine_*.trt``) bằng TensorRT, chạy bằng bộ đệm CuPy — KHÔNG qua
#: Paddle. Chạy được → lỗi nằm ở lớp Paddle C++ ↔ TensorRT 10.9 (Paddle biên dịch với 10.5); sập → lỗi ở TensorRT/driver.
VARIANT_DIRECT_ENGINE: Final[str] = "engine trực tiếp (TensorRT, không qua Paddle)"
#: [v3.23.774] Log 19/11 21:34 (bản 3.23.773): engine trực tiếp CHẠY ĐƯỢC (bộ dò lô 8 ảnh 1600×288: 18,6 ms) → ứng dụng TỰ
#: chạy engine (``trt_direct_runner``). Tự kiểm: (1) dựng engine bằng bước chuyển của Paddle — Paddle có thể sập ở bước
#: dọn sau khi đã GHI tệp engine (log 21:12, ``converter.py:641``), không sao; (2) OCR bằng Paddle GPU thường làm mốc, gắn
#: runner TensorRT trực tiếp, chạy lại và SO kết quả (chữ phải giống hệt, cùng số hộp) + đo tốc độ.
VARIANT_BUILD_ENGINES: Final[str] = "dựng engine (bước chuyển của Paddle)"
#: Rà độc lập 3.23.774: Paddle sập khi chuyển bộ dò thì bộ nhận dạng chưa kịp chuyển → mỗi mô hình một tiến trình con.
VARIANT_BUILD_DET: Final[str] = "dựng engine bộ dò"
VARIANT_BUILD_REC: Final[str] = "dựng engine bộ nhận dạng"
VARIANT_DIRECT_APP: Final[str] = "TensorRT trực tiếp (ứng dụng chạy engine)"
#: Ảnh tổng hợp cỡ dải phụ đề thật (phim 720×1280: ROI 720×118 phóng cao 256 + viền 8 → 1578×272).
SYNTHETIC_SIZE: Final[tuple[int, int]] = (1578, 272)


@dataclass(frozen=True, slots=True)
class SelfTestOutcome:
    ok: bool
    reason: str
    cached: bool = False
    #: [v3.23.766] Kết quả CHẮC CHẮN (chạy được / sập thật / lỗi xác định) mới được ghi nhớ; treo, thiếu VRAM, tiến trình
    #: bị dừng là trục trặc tạm thời — lần sau kiểm lại (rà độc lập: ghi nhớ chúng sẽ tắt TensorRT tới bản sau).
    definitive: bool = True
    #: [v3.23.774] Dữ liệu kèm (vd dấu tệp engine đã kiểm: ``{"engines": {"text_det_model": {...}, …}}``).
    data: dict[str, Any] | None = None


#: Tham số dựng mô hình đi vào ENGINE TensorRT (ngưỡng dò / nhận dạng, số luồng CPU… không — đổi chúng không kiểm lại).
_ENGINE_KEYS: Final[tuple[str, ...]] = (
    "device", "lang", "ocr_version", "text_detection_model_name", "text_recognition_model_name", "use_tensorrt",
    "precision", "use_textline_orientation", "use_doc_orientation_classify", "use_doc_unwarping",
)
_TRANSIENT_MARKERS: Final[tuple[str, ...]] = ("out of memory", "outofmemory", "cudaerrormemoryallocation", "oom ",
                                              "insufficient memory", "alloc failed")
#: Mã thoát do bị DỪNG từ ngoài (không phải sập): POSIX SIGTERM / SIGKILL; Windows ``TerminateProcess`` của
#: multiprocessing (mã 0x10000 = 65536) và 1.
_KILLED_CODES: Final[frozenset[int]] = frozenset({-15, -9, 1, 0x10000})
_LOCK = threading.Lock()


def selftest_signature(app_version: str, init_kwargs: dict[str, Any], det_batch: int) -> str:
    """Khoá kết quả: bản ứng dụng + bản TensorRT + tham số đi vào engine + cỡ lô bộ dò."""
    from subtitles_extractor.infrastructure.ocr.tensorrt_runtime_plan import (
        TENSORRT_VERSION,
        trt_batches,
    )

    engine = {key: init_kwargs.get(key) for key in _ENGINE_KEYS}
    return json.dumps({"app": app_version, "trt": TENSORRT_VERSION, "engine": engine, "det_batch": int(det_batch),
                       "engine_batches": list(trt_batches())},                    # [v3.23.776] cỡ lô trong cài đặt
                      sort_keys=True, default=str)


def _load(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def synthetic_images(count: int) -> list[Any]:
    """Ảnh BGR nền tối, chữ trắng (đủ để bộ dò thấy hộp chữ và bộ nhận dạng chạy — kiểm trên CPU: PP-OCRv6_small đọc
    đúng "Subtitles Extractor 000 TensorRT")."""
    import cv2
    import numpy as np

    width, height = SYNTHETIC_SIZE
    images = []
    for index in range(max(1, count)):
        image = np.full((height, width, 3), 30, np.uint8)
        cv2.putText(image, f"Subtitles Extractor {index:03d} TensorRT", (60, height // 2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.4, (255, 255, 255), 6, cv2.LINE_AA)
        images.append(image)
    return images


def _watch_parent() -> None:  # pragma: no cover — tiến trình con
    """Tiến trình chính chết (bị tắt cưỡng bức / sập vì lý do khác) → con tự thoát, không giữ VRAM / cache engine."""
    parent = multiprocessing.parent_process()
    if parent is None:
        return

    def watch() -> None:
        while True:
            time.sleep(2.0)
            if not parent.is_alive():
                os._exit(3)

    threading.Thread(target=watch, name="selftest-parent-watch", daemon=True).start()


def _rearm_faulthandler(fault_file: Any) -> None:  # pragma: no cover — tiến trình con
    """Paddle cài bộ bắt SIGSEGV riêng ĐÈ lên faulthandler (xem ``crash_hooks.rearm``, v3.23.650; rà độc lập 3.23.766
    tái hiện: bật trước ``import paddle`` → tệp fault rỗng) — bật lại sau mỗi bước nạp Paddle."""
    import faulthandler

    try:
        faulthandler.disable()
        faulthandler.enable(fault_file, all_threads=True)
    except (OSError, ValueError, RuntimeError):
        pass


def _selftest_worker(conn: Any, variant: str, payload: dict[str, Any], det_batch: int, fault_path: str,
                     prepare: bool) -> None:  # pragma: no cover — chạy ở tiến trình con (exe Windows + GPU)
    """[v3.23.768] Tiến trình con chạy ĐÚNG đường mã của ứng dụng: ``PaddleOcrAdapter`` thật với cấu hình của người dùng,
    ``initialize()`` từ một luồng như luồng nạp sẵn, ``infer_batch()`` từ một luồng như luồng trích xuất — mọi lớp vá /
    luồng Paddle chạy y như tiến trình chính. Gửi ``("ok"|"error", mô tả + log chẩn đoán)``.

    Bản 3.23.766–767 tự dựng PaddleOCR riêng trong con (KHÔNG giống tiến trình chính) → log 19/11 17:23: cả hai bố cục lỗi
    "load_inference_model() is only supported in static graph mode" mà không tái hiện được trên máy đám mây.
    """
    import logging as _logging

    fault_file = None
    records: list[str] = []

    class _Collect(_logging.Handler):
        def emit(self, record: _logging.LogRecord) -> None:
            try:
                records.append(f"{record.levelname} {record.name.rsplit('.', 1)[-1]}: {record.getMessage()}"[:1500])
            except Exception:  # noqa: BLE001
                pass

    def diagnostics() -> str:
        tail = [line for line in records if not line.startswith("DEBUG")][-14:]
        return (" || nhật ký con: " + " | ".join(tail)) if tail else ""

    import builtins
    import sys as _sys

    # [v3.23.769] Log 19/11 18:47 (bản 3.23.768): trong tiến trình con trên Windows "paddle đã nạp trước khi dựng: True"
    # (trên Linux: False) — có mã nạp paddle SỚM trên luồng chính của con → luồng Paddle không phải luồng nạp → bộ chuyển
    # đổi tắt TensorRT (và trước đó 3.23.766–767: "static graph mode"). Ghi lại AI nạp paddle (luồng + ngăn xếp).
    preloaded_at_start = "paddle" in _sys.modules
    import_origin: list[str] = []
    real_import = builtins.__import__

    def spy_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if not import_origin and (name == "paddle" or name.startswith("paddle.")) and "paddle" not in _sys.modules:
            import traceback

            stack = [line.strip() for line in traceback.format_stack(limit=14)[:-1]]
            import_origin.append(f"{threading.current_thread().name}: " + " < ".join(
                line.split("\n")[0] for line in reversed(stack))[:900])
        return real_import(name, *args, **kwargs)

    try:
        builtins.__import__ = spy_import
        _watch_parent()
        os.environ["SE_TRT_SELFTEST"] = "0"            # không tự kiểm lồng trong tiến trình tự kiểm
        os.environ["SE_TRT_SELFTEST_CHILD"] = "1"      # không ghi dấu chống sập của tiến trình chính
        if variant == VARIANT_CROSS_THREAD:
            os.environ["SE_TRT_LEGACY_THREADS"] = "1"  # bố cục CŨ — chỉ để chẩn đoán
        if prepare:
            from subtitles_extractor.composition.child_process_setup import (
                prepare_child_process,
            )

            prepare_child_process()
        if os.environ.get("SE_TEST_IMPORT_PADDLE_IN_PREPARE") == "1":    # chỉ cho kiểm thử: mã "lạ" nạp paddle sớm
            import paddle  # noqa: F401
        handler = _Collect(level=_logging.DEBUG)
        root = _logging.getLogger("subtitles_extractor")
        root.addHandler(handler)
        root.setLevel(_logging.DEBUG)
        fault_file = open(fault_path, "w", encoding="utf-8")  # noqa: SIM115 — giữ mở tới khi tiến trình kết thúc
        _rearm_faulthandler(fault_file)
        import sys as _sys

        from subtitles_extractor.infrastructure.ocr import (
            paddle_ocr_adapter as adapter_module,
        )

        builtins.__import__ = real_import
        preloaded = "paddle" in _sys.modules
        records.append(f"INFO selftest: paddle đã nạp trước khi dựng: {preloaded} (lúc con bắt đầu: "
                       f"{preloaded_at_start}){' — nạp bởi ' + import_origin[0] if import_origin else ''}")
        on_main_thread = False
        if preloaded:
            # Paddle đã nạp trên luồng CHÍNH của con (lúc đó con chỉ có luồng này) → chạy bố cục trên luồng chính:
            # "luồng Paddle riêng" = nạp + dựng + OCR cùng luồng chính; "khác luồng" = dựng ở luồng chính, OCR ở luồng
            # khác. Đúng quy tắc đã đo (chỉ luồng nạp paddle chuyển được TensorRT) — như tiến trình chính làm.
            adapter_module._PADDLE_IMPORT_THREAD = adapter_module._PADDLE_IMPORT_THREAD or threading.main_thread().ident
            os.environ["SE_TRT_LEGACY_THREADS"] = "1"
            on_main_thread = True
            records.append("INFO selftest: chạy bố cục trên luồng chính của con (luồng đã nạp paddle)")
        adapter = adapter_module.PaddleOcrAdapter(payload["config"])
        holder: dict[str, Any] = {}

        def in_thread(name: str, work: Callable[[], Any]) -> bool:
            def body() -> None:
                try:
                    holder[name] = work()
                except BaseException as exc:  # noqa: BLE001 — chuyển về tiến trình chính
                    import traceback

                    frames = traceback.format_exception(type(exc), exc, exc.__traceback__)
                    holder["error"] = f"{name}: {type(exc).__name__}: {exc} @ {''.join(frames[-4:])[-900:]}"

            same_thread = on_main_thread and (name == "nạp-sẵn" or variant == VARIANT_DEDICATED_THREAD)
            if same_thread:
                body()
            else:
                thread = threading.Thread(target=body, name=f"selftest-{name}")
                thread.start()
                thread.join()
            _rearm_faulthandler(fault_file)
            return "error" not in holder

        started = time.monotonic()
        if not in_thread("nạp-sẵn", adapter.initialize):
            conn.send(("error", holder["error"] + diagnostics()))
            return
        records.append(f"INFO selftest: luồng nạp paddle = luồng Paddle: "
                       f"{adapter_module._PADDLE_IMPORT_THREAD == adapter_module._PADDLE_THREAD_IDENT}"
                       f"{' (bố cục trên luồng chính)' if on_main_thread else ''}")
        if not adapter._model_uses_trt:
            conn.send(("error", "mô hình KHÔNG dựng được bằng TensorRT (bộ chuyển đổi đã lùi GPU thường)" + diagnostics()))
            return
        built = time.monotonic() - started
        import cv2

        images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in synthetic_images(det_batch)]
        config = payload["config"]
        if not in_thread("trích-xuất", lambda: adapter.infer_batch(images, list(range(len(images))),
                                                                   [0.04 * i for i in range(len(images))], config)):
            conn.send(("error", holder["error"] + diagnostics()))
            return
        if not adapter._model_uses_trt:
            conn.send(("error", "TensorRT lỗi khi chạy (bộ chuyển đổi đã lùi GPU thường)" + diagnostics()))
            return
        lines = sum(1 for result in holder["trích-xuất"] for box in result.text_boxes if (box.text or "").strip())
        if lines <= 0:
            conn.send(("error", f"engine chạy nhưng không đọc được dòng chữ nào trên {len(images)} ảnh tổng hợp "
                                "(kết quả sai)" + diagnostics()))
            return
        conn.send(("ok", f"{len(images)} ảnh {SYNTHETIC_SIZE[0]}×{SYNTHETIC_SIZE[1]}, {lines} dòng chữ, dựng "
                         f"{built:.0f} s, suy luận {time.monotonic() - started - built:.1f} s"))
        try:
            fault_file.flush()
        except OSError:
            pass
        os._exit(0)                                       # không huỷ mô hình lúc thoát (bố cục chẩn đoán có thể sập)
    except BaseException as exc:  # noqa: BLE001 — báo về tiến trình chính thay vì văng hộp thoại
        builtins.__import__ = real_import
        try:
            conn.send(("error", f"{type(exc).__name__}: {exc}"[:800] + diagnostics()))
        except BaseException:  # noqa: BLE001
            pass
    finally:
        if fault_file is not None:
            try:
                fault_file.flush()
            except OSError:
                pass


def _component_worker(conn: Any, variant: str, payload: dict[str, Any], det_batch: int, fault_path: str,
                      prepare: bool) -> None:  # pragma: no cover — chạy ở tiến trình con (exe Windows + GPU)
    """[v3.23.770] Phép thử thành phần: MỘT mô hình (``TextDetection`` / ``TextRecognition`` của PaddleOCR) chạy
    TensorRT, cùng luồng đã nạp paddle (luồng chính của con), các bản vá như ứng dụng (trừ cái biến môi trường tắt)."""
    fault_file = None
    try:
        _watch_parent()
        kind, precision, env = _COMPONENTS[variant]
        os.environ.update(env)
        if prepare:
            from subtitles_extractor.composition.child_process_setup import (
                prepare_child_process,
            )

            prepare_child_process()
        fault_file = open(fault_path, "w", encoding="utf-8")  # noqa: SIM115 — giữ mở tới khi tiến trình kết thúc
        from subtitles_extractor.infrastructure.ocr import (
            paddle_ocr_adapter as adapter_module,
        )

        adapter_module.PaddleOcrAdapter._import_paddle()
        if adapter_module._prepare_paddle_tensorrt() is None:
            conn.send(("error", "không chuẩn bị được TensorRT trong tiến trình tự kiểm"))
            return
        _rearm_faulthandler(fault_file)
        import paddleocr

        config = payload.get("config")
        name = getattr(config, "detection_model_name" if kind == "det" else "recognition_model_name", None) or (
            "PP-OCRv6_small_det" if kind == "det" else "PP-OCRv6_small_rec")
        started = time.monotonic()
        cls = paddleocr.TextDetection if kind == "det" else paddleocr.TextRecognition
        model = cls(model_name=name, device="gpu", use_tensorrt=True, precision=precision)
        _rearm_faulthandler(fault_file)
        built = time.monotonic() - started
        images = synthetic_images(det_batch if kind == "det" else 4)
        if kind == "rec":
            height = SYNTHETIC_SIZE[1]
            images = [image[height // 2 - 45:height // 2 + 55, 40:1540] for image in images]   # dải chữ ~1500×100
        results = list(model.predict(input=images, batch_size=1 if kind == "det" else len(images)))
        if kind == "det":
            found = sum(len(result.get("dt_polys", []) if hasattr(result, "get") else []) for result in results)
            summary = f"{found} hộp chữ"
        else:
            texts = [str(result.get("rec_text", "")) if hasattr(result, "get") else "" for result in results]
            found = sum(1 for text in texts if text.strip())
            summary = f"đọc: {texts[0][:40]!r}" if texts else "không kết quả"
        if found <= 0:
            conn.send(("error", f"{name} chạy nhưng kết quả rỗng (sai) — {summary}"))
            return
        conn.send(("ok", f"{name} {precision}: {summary}, dựng {built:.0f} s"))
        os._exit(0)
    except BaseException as exc:  # noqa: BLE001 — báo về tiến trình chính
        try:
            conn.send(("error", f"{type(exc).__name__}: {exc}"[:700]))
        except BaseException:  # noqa: BLE001
            pass
    finally:
        if fault_file is not None:
            try:
                fault_file.flush()
            except OSError:
                pass


def _child_setup(prepare: bool, fault_path: str) -> tuple[Any, Any]:  # pragma: no cover — tiến trình con
    """Chuẩn bị tiến trình con như ứng dụng + nạp paddle + chuẩn bị TensorRT; trả (tệp fault, mô-đun bộ chuyển đổi)."""
    _watch_parent()
    if prepare:
        from subtitles_extractor.composition.child_process_setup import (
            prepare_child_process,
        )

        prepare_child_process()
    fault_file = open(fault_path, "w", encoding="utf-8")  # noqa: SIM115 — giữ mở tới khi tiến trình kết thúc
    from subtitles_extractor.infrastructure.ocr import (
        paddle_ocr_adapter as adapter_module,
    )

    adapter_module.PaddleOcrAdapter._import_paddle()
    if adapter_module._prepare_paddle_tensorrt() is None:
        raise RuntimeError("không chuẩn bị được TensorRT trong tiến trình tự kiểm")
    _rearm_faulthandler(fault_file)
    return fault_file, adapter_module


def _build_engines_worker(conn: Any, variant: str, payload: dict[str, Any], det_batch: int, fault_path: str,
                          prepare: bool) -> None:  # pragma: no cover — chạy ở tiến trình con (exe Windows + GPU)
    """[v3.23.774] Dựng engine TensorRT bằng bước chuyển của Paddle (tạo mô hình TensorRT, KHÔNG suy luận)."""
    try:
        _fault_file, _adapter_module = _child_setup(prepare, fault_path)
        import paddleocr

        kwargs = payload["init_kwargs"]
        detection = variant == VARIANT_BUILD_DET
        name = kwargs.get("text_detection_model_name" if detection else "text_recognition_model_name") or (
            "PP-OCRv6_small_det" if detection else "PP-OCRv6_small_rec")
        cls = paddleocr.TextDetection if detection else paddleocr.TextRecognition
        started = time.monotonic()
        cls(model_name=name, device="gpu", use_tensorrt=True, precision=kwargs.get("precision", "fp32"))
        conn.send(("ok", f"{name}: dựng / nạp engine xong sau {time.monotonic() - started:.0f} s"))
    except BaseException as exc:  # noqa: BLE001 — báo về tiến trình chính
        try:
            conn.send(("error", f"{type(exc).__name__}: {exc}"[:700]))
        except BaseException:  # noqa: BLE001
            pass
    os._exit(0)


def _direct_app_worker(conn: Any, variant: str, payload: dict[str, Any], det_batch: int, fault_path: str,
                       prepare: bool) -> None:  # pragma: no cover — chạy ở tiến trình con (exe Windows + GPU)
    """[v3.23.774] PaddleOCR GPU thường làm mốc → gắn runner TensorRT trực tiếp → so kết quả + đo tốc độ."""
    try:
        fault_file, adapter_module = _child_setup(prepare, fault_path)
        from subtitles_extractor.infrastructure.ocr.trt_direct_runner import (
            DirectTrtRunner,
            attach_direct_tensorrt,
        )

        _paddle, paddle_ocr_cls = adapter_module.PaddleOcrAdapter._import_paddle()
        kwargs = {key: value for key, value in payload["init_kwargs"].items() if key not in ("use_tensorrt", "precision")}
        model = paddle_ocr_cls(**kwargs)
        try:
            pipeline = model.paddlex_pipeline._pipeline
            pipeline.batch_sampler.batch_size = det_batch
            pipeline.text_det_model.batch_sampler.batch_size = det_batch
        except AttributeError:
            pipeline = None
        images = synthetic_images(det_batch)
        textline = bool(payload["init_kwargs"].get("use_textline_orientation", False))

        import numpy as np

        def ocr() -> tuple[list[list[str]], list[Any], list[Any], float]:
            started = time.monotonic()
            results = list(model.predict(input=images, use_doc_orientation_classify=False, use_doc_unwarping=False,
                                         use_textline_orientation=textline))
            elapsed = time.monotonic() - started
            texts = [list(result.get("rec_texts") or []) for result in results]
            scores = [list(result.get("rec_scores") or []) for result in results]
            polys = [[np.asarray(poly, dtype=np.float64) for poly in (result.get("rec_polys") or [])]
                     for result in results]
            return texts, scores, polys, elapsed

        ocr()                                                       # làm nóng
        base_texts, base_scores, base_polys, base_time = ocr()
        from subtitles_extractor.infrastructure.ocr.trt_direct_runner import (
            engine_fingerprint,
            model_engine_files,
        )

        described = attach_direct_tensorrt(model)
        fingerprints = {name: engine_fingerprint(model_engine_files(getattr(pipeline, name).model_dir)[0])
                        for name in ("text_det_model", "text_rec_model")}
        _rearm_faulthandler(fault_file)
        ocr()                                                       # làm nóng (lần đầu TensorRT)
        trt_texts, trt_scores, trt_polys, trt_time = ocr()
        runners = [getattr(getattr(pipeline, name, None), "runner", None) for name in ("text_det_model", "text_rec_model")]
        used = all(isinstance(runner, DirectTrtRunner) and runner.calls > 0 for runner in runners)
        fallbacks = sum(getattr(runner, "fallbacks", 0) for runner in runners)
        fp16 = str(payload["init_kwargs"].get("precision", "fp32")) == "fp16"
        problem = ""
        if not used or fallbacks:
            # Rà độc lập: lô lùi Paddle thì đang so Paddle với Paddle — không chứng minh được gì.
            problem = f"runner TensorRT trực tiếp không chạy hết (lùi Paddle {fallbacks} lô) — " + "; ".join(described)
        elif trt_texts != base_texts or [len(p) for p in trt_polys] != [len(p) for p in base_polys]:
            problem = (f"kết quả KHÁC Paddle GPU: Paddle {base_texts[:2]} / TensorRT {trt_texts[:2]}"
                       + (" (FP16 — lệch nhỏ có thể chấp nhận; không ghi nhớ)" if fp16 else ""))
        else:
            score_gap = max((abs(a - b) for sa, sb in zip(base_scores, trt_scores, strict=True)
                             for a, b in zip(sa, sb, strict=True)), default=0.0)
            box_gap = max((float(np.max(np.abs(pa - pb))) for la, lb in zip(base_polys, trt_polys, strict=True)
                           for pa, pb in zip(la, lb, strict=True) if pa.shape == pb.shape), default=0.0)
            if score_gap > 0.02 or box_gap > 3.0:
                problem = f"kết quả lệch Paddle GPU: điểm tin cậy lệch {score_gap:.3f}, hộp chữ lệch {box_gap:.1f} px"
        if problem:
            conn.send(("error", problem))
        else:
            speed = base_time / trt_time if trt_time > 0 else 0.0
            conn.send(("ok", f"kết quả GIỐNG Paddle GPU ({sum(len(t) for t in base_texts)} dòng); {len(images)} ảnh: "
                             f"Paddle GPU {base_time * 1000:.0f} ms, TensorRT {trt_time * 1000:.0f} ms (×{speed:.2f}); "
                             + "; ".join(described), {"engines": fingerprints}))
    except BaseException as exc:  # noqa: BLE001 — báo về tiến trình chính
        try:
            conn.send(("error", f"{type(exc).__name__}: {exc}"[:800]))
        except BaseException:  # noqa: BLE001
            pass
    os._exit(0)                                                     # không dọn Paddle / TensorRT lúc thoát (có thể treo)


def find_engine_files(roots: list[Path], limit: int = 6) -> list[Path]:
    """Tệp engine Paddle đã dựng (``<mô hình>/.cache/paddle/trt/**/engine_*.trt``), mới nhất trước."""
    found: dict[str, Path] = {}
    for root in roots:
        try:
            for path in root.glob("*/.cache/paddle/trt/**/engine_*.trt"):
                found[str(path)] = path
        except OSError:
            continue
    files = sorted(found.values(), key=lambda path: path.stat().st_mtime, reverse=True)
    return files[:limit]


def _engine_model_name(path: Path) -> str:
    parts = path.parts
    return parts[parts.index(".cache") - 1] if ".cache" in parts else path.parent.name


def _direct_engine_worker(conn: Any, variant: str, payload: dict[str, Any], det_batch: int, fault_path: str,
                          prepare: bool) -> None:  # pragma: no cover — chạy ở tiến trình con (exe Windows + GPU)
    """[v3.23.771] Nạp + chạy THẲNG tệp engine TensorRT Paddle đã dựng, không qua Paddle (xem ``VARIANT_DIRECT_ENGINE``)."""
    fault_file = None
    try:
        _watch_parent()
        if prepare:
            from subtitles_extractor.composition.child_process_setup import (
                prepare_child_process,
            )

            prepare_child_process()
        fault_file = open(fault_path, "w", encoding="utf-8")  # noqa: SIM115 — giữ mở tới khi tiến trình kết thúc
        # [v3.23.773] Bộ đệm GPU bằng TENSOR PADDLE (``data_ptr``) + luồng CUDA của paddle — không cần CuPy. Log 19/11
        # 20:53 / 21:12 (bản 3.23.771–772): trong tiến trình con CuPy không nạp được ("thiếu mô-đun 'cuda.pathfinder'")
        # dù qua bộ nạp của ứng dụng, trước paddle.
        from subtitles_extractor.infrastructure.ocr import (
            paddle_ocr_adapter as adapter_module,
        )

        adapter_module.PaddleOcrAdapter._import_paddle()       # bước chuẩn bị TensorRT như ứng dụng (DLL, plugin)
        if adapter_module._prepare_paddle_tensorrt() is None:
            conn.send(("error", "không chuẩn bị được TensorRT trong tiến trình tự kiểm"))
            return
        _rearm_faulthandler(fault_file)
        roots: list[Path] = []
        for value in (os.environ.get("PADDLE_PDX_CACHE_HOME"), str(Path.home() / ".paddlex")):
            if value:
                roots.append(Path(value) / "official_models")
        files = find_engine_files(roots)
        if not files:
            conn.send(("error", f"không thấy tệp engine Paddle đã dựng trong {', '.join(map(str, roots))}"))
            return
        import numpy as np
        import paddle
        import tensorrt as trt

        paddle.device.set_device("gpu:0")
        stream: Any = paddle.device.current_stream()
        base = getattr(stream, "stream_base", stream)
        raw = getattr(base, "cuda_stream", None) or getattr(base, "raw_stream", None)
        stream_handle = int(raw if raw is not None else stream.__cuda_stream__()[1])

        def gpu_array(shape: tuple[int, ...], dtype: Any, random: bool) -> Any:
            host = (np.random.random(shape).astype(dtype) if random and dtype.kind == "f" else np.ones(shape, dtype))
            return paddle.to_tensor(host, place=paddle.CUDAPlace(0))

        trt_logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(trt_logger, "")
        runtime = trt.Runtime(trt_logger)
        lines: list[str] = []
        for path in files:
            label = f"{_engine_model_name(path)}/{path.name}"
            engine = runtime.deserialize_cuda_engine(path.read_bytes())
            if engine is None:
                lines.append(f"{label}: KHÔNG nạp được (TensorRT từ chối tệp)")
                continue
            context = engine.create_execution_context()
            buffers: dict[str, Any] = {}
            shapes: list[str] = []
            for index in range(engine.num_io_tensors):
                name = engine.get_tensor_name(index)
                if engine.get_tensor_mode(name) != trt.TensorIOMode.INPUT:
                    continue
                dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(name)))
                if engine.is_shape_inference_io(name):
                    lines.append(f"{label}: đầu vào hình dạng {name} (chưa hỗ trợ trong phép thử)")
                    break
                shape = tuple(int(v) for v in engine.get_tensor_profile_shape(name, 0)[1])
                context.set_input_shape(name, shape)
                buffers[name] = gpu_array(shape, dtype, random=True)
                shapes.append(f"{name}{list(shape)}")
            else:
                for index in range(engine.num_io_tensors):
                    name = engine.get_tensor_name(index)
                    if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                        continue
                    shape = tuple(int(v) for v in context.get_tensor_shape(name))
                    dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(name)))
                    buffers[name] = gpu_array(tuple(max(1, v) for v in shape), dtype, random=False)
                for name, array in buffers.items():
                    context.set_tensor_address(name, int(array.data_ptr()))
                fault_file.write(f"chạy {label}\n")
                fault_file.flush()
                ok = context.execute_async_v3(stream_handle)
                stream.synchronize()
                started = time.monotonic()
                for _ in range(5):
                    context.execute_async_v3(stream_handle)
                stream.synchronize()
                per_run = (time.monotonic() - started) / 5 * 1000
                lines.append(f"{label}: {'chạy được' if ok else 'execute trả False'} ({', '.join(shapes)}; "
                             f"{per_run:.1f} ms/lần)")
        conn.send(("ok" if all("chạy được" in line for line in lines) else "error", "; ".join(lines)))
        os._exit(0)
    except BaseException as exc:  # noqa: BLE001 — báo về tiến trình chính
        try:
            conn.send(("error", f"{type(exc).__name__}: {exc}"[:700]))
        except BaseException:  # noqa: BLE001
            pass
    finally:
        if fault_file is not None:
            try:
                fault_file.flush()
            except OSError:
                pass


Worker = Callable[[Any, str, dict[str, Any], int, str, bool], None]


def _fault_tail(path: Path, lines: int = 12) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
    except OSError:
        return ""
    return " | ".join(line.strip() for line in text[:lines] if line.strip())


def _stop(process: Any) -> None:
    if process.is_alive():
        process.terminate()
        process.join(10)
    if process.is_alive():
        process.kill()
        process.join(10)


def run_child_selftest(variant: str, payload: dict[str, Any], det_batch: int, fault_path: Path, *,
                       worker: Worker | None = None, prepare: bool = True,
                       timeout: float = SELFTEST_TIMEOUT_SEC) -> SelfTestOutcome:
    """Một lượt tự kiểm ở tiến trình con; sập / treo / lỗi đều thành ``SelfTestOutcome(ok=False, lý do)``."""
    try:
        fault_path.unlink(missing_ok=True)
    except OSError:
        pass
    if worker is None:
        worker = (_component_worker if variant in _COMPONENTS else
                  _direct_engine_worker if variant == VARIANT_DIRECT_ENGINE else
                  _build_engines_worker if variant in (VARIANT_BUILD_ENGINES, VARIANT_BUILD_DET, VARIANT_BUILD_REC) else
                  _direct_app_worker if variant == VARIANT_DIRECT_APP else _selftest_worker)
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=worker, args=(child, variant, payload, det_batch, str(fault_path), prepare),
                              name="tensorrt-selftest", daemon=True)
    started = time.monotonic()
    process.start()
    child.close()
    message: tuple[Any, ...] | None = None
    timed_out = False
    try:
        try:
            if parent.poll(timeout):
                message = parent.recv()
            else:
                timed_out = True
        except (EOFError, OSError):              # Windows: PeekNamedPipe lỗi khi con đã chết — coi như không có tin
            message = None
    finally:
        parent.close()
    if timed_out:
        _stop(process)
        return SelfTestOutcome(False, f"{variant}: quá {timeout:.0f} s không xong (treo) — đã dừng tiến trình",
                               definitive=False)
    process.join(60)
    _stop(process)
    elapsed = time.monotonic() - started
    if message is None:
        code = process.exitcode
        # Windows: mã NTSTATUS (0xc0000005 = access violation); POSIX: số âm = tín hiệu (-11 = SIGSEGV).
        shown = ("?" if not isinstance(code, int) else f"tín hiệu {-code}" if code < 0 else f"{code & 0xFFFFFFFF:#010x}")
        tail = _fault_tail(fault_path)
        killed = isinstance(code, int) and code in _KILLED_CODES
        return SelfTestOutcome(False, f"{variant}: tiến trình tự kiểm {'bị dừng' if killed else 'SẬP'} (mã thoát "
                                      f"{shown}, sau {elapsed:.0f} s)" + (f" — {tail}" if tail else ""),
                               definitive=not killed)
    kind, text = message[0], message[1]
    data = message[2] if len(message) > 2 and isinstance(message[2], dict) else None
    if kind == "ok":
        return SelfTestOutcome(True, f"{variant}: chạy được ({text})", data=data)
    transient = any(marker in str(text).lower() for marker in _TRANSIENT_MARKERS)
    return SelfTestOutcome(False, f"{variant}: lỗi — {text}", definitive=not transient)


def ensure_tensorrt_selftest(trt_dir: Path, app_version: str, init_kwargs: dict[str, Any], det_batch: int, *,
                             config: Any = None,
                             runner: Callable[..., SelfTestOutcome] = run_child_selftest) -> SelfTestOutcome:
    """Kết quả tự kiểm cho cấu hình này (chạy nếu chưa có). ``SE_TRT_SELFTEST=0`` để bỏ qua (chẩn đoán)."""
    if os.environ.get("SE_TRT_SELFTEST", "1") == "0":
        return SelfTestOutcome(True, "bỏ qua tự kiểm (SE_TRT_SELFTEST=0)", cached=True)
    signature = selftest_signature(app_version, init_kwargs, det_batch)
    result_path = trt_dir / RESULT_FILE
    with _LOCK:                                   # hai bộ OCR (trích xuất + phụ đề nhúng) cùng lúc: một tiến trình con
        stored = _load(result_path)
        raw_entries = stored.get("entries")
        entries: dict[str, Any] = raw_entries if isinstance(raw_entries, dict) else {}
        entry = entries.get(signature)
        if isinstance(entry, dict) and isinstance(entry.get("ok"), bool):
            data = entry.get("data") if isinstance(entry.get("data"), dict) else None
            return SelfTestOutcome(bool(entry["ok"]), str(entry.get("reason", "")), cached=True, data=data)
        logger.info("TensorRT: tự kiểm trong tiến trình riêng trước khi dùng (lần đầu có thể mất vài phút để dựng "
                    "engine)…")
        fault_path = trt_dir / FAULT_FILE
        payload = {"init_kwargs": init_kwargs, "config": config}
        if os.environ.get("SE_TRT_FULL_DIAG") == "1":
            return _legacy_diagnostics(payload, det_batch, fault_path, runner)
        # [v3.23.774] (1) dựng engine bằng bước chuyển của Paddle (có thể sập ở bước dọn sau khi đã ghi tệp — vẫn tiếp);
        # (2) ứng dụng tự chạy engine, so kết quả với Paddle GPU thường.
        reasons = []
        build_crashed = False
        for step in (VARIANT_BUILD_DET, VARIANT_BUILD_REC):
            build = runner(step, payload, det_batch, fault_path)
            reasons.append(build.reason)
            if not build.ok:
                build_crashed = True
                logger.warning("TensorRT tự kiểm: %s", build.reason)
        direct = runner(VARIANT_DIRECT_APP, payload, det_batch, fault_path)
        reasons.append(direct.reason)
        definitive = direct.definitive
        if not direct.ok and (
                (build_crashed and "tệp engine" in direct.reason)                     # thiếu engine vì bước dựng sập
                or "FP16" in direct.reason                                           # FP16 lệch nhỏ: không ghi nhớ
                or "thiếu VRAM" in direct.reason):                  # [v3.23.775] VRAM đang bận (app khác): kiểm lại sau
            definitive = False
        outcome = SelfTestOutcome(direct.ok, "; ".join(reasons), definitive=definitive, data=direct.data)
        if definitive:
            entries = {key: value for key, value in entries.items() if isinstance(value, dict)}
            entries[signature] = {"ok": outcome.ok, "reason": outcome.reason, "app": app_version,
                                  "data": outcome.data or {}}
            try:
                result_path.write_text(json.dumps({"entries": entries}, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
            except OSError as exc:
                logger.debug("Không ghi được kết quả tự kiểm TensorRT: %s", exc)
        return outcome


def forget_tensorrt_selftest(trt_dir: Path, app_version: str, init_kwargs: dict[str, Any], det_batch: int) -> None:
    """[v3.23.774] Xoá kết quả tự kiểm của cấu hình này (vd tệp engine đổi sau lần kiểm) — lần dựng sau kiểm lại."""
    signature = selftest_signature(app_version, init_kwargs, det_batch)
    result_path = trt_dir / RESULT_FILE
    with _LOCK:
        stored = _load(result_path)
        entries = stored.get("entries")
        if isinstance(entries, dict) and entries.pop(signature, None) is not None:
            try:
                result_path.write_text(json.dumps({"entries": entries}, ensure_ascii=False, indent=1), encoding="utf-8")
            except OSError as exc:
                logger.debug("Không ghi được kết quả tự kiểm TensorRT: %s", exc)


def _legacy_diagnostics(payload: dict[str, Any], det_batch: int, fault_path: Path,
                        runner: Callable[..., SelfTestOutcome]) -> SelfTestOutcome:
    """Chuỗi chẩn đoán 3.23.766–773 (Paddle tự chạy engine) — chỉ khi ``SE_TRT_FULL_DIAG=1``; không ghi nhớ."""
    first = runner(VARIANT_DEDICATED_THREAD, payload, det_batch, fault_path)
    reasons = [first.reason]
    if not first.ok:
        for name in (VARIANT_DIRECT_ENGINE, VARIANT_CROSS_THREAD, *(n for n, *_rest in COMPONENT_VARIANTS)):
            part = runner(name, payload, det_batch, fault_path)
            reasons.append(part.reason)
            logger.warning("TensorRT chẩn đoán đầy đủ: %s", part.reason)
    return SelfTestOutcome(first.ok, "; ".join(reasons), definitive=False)


__all__ = [
    "COMPONENT_VARIANTS", "SELFTEST_TIMEOUT_SEC", "VARIANT_BUILD_DET", "VARIANT_BUILD_ENGINES", "VARIANT_BUILD_REC",
    "VARIANT_DIRECT_APP", "forget_tensorrt_selftest", "VARIANT_DIRECT_ENGINE", "find_engine_files", "VARIANT_CROSS_THREAD", "VARIANT_DEDICATED_THREAD", "SelfTestOutcome",
    "ensure_tensorrt_selftest", "run_child_selftest", "selftest_signature", "synthetic_images",
]
