"""[v3.23.200] Test thread-safety engine cache + nạp giọng VieNeu ở background thread.

Hai vấn đề rà soát:
1. ``_ENGINE_CACHE`` không có lock: nạp giọng nền (UI) + Generate (worker) trùng thời
   điểm -> HAI thread cùng nạp model (~9s + RAM x2) và race trên dict. Fix:
   ``_ENGINE_LOCK`` double-checked — model chỉ nạp đúng MỘT lần.
2. ``list_speakers`` nạp model ~9s trên UI THREAD làm app đứng hình khi đổi engine/chế
   độ (log thực tế 11:35:55 -> 11:36:03). Fix: ``_VieNeuVoiceLoader(QThread)`` nạp nền,
   kết quả kèm mode để bỏ qua STALE khi người dùng đổi chế độ giữa chừng.
"""

from __future__ import annotations

import sys
import threading
import time
import types

from subtitles_extractor.infrastructure.tts import vieneu_tts_adapter as vmod
from subtitles_extractor.infrastructure.tts.vieneu_tts_adapter import VieNeuTtsAdapter


def _install_fake_vieneu(load_counter: list[int], delay_s: float = 0.05) -> None:
    class _FakeVieneu:
        def __init__(self, mode=None, emotion=None):
            load_counter[0] += 1
            time.sleep(delay_s)  # giả lập nạp model chậm

    fake = types.ModuleType("vieneu")
    fake.Vieneu = _FakeVieneu
    sys.modules["vieneu"] = fake


def _teardown_fake() -> None:
    sys.modules.pop("vieneu", None)
    vmod._ENGINE_CACHE.clear()


def test_concurrent_engine_load_happens_once() -> None:
    # 8 thread cùng đòi engine (nạp giọng nền + Generate + check availability...)
    # -> model chỉ được nạp ĐÚNG MỘT lần nhờ double-checked lock.
    vmod._ENGINE_CACHE.clear()
    load_counter = [0]
    _install_fake_vieneu(load_counter)
    try:
        adapters = [VieNeuTtsAdapter() for _ in range(8)]
        threads = [
            threading.Thread(target=a._get_or_load_engine, args=("auto",))
            for a in adapters
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert load_counter[0] == 1
        # Mọi adapter cùng trỏ về MỘT engine instance (cache chia sẻ).
        engines = {id(a._engine) for a in adapters}
        assert len(engines) == 1
    finally:
        _teardown_fake()


def test_different_modes_load_separately() -> None:
    # Mỗi (mode, emotion) là một model riêng -> cache theo key, không trộn lẫn.
    vmod._ENGINE_CACHE.clear()
    load_counter = [0]
    _install_fake_vieneu(load_counter, delay_s=0.0)
    try:
        a1 = VieNeuTtsAdapter(mode="standard")
        a2 = VieNeuTtsAdapter(mode="v3turbo")
        a1._get_or_load_engine("auto")
        a2._get_or_load_engine("auto")
        assert load_counter[0] == 2
        assert id(a1._engine) != id(a2._engine)
    finally:
        _teardown_fake()


def test_engine_lock_exists_and_is_lock() -> None:
    # Khoá module-level phải tồn tại (chống tái phát khi refactor).
    assert isinstance(vmod._ENGINE_LOCK, type(threading.Lock()))


def test_voice_loader_class_emits_mode_with_result() -> None:
    # Khoá thiết kế: loader phát kèm MODE để UI bỏ qua kết quả stale khi đổi chế độ.
    import pathlib

    source = pathlib.Path(
        "src/subtitles_extractor/presentation/pages/tts_page.py"
    ).read_text(encoding="utf-8")
    assert "class _VieNeuVoiceLoader(QThread)" in source
    assert "voices_loaded = Signal(str, list)" in source  # (mode, ids)
    assert "if mode != current_mode:" in source               # stale check
    assert "(Đang nạp giọng…)" in source                       # trạng thái chờ trên UI
