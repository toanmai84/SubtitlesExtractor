"""[v3.23.264] Chống "ngân dài" VieNeu qua tham số API (repetition_penalty/temperature).

**Điều tra nguyên nhân (đọc source vieneu 3.2.3):** model sinh audio TỰ HỒI QUY từng frame
(``_generate_codes`` trong inference_v3_turbo.py), dừng khi (1) tự sinh EOS token hoặc (2)
chạm ``max_new_frames``. Câu NGẮN ("Chú.", "Ừm.") model hay "phân vân" không sinh EOS đúng
lúc -> kéo dài nguyên âm (ngân) tới khi ngẫu nhiên gặp EOS/chạm trần.

**Cơ chế chống từ API:**
- ``repetition_penalty`` (mặc định SDK 1.2): phạt token audio đã xuất hiện
  (``logits[idx] = sel / penalty`` trong modeling) -> chống lặp frame nguyên âm =
  chống ngân. AN TOÀN (không làm câu ngắn bị từ chối). Nâng 1.2 -> 1.3.
- ``temperature`` (SDK 0.8): thấp hơn -> ít ngân, NHƯNG bài học v249: quá thấp làm
  câu ngắn bị từ chối đọc (rỗng). Để None (giữ SDK), chỉ truyền khi người dùng đặt.

App trước đây CHỈ đặt ``max_new_frames`` (trần cứng chặn thảm hoạ) — chưa dùng
repetition_penalty/temperature để giảm ngân NGAY từ đầu.
"""

from __future__ import annotations

from subtitles_extractor.infrastructure.tts.vieneu_tts_adapter import (
    _VIENEU_REPETITION_PENALTY,
    VieNeuTtsAdapter,
)


class _FakeV3Turbo:
    sample_rate = 48_000

    def infer(
        self,
        text: str,
        voice=None,
        style: str = "tu_nhien",
        max_new_frames: int = 300,
        repetition_penalty: float = 1.2,
        temperature: float = 0.8,
        **kw,
    ):
        return {
            "repetition_penalty": repetition_penalty,
            "temperature": temperature,
            "max_new_frames": max_new_frames,
        }


def _adapter(temperature: float | None = None) -> VieNeuTtsAdapter:
    a = VieNeuTtsAdapter.__new__(VieNeuTtsAdapter)
    a._mode = "v3turbo"
    a._emotion = "natural"
    a._temperature = temperature
    a._force_cpu = True  # [v3.23.364] test đường CPU: bỏ qua nhánh GPU worker
    return a


def test_repetition_penalty_được_nâng() -> None:
    # Chống ngân: repetition_penalty nâng từ mặc định 1.2 lên 1.3.
    assert _VIENEU_REPETITION_PENALTY == 1.3
    r = _adapter()._infer_once(_FakeV3Turbo(), "Chú.", {"v": 1})
    assert r["repetition_penalty"] == 1.3


def test_temperature_none_giữ_mặc_định_sdk() -> None:
    # temperature None -> KHÔNG truyền -> SDK giữ 0.8 (tránh câu ngắn bị từ chối).
    r = _adapter(temperature=None)._infer_once(_FakeV3Turbo(), "Ơ.", {"v": 1})
    assert r["temperature"] == 0.8  # mặc định SDK, app không ép


def test_temperature_đặt_thì_truyền() -> None:
    # Người dùng chủ động đặt temperature -> truyền đúng.
    r = _adapter(temperature=0.6)._infer_once(_FakeV3Turbo(), "Chú.", {"v": 1})
    assert r["temperature"] == 0.6


def test_penalty_không_đặt_khi_sdk_không_hỗ_trợ() -> None:
    # Engine cũ không có repetition_penalty -> không truyền (không lỗi).
    class EngineOld:
        sample_rate = 24_000

        def infer(self, text, voice=None, **kw):
            return {"ok": True, "keys": list(kw.keys())}

    r = _adapter()._infer_once(EngineOld(), "Chú.", {"v": 1})
    assert "repetition_penalty" not in r.get("keys", [])


def test_temperature_khởi_tạo_mặc_định_none() -> None:
    # __init__ đặt _temperature=None (getattr an toàn kể cả khi chưa gọi __init__).
    a = VieNeuTtsAdapter(mode="v3turbo")
    assert a._temperature is None
