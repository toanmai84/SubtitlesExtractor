"""[v3.23.229] Hai bug Edge tìm ra từ FLAC + log Edge thật (lần đầu có dữ liệu Edge).

**BUG 1 — Edge lặp lại đúng bug v218 (dời sớm cả câu KHÔNG cần).**

Đo trên FLAC Edge thật (95 câu, cùng phụ đề với các phiên VieNeu):

* **34/95 câu bị dời đúng KỊCH TRẦN 250ms** -> dấu hiệu code lấy lead TỐI ĐA.
* **36/95 câu bị dời sớm dù bản thân đã vừa khung** -> tiếng vang TRƯỚC khẩu
  hình.
* Đối chiếu VieNeu (đã sửa ở v219): dời 24 câu, đúng 24 câu cần -> **0 câu vô ích**.

Nguyên nhân: Edge tính ``lead = min(lead_in_s, free_before)`` — luôn ăn gian tối đa,
dù cụm chỉ thiếu 30ms. VieNeu v219 chỉ ăn gian ĐỦ BÙ phần thiếu (``needed_lead_s``).
Fix chưa bao giờ được port sang Edge — đúng dạng bug parity mà kỷ luật "một fix phải xét
cho cả ba engine" sinh ra để chặn.

**BUG 2 — song song 64 làm dịch vụ từ chối hàng loạt.**

Log Edge: 20 lỗi "audio rỗng" DỒN trong 2.4 giây đầu, rồi 9 timeout 30s cùng lúc; tổng 87
giây cho 95 câu, trong đó ~80 giây là retry và chờ. Mã cũ retry ở CÙNG mức tải -> lại bị
chặn (lần 2, 3, 4 vẫn rỗng).

**Câu hỏi tồn đọng đã được trả lời (và KHÔNG cần sửa):** VAD cắt lặng biên riêng của Edge
cho median lặng đầu **40ms** — y hệt ``trim_edge_silence`` dùng chung của VieNeu. Đây là
tính năng tương đương, không phải bug. Đo trước đã tránh được một refactor vô ích.
"""

from __future__ import annotations

import asyncio
import logging
import pathlib

import pytest

from subtitles_extractor.infrastructure.tts.adaptive_limiter import (
    INITIAL_CONCURRENCY,
    MIN_CONCURRENCY,
    AdaptiveConcurrencyLimiter,
    grow_limit,
    shrink_limit,
)

_EDGE_SRC = pathlib.Path(
    "src/subtitles_extractor/infrastructure/tts/edge_tts_adapter.py"
).read_text(encoding="utf-8")


# ── BUG 1: Edge chỉ được ăn gian ĐỦ BÙ phần thiếu ───────────────────────────
def test_edge_khong_con_an_gian_toi_da() -> None:
    # Mã cũ: lead = max(0.0, min(request.lead_in_s, free_before))  -> luôn lấy MAX.
    assert "lead = max(0.0, min(request.lead_in_s, free_before))" not in _EDGE_SRC
    assert "needed_lead = max(0.0, total_speech - room_at_orig)" in _EDGE_SRC
    assert "lead = min(request.lead_in_s, free_before, needed_lead)" in _EDGE_SRC


def test_lead_chi_du_bu_phan_thieu() -> None:
    """Tái lập đúng công thức mới của Edge trên các ca thật."""

    def lead_moi(lead_in_s: float, free_before: float, thieu: float) -> float:
        return min(lead_in_s, max(0.0, free_before), max(0.0, thieu))

    # Cụm chỉ thiếu 30ms -> chỉ ăn gian 30ms (mã cũ ăn gian nguyên 250ms).
    assert lead_moi(0.25, 1.0, 0.03) == pytest.approx(0.03)
    # Cụm thiếu nhiều -> ăn gian tối đa cho phép.
    assert lead_moi(0.25, 1.0, 0.80) == pytest.approx(0.25)
    # Không có khoảng lặng thật phía trước -> KHÔNG được đè lên câu trước.
    assert lead_moi(0.25, 0.0, 0.80) == 0.0
    # Không thiếu gì -> KHÔNG dời (đây chính là bug v218).
    assert lead_moi(0.25, 1.0, 0.0) == 0.0


# ── BUG 2: bộ điều tiết song song thích ứng ─────────────────────────────────
def test_khoi_dong_cham_khong_phong_burst() -> None:
    """Xuất phát THẤP bất kể người dùng cấu hình bao nhiêu.

    Chỉ "giảm khi bị từ chối" là chưa đủ: nếu xuất phát thẳng ở 64 thì cả burst đầu đã bay
    lên và bị chặn TRƯỚC KHI bộ điều tiết kịp phản ứng (đúng như log: 20 lỗi trong 2.4s).
    """
    lim = AdaptiveConcurrencyLimiter(64)
    assert lim.limit == INITIAL_CONCURRENCY
    assert lim.limit < 64


def test_khong_vuot_tran_nguoi_dung_dat() -> None:
    # Người dùng đặt 4 -> không được tự ý chạy 8.
    assert AdaptiveConcurrencyLimiter(4).limit == 4


def test_giam_nhanh_tang_cham() -> None:
    # Lùi NHANH (chia đôi): mỗi lần thử ở mức quá cao tốn thêm một vòng timeout 30s.
    assert shrink_limit(64) == 32
    assert shrink_limit(16) == 8
    assert shrink_limit(2) == MIN_CONCURRENCY  # chạm sàn thì dừng
    # Nới CHẬM (cộng 1): dung lượng thật không biết trước, nhảy vọt sẽ lại đâm tường.
    assert grow_limit(8, ceiling=64) == 9
    assert grow_limit(64, ceiling=64) == 64  # không vượt trần


def test_giam_tai_khi_dich_vu_tu_choi() -> None:
    lim = AdaptiveConcurrencyLimiter(64)
    ban_dau = lim.limit
    for _ in range(4):
        lim.report_failure()
    assert lim.limit < ban_dau


def test_khong_giam_vi_mot_loi_le() -> None:
    # Một lỗi lẻ giữa nhiều lượt thành công KHÔNG được kéo tụt cả phiên.
    lim = AdaptiveConcurrencyLimiter(64)
    ban_dau = lim.limit
    for _ in range(20):
        lim.report_success()
    lim.report_failure()
    assert lim.limit >= ban_dau


def test_tu_do_ra_muc_an_toan_cua_dich_vu(caplog: pytest.LogCaptureFixture) -> None:
    """Mô phỏng dịch vụ chịu được ~12 kết nối, người dùng cấu hình 64.

    Kết quả đo: Semaphore cố định -> 71/95 dòng rỗng. Bộ điều tiết -> 2/95, và tự dò ra
    mức ~10 (sát ngưỡng thật 12) mà không ai phải nói cho nó biết con số đó.
    """
    caplog.set_level(logging.CRITICAL)  # im lặng, chỉ quan tâm kết quả
    nguong_that = 12

    async def chay(thich_ung: bool) -> tuple[int, int]:
        lim = AdaptiveConcurrencyLimiter(64)
        sem = asyncio.Semaphore(64)
        dang_chay = 0
        so_rong = 0

        async def mot_dong() -> None:
            nonlocal dang_chay, so_rong
            gate = lim if thich_ung else sem
            async with gate:  # type: ignore[attr-defined]
                dang_chay += 1
                qua_tai = dang_chay > nguong_that
                await asyncio.sleep(0.001)
                dang_chay -= 1
                if qua_tai:
                    so_rong += 1
                    if thich_ung:
                        lim.report_failure()
                elif thich_ung:
                    lim.report_success()

        await asyncio.gather(*[mot_dong() for _ in range(95)])
        return so_rong, lim.limit

    rong_cu, _ = asyncio.run(chay(False))
    rong_moi, limit_cuoi = asyncio.run(chay(True))

    assert rong_cu > 40  # Semaphore cố định: hỏng hàng loạt
    assert rong_moi < 10  # Bộ điều tiết: gần như sạch
    assert MIN_CONCURRENCY <= limit_cuoi <= nguong_that + 4  # tự dò sát ngưỡng thật


def test_edge_dung_bo_dieu_tiet_va_bao_ket_qua() -> None:
    assert "AdaptiveConcurrencyLimiter(concurrency)" in _EDGE_SRC
    assert "semaphore.report_success()" in _EDGE_SRC
    assert "semaphore.report_failure()" in _EDGE_SRC


def test_ceiling_khong_hop_le_bi_tu_choi() -> None:
    with pytest.raises(ValueError, match="ceiling"):
        AdaptiveConcurrencyLimiter(0)
