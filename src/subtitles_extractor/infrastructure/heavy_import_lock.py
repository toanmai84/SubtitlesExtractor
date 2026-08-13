"""Lock điều phối import các thư viện ML nặng (paddle, torch).

Vấn đề: ``import paddle`` và ``import torch`` đều KHÔNG an toàn khi chạy ĐỒNG THỜI
trên nhiều thread — Python có thể trả về module nửa-khởi-tạo cho thread thứ hai,
gây ``partially initialized module ... circular import``. Trong app này, preload
OCR (background thread) và WhisperX STT (thread khác) có thể import paddle/torch
cùng lúc.

Giải pháp: một lock TÁI NHẬP (RLock) chia sẻ toàn tiến trình; mọi nơi import các
thư viện này phải acquire lock trước, biến import đồng thời thành tuần tự. Sau khi
một module đã import xong, các lần import sau chỉ lấy từ cache (rất nhanh), nên lock
gần như không gây chậm.
"""

from __future__ import annotations

import threading

# RLock để cùng một thread có thể acquire lồng nhau (vd paddleocr import lại paddle).
HEAVY_IMPORT_LOCK = threading.RLock()


__all__ = ["HEAVY_IMPORT_LOCK"]
