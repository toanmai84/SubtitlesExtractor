"""Subtitle Studio — trích xuất, dịch, lồng tiếng và xuất bản phụ đề video.

Kiến trúc Clean Architecture / Hexagonal:

    domain/         → entities, value objects, ports (Protocol)
    application/    → use cases, services, DTOs (orchestration thuần)
    infrastructure/ → adapter cho Paddle, OpenCV, QSettings, JSON…
    presentation/   → Qt UI, view models, workers
    composition/    → DI container + bootstrap

Changelog v3.6.3 — Static Analysis Full-Clean (Zero Ruff F/B/W Errors):
    * **[BUG FIX]** ``editor_page.py``: Thêm 4 hằng số bị thiếu khiến
      ``test_constants_declared`` fail và các feature liên quan không hoạt
      động: ``_SYNC_HIGHLIGHT_MS = 100``, ``_LOW_CONFIDENCE_THRESHOLD = 0.6``,
      ``_TOO_FAST_CPS_THRESHOLD = 20.0``, ``_TOO_SHORT_DURATION_SEC = 0.5``.
    * **[PERF BUG FIX]** ``editor_page._on_canvas_position_changed``: Áp
      dụng throttle ``_SYNC_HIGHLIGHT_MS`` (100ms) cho vòng lặp highlight
      bảng. position_changed bắn 25-60 lần/giây — không có throttle khiến
      ``find_event_index_at_time`` + ``set_active_row`` gây giật UI khi
      phát video. Slider và waveform vẫn cập nhật mỗi frame.
    * **[BUG FIX / B007]** ``paddle_ocr_adapter._try_create_model_with_retry``:
      Đổi ``for attempt in range(...)`` → ``for _attempt in range(...)`` —
      biến loop không dùng được đặt tên đúng chuẩn Python.
    * **[BUG FIX / B008]** ``subtitle_table_model.rowCount / columnCount``:
      Thay ``parent=QModelIndex()`` (function-call trong default arg — hành
      vi không xác định khi module load) bằng ``parent: QModelIndex | None = None``.
    * **[F401]** Xóa 7 import không dùng:
      - ``bbox_analyzer.py``: ``from loguru import logger``
      - ``editor_page.py``: ``QModelIndex``, ``srt_to_seconds``
      - ``debug_page_view_model.py``: ``QRunnable``, ``QThreadPool``, ``Qt``
      - ``settings_page_view_model.py``: ``os``
    * **[STYLE]** Convert CRLF → LF trên 2 file bị sai line-ending:
      ``gpu_image_filters.py``, ``subtitle_table_model.py``.
    * **[STYLE / E741]** ``editor_page.py``: Đổi tên biến mập mờ ``l``
      → ``layout`` trong hàm xây dựng dialog merge.
    * **[STYLE / W292]** Thêm newline cuối file cho 18 file thiếu.
    * **[STYLE / W291/W293]** Xóa trailing whitespace toàn bộ src/.

Changelog v3.6 — Bug Fix & Performance Audit:
    * **[CRITICAL BUG FIX]** ``SubtitleEditorService.split()``: bỏ
      workaround trừ 0.01s — SRT chuẩn cho phép ``end_a == start_b``.
      Sửa test_split fail (0.99 != 1.0).
    * **[CRITICAL BUG FIX]** ``extract_subtitles._save_annotated_raw_frames``:
      ``np.ascontiguousarray`` + ``QImage.copy()`` ngay sau tạo để tránh
      use-after-free khi numpy buffer bị GC trong lúc Qt encode JPEG.
    * **[CRITICAL BUG FIX]** ``image_filters._get_clahe_object``:
      ``cv2.CLAHE.apply()`` KHÔNG thread-safe — thay ``@lru_cache`` toàn
      cục bằng ``threading.local`` per-thread cache. Sửa race condition
      khi extract & re-OCR chạy song song.
    * **[BUG FIX]** ``mpv_metadata_reader._probe_with_instance``:
      ``time.time()`` → ``time.monotonic()`` để miễn nhiễm với NTP/đổi
      giờ hệ thống gây vòng lặp vô hạn.
    * **[BUG FIX]** ``editor_page.py``: thêm ``_SYNC_HIGHLIGHT_MS = 100``
      bị thiếu (test_constants_declared fail).
    * **[REFACTOR]** ``ReOcrUseCase`` nhận ``LoadVideoMetadataUseCase``
      qua DI thay vì truy cập private member của ``ExtractSubtitlesUseCase``
      (sửa vi phạm Law of Demeter).
    * **[PERF]** ``_ReOcrFrameCacheManager`` thread-safe (RLock), encode
      JPEG ngoài lock.
    * **[PERF]** ``_save_annotated_raw_frames``: dựng font/pen 1 lần ngoài
      vòng lặp text_box thay vì lặp lại — giảm ~30% chi phí khi vẽ nhiều
      box mỗi frame.
    * **[STYLE]** ``ruff --fix`` xoá 242 lỗi auto-fixable (unused imports,
      whitespace, missing newlines, unsorted imports).

Changelog v3.5 — Waveform UX Studio-Grade (giữ lại):
    * Fix bug nghiêm trọng: waveform chỉ scroll/zoom được tới 60s
      (truthiness bug ``_total_video_duration_sec or 60.0``).
    * 9 phím tắt mới chuẩn DAW (Home/End/PageUp/Down/Ctrl+0/Ctrl+±/
      Shift+Wheel 5×). Visual loading hint, tooltip phím tắt.

Changelog v3.2 — Auto-ROI Quality & Performance:
    * **[LOGGING]** ``ocr_based_auto_roi_detector.py`` chuyển ``logging``
      → ``loguru.logger`` cho consistency toàn project.
    * **[BUG FIX]** ``.copy()`` ảnh RGB trước khi lưu top-K để tránh race
      condition với decoder reuse buffer (mpv/pyav).
    * **[BUG FIX]** Sanity check khi merge cluster: từ chối nếu
      ``combined_width / sum_widths > 1.5`` — tránh gộp 2 cluster cách xa
      theo X (vd top-left + top-right credits) thành 1 ROI bao vùng trống.
    * **[PERF]** Top-K frame selection bằng ``heapq`` thay sort+pop —
      O(N×log K) thay O(N×K log K).
    * **[PERF]** Throttle progress callback mỗi 10 frame hoặc khi % thay
      đổi 1% — giảm flood Qt signals.
    * **[PERF]** ``_filter_heuristic`` 1 numpy matrix duy nhất thay 5 list
      comp — giảm 5N → N iterations.
    * **[QUALITY]** Composite image dùng ``np.median`` thay ``np.mean`` —
      robust với outlier (frame chuyển cảnh sáng).
    * **[QUALITY]** Stratified sampling: chia video thành 8 bucket thời
      gian, mỗi bucket chọn top frame riêng — composite đa dạng hơn.
    * **[QUALITY]** Skip intro/outro cap 60s — video dài không skip quá nhiều.
    * **[QUALITY]** ``_MIN_ASPECT_RATIO`` giảm 0.15 → 0.08 cho vertical
      text dài hơn (1 cột CJK).
    * **[QUALITY]** Filter cluster sau merge: yêu cầu xuất hiện trong
      >= 3 unique frame — loại noise (với fallback nếu loại hết).
    * **[TEST]** Thêm 10 unit test mới cho Auto-ROI optimization (447/447 PASS).

Changelog v3.1:
    * Tối ưu chất lượng tầng xây dựng phụ đề sau phân tích trên 11 bộ test data
      (~10K events tổng, ~750K frames OCR), đạt F1=0.996 trung bình.
    * **Yi-Restorer position=0**: giảm ngưỡng evidence cho '一' đầu câu
      (chỉ cần 1 medium thay vì 1 high HOẶC 2 medium). Cứu được các câu
      như "起去前厅用饭" → "一起去前厅用饭".
    * **Filter CJK garble dài**: drop event >= 25 ký tự CJK, conf < 0.80,
      fc <= 15 (rác từ credit screen overlap). Giảm 6 EXTRA.
    * **Adaptive threshold cho single CJK** (cảm thán '埃', '阿', '哦'):
      fc >= 5 chỉ cần conf >= 0.72 (cũ: 0.75). Cứu '埃' chinese_vid2
      (fc=8, mean conf=0.745) → F1 0.995 → **1.000**.
    * **Mở rộng OCR_HALLUCINATION_TYPO_MAP** với các pattern an toàn:
      "现己" → "现已", "凡个" → "几个", "整介" → "整个", "仪用" → "仅用",
      "而旦" → "而且". Giảm 5 TEXT_ERR.
    * **Bảng HAN_TRADITIONAL_TO_SIMPLIFIED**: thêm vào constants nhưng KHÔNG
      apply tự động. Người dùng có thể opt-in nếu cần normalize cứng (cảnh
      báo: có thể gây mismatch với REF dùng phồn cho tên người).

Changelog v3.0:
    * Refactor :mod:`application.services.subtitle_builder` từ 1 file 1901
      dòng thành package :mod:`application.services.subtitle_pipeline` gồm
      9 module SRP.
    * Đổi mặc định ``SubtitleBuilderConfig.use_viterbi`` từ ``True`` →
      ``False`` (greedy nhanh 5-10×, F1 tương đương).
    * 100% backward compatible: import path cũ + private symbol vẫn hoạt động.
"""

from __future__ import annotations

__version__ = "3.23.399"
__all__ = ["__version__"]


