"""Use case: phân tích ngữ cảnh toàn cục của phụ đề bằng AI.

Gửi TOÀN BỘ phụ đề nguồn (sau khử trùng lặp liên tiếp) đến Gemini để tự động
điền vào 3 trường ngữ cảnh: ngôn ngữ gốc, danh sách nhân vật đầy đủ, và tóm tắt
bối cảnh/cốt truyện đầy đủ. Người dùng chọn model AI dùng cho phân tích.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import replace

from subtitles_extractor.application.services.analysis_audit import (
    AUDIT_MAX_CALLS,
    DEFAULT_AUDIT_MODEL,
    MAX_AUDIT_ROUNDS,
    STALE_GLOSSARY_KIND,
    AuditReport,
    apply_audit_patch,
    audit_film_id,
    defect_units,
    defects_to_prompt,
    filter_declined,
    find_analysis_defects,
    note_declined,
    plan_audit_batches,
)
from subtitles_extractor.application.services.cue_completion import speaker_coverage
from subtitles_extractor.application.services.cue_label_canonicalizer import (
    canonicalize_cue_labels,
)
from subtitles_extractor.application.services.name_reading_check import (
    NameReadings,
    correct_context_names,
    describe_fix,
)
from subtitles_extractor.application.services.native_address import (
    describe_changes,
    nativize_context,
)
from subtitles_extractor.application.services.roster_register import normalize_roster_register
from subtitles_extractor.application.services.title_reconciliation import (
    describe_title_fix,
    reconcile_context_titles,
)
from subtitles_extractor.application.services.visual_cue_serializer import (
    deserialize_visual_cues,
    serialize_visual_cues,
)
from subtitles_extractor.application.use_cases.complete_speakers import (
    CompleteSpeakersUseCase,
    GeminiQuotaAwareUseCase,
)
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    SubtitleContextAnalysis,
    SubtitleTranslationError,
    SubtitleTranslatorPort,
    TranslationCancelledError,
    TranslationLine,
)

logger = logging.getLogger(__name__)

CancellationCallback = Callable[[], bool]


class AnalyzeSubtitleContextUseCase(GeminiQuotaAwareUseCase):
    """Phân tích ngữ cảnh toàn cục từ toàn bộ phụ đề bằng AI."""

    def __init__(
        self,
        translator: SubtitleTranslatorPort,
        name_readings: NameReadings | None = None,
        native_address: bool = False,
        audit_ledger: object | None = None,
    ) -> None:
        """Khởi tạo use case.

        Args:
            translator: Adapter dịch/phân tích.
            name_readings: [v3.23.567] Bảng âm Hán-Việt để sửa phiên âm tên sai chắc chắn
                (None = không soát).
            native_address: [v3.23.612] Đổi tiếng gọi kiểu Hán-Việt ("Kiếm ca") trong kết
                quả phân tích sang thuần Việt ("anh Kiếm").
            audit_ledger: [v3.23.631] Sổ "model đã bỏ qua" của bước tự kiểm tra (đối tượng có
                ``load(film_id) -> dict`` và ``save(film_id, dict)``); ``None`` = không nhớ
                giữa các lần chạy.
        """
        self._translator = translator
        self._name_readings = name_readings
        self._native_address = bool(native_address)
        self._audit_ledger = audit_ledger
        #: [v3.23.631] Tóm tắt lần tự kiểm tra gần nhất (``None`` nếu chưa chạy).
        self.last_audit_report: AuditReport | None = None

    def execute(
        self,
        source_lines: list[TranslationLine],
        target_lang: str,
        model_name: str = "gemini-3.1-flash-lite",
        cancel_cb: CancellationCallback | None = None,
        video_refs: list | None = None,
        enable_visual_cues: bool = False,
        visual_cues_batch_size: int = 150,
        progress_cb: Callable[[float, str], None] | None = None,
        prior_context: str = "",
        video_reupload_cb: Callable[..., list] | None = None,
        text_cues: bool = False,
        text_cues_model: str = "gemini-3.5-flash-lite",
        voice_source_path: str = "",
        self_audit: bool = True,
        audit_model: str = "",
        audit_fallbacks: Sequence[str] = (),
    ) -> SubtitleContextAnalysis:
        """Gửi toàn bộ phụ đề đến AI, nhận ngôn ngữ gốc + nhân vật + tóm tắt.

        Args:
            source_lines: Toàn bộ dòng phụ đề nguồn.
            target_lang:  Ngôn ngữ viết tóm tắt (vd ``"Vietnamese"``).
            model_name:   Model AI thực hiện phân tích (DÙNG CHUNG cho cả Visual Cues).
            cancel_cb:    Callback kiểm tra huỷ.
            video_refs:   (Tuỳ chọn) các đoạn video đã tải lên để đính làm ngữ cảnh —
                          gửi GỘP tất cả đoạn cho phân tích toàn cục.
            enable_visual_cues: Nếu True VÀ có video_refs, phân tích thêm "ai nói/nói
                          với ai" NGAY trong bước này, dùng CÙNG model_name.
            visual_cues_batch_size: Cỡ lô cho Visual Cues.
            progress_cb:  Callback tiến độ (cho bước Visual Cues).
            text_cues:    [v3.23.623] KHÔNG có video thì vẫn gán ai nói / nói với ai bằng
                          CHỮ (một lời gọi nhỏ cho cả tập) sau khi bảng nhân vật đã chốt.
            text_cues_model: Model cho bước đó (mặc định bản Lite rẻ, đo đúng ~96%).
            voice_source_path: Video có tiếng cạnh phụ đề → tách giọng từ âm thanh (không tải
                video lên) để gán người nói chính xác hơn (đo 97-100%).
            self_audit: [v3.23.624] Sau phân tích tự đối chiếu kết quả với phụ đề và VÁ khiếm
                          khuyết (xem :meth:`refine`); không phân tích lại từ đầu.
            audit_model: Model kiểm định (rỗng = ``DEFAULT_AUDIT_MODEL``, model rẻ 500 request/ngày;
                KHÔNG dùng ``model_name`` vì model phân tích thường chỉ 20 request/ngày).
            audit_fallbacks: [v3.23.640] Model kiểm định dự phòng, thử theo thứ tự khi model kiểm
                định hết hạn mức ngày / lỗi.

        Returns:
            :class:`SubtitleContextAnalysis` với source_lang, characters, overview,
            glossary và (tuỳ chọn) visual_cues.
        """
        if not source_lines:
            logger.warning("Không có dòng phụ đề nào để phân tích ngữ cảnh.")
            return SubtitleContextAnalysis()

        # [v3.23.132] Đăng ký callback tải lại video bằng key hiện tại (chống 403 khi
        # adapter xoay sang API key khác — file của key cũ không truy cập được).
        if video_reupload_cb is not None and hasattr(
            self._translator, "set_video_reupload_callback"
        ):
            self._translator.set_video_reupload_callback(video_reupload_cb)

        logger.info(
            "Phân tích ngữ cảnh: %d dòng nguồn → model='%s'%s%s.",
            len(source_lines), model_name,
            f", kèm {len(video_refs)} đoạn video" if video_refs else "",
            " + Visual Cues (gộp chung)" if (enable_visual_cues and video_refs) else "",
        )
        # [v3.23.37] Khi bật Visual Cues VÀ có video: lấy LUÔN cues trong cùng request
        # phân tích (đã gửi video) → tiết kiệm quota, KHÔNG gọi analyze_visual_cues riêng.
        want_cues_inline = bool(enable_visual_cues and video_refs)
        # [v3.23.643] Giữ tiến độ theo đoạn trên đĩa cho tới khi cả các bước hoàn thiện sau đó xong.
        hold = getattr(self._translator, "hold_analysis_progress", None)
        release = getattr(self._translator, "release_analysis_progress", None)
        if callable(hold) and callable(release):
            hold()
            try:
                result = self._analyze_and_refine(
                    source_lines, target_lang, model_name, cancel_cb, video_refs, want_cues_inline,
                    prior_context, visual_cues_batch_size, progress_cb, text_cues, text_cues_model,
                    voice_source_path, self_audit, audit_model, audit_fallbacks,
                )
            except BaseException:
                release(clear=False)
                raise
            release(clear=True)
        else:
            result = self._analyze_and_refine(
                source_lines, target_lang, model_name, cancel_cb, video_refs, want_cues_inline,
                prior_context, visual_cues_batch_size, progress_cb, text_cues, text_cues_model,
                voice_source_path, self_audit, audit_model, audit_fallbacks,
            )
        char_preview = result.characters[:80] + "…" if len(result.characters) > 80 else result.characters
        logger.info(
            "Phân tích hoàn tất: lang=%r, nhân vật=%r, tóm tắt=%d ký tự%s.",
            result.source_lang, char_preview, len(result.overview),
            f", {len(result.visual_cues)} ký tự cues" if result.visual_cues else "",
        )
        return result

    def _analyze_and_refine(
        self,
        source_lines: list[TranslationLine],
        target_lang: str,
        model_name: str,
        cancel_cb: CancellationCallback | None,
        video_refs: list | None,
        want_cues_inline: bool,
        prior_context: str,
        visual_cues_batch_size: int,
        progress_cb: Callable[[float, str], None] | None,
        text_cues: bool,
        text_cues_model: str,
        voice_source_path: str,
        self_audit: bool,
        audit_model: str,
        audit_fallbacks: Sequence[str],
    ) -> SubtitleContextAnalysis:
        """[v3.23.643] Phân tích + hoàn thiện người nói + tự kiểm tra (tách ra để bọc giữ tiến độ)."""
        result = self._translator.analyze_global_context(
            source_lines=source_lines,
            target_lang=target_lang,
            model_name=model_name,
            cancel_cb=cancel_cb,
            video_refs=video_refs,
            with_visual_cues=want_cues_inline,
            prior_context=prior_context,
        )
        result = self._normalize_context(result)
        # [v3.23.600] Dòng có lời mà phân tích bỏ trống NGƯỜI NÓI → hỏi lại đúng các dòng
        # đó, để lồng tiếng có người nói cho mọi dòng.
        if want_cues_inline:
            result = self._complete_speakers(
                result, source_lines, target_lang, model_name, video_refs or [],
                visual_cues_batch_size, cancel_cb, progress_cb,
            )
        if text_cues and not want_cues_inline and not result.visual_cues:
            result = self._label_speakers_from_text(
                result, source_lines, target_lang, text_cues_model or model_name,
                model_name, cancel_cb, progress_cb, voice_source_path,
            )
        if self_audit:
            result = self.refine(
                result, source_lines, target_lang, audit_model or DEFAULT_AUDIT_MODEL,
                cancel_cb, fallback_models=audit_fallbacks)
        return result

    def _normalize_context(self, result: SubtitleContextAnalysis) -> SubtitleContextAnalysis:
        """Sửa tất định kết quả phân tích: phiên âm tên, chức danh, xưng hô thuần Việt.

        Dùng cho cả lượt phân tích lẫn bản vá của :meth:`refine` (bản vá cũng có thể thêm tên
        sai âm Hán-Việt).
        """
        # [v3.23.567] Model đọc sai âm Hán-Việt của tên (ca thật: 柳慕云 → "Liễu Mộc Vân",
        # đúng là "Liễu Mộ Vân") thì bản dịch chép sai theo ở MỌI dòng nhắc tên. Sửa tất
        # định ngay tại nguồn, trước khi kết quả tới giao diện và được lưu vào phiên.
        result, fixes = correct_context_names(result, self._name_readings)
        for fix in fixes:
            logger.warning("Sửa phiên âm tên sai âm Hán-Việt — %s.", describe_fix(fix))
        # [v3.23.589] Bảng nhân vật gọi chức danh bằng âm Hán-Việt ("gọi 'Sở trưởng'")
        # trong khi bảng thuật ngữ của CHÍNH lượt phân tích này dịch khác ("所長 = Trưởng
        # đồn công an") → bản dịch dùng lẫn cả hai. Bảng thuật ngữ là chuẩn.
        result, title_fixes = reconcile_context_titles(result, self._name_readings)
        for title_fix in title_fixes:
            logger.warning(
                "Bảng nhân vật gọi chức danh khác bảng thuật ngữ — đổi theo bảng thuật "
                "ngữ: %s.",
                describe_title_fix(title_fix),
            )
        # [v3.23.637] Người dưới tự xưng 'ta' với bậc trên (sư phụ/sư thúc/chưởng giáo…) → 'đệ tử'.
        characters, register_fixes = normalize_roster_register(result.characters, result.overview)
        if register_fixes:
            logger.info(
                "Sửa %d mệnh đề xưng hô sai thế giới phim (đại từ hiện đại / người dưới xưng 'ta' "
                "với bậc trên).", register_fixes,
            )
            result = replace(result, characters=characters)
        # [v3.23.612] Tuỳ chọn xưng hô thuần Việt: prompt đã dặn model, phần model vẫn
        # viết kiểu Hán-Việt thì đổi tất định — sau khi sửa tên để đổi trên tên đúng.
        if self._native_address:
            result, native_changes = nativize_context(result)
            if native_changes:
                logger.info(
                    "Xưng hô thuần Việt trong kết quả phân tích: %s.",
                    describe_changes(native_changes),
                )
        return result

    def refine(
        self,
        analysis: SubtitleContextAnalysis,
        source_lines: list[TranslationLine],
        target_lang: str,
        model_name: str,
        cancel_cb: CancellationCallback | None = None,
        max_rounds: int = MAX_AUDIT_ROUNDS,
        fallback_models: Sequence[str] = (),
    ) -> SubtitleContextAnalysis:
        """[v3.23.624] Tự kiểm tra phân tích cũ rồi VÁ khiếm khuyết (không phân tích lại).

        Mỗi lượt: đối chiếu tất định (:func:`find_analysis_defects`); không có khiếm khuyết thì
        dừng, KHÔNG gọi AI. Có thì hỏi model bản vá và gộp vào kết quả cũ — mục người dùng đã sửa
        hay đã đúng được giữ. Lỗi (trừ huỷ) không làm hỏng kết quả cũ.

        [v3.23.631] RẺ và HỘI TỤ: khiếm khuyết chia thành các lô nhỏ theo vị trí trong phim (mỗi
        lô một lời gọi, prompt ≤ ``AUDIT_MAX_PROMPT_LINES`` dòng, tối đa ``AUDIT_MAX_CALLS`` lời
        gọi); đơn vị khiếm khuyết đã gửi mà vẫn còn sau bản vá được ghi vào SỔ — bị bỏ qua đủ
        ``DECLINE_LIMIT`` lần thì thôi không gửi nữa (cả ở các lần chạy sau). Kết quả tóm tắt ở
        :attr:`last_audit_report`.

        Returns:
            Phân tích đã vá (hoặc nguyên bản cũ).
        """
        audit = getattr(self._translator, "audit_analysis", None)
        report = AuditReport()
        self.last_audit_report = report
        if audit is None or analysis.stop_reason or not source_lines:
            return analysis
        models = list(dict.fromkeys(m for m in (model_name, *fallback_models) if m))
        valid_ids = {ln.index for ln in source_lines}
        line_order = [ln.index for ln in source_lines]
        source_text = "\n".join(
            str(getattr(ln, "original_text", None) or getattr(ln, "text", "") or "")
            for ln in source_lines)
        film_id = audit_film_id(source_lines, target_lang)
        ledger = self._audit_ledger
        declined: dict[str, int] = dict(ledger.load(film_id)) if ledger is not None else {}
        calls_left = AUDIT_MAX_CALLS
        stopped = False
        for round_no in range(1, max_rounds + 1):
            found = find_analysis_defects(analysis, source_lines)
            defects = filter_declined(found, declined)
            if round_no == 1:
                report.found = sum(len(defect_units(d)) for d in found)
                report.blocked = sum(
                    len(defect_units(d)) for d in found if d.kind != STALE_GLOSSARY_KIND
                ) - sum(
                    len(defect_units(d)) for d in defects if d.kind != STALE_GLOSSARY_KIND
                )
            # Thuật ngữ "không thấy trong phụ đề" chỉ là gợi ý (OCR có thể viết phồn thể):
            # đi kèm khiếm khuyết khác thì gửi, đứng một mình thì không đáng một lời gọi AI.
            if not any(d.kind != STALE_GLOSSARY_KIND for d in defects):
                break
            logger.info(
                "Tự kiểm tra phân tích (lượt %d): %d khiếm khuyết (%s)%s.", round_no, len(defects),
                ", ".join(sorted({d.kind for d in defects})),
                f", bỏ qua {report.blocked} mục model đã không sửa được"
                if report.blocked else "")
            round_changes = 0
            for batch in plan_audit_batches(defects, line_order):
                if calls_left <= 0:
                    stopped = True
                    break
                calls_left -= 1
                cues = {c.line_no: (c.speaker, c.addressee)
                        for c in deserialize_visual_cues(analysis.visual_cues)}
                focus = sorted({i for d in batch for i in d.line_ids})
                patch = None
                try:
                    patch = self._audit_with_fallbacks(
                        audit, models, source_lines, target_lang, analysis, cues,
                        defects_to_prompt(batch), focus, cancel_cb)
                except TranslationCancelledError:
                    if cancel_cb is not None and cancel_cb():
                        raise
                    stopped = True
                    break
                if patch is None:
                    logger.warning("Tự kiểm tra phân tích thất bại ở mọi model — giữ kết quả cũ.")
                    stopped = True
                    break
                report.calls += 1
                try:
                    analysis, changes = apply_audit_patch(
                        analysis, patch or {}, valid_ids, source_text)
                except Exception as exc:  # noqa: BLE001 — bản vá hỏng không được làm mất phân tích
                    logger.warning("Bản vá tự kiểm tra không áp được — giữ kết quả cũ: %s", exc)
                    stopped = True
                    break
                if changes:
                    analysis = self._normalize_context(analysis)
                round_changes += changes
                report.patched += changes
                declined = note_declined(
                    declined, batch, find_analysis_defects(analysis, source_lines))
            logger.info(
                "Tự kiểm tra phân tích (lượt %d): đã vá %d mục.", round_no, round_changes)
            if stopped or not round_changes:
                break
        if ledger is not None:
            ledger.save(film_id, declined)
        report.remaining = sum(
            len(defect_units(d))
            for d in filter_declined(find_analysis_defects(analysis, source_lines), declined)
            if d.kind != STALE_GLOSSARY_KIND
        )
        return analysis

    def _audit_with_fallbacks(
        self,
        audit: Callable[..., dict],
        models: Sequence[str],
        source_lines: list[TranslationLine],
        target_lang: str,
        analysis: SubtitleContextAnalysis,
        cues: dict[int, tuple[str, str]],
        defects_json: str,
        focus: list[int],
        cancel_cb: CancellationCallback | None,
    ) -> dict | None:
        """[v3.23.640] Hỏi bản vá bằng model kiểm định đầu tiên dùng được; hỏng thì thử model kế.

        Trước đây model kiểm định bị CỐ ĐỊNH ``gemini-3.1-flash-lite`` (bỏ qua lựa chọn của người
        dùng), còn lỗi hết hạn mức ngày (``QuotaExhaustedError``) của model đó không được bắt ở
        đây và làm hỏng cả lượt phân tích. Model hết hạn mức ngày ở mọi key bị bỏ qua không tốn lời
        gọi nào.

        Returns:
            Bản vá, hoặc ``None`` nếu mọi model đều không dùng được.
        """
        has_quota = getattr(self._translator, "has_any_daily_quota", None)
        for model in models:
            if has_quota is not None:
                try:
                    if not has_quota(model):
                        logger.info("Tự kiểm tra: model '%s' đã hết hạn mức ngày ở mọi key → bỏ qua.",
                                    model)
                        continue
                except (AttributeError, TypeError):
                    pass
            try:
                return audit(
                    source_lines, target_lang, model, analysis.characters, analysis.glossary,
                    analysis.overview, cues, defects_json, focus, cancel_cb=cancel_cb)
            except TranslationCancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — bước tuỳ chọn: lỗi bất kỳ chỉ thử model kế
                logger.warning("Tự kiểm tra bằng '%s' thất bại (%s)%s.", model, exc,
                               " → thử model dự phòng" if model != models[-1] else "")
        return None

    def _label_speakers_from_text(
        self,
        result: SubtitleContextAnalysis,
        source_lines: list[TranslationLine],
        target_lang: str,
        model_name: str,
        fallback_model: str,
        cancel_cb: CancellationCallback | None,
        progress_cb: Callable[[float, str], None] | None,
        voice_source_path: str = "",
    ) -> SubtitleContextAnalysis:
        """[v3.23.623] Gán người nói/người nghe từ CHỮ khi không có video.

        Chạy SAU khi bảng nhân vật đã chốt (đã sửa phiên âm tên) để nhãn khớp tên trong bảng.
        Lỗi (trừ huỷ) không làm hỏng kết quả phân tích.
        """
        if result.stop_reason:
            return result
        analyze = getattr(self._translator, "analyze_text_cues", None)
        if analyze is None:
            return result
        characters = result.characters or ""
        voice_kwargs = (
            {"voice_source": voice_source_path, "source_lang": result.source_lang or ""}
            if voice_source_path and hasattr(self._translator, "label_voices") else {}
        )
        for model in dict.fromkeys(m for m in (model_name, fallback_model) if m):
            try:
                cues = analyze(
                    source_lines, target_lang, model_name=model, characters=characters,
                    cancel_cb=cancel_cb, **voice_kwargs,
                    progress_cb=(lambda p: progress_cb(p, "Gán người nói từ phụ đề"))
                    if progress_cb is not None else None,
                )
            except TranslationCancelledError:
                if cancel_cb is not None and cancel_cb():
                    raise
                continue
            except SubtitleTranslationError as exc:
                logger.warning("Gán người nói bằng chữ thất bại với '%s' — %s", model, exc)
                continue
            if cues:
                cues = canonicalize_cue_labels(cues, characters)
                logger.info(
                    "Đã gán người nói/người nghe từ phụ đề cho %d dòng (model '%s').",
                    len(cues), model)
                return replace(result, visual_cues=serialize_visual_cues(cues))
        return result

    def _complete_speakers(
        self,
        result: SubtitleContextAnalysis,
        source_lines: list[TranslationLine],
        target_lang: str,
        model_name: str,
        video_refs: list,
        batch_size: int,
        cancel_cb: CancellationCallback | None,
        progress_cb: Callable[[float, str], None] | None,
    ) -> SubtitleContextAnalysis:
        """[v3.23.600] Hỏi lại người nói cho mọi dòng có lời mà phân tích bỏ trống.

        Dùng chung lõi với nút "Bổ sung người nói" (:class:`CompleteSpeakersUseCase`), kèm
        nhãn đã có làm mạch hội thoại, tối đa hai lượt; nhãn vừa điền mà "lạc cảnh" bị bỏ.
        Phân tích dừng giữa chừng (hết hạn mức, quá tải) thì không hỏi — lượt chạy tiếp sẽ
        làm. Lỗi của lượt hỏi không làm hỏng kết quả phân tích.

        Returns:
            Kết quả phân tích với Visual Cues đã điền (hoặc nguyên kết quả cũ).
        """
        if not result.visual_cues or result.stop_reason:
            return result
        if getattr(self._translator, "analyze_visual_cues", None) is None:
            return result
        characters = result.characters or ""
        cues = canonicalize_cue_labels(
            deserialize_visual_cues(result.visual_cues), characters
        )
        before = speaker_coverage(cues, source_lines)
        if not before.missing:
            return result
        logger.warning(
            "Phân tích để trống NGƯỜI NÓI ở %d/%d dòng có lời → hỏi lại đúng các dòng đó "
            "bằng '%s' (lồng tiếng cần người nói cho mọi dòng).",
            len(before.missing), before.spoken_lines, model_name,
        )
        outcome = CompleteSpeakersUseCase(self._translator).complete(
            cues, source_lines, characters=characters, target_lang=target_lang,
            model_name=model_name, video_refs=list(video_refs), batch_size=batch_size,
            cancel_cb=cancel_cb, progress_cb=progress_cb,
        )
        if not outcome.changed:
            return result
        return replace(result, visual_cues=serialize_visual_cues(outcome.cues))


__all__ = ["AnalyzeSubtitleContextUseCase"]
