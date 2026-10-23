"""[v3.23.593] Điều phối LỒNG TIẾNG trang TTS: nạp vai → giao giọng → giọng từng dòng.

Tách khỏi ``tts_page`` (đã gần 3000 dòng) để trang chỉ việc nối: bảng phân vai
(:class:`DubbingCastPanel`), view model (tìm người nói, phân vai), bộ giọng của engine
đang chọn, và QSettings (nhớ lựa chọn giọng của người dùng theo từng engine).

[v3.23.597] Hai cách phân giọng: mỗi nhân vật một giọng, hoặc CHỈ HAI GIỌNG (một nam, một
nữ). Ở chế độ hai giọng, giọng nam/nữ nhớ theo engine; lựa chọn "vai này đọc giọng nam hay
nữ" nhớ theo TÊN VAI (không phụ thuộc engine — giới tính là của nhân vật).

[v3.23.598] Sửa tay người nói TỪNG DÒNG (khâu phân tích gán nhầm người): bảng dòng của
trang TTS hỏi người nói/giọng từng dòng và vai để chọn qua :meth:`line_speaker_cells`,
:meth:`role_choices`; chọn xong gọi :meth:`set_line_speaker`.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from PySide6.QtCore import QObject, QSettings, Signal

from subtitles_extractor.application.services.cue_label_canonicalizer import (
    canonical_label,
    roster_label_map,
)
from subtitles_extractor.application.services.dubbing_cast import (
    CAST_MODE_PER_ROLE,
    CAST_MODE_TWO_VOICES,
    CAST_MODES,
    CastMember,
    DubbingSource,
    canonical_voice_id,
    default_gender_voices,
    role_choices,
    two_voice_sides,
)
from subtitles_extractor.application.services.dubbing_line_edits import (
    LineSpeaker,
    describe_line_speakers,
)
from subtitles_extractor.application.services.speaker_review import (
    REVIEW_GENDER,
    REVIEW_SELF_CALL,
    REVIEW_UNKNOWN,
    REVIEW_VOICE_ONLY,
    SpeakerIssue,
)
from subtitles_extractor.domain.value_objects.voice_profile import (
    VOICE_FEMALE,
    VOICE_MALE,
    VoiceProfile,
)
from subtitles_extractor.presentation.view_models.tts_page_view_model import (
    TTSPageViewModel,
)
from subtitles_extractor.presentation.widgets.dubbing_cast_panel import DubbingCastPanel

logger = logging.getLogger(__name__)

#: Khoá QSettings: bật lồng tiếng; lựa chọn giọng theo engine (``{tên vai: voice_id}``).
SETTINGS_ENABLED_KEY = "dubbing_enabled"
SETTINGS_CASTING_PREFIX = "dubbing_cast_"
#: Số vai tối đa nhớ giọng cho mỗi engine (nhiều phim/tập).
MAX_SAVED_ROLES = 400
#: [v3.23.597] Cách phân giọng; giọng nam/nữ theo engine (``{"male", "female"}``); giới
#: cho vai chưa rõ giới tính; giới người dùng tự chọn cho từng vai (``{tên vai: giới}``).
SETTINGS_MODE_KEY = "dubbing_mode"
SETTINGS_GENDER_VOICES_PREFIX = "dubbing_gender_voices_"
SETTINGS_UNKNOWN_SIDE_KEY = "dubbing_unknown_side"
SETTINGS_ROLE_SIDES_KEY = "dubbing_role_sides"

CatalogProvider = Callable[[], tuple[str, list[VoiceProfile]]]
SourceProvider = Callable[[], DubbingSource]
MainVoiceProvider = Callable[[], str]
#: Dấu đánh dấu dòng người dùng chọn tay người nói.
EDITED_MARK = "✎"
#: [v3.23.599] Dấu đánh dấu dòng cần soát người nói.
REVIEW_MARK = "⚠"


@dataclass(frozen=True, slots=True)
class SpeakerCell:
    """[v3.23.598] Ô "Người nói · Giọng" của một dòng trên bảng dòng.

    Attributes:
        text: Chữ hiện trong ô.
        tooltip: Giải thích (vai tự tìm khi đã sửa tay; [v3.23.599] lý do cần soát).
        overridden: Người dùng đã chọn tay người nói.
        needs_review: [v3.23.599] Dòng cần soát người nói (chưa rõ / nhãn chắc sai).
    """

    text: str
    tooltip: str
    overridden: bool
    needs_review: bool = False


def short_voice_label(label: str) -> str:
    """Tên giọng gọn cho ô bảng: "⭐ Adam bựa — Nam · Bắc · …" → "Adam bựa".

    Args:
        label: Nhãn giọng của engine.

    Returns:
        Phần tên trước dấu " — " (bỏ ngôi sao "giọng nổi bật" của VieNeu).
    """
    return label.replace("⭐", "").split(" — ", 1)[0].strip()


class TTSDubbingCoordinator(QObject):
    """Nối bảng phân vai với view model, bộ giọng engine và cài đặt đã lưu.

    Signals:
        line_speakers_changed: [v3.23.598] Người nói / giọng của các dòng có thể đã đổi
            (nạp vai, đổi giọng một vai, đổi cách phân giọng, sửa tay người nói).
    """

    line_speakers_changed = Signal()

    def __init__(
        self,
        panel: DubbingCastPanel,
        view_model: TTSPageViewModel,
        settings: QSettings,
        translate: Callable[[str], str],
        catalog_provider: CatalogProvider,
        parent: QObject | None = None,
        main_voice_provider: MainVoiceProvider | None = None,
    ) -> None:
        """Khởi tạo và nối tín hiệu của bảng phân vai.

        Args:
            panel: Bảng phân vai.
            view_model: View model trang TTS (giữ phụ đề nguồn, tìm người nói).
            settings: QSettings của trang TTS.
            translate: Hàm dịch chuỗi giao diện (``"tts.<khoá>"`` → chuỗi).
            catalog_provider: Trả ``(khoá engine, bộ giọng)`` của engine đang chọn.
            parent: QObject cha.
            main_voice_provider: [v3.23.597] Giọng chung đang chọn ở tab Engine — giữ cho
                giới của nó khi lần đầu dùng chế độ hai giọng.
        """
        super().__init__(parent)
        self._panel = panel
        self._view_model = view_model
        self._settings = settings
        self._translate = translate
        self._catalog_provider = catalog_provider
        self._main_voice_provider = main_voice_provider
        self._source_provider: SourceProvider | None = None
        # [v3.23.596] Số vai đang dùng giọng ĐÃ LƯU khác với giọng phân vai tự động.
        self._saved_overrides = 0
        panel.load_requested.connect(self.load_cast)
        panel.auto_cast_requested.connect(self.auto_cast)
        panel.casting_changed.connect(self._save_casting)
        panel.enabled_changed.connect(self._save_enabled)
        panel.mode_changed.connect(self._on_mode_changed)
        panel.gender_voices_changed.connect(self._on_gender_voices_changed)
        panel.set_dubbing_enabled(
            bool(settings.value(SETTINGS_ENABLED_KEY, False, type=bool))
        )
        mode = str(settings.value(SETTINGS_MODE_KEY, CAST_MODE_PER_ROLE, type=str))
        panel.set_cast_mode(mode if mode in CAST_MODES else CAST_MODE_PER_ROLE)

    def set_source_provider(self, provider: SourceProvider) -> None:
        """Nguồn ngữ cảnh phân vai (bảng nhân vật + Visual Cues ở trang Dịch)."""
        self._source_provider = provider

    def load_cast(self) -> bool:
        """Tìm người nói từng dòng, lập danh sách vai, giao giọng (đã lưu/tự động).

        Returns:
            ``True`` nếu tìm được ít nhất một vai.
        """
        if not self._view_model.has_source:
            self._panel.set_cast([], [], {})
            self._panel.set_status(self._translate("tts.dub_status_no_source"))
            self.line_speakers_changed.emit()
            return False
        source = self._current_source()
        speakers = self._view_model.prepare_dubbing(source)
        # [v3.23.594] Trang Dịch có cues mà không ghép được → phụ đề TTS là phim/tập khác.
        mismatch = bool(source.visual_cues.strip()) and not (
            self._view_model.dubbing_cues_matched
        )
        if not self._view_model.dubbing_cast:
            self._panel.set_cast([], [], {})
            self._panel.set_status(self._translate(
                "tts.dub_status_mismatch" if mismatch else "tts.dub_status_none"
            ))
            self.line_speakers_changed.emit()
            return False
        self._show_cast(prefer_saved=True)
        if mismatch:
            self._panel.set_status(
                f"{self._panel.status()} {self._translate('tts.dub_status_tags_only')}"
            )
        voiced = sum(1 for speaker in speakers if speaker)
        logger.info(
            "Lồng tiếng: %d vai, %d/%d dòng biết người nói%s.",
            len(self._view_model.dubbing_cast), voiced, len(speakers),
            " (Visual Cues không khớp phụ đề này — chỉ dùng tag)" if mismatch else "",
        )
        return True

    def auto_cast(self) -> None:
        """Giao lại giọng tự động, bỏ qua lựa chọn đã lưu."""
        if self._view_model.dubbing_cast:
            self._show_cast(prefer_saved=False)
            self._save_casting()

    def refresh_catalog(self) -> None:
        """Engine/ngôn ngữ/chế độ đổi → đổi bộ giọng trong bảng (giữ lựa chọn đã lưu)."""
        if self._view_model.dubbing_cast:
            self._show_cast(prefer_saved=True)

    def line_voices(self) -> tuple[str, ...]:
        """Giọng từng dòng cho lần TTS sắp chạy; rỗng khi tắt lồng tiếng.

        [v3.23.594] Nạp lại vai NGAY lúc tổng hợp từ ngữ cảnh HIỆN TẠI của trang Dịch. Bản
        593 dùng vai nạp lúc trước: nạp phụ đề khác, "Tổng hợp cả bộ" các tập khác, hay mở
        lại ứng dụng (ô lồng tiếng vẫn bật nhưng chưa nạp vai) đều cho kết quả sai — giọng
        vai phim cũ gán cho phụ đề mới, hoặc âm thầm chỉ một giọng. Lựa chọn giọng của
        người dùng nằm trong QSettings nên nạp lại không mất.
        """
        if not self._panel.is_dubbing_enabled():
            return ()
        if not self.load_cast():
            logger.warning(
                "Lồng tiếng đang bật nhưng không tìm được người nói cho phụ đề này — "
                "đọc bằng một giọng."
            )
            return ()
        voices = self._view_model.dubbing_line_voices(self._panel.casting())
        if not voices:
            logger.warning(
                "Lồng tiếng đang bật nhưng chưa vai nào có giọng (engine chưa có danh "
                "sách giọng?) — đọc bằng một giọng."
            )
        return voices

    def line_speakers(self) -> list[LineSpeaker]:
        """[v3.23.598] Người nói (sau sửa tay), vai tự tìm, giọng của từng dòng nguồn."""
        return describe_line_speakers(
            self._view_model.auto_line_speakers(),
            self._view_model.speaker_overrides,
            self._panel.casting(),
        )

    def line_speaker_cells(self) -> list[SpeakerCell]:
        """[v3.23.598] Chữ ô "Người nói · Giọng" của từng dòng nguồn."""
        names = self._display_names()
        _engine_key, catalog = self._catalog_provider()
        voices = {
            profile.voice_id: short_voice_label(profile.display) for profile in catalog
        }
        shared = self._translate("tts.dub_voice_shared")
        issues = self._view_model.speaker_issues()
        cells: list[SpeakerCell] = []
        for position, line in enumerate(self.line_speakers()):
            if line.speaker:
                voice = voices.get(line.voice, line.voice) if line.voice else shared
                text = f"{names.get(line.speaker, line.speaker)} · {voice}"
            else:
                text = shared
            if line.overridden:
                auto = names.get(line.auto_speaker, line.auto_speaker) or "?"
                tip = self._translate("tts.line_speaker_tip_edited")
                cells.append(SpeakerCell(
                    f"{EDITED_MARK} {text}", tip.replace("{auto}", auto), True
                ))
            elif position in issues:
                cells.append(SpeakerCell(
                    f"{REVIEW_MARK} {text}", self.issue_text(issues[position]), False,
                    needs_review=True,
                ))
            else:
                cells.append(SpeakerCell(
                    text, self._translate("tts.line_speaker_tip"), False
                ))
        return cells

    def review_positions(self) -> list[int]:
        """[v3.23.599] Vị trí các dòng cần soát người nói."""
        return sorted(self._view_model.speaker_issues())

    def issue_text(self, issue: SpeakerIssue) -> str:
        """[v3.23.599] Lý do một dòng cần soát người nói, đọc được cho người dùng."""
        if issue.reason == REVIEW_SELF_CALL:
            name = self.display_name(issue.detail)
            return (
                self._translate("tts.review_self_call")
                .replace("{word}", issue.word).replace("{name}", name)
            )
        if issue.reason == REVIEW_VOICE_ONLY:
            return self._translate("tts.review_voice_only").replace(
                "{voice}", issue.detail
            )
        if issue.reason == REVIEW_GENDER:
            return (
                self._translate("tts.review_gender")
                .replace("{word}", issue.word)
                .replace("{gender}", self._gender_word(issue.detail))
            )
        return self._translate("tts.line_speaker_tip_unknown")

    def unknown_gender_roles(self) -> list[CastMember]:
        """[v3.23.599] Vai chưa rõ giới tính (giọng nam hay nữ do người dùng quyết)."""
        return [member for member in self._view_model.dubbing_cast if not member.gender]

    def _gender_word(self, gender: str) -> str:
        return self._translate(
            "tts.dub_gender_female" if gender == VOICE_FEMALE else "tts.dub_gender_male"
        )

    def role_choices(self) -> list[CastMember]:
        """[v3.23.598] Vai để chọn khi sửa tay người nói một dòng.

        Nạp vai trước nếu phụ đề đang nạp chưa được tìm người nói.
        """
        if self._view_model.has_source and not self._view_model.dubbing_prepared:
            self.load_cast()
        return role_choices(self._view_model.dubbing_cast, self._subtitle_roster())

    def display_name(self, speaker: str) -> str:
        """[v3.23.598] Tên đọc được của vai (phiên âm trong bảng nhân vật nếu có)."""
        return self._display_names().get(speaker, speaker)

    def voice_label(self, voice_id: str) -> str:
        """[v3.23.598] Tên giọng gọn của engine đang chọn (rỗng = giọng chung)."""
        if not voice_id:
            return self._translate("tts.dub_voice_shared")
        _engine_key, catalog = self._catalog_provider()
        for profile in catalog:
            if profile.voice_id == voice_id:
                return short_voice_label(profile.display)
        return voice_id

    def set_line_speaker(self, positions: Iterable[int], speaker: str | None) -> int:
        """[v3.23.598] Người dùng sửa người nói của các dòng.

        Tên gõ tay được quy về tên chuẩn trong bảng nhân vật (gõ "Chân Kiếm Thanh" cũng
        là vai 甄劍聲). Sửa người nói là muốn đọc nhiều giọng nên lồng tiếng được bật nếu
        đang tắt; bảng phân vai nạp lại để vai mới (nếu có) nhận giọng.

        Args:
            positions: Vị trí dòng.
            speaker: Tên vai; rỗng = giọng chung; ``None`` = trả về người nói tự tìm.

        Returns:
            Số dòng thay đổi.
        """
        if speaker:
            label_map = roster_label_map(self._subtitle_roster())
            speaker = canonical_label(speaker.strip(), label_map)
        changed = self._view_model.set_speaker_override(positions, speaker)
        if not changed:
            return 0
        if not self._panel.is_dubbing_enabled():
            self._panel.set_dubbing_enabled(True)
            self._save_enabled(True)
        self._reload_keeping_voices()
        return changed

    def revert_lines(self, positions: Iterable[int]) -> list[int]:
        """[v3.23.598] Bỏ sửa tay của các dòng (lời về chữ lúc nạp, người nói tự tìm).

        Args:
            positions: Vị trí dòng.

        Returns:
            Vị trí các dòng vừa được trả lại lời gốc.
        """
        wanted = list(positions)
        overridden = self._view_model.speaker_overrides
        had_speaker = any(position in overridden for position in wanted)
        reverted = self._view_model.revert_line_edits(wanted)
        if had_speaker or (reverted and self._view_model.dubbing_prepared):
            self._reload_keeping_voices()
        return reverted

    def reload_cast(self) -> None:
        """[v3.23.625] Nạp lại vai sau khi danh sách dòng đổi (thêm/xoá dòng), giữ giọng đã chọn."""
        if self._view_model.has_source:
            self._reload_keeping_voices()

    def _reload_keeping_voices(self) -> None:
        """Nạp lại vai sau khi người nói các dòng đổi, GIỮ giọng các vai đang có.

        Phân vai tự động tính theo số câu và mức đối đáp giữa các vai — đổi người nói một
        dòng là đổi cả hai thứ đó, nên vai KHÁC có thể bị giao giọng khác trong khi audio
        đã tổng hợp của chúng vẫn là giọng cũ. Ghim giọng đang hiện (như người dùng tự
        chọn) trước khi nạp lại: chỉ vai mới xuất hiện mới nhận giọng tự động.
        """
        if self._panel.cast_mode() == CAST_MODE_PER_ROLE and self._panel.has_cast():
            self._save_casting()
        self.load_cast()

    def _current_source(self) -> DubbingSource:
        return self._source_provider() if self._source_provider else DubbingSource()

    def _subtitle_roster(self) -> str:
        """Bảng nhân vật của trang Dịch, nếu nó thuộc CÙNG phim với phụ đề TTS.

        Visual Cues không ghép được vào phụ đề TTS (phim/tập khác) thì nhân vật trong bảng
        cũng là của phim khác — không đưa vào danh sách vai để chọn.
        """
        source = self._current_source()
        mismatch = bool(source.visual_cues.strip()) and not (
            self._view_model.dubbing_cues_matched
        )
        return "" if mismatch else source.characters

    def _display_names(self) -> dict[str, str]:
        return {
            member.name: member.display_name for member in self._view_model.dubbing_cast
        }

    def _show_cast(self, *, prefer_saved: bool) -> None:
        engine_key, catalog = self._catalog_provider()
        cast = self._view_model.dubbing_cast
        if not catalog:
            self._panel.set_cast(cast, [], {})
            self._panel.set_status(self._translate("tts.dub_status_no_voices"))
            self.line_speakers_changed.emit()
            return
        if self._panel.cast_mode() == CAST_MODE_TWO_VOICES:
            self._show_two_voice_cast(engine_key, catalog, prefer_saved=prefer_saved)
            return
        casting = self._view_model.auto_cast(catalog)
        self._saved_overrides = 0
        if prefer_saved:
            known = {profile.voice_id for profile in catalog}
            saved = {
                name: voice
                for name, voice in (
                    (name, canonical_voice_id(voice, catalog))
                    for name, voice in self._saved_casting(engine_key).items()
                )
                if name in casting and (voice in known or voice == "")
            }
            # [v3.23.596] Lựa chọn đã lưu THẮNG phân vai tự động một cách lặng lẽ — lượt
            # thật 10/10/2026: 19 vai nhưng chỉ 2 giọng (nam/nữ) do lựa chọn lưu từ trước,
            # trong khi phân vai tự động cho 19 giọng. Đếm để dòng trạng thái nói ra.
            self._saved_overrides = sum(
                1 for name, voice in saved.items() if casting.get(name) != voice
            )
            casting.update(saved)
        self._panel.set_cast(cast, catalog, casting)
        self._update_status()

    def _show_two_voice_cast(
        self, engine_key: str, catalog: list[VoiceProfile], *, prefer_saved: bool
    ) -> None:
        """[v3.23.597] Chế độ hai giọng: vai nam đọc giọng nam, vai nữ đọc giọng nữ.

        ``prefer_saved=False`` ("Phân vai tự động") bỏ các lựa chọn nam/nữ tự đổi của từng
        vai, quay về giới tính của vai; giọng nam/nữ đã chọn giữ nguyên.
        """
        self._saved_overrides = 0
        unknown_side = self._saved_unknown_side()
        overrides = self._saved_role_sides() if prefer_saved else {}
        self._panel.set_two_voice_cast(
            self._view_model.dubbing_cast,
            catalog,
            self._gender_voices(engine_key, catalog),
            two_voice_sides(self._view_model.dubbing_cast, unknown_side, overrides),
            unknown_side,
        )
        self._update_status()

    def _gender_voices(
        self, engine_key: str, catalog: list[VoiceProfile]
    ) -> tuple[str, str]:
        """Giọng nam/nữ đã lưu cho engine (nếu engine còn giọng đó) hoặc mặc định."""
        main_voice = self._main_voice_provider() if self._main_voice_provider else ""
        male, female = default_gender_voices(catalog, main_voice)
        known = {profile.voice_id for profile in catalog}
        saved = self._load_json(f"{SETTINGS_GENDER_VOICES_PREFIX}{engine_key}")
        saved = {
            side: canonical_voice_id(voice, catalog) for side, voice in saved.items()
        }
        if saved.get(VOICE_MALE) in known:
            male = saved[VOICE_MALE]
        if saved.get(VOICE_FEMALE) in known:
            female = saved[VOICE_FEMALE]
        return male, female

    def _saved_unknown_side(self) -> str:
        side = str(self._settings.value(SETTINGS_UNKNOWN_SIDE_KEY, VOICE_MALE, type=str))
        return side if side in (VOICE_MALE, VOICE_FEMALE) else VOICE_MALE

    def _saved_role_sides(self) -> dict[str, str]:
        return {
            name: side for name, side in self._load_json(SETTINGS_ROLE_SIDES_KEY).items()
            if side in (VOICE_MALE, VOICE_FEMALE)
        }

    def _load_json(self, key: str) -> dict[str, str]:
        raw = self._settings.value(key, "", type=str)
        try:
            loaded = json.loads(raw) if raw else {}
        except ValueError:
            logger.warning("Cài đặt lồng tiếng '%s' hỏng — bỏ qua.", key)
            return {}
        if not isinstance(loaded, dict):
            return {}
        return {str(name): str(value) for name, value in loaded.items()}

    def _save_two_voice_settings(self) -> None:
        """Lưu giọng nam/nữ (theo engine), giới vai chưa rõ, giới tự chọn của từng vai."""
        # Bảng đang hiện giới mặc định theo lựa chọn "vai chưa rõ giới" ĐÃ LƯU: nếu người
        # dùng vừa đổi ô đó thì các vai chưa rõ giới chưa kịp đổi theo — so với giá trị cũ
        # để không ghi nhầm chúng thành lựa chọn riêng.
        shown_unknown = self._saved_unknown_side()
        engine_key, _catalog = self._catalog_provider()
        male, female = self._panel.gender_voices()
        if male or female:
            self._settings.setValue(
                f"{SETTINGS_GENDER_VOICES_PREFIX}{engine_key}",
                json.dumps({VOICE_MALE: male, VOICE_FEMALE: female}, ensure_ascii=False),
            )
        unknown_side = self._panel.unknown_side()
        self._settings.setValue(SETTINGS_UNKNOWN_SIDE_KEY, unknown_side)
        cast = {member.name: member for member in self._view_model.dubbing_cast}
        # Chỉ nhớ vai người dùng ĐỔI khác giới mặc định; vai về lại mặc định thì xoá.
        merged = {
            name: side for name, side in self._saved_role_sides().items()
            if name not in cast
        }
        for name, side in self._panel.sides().items():
            member = cast.get(name)
            if member is not None and side != (member.gender or shown_unknown):
                merged[name] = side
        kept = dict(list(merged.items())[-MAX_SAVED_ROLES:])
        self._settings.setValue(
            SETTINGS_ROLE_SIDES_KEY, json.dumps(kept, ensure_ascii=False)
        )

    def _on_mode_changed(self, mode: str) -> None:
        self._settings.setValue(SETTINGS_MODE_KEY, mode)
        if self._view_model.dubbing_cast:
            self._show_cast(prefer_saved=True)

    def _on_gender_voices_changed(self) -> None:
        self._save_two_voice_settings()
        # Giới cho vai chưa rõ có thể vừa đổi → giao lại các vai đó (giữ lựa chọn riêng).
        if self._view_model.dubbing_cast:
            self._show_cast(prefer_saved=True)

    def _update_status(self) -> None:
        cast = self._view_model.dubbing_cast
        if self._panel.cast_mode() == CAST_MODE_TWO_VOICES:
            sides = list(self._panel.sides().values())
            status = (
                self._translate("tts.dub_status_two")
                .replace("{cast}", str(len(cast)))
                .replace("{voiced}", str(sum(member.line_count for member in cast)))
                .replace("{lines}", str(len(self._view_model.source_events)))
                .replace("{male}", str(sides.count(VOICE_MALE)))
                .replace("{female}", str(sides.count(VOICE_FEMALE)))
            )
        else:
            distinct = {voice for voice in self._panel.casting().values() if voice}
            status = (
                self._translate("tts.dub_status_loaded")
                .replace("{cast}", str(len(cast)))
                .replace("{voiced}", str(sum(member.line_count for member in cast)))
                .replace("{lines}", str(len(self._view_model.source_events)))
                .replace("{voices}", str(len(distinct)))
            )
            if self._saved_overrides:
                status += " " + self._translate("tts.dub_status_saved").replace(
                    "{count}", str(self._saved_overrides)
                )
        overrides = len(self._view_model.speaker_overrides)
        if overrides:
            status += " " + self._translate("tts.dub_status_line_edits").replace(
                "{count}", str(overrides)
            )
        status += self._review_status()
        self._panel.set_status(status)
        self.line_speakers_changed.emit()

    def _review_status(self) -> str:
        """[v3.23.599] Phần trạng thái soát người nói: dòng chưa rõ, dòng nhãn chắc sai,
        vai chưa rõ giới tính."""
        issues = list(self._view_model.speaker_issues().values())
        unknown = sum(1 for issue in issues if issue.reason == REVIEW_UNKNOWN)
        voice_only = sum(1 for issue in issues if issue.reason == REVIEW_VOICE_ONLY)
        suspect = len(issues) - unknown - voice_only
        parts: list[str] = []
        if unknown:
            parts.append(
                self._translate("tts.dub_status_unknown").replace("{count}", str(unknown))
            )
        if voice_only:
            parts.append(
                self._translate("tts.dub_status_voice_only")
                .replace("{count}", str(voice_only))
            )
        if suspect:
            parts.append(
                self._translate("tts.dub_status_suspect").replace("{count}", str(suspect))
            )
        roles = self.unknown_gender_roles()
        if roles:
            names = ", ".join(member.display_name for member in roles[:4])
            if len(roles) > 4:
                names += ", …"
            parts.append(
                self._translate("tts.dub_status_no_gender")
                .replace("{count}", str(len(roles)))
                .replace("{lines}", str(sum(member.line_count for member in roles)))
                .replace("{names}", names)
            )
        return f" {' '.join(parts)}" if parts else ""

    def _settings_key(self, engine_key: str) -> str:
        return f"{SETTINGS_CASTING_PREFIX}{engine_key}"

    def _saved_casting(self, engine_key: str) -> dict[str, str]:
        raw = self._settings.value(self._settings_key(engine_key), "", type=str)
        try:
            loaded = json.loads(raw) if raw else {}
        except ValueError:
            logger.warning("Bảng phân vai đã lưu của %s hỏng — bỏ qua.", engine_key)
            return {}
        if not isinstance(loaded, dict):
            return {}
        return {str(name): str(voice) for name, voice in loaded.items()}

    def _save_casting(self) -> None:
        if self._panel.cast_mode() == CAST_MODE_TWO_VOICES:
            self._save_two_voice_settings()
            self._update_status()
            return
        engine_key, _catalog = self._catalog_provider()
        casting = self._panel.casting()
        if not casting:
            return
        # Người dùng vừa tự chọn (hoặc vừa phân vai lại): không còn là "giọng cũ ghi đè".
        self._saved_overrides = 0
        # [v3.23.594] Gộp với lựa chọn đã lưu thay vì ghi đè: bản 593 chỉ giữ vai của phim
        # đang mở, nên chỉnh giọng phim B làm mất lựa chọn của phim A (và của các tập khác
        # cùng bộ, vốn chung tên vai). Vai vừa chỉnh dời xuống cuối; giữ tối đa
        # MAX_SAVED_ROLES vai gần nhất để QSettings không phình mãi.
        merged = {
            name: voice for name, voice in self._saved_casting(engine_key).items()
            if name not in casting
        }
        merged.update(casting)
        kept = dict(list(merged.items())[-MAX_SAVED_ROLES:])
        self._settings.setValue(
            self._settings_key(engine_key), json.dumps(kept, ensure_ascii=False)
        )
        self._update_status()

    def _save_enabled(self, enabled: bool) -> None:
        self._settings.setValue(SETTINGS_ENABLED_KEY, enabled)


__all__ = [
    "EDITED_MARK",
    "MAX_SAVED_ROLES",
    "REVIEW_MARK",
    "SETTINGS_CASTING_PREFIX",
    "SETTINGS_ENABLED_KEY",
    "SETTINGS_GENDER_VOICES_PREFIX",
    "SETTINGS_MODE_KEY",
    "SETTINGS_ROLE_SIDES_KEY",
    "SETTINGS_UNKNOWN_SIDE_KEY",
    "SpeakerCell",
    "TTSDubbingCoordinator",
    "short_voice_label",
]
