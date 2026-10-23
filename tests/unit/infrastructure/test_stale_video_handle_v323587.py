"""[v3.23.587] Sau khi xoay API key, đoạn phân tích kế tiếp không gửi file của key cũ.

Nhật ký thật v3.23.586 (phim "灭门惨案II", phân tích 17 đoạn bằng gemini-3.8-flash): key 1
hết hạn mức ngày ở đoạn 12, ứng dụng xoay sang key 2 và tải lại đúng đoạn 12. Năm đoạn còn
lại (13 tới 17) vẫn mang handle video resolve bằng key 1 từ lúc bắt đầu, nên lời gọi ĐẦU
của từng đoạn nhận "403 PERMISSION_DENIED … permission to access the File" rồi mới tải
lại: 5 request của key 2 mất trắng trên hạn mức 20/ngày. Trên 12 nhật ký: 87 lượt như
vậy, 86 lượt nằm ở khâu phân tích.

Các lượt 403 đó server VẪN TÍNH vào hạn mức ngày nhưng ứng dụng không đếm. Trên 6 nhật ký
có 10 lần một key vừa xoay (còn đủ 20 request) bị server chặn: cộng lượt thành công + 503
+ 403 thì ra 19-21 lượt ở cả 10 lần; bỏ 403 ra chỉ còn 10-19.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from subtitles_extractor.domain.ports.subtitle_translator_port import TranslationLine
from subtitles_extractor.infrastructure.translation.gemini_quota_manager import (
    GeminiQuotaManager,
)
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
)
from tests.unit.infrastructure.gemini_call_harness import (
    REAL_DAILY_429,
    SCARCE_MODEL,
    FakeClock,
    build_translator,
    call_gemini,
    confirmed_scarce_manager,
    daily_requests_used,
    install_fake_clock,
)

SEGMENT_SEC = 340.0

#: Chép nguyên từ nhật ký (chỉ thay mã file).
REAL_FILE_403 = (
    "403 PERMISSION_DENIED. {{'error': {{'code': 403, 'message': 'You do not have "
    "permission to access the File {name} or it may not exist.', "
    "'status': 'PERMISSION_DENIED'}}}}"
)


class _Clip:
    """Một đoạn video đã tải lên Gemini."""

    def __init__(self, remote_name: str, index: int) -> None:
        self.remote_name = remote_name
        self.chunk_index = index
        self.start_sec = index * SEGMENT_SEC
        self.end_sec = (index + 1) * SEGMENT_SEC
        self.state = "ACTIVE"


class _FileBoundGemini:
    """Gemini giả giữ đúng ba luật của server thật.

    * File chỉ dùng được bằng key (dự án) đã tải nó lên; key khác nhận 403.
    * Lượt gọi bị 403 vì file VẪN bị tính vào hạn mức ngày của key đang gọi.
    * Vượt hạn mức ngày thì trả 429 (lượt này không bị tính).
    """

    def __init__(self, daily_left: dict[str, int], projects: dict[str, str]) -> None:
        self.daily_left = dict(daily_left)
        self.projects = projects
        self.current_key = next(iter(daily_left))
        self.file_project: dict[str, str] = {}
        self.counted: Counter[str] = Counter()
        self.file_403 = 0
        self.uploads: list[tuple[str, int]] = []
        self.models = SimpleNamespace(generate_content=self._generate)
        self.files = SimpleNamespace(get=self._get_file)

    def upload(self, key: str, index: int) -> _Clip:
        name = f"files/{key.lower()}_{index}_{len(self.uploads)}"
        self.file_project[name] = self.projects[key]
        self.uploads.append((key, index))
        return _Clip(name, index)

    def _owns(self, name: str) -> bool:
        return self.file_project.get(name) == self.projects[self.current_key]

    def _get_file(self, name: str) -> str:
        if not self._owns(name):
            raise RuntimeError(REAL_FILE_403.format(name=name.split("/")[-1]))
        return name

    def _generate(self, model: str, contents: list, config: object) -> object:
        key = self.current_key
        if self.daily_left[key] <= 0:
            raise RuntimeError(REAL_DAILY_429)
        self.daily_left[key] -= 1
        self.counted[key] += 1
        for part in contents[:-1]:
            if not self._owns(part):
                self.file_403 += 1
                raise RuntimeError(REAL_FILE_403.format(name=part.split("/")[-1]))
        answer = {
            "source_lang": "zh",
            "characters": "龍傳人 (Long Truyền Nhân) — Nam — trưởng đồn",
            "overview": "Diễn biến.",
            "glossary": "派出所 = đồn công an",
            "segment_summary": "Diễn biến đoạn.",
        }
        return SimpleNamespace(text=json.dumps(answer), usage_metadata=None)


def _wire(
    monkeypatch: pytest.MonkeyPatch,
    clock: FakeClock,
    manager: GeminiQuotaManager,
    server: _FileBoundGemini,
) -> GeminiSubtitleTranslator:
    """Adapter thật hai key nối với server giả; file gắn theo key đang dùng."""
    translator, _slept = build_translator(
        monkeypatch, clock, manager, server.models, keys="KEY_A\nKEY_B"
    )

    def _client() -> _FileBoundGemini:
        server.current_key = translator._api_key
        return server

    monkeypatch.setattr(translator, "_get_client", _client)
    monkeypatch.setattr(
        translator, "_apply_video_fps", lambda handles: list(handles or [])
    )
    translator.set_video_reupload_callback(
        lambda api_key, ranges: [
            server.upload(api_key, round(start / SEGMENT_SEC)) for start, _end in ranges
        ]
    )
    return translator


def _analyze(translator: GeminiSubtitleTranslator, refs: list[_Clip]) -> object:
    lines = [
        TranslationLine(index=i, start_ms=0, end_ms=1, text=f"臺詞{i}")
        for i in range(1, 6)
    ]
    return translator.analyze_global_context(
        lines, "Vietnamese", video_refs=refs, model_name=SCARCE_MODEL
    )


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    return install_fake_clock(monkeypatch)


SEPARATE_PROJECTS = {"KEY_A": "project_a", "KEY_B": "project_b"}


class TestAnalysisAfterKeyRotation:
    def test_later_segments_do_not_send_the_old_keys_file(
        self, monkeypatch: pytest.MonkeyPatch, clock: FakeClock, tmp_path: Path
    ) -> None:
        """Ca thật thu nhỏ: key 1 cạn ở đoạn 3/6 → ba đoạn sau từng mất một request."""
        server = _FileBoundGemini({"KEY_A": 2, "KEY_B": 20}, SEPARATE_PROJECTS)
        manager = confirmed_scarce_manager(tmp_path)
        translator = _wire(monkeypatch, clock, manager, server)
        refs = [server.upload("KEY_A", index) for index in range(6)]

        result = _analyze(translator, refs)

        assert result.missing_segments == ()
        assert server.file_403 == 0  # trước đây 3: đoạn 4, 5, 6
        assert server.counted == {"KEY_A": 2, "KEY_B": 4}  # trước đây KEY_B = 7
        # Tải lại đúng bốn đoạn key mới phải gửi, mỗi đoạn một lần.
        assert server.uploads[6:] == [("KEY_B", index) for index in (2, 3, 4, 5)]

    def test_app_counter_matches_the_server(
        self, monkeypatch: pytest.MonkeyPatch, clock: FakeClock, tmp_path: Path
    ) -> None:
        server = _FileBoundGemini({"KEY_A": 2, "KEY_B": 20}, SEPARATE_PROJECTS)
        manager = confirmed_scarce_manager(tmp_path)
        translator = _wire(monkeypatch, clock, manager, server)
        _analyze(translator, [server.upload("KEY_A", index) for index in range(6)])
        used = daily_requests_used(manager, SCARCE_MODEL, "KEY_B")
        assert used == server.counted["KEY_B"]

    def test_refs_are_replaced_for_the_caller(
        self, monkeypatch: pytest.MonkeyPatch, clock: FakeClock, tmp_path: Path
    ) -> None:
        """Ref mới phải tới tay nơi gọi: khâu sau dùng lại, không tải lên lần nữa."""
        server = _FileBoundGemini({"KEY_A": 2, "KEY_B": 20}, SEPARATE_PROJECTS)
        translator = _wire(monkeypatch, clock, confirmed_scarce_manager(tmp_path), server)
        refs = [server.upload("KEY_A", index) for index in range(6)]
        _analyze(translator, refs)
        owners = [server.file_project[ref.remote_name] for ref in refs]
        assert owners == ["project_a"] * 2 + ["project_b"] * 4
        assert [ref.chunk_index for ref in refs] == list(range(6))

    def test_keys_of_one_project_share_files(
        self, monkeypatch: pytest.MonkeyPatch, clock: FakeClock, tmp_path: Path
    ) -> None:
        """Hai key cùng dự án dùng chung file: không tải lại từng đoạn còn lại."""
        shared = {"KEY_A": "project", "KEY_B": "project"}
        server = _FileBoundGemini({"KEY_A": 2, "KEY_B": 20}, shared)
        translator = _wire(monkeypatch, clock, confirmed_scarce_manager(tmp_path), server)
        refs = [server.upload("KEY_A", index) for index in range(6)]
        result = _analyze(translator, refs)
        assert result.missing_segments == ()
        assert server.file_403 == 0
        # Chỉ đoạn đang gửi lúc đổi key được tải lại (chưa biết hai key chung dự án).
        assert server.uploads[6:] == [("KEY_B", 2)]

    def test_no_rotation_means_no_extra_work(
        self, monkeypatch: pytest.MonkeyPatch, clock: FakeClock, tmp_path: Path
    ) -> None:
        server = _FileBoundGemini({"KEY_A": 20, "KEY_B": 20}, SEPARATE_PROJECTS)
        translator = _wire(monkeypatch, clock, confirmed_scarce_manager(tmp_path), server)
        refs = [server.upload("KEY_A", index) for index in range(6)]
        _analyze(translator, refs)
        assert len(server.uploads) == 6
        assert server.counted == {"KEY_A": 6}


class TestFile403IsCounted:
    def test_rejected_file_still_costs_a_request(
        self, monkeypatch: pytest.MonkeyPatch, clock: FakeClock, tmp_path: Path
    ) -> None:
        """Đường dự phòng (403 rồi mới tải lại) vẫn còn: lượt hỏng phải được đếm."""
        server = _FileBoundGemini({"KEY_A": 20, "KEY_B": 20}, SEPARATE_PROJECTS)
        manager = confirmed_scarce_manager(tmp_path)
        translator = _wire(monkeypatch, clock, manager, server)
        foreign = server.upload("KEY_B", 0)  # file của key khác, key hiện tại là KEY_A

        data = call_gemini(
            translator, SCARCE_MODEL,
            video_files=[foreign.remote_name], video_refs=[foreign],
        )

        assert data["source_lang"] == "zh"
        assert server.counted["KEY_A"] == 2  # một lượt 403 + một lượt thành công
        assert daily_requests_used(manager, SCARCE_MODEL) == 2  # trước đây 1
