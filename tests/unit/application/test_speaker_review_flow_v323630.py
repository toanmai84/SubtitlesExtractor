"""[v3.23.630] Duyệt nhanh người nói: hàng đợi, gợi ý, đoạn liền kề, phiên (hàm thuần).

Dữ liệu: cảnh thật dòng 490-499 phim "灭门惨案II" (lượt dịch 10/10/2026) — dòng 495 mang nhãn
vợ (秦可秀) trong khi câu gốc gọi chính cô là "阿秀": người nói là chồng (甄劍聲).
"""

from __future__ import annotations

from subtitles_extractor.application.services.speaker_review import (
    REVIEW_GENDER,
    REVIEW_SELF_CALL,
    REVIEW_UNKNOWN,
    REVIEW_VOICE_ONLY,
    SpeakerIssue,
)
from subtitles_extractor.application.services.speaker_review_flow import (
    ReviewLine,
    candidate_speakers,
    review_queue,
    same_speaker_run,
)
from subtitles_extractor.application.services.speaker_review_session import (
    SpeakerReviewSession,
)

WIFE, HUSBAND, DOCTOR = "秦可秀", "甄劍聲", "醫生"
#: (bắt đầu, kết thúc, người nói, người nghe)
SCENE = [
    (1897.6, 1898.5, HUSBAND, WIFE),
    (1899.2, 1900.0, WIFE, HUSBAND),
    (2024.0, 2025.2, HUSBAND, WIFE),
    (2026.7, 2028.6, HUSBAND, WIFE),
    (2028.7, 2032.0, WIFE, HUSBAND),
    (2033.4, 2037.4, WIFE, HUSBAND),     # sai: phải là chồng
    (2038.7, 2040.2, HUSBAND, WIFE),
    (2050.2, 2052.2, WIFE, HUSBAND),
    (2059.8, 2062.9, HUSBAND, WIFE),
    (2063.4, 2064.2, HUSBAND, WIFE),
]
P495 = 5


def _lines(scene=SCENE) -> list[ReviewLine]:
    return [
        ReviewLine(i, s, e, f"Dịch {i}", f"Gốc {i}", who, to)
        for i, (s, e, who, to) in enumerate(scene)
    ]


def _times(scene=SCENE):
    return [r[0] for r in scene], [r[1] for r in scene]


class TestQueue:
    def test_strongest_evidence_first_then_film_order(self) -> None:
        issues = {
            9: SpeakerIssue(REVIEW_UNKNOWN),
            2: SpeakerIssue(REVIEW_VOICE_ONLY, "Giọng nam"),
            7: SpeakerIssue(REVIEW_SELF_CALL, WIFE, "阿秀"),
            1: SpeakerIssue(REVIEW_UNKNOWN),
            4: SpeakerIssue(REVIEW_GENDER, "female", "我嫁"),
            3: SpeakerIssue(REVIEW_SELF_CALL, WIFE, "阿秀"),
        }
        assert [item.position for item in review_queue(issues)] == [3, 7, 4, 2, 1, 9]

    def test_empty(self) -> None:
        assert review_queue({}) == []


class TestRun:
    def test_run_follows_same_speaker_within_gap(self) -> None:
        speakers = [r[2] for r in SCENE]
        starts, ends = _times()
        assert same_speaker_run(speakers, starts, ends, 2) == [2, 3]
        assert same_speaker_run(speakers, starts, ends, 9) == [8, 9]
        assert same_speaker_run(speakers, starts, ends, 0) == [0]

    def test_long_silence_ends_the_run(self) -> None:
        speakers = [HUSBAND, HUSBAND, HUSBAND]
        starts, ends = [0.0, 1.0, 20.0], [0.8, 1.8, 21.0]
        assert same_speaker_run(speakers, starts, ends, 0) == [0, 1]
        assert same_speaker_run(speakers, starts, ends, 2) == [2]

    def test_unknown_speaker_is_alone_and_bad_positions_are_safe(self) -> None:
        assert same_speaker_run(["", "", ""], [0, 1, 2], [0.5, 1.5, 2.5], 1) == [1]
        assert same_speaker_run(["a"], [0.0], [1.0], 5) == []
        assert same_speaker_run([], [], [], 0) == []


class TestCandidates:
    def test_scene_candidates_are_the_two_people_talking(self) -> None:
        speakers = [r[2] for r in SCENE]
        got = candidate_speakers(speakers, P495, addressee=HUSBAND)
        assert set(got) == {WIFE, HUSBAND} and len(got) == 2

    def test_addressee_and_recent_pick_boost_people_not_nearby(self) -> None:
        speakers = [WIFE, HUSBAND, WIFE, HUSBAND, WIFE]
        got = candidate_speakers(speakers, 2, recent=[DOCTOR], addressee="")
        assert got[:2] == [WIFE, HUSBAND] or got[:2] == [HUSBAND, WIFE]
        assert DOCTOR in got
        assert DOCTOR in candidate_speakers(speakers, 2, addressee=DOCTOR)[:3]
        assert DOCTOR not in candidate_speakers(speakers, 2)

    def test_no_duplicates_no_blank_and_limit(self) -> None:
        speakers = [f"v{i}" for i in range(30)]
        got = candidate_speakers(speakers, 15, limit=9)
        assert len(got) == 9 and len(set(got)) == 9 and "" not in got
        assert candidate_speakers(["", ""], 0) == []
        assert candidate_speakers([], 0) == []


class TestSession:
    def _session(self, queue=(P495,)) -> SpeakerReviewSession:
        return SpeakerReviewSession(_lines(), queue)

    def test_choose_changes_only_the_line_and_moves_on(self) -> None:
        session = self._session((P495, 7))
        assert session.position == P495 and session.progress == (1, 2)
        assert session.choose(HUSBAND) == [P495]
        assert session.position == 7 and session.progress == (2, 2)
        assert session.changes() == {P495: HUSBAND}
        assert session.speakers[P495] == HUSBAND

    def test_keep_leaves_no_change(self) -> None:
        session = self._session((P495, 7))
        assert session.next()
        assert not session.next() and session.finished and session.position is None
        assert session.changes() == {} and session.reviewed_count == 2

    def test_choosing_the_same_speaker_is_not_a_change(self) -> None:
        session = self._session((P495,))
        assert session.choose(WIFE) == []
        assert session.changes() == {}

    def test_shared_voice_is_an_empty_speaker(self) -> None:
        session = self._session((P495,))
        session.choose("")
        assert session.changes() == {P495: ""}

    def test_whole_run_changes_adjacent_lines_of_the_same_speaker(self) -> None:
        session = SpeakerReviewSession(_lines(), (2,))
        assert session.run() == [2, 3]
        assert session.choose(DOCTOR, whole_run=True) == [2, 3]
        assert session.changes() == {2: DOCTOR, 3: DOCTOR}

    def test_undo_restores_speakers_cursor_and_recent_picks(self) -> None:
        session = SpeakerReviewSession(_lines(), (2, P495))
        session.choose(DOCTOR, whole_run=True)
        assert session.position == P495 and session.candidates()[0] != "x"
        assert session.can_undo and session.undo()
        assert session.position == 2 and session.changes() == {}
        assert DOCTOR not in session.candidates()
        assert not session.undo()

    def test_recent_pick_is_offered_next(self) -> None:
        session = SpeakerReviewSession(_lines(), (P495, 7))
        session.choose(DOCTOR)
        assert DOCTOR in session.candidates()

    def test_all_lines_scope_and_start_at(self) -> None:
        session = self._session((P495,))
        session.set_scope(True)
        assert session.total == len(SCENE) and session.position == P495
        session.start_at(8)
        assert session.position == 8
        session.set_scope(False)             # dòng 8 không nghi sai
        assert session.finished
        session.start_at(P495)
        assert session.position == P495 and not session.all_lines

    def test_start_at_a_line_outside_the_queue_switches_to_all_lines(self) -> None:
        session = self._session((P495,))
        session.start_at(1)
        assert session.all_lines and session.position == 1

    def test_previous_and_context(self) -> None:
        session = self._session((2, P495))
        assert not session.previous()
        session.next()
        assert session.previous() and session.position == 2
        context = session.context(2)
        assert [line.position for line in context] == [0, 1, 2, 3, 4]
        session.choose(DOCTOR)
        assert session.line(2).speaker == DOCTOR
        assert [line.position for line in session.context(1)] == [4, 5, 6]   # đã sang dòng 5

    def test_empty_queue_is_finished(self) -> None:
        session = SpeakerReviewSession(_lines(), ())
        assert session.finished and session.candidates() == [] and session.run() == []
        assert session.choose(WIFE) == [] and not session.next()
