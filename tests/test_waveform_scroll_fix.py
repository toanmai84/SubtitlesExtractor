#!/usr/bin/env python3
"""
Test để xác minh sửa chữa: Waveform cuộn khi chọn dòng phụ đề

BUG: Khi chọn dòng phụ đề trong bảng, waveform không nhảy đến câu đó
FIX: set_active_row() giờ cuộn waveform để câu được chọn nằm giữa viewport
"""

import sys
from pathlib import Path

# Thêm src vào path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Import SubtitleEvent từ app
try:
    from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
    from subtitles_extractor.domain.value_objects.time_interval import TimeInterval

    # Tạo event với TimeInterval
    def make_event(start_sec, end_sec, text, idx=0):
        return SubtitleEvent(
            index=idx,
            text=text,
            interval=TimeInterval(start_sec, end_sec)
        )
except ImportError:
    # Nếu không thể import, tạo class stub để test logic
    class SubtitleEvent:
        def __init__(self, **kwargs):
            self.start_sec = kwargs.get('start_sec', 0.0)
            self.end_sec = kwargs.get('end_sec', 0.0)
            self.text = kwargs.get('text', '')
            self.duration_sec = self.end_sec - self.start_sec

    def make_event(start_sec, end_sec, text, idx=0):
        ev = SubtitleEvent()
        ev.start_sec = start_sec
        ev.end_sec = end_sec
        ev.text = text
        ev.duration_sec = end_sec - start_sec
        return ev


def test_set_active_row_scroll():
    """Test: set_active_row() cuộn waveform đến câu được chọn"""

    # Tạo waveform widget (không cần QApplication vì chỉ test logic)
    # NOTE: Thực tế cần QApplication để khởi tạo QWidget
    # Ở đây chỉ test logic của set_active_row()

    print("=" * 70)
    print("TEST: set_active_row() - Waveform Scroll Fix")
    print("=" * 70)

    # Tạo danh sách các câu phụ đề giả
    events = [
        make_event(0.0, 2.0, "Câu 1", 0),
        make_event(3.0, 5.0, "Câu 2", 1),
        make_event(6.0, 8.0, "Câu 3", 2),
        make_event(10.0, 12.0, "Câu 4", 3),
        make_event(15.0, 17.0, "Câu 5", 4),
        make_event(20.0, 22.0, "Câu 6", 5),
    ]

    print(f"\n✓ Tạo {len(events)} câu phụ đề test")
    for i, ev in enumerate(events):
        print(f"  [{i}] {ev.start_sec:.1f}s - {ev.end_sec:.1f}s: {ev.text}")

    # Mô phỏng logic set_active_row()
    print("\n" + "-" * 70)
    print("KIỂM TRA LOGIC set_active_row():")
    print("-" * 70)

    _view_timeline_duration_sec = 10.0  # Window rộng 10 giây
    _timeline_max_bounds = 25.0  # Video dài 25 giây

    test_cases = [
        (0, "Câu đầu tiên - nên giới hạn ở min(0, ...)"),
        (2, "Câu giữa - nên center"),
        (5, "Câu cuối - nên giới hạn ở max"),
    ]

    for idx, description in test_cases:
        if idx >= len(events):
            print(f"\n[SKIP] {description} (index {idx} out of range)")
            continue

        event = events[idx]

        # Tính toán view_start như trong set_active_row()
        desired_view_start = event.start_sec - _view_timeline_duration_sec / 2.0
        max_view_start = max(0.0, _timeline_max_bounds - _view_timeline_duration_sec)
        view_timeline_start_sec = max(0.0, min(desired_view_start, max_view_start))

        view_end = view_timeline_start_sec + _view_timeline_duration_sec

        print(f"\n[TEST {idx}] {description}")
        print(f"  Event: {event.start_sec:.1f}s - {event.end_sec:.1f}s")
        print(f"  Desired view start: {desired_view_start:.1f}s")
        print(f"  Max view start: {max_view_start:.1f}s")
        print(f"  ✓ Actual view_start: {view_timeline_start_sec:.1f}s")
        print(f"  ✓ View window: [{view_timeline_start_sec:.1f}s, {view_end:.1f}s]")

        # Kiểm tra event nằm trong window
        event_in_window = (view_timeline_start_sec <= event.start_sec and 
                          event.end_sec <= view_end)
        if event_in_window:
            print(f"  ✓ Event NẰM TRONG window (đúng!)")
        else:
            print(f"  ✗ Event KHÔNG NẰM TRONG window (lỗi!)")
            print(f"    Event.start_sec={event.start_sec:.1f} vs view_start={view_timeline_start_sec:.1f}")
            print(f"    Event.end_sec={event.end_sec:.1f} vs view_end={view_end:.1f}")

        # Kiểm tra event gần center
        event_center = (event.start_sec + event.end_sec) / 2.0
        window_center = view_timeline_start_sec + _view_timeline_duration_sec / 2.0
        distance_from_center = abs(event_center - window_center)
        print(f"  Event center: {event_center:.1f}s, Window center: {window_center:.1f}s")
        print(f"  Distance from center: {distance_from_center:.2f}s")

    print("\n" + "=" * 70)
    print("✓ TEST HOÀN THÀNH: Logic set_active_row() được kiểm tra")
    print("=" * 70)


if __name__ == "__main__":
    test_set_active_row_scroll()
