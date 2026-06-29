"""Unit test cho các hàm thuần (không cần model)."""

from vsf_diarization.core.utils import (
    merge_segments, merge_text_segments, speaker_at, compute_asr_metrics,
)


def test_merge_segments_drops_short_and_merges_gap():
    segs = [
        {"speaker": "A", "start": 0.0, "end": 0.1},     # < min_dur 0.3 → bỏ
        {"speaker": "A", "start": 1.0, "end": 2.0},
        {"speaker": "A", "start": 2.3, "end": 3.0},     # gap 0.3 ≤ 0.5 → gộp
        {"speaker": "B", "start": 3.1, "end": 4.0},
    ]
    out = merge_segments(segs)
    assert [(s["speaker"], s["start"], s["end"]) for s in out] == [
        ("A", 1.0, 3.0), ("B", 3.1, 4.0)]


def test_merge_text_segments_joins_same_speaker():
    segs = [
        {"speaker": "A", "start": 0.0, "end": 1.0, "text": "xin"},
        {"speaker": "A", "start": 1.5, "end": 2.0, "text": "chào"},
        {"speaker": "B", "start": 2.5, "end": 3.0, "text": "vâng"},
    ]
    out = merge_text_segments(segs)
    assert len(out) == 2
    assert out[0]["text"] == "xin chào"
    assert out[1]["text"] == "vâng"


def test_speaker_at_picks_active_then_nearest():
    turns = [{"speaker": "A", "start": 0.0, "end": 2.0},
             {"speaker": "B", "start": 3.0, "end": 5.0}]
    assert speaker_at(turns, 1.0) == "A"          # bên trong A
    assert speaker_at(turns, 2.6) == "B"          # trong gap → gần B hơn
    assert speaker_at([], 1.0) == "UNKNOWN"


def test_asr_metrics_perfect_and_errors():
    gt = [{"start": 0, "end": 1, "text": "xin chào bác sĩ"}]
    assert compute_asr_metrics(gt, gt)["WER_%"] == 0.0
    hyp = [{"start": 0, "end": 1, "text": "xin chào bác"}]   # mất 1/4 từ
    assert compute_asr_metrics(gt, hyp)["WER_%"] == 25.0
    assert compute_asr_metrics(gt, [])["WER_%"] is None       # rỗng → None
