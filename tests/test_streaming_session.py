"""Test StreamingSession (real-time engine) với FakePipeline — không cần model."""

import numpy as np

from vsf_diarization.core.streaming_session import StreamingSession
from tests.fakes import FakePipeline, SAMPLE_RATE


def _run(timeline, total_s, *, block_s=0.5, chunk_s=6.0, step_s=1.0, num_speakers=None):
    """Đẩy audio im lặng theo block để bắt chước mic; trả về các turn (no-ASR)."""
    pipe = FakePipeline(timeline, step_s=step_s)
    sess = StreamingSession(pipe, whisper=None, chunk_s=chunk_s, step_s=step_s,
                            num_speakers=num_speakers, min_asr=0.3)
    block = int(block_s * SAMPLE_RATE)
    audio = np.zeros(int(total_s * SAMPLE_RATE), dtype=np.float32)
    turns = []
    for i in range(0, len(audio), block):
        turns += sess.feed(audio[i:i + block])
    turns += sess.finalize()
    return turns


def test_two_speaker_alternation_recovered():
    # A: 0–10s, B: 10–20s, A: 20–30s
    timeline = [(0, 10, "A"), (10, 20, "B"), (20, 30, "A")]
    turns = _run(timeline, 30.0)
    speakers = [t["speaker"] for t in turns]
    assert speakers == ["A", "B", "A"], speakers
    # biên turn khớp timeline trong khoảng dung sai 1 cửa sổ-step
    assert abs(turns[0]["end"] - 10.0) <= 1.5
    assert abs(turns[1]["start"] - 10.0) <= 1.5


def test_feed_is_incremental_not_all_at_end():
    """Turn đầu phải được chốt TRƯỚC khi đẩy hết audio (đặc tính real-time)."""
    timeline = [(0, 8, "A"), (8, 24, "B")]
    pipe = FakePipeline(timeline, step_s=1.0)
    sess = StreamingSession(pipe, whisper=None, chunk_s=6.0, step_s=1.0, min_asr=0.3)
    block = int(0.5 * SAMPLE_RATE)
    audio = np.zeros(int(24 * SAMPLE_RATE), dtype=np.float32)

    emitted_before_end = 0
    n_blocks = len(audio) // block
    for i in range(n_blocks):
        out = sess.feed(audio[i * block:(i + 1) * block])
        if i < n_blocks - 1:           # trước block cuối
            emitted_before_end += len(out)
    assert emitted_before_end >= 1, "không có turn nào được chốt giữa chừng"


def test_feed_equals_single_shot():
    """Đẩy theo block nhỏ phải cho cùng kết quả với đẩy nguyên mảng."""
    timeline = [(0, 7, "A"), (7, 14, "B"), (14, 21, "A")]
    chunked = _run(timeline, 21.0, block_s=0.5)

    pipe = FakePipeline(timeline, step_s=1.0)
    sess = StreamingSession(pipe, whisper=None, chunk_s=6.0, step_s=1.0, min_asr=0.3)
    whole = sess.feed(np.zeros(int(21 * SAMPLE_RATE), dtype=np.float32)) + sess.finalize()

    norm = lambda ts: [(t["speaker"], round(t["start"]), round(t["end"])) for t in ts]
    assert norm(chunked) == norm(whole)
