"""
Real-time streaming session — feed live audio chunks, get finalized speaker turns.

`stream_online()` (diarize_online.py) processes a *fixed* numpy array. This class
generalizes the same sliding-window + 50ms majority-vote algorithm to *incremental*
input: push audio as it arrives (mic / WebSocket) and receive turns the moment they
become stable. Same accuracy as stream_online; only the input delivery differs.

Latency ≈ chunk_s × RTF ≈ 0.9s at the default chunk_s=6s.

Usage:
    session = StreamingSession(pipeline, whisper, language="vi", num_speakers=2)
    for block in audio_blocks:                 # e.g. 0.5s mic blocks
        for turn in session.feed(block):       # turns finalized so far
            print(turn["speaker"], turn["text"])
    for turn in session.finalize():            # flush the tail
        print(turn["speaker"], turn["text"])
"""

from collections import defaultdict

import numpy as np

from vsf_diarization.core.utils import SAMPLE_RATE, transcribe_turn
from vsf_diarization.core.diarize_online import (
    SpeakerRegistry, _frames_to_segs, FRAME_DUR,
)


class StreamingSession:
    def __init__(self, pipeline, whisper=None, *, chunk_s: float = 6.0,
                step_s: float = 1.0, threshold: float = 0.70,
                num_speakers: int | None = None, language: str | None = "vi",
                min_asr: float = 1.0):
        self.pipeline = pipeline
        self.whisper = whisper
        self.chunk = int(chunk_s * SAMPLE_RATE)
        self.step = int(step_s * SAMPLE_RATE)
        self.kwargs = {"num_speakers": num_speakers} if num_speakers else {}
        self.language = language
        self.min_asr = min_asr

        self.registry = SpeakerRegistry(threshold=threshold)
        self.buf = np.empty(0, dtype=np.float32)
        self.votes: dict[int, dict[str, int]] = defaultdict(dict)
        self.pos = 0          # sample index of next window start
        self.emit_fi = 0      # next frame index not yet finalized
        self.cur: dict | None = None   # turn being accumulated

    # ── Public API ─────────────────────────────────────────────────────────────

    def feed(self, audio: np.ndarray) -> list[dict]:
        """Append audio; return turns that became final (each {speaker,start,end,text})."""
        if audio is not None and len(audio):
            self.buf = np.concatenate([self.buf, np.asarray(audio, dtype=np.float32)])
        done: list[dict] = []
        while self.pos + self.chunk <= len(self.buf):
            self._cast_window(self.pos, self.pos + self.chunk)
            self.pos += self.step
            done += self._finalize(self.pos)            # frames before pos are stable
        return done

    def finalize(self) -> list[dict]:
        """Process the trailing partial window and flush the open turn."""
        done: list[dict] = []
        if len(self.buf) - self.pos >= int(0.5 * SAMPLE_RATE):
            self._cast_window(self.pos, len(self.buf))
            self.pos = len(self.buf)
        done += self._finalize_to_frame(int((len(self.buf) / SAMPLE_RATE) / FRAME_DUR) + 1)
        last = self._flush(self.cur)
        self.cur = None
        if last:
            done.append(last)
        return done

    # ── Internals ──────────────────────────────────────────────────────────────

    def _cast_window(self, start: int, end: int) -> None:
        import torch
        chunk = self.buf[start:end]
        if len(chunk) < int(0.5 * SAMPLE_RATE):
            return
        offset_s = start / SAMPLE_RATE
        waveform = torch.from_numpy(chunk[np.newaxis, :])
        try:
            result = self.pipeline({"waveform": waveform, "sample_rate": SAMPLE_RATE},
                                   **self.kwargs)
        except Exception:
            return
        label_map = self.registry.resolve(result)
        annotation = (result.speaker_diarization if hasattr(result, "speaker_diarization") else result)
        for seg, _, spk in annotation.itertracks(yield_label=True):
            g = label_map.get(spk, spk)
            fi0 = int((seg.start + offset_s) / FRAME_DUR)
            fi1 = int((seg.end + offset_s) / FRAME_DUR)
            for fi in range(fi0, fi1 + 1):
                self.votes[fi][g] = self.votes[fi].get(g, 0) + 1

    def _finalize(self, sample_boundary: int) -> list[dict]:
        return self._finalize_to_frame(int((sample_boundary / SAMPLE_RATE) / FRAME_DUR))

    def _finalize_to_frame(self, new_emit_fi: int) -> list[dict]:
        if new_emit_fi <= self.emit_fi:
            return []
        frames = []
        for fi in range(self.emit_fi, new_emit_fi):
            v = self.votes.get(fi)
            frames.append((fi * FRAME_DUR, max(v, key=lambda k: v[k]) if v else None))
            self.votes.pop(fi, None)                     # free finalized frames
        self.emit_fi = new_emit_fi
        return self._ingest(_frames_to_segs(frames))

    def _ingest(self, segs: list[dict]) -> list[dict]:
        done: list[dict] = []
        for seg in segs:
            if (self.cur and seg["speaker"] == self.cur["speaker"]
                    and seg["start"] <= self.cur["end"] + 0.6):
                self.cur["end"] = seg["end"]             # same speaker → extend turn
            else:
                t = self._flush(self.cur)                # speaker changed → close turn
                if t:
                    done.append(t)
                self.cur = {"speaker": seg["speaker"],
                            "start": seg["start"], "end": seg["end"]}
        return done

    def _flush(self, turn: dict | None) -> dict | None:
        if not turn:
            return None
        dur = turn["end"] - turn["start"]
        if dur < 0.3:                                    # nhiễu biên
            return None
        if self.whisper is None or dur < self.min_asr:
            text = "..."                                 # ngắn → bỏ ASR (tránh hallucinate)
        else:
            text = transcribe_turn(self.whisper, self.buf, turn["start"], turn["end"], self.language) or "..."
        return {"speaker": turn["speaker"], "start": round(turn["start"], 3),
                "end": round(turn["end"], 3), "text": text}
