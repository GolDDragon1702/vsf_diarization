"""
Online (streaming) speaker diarization — diart-style sliding window.

Two modes:
  run_online()    — batch mode, processes full audio, returns all segments at once.
  stream_online() — generator mode, yields segments incrementally as chunks are processed.
                    Latency ≈ chunk_s × RTF ≈ 0.9s (default chunk_s=6s). Use for true streaming output.

Optimised batch parameters (validated against 10 reviewed GT files via sweep_online.py):
    window_s=6, step_s=1, registry_threshold=0.70  → DER 13.81% (default: streaming latency ~0.9s)
    window_s=9, step_s=1, registry_threshold=0.70  → DER 12.65% (best accuracy, latency ~1.35s)
    Optimum phụ thuộc dataset (11 file test01–11); xem REPORT.md mục 6.2.

Streaming parameters (latency vs accuracy trade-off):
    chunk_s=6, step_s=1, registry_threshold=0.70  → matches batch accuracy, ~0.9s latency

Usage:
    # Batch
    from diarize_online import run_online
    turns, elapsed = run_online(data, pipeline, window_s=6, step_s=1, num_speakers=2)

    # Streaming
    from diarize_online import stream_online
    for segments in stream_online(data, pipeline, chunk_s=3.0):
        for seg in segments:
            print(f"[{seg['start']:.1f}s] {seg['speaker']}: ...")
"""

from typing import Generator
import time

import numpy as np
import torch

from vsf_diarization.core.utils import SAMPLE_RATE, merge_segments

FRAME_DUR = 0.05  # 50ms voting resolution


# ── Speaker registry ──────────────────────────────────────────────────────────

class SpeakerRegistry:
    """
    Maps per-window local speaker labels to persistent global IDs using
    speaker embedding cosine similarity.
    """

    def __init__(self, threshold: float = 0.70):
        self.threshold = threshold
        self._embeddings: list[np.ndarray] = []
        self._counts: list[int] = []

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        d = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / d) if d > 0 else 0.0

    def resolve(self, diarization_output) -> dict[str, str]:
        """Return a mapping {local_label → global_SPEAKER_XX} for this window's output."""
        annotation = (diarization_output.speaker_diarization
                    if hasattr(diarization_output, "speaker_diarization")
                    else diarization_output)
        raw_embs = getattr(diarization_output, "speaker_embeddings", None)
        labels = sorted(annotation.labels())
        mapping: dict[str, str] = {}

        for pos, label in enumerate(labels):
            emb = None
            if raw_embs is not None and pos < len(raw_embs):
                e = np.asarray(raw_embs[pos], dtype=float)
                if np.linalg.norm(e) > 0:
                    emb = e

            if emb is None:
                mapping[label] = label
                continue

            best_idx, best_sim = -1, -1.0
            for i, known in enumerate(self._embeddings):
                s = self._cosine(emb, known)
                if s > best_sim:
                    best_sim, best_idx = s, i

            if best_sim >= self.threshold:
                n = self._counts[best_idx]
                self._embeddings[best_idx] = (self._embeddings[best_idx] * n + emb) / (n + 1)
                self._counts[best_idx] += 1
                mapping[label] = f"SPEAKER_{best_idx:02d}"
            else:
                new_id = len(self._embeddings)
                self._embeddings.append(emb.copy())
                self._counts.append(1)
                mapping[label] = f"SPEAKER_{new_id:02d}"

        return mapping


# ── Ghost-speaker cleanup ─────────────────────────────────────────────────────

def _merge_shortest_speaker(segs: list[dict]) -> list[dict]:
    """Merge the speaker with the least total speech into its most-adjacent neighbour."""
    if not segs:
        return segs
    durations: dict[str, float] = {}
    for s in segs:
        durations[s["speaker"]] = durations.get(s["speaker"], 0) + (s["end"] - s["start"])
    if len(durations) <= 1:
        return segs

    ghost = min(durations, key=durations.get)
    real = {spk for spk in durations if spk != ghost}

    adj: dict[str, int] = {}
    for i, seg in enumerate(segs):
        if seg["speaker"] == ghost:
            for j in (i - 1, i + 1):
                if 0 <= j < len(segs) and segs[j]["speaker"] in real:
                    nbr = segs[j]["speaker"]
                    adj[nbr] = adj.get(nbr, 0) + 1
    target = max(adj, key=adj.get) if adj else next(iter(real))

    return [
        {**s, "speaker": target if s["speaker"] == ghost else s["speaker"]}
        for s in segs
    ]


def limit_speakers(segs: list[dict], max_speakers: int) -> list[dict]:
    """Reduce speaker count to max_speakers by repeatedly merging the shortest one."""
    while len({s["speaker"] for s in segs}) > max_speakers:
        segs = _merge_shortest_speaker(segs)
    return segs


# ── Shared helpers ────────────────────────────────────────────────────────────

def _cast_votes(votes, annotation, label_map, offset_s: float, n_frames: int) -> None:
    """Add votes from one window's diarization result into the votes array."""
    for t, _, spk in annotation.itertracks(yield_label=True):
        global_spk = label_map.get(spk, spk)
        fi_start = int((t.start + offset_s) / FRAME_DUR)
        fi_end   = int((t.end   + offset_s) / FRAME_DUR)
        for fi in range(fi_start, min(fi_end + 1, n_frames)):
            votes[fi][global_spk] = votes[fi].get(global_spk, 0) + 1


def _frames_to_segs(frames: list[tuple[float, str | None]]) -> list[dict]:
    """Convert [(time_s, speaker), ...] frame sequence to segment dicts."""
    if not frames:
        return []
    segs: list[dict] = []
    cur_spk, cur_start = frames[0][1], frames[0][0]
    cur_end = cur_start + FRAME_DUR
    for t, spk in frames[1:]:
        if spk == cur_spk:
            cur_end = t + FRAME_DUR
        else:
            if cur_spk is not None:
                segs.append({"speaker": cur_spk,
                             "start": round(cur_start, 3), "end": round(cur_end, 3),
                             "duration": round(cur_end - cur_start, 3)})
            cur_spk, cur_start, cur_end = spk, t, t + FRAME_DUR
    if cur_spk is not None:
        segs.append({"speaker": cur_spk,
                     "start": round(cur_start, 3), "end": round(cur_end, 3),
                     "duration": round(cur_end - cur_start, 3)})
    return segs


# ── Batch mode ────────────────────────────────────────────────────────────────

def run_online(
    data: np.ndarray,
    pipeline,
    window_s: float = 6.0,
    step_s: float = 1.0,
    registry_threshold: float = 0.70,
    num_speakers: int | None = None,
) -> tuple[list[dict], float]:
    """
    Batch sliding-window diarization. Processes all audio and returns all segments.
    Best accuracy. Use for offline evaluation / comparison.

    Returns: (segments, elapsed_seconds)
    """
    window = int(window_s * SAMPLE_RATE)
    step   = int(step_s   * SAMPLE_RATE)
    n = len(data)
    duration = n / SAMPLE_RATE
    n_frames = int(duration / FRAME_DUR) + 2
    registry = SpeakerRegistry(threshold=registry_threshold)
    kwargs = {"num_speakers": num_speakers} if num_speakers else {}
    votes: list[dict[str, int]] = [{} for _ in range(n_frames)]

    t0 = time.perf_counter()
    pos = 0
    while pos < n:
        end = min(pos + window, n)
        chunk = data[pos:end]
        if len(chunk) < int(0.5 * SAMPLE_RATE):
            break
        waveform = torch.from_numpy(chunk[np.newaxis, :])
        try:
            result = pipeline({"waveform": waveform, "sample_rate": SAMPLE_RATE}, **kwargs)
        except Exception:
            pos += step
            continue
        label_map  = registry.resolve(result)
        annotation = result.speaker_diarization if hasattr(result, "speaker_diarization") else result
        _cast_votes(votes, annotation, label_map, pos / SAMPLE_RATE, n_frames)
        pos += step
    elapsed = time.perf_counter() - t0

    frames = [(fi * FRAME_DUR,
               max(votes[fi], key=votes[fi].get) if votes[fi] else None)
              for fi in range(n_frames) if fi * FRAME_DUR < duration]
    raw_segs = _frames_to_segs(frames)
    result_segs = merge_segments(raw_segs)
    if num_speakers is not None:
        result_segs = merge_segments(limit_speakers(result_segs, num_speakers))
    return result_segs, elapsed


# ── Streaming mode ────────────────────────────────────────────────────────────

def stream_online(
    data: np.ndarray,
    pipeline,
    chunk_s: float = 6.0,
    step_s: float = 1.0,
    registry_threshold: float = 0.70,
    num_speakers: int | None = None,
) -> Generator[list[dict], None, None]:
    """
    Generator that yields speaker segments incrementally as audio is processed.
    Latency ≈ chunk_s × RTF ≈ 0.9s at the default chunk_s=6s.

    After processing each chunk, segments that won't receive further votes are
    yielded immediately. Total throughput matches run_online(); only the output
    delivery differs.

    Latency vs accuracy trade-off:
        chunk_s=4s  → lower latency, but confusion rises (too short to separate 2 voices)
        chunk_s=6s  → default: low latency ~0.9s (DER 13.81% on 11 GT files)
        chunk_s=9s  → robust on long turns, blurs fast-alternating short turns

    Note: adjacent yields may split a continuous speaker turn at a step boundary.
    Merge consecutive same-speaker segments if needed after collecting all yields.

    Usage:
        for segs in stream_online(data, pipeline, chunk_s=6.0):
            for seg in segs:
                print(f"[{seg['start']:.1f}s] {seg['speaker']}: ...")
    """
    chunk = int(chunk_s * SAMPLE_RATE)
    step  = int(step_s  * SAMPLE_RATE)
    n = len(data)
    duration = n / SAMPLE_RATE
    n_frames = int(duration / FRAME_DUR) + 2
    registry = SpeakerRegistry(threshold=registry_threshold)
    kwargs = {"num_speakers": num_speakers} if num_speakers else {}
    votes: list[dict[str, int]] = [{} for _ in range(n_frames)]

    emit_fi = 0  # next frame index not yet yielded
    pos = 0

    while pos < n:
        end = min(pos + chunk, n)
        audio_chunk = data[pos:end]
        if len(audio_chunk) < int(0.5 * SAMPLE_RATE):
            break
        offset_s = pos / SAMPLE_RATE

        waveform = torch.from_numpy(audio_chunk[np.newaxis, :])
        try:
            result = pipeline({"waveform": waveform, "sample_rate": SAMPLE_RATE}, **kwargs)
        except Exception:
            pos += step
            continue

        label_map  = registry.resolve(result)
        annotation = result.speaker_diarization if hasattr(result, "speaker_diarization") else result
        _cast_votes(votes, annotation, label_map, offset_s, n_frames)

        pos += step

        # Frames before pos/SAMPLE_RATE are now stable (no future window covers them)
        new_emit_fi = int((pos / SAMPLE_RATE) / FRAME_DUR)
        if new_emit_fi <= emit_fi:
            continue

        frames = [
            (fi * FRAME_DUR, max(votes[fi], key=votes[fi].get) if votes[fi] else None)
            for fi in range(emit_fi, min(new_emit_fi, n_frames))
        ]
        new_segs = _frames_to_segs(frames)
        emit_fi = new_emit_fi

        if new_segs:
            yield new_segs

    # Flush remaining frames
    frames = []
    for fi in range(emit_fi, n_frames):
        t = fi * FRAME_DUR
        if t >= duration:
            break
        frames.append((t, max(votes[fi], key=votes[fi].get) if votes[fi] else None))

    final_segs = _frames_to_segs(frames)
    if final_segs:
        yield final_segs
