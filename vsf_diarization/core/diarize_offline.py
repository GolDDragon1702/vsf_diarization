"""
Offline (batch) speaker diarization using pyannote/speaker-diarization-3.1.

Processes the entire audio at once — highest accuracy, not suitable for streaming.

Usage:
    from diarize_offline import run_offline
    turns, elapsed = run_offline(data, pipeline, num_speakers=2)
"""

import time

import numpy as np
import torch

from vsf_diarization.core.utils import SAMPLE_RATE, merge_segments


def run_offline(
    data: np.ndarray,
    pipeline,
    num_speakers: int | None = None,
) -> tuple[list[dict], float]:
    """
    Run pyannote full-audio diarization.

    Returns:
        turns   — merged segment list: [{speaker, start, end, duration}]
        elapsed — wall-clock seconds
    """
    waveform = torch.from_numpy(data[np.newaxis, :])
    kwargs = {"num_speakers": num_speakers} if num_speakers else {}

    t0 = time.perf_counter()
    result = pipeline({"waveform": waveform, "sample_rate": SAMPLE_RATE}, **kwargs)
    elapsed = time.perf_counter() - t0

    annotation = (result.speaker_diarization
                if hasattr(result, "speaker_diarization") else result)
    raw = [
        {"speaker": spk, "start": t.start, "end": t.end, "duration": t.end - t.start}
        for t, _, spk in annotation.itertracks(yield_label=True)
    ]
    return merge_segments(raw), elapsed
