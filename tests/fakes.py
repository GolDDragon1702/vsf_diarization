"""Fake pyannote pipeline + annotation cho test không cần model/GPU/HF token.

Duck-types đủ những gì StreamingSession / SpeakerRegistry dùng:
  - annotation.labels()
  - annotation.itertracks(yield_label=True) -> (Segment(start,end), track, label)
Không có speaker_embeddings → SpeakerRegistry map local-label thẳng thành global,
nên label phải nhất quán giữa các cửa sổ (ta dùng timeline cố định).

StreamingSession cast cửa sổ theo thứ tự pos = 0, step, 2·step, … (luôn +step mỗi
lần). Nên offset của lần gọi thứ k là k·step_s — fake tái dựng được mà không cần
session truyền offset, và trả về toạ độ cục bộ trong cửa sổ (như pyannote thật).
"""

from dataclasses import dataclass

SAMPLE_RATE = 16000


@dataclass
class _Seg:
    start: float
    end: float


class FakeAnnotation:
    def __init__(self, tracks):           # tracks: list[(start, end, label)]
        self._tracks = tracks

    def labels(self):
        return sorted({lbl for _, _, lbl in self._tracks})

    def itertracks(self, yield_label=False):
        for i, (s, e, lbl) in enumerate(self._tracks):
            yield (_Seg(s, e), i, lbl) if yield_label else (_Seg(s, e), i)


class FakePipeline:
    """timeline: list[(start_s, end_s, label)] theo thời gian tuyệt đối của cả audio."""

    def __init__(self, timeline, step_s: float = 1.0):
        self.timeline = timeline
        self.step_s = step_s
        self._calls = 0

    def __call__(self, payload, **kwargs):
        win_len = payload["waveform"].shape[-1] / SAMPLE_RATE
        offset = self._calls * self.step_s
        self._calls += 1
        tracks = []
        for s, e, lbl in self.timeline:
            cs, ce = max(s, offset), min(e, offset + win_len)
            if ce - cs > 0.05:
                tracks.append((round(cs - offset, 3), round(ce - offset, 3), lbl))
        return FakeAnnotation(tracks)
