#!/usr/bin/env python3
"""
In bảng so sánh Whisper vs Qwen3 vs Nemotron từ các JSON đã lưu trong outputs/.
Chạy sau khi đã có asr_whisper.json / asr_qwen.json / asr_nemotron.json.
Không cần GPU:  python compare_asr_3models.py
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

SOURCES = {
    "Whisper": "outputs/asr_whisper.json",
    "Qwen3": "outputs/asr_qwen.json",
    "Nemotron": "outputs/asr_nemotron.json",
}

data = {n: json.load(open(p, encoding="utf-8")) for n, p in SOURCES.items() if Path(p).exists()}
if not data:
    sys.exit("Không tìm thấy JSON ASR nào trong outputs/ — chạy eval_asr_models.py trước.")

names = list(data.keys())

print(f"\n{'Model':<12}{'mean WER':>10}{'mean CER':>10}{'mean RTF':>10}")
print("-" * 42)
for n in names:
    d = data[n]
    print(f"{n:<12}{d['mean_wer']:>9}%{d['mean_cer']:>9}%{d['mean_rtf']:>9}x")

files = sorted({f["file"] for d in data.values() for f in d["files"]})
per = {n: {f["file"]: f for f in data[n]["files"]} for n in names}

print(f"\nPer-file WER:\n{'file':<10}" + "".join(f"{n:>11}" for n in names))
for fn in files:
    row = f"{fn.replace('.wav',''):<10}"
    for n in names:
        m = per[n].get(fn)
        row += f"{(str(m['WER_%'])+'%') if m else '—':>11}"
    print(row)

if len(names) == 3:
    print("\n→ Model WER trung bình thấp nhất:",
          min(names, key=lambda n: data[n]["mean_wer"]))
