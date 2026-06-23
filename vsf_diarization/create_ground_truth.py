#!/usr/bin/env python3
"""
Tạo draft ground truth tự động với pipeline chính xác nhất so với GT reviewed:
offline pyannote + Whisper turbo word-level (DER 5.82%, WER 2.12% trên test1).

Lưu ý: online (chunk=6s) tuy DER raw-diarization thấp hơn nhưng output segment
quá thô (3 segs vs 17 segs reviewed) — không phù hợp làm draft annotation.

User mở file ground_truth/<name>.json, sửa những chỗ sai, lưu lại.

Usage:
  python create_ground_truth.py test/test01.wav --language vi
  python create_ground_truth.py test/ --language vi          # toàn bộ folder, bỏ qua file đã có GT
  python create_ground_truth.py test/ --num-speakers 2       # ép số speaker nếu biết trước
  python create_ground_truth.py test/test01.wav --force      # ghi đè GT đã có
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from vsf_diarization.core.utils import (
    get_hf_token, load_audio, load_diarization_pipeline, load_whisper,
    whisper_transcribe, SAMPLE_RATE,
)
from vsf_diarization.core.diarize_offline import run_offline
from vsf_diarization.core.diarize_online import run_online

GT_DIR = Path("ground_truth")
GT_DIR.mkdir(exist_ok=True)


def build_draft(
    audio_path: str, diar_pipeline, whisper,
    language: str | None, strategy: str, num_speakers: int | None,
) -> dict:
    data = load_audio(audio_path)
    duration = len(data) / SAMPLE_RATE

    # ── Diarization — online chunk=6s là chiến lược DER thấp nhất (5.86%) ────
    if strategy == "online":
        turns, _ = run_online(data, diar_pipeline, window_s=6.0, step_s=1.0,
                              num_speakers=num_speakers)
    else:
        turns, _ = run_offline(data, diar_pipeline, num_speakers)

    # ── ASR word-level + speaker assignment ──────────────────────────────────
    results, detected_lang = whisper_transcribe(data, turns, whisper, language)

    speakers = sorted({r["speaker"] for r in results})
    speaker_map = {spk: spk for spk in speakers}  # user thay đổi ở đây

    return {
        "file": audio_path,
        "duration_s": round(duration, 3),
        "detected_language": detected_lang,
        "diarization_strategy": strategy,
        "annotator": "",  # <-- điền tên người annotate
        "annotation_status": "draft",  # draft | reviewed | verified
        # ── Hướng dẫn cho người annotate ─────────────────────────────────────
        "_instructions": [
            "1. Đổi tên speaker: thay 'SPEAKER_00' -> tên thật, ví dụ: 'BACSI', 'BENHNHAN'",
            "2. Sửa text nếu Whisper nhận sai: kiểm tra từng đoạn, sửa trực tiếp",
            "3. Sửa timestamp nếu lệch rõ (chấp nhận sai số ±0.3s)",
            "4. Thêm/xóa segment nếu cần (giữ đúng thứ tự start time)",
            "5. Đổi annotation_status: 'draft' -> 'reviewed' khi xong",
            "6. Lưu file. Dùng evaluate.py để đánh giá pipeline.",
        ],
        "speaker_labels": speaker_map,
        "segments": [
            {
                "speaker": r["speaker"],
                "start": r["start"],
                "end":   r["end"],
                "text":  r["text"],
            }
            for r in results
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Create ground truth draft")
    parser.add_argument("input", nargs="+", help="WAV file(s) or folder")
    parser.add_argument("--hf-token")
    parser.add_argument("--whisper-model", default="turbo")
    parser.add_argument("--language")
    parser.add_argument("--strategy", choices=["online", "offline"], default="offline",
                        help="Diarization strategy (offline cho draft chi tiết nhất)")
    parser.add_argument("--num-speakers", type=int)
    parser.add_argument("--force", action="store_true", help="Overwrite existing GT files")
    args = parser.parse_args()

    wav_files = []
    for inp in args.input:
        p = Path(inp)
        wav_files.extend(sorted(p.glob("*.wav")) if p.is_dir() else [p])
    if not wav_files:
        sys.exit("No WAV files found")

    hf_token = get_hf_token(args.hf_token)
    print("Loading models...")
    diar_pipeline = load_diarization_pipeline(hf_token)
    whisper = load_whisper(args.whisper_model)
    print(f"Models loaded. Strategy: {args.strategy}\n")

    for wav in wav_files:
        out_path = GT_DIR / (wav.stem + ".json")
        if out_path.exists() and not args.force:
            print(f"  SKIP {wav.name} (already exists: {out_path}). Use --force to overwrite.")
            continue

        print(f"Processing {wav.name}...")
        t0 = time.perf_counter()
        draft = build_draft(str(wav), diar_pipeline, whisper,
                            args.language, args.strategy, args.num_speakers)
        elapsed = time.perf_counter() - t0

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(draft, f, indent=2, ensure_ascii=False)

        n_segs = len(draft["segments"])
        speakers = list(draft["speaker_labels"].keys())
        print(f"  -> {out_path}  ({n_segs} segments, speakers: {speakers}, {elapsed:.1f}s)\n")

    print("Done. Mở folder ground_truth/ để chỉnh sửa.")


if __name__ == "__main__":
    main()
