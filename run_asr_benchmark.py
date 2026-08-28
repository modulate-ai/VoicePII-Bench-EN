"""
run_asr_benchmark.py

Runs one of the paper's two ASR conditions (Cohere Transcribe or Parakeet
TDT) plus one of the paper's three detectors (GLiNER, OpenPipe, Piiranha),
scoring against ground truth. Since the detector operates on the ASR
transcript (not ground-truth text), predicted spans are mapped back to
ground-truth coordinates via word-level alignment (paper_scoring.py)
before scoring.

Transcripts are cached separately from detection results, so re-running
with a different --detector does not require re-transcribing.

USAGE
    python3 run_asr_benchmark.py --asr cohere --detector gliner
    python3 run_asr_benchmark.py --asr parakeet --detector openpipe
    python3 run_asr_benchmark.py --asr cohere --detector piiranha --limit 10

OUTPUT
    transcripts_<asr>/<script_id>.txt        cached transcripts (shared across detectors)
    <asr>_<detector>_results.csv             per-clip detail
    <asr>_<detector>_overall.csv              corpus-wide precision/recall/F1
    <asr>_wer.csv                             per-clip WER for this ASR condition
"""

import argparse
import csv
import json
import os

from paper_asr import ASR_CONDITIONS, transcribe
from paper_detectors import DETECTORS, ENTITY_MAPS, run_detector
from paper_scoring import (score_row, write_grouped_metrics, aggregate,
                            build_alignment, alignment_is_reliable, map_span_to_original,
                            tokenize_words_with_offsets)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(SCRIPT_DIR, "pii_phi_voice_benchmark_dataset.csv")
AUDIO_DIR = os.path.join(SCRIPT_DIR, "audio_en")


def word_error_rate(reference, hypothesis):
    ref_words, _ = tokenize_words_with_offsets(reference)
    hyp_words, _ = tokenize_words_with_offsets(hypothesis)
    if not ref_words:
        return 0.0
    # Standard WER via Levenshtein distance on word sequences.
    n, m = len(ref_words), len(hyp_words)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return round(dp[n][m] / n, 4)


def main():
    parser = argparse.ArgumentParser(description="Run an ASR-condition benchmark for the paper.")
    parser.add_argument("--asr", type=str, required=True, choices=ASR_CONDITIONS)
    parser.add_argument("--detector", type=str, required=True, choices=DETECTORS)
    parser.add_argument("--input", type=str, default=None,
                         help="Path to the dataset CSV (default: pii_phi_voice_benchmark_dataset.csv)")
    parser.add_argument("--audio-dir", type=str, default=None, help="Default: audio_en")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N rows.")
    args = parser.parse_args()

    dataset_path = args.input if args.input else DATASET_PATH
    audio_dir = args.audio_dir if args.audio_dir else AUDIO_DIR
    transcript_dir = os.path.join(SCRIPT_DIR, f"transcripts_{args.asr}")
    os.makedirs(transcript_dir, exist_ok=True)

    entity_map = ENTITY_MAPS[args.detector]

    with open(dataset_path, encoding="utf-8") as f:
        dataset_rows = list(csv.DictReader(f))
    if args.limit:
        dataset_rows = dataset_rows[:args.limit]

    # ---- Phase 1: transcribe (cached, shared across all detectors) ----
    transcripts = {}
    wer_rows = []
    n_transcribe_fail = 0
    unique_clips = {row["script_id"]: row for row in dataset_rows}

    for i, (script_id, row) in enumerate(unique_clips.items(), 1):
        transcript_path = os.path.join(transcript_dir, f"{script_id}.txt")
        if os.path.exists(transcript_path):
            with open(transcript_path, encoding="utf-8") as f:
                transcripts[script_id] = f.read()
            continue
        audio_path = os.path.join(audio_dir, f"{script_id}.mp3")
        if not os.path.exists(audio_path):
            print(f"[{i}/{len(unique_clips)}] SKIP {script_id}: audio file not found")
            n_transcribe_fail += 1
            continue
        try:
            text = transcribe(args.asr, audio_path)
        except Exception as e:
            print(f"[{i}/{len(unique_clips)}] FAIL {script_id} (transcription): {e}")
            n_transcribe_fail += 1
            continue
        transcripts[script_id] = text
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[{i}/{len(unique_clips)}] transcribed {script_id}")

    for script_id, row in unique_clips.items():
        if script_id in transcripts:
            wer = word_error_rate(row["script_text"], transcripts[script_id])
            wer_rows.append({"script_id": script_id, "error_rate": wer})

    wer_out = os.path.join(SCRIPT_DIR, f"{args.asr}_wer.csv")
    with open(wer_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["script_id", "error_rate"])
        writer.writeheader()
        writer.writerows(wer_rows)
    print(f"Wrote {wer_out}")
    if wer_rows:
        avg_wer = sum(r["error_rate"] for r in wer_rows) / len(wer_rows)
        print(f"Average WER ({args.asr}): {avg_wer:.4f}")

    # ---- Phase 2: run the detector on each transcript, align, score ----
    detail_rows = []
    scored_rows = []
    n_alignment_failed = 0

    for row in dataset_rows:
        script_id = row["script_id"]
        original = row["script_text"]
        gold_start, gold_end = int(row["char_start"]), int(row["char_end"])
        entity = row["entity_class"]

        transcript = transcripts.get(script_id)
        if transcript is None:
            continue

        try:
            predicted_spans_transcript = run_detector(args.detector, transcript)
        except Exception as e:
            print(f"FAIL {script_id} (detection): {e}")
            continue

        orig_words, orig_spans, trans_words, trans_spans, opcodes = build_alignment(original, transcript)
        if not alignment_is_reliable(orig_words, opcodes):
            n_alignment_failed += 1
            detail_rows.append({
                "script_id": script_id, "category": row["category"], "entity_class": entity,
                "gold_value": row["pii_value"], "transcript": transcript, "alignment_ok": False,
                "mapped_spans": "", "detected": "", "exact_match": "",
                "trimmed_exact_match": "", "compound_exact_match": "", "supported_by_detector": bool(entity_map.get(entity)),
            })
            continue

        mapped_spans = []
        for ps, pe, ptype in predicted_spans_transcript:
            mapped = map_span_to_original(ps, pe, orig_spans, trans_spans, opcodes)
            if mapped:
                mapped_spans.append((mapped[0], mapped[1], ptype))

        allowed_types = entity_map.get(entity)
        candidate_spans = ([s for s in mapped_spans if s[2] in allowed_types]
                            if allowed_types else mapped_spans)

        result = score_row(original, gold_start, gold_end, candidate_spans,
                            all_spans=mapped_spans, allowed_types=allowed_types)
        detail_rows.append({
            "script_id": script_id, "category": row["category"], "entity_class": entity,
            "gold_value": row["pii_value"], "transcript": transcript, "alignment_ok": True,
            "mapped_spans": json.dumps(mapped_spans),
            "detected": result["detected"], "exact_match": result["strict_tp"],
            "trimmed_exact_match": result["trimmed_strict_tp"],
            "compound_exact_match": result["compound_strict_tp"],
            "supported_by_detector": bool(allowed_types),
        })
        if allowed_types:
            scored_rows.append(result)

    detail_out = os.path.join(SCRIPT_DIR, f"{args.asr}_{args.detector}_results.csv")
    overall_out = os.path.join(SCRIPT_DIR, f"{args.asr}_{args.detector}_overall.csv")

    with open(detail_out, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["script_id", "category", "entity_class", "gold_value", "transcript",
                      "alignment_ok", "mapped_spans", "detected", "exact_match",
                      "trimmed_exact_match", "compound_exact_match", "supported_by_detector"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detail_rows)
    print(f"Wrote {detail_out}")

    write_grouped_metrics(overall_out, {"overall_supported_only": scored_rows})

    m = aggregate(scored_rows) if scored_rows else None
    print(f"\n=== {args.detector} on {args.asr} transcripts ===")
    print(f"Clips scored (supported entities only): {len(scored_rows)}  |  "
          f"transcription failures: {n_transcribe_fail}  |  alignment failures: {n_alignment_failed}")
    if m:
        print(f"Strict:          F1={m['span_strict_f1']}")
        print(f"Trimmed-strict:  F1={m['span_trimmed_strict_f1']}")
        print(f"Compound-strict: F1={m['span_compound_strict_f1']}")
        print(f"Lenient:         F1={m['span_lenient_f1']}")


if __name__ == "__main__":
    main()
