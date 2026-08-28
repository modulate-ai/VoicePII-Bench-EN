"""
run_ground_truth_benchmark.py

Scores one of the paper's three detectors (GLiNER, OpenPipe, Piiranha)
directly against ground-truth script text -- the oracle, 0%-WER condition
in Table 1 of the paper. No transcription or alignment needed, since
predicted spans are already in ground-truth coordinates.

USAGE
    python3 run_ground_truth_benchmark.py --detector gliner
    python3 run_ground_truth_benchmark.py --detector openpipe
    python3 run_ground_truth_benchmark.py --detector piiranha
    python3 run_ground_truth_benchmark.py --detector gliner --limit 10   # smoke test

OUTPUT
    ground_truth_<detector>_results.csv    per-clip detail
    ground_truth_<detector>_overall.csv     corpus-wide precision/recall/F1
"""

import argparse
import csv
import json
import os

from paper_detectors import DETECTORS, ENTITY_MAPS, run_detector
from paper_scoring import score_row, write_grouped_metrics, aggregate

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(SCRIPT_DIR, "pii_phi_voice_benchmark_dataset.csv")
RAW_DIR_TEMPLATE = os.path.join(SCRIPT_DIR, "ground_truth_raw_{detector}")


def main():
    parser = argparse.ArgumentParser(description="Score a paper detector against ground-truth text.")
    parser.add_argument("--detector", type=str, required=True, choices=DETECTORS)
    parser.add_argument("--input", type=str, default=None,
                         help="Path to the dataset CSV (default: pii_phi_voice_benchmark_dataset.csv)")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N rows.")
    args = parser.parse_args()

    dataset_path = args.input if args.input else DATASET_PATH
    raw_dir = RAW_DIR_TEMPLATE.format(detector=args.detector)
    os.makedirs(raw_dir, exist_ok=True)

    entity_map = ENTITY_MAPS[args.detector]

    with open(dataset_path, encoding="utf-8") as f:
        dataset_rows = list(csv.DictReader(f))
    if args.limit:
        dataset_rows = dataset_rows[:args.limit]

    detail_rows = []
    scored_rows = []
    n_failed = 0

    # Group rows by unique script_text so each clip's text is only run
    # through the detector once, even if multiple entity rows share it.
    text_to_rows = {}
    for row in dataset_rows:
        text_to_rows.setdefault(row["script_text"], []).append(row)

    predicted_spans_by_text = {}
    for i, text in enumerate(text_to_rows, 1):
        script_id = text_to_rows[text][0]["script_id"]
        raw_path = os.path.join(raw_dir, f"{script_id}.json")
        if os.path.exists(raw_path):
            with open(raw_path, encoding="utf-8") as f:
                predicted_spans_by_text[text] = json.load(f)["spans"]
            continue
        try:
            spans = run_detector(args.detector, text)
        except Exception as e:
            print(f"[{i}/{len(text_to_rows)}] FAIL {script_id}: {e}")
            n_failed += 1
            predicted_spans_by_text[text] = []
            continue
        predicted_spans_by_text[text] = spans
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump({"script_id": script_id, "spans": spans}, f)
        print(f"[{i}/{len(text_to_rows)}] OK {script_id}")

    for row in dataset_rows:
        text = row["script_text"]
        gold_start, gold_end = int(row["char_start"]), int(row["char_end"])
        entity = row["entity_class"]
        predicted_spans = [tuple(s) for s in predicted_spans_by_text.get(text, [])]

        allowed_types = entity_map.get(entity)
        candidate_spans = ([s for s in predicted_spans if s[2] in allowed_types]
                            if allowed_types else predicted_spans)

        result = score_row(text, gold_start, gold_end, candidate_spans,
                            all_spans=predicted_spans, allowed_types=allowed_types)
        detail_rows.append({
            "script_id": row["script_id"], "category": row["category"], "entity_class": entity,
            "gold_value": row["pii_value"],
            "predicted_spans": json.dumps(predicted_spans),
            "detected": result["detected"], "exact_match": result["strict_tp"],
            "trimmed_exact_match": result["trimmed_strict_tp"],
            "compound_exact_match": result["compound_strict_tp"],
            "supported_by_detector": bool(allowed_types),
        })
        if allowed_types:
            scored_rows.append(result)

    detail_out = os.path.join(SCRIPT_DIR, f"ground_truth_{args.detector}_results.csv")
    overall_out = os.path.join(SCRIPT_DIR, f"ground_truth_{args.detector}_overall.csv")

    with open(detail_out, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["script_id", "category", "entity_class", "gold_value", "predicted_spans",
                      "detected", "exact_match", "trimmed_exact_match", "compound_exact_match",
                      "supported_by_detector"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detail_rows)
    print(f"Wrote {detail_out}")

    write_grouped_metrics(overall_out, {"overall_supported_only": scored_rows})

    m = aggregate(scored_rows) if scored_rows else None
    print(f"\n=== {args.detector} on ground-truth text ===")
    print(f"Clips scored (supported entities only): {len(scored_rows)}  |  failed: {n_failed}")
    if m:
        print(f"Strict:          F1={m['span_strict_f1']}")
        print(f"Trimmed-strict:  F1={m['span_trimmed_strict_f1']}")
        print(f"Compound-strict: F1={m['span_compound_strict_f1']}")
        print(f"Lenient:         F1={m['span_lenient_f1']}")


if __name__ == "__main__":
    main()
