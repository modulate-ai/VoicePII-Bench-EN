"""
build_paper_table.py

Assembles the paper's Table 1: span-level F1 (strict, compound-strict,
lenient) for the 3 detectors (GLiNER, OpenPipe, Piiranha) under the 3
conditions (ground truth, Cohere Transcribe, Parakeet TDT).

Run this after all 9 combinations have been scored:
    python3 run_ground_truth_benchmark.py --detector {gliner,openpipe,piiranha}
    python3 run_asr_benchmark.py --asr {cohere,parakeet} --detector {gliner,openpipe,piiranha}

USAGE
    python3 build_paper_table.py
"""

import csv
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DETECTORS = ["gliner", "openpipe", "piiranha"]
CONDITIONS = [
    ("Ground Truth", None),  # (label, asr condition or None for ground truth)
    ("Cohere Transcribe", "cohere"),
    ("Parakeet TDT", "parakeet"),
]

# For Piiranha specifically, raw strict is corrupted by a confirmed
# tokenizer artifact (see paper Methodology/Appendix) -- use trimmed_strict
# as its "strict" figure. GLiNER and OpenPipe show no such gap.
STRICT_METRIC_OVERRIDE = {"piiranha": "span_trimmed_strict_f1"}


def read_overall(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if r["group"] == "overall_supported_only":
            return r
    return None


def main():
    print(f"{'Detector':10s} {'Condition':20s} {'Strict':>8s} {'Compound':>10s} {'Lenient':>8s}")
    print("-" * 60)

    for detector in DETECTORS:
        strict_key = STRICT_METRIC_OVERRIDE.get(detector, "span_strict_f1")
        for label, asr in CONDITIONS:
            if asr is None:
                path = os.path.join(SCRIPT_DIR, f"ground_truth_{detector}_overall.csv")
            else:
                path = os.path.join(SCRIPT_DIR, f"{asr}_{detector}_overall.csv")
            row = read_overall(path)
            if row is None:
                print(f"{detector:10s} {label:20s} {'not run':>8s} {'not run':>10s} {'not run':>8s}")
                continue
            strict = row[strict_key]
            compound = row["span_compound_strict_f1"]
            lenient = row["span_lenient_f1"]
            print(f"{detector:10s} {label:20s} {strict:>8s} {compound:>10s} {lenient:>8s}")
        print()

    print("Note: Piiranha's 'Strict' column uses trimmed_strict (see paper Methodology) --")
    print("raw strict is corrupted by a confirmed tokenizer offset artifact affecting nearly")
    print("every predicted span. GLiNER and OpenPipe use raw strict (no such gap observed).")


if __name__ == "__main__":
    main()
