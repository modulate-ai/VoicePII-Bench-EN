"""
paper_scoring.py

Shared scoring logic for all three conditions (ground truth, Cohere
Transcribe, Parakeet TDT) evaluated in the paper. Three match-stringency
levels are computed for every prediction:
  - strict: exact character-offset match.
  - trimmed_strict: exact match after stripping leading/trailing
    whitespace/punctuation a detector's own offset computation sometimes
    includes (Section on Piiranha's tokenizer artifact in the paper).
  - compound_strict: credits a detector for correctly decomposing one
    gold entity into 2+ adjacent sub-spans (e.g. GIVENNAME + SURNAME).
    Critically, this credit requires 2+ spans to be genuinely merged; a
    single span must still match an allowed type. See Appendix on
    scoring methodology in the paper for the bug this fixes.
  - lenient: any character overlap between predicted and gold spans.

Also includes the word-level fuzzy alignment used to map a detector's
predicted spans (computed on an ASR transcript) back to the corresponding
character positions in the original ground-truth text, needed for the
Cohere Transcribe / Parakeet TDT conditions.
"""

import difflib
import re

# ---------------------------------------------------------------------------
# Core span matching
# ---------------------------------------------------------------------------

TRIM_CHARS = " \t\n,;:.!?\"'"  # whitespace + common trailing punctuation
COMPOUND_ADJACENT_GAP = 3  # merge predicted spans within this many characters
                            # of the gold span before checking compound-strict


def overlaps(a_start, a_end, b_start, b_end):
    return a_start < b_end and b_start < a_end


def trim_span(text, start, end, gold_start, gold_end):
    """Trims whitespace/punctuation from a predicted span, but only
    characters that extend *beyond* the true gold boundary -- never into
    it."""
    while start < end and start < gold_start and text[start] in TRIM_CHARS:
        start += 1
    while end > start and end > gold_end and text[end - 1] in TRIM_CHARS:
        end -= 1
    return start, end


def merge_adjacent_spans(spans, gold_start, gold_end, allowed_types=None):
    """Merges predicted spans within COMPOUND_ADJACENT_GAP characters of
    the gold span into one covering region, crediting a detector that
    decomposes one gold entity into several adjacent sub-spans.

    The type-agnostic merge is only applied when 2+ spans are genuinely
    combined; a single span must still pass the allowed_types check the
    same way strict/lenient do. Without this restriction, a single
    correctly-positioned but wrongly-typed span would get undeserved
    credit -- confirmed via a real anomaly where a detector's
    compound-strict F1 exceeded its own lenient F1, which should be
    impossible (see paper Appendix)."""
    relevant = [s for s in spans
                if overlaps(s[0], s[1], gold_start - COMPOUND_ADJACENT_GAP, gold_end + COMPOUND_ADJACENT_GAP)]
    if not relevant:
        return []
    if len(relevant) == 1 and allowed_types is not None and relevant[0][2] not in allowed_types:
        return []
    relevant.sort(key=lambda s: s[0])
    merged_start = min(s[0] for s in relevant)
    merged_end = max(s[1] for s in relevant)
    return [(merged_start, merged_end, relevant[0][2])]


def score_row(text, gold_start, gold_end, predicted_spans, all_spans=None, allowed_types=None):
    """predicted_spans: type-filtered candidates, used for strict/trimmed/
    lenient. all_spans: the FULL unfiltered set of spans for this clip,
    used ONLY for compound-strict merging (so a detector splitting one
    entity across different type labels, e.g. BUILDINGNUM+STREET+CITY,
    isn't penalized by pre-filtering on a single expected type).
    allowed_types: valid types for this gold entity, used by
    merge_adjacent_spans to reject a lone, wrongly-typed span."""
    if all_spans is None:
        all_spans = predicted_spans

    exact_hits = [s for s in predicted_spans if s[0] == gold_start and s[1] == gold_end]
    overlap_hits = [s for s in predicted_spans if overlaps(s[0], s[1], gold_start, gold_end)]

    trimmed_spans = [(*trim_span(text, s, e, gold_start, gold_end), t) for s, e, t in predicted_spans]
    trimmed_exact_hits = [s for s in trimmed_spans if s[0] == gold_start and s[1] == gold_end]

    all_trimmed_spans = [(*trim_span(text, s, e, gold_start, gold_end), t) for s, e, t in all_spans]
    compound_spans = merge_adjacent_spans(all_trimmed_spans, gold_start, gold_end, allowed_types=allowed_types)
    compound_exact_hits = [s for s in compound_spans if s[0] == gold_start and s[1] == gold_end]

    return {
        "strict_tp": 1 if exact_hits else 0,
        "strict_fp": max(0, len(predicted_spans) - len(exact_hits)),
        "strict_fn": 0 if exact_hits else 1,
        "trimmed_strict_tp": 1 if trimmed_exact_hits else 0,
        "trimmed_strict_fp": max(0, len(trimmed_spans) - len(trimmed_exact_hits)),
        "trimmed_strict_fn": 0 if trimmed_exact_hits else 1,
        "compound_strict_tp": 1 if compound_exact_hits else 0,
        "compound_strict_fp": max(0, len(compound_spans) - len(compound_exact_hits)),
        "compound_strict_fn": 0 if compound_exact_hits else 1,
        "lenient_tp": 1 if overlap_hits else 0,
        "lenient_fp": max(0, len(predicted_spans) - len(overlap_hits)),
        "lenient_fn": 0 if overlap_hits else 1,
        "detected": 1 if overlap_hits else 0,
    }


def prf(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return round(precision, 4), round(recall, 4), round(f1, 4)


def aggregate(rows):
    def sums(prefix):
        return (sum(r[f"{prefix}_tp"] for r in rows), sum(r[f"{prefix}_fp"] for r in rows),
                sum(r[f"{prefix}_fn"] for r in rows))
    sp, sr, sf = prf(*sums("strict"))
    tsp, tsr, tsf = prf(*sums("trimmed_strict"))
    csp, csr, csf = prf(*sums("compound_strict"))
    lp, lr, lf = prf(*sums("lenient"))
    return {
        "n_clips": len(rows),
        "span_strict_precision": sp, "span_strict_recall": sr, "span_strict_f1": sf,
        "span_trimmed_strict_precision": tsp, "span_trimmed_strict_recall": tsr, "span_trimmed_strict_f1": tsf,
        "span_compound_strict_precision": csp, "span_compound_strict_recall": csr, "span_compound_strict_f1": csf,
        "span_lenient_precision": lp, "span_lenient_recall": lr, "span_lenient_f1": lf,
    }


def write_grouped_metrics(path, groups):
    import csv
    fieldnames = ["group", "n_clips", "span_strict_precision", "span_strict_recall", "span_strict_f1",
                  "span_trimmed_strict_precision", "span_trimmed_strict_recall", "span_trimmed_strict_f1",
                  "span_compound_strict_precision", "span_compound_strict_recall", "span_compound_strict_f1",
                  "span_lenient_precision", "span_lenient_recall", "span_lenient_f1"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for group_name, rows in sorted(groups.items()):
            writer.writerow({"group": group_name, **aggregate(rows)})
    print(f"Wrote {path}")


# ---------------------------------------------------------------------------
# Word-level alignment (ASR conditions only: maps a detector's predicted
# spans, computed on a transcript, back to original ground-truth
# coordinates)
# ---------------------------------------------------------------------------

WORD_RE = re.compile(r"[\w']+", re.UNICODE)
MIN_MATCH_RATIO = 0.3  # below this, the transcript is too different from
                        # the original to trust any span mapping from it


def tokenize_words_with_offsets(text):
    spans = [(m.start(), m.end()) for m in WORD_RE.finditer(text)]
    words = [text[s:e].lower() for s, e in spans]
    return words, spans


def build_alignment(original, transcript):
    orig_words, orig_spans = tokenize_words_with_offsets(original)
    trans_words, trans_spans = tokenize_words_with_offsets(transcript)
    matcher = difflib.SequenceMatcher(None, orig_words, trans_words, autojunk=False)
    return orig_words, orig_spans, trans_words, trans_spans, matcher.get_opcodes()


def alignment_is_reliable(orig_words, opcodes):
    if not orig_words:
        return True
    matched = sum((i2 - i1) for tag, i1, i2, j1, j2 in opcodes if tag == "equal")
    return (matched / len(orig_words)) >= MIN_MATCH_RATIO


def map_span_to_original(pred_start, pred_end, orig_spans, trans_spans, opcodes):
    """Maps a (start, end) char span in transcript coordinates to the
    corresponding (start, end) span in original coordinates, or None if it
    can't be reasonably mapped."""
    trans_idxs = [i for i, (s, e) in enumerate(trans_spans) if s < pred_end and pred_start < e]
    if not trans_idxs:
        return None
    j1_target, j2_target = min(trans_idxs), max(trans_idxs) + 1

    mapped_word_idxs = []
    for tag, i1, i2, j1, j2 in opcodes:
        if j2 <= j1_target or j1 >= j2_target:
            continue
        if tag == "equal":
            ov_j1, ov_j2 = max(j1, j1_target), min(j2, j2_target)
            mapped_word_idxs.append(i1 + (ov_j1 - j1))
            mapped_word_idxs.append(i1 + (ov_j2 - j1) - 1)
        elif i2 > i1:
            mapped_word_idxs.append(i1)
            mapped_word_idxs.append(i2 - 1)

    if not mapped_word_idxs:
        return None
    start_w = max(0, min(mapped_word_idxs))
    end_w = max(0, min(max(mapped_word_idxs), len(orig_spans) - 1))
    if start_w >= len(orig_spans):
        return None
    return orig_spans[start_w][0], orig_spans[end_w][1]
