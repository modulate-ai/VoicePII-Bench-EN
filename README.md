# PII/PHI Detection Under Real-World ASR Noise: Benchmark and Code

This repository contains the dataset and benchmark harness for our paper
evaluating how automatic speech recognition (ASR) transcription quality
affects downstream PII/PHI detection across three open-source detector
architectures.

**Note (anonymous review copy):** this repository has been anonymized for
double-blind review. The full dataset (audio + ground-truth labels), a
persistent citation, and any author-identifying information will be added
upon acceptance.

## What's Included

| Component | Description |
|---|---|
| `pii_phi_voice_benchmark_dataset.csv` | 545 ground-truth clips: script text, entity class, target PII/PHI value, and character-offset span |
| `audio_en/` | Corresponding synthesized audio for each clip (`<script_id>.mp3`) |
| `detectors.py` | Entity maps and detection functions for the 3 evaluated detectors |
| `asr.py` | Transcription functions for the 2 evaluated ASR systems |
| `scoring.py` | Span-matching scoring logic (strict / trimmed-strict / compound-strict / lenient) |
| `run_ground_truth_benchmark.py` | Scores a detector directly against ground-truth text (the oracle condition) |
| `run_asr_benchmark.py` | Transcribes audio with a chosen ASR system, then scores a detector against the resulting transcript |
| `build_table.py` | Assembles the paper's main results table from all scored conditions |

This repository is intentionally scoped to only what is reported in the
paper: three detector architectures (GLiNER, OpenPipe, Piiranha) and two
ASR systems (Cohere Transcribe, Parakeet TDT), evaluated on the
English-language portion of a larger benchmark we constructed. A broader
multilingual extension exists but is outside the scope of this paper (see
Limitations in the paper).

## Detectors and ASR Systems Evaluated

**Detectors** (span different architectural approaches to PII/PHI detection):
- **GLiNER** (`nvidia/gliner-PII`) -- zero-shot NER, targets arbitrary entity labels at inference time
- **OpenPipe** (`PII-Redact-General`) -- generative model, tags entities inline in rewritten text
- **Piiranha** (`iiiorg/piiranha-v1-detect-personal-information`) -- fixed-taxonomy token classifier

**ASR systems** (selected for a substantial accuracy gap on our audio):
- **Cohere Transcribe** (`CohereLabs/cohere-transcribe-03-2026`) -- 6.49% WER
- **Parakeet TDT** (`nvidia/parakeet-tdt-0.6b-v3`) -- 20.64% WER

## Setup

We recommend a virtual environment rather than installing into your
system Python; on Debian/Ubuntu-based systems in particular, a bare `pip
install` outside a virtual environment will typically fail outright with
an `externally-managed-environment` error.

```bash
python3 -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate

pip install torch transformers>=5.4.0 gliner pii-redaction

# Cohere Transcribe is a gated model -- accept its license on the model
# page, then authenticate:
huggingface-cli login
```

Each new terminal session requires re-running `source venv/bin/activate`
before using any script in this repository -- if a command fails with a
`ModuleNotFoundError` for a package you already installed, this is the
most common cause.

GPU acceleration (CUDA or Apple Silicon MPS) is auto-detected and strongly
recommended; all three detectors and both ASR systems fall back to CPU
but will be substantially slower.

## Reproducing the Paper's Results Table

```bash
# Ground-truth (oracle) condition
python3 run_ground_truth_benchmark.py --detector gliner
python3 run_ground_truth_benchmark.py --detector openpipe
python3 run_ground_truth_benchmark.py --detector piiranha

# Cohere Transcribe condition
python3 run_asr_benchmark.py --asr cohere --detector gliner
python3 run_asr_benchmark.py --asr cohere --detector openpipe
python3 run_asr_benchmark.py --asr cohere --detector piiranha

# Parakeet TDT condition
python3 run_asr_benchmark.py --asr parakeet --detector gliner
python3 run_asr_benchmark.py --asr parakeet --detector openpipe
python3 run_asr_benchmark.py --asr parakeet --detector piiranha

# Assemble the final table
python3 build_table.py
```

Each script is independently resumable: cached transcripts
(`transcripts_<asr>/`) and cached detector outputs
(`ground_truth_raw_<detector>/`) are skipped on re-run, so an interrupted
run can simply be restarted. Add `--limit N` to any command to smoke-test
on a small subset before committing to the full 545-clip run.

## Scoring Methodology

We report four span-matching metrics for every prediction:

- **Strict**: exact character-offset match.
- **Trimmed-strict**: exact match after stripping leading/trailing
  whitespace or punctuation a detector's own offset computation
  sometimes includes. Reported separately from raw strict for
  transparency; for Piiranha specifically, this correction is necessary
  because a confirmed tokenizer artifact otherwise affects nearly every
  predicted span (raw strict F1 of 0.0 vs. trimmed-strict F1 in line with
  lenient F1). No analogous gap was observed for GLiNER or OpenPipe.
- **Compound-strict**: credits a detector for correctly decomposing one
  ground-truth entity into two or more adjacent predicted spans (e.g. a
  full name split into separate given-name and surname spans). This
  credit is only applied when at least two spans are genuinely being
  merged; a single span must still match an expected type for the gold
  entity, which prevents a lone, wrongly-typed detection from receiving
  undeserved credit (see full discussion in the paper).
- **Lenient**: any character overlap between predicted and gold spans.

All metrics are computed only over each detector's own supported entity
subset (i.e., entity types it was designed or mapped to detect), so a
detector is never penalized for entity types outside its scope.

For the two ASR conditions, a detector's predicted spans are computed on
the ASR transcript, then mapped back to the original ground-truth text's
character coordinates via word-level sequence alignment
(`scoring.py`). Clips where the transcript has diverged too far
from the original text to trust this mapping (aligned word overlap below
30%) are excluded from scoring for that condition and reported as
alignment failures in each script's console output.

## Dataset

545 short synthetic spoken utterances (21-35 words, ~8-14 seconds each),
each containing exactly one PII/PHI value from a taxonomy of 109 entity
classes across 10 categories. Script text was LLM-generated and
subsequently reviewed and corrected by human annotators; audio was
synthesized via text-to-speech across a 12-voice pool stratified across 8
accent categories and both genders, with all audio likewise
human-reviewed for synthesis artifacts.

**All PII/PHI values in this dataset are synthetically generated and do
not correspond to any real individual.**

## License

- **Code** (all `.py` files in this repository): MIT License.
- **Dataset** (`pii_phi_voice_benchmark_dataset.csv` and `audio_en/`): CC-BY-4.0.

This license applies only to the code and dataset contained in this
repository. It does **not** extend to the third-party models this code
downloads and calls at runtime (GLiNER, OpenPipe, Piiranha, Cohere
Transcribe, Parakeet TDT), each of which is subject to its own separate
license and terms of use set by its respective publisher. Notably,
Piiranha is distributed under CC-BY-NC-ND-4.0 (noncommercial, no
derivatives) and Cohere Transcribe is a gated model requiring separate
license acceptance -- users of this repository are responsible for
independently reviewing and complying with each model's own terms before
use.

## Citation

A citation will be provided here upon acceptance.

