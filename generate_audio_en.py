"""
generate_audio_en.py

Regenerates audio_en/<script_id>.mp3 for every row in
pii_phi_voice_benchmark_dataset.csv using two open-source TTS engines in
place of the original ElevenLabs-based audio, whose terms of service do
not allow redistributing generated speech as part of an open dataset.
See voices.py for the voice-slot -> engine mapping and the reasoning
behind it, and fetch_reference_clips.py for how Chatterbox's reference
clips were sourced. tts_overrides.py lets individual clips override what
text is actually spoken (e.g. to fix a number mispronunciation) without
touching the CSV's ground-truth script_text/spans.

Resumable: skips any script_id whose audio_en/<script_id>.mp3 already
exists, unless --overwrite is passed. Clips are processed grouped by
voice_name so a Kokoro pipeline or Chatterbox voice-cloning prompt is
loaded/prepared once per voice slot, not once per clip.

USAGE
    python3 generate_audio_en.py                            # regenerate all 545 clips
    python3 generate_audio_en.py --limit 10                 # smoke test
    python3 generate_audio_en.py --voice "Irish Female 1"    # just one voice slot
    python3 generate_audio_en.py --script-id AB-03 --overwrite  # regenerate a single clip
    python3 generate_audio_en.py --script-id AB-03,AB-05 --overwrite  # regenerate several
    python3 generate_audio_en.py --overridden-only --overwrite  # regenerate every clip with a tts_overrides.py entry
    python3 generate_audio_en.py --overwrite                # ignore existing files
"""

import argparse
import csv
import os
import subprocess

from tts_overrides import TTS_TEXT_OVERRIDES
from voices import VOICE_SYNTHESIS

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(SCRIPT_DIR, "pii_phi_voice_benchmark_dataset.csv")
AUDIO_DIR = os.path.join(SCRIPT_DIR, "audio_en")
REFERENCE_CLIPS_DIR = os.path.join(SCRIPT_DIR, "reference_clips")

# ---------------------------------------------------------------------------
# Kokoro-82M
# ---------------------------------------------------------------------------

_kokoro_pipelines = {}


def get_kokoro_pipeline(lang_code):
    if lang_code not in _kokoro_pipelines:
        from kokoro import KPipeline
        print(f"Loading Kokoro pipeline (lang_code={lang_code!r})...")
        _kokoro_pipelines[lang_code] = KPipeline(lang_code=lang_code)
    return _kokoro_pipelines[lang_code]


def synthesize_kokoro(text, voice):
    import numpy as np
    # Voice prefix encodes the phonemization ruleset Kokoro needs: 'a' for
    # American English voices (af_/am_), 'b' for British (bf_/bm_).
    lang_code = voice[0]
    pipeline = get_kokoro_pipeline(lang_code)
    chunks = [audio for _, _, audio in pipeline(text, voice=voice)]
    audio = np.concatenate([c.numpy() if hasattr(c, "numpy") else c for c in chunks])
    return audio, 24000


# ---------------------------------------------------------------------------
# Chatterbox
# ---------------------------------------------------------------------------

_chatterbox_model = None
_chatterbox_last_reference_clip = None


def get_chatterbox_model():
    global _chatterbox_model
    if _chatterbox_model is None:
        from chatterbox.tts import ChatterboxTTS
        import torch
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        print(f"Loading Chatterbox (device={device})...")
        _chatterbox_model = ChatterboxTTS.from_pretrained(device=device)
    return _chatterbox_model


def synthesize_chatterbox(text, reference_clip):
    global _chatterbox_last_reference_clip
    model = get_chatterbox_model()
    # Re-cloning the reference voice (prepare_conditionals) is the expensive
    # part -- only pass audio_prompt_path when the voice slot has changed
    # since the last call.
    if reference_clip == _chatterbox_last_reference_clip:
        audio_prompt_path = None
    else:
        audio_prompt_path = os.path.join(REFERENCE_CLIPS_DIR, reference_clip)
    wav = model.generate(text, audio_prompt_path=audio_prompt_path)
    _chatterbox_last_reference_clip = reference_clip
    return wav.squeeze(0).cpu().numpy(), model.sr


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def synthesize(text, voice_name):
    spec = VOICE_SYNTHESIS[voice_name]
    if spec["engine"] == "kokoro":
        return synthesize_kokoro(text, spec["voice"])
    if spec["engine"] == "chatterbox":
        return synthesize_chatterbox(text, spec["reference_clip"])
    raise ValueError(f"Unknown engine for voice {voice_name!r}: {spec['engine']!r}")


def write_mp3(audio, sample_rate, out_path):
    import soundfile as sf
    wav_path = out_path[: -len(".mp3")] + ".wav"
    sf.write(wav_path, audio, sample_rate)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", wav_path, out_path], check=True)
    os.remove(wav_path)


def main():
    parser = argparse.ArgumentParser(description="Regenerate audio_en/ with open-source TTS.")
    parser.add_argument("--input", type=str, default=None,
                         help="Path to the dataset CSV (default: pii_phi_voice_benchmark_dataset.csv)")
    parser.add_argument("--voice", type=str, default=None, choices=list(VOICE_SYNTHESIS),
                         help="Only regenerate clips for this one voice slot.")
    parser.add_argument("--script-id", type=str, default=None,
                         help="Only regenerate these script_ids (comma-separated).")
    parser.add_argument("--overridden-only", action="store_true",
                         help="Only regenerate clips that have a tts_overrides.py entry.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N clips.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate even if the mp3 already exists.")
    args = parser.parse_args()

    dataset_path = args.input if args.input else DATASET_PATH
    os.makedirs(AUDIO_DIR, exist_ok=True)

    with open(dataset_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if args.voice:
        rows = [r for r in rows if r["voice_name"] == args.voice]
    if args.script_id:
        wanted = set(args.script_id.split(","))
        rows = [r for r in rows if r["script_id"] in wanted]
    if args.overridden_only:
        rows = [r for r in rows if r["script_id"] in TTS_TEXT_OVERRIDES]
    rows.sort(key=lambda r: r["voice_name"])
    if args.limit:
        rows = rows[: args.limit]

    n_done = n_skipped = n_failed = 0
    for i, row in enumerate(rows, 1):
        script_id = row["script_id"]
        out_path = os.path.join(AUDIO_DIR, f"{script_id}.mp3")
        if os.path.exists(out_path) and not args.overwrite:
            n_skipped += 1
            continue
        try:
            text = TTS_TEXT_OVERRIDES.get(script_id, row["script_text"])
            audio, sample_rate = synthesize(text, row["voice_name"])
            write_mp3(audio, sample_rate, out_path)
        except Exception as e:
            print(f"[{i}/{len(rows)}] FAIL {script_id} ({row['voice_name']}): {e}")
            n_failed += 1
            continue
        n_done += 1
        print(f"[{i}/{len(rows)}] OK {script_id} ({row['voice_name']})")

    print(f"\nGenerated: {n_done}  |  skipped (already existed): {n_skipped}  |  failed: {n_failed}")


if __name__ == "__main__":
    main()
