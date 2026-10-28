"""
relabel_voice_metadata.py

One-time, idempotent rewrite of pii_phi_voice_benchmark_dataset.csv's
voice/audio metadata to reflect the open-source regeneration pipeline
(see voices.py), in place of the original ElevenLabs-based values:

  - voice_name / voice_accent: renamed per RENAMED_SLOTS (see voices.py
    for why each rename happened -- e.g. "Southern US Male 1" and
    "African-American Female 1" both become a second American voice,
    since neither is a distinguishable accent in the new pipeline).
  - voice_id (an ElevenLabs voice ID) is replaced by two columns:
    tts_engine ("kokoro" or "chatterbox") and tts_voice_id (the Kokoro
    voice ID, or the Common Voice clip ID a cloned voice's reference
    clip came from -- never a local filename, and never a Common Voice
    client_id, which is a speaker-level identifier this project does
    not publish).
  - est_duration_sec (a pre-synthesis word-count estimate) is replaced
    by duration_sec, the actual measured duration of the regenerated
    clip in audio_en/, via ffprobe.

Safe to re-run: every row is recomputed from the current audio_en/ and
voices.py, so re-running after a clip is regenerated just refreshes its
duration_sec.

USAGE
    python3 relabel_voice_metadata.py
"""

import csv
import os
import subprocess

from voices import RENAMED_SLOTS, tts_engine_for, tts_voice_id_for

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(SCRIPT_DIR, "pii_phi_voice_benchmark_dataset.csv")
AUDIO_DIR = os.path.join(SCRIPT_DIR, "audio_en")

NEW_FIELDNAMES = [
    "category", "entity_class", "script_id", "script_text", "pii_value",
    "char_start", "char_end", "word_count", "duration_sec",
    "voice_name", "voice_gender", "voice_accent", "tts_engine", "tts_voice_id",
]


def measured_duration(script_id):
    audio_path = os.path.join(AUDIO_DIR, f"{script_id}.mp3")
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
        capture_output=True, text=True, check=True,
    )
    return round(float(result.stdout.strip()), 2)


def main():
    with open(DATASET_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    n_renamed = 0
    for row in rows:
        key = (row["voice_name"], row["voice_accent"])
        if key in RENAMED_SLOTS:
            row["voice_name"], row["voice_accent"] = RENAMED_SLOTS[key]
            n_renamed += 1

        row["tts_engine"] = tts_engine_for(row["voice_name"])
        row["tts_voice_id"] = tts_voice_id_for(row["voice_name"])
        row["duration_sec"] = measured_duration(row["script_id"])
        row.pop("voice_id", None)
        row.pop("est_duration_sec", None)

    with open(DATASET_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=NEW_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Relabeled {n_renamed} row(s)")
    print(f"Recomputed tts_engine / tts_voice_id / duration_sec for all {len(rows)} rows")


if __name__ == "__main__":
    main()
