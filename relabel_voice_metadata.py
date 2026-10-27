"""
relabel_voice_metadata.py

One-time, idempotent rewrite of pii_phi_voice_benchmark_dataset.csv's
voice metadata columns (voice_name, voice_gender, voice_accent, voice_id)
to reflect the open-source regeneration pipeline (see voices.py), in place
of the original ElevenLabs-based values:

  - voice_id previously held an ElevenLabs voice ID; rewritten to the
    `kokoro:<voice>` / `chatterbox:<reference_clip>` identifier actually
    used to synthesize that row's audio.
  - "Canadian Male 1" / "Canadian" is renamed to "British Male 2" /
    "British" (voice_gender unchanged) -- see voices.py for why.

Safe to re-run: rows already matching the new voice_id values are left
untouched.

USAGE
    python3 relabel_voice_metadata.py
"""

import csv
import os

from voices import RENAMED_SLOTS, voice_id_for

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(SCRIPT_DIR, "pii_phi_voice_benchmark_dataset.csv")


def main():
    with open(DATASET_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    n_renamed = 0
    n_id_updated = 0
    for row in rows:
        key = (row["voice_name"], row["voice_accent"])
        if key in RENAMED_SLOTS:
            row["voice_name"], row["voice_accent"] = RENAMED_SLOTS[key]
            n_renamed += 1

        new_voice_id = voice_id_for(row["voice_name"])
        if row["voice_id"] != new_voice_id:
            row["voice_id"] = new_voice_id
            n_id_updated += 1

    with open(DATASET_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Relabeled {n_renamed} row(s) from Canadian Male 1 -> British Male 2")
    print(f"Updated voice_id on {n_id_updated} row(s)")


if __name__ == "__main__":
    main()
