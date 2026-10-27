"""
fetch_reference_clips.py

Sources one short reference clip per (accent, gender) slot that Kokoro
cannot cover natively (see generate_audio_en.py for the full voice map),
for use as a zero-shot voice-cloning prompt in Chatterbox.

Clips come from Mozilla Common Voice accent-filtered subsets distributed
via Mozilla Data Collective (CC0-1.0) -- each subset keeps only rows whose
self-declared `accents` field matches a given English regional accent.
"British Male 2" (standing in for the dataset's original "Canadian Male 1"
slot, which has no Common Voice accent-tagged equivalent) is additionally
filtered to rows tagged "England English", since the British subset itself
bundles England/Scottish/Welsh/Northern-Irish accents together.

Per MDC's terms for these datasets, the source corpus itself must not be
rehosted: this script downloads each subset to mdc_cache/ (gitignored),
extracts only the one clip selected per slot into reference_clips/ (also
gitignored), and never commits or redistributes the wider corpus.

Candidate selection is deterministic (not random) for reproducibility:
among validated, adult-age, zero-downvote rows with a sentence length in
a range long enough to give the cloning model real phonetic coverage, it
picks the longest sentence, breaking ties on clip path.

USAGE
    python3 fetch_reference_clips.py                       # fetch all slots
    python3 fetch_reference_clips.py --slot "Irish Female 1"  # single slot
"""

import argparse
import csv
import json
import os
import tarfile

from dotenv import load_dotenv

load_dotenv()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(SCRIPT_DIR, "mdc_cache")
OUT_DIR = os.path.join(SCRIPT_DIR, "reference_clips")
MANIFEST_PATH = os.path.join(OUT_DIR, "manifest.json")

MIN_SENTENCE_CHARS = 40
MAX_SENTENCE_CHARS = 160
ADULT_AGE_BRACKETS = {"twenties", "thirties", "fourties", "fifties", "sixties"}

REFERENCE_SLOTS = {
    "Australian Female 1": {
        "dataset_id": "cmrt710620013mm071t45y6wb",
        "dataset_name": "Common Voice Scripted Speech 26.0 - Australian English",
        "gender": "female_feminine",
    },
    "Australian Male 1": {
        "dataset_id": "cmrt710620013mm071t45y6wb",
        "dataset_name": "Common Voice Scripted Speech 26.0 - Australian English",
        "gender": "male_masculine",
    },
    "Indian-English Female 1": {
        "dataset_id": "cmrt70sar001umm07jwxzhw89",
        "dataset_name": "Common Voice Scripted Speech 26.0 - South Asian English",
        "gender": "female_feminine",
    },
    "Indian-English Male 1": {
        "dataset_id": "cmrt70sar001umm07jwxzhw89",
        "dataset_name": "Common Voice Scripted Speech 26.0 - South Asian English",
        "gender": "male_masculine",
    },
    "Irish Female 1": {
        "dataset_id": "cmrt71fck001bmm07cj9h312i",
        "dataset_name": "Common Voice Scripted Speech 26.0 - Irish English",
        "gender": "female_feminine",
    },
    "British Male 2": {
        "dataset_id": "cmrt6zrob000zmm07yqwjlpwi",
        "dataset_name": "Common Voice Scripted Speech 26.0 - British English",
        "gender": "male_masculine",
        "accent_substring": "England English",
    },
}


def ensure_archive(dataset_id):
    import datacollective
    return datacollective.download_dataset(dataset_id, download_directory=CACHE_DIR)


def load_metadata_rows(archive_path):
    with tarfile.open(archive_path, "r:gz") as tar:
        rows = []
        for split in ("train.tsv", "dev.tsv", "test.tsv"):
            member = tar.extractfile(split)
            text = member.read().decode("utf-8").splitlines()
            rows.extend(csv.DictReader(text, delimiter="\t"))
    return rows


def pick_candidate(rows, gender, accent_substring=None):
    candidates = [
        r for r in rows
        if r["gender"] == gender
        and r["age"] in ADULT_AGE_BRACKETS
        and r["down_votes"] == "0"
        and int(r["up_votes"]) >= 2
        and MIN_SENTENCE_CHARS <= len(r["sentence"]) <= MAX_SENTENCE_CHARS
        and (accent_substring is None or accent_substring in r["accents"])
    ]
    if not candidates:
        raise RuntimeError(f"No candidate rows matched gender={gender!r} accent_substring={accent_substring!r}")
    candidates.sort(key=lambda r: (-len(r["sentence"]), r["path"]))
    return candidates[0]


def extract_clip(archive_path, clip_path, out_path):
    with tarfile.open(archive_path, "r:gz") as tar:
        member = tar.extractfile(f"clips/{clip_path}")
        with open(out_path, "wb") as f:
            f.write(member.read())


def fetch_slot(voice_name, spec, manifest):
    slug = voice_name.lower().replace(" ", "_").replace("-", "_")
    out_path = os.path.join(OUT_DIR, f"{slug}.mp3")
    if os.path.exists(out_path):
        print(f"SKIP {voice_name}: {out_path} already exists")
        return

    archive_path = ensure_archive(spec["dataset_id"])
    rows = load_metadata_rows(archive_path)
    candidate = pick_candidate(rows, spec["gender"], spec.get("accent_substring"))
    extract_clip(archive_path, candidate["path"], out_path)

    manifest[voice_name] = {
        "source_dataset": spec["dataset_name"],
        "source_dataset_id": spec["dataset_id"],
        "clip_path": candidate["path"],
        "sentence_id": candidate["sentence_id"],
        "sentence": candidate["sentence"],
        "age": candidate["age"],
        "accents": candidate["accents"],
        "license": "CC0-1.0",
    }
    print(f"OK {voice_name}: {out_path} (from {candidate['path']!r}, {len(candidate['sentence'])} chars)")


def main():
    parser = argparse.ArgumentParser(description="Fetch Chatterbox reference clips from Mozilla Data Collective.")
    parser.add_argument("--slot", type=str, default=None, choices=list(REFERENCE_SLOTS),
                         help="Only fetch this one voice slot (default: all).")
    args = parser.parse_args()

    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    manifest = {}
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, encoding="utf-8") as f:
            manifest = json.load(f)

    slots = {args.slot: REFERENCE_SLOTS[args.slot]} if args.slot else REFERENCE_SLOTS
    for voice_name, spec in slots.items():
        fetch_slot(voice_name, spec, manifest)

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
