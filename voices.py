"""
voices.py

Maps each of the dataset's 12 (accent, gender) voice slots to how its audio
is synthesized, now that the original ElevenLabs-based generation (no
longer redistributable as part of an open dataset -- see repo history) has
been replaced with two open-source engines:

  - Kokoro-82M (Apache 2.0): used wherever its built-in voice bank has a
    native match -- American and British English. It also covers the two
    slots originally sold as "Southern US" and "African-American" accents:
    Kokoro's voice bank doesn't model either as a distinct accent, and
    Common Voice (the source for every other cloned voice here) tracks
    neither a US sub-region nor a speaker's race/ethnicity, so there is no
    responsible way to source or label a voice for either claim. Those two
    slots are instead renamed "American Male 2" / "American Female 2" and
    given a second, distinct Kokoro American voice (not a reuse of
    "American Male/Female 1"), and the accent list is reported honestly as
    5 categories, not 7.

  - Chatterbox (MIT, zero-shot voice cloning) conditioned on a short
    reference clip for the remaining accents Kokoro doesn't cover
    natively: Australian, Irish, and Indian/South Asian. Reference clips
    come from Mozilla Common Voice accent-filtered subsets (CC0-1.0,
    contributors consented to reuse) -- see fetch_reference_clips.py. The
    "Indian/South Asian" label matches Common Voice's own accent tag
    ("India and South Asia (India, Pakistan, Sri Lanka)"), which spans more
    than just India -- the raw tag is preserved verbatim in
    reference_clips/manifest.json for anyone who wants the exact source
    category.

  - "Canadian Male 1" has no Common Voice accent tag of its own, so this
    slot was renamed to "British Male 2" and given its own Chatterbox
    reference clip (distinct from the Kokoro-voiced "British Male 1"),
    rather than inventing an unlabeled or misattributed accent.

pii_phi_voice_benchmark_dataset.csv records provenance in two columns:
tts_engine ("kokoro" or "chatterbox") and tts_voice_id (the Kokoro voice
ID, or the Common Voice clip ID for a cloned voice -- never a local
filename, and never a Common Voice client_id, which is a speaker-level
identifier this project does not publish). relabel_voice_metadata.py
computes both from this file plus reference_clips/manifest.json.
"""

import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MANIFEST_PATH = os.path.join(SCRIPT_DIR, "reference_clips", "manifest.json")

RENAMED_SLOTS = {
    # original (voice_name, voice_accent) -> new (voice_name, voice_accent)
    ("Canadian Male 1", "Canadian"): ("British Male 2", "British"),
    ("Southern US Male 1", "Southern US"): ("American Male 2", "American"),
    ("African-American Female 1", "African-American"): ("American Female 2", "American"),
    ("Indian-English Female 1", "Indian-English"): ("Indian/South Asian Female 1", "Indian/South Asian"),
    ("Indian-English Male 1", "Indian-English"): ("Indian/South Asian Male 1", "Indian/South Asian"),
}

VOICE_SYNTHESIS = {
    "American Female 1": {"engine": "kokoro", "voice": "af_heart"},
    "American Female 2": {"engine": "kokoro", "voice": "af_bella"},
    "American Male 1": {"engine": "kokoro", "voice": "am_michael"},
    "American Male 2": {"engine": "kokoro", "voice": "am_adam"},
    "British Female 1": {"engine": "kokoro", "voice": "bf_emma"},
    "British Male 1": {"engine": "kokoro", "voice": "bm_george"},
    "British Male 2": {"engine": "chatterbox", "reference_clip": "british_male_2.mp3"},
    "Australian Female 1": {"engine": "chatterbox", "reference_clip": "australian_female_1.mp3"},
    "Australian Male 1": {"engine": "chatterbox", "reference_clip": "australian_male_1.mp3"},
    "Irish Female 1": {"engine": "chatterbox", "reference_clip": "irish_female_1.mp3"},
    "Indian/South Asian Female 1": {"engine": "chatterbox", "reference_clip": "indian_south_asian_female_1.mp3"},
    "Indian/South Asian Male 1": {"engine": "chatterbox", "reference_clip": "indian_south_asian_male_1.mp3"},
}


def tts_engine_for(voice_name):
    return VOICE_SYNTHESIS[voice_name]["engine"]


def tts_voice_id_for(voice_name):
    """The Kokoro voice ID, or -- for a cloned voice -- the Common Voice
    clip ID its reference clip was sourced from (never a local filename
    or a speaker-level client_id)."""
    spec = VOICE_SYNTHESIS[voice_name]
    if spec["engine"] == "kokoro":
        return spec["voice"]
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)
    clip_path = manifest[voice_name]["clip_path"]
    return os.path.splitext(clip_path)[0]
