"""
voices.py

Maps each of the dataset's 12 (accent, gender) voice slots to how its audio
is synthesized, now that the original ElevenLabs-based generation (no
longer redistributable as part of an open dataset -- see repo history) has
been replaced with two open-source engines:

  - Kokoro-82M (Apache 2.0): used wherever its built-in voice bank has a
    native match -- American and British English. Also used for the
    "Southern US Male 1" and "African-American Female 1" slots, since
    neither a US-regional accent nor a speaker's race is a category Common
    Voice contributors self-report, so a Common Voice clip cannot be
    responsibly labeled with either; these two slots are instead given a
    distinct Kokoro American voice rather than reusing "American Male 1"
    / "American Female 1" verbatim.

  - Chatterbox (MIT, zero-shot voice cloning) conditioned on a short
    reference clip for the remaining accents Kokoro doesn't cover
    natively: Australian, Indian-English, and Irish. Reference clips come
    from Mozilla Common Voice accent-filtered subsets (CC0-1.0, contributors
    consented to reuse) -- see fetch_reference_clips.py.

  - "Canadian Male 1" has no Common Voice accent tag of its own, so this
    slot was renamed to "British Male 2" and given its own Chatterbox
    reference clip (distinct from the Kokoro-voiced "British Male 1"),
    rather than inventing an unlabeled or misattributed accent.

`voice_id` in pii_phi_voice_benchmark_dataset.csv previously held an
ElevenLabs voice ID; relabel_voice_metadata.py rewrites it to the engine
identifier below (and renames the Canadian slot), so the CSV's own
metadata stays accurate for the new synthesis pipeline.
"""

RENAMED_SLOTS = {
    # original (voice_name, voice_accent) -> new (voice_name, voice_accent)
    ("Canadian Male 1", "Canadian"): ("British Male 2", "British"),
}

VOICE_SYNTHESIS = {
    "American Female 1": {"engine": "kokoro", "voice": "af_heart"},
    "American Male 1": {"engine": "kokoro", "voice": "am_michael"},
    "British Female 1": {"engine": "kokoro", "voice": "bf_emma"},
    "British Male 1": {"engine": "kokoro", "voice": "bm_george"},
    "Southern US Male 1": {"engine": "kokoro", "voice": "am_adam"},
    "African-American Female 1": {"engine": "kokoro", "voice": "af_bella"},
    "Australian Female 1": {"engine": "chatterbox", "reference_clip": "australian_female_1.mp3"},
    "Australian Male 1": {"engine": "chatterbox", "reference_clip": "australian_male_1.mp3"},
    "Indian-English Female 1": {"engine": "chatterbox", "reference_clip": "indian_english_female_1.mp3"},
    "Indian-English Male 1": {"engine": "chatterbox", "reference_clip": "indian_english_male_1.mp3"},
    "Irish Female 1": {"engine": "chatterbox", "reference_clip": "irish_female_1.mp3"},
    "British Male 2": {"engine": "chatterbox", "reference_clip": "british_male_2.mp3"},
}


def voice_id_for(voice_name):
    spec = VOICE_SYNTHESIS[voice_name]
    if spec["engine"] == "kokoro":
        return f"kokoro:{spec['voice']}"
    return f"chatterbox:{spec['reference_clip']}"
