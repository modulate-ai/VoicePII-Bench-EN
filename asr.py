"""
asr.py

The two ASR conditions evaluated in the paper, alongside ground-truth text
(which needs no transcription step at all):
  - Cohere Transcribe (cohere-transcribe-03-2026): 6.49% WER on our audio.
  - NVIDIA Parakeet TDT (parakeet-tdt-0.6b-v3): 20.64% WER on our audio.

Both were selected from top-ranked entries on the Hugging Face Open ASR
Leaderboard (August 2026), chosen to differ substantially in accuracy.

This module intentionally excludes Whisper, used during earlier
exploratory work but not part of this paper's final ASR comparison.
"""

# ---------------------------------------------------------------------------
# Cohere Transcribe
# ---------------------------------------------------------------------------

COHERE_TRANSCRIBE_MODEL_ID = "CohereLabs/cohere-transcribe-03-2026"
# Gated model -- requires accepting the license on the HF model page and
# `huggingface-cli login` before first use. Requires transformers>=5.4.0.
# `language` is a PREPROCESSING parameter (passed to the processor), not a
# generate() kwarg -- confirmed via direct testing; Cohere Transcribe does
# no language auto-detection at all.
_cohere_transcribe_model = None
_cohere_transcribe_processor = None


def get_cohere_transcribe_model_and_processor():
    global _cohere_transcribe_model, _cohere_transcribe_processor
    if _cohere_transcribe_model is None:
        from transformers import AutoProcessor, CohereAsrForConditionalGeneration
        print(f"Loading {COHERE_TRANSCRIBE_MODEL_ID} (first run downloads the model; "
              f"gated -- run `huggingface-cli login` first if this fails with an auth error)...")
        _cohere_transcribe_processor = AutoProcessor.from_pretrained(COHERE_TRANSCRIBE_MODEL_ID)
        _cohere_transcribe_model = CohereAsrForConditionalGeneration.from_pretrained(
            COHERE_TRANSCRIBE_MODEL_ID, device_map="auto")
    return _cohere_transcribe_model, _cohere_transcribe_processor


def transcribe_with_cohere(audio_path, language_code="en"):
    from transformers.audio_utils import load_audio
    model, processor = get_cohere_transcribe_model_and_processor()
    audio = load_audio(audio_path, sampling_rate=16000)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", language=language_code)
    inputs = inputs.to(model.device, dtype=model.dtype)
    outputs = model.generate(**inputs, max_new_tokens=256)
    text = processor.decode(outputs, skip_special_tokens=True)
    # processor.decode() returns a list of strings (batch-decode behavior),
    # even for a batch size of 1 -- confirmed via direct testing.
    if isinstance(text, list):
        text = text[0] if text else ""
    return text.strip()


# ---------------------------------------------------------------------------
# NVIDIA Parakeet TDT
# ---------------------------------------------------------------------------

PARAKEET_MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"
# Works with the standard transformers ASR pipeline -- no special language
# handling needed for English (this paper's scope), no gating.
_parakeet_pipeline = None


def get_parakeet_pipeline():
    global _parakeet_pipeline
    if _parakeet_pipeline is None:
        from transformers import pipeline
        import torch
        device = 0 if torch.cuda.is_available() else -1
        print(f"Loading {PARAKEET_MODEL_ID} (first run downloads the model)...")
        _parakeet_pipeline = pipeline(task="automatic-speech-recognition",
                                       model=PARAKEET_MODEL_ID, device=device)
    return _parakeet_pipeline


def transcribe_with_parakeet(audio_path, language_code="en"):
    pipe = get_parakeet_pipeline()
    result = pipe(audio_path, max_new_tokens=440)
    return result["text"].strip()


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

ASR_CONDITIONS = ["cohere", "parakeet"]


def transcribe(asr_condition, audio_path, language_code="en"):
    if asr_condition == "cohere":
        return transcribe_with_cohere(audio_path, language_code)
    if asr_condition == "parakeet":
        return transcribe_with_parakeet(audio_path, language_code)
    raise ValueError(f"Unknown ASR condition: {asr_condition!r}. Must be one of {ASR_CONDITIONS}")
