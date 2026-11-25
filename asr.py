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
    # max_new_tokens does NOT bound total output length against a decoder
    # repetition loop: on 4/545 clips (all Chatterbox-synthesized) the
    # decoder re-emits a single token for the rest of the clip instead of
    # terminating, producing transcripts of 500-1300+ words (confirmed via
    # word count vs. audio duration) and per-clip WER as high as 46.8.
    result = pipe(audio_path, max_new_tokens=440)
    return result["text"].strip()


# ---------------------------------------------------------------------------
# Oruk Fourier 2B (oruk/oruk-fourier-2b, CC BY-NC 4.0)
# ---------------------------------------------------------------------------

ORUK_FOURIER_2B_MODEL_ID = "oruk/oruk-fourier-2b"
ORUK_FOURIER_2B_REVISION = "8c2a8871f1304aa2f634da4fb95e24438c4945b9"
ORUK_FOURIER_2B_NATIVE_FILES = [
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
]
_oruk_fourier_model = None
_oruk_fourier_processor = None


def get_oruk_fourier_2b_model_and_processor():
    global _oruk_fourier_model, _oruk_fourier_processor
    if _oruk_fourier_model is None:
        from huggingface_hub import snapshot_download
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        print(f"Loading {ORUK_FOURIER_2B_MODEL_ID}@{ORUK_FOURIER_2B_REVISION[:8]} (native BF16)...")
        local_dir = snapshot_download(
            repo_id=ORUK_FOURIER_2B_MODEL_ID,
            revision=ORUK_FOURIER_2B_REVISION,
            allow_patterns=ORUK_FOURIER_2B_NATIVE_FILES,
        )
        _oruk_fourier_processor = AutoProcessor.from_pretrained(local_dir, local_files_only=True)
        _oruk_fourier_processor.tokenizer.padding_side = "left"
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        dtype = torch.bfloat16 if device in ("cuda", "mps") else torch.float32
        _oruk_fourier_model = (
            AutoModelForMultimodalLM.from_pretrained(
                local_dir, local_files_only=True, dtype=dtype
            )
            .to(device)
            .eval()
        )
    return _oruk_fourier_model, _oruk_fourier_processor


def transcribe_with_oruk_fourier_2b(audio_path, language_code="en"):
    import librosa
    import numpy as np
    import torch

    model, processor = get_oruk_fourier_2b_model_and_processor()
    audio, _ = librosa.load(audio_path, sr=16000, mono=True)
    audio = np.asarray(audio, dtype=np.float32)
    lang = "English" if language_code in ("en", "English") else language_code
    inputs = processor.apply_transcription_request(
        audio=[audio],
        language=[lang],
        return_tensors="pt",
        processor_kwargs={"padding": True},
    )
    inputs = inputs.to(model.device, dtype=model.dtype)
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=256)
    gen_ids = output_ids[:, inputs["input_ids"].shape[1] :]
    texts = processor.decode(gen_ids, return_format="transcription_only")
    text = texts[0] if isinstance(texts, list) and texts else str(texts)
    return text.strip()


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

ASR_CONDITIONS = ["cohere", "parakeet", "oruk_fourier_2b"]


def transcribe(asr_condition, audio_path, language_code="en"):
    if asr_condition == "cohere":
        return transcribe_with_cohere(audio_path, language_code)
    if asr_condition == "parakeet":
        return transcribe_with_parakeet(audio_path, language_code)
    if asr_condition == "oruk_fourier_2b":
        return transcribe_with_oruk_fourier_2b(audio_path, language_code)
    raise ValueError(f"Unknown ASR condition: {asr_condition!r}. Must be one of {ASR_CONDITIONS}")
