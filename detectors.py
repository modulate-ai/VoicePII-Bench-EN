"""
detectors.py

The three PII/PHI detection architectures evaluated in the paper:
  - GLiNER (nvidia/gliner-PII): zero-shot NER, full 109/109 entity coverage.
  - OpenPipe (PII-Redact-General): generative tagging, 76/109 coverage.
  - Piiranha (iiiorg/piiranha-v1-detect-personal-information): fixed-taxonomy
    token classifier, 26/109 coverage.

Each run_* function takes a text string and returns a list of
(start, end, type) character-offset spans in THAT text's own coordinates.
For ground-truth scoring, this is used directly. For ASR-condition
scoring, the caller maps these spans from transcript coordinates back to
original ground-truth coordinates (see paper_scoring.py's
build_alignment/map_span_to_original).

This module intentionally excludes every other detector explored during
the broader project (Presidio, OpenAI Privacy Filter, OpenMed, GLiNER2,
Azure Conversational PII) -- those are out of scope for this paper.
"""

import re
import difflib

# ---------------------------------------------------------------------------
# GLiNER (nvidia/gliner-PII) -- zero-shot NER
# ---------------------------------------------------------------------------

GLINER_MODEL_NAME = "nvidia/gliner-PII"
GLINER_LABEL_CHUNK_SIZE = 25  # too many simultaneous zero-shot labels can hurt accuracy

GLINER_ENTITY_MAP = {'First name': 'first name', 'Last name': 'last name', 'Full name': 'full name', 'Middle name': 'middle name', 'Middle initial': 'middle initial', 'Maiden name': 'maiden name', 'Prefix/Suffix': 'name title or suffix', 'Aliases': 'alias or assumed name', 'Nicknames': 'nickname', 'Personal email': 'email address', 'Phone number': 'phone number', 'Home address': 'home address', 'Zipcode': 'zip code', 'Postcode': 'postcode', 'State/Province': 'state or province', 'Country': 'country', 'Mailing address': 'mailing address', 'Social Security Number (SSN)': 'social security number', 'National Insurance Number (NIN)': 'national insurance number', 'National ID': 'national identification number', 'Tax ID': 'tax identification number', 'Passport Numbers': 'passport number', "Driver's license": "driver's license number", 'CA Social Insurance Number (SIN)': 'Canadian social insurance number', "AU Driver's license Number": "Australian driver's license number", 'AU Tax File Number': 'Australian tax file number', 'Business Number': 'business registration number', 'Company Number': 'company registration number', 'Credit/debit card': 'credit card number', 'Card expiration date': 'card expiration date', 'CVV': 'card security code', 'Bank account (US)': 'bank account number', 'Routing numbers (US)': 'bank routing number', 'Short code (UK)': 'UK bank sort code', 'Account number (UK)': 'UK bank account number', 'IBAN (EU)': 'IBAN number', 'Pins': 'PIN code', 'Account balances': 'account balance amount', 'Transaction history': 'transaction amount and date', 'Credit scores': 'credit score', 'Financial history': 'financial account history year', 'Loan/Mortgage details': 'loan or mortgage amount', 'Investment / Brokerage Account Info': 'brokerage account number', 'Bankruptcy or Debt Records': 'bankruptcy chapter or debt record', 'Cryptocurrency Wallet Address': 'cryptocurrency wallet address', 'Swift Code': 'SWIFT bank code', 'VAT Number': 'VAT registration number', 'Medical record number': 'medical record number', 'Health Insurance ID': 'health insurance ID number', 'NHS Number (UK)': 'NHS number', 'European Health Insurance Card (EHIC)': 'European health insurance card number', 'Global Health Insurance Card (GHIC) (UK)': 'UK global health insurance card number', 'Prescription details': 'prescription medication and dosage', 'Diagnosis': 'medical diagnosis', 'Treatment plans': 'medical treatment plan', 'Mental Health Records': 'mental health condition', 'Lab or Test Results': 'lab test result', 'Disability Status': 'disability condition', 'Vaccination Records': 'vaccine name', 'Substance Use History': 'substance use or addiction history', 'Genetic / DNA Information': 'genetic marker or gene name', 'Pregnancy or Reproductive Info': 'pregnancy or reproductive health information', 'Date of Birth': 'date of birth', 'Gender': 'gender', 'Blood Type': 'blood type', 'Employee ID': 'employee ID number', 'Employer name': 'employer company name', 'Previous employers': 'previous employer name', 'Job Title': 'job title', 'Salary details': 'salary amount', 'Bonus': 'bonus amount', 'RSU grant': 'RSU stock grant amount', 'Dates of employment': 'employment start date', 'Performance reviews': 'performance review rating', 'Manager/Supervisor Name': 'manager or supervisor name', 'Immigration / Work Authorization Status': 'immigration or work visa status', 'Professional License Number': 'professional license number', 'Username': 'username', 'Passwords': 'password', 'Device IDs': 'device ID number', 'IP Address': 'IP address', 'MAC Address': 'MAC address', 'File Name': 'file name', 'Policy numbers': 'insurance policy number', 'Claim details': 'insurance claim description', 'Beneficiary info': 'insurance beneficiary name', "Security Question — Mother's Maiden Name": "mother's maiden name", "Security Question — First Pet's Name": "first pet's name", 'Security Question — City You Grew Up In': 'childhood city', 'Security Question — High School Name': 'high school name', 'Security Question — Childhood Best Friend': 'childhood best friend name', 'One-Time Passcode (OTP)': 'one-time passcode', 'PIN': 'PIN code', 'Account Recovery Code': 'account recovery code', 'Age': 'age', 'Criminal History': 'criminal conviction or offense', 'Date': 'date', 'Date Range': 'date range', 'Event Name': 'event name', 'Location Coordinates': 'GPS coordinates', 'Marital Status': 'marital status', 'Nationality': 'nationality', 'Number Squence': 'reference number sequence', 'Physical Attribute': 'physical description', 'Political Affiliation': 'political party affiliation', 'Religion': 'religion', 'License Plate Number': 'license plate number', 'Vehicle Identification Number': 'vehicle identification number', 'Zodiac Sign (Astrological)': 'zodiac sign'}

_gliner_model = None


def get_gliner_model():
    global _gliner_model
    if _gliner_model is None:
        from gliner import GLiNER
        print(f"Loading {GLINER_MODEL_NAME} (first run downloads the model)...")
        _gliner_model = GLiNER.from_pretrained(GLINER_MODEL_NAME)
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
            _gliner_model = _gliner_model.to(device)
            print(f"GLiNER running on: {device}")
        except Exception as e:
            print(f"Could not check/set device acceleration ({e}) -- continuing on default device.")
    return _gliner_model


def run_gliner(text):
    """Queries GLiNER with ALL entity labels (chunked, since too many
    simultaneous zero-shot labels can hurt its own accuracy)."""
    model = get_gliner_model()
    all_labels = sorted(set(GLINER_ENTITY_MAP.values()))
    spans = []
    for i in range(0, len(all_labels), GLINER_LABEL_CHUNK_SIZE):
        chunk = all_labels[i:i + GLINER_LABEL_CHUNK_SIZE]
        entities = model.predict_entities(text, chunk, threshold=0.3)
        for e in entities:
            spans.append((e["start"], e["end"], e["label"]))
    return spans


# ---------------------------------------------------------------------------
# OpenPipe (PII-Redact-General) -- generative tagging model
# ---------------------------------------------------------------------------

OPENPIPE_ENTITY_MAP = {'First name': 'person_name', 'Last name': 'person_name', 'Full name': 'person_name', 'Middle name': 'person_name', 'Middle initial': 'person_name', 'Maiden name': 'person_name', 'Aliases': 'person_name', 'Nicknames': 'person_name', 'Personal email': 'email_address', 'Phone number': 'phone_number', 'Home address': 'street_address', 'Mailing address': 'street_address', 'Social Security Number (SSN)': 'personal_id', 'National Insurance Number (NIN)': 'personal_id', 'National ID': 'personal_id', 'Tax ID': 'personal_id', 'Passport Numbers': 'personal_id', "Driver's license": 'personal_id', 'CA Social Insurance Number (SIN)': 'personal_id', "AU Driver's license Number": 'personal_id', 'AU Tax File Number': 'personal_id', 'Business Number': 'other_id', 'Company Number': 'other_id', 'Credit/debit card': 'credit_card_info', 'Card expiration date': 'credit_card_info', 'CVV': 'credit_card_info', 'Bank account (US)': 'banking_number', 'Routing numbers (US)': 'banking_number', 'Short code (UK)': 'banking_number', 'Account number (UK)': 'banking_number', 'IBAN (EU)': 'banking_number', 'Swift Code': 'banking_number', 'Pins': 'secure_credential', 'Cryptocurrency Wallet Address': 'other_id', 'VAT Number': 'other_id', 'Medical record number': 'personal_id', 'Health Insurance ID': 'personal_id', 'NHS Number (UK)': 'personal_id', 'European Health Insurance Card (EHIC)': 'personal_id', 'Global Health Insurance Card (GHIC) (UK)': 'personal_id', 'Prescription details': 'medical_condition', 'Diagnosis': 'medical_condition', 'Treatment plans': 'medical_condition', 'Mental Health Records': 'medical_condition', 'Lab or Test Results': 'medical_condition', 'Disability Status': 'medical_condition', 'Vaccination Records': 'medical_condition', 'Substance Use History': 'medical_condition', 'Genetic / DNA Information': 'medical_condition', 'Pregnancy or Reproductive Info': 'medical_condition', 'Date of Birth': 'date_of_birth', 'Gender': 'gender', 'Employee ID': 'other_id', 'Employer name': 'organization_name', 'Previous employers': 'organization_name', 'Dates of employment': 'date', 'Manager/Supervisor Name': 'person_name', 'Professional License Number': 'personal_id', 'Passwords': 'password', 'Device IDs': 'other_id', 'Policy numbers': 'other_id', 'Beneficiary info': 'person_name', "Security Question — Mother's Maiden Name": 'person_name', 'Security Question — High School Name': 'organization_name', 'Security Question — Childhood Best Friend': 'person_name', 'One-Time Passcode (OTP)': 'secure_credential', 'PIN': 'secure_credential', 'Account Recovery Code': 'secure_credential', 'Age': 'age', 'Date': 'date', 'Date Range': 'date', 'Nationality': 'nationality', 'Number Squence': 'other_id', 'Religion': 'religious_affiliation', 'License Plate Number': 'other_id', 'Vehicle Identification Number': 'other_id'}

OPENPIPE_TAG_RE = re.compile(r"<PII:([\w]+)>(.*?)</PII:\1>", re.DOTALL)
_WORD_RE = re.compile(r"[\w']+", re.UNICODE)


def _tokenize_words_with_offsets(text):
    spans = [(m.start(), m.end()) for m in _WORD_RE.finditer(text)]
    words = [text[s:e].lower() for s, e in spans]
    return words, spans


def _align_tags_to_original(original, tagged_text):
    """Word-level fuzzy alignment: tokenize both into words, replace each
    tag with a sentinel so its position survives tokenization, align via
    difflib, and derive each tag's span in the ORIGINAL text's coordinates
    from the word-alignment opcodes. Returns [] if no tags, None if the
    tagged text is too different from the original to trust (see
    Appendix on OpenPipe alignment in the paper)."""
    tags_found = list(OPENPIPE_TAG_RE.finditer(tagged_text))
    if not tags_found:
        return []

    tag_types = [m.group(1) for m in tags_found]
    sentinel_text = tagged_text
    for idx in range(len(tags_found) - 1, -1, -1):
        m = tags_found[idx]
        sentinel_text = sentinel_text[:m.start()] + f" \x00TAG{idx}\x00 " + sentinel_text[m.end():]

    orig_words, orig_spans = _tokenize_words_with_offsets(original)
    combined_re = re.compile(r"\x00TAG\d+\x00|[\w']+", re.UNICODE)
    sentinel_re = re.compile(r"\x00TAG(\d+)\x00")

    red_tokens = combined_re.findall(sentinel_text)
    red_words, red_is_tag, red_tag_idx = [], [], []
    for tok in red_tokens:
        sm = sentinel_re.fullmatch(tok)
        if sm:
            red_words.append(tok)
            red_is_tag.append(True)
            red_tag_idx.append(int(sm.group(1)))
        else:
            red_words.append(tok.lower())
            red_is_tag.append(False)
            red_tag_idx.append(None)

    matcher = difflib.SequenceMatcher(None, orig_words, red_words, autojunk=False)
    opcodes = matcher.get_opcodes()

    total_orig_words = len(orig_words)
    matched_orig_words = sum((i2 - i1) for tag, i1, i2, j1, j2 in opcodes if tag == "equal")
    if total_orig_words > 0 and matched_orig_words / total_orig_words < 0.3:
        return None

    spans_by_tag_idx = {}
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            continue
        sentinel_positions = [j for j in range(j1, j2) if red_is_tag[j]]
        if not sentinel_positions:
            continue
        n_sent = len(sentinel_positions)
        n_orig = i2 - i1
        if n_orig == 0:
            boundary = orig_spans[i1 - 1][1] if i1 > 0 else (orig_spans[0][0] if orig_spans else 0)
            for j in sentinel_positions:
                spans_by_tag_idx[red_tag_idx[j]] = (boundary, boundary)
            continue
        chunk = max(1, n_orig // n_sent)
        for k, j in enumerate(sentinel_positions):
            start_word = min(i1 + k * chunk, i2 - 1)
            end_word = i2 if k == n_sent - 1 else min(i1 + (k + 1) * chunk, i2)
            end_word = max(end_word, start_word + 1)
            spans_by_tag_idx[red_tag_idx[j]] = (orig_spans[start_word][0], orig_spans[end_word - 1][1])

    spans = []
    for idx, tag_type in enumerate(tag_types):
        if idx in spans_by_tag_idx:
            s, e = spans_by_tag_idx[idx]
            spans.append((s, e, tag_type))
    return spans


_openpipe_device = None


def get_openpipe_device():
    global _openpipe_device
    if _openpipe_device is None:
        try:
            import torch
            if torch.cuda.is_available():
                _openpipe_device = "cuda"
            elif torch.backends.mps.is_available():
                _openpipe_device = "mps"
            else:
                _openpipe_device = "cpu"
        except Exception:
            _openpipe_device = "cpu"
        print(f"OpenPipe pii-redaction running on: {_openpipe_device}")
    return _openpipe_device


def run_openpipe(text):
    from pii_redaction import tag_pii_in_documents, PIIHandlingMode
    tagged = tag_pii_in_documents([text], device=get_openpipe_device(), mode=PIIHandlingMode.TAG)[0]
    spans = _align_tags_to_original(text, tagged)
    return spans if spans else []


# ---------------------------------------------------------------------------
# Piiranha (iiiorg/piiranha-v1-detect-personal-information) -- fixed-taxonomy
# token classifier
# ---------------------------------------------------------------------------

PIIRANHA_MODEL_NAME = "iiiorg/piiranha-v1-detect-personal-information"

PIIRANHA_ENTITY_MAP = {'First name': 'GIVENNAME', 'Last name': 'SURNAME', 'Full name': ['GIVENNAME', 'SURNAME'], 'Personal email': 'EMAIL', 'Phone number': 'TELEPHONENUM', 'Social Security Number (SSN)': 'SOCIALNUM', 'National Insurance Number (NIN)': 'SOCIALNUM', 'CA Social Insurance Number (SIN)': 'SOCIALNUM', 'Credit/debit card': 'CREDITCARDNUMBER', 'Date of Birth': 'DATEOFBIRTH', "Driver's license": 'DRIVERLICENSENUM', "AU Driver's license Number": 'DRIVERLICENSENUM', 'Bank account (US)': 'ACCOUNTNUM', 'Account number (UK)': 'ACCOUNTNUM', 'Tax ID': 'TAXNUM', 'AU Tax File Number': 'TAXNUM', 'VAT Number': 'TAXNUM', 'National ID': 'IDCARDNUM', 'Passport Numbers': 'IDCARDNUM', 'Passwords': 'PASSWORD', 'Username': 'USERNAME', 'Home address': 'STREET', 'Mailing address': 'STREET', 'Security Question — City You Grew Up In': 'CITY', 'Zipcode': 'ZIPCODE', 'Postcode': 'ZIPCODE'}

_piiranha_pipeline = None


def get_piiranha_pipeline():
    global _piiranha_pipeline
    if _piiranha_pipeline is None:
        from transformers import pipeline
        import torch
        if torch.cuda.is_available():
            device = 0
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = -1
        print(f"Loading {PIIRANHA_MODEL_NAME} (first run downloads the model)...")
        _piiranha_pipeline = pipeline("token-classification", model=PIIRANHA_MODEL_NAME,
                                       aggregation_strategy="simple", device=device)
        print(f"Piiranha running on device: {device}")
    return _piiranha_pipeline


def run_piiranha(text):
    """Returns Piiranha's RAW spans, uncorrected -- see paper_scoring.py's
    trimmed_strict metric for the whitespace/punctuation-offset
    correction, kept as a separate, visible metric for transparency."""
    pipe = get_piiranha_pipeline()
    results = pipe(text)
    return [(r["start"], r["end"], r["entity_group"]) for r in results]


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

DETECTORS = ["gliner", "openpipe", "piiranha"]

ENTITY_MAPS = {
    "gliner": {k: [v] if not isinstance(v, list) else v for k, v in GLINER_ENTITY_MAP.items()},
    "openpipe": {k: [v] if not isinstance(v, list) else v for k, v in OPENPIPE_ENTITY_MAP.items()},
    "piiranha": {k: [v] if not isinstance(v, list) else v for k, v in PIIRANHA_ENTITY_MAP.items()},
}


def run_detector(detector, text):
    if detector == "gliner":
        return run_gliner(text)
    if detector == "openpipe":
        return run_openpipe(text)
    if detector == "piiranha":
        return run_piiranha(text)
    raise ValueError(f"Unknown detector: {detector!r}. Must be one of {DETECTORS}")
