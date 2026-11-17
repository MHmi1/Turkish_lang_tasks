"""Central configuration: paths, taxonomy, dataset sources, label maps."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORK = Path(os.environ.get("PII_WORK", ROOT / "work"))
DATA = WORK / "data"
SPLITS = DATA / "splits"
SYNTH = DATA / "synth"
CACHE = WORK / "hf_cache"
MODELS = WORK / "models"
REPORTS = WORK / "reports"
PLOTS = REPORTS / "plots"
LOGS = WORK / "logs"
EXPORT = WORK / "export"
for _p in (SPLITS, SYNTH, CACHE, MODELS, REPORTS, PLOTS, LOGS, EXPORT):
    _p.mkdir(parents=True, exist_ok=True)

BASE_MODEL = "ytu-ce-cosmos/modernbert-tr-base"
BASELINE_MODEL = "ytu-ce-cosmos/modernbert-tr-pii-ner"

# Gemini credentials are read from the environment at runtime.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()

# Candidate models for optional synthetic generation. Availability and quotas vary.
DEFAULT_GEMINI_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.1-flash-lite-preview",
    "gemini-flash-lite-latest",
    "gemini-3-flash-preview",
    "gemini-3.5-flash",
    "gemini-3.6-flash",
]
_env_models = [m.strip() for m in os.environ.get("GEMINI_MODELS", "").split(",") if m.strip()]
GEMINI_MODELS = _env_models if _env_models else DEFAULT_GEMINI_MODELS
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", GEMINI_MODELS[0])

# ---- unified taxonomy -------------------------------------------------------
LABELS = ["PERSON", "ADDRESS", "PHONE", "EMAIL", "DATE", "URL", "ACCOUNT_NUMBER",
          "SECRET", "TCKN", "VKN", "IBAN", "CARD", "PLATE"]
BIO = ["O"] + [x for l in LABELS for x in (f"B-{l}", f"I-{l}")]
L2I = {l: i for i, l in enumerate(BIO)}
I2L = {i: l for l, i in L2I.items()}
STRUCT_LABELS = {"TCKN", "VKN", "IBAN", "CARD", "PHONE", "EMAIL", "ACCOUNT_NUMBER"}

# ---- public sources ---------------------------------------------------------
_PRIV = {"private_person": "PERSON", "private_address": "ADDRESS", "private_phone": "PHONE",
         "private_email": "EMAIL", "private_date": "DATE", "private_url": "URL",
         "account_number": "ACCOUNT_NUMBER", "secret": "SECRET",
         "tckn": "TCKN", "vkn": "VKN", "iban": "IBAN"}
SOURCES = {
    "btx24": {"repo": "BTX24/turkish-privacy-pii-ner", "map": _PRIV,
              "files": {"train": "train.jsonl", "val": "validation.jsonl", "test": "test.jsonl"},
              "license": "CC-BY-4.0"},
    "yusuf": {"repo": "yusuf-said/turkish-privacy-filter-dataset", "map": _PRIV,
              "files": {"all": "tr_privacy_tr_curated.jsonl"}, "license": "MIT"},
    "belgin": {"repo": "negentropi/belgin-pii-dataset", "map": _PRIV,
               "files": {"train": "data/train.jsonl", "val": "data/validation.jsonl",
                         "test": "data/test.jsonl"}, "license": "CC-BY-4.0"},
    "fevzi": {"repo": "fevziegeyurtsevenler/turkish-pii-corpus",
              "map": {"PHONE": "PHONE", "IBAN": "IBAN", "VKN": "VKN", "TCKN": "TCKN",
                      "CREDIT_CARD": "CARD", "PLAKA": "PLATE", "EMAIL": "EMAIL"},
              "files": {"all": "pii_corpus.jsonl"}, "license": "Apache-2.0"},
}
# external long-document test (never trained on); labels outside our taxonomy are ignored
NM_REPO = "newmindai/nm-kvkk-pii-6K"
NM_FILE = "spans/test_hard.jsonl"
NM_MAP = {"full_name": "PERSON", "last_name": "PERSON", "full_address": "ADDRESS",
          "phone_number": "PHONE", "email_address": "EMAIL", "date_of_birth": "DATE",
          "url_with_pii": "URL", "national_id_number": "TCKN", "tax_id_number": "VKN",
          "company_tax_number": "VKN", "iban": "IBAN", "card_number": "CARD",
          "license_plate": "PLATE", "password": "SECRET", "session_or_token": "SECRET"}

# pretrained YTU baseline label names -> unified
YTU_MAP = {"KISI_AD_SOYAD": "PERSON", "TCKN": "TCKN", "TELEFON": "PHONE", "EMAIL": "EMAIL",
           "ADRES": "ADDRESS", "DOGUM_TARIHI": "DATE", "PLAKA": "PLATE", "IBAN_TR": "IBAN",
           "KART_NO": "CARD", "HESAP_NO": "ACCOUNT_NUMBER", "VKN": "VKN"}
