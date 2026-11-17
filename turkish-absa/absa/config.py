"""Configuration for Turkish Aspect-Based Sentiment Analysis."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORK = Path(os.environ.get("ABSA_WORK", ROOT / "work"))
DATA = WORK / "data"
SPLITS = DATA / "splits"
SYNTH = DATA / "synth"
CACHE = WORK / "hf_cache"
MODELS = WORK / "models"
REPORTS = WORK / "reports"
LOGS = WORK / "logs"
EXPORT = WORK / "export"
for _p in (SPLITS, SYNTH, CACHE, MODELS, REPORTS, LOGS, EXPORT):
    _p.mkdir(parents=True, exist_ok=True)

BASE_MODEL = "ytu-ce-cosmos/modernbert-tr-base"
BASELINE_ATE = "ytu-ce-cosmos/modernbert-tr-absa-extraction"
BASELINE_APC = "ytu-ce-cosmos/modernbert-tr-absa-polarity"


def model_revision(path):
    """Return a run-pinned Hugging Face SHA for remote model repositories."""
    return json.loads(os.environ.get("ABSA_MODEL_REVISIONS", "{}")).get(str(path))


GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")

# Candidate models for the optional synthetic-data stage. Availability varies.
GEMINI_MODEL_POOL = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-3-flash",
    "gemini-3.6-flash",
    "gemini-3.8-flash",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
    "gemma-4-31b-it",
    "gemma-4-26b-it",
]

# ---- ATE (Aspect Term Extraction) BIO labels --------------------------------
ATE_BIO = ["O", "B-ASP", "I-ASP"]
ATE_L2I = {l: i for i, l in enumerate(ATE_BIO)}
ATE_I2L = {i: l for l, i in ATE_L2I.items()}

# ---- APC (Aspect Polarity Classification) labels ----------------------------
APC_LABELS = ["negative", "neutral", "positive", "not_mentioned"]
APC_L2I = {l: i for i, l in enumerate(APC_LABELS)}
APC_I2L = {i: l for l, i in APC_L2I.items()}

POL_MAP = {
    "negative": "negative", "negatif": "negative", "neg": "negative",
    "neutral": "neutral", "nötr": "neutral", "notr": "neutral", "neu": "neutral",
    "positive": "positive", "pozitif": "positive", "pos": "positive",
}

# ---- Public HF datasets -----------------------------------------------------
ABSA_TR_REPO = "ytu-ce-cosmos/absa-tr"
ABSA_TR_FILES = {"train": "train.jsonl", "val": "dev.jsonl", "test": "test.jsonl"}

M_ABSA_REPO = "Multilingual-NLP/M-ABSA"
M_ABSA_DOMAINS = ["phone", "laptop", "restaurant", "hotel", "food", "coursera", "sight"]
# Candidate aspects used for explicit+implicit classification. A 4th APC class
# (not_mentioned) is essential: neutral sentiment does not imply absent aspect.
IMPLICIT_CANDIDATES = ["ürün", "fiyat", "teslimat", "paketleme", "kalite", "performans", "batarya", "kamera", "ekran", "ses", "iade", "müşteri hizmetleri", "beden", "kumaş"]
ASPECT_ALIASES = {
    "kargo":"teslimat", "kargosu":"teslimat", "kargo firması":"teslimat",
    "teslimatı":"teslimat", "teslimat hızı":"teslimat",
    "bataryası":"batarya", "pil ömrü":"batarya", "şarj süresi":"batarya",
    "kamera":"kamera", "kameranın":"kamera", "kamerası":"kamera",
    "ekranı":"ekran", "ekran parlaklığı":"ekran",
    "ses kalitesi":"ses", "hoparlör":"ses", "fiyatlandırma":"fiyat",
    "kumaş kalitesi":"kumaş", "bedeni":"beden", "kalıbı":"beden",
    "teslimat süresi": "teslimat", "kargo süresi": "teslimat", "şarj": "batarya",
    "pil": "batarya", "pil ömrü": "batarya", "kamerası": "kamera", "fiyatı": "fiyat",
    "ürün kalitesi": "kalite", "kumaşı": "kumaş", "müşteri temsilcisi": "müşteri hizmetleri",
    "geri ödeme": "iade", "iade süreci": "iade", "kutusu": "paketleme",
}
