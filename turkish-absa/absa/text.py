"""Turkish-aware normalization shared by data preparation and inference."""
import unicodedata


def tr_lower(value):
    text = unicodedata.normalize("NFC", str(value or ""))
    return text.replace("I", "ı").replace("İ", "i").lower().replace("\u0307", "")
