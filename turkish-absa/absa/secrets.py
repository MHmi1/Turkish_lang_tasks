"""Resolve optional Gemini credentials without storing values in project files."""
import os


NAMES = ("GEMINI_API_KEY", "GEMINI_API_KEY_1")


def gemini_key():
    for name in NAMES:
        value = os.environ.get(name)
        if value:
            return value
    try:
        from google.colab import userdata
    except ImportError:
        return None
    for name in NAMES:
        try:
            value = userdata.get(name)
        except Exception:
            continue
        if value:
            return value
    return None
