"""Conservative, offset-preserving quality checks for public ABSA training rows.

Text is never rewritten: changing review characters would invalidate gold spans.
Diagnostic flags are distinct from the small set of automatic rejection rules.
"""
import html
import re
import unicodedata
from collections import Counter, defaultdict
from .text import tr_lower


URL = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
EMAIL = re.compile(r"(?i)\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b")
PHONE = re.compile(r"(?<!\d)(?:\+?90[\s.-]?)?0?5\d{2}[\s.-]?\d{3}[\s.-]?\d{2}[\s.-]?\d{2}(?!\d)")
HTML_TAG = re.compile(r"<\s*/?\s*[a-z][^>]*>", re.I)
HTML_ENTITY = re.compile(r"&(?:[a-z][a-z0-9]+|\#\d+|\#x[0-9a-f]+);", re.I)
ZERO_WIDTH = re.compile(r"[\u200b-\u200f\u2060\ufeff]")
REPEATED_PUNCT = re.compile(r"[!?.,;:…\-_=]{5,}")
REPEATED_CHAR = re.compile(r"([^\W\d_])\1{5,}", re.I | re.UNICODE)
PLACEHOLDERS = {"nan", "none", "null", "n/a", "na", "-", "...", "?"}


def canonical_text(text):
    """Normalize only for duplicate detection, never for model input or span offsets."""
    s = html.unescape(unicodedata.normalize("NFKC", str(text or "")))
    s = tr_lower(ZERO_WIDTH.sub("", s))
    return re.sub(r"\s+", " ", s).strip()


def quality_flags(text):
    text = str(text or "")
    flags = []
    for name, pattern in (("url", URL), ("email", EMAIL), ("phone_number", PHONE),
                          ("html_tag", HTML_TAG), ("html_entity", HTML_ENTITY),
                          ("zero_width", ZERO_WIDTH), ("repeated_punctuation", REPEATED_PUNCT),
                          ("elongated_word", REPEATED_CHAR)):
        if pattern.search(text):
            flags.append(name)
    if any(unicodedata.category(ch) == "Cc" and ch not in "\t\n\r" for ch in text):
        flags.append("control_character")
    if "\ufffd" in text:
        flags.append("replacement_character")
    if len(text) > 1000:
        flags.append("very_long_review")
    if text and sum(ch.isalpha() for ch in text) / len(text) < 0.25:
        flags.append("low_letter_ratio")
    if len(re.findall(r"\w+", text, re.UNICODE)) <= 2:
        flags.append("very_short_review")
    return flags


def rejection_reason(doc):
    text = doc.get("text")
    if not isinstance(text, str) or not text.strip():
        return "blank_text"
    if canonical_text(text) in PLACEHOLDERS:
        return "placeholder_text"
    if not doc.get("aspects"):
        return "no_valid_aspect"
    content = HTML_TAG.sub(" ", URL.sub(" ", EMAIL.sub(" ", text)))
    if sum(ch.isalpha() for ch in content) < 3:
        return "non_linguistic_text"
    for a in doc["aspects"]:
        span = a.get("span")
        start, end = a.get("start"), a.get("end")
        if span is None:
            if start is not None or end is not None:
                return "invalid_span"
        elif not (isinstance(start, int) and isinstance(end, int)
                  and 0 <= start < end <= len(text) and text[start:end] == span):
            return "invalid_span"
    return None


def clean_training(train, heldout):
    """Filter clear corruption and normalized overlap; leave held-out rows untouched."""
    held_keys = {canonical_text(d["text"]) for docs in heldout.values() for d in docs}
    seen = set()
    accepted = []
    dropped = Counter()
    source_before = Counter()
    source_after = Counter()
    flagged = Counter()
    example_ids = defaultdict(list)
    for d in train:
        source = d.get("source", "unknown")
        source_before[source] += 1
        for flag in quality_flags(d.get("text")):
            flagged[flag] += 1
        reason = rejection_reason(d)
        key = canonical_text(d.get("text"))
        if reason is None and key in held_keys:
            reason = "normalized_heldout_overlap"
        if reason is None and key in seen:
            reason = "normalized_train_duplicate"
        if reason:
            dropped[reason] += 1
            if len(example_ids[reason]) < 5:
                example_ids[reason].append(d.get("id"))
            continue
        seen.add(key)
        accepted.append(d)
        source_after[source] += 1
    report = {
        "policy": "Train only; no text rewriting or held-out filtering; diagnostic flags do not remove rows.",
        "counts": {"input": len(train), "accepted": len(accepted), "dropped": len(train) - len(accepted)},
        "dropped_by_reason": dict(dropped),
        "source_before": dict(source_before),
        "source_after": dict(source_after),
        "diagnostic_flags_before": dict(flagged),
        "example_ids_by_reason": dict(example_ids),
    }
    return accepted, report


def filter_near_heldout(train, heldout, threshold=0.9, min_chars=40):
    """Remove high-overlap train reviews using rare-shingle retrieval and exact Jaccard.

    This is a conservative candidate search, not a guarantee of semantic deduplication.
    Held-out documents are never changed.
    """
    def shingles(text):
        normalized = re.sub(r"[^\w\s]", "", canonical_text(text))
        return {normalized[i:i + 5] for i in range(len(normalized) - 4)}

    held_rows = [doc for split in heldout.values() for doc in split
                 if len(canonical_text(doc.get("text"))) >= min_chars]
    held_sets = [shingles(doc["text"]) for doc in held_rows]
    index = defaultdict(list)
    for i, grams in enumerate(held_sets):
        for gram in grams:
            index[gram].append(i)
    kept, matches = [], []
    for doc in train:
        normalized = canonical_text(doc.get("text"))
        if len(normalized) < min_chars:
            kept.append(doc)
            continue
        grams = shingles(normalized)
        candidates = set()
        for gram in sorted((g for g in grams if g in index), key=lambda g: len(index[g]))[:16]:
            candidates.update(index.get(gram, ()))
        best = None
        for i in candidates:
            other = held_sets[i]
            if min(len(grams), len(other)) / max(len(grams), len(other)) < threshold:
                continue
            intersection = len(grams & other)
            similarity = intersection / (len(grams) + len(other) - intersection)
            if similarity >= threshold and (best is None or similarity > best[1]):
                best = (held_rows[i].get("id"), similarity)
        if best is None:
            kept.append(doc)
        else:
            matches.append({"train_id": doc.get("id"), "heldout_id": best[0],
                            "char_5gram_jaccard": round(best[1], 4)})
    return kept, {"threshold": threshold, "min_chars": min_chars,
                  "candidate_search": "16 rarest held-out character five-grams",
                  "removed": len(matches), "examples": matches[:50]}
