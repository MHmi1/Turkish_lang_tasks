"""Gemini-based generation of Turkish PII edge cases with multi-model rotation & fast-fail.

Gemini writes text with inline markup `[[LABEL|surface]]`; offsets are derived deterministically
(no LLM-computed offsets). All outputs go through rule validation + dedup. TRAIN/VAL/TEST routing is
done by the caller. Uses multi-model rotation across verified Free Tier models with at most 1 fallback
model per request (never blocks in exponential backoff loops).
"""
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from .config import GEMINI_API_KEY, GEMINI_MODELS, LABELS
from .data import make_doc, text_hash
from .rules import _digits

ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent"
MARK = re.compile(r"\[\[([A-Z_]+)\|([^\[\]]+?)\]\]")

POLICY = """LABEL POLICY (follow strictly):
PERSON: private individual's name (first/last/full). NOT titles alone, NOT generic roles ("müşteri temsilcisi"), NOT company or bank names.
ADDRESS: street/neighbourhood-level postal address. A city or district alone is NOT an address.
PHONE: personal phone number in any format.  EMAIL: personal e-mail address.
DATE: a date tied to a specific person's record (birth date, personal appointment, personal delivery/payment date). Public calendar dates (holidays, campaign deadlines, announcements) are NOT PII.
URL: link containing a personal token, personal id or user-specific path. Public sites (turkiye.gov.tr, trendyol.com) are NOT PII.
ACCOUNT_NUMBER: customer / account / contract / policy / subscription / reference numbers that belong to a person.
SECRET: OTP codes, CVV, PIN, passwords, API keys/tokens.
TCKN: 11-digit Turkish national id.  VKN: 10-digit tax id.  IBAN: Turkish IBAN (TR..).  CARD: payment card number.  PLATE: vehicle plate.
Prices, quantities, product codes/barcodes, versions, HTTP codes, ticket numbers, percentages, branch names are NOT PII."""

CATEGORIES = {
    "suffix_boundary": "Entities followed by Turkish apostrophe case suffixes (Ahmet Yılmaz'ın, 05301234567'yi, TR12...'ye, kartım 4111...'dan). The markup span MUST EXCLUDE the apostrophe and suffix. Use all cases (gen/dat/loc/abl/acc/ins) and vowel-harmony variants; for numbers, suffix follows pronunciation of the last digit.",
    "format_variants": "Structured ids in diverse formatting: phone with spaces/dashes/parentheses/+90/0090, IBAN grouped/compact/lowercase, card with spaces/dashes, TCKN and VKN plain or space-grouped, plates with/without spaces, emails with plus-tags and Turkish transliterations. Use obviously fictional values.",
    "ocr_asr_noise": "Noisy text as from OCR or speech-to-text: broken diacritics (ı/i, ş/s), missing punctuation, digits separated by spaces or dots, merged words, wrong casing, stray characters, form-field style (ad:..;tc:..). Labels still mark the sensitive text exactly as it appears.",
    "hard_negatives": "Texts that LOOK sensitive but contain NO PII per the policy (11-digit order/stock-like codes next to non-id words, 10-digit barcodes, prices like 1.250,50 TL, public holiday dates, campaign deadlines, version numbers, public gov/shop URLs, generic roles, bank/product names, city names alone). Output NO markup at all for these.",
    "dense_multi_entity": "3-7 sensitive entities in one message (several persons, several ids, address, phone, email) incl. two different people with different ids in the same text.",
    "colloquial_codeswitch": "Informal, typo-ridden chat Turkish, slang, abbreviations, emojis, Turkish-English code-switching (e.g. 'mailimi update eder misiniz', 'kartım block olmuş').",
    "context_dependent": "Pairs of near-identical numbers/strings where context decides: e.g. 'müşteri numaram 48291736' (PII: ACCOUNT_NUMBER) vs 'stok kodu 48291736' (not PII); 'doğum tarihim 14.03.1988' (PII) vs 'kampanya 14.03.2026'a kadar' (not). Mix both kinds in the batch.",
    "secrets": "OTP/CVV/PIN/passwords/API tokens in chats, logs, config snippets, ticket comments, JSON logs. Use clearly fictional demo-looking tokens (never real provider key formats).",
    "address_variants": "Turkish postal addresses with abbreviations (Mah./Mh./Sok./Sk./Cd./Bulvarı/Apt./Blok/Kat/Daire/No), lowercase unpunctuated variants, building names, 'karşısı/yanı' landmarks, and negatives with only a city/district.",
}
SCENARIOS = ["banka müşteri hizmetleri", "kargo ve teslimat", "e-ticaret iade", "sigorta poliçesi", "telekom abonelik",
             "hastane randevu", "kira sözleşmesi", "fatura ve ödeme", "çağrı merkezi transkripti", "destek bileti",
             "sunucu logu", "e-posta gövdesi", "kimlik formu OCR", "sosyal medya DM", "avukat başvurusu"]

SCHEMA = {"type": "OBJECT", "properties": {"samples": {"type": "ARRAY", "items": {
    "type": "OBJECT", "properties": {"text": {"type": "STRING"}, "note": {"type": "STRING"}},
    "required": ["text"]}}}, "required": ["samples"]}


def build_prompt(category, scenarios, n):
    return f"""You generate TRAINING DATA for a Turkish PII-detection model.
Write {n} DIFFERENT, entirely FICTIONAL Turkish texts (30-400 characters each) in these scenarios: {', '.join(scenarios)}.
Never use real people, real accounts, real credentials. Use plausible but invented values.

FOCUS CATEGORY: {category} — {CATEGORIES[category]}

Mark every sensitive span inline as [[LABEL|exact text]] where LABEL is one of: {', '.join(LABELS)}.
Do not nest markers, do not put suffixes/apostrophes/punctuation inside a marker unless they are part of the value.
Everything that is not sensitive per the policy must stay unmarked.

{POLICY}

Example of the format (do not copy):
"Ben [[PERSON|Elif Kaya]], kimlik numaram [[TCKN|12345678950]]'dır, [[PHONE|0532 111 22 33]]'ten arayın."
"Kampanya 29 Ekim'e kadar geçerli, ürün kodu 4829173." (no markers)

Be diverse in length, tone, structure, punctuation, casing. Return JSON: {{"samples":[{{"text": "...marked text...", "note": "why this is tricky"}}]}}"""


def parse_markup(marked):
    out, spans, last, cur = [], [], 0, 0
    for m in MARK.finditer(marked):
        pre = marked[last:m.start()]
        out.append(pre)
        cur += len(pre)
        label, surf = m.group(1), m.group(2)
        if label not in LABELS:
            return None
        lead = len(surf) - len(surf.lstrip())
        s2 = surf.strip()
        if not s2:
            return None
        out.append(surf)
        spans.append([cur + lead, cur + lead + len(s2), label])
        cur += len(surf)
        last = m.end()
    out.append(marked[last:])
    text = "".join(out)
    if "[[" in text or "]]" in text:
        return None
    return text, spans


def _semantic_ok(label, s):
    d = len(_digits(s))
    if label == "EMAIL":
        return "@" in s and "." in s.split("@")[-1]
    if label == "IBAN":
        return re.fullmatch(r"TR\d{24}", re.sub(r"\s", "", s).upper()) is not None
    if label == "PHONE":
        return 10 <= d <= 13
    if label == "TCKN":
        return d == 11
    if label == "VKN":
        return d == 10
    if label == "CARD":
        return 13 <= d <= 19
    if label == "URL":
        return s.lower().startswith(("http", "www"))
    if label == "DATE":
        return d >= 2 or any(c.isalpha() for c in s)
    if label == "PLATE":
        return 1 <= len(s) <= 12 and d >= 2
    if label == "SECRET":
        return len(s) >= 3
    if label == "ACCOUNT_NUMBER":
        return d >= 4 or len(s) >= 5
    return len(s) >= 2


def validate(marked, category):
    p = parse_markup(marked)
    if p is None:
        return None
    text, spans = p
    if not 15 <= len(text) <= 700:
        return None
    if category == "hard_negatives":
        pass  # negatives may still contain a few true spans; keep whatever Gemini marked
    for s, e, l in spans:
        if not _semantic_ok(l, text[s:e]):
            return None
    if sum(c.isalpha() for c in text) < 8:
        return None
    return text, spans


class _ModelPool:
    """Thread-safe multi-model router with per-model rate spacing and instant fallback."""

    def __init__(self, models=None):
        raw_models = list(models or GEMINI_MODELS)
        self.models = []
        for m in raw_models:
            if m not in self.models:
                self.models.append(m)
        # Weight high-RPD Flash-Lite models 3x so they carry most of the load
        self.weighted = []
        for m in self.models:
            w = 3 if "flash-lite" in m else 1
            self.weighted.extend([m] * w)
        self.lock = threading.Lock()
        self.idx = 0
        self.dead = set()
        self.cooldown_until = {m: 0.0 for m in self.models}
        self.next_slot = {m: 0.0 for m in self.models}
        # Per-model spacing: ~3.2s for 15-RPM flash-lite models, ~6.0s for 5-RPM flash models
        self.interval = {
            m: (3.2 if "flash-lite" in m else 6.0)
            for m in self.models
        }

    def pick(self, exclude=None):
        """Pick the next available model and return (model_name, wait_seconds)."""
        now = time.time()
        with self.lock:
            alive = [m for m in self.models if m not in self.dead and m != exclude]
            if not alive:
                alive = [m for m in self.models if m != exclude] or list(self.models)
            for _ in range(len(self.weighted)):
                cand = self.weighted[self.idx % len(self.weighted)]
                self.idx += 1
                if cand in alive and self.cooldown_until.get(cand, 0.0) <= now:
                    wait = max(0.0, self.next_slot[cand] - now)
                    if wait <= 4.0:
                        self.next_slot[cand] = max(now, self.next_slot[cand]) + self.interval[cand]
                        return cand, wait
            best = min(alive, key=lambda m: max(self.cooldown_until.get(m, 0.0), self.next_slot.get(m, 0.0)))
            ready_at = max(now, self.cooldown_until.get(best, 0.0), self.next_slot.get(best, 0.0))
            wait = max(0.0, min(3.5, ready_at - now))
            self.next_slot[best] = now + wait + self.interval[best]
            return best, wait

    def mark_error(self, model, status_code, body_text=""):
        body_low = (body_text or "").lower()
        with self.lock:
            if status_code in (404, 400) and ("not found" in body_low or "no longer available" in body_low):
                self.dead.add(model)
            elif status_code == 429:
                if "perday" in body_low or "per_day" in body_low or "daily" in body_low:
                    self.dead.add(model)
                else:
                    self.cooldown_until[model] = time.time() + 12.0
            elif status_code in (500, 502, 503, 504):
                self.cooldown_until[model] = time.time() + 15.0


def _extract_samples(resp_json):
    cands = resp_json.get("candidates") or []
    if not cands:
        return None
    parts = cands[0].get("content", {}).get("parts") or []
    txt = "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()
    if not txt:
        return None
    if txt.startswith("```"):
        m = re.search(r"\{.*\}", txt, re.S)
        if m:
            txt = m.group(0)
    data = json.loads(txt)
    if isinstance(data, dict) and isinstance(data.get("samples"), list):
        return data["samples"]
    if isinstance(data, list):
        return data
    return None


def _send_once(prompt, key, model, temperature=0.95, timeout=25):
    """Single HTTP attempt to one model. Returns (samples_or_None, err_str_or_None)."""
    url = ENDPOINT.format(m=model)
    gen = {
        "temperature": temperature,
        "maxOutputTokens": 4096,
        "responseMimeType": "application/json",
        "responseSchema": SCHEMA,
    }
    body = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": gen}
    try:
        r = requests.post(
            url,
            headers={"x-goog-api-key": key, "Content-Type": "application/json"},
            json=body,
            timeout=timeout,
        )
    except requests.RequestException as e:
        return None, (0, f"net:{type(e).__name__}")
    if r.status_code == 200:
        try:
            samples = _extract_samples(r.json())
            if samples is not None:
                return samples, None
            return None, (200, "parse_empty")
        except Exception as e:
            return None, (200, f"json:{type(e).__name__}")
    return None, (r.status_code, r.text[:200])


def _call_with_one_fallback(prompt, key, pool, temperature=0.95):
    """Try primary model once; on failure, try at most 1 fallback model, then skip immediately."""
    m1, w1 = pool.pick()
    if w1 > 0:
        time.sleep(w1)
    samples, err1 = _send_once(prompt, key, m1, temperature=temperature)
    if samples is not None:
        return samples, m1, "ok"

    code1, msg1 = err1
    pool.mark_error(m1, code1, msg1)

    # Strict rule: try at most 1 alternative model, then move on immediately
    m2, w2 = pool.pick(exclude=m1)
    if w2 > 0:
        time.sleep(min(w2, 1.5))
    samples2, err2 = _send_once(prompt, key, m2, temperature=temperature)
    if samples2 is not None:
        return samples2, m2, f"fallback_from:{m1}({code1})"

    code2, msg2 = err2
    pool.mark_error(m2, code2, msg2)
    return [], f"{m1}->{m2}", f"skipped({m1}:{code1},{m2}:{code2})"


def generate(out_path, total_calls=120, per_call=10, workers=8, min_interval=0.5, seed=0, model=None, api_key=None):
    """Resumable: appends validated samples to out_path (jsonl) using multi-model rotation."""
    key = (api_key or os.environ.get("GEMINI_API_KEY", "") or GEMINI_API_KEY).strip()
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    models = [m.strip() for m in model.split(",") if m.strip()] if model else list(GEMINI_MODELS)
    pool = _ModelPool(models)
    rng = random.Random(seed)
    cats = list(CATEGORIES)
    seen, rows = set(), []
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            for l in f:
                if l.strip():
                    r = json.loads(l)
                    rows.append(r)
                    seen.add(text_hash(r["text"]))
    done_calls = {r.get("call") for r in rows if r.get("call") is not None}
    all_plan = [(k, c, rng.sample(SCENARIOS, 3)) for k, c in enumerate((cats * (total_calls // len(cats) + 1))[:total_calls])]
    todo = [(k, c, s) for k, c, s in all_plan if k not in done_calls]
    print(f"[gemini] models={pool.models} workers={workers} resume={len(rows)} samples, {len(todo)} calls to run", flush=True)
    if not todo:
        return {"ok": len(rows), "rejected": 0, "dup": 0, "skipped_calls": 0}

    lock = threading.Lock()
    stats = {"ok": 0, "rejected": 0, "dup": 0, "skipped_calls": 0}

    def work(call_id, cat, scs):
        prompt = build_prompt(cat, scs, per_call)
        samples, used_model, status = _call_with_one_fallback(prompt, key, pool, temperature=0.95)
        return call_id, cat, samples, used_model, status

    with ThreadPoolExecutor(max_workers=workers) as ex, open(out_path, "a", encoding="utf-8") as f:
        futs = [ex.submit(work, cid, c, s) for cid, c, s in todo]
        for fut in as_completed(futs):
            i, cat, res, used_model, status = fut.result()
            if not res:
                stats["skipped_calls"] += 1
            for it in res:
                if not isinstance(it, dict):
                    continue
                v = validate(it.get("text", ""), cat)
                if v is None:
                    stats["rejected"] += 1
                    continue
                text, spans = v
                h = text_hash(text)
                with lock:
                    if h in seen:
                        stats["dup"] += 1
                        continue
                    seen.add(h)
                    f.write(json.dumps({"text": text, "spans": spans, "category": cat, "call": i,
                                        "note": it.get("note", ""), "model": used_model}, ensure_ascii=False) + "\n")
                    f.flush()
                    stats["ok"] += 1
            print(f"[gemini] call {i} {cat} [{used_model} {status}]: total ok={stats['ok']} rej={stats['rejected']} dup={stats['dup']} skip={stats['skipped_calls']}", flush=True)
    return stats


def load(path):
    docs = []
    if not os.path.exists(path):
        return docs
    with open(path, encoding="utf-8") as f:
        for i, l in enumerate(f):
            if l.strip():
                r = json.loads(l)
                docs.append(make_doc(f"gem_{i}", r["text"], r["spans"], "gemini", r.get("category", "")))
    return docs
