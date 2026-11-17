"""Gemini-powered generator for Turkish ABSA edge cases:
sarcasm/irony, heavy Turkish idioms ("kazık yedik", "elimizde patladı", "saç baş yoldurdu"),
Gen-Z / social media slang, formal corporate complaints, contrastive multi-aspect, and implicit aspects.

Uses inline markup `[[ASP|surface|polarity]]` for explicit aspects so character offsets are derived
deterministically in Python (never guessed by the LLM).
"""
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from .config import GEMINI_MODEL, GEMINI_MODEL_POOL, POL_MAP
from .data import locate_spans, text_hash

ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent"
MODELS_LIST_URL = "https://generativelanguage.googleapis.com/v1beta/models"
MARK = re.compile(r"\[\[ASP\|([^|\[\]]+?)\|([a-zA-ZöüşığçÖÜŞİĞÇ]+)\]\]")

CATEGORIES = {
    "heavy_sarcasm_irony": (
        "Biting Turkish sarcasm & irony where surface words sound positive ('Harika(!)', 'Maşallah', 'Bayıldım', "
        "'Bravo', 'Çok hızlı(!)') but the true sentiment toward the aspect is strongly NEGATIVE (or vice versa)."
    ),
    "turkish_idioms_metaphors": (
        "Rich Istanbul Turkish idioms & metaphorical expressions where literal translation fails: "
        "'kazık yedik', 'elimizde patladı', 'paramız çöp oldu', 'saç baş yoldurdu', 'kan kusturdu', 'ömür törpüsü', "
        "'göz boyama', 'taş çıkartır', 'ateş ediyor', 'yağ gibi akıyor', 'kafa ütülüyor', 'teneke gibi', 'taş gibi'."
    ),
    "gen_z_social_slang": (
        "Gen-Z & Twitter/TikTok/Trendyol youth slang, abbreviations, emojis, and code-switching: "
        "'efso', 'aşırı iyi', 'çöp', 'f/p kralı', 'şaka mı', 'cringe', 'vibe', 'buglı', 'drop yiyor', 'uçtu resmen', "
        "'10/10', 'baya iyi', 'yok artık', lowercase/typo-heavy casual writing."
    ),
    "formal_corporate_polite": (
        "Polite, formal corporate or legalistic Turkish reviews where strong dissatisfaction or praise is expressed "
        "indirectly ('beklentilerimin oldukça altında kalmıştır', 'tarafıma kusurlu ulaşmıştır', "
        "'fiyat-performans dengesi açısından tatmin edici değildir')."
    ),
    "contrastive_multi_aspect": (
        "Reviews with 2 to 4 aspects having OPPOSITE polarities connected by 'ama', 'fakat', 'lakin', 'buna karşın', "
        "'gel gör ki', 'öte yandan' (e.g., camera is amazing, battery is terrible, packaging is average)."
    ),
    "tricky_negation": (
        "Double negation or deceptive negation structures: 'kötü diyemem' (positive/neutral), 'hiç fena değil' (positive), "
        "'ucuz sayılmaz' (negative), 'beğenmedim değil' (positive), 'anlatıldığı kadar iyi değil' (negative)."
    ),
    "implicit_aspect": (
        "Reviews where one or more aspects are IMPLICIT (not named as a literal noun span in the sentence, e.g., "
        "'Dün sipariş verdim bugün kapımdaydı' -> implicit aspect 'kargo/teslimat' positive; "
        "'Bu paraya resmen soygun' -> implicit aspect 'fiyat' negative). Include `implicit_aspects` in JSON."
    ),
}

DOMAINS = [
    "akıllı telefon (Samsung S24, iPhone 15, Xiaomi)",
    "bilgisayar ve oyuncu ekipmanı (laptop, kulaklık, mouse)",
    "e-ticaret giyim ve ayakkabı (kumaş, kalıp, beden, dikiş)",
    "ev aletleri (robot süpürge, kahve makinesi, airfryer)",
    "restoran ve yemek siparişi (lezzet, porsiyon, servis hızı, sıcaklık)",
    "otel ve konaklama (oda temizliği, personel, konum, kahvaltı)",
    "mobil uygulama ve dijital servis (arayüz, hız, müşteri hizmetleri)",
    "kozmetik ve takviye (etki, koku, ambalaj, içerik)",
]

SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "samples": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "marked_text": {"type": "STRING"},
                    "register": {"type": "STRING"},
                    "implicit_aspects": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "aspect": {"type": "STRING"},
                                "polarity": {"type": "STRING"},
                            },
                            "required": ["aspect", "polarity"],
                        },
                    },
                    "note": {"type": "STRING"},
                },
                "required": ["marked_text", "register"],
            },
        }
    },
    "required": ["samples"],
}


def build_prompt(category, domains, n):
    desc = CATEGORIES[category]
    return f"""You are a native Istanbul Turkish linguist creating high-value training data for an Aspect-Based Sentiment Analysis (ABSA) model.
Generate {n} distinct, natural Turkish user reviews across these domains: {', '.join(domains)}.

FOCUS STYLE / CHALLENGE: {category}
{desc}

HOW TO ANNOTATE:
1. Mark every EXPLICIT aspect term (the exact noun/phrase inside the sentence being evaluated, such as "kamerası", "batarya ömrü", "kargo", "kumaşı", "müşteri hizmetleri") inline using:
   [[ASP|exact_surface_in_text|polarity]]
   where `polarity` is strictly one of: `positive`, `negative`, `neutral`.
2. Do NOT include Turkish case suffixes outside the noun unless natural, and NEVER nest markers.
3. For IMPLICIT aspects (where the user expresses sentiment about delivery, price, or overall product through an idiom/sarcasm without naming the aspect word in the sentence), list them in `implicit_aspects`: [{{"aspect": "fiyat", "polarity": "negative"}}].
4. Set `register` to one of: `sarcastic`, `gen_z`, `informal`, `formal`, `standard`.

Examples of `marked_text`:
- "Bu telefonu alarak resmen kazık yedik, [[ASP|kamerası|negative]] tost makinesi gibi, ama [[ASP|tasarımı|positive]] fena değil."
- "abi [[ASP|kargo|positive]] uçtu resmen dün verdim bugün elimde, [[ASP|batarya|negative]] ise tam çöp :/"
- "Maşallah [[ASP|teslimat|negative]] hızınıza hayran kaldım (!), sadece üç haftada geldi."

IMPORTANT: Do NOT use any of the examples as templates. Vary product, incident, time, emotion,
syntax, polarity, spelling (Turkish ı/i, ş/s, abbreviations), idioms, and discourse context.
Do not always make sarcasm negative: occasionally use sincere positive/neutral texts with
markers like 'çok iyi' to avoid shortcut learning. Include context necessary to interpret
metaphor; an ambiguous idiom alone is insufficient. Avoid irrelevant nonsense and
avoid presenting profanity or sexual language as mandatory for Turkish internet slang.
Keep the exact surface inside every [[ASP|surface|polarity]] as written in its sentence.
Return valid JSON matching the schema."""


def parse_sample(item, category, domain):
    marked = (item.get("marked_text") or "").strip()
    if not marked:
        return None
    out, aspects, last, cur = [], [], 0, 0
    if marked.count("[[ASP|") != len(MARK.findall(marked)):
        return None
    for m in MARK.finditer(marked):
        pre = marked[last:m.start()]
        out.append(pre)
        cur += len(pre)
        surf = m.group(1).strip()
        pol = POL_MAP.get(m.group(2).strip().lower())
        if not surf or not pol:
            return None
        out.append(surf)
        aspects.append({
            "span": surf,
            "start": cur,
            "end": cur + len(surf),
            "aspect": surf.lower(),
            "polarity": pol,
        })
        cur += len(surf)
        last = m.end()
    out.append(marked[last:])
    text = "".join(out).strip()
    if "[[" in text or "]]" in text or len(text) < 12 or len(text) > 650:
        return None
    for imp in item.get("implicit_aspects") or []:
        pol = POL_MAP.get(str(imp.get("polarity", "")).lower().strip())
        asp = str(imp.get("aspect", "")).strip().lower()
        if pol and asp:
            aspects.append({"span": None, "start": None, "end": None, "aspect": asp, "polarity": pol})
    aligned = locate_spans(text, aspects)
    if len(aligned) != len(aspects):
        return None
    aspects = aligned
    if not aspects:
        return None
    tagmap = {"heavy_sarcasm_irony":"irony", "turkish_idioms_metaphors":"idiom",
              "gen_z_social_slang":"gen_z", "formal_corporate_polite":"formal",
              "contrastive_multi_aspect":"contrast", "tricky_negation":"negation",
              "implicit_aspect":"implicit"}
    flags = [tagmap.get(category, category)]
    if any(a["span"] is None for a in aspects):
        flags.append("implicit")
    if len({a["polarity"] for a in aspects}) > 1:
        flags.append("contrast")
    if len(aspects) > 1:
        flags.append("multi_aspect")
    return {
        "text": text,
        "domain": domain.split("(")[0].strip(),
        "register": item.get("register", "informal"),
        "category": category,
        "flags": flags,
        "aspects": aspects,
        "note": item.get("note", ""),
    }


class _ModelRotator:
    """Thread-safe multi-model rotator with per-model cooldown and RPD/404 blacklisting."""

    def __init__(self, key, preferred=None, min_interval=0.8):
        self.key = key
        self.min_interval = min_interval
        self.lock = threading.Lock()
        self.models = self._discover_models(key, preferred)
        self.last_used = {m: 0.0 for m in self.models}
        self.cooldown_until = {m: 0.0 for m in self.models}
        self.dead = set()
        self.thinking_ok = {m: not m.startswith("gemma") for m in self.models}
        self.schema_ok = {m: not m.startswith("gemma") for m in self.models}
        self.cursor = 0

    @staticmethod
    def _discover_models(key, preferred):
        pool = []
        if preferred:
            for m in str(preferred).split(","):
                m = m.strip()
                if m and m not in pool:
                    pool.append(m)
        for m in GEMINI_MODEL_POOL:
            if m not in pool:
                pool.append(m)
        try:
            r = requests.get(MODELS_LIST_URL, headers={"x-goog-api-key": key}, timeout=15)
            if r.status_code == 200:
                avail = {
                    m["name"].split("/")[-1]
                    for m in r.json().get("models", [])
                    if "generateContent" in m.get("supportedGenerationMethods", [])
                }
                filtered = [m for m in pool if m in avail]
                # Also append any other flash-lite / flash / gemma-4 models available on the account
                for m in sorted(avail):
                    if m not in filtered and any(k in m for k in ("flash-lite", "flash", "gemma-4")) and "tts" not in m and "image" not in m and "audio" not in m and "live" not in m:
                        filtered.append(m)
                if filtered:
                    return filtered
        except Exception:
            pass
        return pool

    def acquire(self):
        """Return (model_name, use_thinking_cfg, use_schema_cfg) waiting if needed."""
        while True:
            with self.lock:
                alive = [m for m in self.models if m not in self.dead]
                if not alive:
                    return None, False, False
                now = time.time()
                best_m, best_wait = None, 1e9
                for offset in range(len(self.models)):
                    idx = (self.cursor + offset) % len(self.models)
                    m = self.models[idx]
                    if m in self.dead:
                        continue
                    ready_at = max(self.cooldown_until[m], self.last_used[m] + self.min_interval)
                    wait = max(0.0, ready_at - now)
                    if wait < best_wait:
                        best_wait = wait
                        best_m = m
                        if wait == 0.0:
                            self.cursor = (idx + 1) % len(self.models)
                            break
                if best_m is not None and best_wait <= 0.05:
                    self.last_used[best_m] = time.time()
                    return best_m, self.thinking_ok[best_m], self.schema_ok[best_m]
            time.sleep(min(best_wait, 2.0))

    def mark_rate_limited(self, model, body_text="", retry_after=15.0):
        with self.lock:
            low = (body_text or "").lower()
            if "per_day" in low or "perday" in low or "daily" in low or "quota exceeded for quota metric" in low and "day" in low:
                self.dead.add(model)
                print(f"[gemini-absa] model {model} hit daily RPD quota -> switching to remaining pool", flush=True)
            else:
                self.cooldown_until[model] = max(self.cooldown_until.get(model, 0.0), time.time() + retry_after)

    def mark_dead(self, model, reason=""):
        with self.lock:
            if model not in self.dead:
                self.dead.add(model)
                print(f"[gemini-absa] disabling model {model} ({reason})", flush=True)

    def disable_thinking(self, model):
        with self.lock:
            self.thinking_ok[model] = False

    def disable_schema(self, model):
        with self.lock:
            self.schema_ok[model] = False


def _extract_samples_from_text(raw_text):
    raw_text = (raw_text or "").strip()
    if raw_text.startswith("```"):
        raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
        raw_text = re.sub(r"\s*```$", "", raw_text)
    try:
        obj = json.loads(raw_text)
        if isinstance(obj, dict) and "samples" in obj:
            return obj["samples"]
        if isinstance(obj, list):
            return obj
    except Exception:
        m = re.search(r"\{.*\}|\[.*\]", raw_text, re.S)
        if m:
            try:
                obj = json.loads(m.group(0))
                return obj.get("samples", []) if isinstance(obj, dict) else (obj if isinstance(obj, list) else [])
            except Exception:
                pass
    return []


def _call_with_fallback(prompt, rotator, max_attempts=10):
    for attempt in range(max_attempts):
        model, use_thinking, use_schema = rotator.acquire()
        if model is None:
            return [], None
        url = ENDPOINT.format(m=model)
        gen = {"temperature": 1.0, "maxOutputTokens": 8192}
        if use_schema:
            gen["responseMimeType"] = "application/json"
            gen["responseSchema"] = SCHEMA
        if use_thinking:
            gen["thinkingConfig"] = {"thinkingBudget": 0}
        body = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": gen}
        try:
            r = requests.post(
                url,
                headers={"x-goog-api-key": rotator.key, "Content-Type": "application/json"},
                json=body,
                timeout=90,
            )
        except requests.RequestException:
            rotator.mark_rate_limited(model, retry_after=5.0)
            continue

        if r.status_code == 200:
            try:
                parts = r.json()["candidates"][0]["content"]["parts"]
                text = "".join(p.get("text", "") for p in parts)
                samples = _extract_samples_from_text(text)
                if samples:
                    return samples, model
            except Exception:
                pass
            continue

        if r.status_code == 429:
            rotator.mark_rate_limited(model, r.text, retry_after=12.0 + random.random() * 6.0)
            continue

        if r.status_code in (500, 502, 503, 504):
            rotator.mark_rate_limited(model, retry_after=8.0)
            continue

        if r.status_code in (404, 403):
            rotator.mark_dead(model, f"HTTP {r.status_code}")
            continue

        if r.status_code == 400:
            low = r.text.lower()
            if use_thinking and ("thinking" in low or "thinkingbudget" in low):
                rotator.disable_thinking(model)
                continue
            if use_schema and ("responseschema" in low or "mimetypes" in low or "response_mime_type" in low or "json" in low):
                rotator.disable_schema(model)
                continue
            rotator.mark_dead(model, f"HTTP 400: {r.text[:120]}")
            continue

    return [], None


def generate(out_path, total_calls=80, per_call=8, workers=3, min_interval=0.8, seed=42, model=None):
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    rng = random.Random(seed)
    cats = list(CATEGORIES.keys())
    seen, rows = set(), []
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            for l in f:
                if l.strip():
                    r = json.loads(l)
                    rows.append(r)
                    seen.add(text_hash(r["text"]))
    done_calls = len({r.get("call") for r in rows})
    todo = [(c, rng.sample(DOMAINS, 2)) for c in (cats * (total_calls // len(cats) + 1))[:total_calls]][done_calls:]
    rotator = _ModelRotator(key, preferred=model, min_interval=min_interval)
    print(f"[gemini-absa] model_pool={rotator.models} existing={len(rows)} remaining_calls={len(todo)}", flush=True)
    lock = threading.Lock()
    stats = {"ok": 0, "rejected": 0, "dup": 0, "by_model": {}}

    def work(i, cat, doms):
        samples, used_model = _call_with_fallback(build_prompt(cat, doms, per_call), rotator)
        return i, cat, doms[0], samples, used_model

    with ThreadPoolExecutor(workers) as ex, open(out_path, "a", encoding="utf-8") as f:
        futs = [ex.submit(work, done_calls + k, c, d) for k, (c, d) in enumerate(todo)]
        for fut in as_completed(futs):
            i, cat, dom, res, used_model = fut.result()
            if not res:
                print(f"[gemini-absa] call {i} ({cat}): 0 samples returned", flush=True)
                continue
            for it in res:
                parsed = parse_sample(it, cat, dom)
                if parsed is None:
                    stats["rejected"] += 1
                    continue
                h = text_hash(parsed["text"])
                with lock:
                    if h in seen:
                        stats["dup"] += 1
                        continue
                    seen.add(h)
                    parsed["call"] = i
                    parsed["model"] = used_model or "unknown"
                    stats["by_model"][parsed["model"]] = stats["by_model"].get(parsed["model"], 0) + 1
                    f.write(json.dumps(parsed, ensure_ascii=False) + "\n")
                    f.flush()
                    stats["ok"] += 1
            print(f"[gemini-absa] call {i} ({cat} via {used_model}): ok={stats['ok']} rej={stats['rejected']} dup={stats['dup']}", flush=True)
    return stats


def load(path):
    docs = []
    if not os.path.exists(path):
        return docs
    with open(path, encoding="utf-8") as f:
        for i, l in enumerate(f):
            if l.strip():
                r = json.loads(l)
                docs.append({
                    "id": f"gem_absa_{i}",
                    "text": r["text"],
                    "domain": r.get("domain", "ecommerce"),
                    "source": "synth_gemini",
                    "register": r.get("register", "informal"),
                    "flags": r.get("flags", [r.get("category", "edge")]),
                    "group": f"gem_{i}",
                    "aspects": locate_spans(r["text"], r.get("aspects", [])),
                })
    return [d for d in docs if d["aspects"]]
