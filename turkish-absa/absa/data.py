"""Dataset download, parsing (ABSA-TR + M-ABSA Turkish), span alignment, and leakage-safe splits.

Unified document schema:
{
  "id": str,
  "text": str,
  "domain": str,
  "source": str,
  "register": str,          # "formal" | "informal" | "gen_z" | "sarcastic" | "standard"
  "flags": [str, ...],      # ["irony", "idiom", "gen_z", "contrast", "negation", "implicit", ...]
  "aspects": [
    {"span": str|None, "start": int|None, "end": int|None, "aspect": str, "polarity": "negative"|"neutral"|"positive"}
  ]
}
"""
import ast
import hashlib
import json
import os
import re
from bisect import bisect_right
from collections import Counter
from pathlib import Path

from .config import ABSA_TR_FILES, ABSA_TR_REPO, APC_L2I, ATE_L2I, CACHE, M_ABSA_DOMAINS, M_ABSA_REPO, POL_MAP
from .text import tr_lower


ALIGNMENT_STATS = Counter()


def fetch(repo, filename):
    from huggingface_hub import hf_hub_download
    revisions = json.loads(os.environ.get("ABSA_DATASET_REVISIONS", "{}"))
    return hf_hub_download(repo_id=repo, filename=filename, repo_type="dataset",
                           revision=revisions.get(repo), cache_dir=str(CACHE))


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def text_hash(text):
    return hashlib.md5(re.sub(r"\s+", " ", tr_lower(text)).strip().encode()).hexdigest()


def bucket(key, salt="absa"):
    return int(hashlib.md5((salt + key).encode()).hexdigest()[:8], 16) % 1000


def locate_spans(text, aspects, diagnostics=None):
    """Resolve offsets; reject a document if any explicit annotation cannot align."""
    used = []
    out = []
    for a in aspects:
        pol = POL_MAP.get(str(a.get("polarity", "")).lower().strip())
        if not pol:
            if diagnostics is not None:
                diagnostics.append({"reason": "unsupported_polarity", "polarity": a.get("polarity")})
            return []
        surf = a.get("span")
        asp_name = (a.get("aspect") or surf or "genel").strip()
        s, e = a.get("start"), a.get("end")
        if surf is None or str(surf).strip() == "" or str(surf).lower() == "null":
            out.append({"span": None, "start": None, "end": None, "aspect": asp_name, "polarity": pol})
            continue
        surf = str(surf).strip()
        if isinstance(s, int) and isinstance(e, int) and 0 <= s < e <= len(text) and text[s:e] == surf:
            used.append((s, e))
            out.append({"span": surf, "start": s, "end": e, "aspect": asp_name, "polarity": pol})
            continue
        found = None
        for m in re.finditer(re.escape(surf), text):
            if all(m.end() <= u0 or m.start() >= u1 for u0, u1 in used):
                found = (m.start(), m.end())
                break
        if found is None:
            for m in re.finditer(re.escape(surf), text, re.I):
                if all(m.end() <= u0 or m.start() >= u1 for u0, u1 in used):
                    found = (m.start(), m.end())
                    surf = text[found[0]:found[1]]
                    break
        if found is not None:
            used.append(found)
            out.append({"span": surf, "start": found[0], "end": found[1], "aspect": asp_name, "polarity": pol})
        else:
            if diagnostics is not None:
                diagnostics.append({"span": surf, "aspect": asp_name})
            return []
    return out


def parse_absa_tr(split):
    path = fetch(ABSA_TR_REPO, ABSA_TR_FILES[split])
    docs = []
    for i, r in enumerate(read_jsonl(path)):
        text = r.get("text", "").strip()
        if not text:
            continue
        ALIGNMENT_STATS[f"absa_tr_{split}_attempted_docs"] += 1
        issues = []
        asps = locate_spans(text, r.get("aspects", []), diagnostics=issues)
        if issues:
            ALIGNMENT_STATS[f"absa_tr_{split}_unaligned_docs"] += 1
        if not asps:
            continue
        flags = list(r.get("flags") or [])
        if any(a["span"] is None for a in asps) and "implicit" not in flags:
            flags.append("implicit")
        docs.append({
            "id": f"absatr_{split}_{r.get('pool_id', i)}",
            "text": text,
            "domain": r.get("domain", "general"),
            "source": "absa_tr",
            "register": "standard",
            "flags": flags,
            "aspects": asps,
        })
    return docs


def parse_m_absa(split):
    hf_split = "dev" if split == "val" else split
    docs = []
    for dom in M_ABSA_DOMAINS:
        try:
            path = fetch(M_ABSA_REPO, f"{dom}/tr/{hf_split}.txt")
        except Exception as exc:
            raise RuntimeError(f"M-ABSA missing {dom}/tr/{hf_split}.txt; refuse silent dataset loss") from exc
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if "####" not in line:
                    continue
                text, raw = line.strip().split("####", 1)
                text = text.lstrip("\u200e\u200f").strip()
                ALIGNMENT_STATS[f"m_absa_{dom}_{split}_attempted_docs"] += 1
                try:
                    triplets = ast.literal_eval(raw)
                except Exception:
                    ALIGNMENT_STATS[f"m_absa_{dom}_{split}_malformed_docs"] += 1
                    continue
                if not isinstance(triplets, list) or any(
                        not isinstance(item, (list, tuple)) or len(item) < 3
                        for item in triplets):
                    ALIGNMENT_STATS[f"m_absa_{dom}_{split}_malformed_docs"] += 1
                    continue
                raw_asps = []
                seen = set()
                for item in triplets:
                    sp, cat, pol = item[0], item[1], item[2]
                    key = (sp, pol)
                    if key in seen:
                        continue
                    seen.add(key)
                    raw_asps.append({"span": None if sp in ("NULL", "null", "") else sp,
                                     "aspect": str(cat).split("#")[0].lower(),
                                     "polarity": pol})
                issues = []
                asps = locate_spans(text, raw_asps, diagnostics=issues)
                if issues:
                    ALIGNMENT_STATS[f"m_absa_{dom}_{split}_unaligned_docs"] += 1
                if not asps:
                    continue
                docs.append({
                    "id": f"mabsa_{dom}_{split}_{i}",
                    "text": text,
                    "domain": dom,
                    "source": "m_absa",
                    "register": "standard",
                    "flags": ["multi_aspect"] if len(asps) > 1 else [],
                    "aspects": asps,
                })
    return docs


def parse_clothing(split):
    """Clothing ABSA: aspect CATEGORY + polarity, no span supervision.
    Reads official parquet directly via hf_hub_download + pandas to avoid
    datasets/huggingface_hub version conflicts on Kaggle.
    """
    from collections import defaultdict
    import pandas as pd
    repo = "ebrukilic/clothing_products_ABSA"
    fname = "data/test-00000-of-00001.parquet" if split == "test" else "data/train-00000-of-00001.parquet"
    path = fetch(repo, fname)
    df = pd.read_parquet(path)
    grouped = defaultdict(dict)
    for r in df.to_dict("records"):
        text = str(r.get("normalized_yorum") or "").strip()
        category = str(r.get("aspects") or "").strip().lower()
        pol = POL_MAP.get(str(r.get("polarity") or "").strip().lower())
        if len(text) < 8 or not category or not pol:
            continue
        key = text_hash(text)
        b = bucket(key, salt="clothing_2026")
        if split == "val" and b >= 120:
            continue
        if split == "train" and b < 120:
            continue
        if key not in grouped:
            grouped[key] = {
                "id": f"clothing_{split}_{key[:12]}",
                "text": text,
                "domain": "clothing",
                "source": "clothing_absa",
                "register": "standard",
                "flags": ["category_only"],
                "span_supervision": False,
                "aspects": [],
                "_pairs": {},
            }
        d = grouped[key]
        if category in d["_pairs"] and d["_pairs"][category] != pol:
            d["_pairs"][category] = "CONFLICT"
        elif category not in d["_pairs"]:
            d["_pairs"][category] = pol
    docs = []
    for d in grouped.values():
        d["aspects"] = [
            {"span": None, "start": None, "end": None, "aspect": cat, "polarity": pol}
            for cat, pol in d.pop("_pairs").items()
            if pol != "CONFLICT"
        ]
        if d["aspects"]:
            docs.append(d)
    return docs


def load_public(with_clothing=True):
    ALIGNMENT_STATS.clear()
    tr = parse_absa_tr("train") + parse_m_absa("train")
    va = parse_absa_tr("val") + parse_m_absa("val")
    te_absatr = parse_absa_tr("test")
    te_mabsa = parse_m_absa("test")
    clothes_test = []
    if with_clothing:
        # Explicit failure rather than silently claiming a third data source.
        clothes_train, clothes_val, clothes_test = (parse_clothing("train"),
                                                     parse_clothing("val"),parse_clothing("test"))
        tr += clothes_train
        va += clothes_val
    held = {text_hash(d["text"]) for d in va + te_absatr + te_mabsa + clothes_test}
    seen = set()
    clean_tr = []
    for d in tr:
        h = text_hash(d["text"])
        if h not in held and h not in seen:
            seen.add(h)
            clean_tr.append(d)
    stats = {
        "train": len(clean_tr),
        "val": len(va),
        "absa_tr_test": len(te_absatr),
        "m_absa_test": len(te_mabsa),
        "polarity_dist": dict(Counter(a["polarity"] for d in clean_tr for a in d["aspects"])),
        "implicit_count": sum(1 for d in clean_tr if d.get("span_supervision", True)
                              for a in d["aspects"] if a["span"] is None),
        "sources": dict(Counter(d["source"] for d in clean_tr)),
        "dataset_caveat": "ABSA-TR uses LLM annotations; M-ABSA Turkish is translated/human reviewed; neither is guaranteed native-ecommerce gold.",
        "alignment_counts": dict(ALIGNMENT_STATS),
    }
    stats["clothing_apc_test"] = len(clothes_test)
    stats["alignment_drop_rates"] = {
        key.removesuffix("_unaligned_docs"): count /
        max(1, ALIGNMENT_STATS[key.removesuffix("_unaligned_docs") + "_attempted_docs"])
        for key, count in ALIGNMENT_STATS.items() if key.endswith("_unaligned_docs")}
    return {"train": clean_tr, "val": va, "absa_tr_test": te_absatr,
            "m_absa_test": te_mabsa, "clothing_apc_test": clothes_test}, stats


# --------------------------------------------------------- tokenization for ATE
def _align_ate(text, spans, offsets):
    spans = sorted([s for s in spans if s[0] is not None and s[1] is not None])
    starts = [s for s, _ in spans]
    labels, prev = [], -1
    for ts, te in offsets:
        if te <= ts:
            labels.append(-100)
            continue
        while ts < te and text[ts].isspace():
            ts += 1
        if ts >= te:
            labels.append(-100)
            continue
        idx = bisect_right(starts, ts) - 1
        if idx >= 0 and not (spans[idx][0] <= ts < spans[idx][1]):
            idx = -1
        if idx < 0:
            j = bisect_right(starts, ts)
            if j < len(spans) and spans[j][0] < te and (te - spans[j][0]) * 2 >= (te - ts):
                idx = j
        if idx < 0:
            labels.append(ATE_L2I["O"])
            prev = -1
        else:
            labels.append(ATE_L2I["I-ASP" if prev == idx else "B-ASP"])
            prev = idx
    return labels


def encode_ate(tok, docs, max_len=256):
    # only docs that have at least one explicit span or are genuine review sentences
    enc = tok([d["text"] for d in docs], truncation=True, max_length=max_len,
              stride=max(1, min(64, max_len // 4)), return_overflowing_tokens=True,
              return_offsets_mapping=True)
    items = []
    owners = enc.get("overflow_to_sample_mapping", range(len(docs)))
    for owner, ids, offs in zip(owners, enc["input_ids"], enc["offset_mapping"]):
        d = docs[int(owner)]
        sp = [(a["start"], a["end"]) for a in d["aspects"] if isinstance(a.get("start"), int) and isinstance(a.get("end"), int)]
        items.append({"input_ids": ids, "labels": _align_ate(d["text"], sp, offs)})
    return items


# --------------------------------------------------------- tokenization for APC
def canonical_aspect(name):
    from .config import ASPECT_ALIASES
    k = re.sub(r"\s+", " ", tr_lower(name)).strip()
    return ASPECT_ALIASES.get(k, k)


def build_implicit_ontology(docs, min_count=2, max_per_domain=40):
    """Fit the candidate vocabulary on training annotations only, per domain."""
    counts = {}
    for d in docs:
        domain = str(d.get("domain") or "general")
        tally = counts.setdefault(domain, Counter())
        for a in d["aspects"]:
            if a.get("polarity") in APC_L2I:
                tally[canonical_aspect(a.get("aspect") or a.get("span"))] += 1
    return {domain: [name for name, count in tally.most_common(max_per_domain)
                     if name and count >= min_count]
            for domain, tally in counts.items()}


# Conservative weak-supervision guard. Exclude negatives whenever an aspect can
# plausibly be IMPLIED by the sentence, even without a literal aspect word.
NEGATIVE_CUE_STEMS = {
    "teslimat": ["sipariş", "kurye", "kapımda", "geldi", "ulaştı", "bekle", "karg", "gün içinde", "gecik"],
    "fiyat": ["para", "tl", "₺", "ucuz", "pahalı", "bedava", "kazık", "fiyat", "değer", "soygun", "kuruş"],
    "kalite": ["bozul", "kır", "dağıl", "sağlam", "dayan", "kalite", "kaliteli", "kusur", "çöp", "malzeme"],
    "performans": ["hız", "don", "kas", "yavaş", "akıcı", "performans", "çalışmıyor", "bug", "drop", "ısını"],
    "batarya": ["pil", "batarya", "şarj", "enerji", "yüzde", "günü çıkar"],
    "kamera": ["foto", "video", "çekim", "kamera", "lens", "gece mod"],
    "ekran": ["ekran", "parlak", "panel", "hz", "piksel", "dokunmat"],
    "paketleme": ["kutu", "ambalaj", "paket", "sarma", "ezik", "baloncuk"],
    "iade": ["iade", "para yattı", "geri ödeme", "değişim", "geri gönd"],
    "müşteri hizmetleri": ["destek", "temsilci", "müşteri", "satıcı", "aram", "şikayet"],
    "ses": ["ses", "bas", "hoparlör", "mikrofon", "gürültü", "fan", "teneke"],
    "beden": ["beden", "kalıp", "dar", "geniş", "bol", "küçük geldi", "büyük geldi", "xl"],
    "kumaş": ["kumaş", "doku", "pamuk", "incecik", "tüylen", "kıyafet", "dikiş"],
}


def _has_cue(text, cue):
    words = [re.escape(part) for part in cue.split()]
    pattern = r"(?<!\w)" + r"\s+".join(words) + (r"\w*(?!\w)" if len(cue) >= 4 else r"(?!\w)")
    return re.search(pattern, text, flags=re.UNICODE) is not None


def safe_negative_candidates(text, positive, candidates):
    normalized = {canonical_aspect(p) for p in positive}
    lower = tr_lower(text)
    return [c for c in candidates
            if c not in normalized and c != "ürün"
            and not _has_cue(lower, c)
            and not any(_has_cue(lower, term) for term in NEGATIVE_CUE_STEMS.get(c, []))]


def apc_views(doc, policy="category"):
    """Yield unambiguous (target, polarity, aspect) views for a text-only APC pair.

    A sentence may express opposite opinions about the same category or repeat
    the same surface with different sentiment. Without a target offset, those
    identical (text, target) inputs cannot have two labels. Keep the document
    for ATE/E2E, but exclude only its ambiguous APC views.
    """
    by_target = {}
    views = []
    for a in doc["aspects"]:
        pol = a.get("polarity")
        if pol not in APC_L2I:
            continue
        target = canonical_aspect(a.get("aspect") or a.get("span") or "ürün")
        candidates = [target]
        surf = a.get("span")
        if policy == "surface" and surf:
            candidates = [str(surf).strip()]
        elif policy == "both" and surf and tr_lower(surf).strip() != target and target not in ("", "genel"):
            candidates.append(str(surf).strip())
        for view in dict.fromkeys(candidates):
            key = tr_lower(view)
            by_target.setdefault(key, set()).add(pol)
            views.append((view, pol, a))
    seen = set()
    for view, pol, a in views:
        key = tr_lower(view)
        if len(by_target[key]) != 1 or key in seen:
            continue
        seen.add(key)
        yield view, pol, a


def encode_apc(tok, docs, max_len=256, train=False, negative_per_doc=2, seed=42,
               add_negatives=False, ontology=None, view_policy=None):
    """4-way APC with hard absent-aspect negatives (NEUTRAL != NOT_MENTIONED).
    Use normalized categories as well as explicit surface forms as positive views.
    Negative aspect candidates are sampled only from a closed list absent from
    both the annotated category set and the text surface, as a weak safeguard.
    """
    import random
    rng = random.Random(seed)
    texts, pairs, labels, meta = [], [], [], []
    for d in docs:
        present = {canonical_aspect(a.get("aspect") or a.get("span") or "ürün")
                   for a in d["aspects"] if a.get("polarity") in APC_L2I}
        for view, pol, a in apc_views(d, policy=view_policy or ("both" if train else "surface")):
            texts.append(d["text"]); pairs.append(view); labels.append(APC_L2I[pol])
            meta.append({"pair_id": len(meta), "id": d["id"], "flags": d.get("flags", []),
                         "register": d.get("register", "standard"),
                         "implicit": a.get("span") is None and d.get("span_supervision", True), "target": view})
        if (train or add_negatives) and negative_per_doc and d.get("span_supervision", True):
            # The closed ontology includes general categories. Avoid 'ürün' as an
            # absent negative when an annotated aspect belongs to that product.
            candidates = safe_negative_candidates(d["text"], present,
                                                  (ontology or {}).get(str(d.get("domain") or "general"), []))
            rng.shuffle(candidates)
            for neg in candidates[:negative_per_doc]:
                texts.append(d["text"]); pairs.append(neg)
                labels.append(APC_L2I["not_mentioned"])
                meta.append({"pair_id": len(meta), "id": d["id"], "flags": d.get("flags", []),
                             "register": d.get("register", "standard"),
                             "implicit": False, "target": neg, "negative_sample": True})
    if not texts:
        return [], []
    enc = tok(texts, pairs, truncation="only_first", max_length=max_len,
              stride=max(1, min(64, max_len // 4)), return_overflowing_tokens=True)
    owners = enc.get("overflow_to_sample_mapping", range(len(texts)))
    items = [{"input_ids": ids, "attention_mask": mask, "labels": lab}
             for ids, mask, lab in zip(enc["input_ids"], enc["attention_mask"],
                                       [labels[int(i)] for i in owners])]
    return items, [meta[int(i)] for i in owners]
