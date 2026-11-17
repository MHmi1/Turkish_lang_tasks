"""Dataset download, parsing, unified schema, leakage-safe splitting, tokenization."""
import hashlib
import json
import random
import re
from bisect import bisect_right
from collections import Counter, defaultdict
from pathlib import Path

from .config import (CACHE, SOURCES, NM_REPO, NM_FILE, NM_MAP, L2I, LABELS, SPLITS)

# Doc schema: {"id","text","spans":[[s,e,LABEL],...],"source","group","case"}


# ------------------------------------------------------------------- io helpers
def fetch(repo, filename):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=repo, filename=filename, repo_type="dataset", cache_dir=str(CACHE))


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ----------------------------------------------------------------- span hygiene
def clean_spans(text, spans):
    """Trim whitespace, drop empty/invalid, resolve overlaps (keep longer)."""
    out = []
    for s, e, l in spans:
        if not (0 <= s < e <= len(text)):
            continue
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if s < e:
            out.append([s, e, l])
    out.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    res = []
    for sp in out:
        if res and sp[0] < res[-1][1]:
            if sp[1] - sp[0] > res[-1][1] - res[-1][0]:
                res[-1] = sp
            continue
        res.append(sp)
    return res


def skeleton(text, spans):
    """Template fingerprint: spans masked by label, digits->0, lowercase."""
    out, last = [], 0
    for s, e, l in sorted(spans):
        out.append(text[last:s])
        out.append(f"<{l}>")
        last = e
    out.append(text[last:])
    t = re.sub(r"\d", "0", "".join(out).lower())
    return hashlib.md5(re.sub(r"\s+", " ", t).strip().encode()).hexdigest()[:12]


def text_hash(text):
    return hashlib.md5(re.sub(r"\s+", " ", text.lower()).strip().encode()).hexdigest()


def bucket(key, salt="pii"):
    return int(hashlib.md5((salt + key).encode()).hexdigest()[:8], 16) % 1000


def make_doc(uid, text, spans, source, case=""):
    spans = clean_spans(text, spans)
    return {"id": uid, "text": text, "spans": spans, "source": source,
            "group": skeleton(text, spans), "case": case}


# ---------------------------------------------------------------------- parsers
def _parse_rows(name, rows, split, stats):
    cfg = SOURCES[name]
    lmap = cfg["map"]
    docs = []
    for i, r in enumerate(rows):
        text = r["text"]
        raw = []
        if name == "btx24":
            raw = [(x["start"], x["end"], x["category"]) for x in r.get("label", [])]
        elif name in ("yusuf", "belgin"):
            for k, v in (r.get("spans") or {}).items():
                lab = k.split(":", 1)[0]
                raw += [(s, e, lab) for s, e in v]
        elif name == "fevzi":
            raw = [(x["start"], x["end"], x["type"]) for x in r.get("entities", [])]
        bad = [l for _, _, l in raw if l not in lmap]
        if bad:
            stats["dropped_unmapped"][name] += 1
            for l in bad:
                stats["unmapped_labels"][l] += 1
            continue
        info = r.get("info") or {}
        case = info.get("case") or info.get("scenario") or ""
        docs.append(make_doc(f"{name}_{split}_{i}", text, [(s, e, lmap[l]) for s, e, l in raw], name, case))
    return docs


def load_public(seed=42, btx_val_max=2500, max_per_skel=4):
    """Return dict split-name -> docs, plus stats. Test sets are frozen & per-source."""
    stats = {"dropped_unmapped": Counter(), "unmapped_labels": Counter()}
    rng = random.Random(seed)
    raw = defaultdict(dict)
    for name, cfg in SOURCES.items():
        for split, fn in cfg["files"].items():
            raw[name][split] = _parse_rows(name, read_jsonl(fetch(cfg["repo"], fn)), split, stats)

    train, val, tests = [], [], {}
    # official splits
    for name in ("btx24", "belgin"):
        train += raw[name]["train"]
        v = raw[name]["val"]
        if name == "btx24":
            rng.shuffle(v)
            v = v[:btx_val_max]
        val += v
        tests[f"{name}_test"] = raw[name]["test"]
    # grouped hash splits (template families never straddle splits)
    for name, cut in (("yusuf", (850, 925)), ("fevzi", (600, 700))):
        tr, va, te = [], [], []
        for d in raw[name]["all"]:
            b = bucket(d["group"], salt=f"{seed}")
            (tr if b < cut[0] else va if b < cut[1] else te).append(d)
        train += tr
        val += va
        tests[f"{name}_test"] = te

    # leakage control: exact text collisions removed from train
    held = {text_hash(d["text"]) for d in val} | {text_hash(d["text"]) for t in tests.values() for d in t}
    n0 = len(train)
    train = [d for d in train if text_hash(d["text"]) not in held]
    stats["removed_exact_overlap"] = n0 - len(train)
    # template-overlap diagnostics (not purged: official BTX24/Belgin splits share templates)
    tg = {d["group"] for d in train}
    stats["skeleton_overlap_with_train"] = {
        k: round(sum(d["group"] in tg for d in v) / max(1, len(v)), 4) for k, v in tests.items()}
    # cap redundant templates to avoid memorizing a few hundred BTX24 patterns
    by_g = defaultdict(list)
    for d in train:
        by_g[d["group"]].append(d)
    capped = []
    for g, ds in by_g.items():
        rng.shuffle(ds)
        capped += ds[:max_per_skel] if ds[0]["source"] in ("btx24",) else ds
    rng.shuffle(capped)
    stats["train_before_cap"], stats["train_after_cap"] = n0 - stats["removed_exact_overlap"], len(capped)
    return {"train": capped, "val": val, **tests}, stats


def load_nm_test(max_docs=None):
    """External long-document test (nm-kvkk test_hard). Labels outside taxonomy -> 'ignore' spans."""
    docs = []
    for i, r in enumerate(read_jsonl(fetch(NM_REPO, NM_FILE))):
        text = r["text"]
        spans, ignore = [], []
        for e in r.get("entities", []):
            if e["label"] in NM_MAP:
                spans.append((e["start"], e["end"], NM_MAP[e["label"]]))
            else:
                ignore.append([e["start"], e["end"]])
        d = make_doc(f"nm_{i}", text, spans, "nm_kvkk")
        d["ignore"] = ignore
        docs.append(d)
        if max_docs and len(docs) >= max_docs:
            break
    return docs


# ------------------------------------------------------------- chunking (train)
def chunk_doc(doc, max_chars=700):
    """Split long training docs at whitespace outside spans; keep spans fully inside a chunk."""
    text = doc["text"]
    if len(text) <= max_chars:
        return [doc]
    spans = doc["spans"]
    cuts, start = [], 0
    while len(text) - start > max_chars:
        cut = start + max_chars
        while cut > start + 50 and (not text[cut].isspace() or any(s < cut < e for s, e, _ in spans)):
            cut -= 1
        if cut <= start + 50:
            cut = start + max_chars
        cuts.append((start, cut))
        start = cut
    cuts.append((start, len(text)))
    out = []
    for k, (a, b) in enumerate(cuts):
        sp = [[s - a, e - a, l] for s, e, l in spans if s >= a and e <= b]
        out.append({**doc, "id": f"{doc['id']}#{k}", "text": text[a:b], "spans": sp})
    return out


# ------------------------------------------------------------------ tokenization
def _align(text, spans, offsets):
    starts = [s for s, _, _ in spans]
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
        if idx < 0:  # token begins before a span but mostly overlaps it
            j = bisect_right(starts, ts)
            if j < len(spans) and spans[j][0] < te and (te - spans[j][0]) * 2 >= (te - ts):
                idx = j
        if idx < 0:
            labels.append(L2I["O"])
            prev = -1
        else:
            lab = spans[idx][2]
            labels.append(L2I[f"{'I' if prev == idx else 'B'}-{lab}"])
            prev = idx
    return labels


def encode_docs(tok, docs, max_len=256):
    enc = tok([d["text"] for d in docs], truncation=True, max_length=max_len,
              return_offsets_mapping=True)
    items, trunc = [], 0
    for d, ids, offs in zip(docs, enc["input_ids"], enc["offset_mapping"]):
        items.append({"input_ids": ids, "labels": _align(d["text"], d["spans"], offs)})
        trunc += len(ids) >= max_len
    return items, trunc


def label_stats(docs):
    c = Counter(l for d in docs for _, _, l in d["spans"])
    return dict(c.most_common()), sum(1 for d in docs if not d["spans"])
