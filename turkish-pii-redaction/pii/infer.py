"""Inference: windowed token probabilities, span decoding, regex+neural hybrid, redaction."""
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .config import BIO, L2I, LABELS, STRUCT_LABELS
from .rules import find_structured

ID_OVERRIDE = {"TCKN", "VKN", "IBAN", "CARD", "PHONE", "EMAIL", "ACCOUNT_NUMBER"}


def build_projection(id2label, label_map=None):
    """Matrix mapping a model's BIO outputs to the unified BIO space (unmapped classes -> O)."""
    M = np.zeros((len(id2label), len(BIO)), np.float32)
    for i, name in id2label.items():
        i = int(i)
        if name == "O":
            M[i, 0] = 1
            continue
        pre, ent = name[0], name[2:]
        u = ent if label_map is None else label_map.get(ent)
        M[i, L2I.get(f"{pre}-{u}", 0) if u else 0] = 1
    return M


class NerModel:
    def __init__(self, path, device=None, label_map=None, fp16=True):
        import torch
        from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(path)
        cfg = AutoConfig.from_pretrained(path)
        if hasattr(cfg, "reference_compile"):
            cfg.reference_compile = False
        for attr in ("bos_token_id", "eos_token_id", "pad_token_id", "cls_token_id", "sep_token_id"):
            v = getattr(cfg, attr, None)
            if isinstance(v, int) and v >= getattr(cfg, "vocab_size", 1 << 30):
                setattr(cfg, attr, None)
        try:
            self.model = AutoModelForTokenClassification.from_pretrained(
                path, config=cfg, attn_implementation="sdpa"
            ).to(self.device).eval()
        except Exception:
            self.model = AutoModelForTokenClassification.from_pretrained(path, config=cfg).to(self.device).eval()
        self.fp16 = fp16 and self.device.startswith("cuda")
        if self.fp16:
            self.model.half()
        self.proj = build_projection(self.model.config.id2label, label_map)

    @staticmethod
    def _windows(text, size=1200, overlap=200):
        """[(start, end)] char windows split at spaces, consecutive windows overlap."""
        if len(text) <= size:
            return [(0, len(text))]
        out, start = [], 0
        while True:
            end = min(len(text), start + size)
            if end < len(text):
                cut = text.rfind(" ", start + size // 2, end)
                end = cut if cut > 0 else end
            out.append((start, end))
            if end >= len(text):
                break
            nxt = text.rfind(" ", start + 1, max(start + 2, end - overlap))
            start = nxt + 1 if nxt > start else end
        return out

    def token_probs(self, texts, batch_size=64, max_len=512):
        """Per text: (offsets[N,2], probs[N,len(BIO)]) in document char coordinates."""
        torch = self.torch
        wins = []  # (doc, start, lo, hi)
        for di, t in enumerate(texts):
            ws = self._windows(t)
            cuts = [(ws[k + 1][0] + ws[k][1]) // 2 for k in range(len(ws) - 1)]
            for k, (s, e) in enumerate(ws):
                wins.append((di, s, e, cuts[k - 1] if k else 0, cuts[k] if k < len(cuts) else 1 << 60))
        order = sorted(range(len(wins)), key=lambda i: wins[i][2] - wins[i][1])
        res = defaultdict(list)
        for b in range(0, len(order), batch_size):
            idx = order[b:b + batch_size]
            enc = self.tok([texts[wins[i][0]][wins[i][1]:wins[i][2]] for i in idx], truncation=True,
                           max_length=max_len, padding=True, return_offsets_mapping=True, return_tensors="pt")
            offs = enc.pop("offset_mapping").numpy()
            mask = enc["attention_mask"].numpy().astype(bool)
            with torch.inference_mode():
                logits = self.model(**{k: v.to(self.device) for k, v in enc.items()}).logits.float()
            probs = torch.softmax(logits, -1).cpu().numpy() @ self.proj
            for j, wi in enumerate(idx):
                di, s, _, lo, hi = wins[wi]
                o, p = offs[j][mask[j]] + s, probs[j][mask[j]]
                keep = (offs[j][mask[j]][:, 1] > offs[j][mask[j]][:, 0]) & (o[:, 0] >= lo) & (o[:, 0] < hi)
                res[di].append((o[keep], p[keep]))
        out = []
        for di in range(len(texts)):
            parts = res.get(di, [])
            parts.sort(key=lambda x: x[0][0, 0] if len(x[0]) else 1 << 60)
            if parts:
                out.append((np.concatenate([a for a, _ in parts]), np.concatenate([b for _, b in parts])))
            else:
                out.append((np.zeros((0, 2), int), np.zeros((0, len(BIO)), np.float32)))
        return out


_TRAIL = re.compile(r"[.,;:!?)\"']+$")
_SUFFIX = re.compile(r"^(.+?)['’`´][^\W\d_]{1,6}$")


def _refine(text, s, e, label):
    while s < e and text[s].isspace():
        s += 1
    if label not in ("URL", "EMAIL"):
        m = _SUFFIX.match(text[s:e])
        if m:
            e = s + len(m.group(1))
    if label != "ADDRESS":
        m = _TRAIL.search(text[s:e])
        if m and label not in ("SECRET",):
            e = s + m.start()
    while e > s and text[e - 1].isspace():
        e -= 1
    return s, e


def decode(text, offsets, probs, tau=0.5):
    """Flag tokens with P(non-O) > tau, type by summed B/I mass, merge into char spans."""
    n = len(offsets)
    if n == 0:
        return []
    flag = (1.0 - probs[:, 0]) > tau
    B, I = probs[:, 1::2], probs[:, 2::2]
    tscore = B + I
    typ = tscore.argmax(1)
    ar = np.arange(n)
    bp, ip, sc = B[ar, typ], I[ar, typ], tscore[ar, typ]
    spans, cur = [], None

    def close():
        if cur:
            s, e = _refine(text, cur[0], cur[1], LABELS[cur[2]])
            if e > s:
                spans.append({"start": s, "end": e, "label": LABELS[cur[2]],
                              "score": float(np.mean(cur[3])), "source": "neural"})
    for i in range(n):
        if not flag[i]:
            close()
            cur = None
            continue
        if cur is None or typ[i] != cur[2] or bp[i] > ip[i]:
            close()
            cur = [int(offsets[i, 0]), int(offsets[i, 1]), int(typ[i]), [sc[i]]]
        else:
            cur[1] = int(offsets[i, 1])
            cur[3].append(sc[i])
    close()
    return spans


def hybrid(text, neural_spans):
    """Validated structural regex hits override overlapping ID-type neural spans and fill misses."""
    out = list(neural_spans)
    for s, e, l, sc in find_structured(text):
        ov = [sp for sp in out if min(e, sp["end"]) > max(s, sp["start"])]
        if any(sp["label"] not in ID_OVERRIDE for sp in ov):
            continue  # a contextual neural span (e.g. SECRET) wins
        out = [sp for sp in out if sp not in ov]
        out.append({"start": s, "end": e, "label": l, "score": sc, "source": "regex"})
    return sorted(out, key=lambda x: x["start"])


def to_tuples(spans):
    return [(sp["start"], sp["end"], sp["label"]) for sp in spans]


def regex_only(text):
    return [{"start": s, "end": e, "label": l, "score": sc, "source": "regex"} for s, e, l, sc in find_structured(text)]


def redact(text, spans, style="typed", review_below=0.75):
    """Replace spans by placeholders ([PERSON_1]); same surface -> same index inside a doc."""
    counters, seen, parts, last = defaultdict(int), {}, [], 0
    for sp in sorted(spans, key=lambda x: x["start"]):
        if sp["start"] < last:
            continue
        surf = text[sp["start"]:sp["end"]]
        key = (sp["label"], surf.lower())
        if key not in seen:
            counters[sp["label"]] += 1
            seen[key] = counters[sp["label"]]
        ph = f"[{sp['label']}_{seen[key]}]" if style == "typed" else f"[{sp['label']}]" if style == "label" else "[REDACTED]"
        parts += [text[last:sp["start"]], ph]
        last = sp["end"]
    parts.append(text[last:])
    return {"redacted_text": "".join(parts),
            "entities": [{**sp, "surface": text[sp["start"]:sp["end"]]} for sp in spans],
            "needs_review": any(sp["score"] < review_below and sp["source"] == "neural" for sp in spans)}


def calibrate_tau(token_probs, texts, golds, grid=None, evaluate=None):
    """Pick tau maximizing overlap-F2 on validation (recall-weighted, privacy use-case)."""
    grid = grid or [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7]
    best, hist = (-1, 0.5), {}
    for t in grid:
        preds = [to_tuples(decode(tx, o, p, t)) for tx, (o, p) in zip(texts, token_probs)]
        f2 = evaluate(golds, preds, bootstrap=0)["ov"]["f2"]
        hist[t] = f2
        if f2 > best[0]:
            best = (f2, t)
    return best[1], hist


class Pipeline:
    """Production wrapper: neural model + calibrated tau + structural regex fallback."""

    def __init__(self, model_dir, device=None, use_regex=True):
        self.m = NerModel(model_dir, device=device)
        cal = Path(model_dir) / "calibration.json"
        self.tau = json.loads(cal.read_text())["tau"] if cal.exists() else 0.4
        self.use_regex = use_regex

    def detect(self, text):
        (o, p), = self.m.token_probs([text])
        sp = decode(text, o, p, self.tau)
        return hybrid(text, sp) if self.use_regex else sp

    def redact(self, text, style="typed"):
        t0 = time.perf_counter()
        r = redact(text, self.detect(text), style)
        r["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        return r
