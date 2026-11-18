"""Two-stage + implicit-fallback inference for Turkish ABSA (ATE + APC)."""
import re
import time
import json
from pathlib import Path
import numpy as np

from .config import model_revision
from .data import canonical_aspect
from .text import tr_lower

_TRAIL = re.compile(r"[.,;:!?)\"']+$")


def _clean_cfg(cfg):
    if hasattr(cfg, "reference_compile"):
        cfg.reference_compile = False
    for attr in ("bos_token_id", "eos_token_id"):
        v = getattr(cfg, attr, None)
        if isinstance(v, int) and v >= getattr(cfg, "vocab_size", 1 << 30):
            setattr(cfg, attr, None)
    return cfg


class ATEPredictor:
    def __init__(self, path, device=None, fp16=True):
        import torch
        from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        revision = model_revision(path)
        self.tok = AutoTokenizer.from_pretrained(path, revision=revision)
        cfg = _clean_cfg(AutoConfig.from_pretrained(path, revision=revision))
        self.model = AutoModelForTokenClassification.from_pretrained(path, revision=revision, config=cfg).to(self.device).eval()
        labels={i:str(cfg.id2label.get(i, cfg.id2label.get(str(i),''))).upper()
                for i in range(self.model.config.num_labels)}
        expected=('O','B-ASP','I-ASP')
        if sorted(labels.values()) != sorted(expected):
            raise ValueError(f"ATE checkpoint has unsupported label map: {labels}")
        self.label_indices = [next(i for i, v in labels.items() if v == tag) for tag in expected]
        if fp16 and self.device.startswith("cuda"):
            self.model.half()

    def extract_batch(self, texts, tau=0.45, batch_size=32, max_len=256):
        torch = self.torch
        results = []
        for b in range(0, len(texts), batch_size):
            chunk = texts[b:b + batch_size]
            enc = self.tok(chunk, truncation=True, max_length=max_len,
                           stride=max(1, min(64, max_len // 4)),
                           return_overflowing_tokens=True, padding=True,
                           return_offsets_mapping=True, return_tensors="pt")
            offs = enc.pop("offset_mapping").numpy()
            owners = enc.pop("overflow_to_sample_mapping").numpy()
            mask = enc["attention_mask"].numpy().astype(bool)
            probs = []
            for start in range(0, len(owners), batch_size):
                with torch.inference_mode():
                    logits = self.model(**{k: v[start:start+batch_size].to(self.device)
                                           for k, v in enc.items()}).logits.float()
                probs.append(torch.softmax(logits, -1).cpu().numpy()[:, :, self.label_indices])
            probs = np.concatenate(probs, axis=0)
            token_rows = [{} for _ in chunk]
            for owner, o_seq, p_seq, m_seq in zip(owners, offs, probs, mask):
                o = o_seq[m_seq]
                p = p_seq[m_seq]
                keep = o[:, 1] > o[:, 0]
                o, p = o[keep], p[keep]
                for offset, row in zip(o, p):
                    key = (int(offset[0]), int(offset[1]))
                    token_rows[int(owner)].setdefault(key, []).append(row)
            for text, rows in zip(chunk, token_rows):
                ordered = sorted(rows)
                offsets = np.array(ordered, dtype=int).reshape(-1, 2)
                probabilities = np.array([np.mean(rows[key], axis=0) for key in ordered]).reshape(-1, 3)
                results.append(self._decode_one(text, offsets, probabilities, tau))
        return results

    @staticmethod
    def _decode_one(text, offsets, probs, tau):
        spans, cur = [], None

        def close():
            if cur is not None:
                s, e = cur[0], cur[1]
                while s < e and text[s].isspace():
                    s += 1
                m = _TRAIL.search(text[s:e])
                if m:
                    e = s + m.start()
                while e > s and text[e - 1].isspace():
                    e -= 1
                if e > s:
                    spans.append({
                        "start": int(s),
                        "end": int(e),
                        "span": text[s:e],
                        "score": float(np.mean(cur[2])),
                    })

        for (ts, te), pr in zip(offsets, probs):
            asp_probability = 1.0 - float(pr[0])
            label = 1 if pr[1] >= pr[2] else 2
            if asp_probability < tau:
                close(); cur = None
            elif label == 1 or cur is None:
                close()
                cur = [int(ts), int(te), [asp_probability]]
            else:
                cur[1] = int(te)
                cur[2].append(asp_probability)
        close()
        return spans


class APCPredictor:
    def __init__(self, path, device=None, fp16=True):
        import torch
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        revision = model_revision(path)
        self.tok = AutoTokenizer.from_pretrained(path, revision=revision)
        cfg = _clean_cfg(AutoConfig.from_pretrained(path, revision=revision))
        self.model = AutoModelForSequenceClassification.from_pretrained(path, revision=revision, config=cfg).to(self.device).eval()
        if fp16 and self.device.startswith("cuda"):
            self.model.half()
        # Baseline checkpoint might use a different id2label order than ours.
        raw = self.model.config.id2label
        expected = {"negative", "neutral", "positive", "not_mentioned"}
        labels = {i: str(raw.get(i, raw.get(str(i), ""))).lower() for i in range(self.model.config.num_labels)}
        if set(labels.values()) not in ({"negative", "neutral", "positive"}, expected) or len(labels) != len(set(labels.values())):
            raise ValueError(f"APC checkpoint has unsupported label map: {labels}")
        self.label_names = labels
        self.absent_idx = next((i for i, val in self.label_names.items() if val == "not_mentioned"), None)

    def classify_pairs(self, texts, aspects, batch_size=64, max_len=256):
        torch = self.torch
        preds, confs, probs_all = [], [], []
        for b in range(0, len(texts), batch_size):
            t_chunk = texts[b:b + batch_size]
            a_chunk = aspects[b:b + batch_size]
            enc = self.tok(t_chunk, a_chunk, truncation="only_first", max_length=max_len,
                           stride=max(1, min(64, max_len // 4)),
                           return_overflowing_tokens=True, padding=True,
                           return_tensors="pt")
            owners = enc.pop("overflow_to_sample_mapping").numpy()
            sums = np.zeros((len(t_chunk), self.model.config.num_labels), dtype=np.float64)
            counts = np.zeros(len(t_chunk), dtype=np.int32)
            for start in range(0, len(owners), batch_size):
                with torch.inference_mode():
                    logits = self.model(**{k: v[start:start+batch_size].to(self.device)
                                           for k, v in enc.items()}).logits.float().cpu().numpy()
                for owner, row in zip(owners[start:start+batch_size], logits):
                    sums[int(owner)] += row
                    counts[int(owner)] += 1
            if np.any(counts == 0):
                raise RuntimeError("Tokenizer omitted an APC input")
            mean_logits = sums / counts[:, None]
            shifted = mean_logits - mean_logits.max(axis=-1, keepdims=True)
            exp = np.exp(shifted)
            pr = exp / exp.sum(axis=-1, keepdims=True)
            idx = pr.argmax(-1)
            for i, p_row in zip(idx, pr):
                preds.append(int(i))
                confs.append(float(p_row[i]))
                probs_all.append(p_row)
        return preds, confs, probs_all


class ABSAPipeline:
    """Two-stage ABSA with optional 4-way open-set implicit candidate probing.

    Unlike the original unconditional product fallback, infer implicit aspects
    only if a 4-class APC model explicitly distinguishes NOT_MENTIONED from
    NEUTRAL. No explicit aspect -> no forced sentiment label.
    """

    def __init__(self, ate_dir, apc_dir, device=None, tau=0.45,
                 implicit_confidence=0.78, absent_margin=0.12):
        self.ate = ATEPredictor(ate_dir, device=device)
        self.apc = APCPredictor(apc_dir, device=device)
        self.tau = tau
        self.implicit_confidence = implicit_confidence
        self.absent_margin = absent_margin
        ontology_path = Path(apc_dir) / "implicit_ontology.json"
        self.ontology = json.loads(ontology_path.read_text(encoding="utf-8")) if ontology_path.exists() else {}

    def analyze(self, text, include_implicit_fallback=True, domain=None):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        t0 = time.perf_counter()
        spans = self.ate.extract_batch([text], tau=self.tau)[0]
        aspects = []
        if spans:
            targets = [s["span"] for s in spans]
            preds, confs, _ = self.apc.classify_pairs([text]*len(targets), targets)
            for s, p, conf in zip(spans, preds, confs):
                label = self.apc.label_names[p]
                if label == "not_mentioned":
                    continue
                aspects.append({
                    "span":s["span"], "start":s["start"], "end":s["end"],
                    "aspect":canonical_aspect(s["span"]), "polarity":label,
                    "confidence":round(float(conf),4), "implicit":False
                })
        # 3-way baseline APC cannot learn presence/absence, so do NOT fabricate
        # implicit-aspect labels from arbitrary sentiments.
        if include_implicit_fallback and self.apc.absent_idx is not None and domain in self.ontology:
            mentioned = {canonical_aspect(a["aspect"]) for a in aspects}
            candidates = [c for c in self.ontology[domain] if c not in mentioned]
            # A candidate named explicitly in the sentence is not implicit.
            candidates = [c for c in candidates if tr_lower(c) not in tr_lower(text)]
            pred, conf, prob = self.apc.classify_pairs([text]*len(candidates),candidates)
            choices=[]
            for cand, idx, row in zip(candidates,pred,prob):
                label=self.apc.label_names[idx]
                if label == "not_mentioned":
                    continue
                posterior=float(row[idx]); absent=float(row[self.apc.absent_idx])
                if posterior >= getattr(self,'implicit_confidence',0.78) and posterior-absent >= getattr(self,'absent_margin',0.12):
                    choices.append((cand,label,posterior))
            choices.sort(key=lambda x:x[2],reverse=True)
            # A small cap prevents numerous spurious ontology predictions.
            for cand,label,posterior in choices[:3]:
                aspects.append({"span":None,"start":None,"end":None,
                                "aspect":cand,"polarity":label,
                                "confidence":round(posterior,4),"implicit":True})
        return {"text":text,"aspects":aspects,
                "latency_ms":round((time.perf_counter()-t0)*1000,2)}
