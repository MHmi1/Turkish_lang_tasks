"""Metrics for Turkish ABSA:
1. ATE (Aspect Term Extraction): exact & overlap span P / R / F1.
2. APC (Aspect Polarity Classification given gold aspect): accuracy, macro-F1, per-class F1,
   and slice breakdowns (irony/sarcasm, idiom, gen_z, negation, contrast, implicit).
3. End-to-End Tuple (span, polarity) F1.
"""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from .config import APC_I2L, APC_LABELS, ATE_I2L


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


# --------------------------------------------------------- trainer metrics
def ate_trainer_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = np.argmax(logits, -1)
    tp = fp = fn = 0
    for p_seq, l_seq in zip(preds, labels):
        m = l_seq != -100
        g, p = _bio_spans(l_seq[m]), _bio_spans(p_seq[m])
        tp += len(g & p)
        fp += len(p - g)
        fn += len(g - p)
    p, r, f1 = prf(tp, fp, fn)
    return {"precision": p, "recall": r, "f1": f1}


def _bio_spans(seq):
    spans, cur = set(), None
    for i, t in enumerate(seq):
        tag = ATE_I2L[int(t)]
        if tag == "B-ASP" or (tag == "I-ASP" and cur is None):
            if cur is not None:
                spans.add((cur, i))
            cur = i
        elif tag == "O":
            if cur is not None:
                spans.add((cur, i))
                cur = None
    if cur is not None:
        spans.add((cur, len(seq)))
    return spans


def apc_trainer_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = np.argmax(logits, -1)
    acc = float(accuracy_score(labels, preds))
    mf1 = float(f1_score(labels, preds, average="macro", zero_division=0))
    return {"accuracy": acc, "macro_f1": mf1}


# --------------------------------------------------------- full benchmark eval
def eval_ate(docs, pred_spans_per_doc):
    ex_tp = ex_fp = ex_fn = ov_tp = ov_fp = ov_fn = 0
    for d, preds in zip(docs, pred_spans_per_doc):
        golds = [(a["start"], a["end"]) for a in d["aspects"] if isinstance(a.get("start"), int) and isinstance(a.get("end"), int)]
        gs, ps = set(golds), set(preds)
        ex_tp += len(gs & ps)
        ex_fp += len(ps - gs)
        ex_fn += len(gs - ps)
        used = set()
        for g in golds:
            for j, p in enumerate(preds):
                if j not in used and min(g[1], p[1]) > max(g[0], p[0]):
                    used.add(j)
                    ov_tp += 1
                    break
            else:
                ov_fn += 1
        ov_fp += len(preds) - len(used)
    ep, er, ef1 = prf(ex_tp, ex_fp, ex_fn)
    op, orc, of1 = prf(ov_tp, ov_fp, ov_fn)
    return {
        "exact": {"precision": ep, "recall": er, "f1": ef1},
        "overlap": {"precision": op, "recall": orc, "f1": of1},
    }


def eval_apc(y_true, y_pred, meta=None):
    acc = float(accuracy_score(y_true, y_pred)) if len(y_true) else 0.0
    mf1 = float(f1_score(y_true, y_pred, labels=[0,1,2], average="macro", zero_division=0)) if len(y_true) else 0.0
    p, r, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], zero_division=0)
    per_class = {
        APC_I2L[i]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f1[i]), "support": int(sup[i])}
        for i in range(3)
    }
    slices = {}
    if meta:
        for tag in ("irony", "idiom", "gen_z", "negation", "contrast", "implicit", "formal"):
            idx = [i for i, m in enumerate(meta) if tag in m.get("flags", []) or m.get("register") == tag or (tag == "implicit" and m.get("implicit"))]
            if len(idx) >= 5:
                yt = [y_true[i] for i in idx]
                yp = [y_pred[i] for i in idx]
                slices[tag] = {
                    "n": len(idx),
                    "acc": float(accuracy_score(yt, yp)),
                    "macro_f1": float(f1_score(yt, yp, labels=[0,1,2], average="macro", zero_division=0)),
                }
    return {"n": len(y_true), "accuracy": acc, "macro_f1": mf1, "per_class": per_class, "slices": slices}


def eval_e2e(docs, pred_tuples_per_doc):
    """End-to-end tuple (start, end, polarity) exact & overlap F1 on explicit aspects."""
    ex_tp = ex_fp = ex_fn = ov_tp = ov_fp = ov_fn = 0
    for d, preds in zip(docs, pred_tuples_per_doc):
        golds = [(a["start"], a["end"], a["polarity"]) for a in d["aspects"] if isinstance(a.get("start"), int)]
        gs, ps = set(golds), set(preds)
        ex_tp += len(gs & ps)
        ex_fp += len(ps - gs)
        ex_fn += len(gs - ps)
        used = set()
        for g in golds:
            for j, p in enumerate(preds):
                if j not in used and g[2] == p[2] and min(g[1], p[1]) > max(g[0], p[0]):
                    used.add(j)
                    ov_tp += 1
                    break
            else:
                ov_fn += 1
        ov_fp += len(preds) - len(used)
    ep, er, ef1 = prf(ex_tp, ex_fp, ex_fn)
    op, orc, of1 = prf(ov_tp, ov_fp, ov_fn)
    return {"exact_f1": ef1, "overlap_f1": of1, "exact_p": ep, "exact_r": er, "overlap_p": op, "overlap_r": orc}


def eval_implicit(docs, predicted):
    """Micro tuple F1 over explicit-NULL gold aspects only.
    Zero true implicit labels are excluded from the reported recall denominator,
    but all predicted implicit labels count against precision.
    """
    from .data import canonical_aspect
    tp=fp=fn=0
    for d, preds in zip(docs,predicted):
        gold={(canonical_aspect(a.get("aspect","")),a["polarity"])
              for a in d["aspects"] if a.get("span") is None}
        guess={(canonical_aspect(a.get("aspect","")),a["polarity"])
               for a in preds if a.get("implicit")}
        tp+=len(gold & guess); fp+=len(guess-gold); fn+=len(gold-guess)
    p,r,f1=prf(tp,fp,fn)
    return {"n_docs":len(docs),"support":tp+fn,"precision":p,"recall":r,"f1":f1,
            "tp":tp,"fp":fp,"fn":fn}
