"""Span-level evaluation: exact/overlap P-R-F1-F2, per-class, leakage, false redaction, bootstrap CI."""
import numpy as np

from .config import BIO, I2L


def _f(p, r, b):
    return (1 + b * b) * p * r / (b * b * p + r) if (b * b * p + r) > 0 else 0.0


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, _f(p, r, 1), _f(p, r, 2)


# ----------------------------------------------------- training-time token metric
def token_span_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    pred = np.argmax(logits, -1)
    tp = fp = fn = 0
    for p_seq, l_seq in zip(pred, labels):
        m = l_seq != -100
        g, p = _bio_spans(l_seq[m]), _bio_spans(p_seq[m])
        tp += len(g & p)
        fp += len(p - g)
        fn += len(g - p)
    p, r, f1, f2 = prf(tp, fp, fn)
    return {"precision": p, "recall": r, "f1": f1, "f2": f2}


def _bio_spans(seq):
    spans, cur = set(), None
    for i, t in enumerate(seq):
        tag = I2L[int(t)]
        if tag.startswith("B-") or (tag.startswith("I-") and (cur is None or cur[1] != tag[2:])):
            if cur:
                spans.add((cur[0], i, cur[1]))
            cur = (i, tag[2:])
        elif tag == "O":
            if cur:
                spans.add((cur[0], i, cur[1]))
            cur = None
    if cur:
        spans.add((cur[0], len(seq), cur[1]))
    return spans


# ------------------------------------------------------------- char-level eval
def _overlap(a, b):
    return min(a[1], b[1]) - max(a[0], b[0])


def _doc_counts(gold, pred):
    gs, ps = {tuple(g) for g in gold}, {tuple(p) for p in pred}
    ex_tp = len(gs & ps)
    used, ov_tp = set(), 0
    per = {}
    for g in gold:
        best, bo = None, 0
        for j, p in enumerate(pred):
            if j in used or p[2] != g[2]:
                continue
            o = _overlap(g, p)
            if o > bo:
                best, bo = j, o
        if best is not None:
            used.add(best)
            ov_tp += 1
            per.setdefault(g[2], [0, 0, 0, 0])[3] += 1
    gold_chars = unc = missed = 0
    for g in gold:
        cov = np.zeros(g[1] - g[0], bool)
        for p in pred:
            a, b = max(g[0], p[0]), min(g[1], p[1])
            if b > a:
                cov[a - g[0]:b - g[0]] = True
        gold_chars += len(cov)
        unc += int((~cov).sum())
        missed += not cov.any()
    return dict(ex_tp=ex_tp, ex_fp=len(ps) - ex_tp, ex_fn=len(gs) - ex_tp, ov_tp=ov_tp,
                ov_fp=len(pred) - ov_tp, ov_fn=len(gold) - ov_tp, gold_chars=gold_chars, unc=unc,
                missed=missed, n_gold=len(gold), n_pred=len(pred))


def evaluate(golds, preds, restrict=None, bootstrap=300, seed=0):
    """golds/preds: per-doc lists of (start, end, label)."""
    if restrict:
        golds = [[g for g in d if g[2] in restrict] for d in golds]
        preds = [[p for p in d if p[2] in restrict] for d in preds]
    rows = [_doc_counts(g, p) for g, p in zip(golds, preds)]
    keys = list(rows[0].keys())
    arr = np.array([[r[k] for k in keys] for r in rows], dtype=np.float64)
    tot = dict(zip(keys, arr.sum(0)))
    out = {"n_docs": len(rows), "n_gold": int(tot["n_gold"])}
    for tag in ("ex", "ov"):
        p, r, f1, f2 = prf(tot[f"{tag}_tp"], tot[f"{tag}_fp"], tot[f"{tag}_fn"])
        out[tag] = {"precision": p, "recall": r, "f1": f1, "f2": f2}
    out["char_leak_rate"] = tot["unc"] / tot["gold_chars"] if tot["gold_chars"] else 0.0
    out["missed_span_rate"] = tot["missed"] / tot["n_gold"] if tot["n_gold"] else 0.0
    neg = [i for i, g in enumerate(golds) if not g]
    out["n_neg_docs"] = len(neg)
    out["false_redaction_rate"] = (sum(1 for i in neg if preds[i]) / len(neg)) if neg else None
    # per class
    pc = {}
    labels = sorted({g[2] for d in golds for g in d} | {p[2] for d in preds for p in d})
    for l in labels:
        gl = [[g for g in d if g[2] == l] for d in golds]
        pl = [[p for p in d if p[2] == l] for d in preds]
        rr = [_doc_counts(a, b) for a, b in zip(gl, pl)]
        s = {k: sum(x[k] for x in rr) for k in keys}
        _, _, f1, f2 = prf(s["ex_tp"], s["ex_fp"], s["ex_fn"])
        op, orc, of1, of2 = prf(s["ov_tp"], s["ov_fp"], s["ov_fn"])
        pc[l] = {"support": int(s["n_gold"]), "ex_f1": f1, "ov_precision": op, "ov_recall": orc, "ov_f2": of2,
                 "leak": s["unc"] / s["gold_chars"] if s["gold_chars"] else None}
    out["per_class"] = pc
    # bootstrap over documents
    if bootstrap and len(rows) > 5:
        rng = np.random.default_rng(seed)
        ix = {k: i for i, k in enumerate(keys)}
        f2s, leaks = [], []
        for _ in range(bootstrap):
            s = arr[rng.integers(0, len(rows), len(rows))].sum(0)
            f2s.append(prf(s[ix["ov_tp"]], s[ix["ov_fp"]], s[ix["ov_fn"]])[3])
            leaks.append(s[ix["unc"]] / s[ix["gold_chars"]] if s[ix["gold_chars"]] else 0)
        out["ov_f2_ci"] = [float(np.percentile(f2s, 2.5)), float(np.percentile(f2s, 97.5))]
        out["leak_ci"] = [float(np.percentile(leaks, 2.5)), float(np.percentile(leaks, 97.5))]
    return out
