"""Reproducible EDA for prepared Turkish ABSA splits."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from absa.data import canonical_aspect, read_jsonl
from absa.config import model_revision
from absa.quality import canonical_text, quality_flags


SPLIT_NAMES = ("train_public", "train_clean", "val", "absa_tr_test", "m_absa_test", "clothing_apc_test")


def _quantiles(values):
    if not values:
        return {"p50": 0, "p90": 0, "p95": 0, "p99": 0, "max": 0}
    s = pd.Series(values)
    return {f"p{int(q * 100)}": round(float(s.quantile(q)), 1)
            for q in (0.5, 0.9, 0.95, 0.99)} | {"max": int(s.max())}


def _token_lengths(docs, model_name, max_len):
    if not model_name:
        return {"status": "not_requested"}
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_name, revision=model_revision(model_name), use_fast=True)
        lengths = []
        pair_lengths = []
        for i in range(0, len(docs), 256):
            batch = docs[i:i + 256]
            texts = [d["text"] for d in batch]
            targets = [str((d["aspects"][0].get("aspect") or d["aspects"][0].get("span") or ""))
                       if d.get("aspects") else "" for d in batch]
            lengths += [len(x) for x in tok(texts, truncation=False)["input_ids"]]
            pair_lengths += [len(x) for x in tok(texts, targets, truncation=False)["input_ids"]]
        return {"status": "ok", "tokenizer": model_name, "max_len": max_len,
                "ate_lengths": _quantiles(lengths), "apc_pair_lengths": _quantiles(pair_lengths),
                "ate_over_max_len": sum(n > max_len for n in lengths),
                "apc_pair_over_max_len": sum(n > max_len for n in pair_lengths),
                "n_documents": len(docs)}
    except Exception as exc:
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}


def run_eda(work, max_len=256, tokenizer_name=None):
    work = Path(work)
    split_dir = work / "data" / "splits"
    files = {name: read_jsonl(split_dir / f"{name}.jsonl") for name in SPLIT_NAMES
             if (split_dir / f"{name}.jsonl").exists()}
    if "train_clean" not in files or "val" not in files:
        raise FileNotFoundError("Run prepare and clean before EDA")

    summary = []
    by_source = Counter()
    by_domain = Counter()
    by_polarity = Counter()
    by_aspect = Counter()
    noise_flags = Counter()
    ambiguous = Counter()
    keys = {}
    for name, docs in files.items():
        length = [len(d["text"]) for d in docs]
        n_aspects = [len(d.get("aspects", [])) for d in docs]
        raw_keys = [canonical_text(d["text"]) for d in docs]
        keys[name] = set(raw_keys)
        summary.append({"split": name, "documents": len(docs), "aspects": sum(n_aspects),
                        "explicit_aspects": sum(a.get("span") is not None for d in docs for a in d["aspects"]),
                        "implicit_aspects": sum(a.get("span") is None and d.get("span_supervision", True)
                                                for d in docs for a in d["aspects"]),
                        "category_only_documents": sum(not d.get("span_supervision", True) for d in docs),
                        "multi_aspect_documents": sum(n > 1 for n in n_aspects),
                        "normalized_duplicate_rows": len(raw_keys) - len(keys[name]),
                        "char_p50": _quantiles(length)["p50"], "char_p95": _quantiles(length)["p95"],
                        "char_p99": _quantiles(length)["p99"],
                        "aspects_per_doc_p95": _quantiles(n_aspects)["p95"]})
        for d in docs:
            by_source[name, d.get("source", "unknown")] += 1
            by_domain[name, d.get("domain", "unknown")] += 1
            noise_flags.update((name, flag) for flag in quality_flags(d["text"]))
            targets = defaultdict(set)
            for a in d["aspects"]:
                pol = a.get("polarity", "unknown")
                by_polarity[name, pol] += 1
                aspect = canonical_aspect(a.get("aspect") or a.get("span") or "")
                by_aspect[name, aspect] += 1
                targets[aspect].add(pol)
            ambiguous[name] += sum(len(pols) > 1 for pols in targets.values())

    overlap = []
    for test_name in ("val", "absa_tr_test", "m_absa_test", "clothing_apc_test"):
        if test_name in keys:
            for train_name in ("train_public", "train_clean"):
                if train_name in keys:
                    overlap.append({"train": train_name, "heldout": test_name,
                                    "normalized_text_collisions": len(keys[train_name] & keys[test_name])})

    reports = work / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    frames = {
        "summary": pd.DataFrame(summary),
        "source": pd.DataFrame([{"split": k[0], "source": k[1], "documents": v} for k, v in by_source.items()]),
        "domain": pd.DataFrame([{"split": k[0], "domain": k[1], "documents": v} for k, v in by_domain.items()]),
        "polarity": pd.DataFrame([{"split": k[0], "polarity": k[1], "aspects": v} for k, v in by_polarity.items()]),
        "aspect": pd.DataFrame([{"split": k[0], "aspect": k[1], "aspects": v} for k, v in by_aspect.items()]),
        "noise": pd.DataFrame([{"split": k[0], "flag": k[1], "documents": v} for k, v in noise_flags.items()]),
        "overlap": pd.DataFrame(overlap),
    }
    for name, frame in frames.items():
        frame.to_csv(reports / f"eda_{name}.csv", index=False)
    token = _token_lengths(files["train_clean"] + files["val"], tokenizer_name, max_len)
    report = {"files": list(files), "summary": summary,
              "ambiguous_category_polarities_by_split": dict(ambiguous),
              "normalized_cross_split_overlap": overlap,
              "token_lengths_train_clean_plus_val": token,
              "notes": ["Official validation and test texts are never cleaned or removed.",
                        "Diagnostics describe possible noise, not verified annotation errors.",
                        "Train-clean and train-public are alternative versions of the same rows; do not sum them."]}
    (reports / "eda_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "overlap": overlap, "tokens": token}, ensure_ascii=False), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--tokenizer", default=None)
    args = parser.parse_args()
    run_eda(args.work, args.max_len, args.tokenizer)
