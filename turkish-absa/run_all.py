"""Public-data Turkish ABSA pipeline. Synthetic generation remains opt-in."""
import argparse
import glob
import json
import os
import random
import shutil
import subprocess
import sys
import time

from absa import config as C
from absa.data import bucket, load_public, read_jsonl, text_hash, write_jsonl

S = C.SPLITS


def route(docs, cuts=(750, 850), key="group"):
    out = {"train": [], "val": [], "test": []}
    for d in docs:
        b = bucket(str(d.get(key, d["id"])), salt="absa_synth")
        out["train" if b < cuts[0] else "val" if b < cuts[1] else "test"].append(d)
    return out


# ------------------------------------------------------------------- prepare
def stage_smoke(a):
    """Offline checks only: no network or GPU."""
    import unittest
    suite = unittest.defaultTestLoader.discover(str(C.ROOT / "tests"), pattern="test_core.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit("Offline smoke test failure")
    print(f"[smoke] passed {result.testsRun} tests", flush=True)


def stage_prepare(a):
    sp, stats = load_public(with_clothing=not a.no_clothing)
    written_rows = {}
    for k, v in sp.items():
        if a.quick:
            v = v[:250] if k == "train" else v[:100]
        name = "train_public" if k == "train" else k
        write_jsonl(S / f"{name}.jsonl", v)
        written_rows[name] = len(v)
    stats["written_rows"] = written_rows
    stats["quick_mode"] = bool(a.quick)
    json.dump(stats, open(S / "manifest.json", "w"), indent=2, ensure_ascii=False)
    print("[prepare]", json.dumps(stats, ensure_ascii=False), flush=True)


def stage_clean(a):
    from absa.quality import clean_training
    train = read_jsonl(S / "train_public.jsonl")
    held = {name: read_jsonl(S / f"{name}.jsonl") for name in
            ("val", "absa_tr_test", "m_absa_test", "clothing_apc_test")
            if (S / f"{name}.jsonl").exists()}
    cleaned, report = clean_training(train, held)
    write_jsonl(S / "train_clean.jsonl", cleaned)
    (C.REPORTS / "cleaning_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[clean]", json.dumps(report["counts"], ensure_ascii=False), flush=True)


def stage_eda(a):
    from scripts.eda_data import run_eda
    run_eda(C.WORK, max_len=a.max_len, tokenizer_name=C.BASE_MODEL if a.eda_tokenizer else None)


# --------------------------------------------------------------------- synth
def find_gemini_raw():
    own = C.SYNTH / "gemini_absa_raw.jsonl"
    if own.exists() and own.stat().st_size > 0:
        return own
    for p in glob.glob("/kaggle/input/**/gemini_absa_raw*.jsonl", recursive=True):
        shutil.copy(p, own)
        print("[synth] using uploaded Gemini ABSA data:", p)
        return own
    return None


def stage_synth(a):
    from absa import synth_gemini, synth_local
    from absa.secrets import gemini_key

    held = set()
    for pat in ("val.jsonl", "*_test.jsonl", "test_*.jsonl"):
        for f in S.glob(pat):
            held |= {text_hash(d["text"]) for d in read_jsonl(f)}

    loc = synth_local.generate(n=120 if a.quick else a.local_n, seed=a.seed)
    seen, uniq = set(), []
    for d in loc:
        h = text_hash(d["text"])
        if h not in held and h not in seen:
            seen.add(h)
            uniq.append(d)
    # Synthetic text is training augmentation only. Do NOT contaminate real
    # validation or test sets with generated texts (optimistic scores).
    write_jsonl(C.SYNTH / "local_train.jsonl", uniq)
    print("[synth] local augmentation for training:", len(uniq), flush=True)
    from absa.hard_cases import curated_stress_set
    hard = [d for d in curated_stress_set() if text_hash(d["text"]) not in held
            and text_hash(d["text"]) not in seen]
    write_jsonl(S / "curated_stress_test.jsonl", hard)
    print("[synth] separate curated stress set (UNVERIFIED, not real-world benchmark):",len(hard), flush=True)

    raw = C.SYNTH / "gemini_absa_raw.jsonl"
    key = gemini_key() if not a.no_gemini else None
    if key:
        os.environ["GEMINI_API_KEY"] = key
        st = synth_gemini.generate(
            str(raw),
            total_calls=3 if a.quick else a.gemini_calls,
            per_call=a.gemini_per_call,
            workers=a.gemini_workers,
            seed=a.seed,
            model=os.environ.get("GEMINI_MODEL"),
        )
        print("[synth] gemini stats:", st, flush=True)

    r = find_gemini_raw()
    if r:
        gd = [d for d in synth_gemini.load(str(r)) if text_hash(d["text"]) not in held]
        seen_g = set()
        gd = [d for d in gd if not (text_hash(d["text"]) in seen_g or seen_g.add(text_hash(d["text"])))
              and text_hash(d["text"]) not in seen]
        write_jsonl(C.SYNTH / "gemini_train.jsonl", gd)
        print("[synth] Gemini generated train-only examples:", len(gd), flush=True)
    else:
        print("[synth] no Gemini data (optional secret unavailable)", flush=True)


# --------------------------------------------------------------------- train
def stage_train(a):
    import torch

    pub = read_jsonl(S / "train_clean.jsonl")
    extra = []
    if a.include_synthetic:
        for src in ("local", "gemini"):
            f = C.SYNTH / f"{src}_train.jsonl"
            if f.exists():
                extra += read_jsonl(f)
        max_synth = int(len(pub) * a.max_synth_ratio / max(1e-6, 1-a.max_synth_ratio))
        random.Random(a.seed).shuffle(extra)
        extra = extra[:max_synth]
        public_hashes = {text_hash(d["text"]) for d in pub}
        held_hashes = {text_hash(d["text"]) for f in
                       [S/"val.jsonl", S/"absa_tr_test.jsonl", S/"m_absa_test.jsonl"]
                       if f.exists() for d in read_jsonl(f)}
        extra = [d for d in extra if text_hash(d["text"]) not in public_hashes | held_hashes]
    full_rows = pub + extra
    random.Random(a.seed).shuffle(full_rows)
    train_path = S / "train_full.jsonl"
    write_jsonl(train_path, full_rows)

    val_rows = read_jsonl(S / "val.jsonl")
    # Validation is intentionally REAL/ORIGINAL public annotations only.
    val_path = S / "val_full.jsonl"
    write_jsonl(val_path, val_rows)

    print(f"[train] full dataset: {len(pub)} public + {len(extra)} synthetic edge cases (val={len(val_rows)})", flush=True)

    jobs = []
    variants = [("full", train_path)]
    if a.train_real_baseline:
        variants.append(("real", S / "train_clean.jsonl"))
    for variant, variant_train_path in variants:
        for task in ("ate", "apc"):
            name = f"{task}_{variant}"
            out = C.MODELS / name
            if (out / "final").exists() and not a.retrain:
                print(f"[train] {name} already trained, skipping", flush=True)
                continue
            cmd = [
                sys.executable, "-m", "absa.train",
                "--task", task,
                "--train", str(variant_train_path),
                "--val", str(val_path),
                "--out", str(out),
                "--epochs", "1" if a.quick else str(a.epochs),
                "--bs", str(a.bs),
                "--lr", str(a.lr),
                "--seed", str(a.seed),
                "--max_len", str(a.max_len),
                "--grad_accum", str(a.grad_accum),
            ]
            if a.quick:
                cmd += ["--max_train", "200"]
            jobs.append((name, cmd))

    ng = torch.cuda.device_count()
    print(f"[train] launching {len(jobs)} task(s) across {ng} GPU(s)", flush=True)
    i = 0
    while i < len(jobs):
        batch = jobs[i:i + max(1, ng)]
        procs = []
        for gi, (task, cmd) in enumerate(batch):
            env = {**os.environ, "TOKENIZERS_PARALLELISM": "false"}
            if ng:
                env["CUDA_VISIBLE_DEVICES"] = str(gi)
            log = open(C.LOGS / f"train_{task}.log", "w")
            procs.append((task, subprocess.Popen(cmd, cwd=str(C.ROOT), env=env, stdout=log, stderr=subprocess.STDOUT), log))
            print(f"[train] started {task.upper()} on GPU {gi if ng else 'cpu'}", flush=True)
        while any(p.poll() is None for _, p, _ in procs):
            time.sleep(45)
            for task, p, _ in procs:
                lines = open(C.LOGS / f"train_{task}.log", errors="ignore").read().replace("\r", "\n").strip().split("\n")
                tail = lines[-1] if lines else ""
                print(f"  [{task.upper()}] {tail[-150:]}", flush=True)
        for task, p, log in procs:
            log.close()
            if p.returncode != 0:
                print(open(C.LOGS / f"train_{task}.log", errors="ignore").read()[-3000:])
                raise SystemExit(f"training {task} failed")
        i += len(batch)


# ---------------------------------------------------------------------- eval
def _gold_apc_pairs(docs):
    texts, targets, labels, meta = [], [], [], []
    for d in docs:
        rows = []
        for a in d["aspects"]:
            pol = C.APC_L2I.get(a["polarity"])
            if pol is None:
                continue
            target = (a.get("span") or a.get("aspect") or "ürün").strip()
            rows.append((target, pol, a))
        label_sets = {}
        for target, pol, _ in rows:
            label_sets.setdefault(target.casefold(), set()).add(pol)
        seen = set()
        for target, pol, a in rows:
            key = target.casefold()
            if len(label_sets[key]) != 1 or key in seen:
                continue
            seen.add(key)
            texts.append(d["text"])
            targets.append(target)
            labels.append(pol)
            meta.append({
                "id": d["id"],
                "flags": d.get("flags", []),
                "register": d.get("register", "standard"),
                "implicit": a.get("span") is None and d.get("span_supervision", True),
            })
    return texts, targets, labels, meta


def stage_eval(a):
    from absa.infer import APCPredictor, ATEPredictor
    from absa.metrics import eval_apc, eval_ate, eval_e2e

    tests = {}
    for name in ("absa_tr_test", "m_absa_test", "clothing_apc_test"):
        f = S / f"{name}.jsonl"
        if f.exists():
            tests[name] = read_jsonl(f)

    systems = {}
    if not a.skip_baseline:
        systems["ytu_baseline"] = (C.BASELINE_ATE, C.BASELINE_APC)
    if (C.MODELS / "ate_real" / "final").exists() and (C.MODELS / "apc_real" / "final").exists():
        systems["ours_real_only"] = (str(C.MODELS / "ate_real" / "final"), str(C.MODELS / "apc_real" / "final"))
    if (C.MODELS / "ate_full" / "final").exists() and (C.MODELS / "apc_full" / "final").exists():
        systems["ours_full"] = (str(C.MODELS / "ate_full" / "final"), str(C.MODELS / "apc_full" / "final"))

    report = {}
    for sys_name, (ate_path, apc_path) in systems.items():
        print(f"[eval] evaluating system: {sys_name}", flush=True)
        ate = ATEPredictor(ate_path)
        apc = APCPredictor(apc_path)
        report[sys_name] = {}
        for tname, docs in tests.items():
            # 1. ATE evaluation on explicit aspects
            span_docs = [d for d in docs if d.get("span_supervision", True)]
            raw_spans = ate.extract_batch([d["text"] for d in span_docs], tau=0.45) if span_docs else []
            ate_spans = [[(s["start"], s["end"]) for s in sp] for sp in raw_spans]
            ate_m = eval_ate(span_docs, ate_spans) if span_docs else None

            # 2. APC evaluation on gold aspects (including implicit aspects!)
            texts, targets, y_true, meta = _gold_apc_pairs(docs)
            apc_skipped = sum(1 for d in docs for asp in d["aspects"]
                              if asp.get("polarity") in C.APC_L2I) - len(y_true)
            raw_pred, _, _ = apc.classify_pairs(texts, targets)
            # Compare actual label NAMES: checkpoints may store a different
            # numeric id2label order than the locally fine-tuned classifier.
            y_pred = [C.APC_L2I.get(apc.label_names[p], -1) for p in raw_pred]
            apc_m = eval_apc(y_true, y_pred, meta=meta)

            # 3. End-to-End (extracted span + predicted polarity)
            e2e_tuples = []
            for d, sp in zip(span_docs, raw_spans):
                if not sp:
                    e2e_tuples.append([])
                    continue
                t_list = [d["text"]] * len(sp)
                a_list = [s["span"] for s in sp]
                p_list, _, _ = apc.classify_pairs(t_list, a_list)
                e2e_tuples.append([(s["start"], s["end"], apc.label_names[p])
                                   for s, p in zip(sp, p_list) if apc.label_names[p] != "not_mentioned"])
            e2e_m = eval_e2e(span_docs, e2e_tuples) if span_docs else None

            entry = {"ate": ate_m, "apc": apc_m, "apc_ambiguous_or_duplicate_skipped": apc_skipped,
                     "e2e": e2e_m}
            if a.eval_implicit_limit and apc.model.config.num_labels == 4 and span_docs:
                from absa.metrics import eval_implicit
                from absa.infer import ABSAPipeline
                # Reuse loaded predictors (do not load a third copy on GPU).
                obj = ABSAPipeline.__new__(ABSAPipeline)
                obj.ate, obj.apc, obj.tau = ate, apc, 0.45
                obj.implicit_confidence, obj.absent_margin = 0.78, 0.12
                pos = [d for d in span_docs if any(a.get('span') is None for a in d['aspects'])]
                neg = [d for d in span_docs if not any(a.get('span') is None for a in d['aspects'])]
                random.Random(a.seed).shuffle(pos)
                random.Random(a.seed).shuffle(neg)
                half = a.eval_implicit_limit // 2
                subset = pos[:half] + neg[:a.eval_implicit_limit - min(half,len(pos))]
                # If a dataset has few implicit instances, backfill from negatives.
                if len(subset) < a.eval_implicit_limit:
                    already = {d['id'] for d in subset}
                    subset += [d for d in neg if d['id'] not in already][:a.eval_implicit_limit-len(subset)]
                pred = [obj.analyze(d["text"], include_implicit_fallback=True)["aspects"]
                        for d in subset]
                entry["implicit"] = eval_implicit(subset, pred)
            report[sys_name][tname] = entry
        del ate, apc
        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    json.dump(report, open(C.REPORTS / "results.json", "w"), indent=2, ensure_ascii=False)
    md = format_markdown(report)
    (C.REPORTS / "results.md").write_text(md, encoding="utf-8")
    print(md, flush=True)


def format_markdown(report):
    L = ["# Turkish ABSA Evaluation Results (ATE + APC + Figurative/Idiom/Gen-Z Slices)\n"]
    tests = sorted({t for r in report.values() for t in r})
    L.append("## 1. Overall Benchmark Summary\n")
    for tn in tests:
        L.append(f"\n### Test Set: `{tn}`\n")
        L.append("| System | ATE Exact-F1 | ATE Overlap-F1 | APC Gold Pairs | APC Skipped | APC Accuracy | APC Macro-F1 | E2E Explicit Tuple Overlap-F1 | Implicit Category+Polarity F1 (subset) |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for sys_name, res in report.items():
            if tn not in res:
                continue
            m = res[tn]
            L.append(
                f"| **{sys_name}** | {m['ate']['exact']['f1']:.3f} | {m['ate']['overlap']['f1']:.3f} | "
                if m['ate'] else f"| **{sys_name}** | N/A | N/A | "
            )
            L[-1] += (f"{m['apc']['n']} | {m['apc_ambiguous_or_duplicate_skipped']} | "
                      f"{m['apc']['accuracy']:.3f} | **{m['apc']['macro_f1']:.3f}** | "
                      f"{m['e2e']['overlap_f1']:.3f} | " if m['e2e'] else
                      f"{m['apc']['n']} | {m['apc_ambiguous_or_duplicate_skipped']} | "
                      f"{m['apc']['accuracy']:.3f} | **{m['apc']['macro_f1']:.3f}** | N/A | ")
            implicit = m.get("implicit")
            L[-1] += f"{implicit['f1']:.3f} |" if implicit else "N/A |"
    L.append("\n## 2. Challenging Linguistic Slices (APC Macro-F1 on Sarcasm / Idiom / Gen-Z / Negation / Implicit)\n")
    for tn in tests:
        has_slices = any(report[s].get(tn, {}).get("apc", {}).get("slices") for s in report)
        if not has_slices:
            continue
        L.append(f"\n### Slice Breakdown on `{tn}`\n")
        L.append("| System | Irony/Sarcasm | Idiom/Metaphor | Gen-Z Slang | Negation | Contrast | Implicit Aspect |")
        L.append("|---|---|---|---|---|---|---|")
        for sys_name, res in report.items():
            sl = res.get(tn, {}).get("apc", {}).get("slices", {})
            def fmt(k):
                return f"{sl[k]['macro_f1']:.3f} (n={sl[k]['n']})" if k in sl else "-"
            L.append(
                f"| **{sys_name}** | {fmt('irony')} | {fmt('idiom')} | {fmt('gen_z')} | "
                f"{fmt('negation')} | {fmt('contrast')} | {fmt('implicit')} |"
            )
    L.append("\n**Data note:** metrics use separate public test sets; their annotation protocols differ. ")
    L.append("Implicit scores for baseline 3-way APC are N/A; it has no open-set aspect detection. ")
    return "\n".join(L)


# -------------------------------------------------------------------- export
def stage_export(a):
    from absa.infer import ABSAPipeline

    ate_src = C.MODELS / "ate_full" / "final"
    apc_src = C.MODELS / "apc_full" / "final"
    if not (ate_src.exists() and apc_src.exists()):
        print("[export] trained models not found", flush=True)
        return
    dst_ate = C.EXPORT / "ate_model"
    dst_apc = C.EXPORT / "apc_model"
    for s, d in ((ate_src, dst_ate), (apc_src, dst_apc)):
        if d.exists():
            shutil.rmtree(d)
        shutil.copytree(s, d)

    pipe = ABSAPipeline(str(dst_ate), str(dst_apc), device="cpu")
    lines = ["# Inference examples from the public held-out ABSA-TR test split (CPU)\n"]
    test_docs = read_jsonl(S / "absa_tr_test.jsonl")
    sample = random.Random(a.seed).sample(test_docs, min(6, len(test_docs)))
    for doc in sample:
        text = doc["text"]
        r = pipe.analyze(text)
        lines.append(f"**Source ID:** `{doc['id']}`  \n**Review:** `{text}`\n")
        for asp in r["aspects"]:
            tag = "IMPLICIT" if asp["implicit"] else f"SPAN[{asp['start']}:{asp['end']}]"
            lines.append(f"- **{asp['aspect']}** ({tag}) → **{asp['polarity'].upper()}** (conf={asp['confidence']:.2f})")
        lines.append("")
    demo_md = "\n".join(lines)
    (C.REPORTS / "demo.md").write_text(demo_md, encoding="utf-8")
    print(demo_md, flush=True)
    zp = shutil.make_archive(str(C.WORK / "turkish_absa_models"), "zip", C.EXPORT)
    print("[export] zip:", zp, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all", "smoke", "prepare", "clean", "eda", "synth", "train", "eval", "export"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=float, default=5)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_synth_ratio", type=float, default=0.30)
    ap.add_argument("--include_synthetic", action="store_true", help="Opt in to existing synthetic training files")
    ap.add_argument("--lr", type=float, default=3.5e-5)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--eda_tokenizer", action="store_true", help="Profile exact ModernBERT token lengths during EDA")
    ap.add_argument("--local_n", type=int, default=2500)
    ap.add_argument("--gemini_calls", type=int, default=80)
    ap.add_argument("--gemini_per_call", type=int, default=8)
    ap.add_argument("--gemini_workers", type=int, default=4)
    ap.add_argument("--no_gemini", action="store_true")
    ap.add_argument("--no_clothing", action="store_true", help="Use only ABSA-TR + M-ABSA sources")
    ap.add_argument("--skip_baseline", action="store_true")
    ap.add_argument("--eval_implicit_limit", type=int, default=120)
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--train_real_baseline", action="store_true", help="Also train an otherwise identical real-data-only ablation")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--gemini_models", default="", help="Optional comma-separated priority model list for Gemini rotation")
    a = ap.parse_args()
    if a.gemini_models:
        os.environ["GEMINI_MODEL"] = a.gemini_models.strip()
    if not 0 <= a.max_synth_ratio < 0.8:
        ap.error("--max_synth_ratio must be >=0 and <0.8")

    steps = ["smoke", "prepare", "clean", "eda", "train", "eval", "export"] if a.stage == "all" else [a.stage]
    for s in steps:
        t0 = time.time()
        print(f"\n===== {s} =====", flush=True)
        globals()[f"stage_{s}"](a)
        print(f"===== {s} done in {time.time() - t0:.0f}s =====", flush=True)


if __name__ == "__main__":
    main()
