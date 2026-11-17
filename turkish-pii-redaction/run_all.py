"""Orchestrator: prepare -> synth -> train -> eval -> export.

  python run_all.py --stage all                # full run (Kaggle 2xT4)
  python run_all.py --stage synth              # Gemini + local synthetic data only (CPU, run anywhere)
  python run_all.py --stage all --quick        # tiny smoke test
"""
import argparse
import glob
import json
import os
import random
import shutil
import subprocess
import sys
import time

from pii import config as C
from pii.data import (bucket, chunk_doc, label_stats, load_nm_test, load_public, read_jsonl, text_hash,
                      write_jsonl)

S = C.SPLITS
VARIANTS = {"clean": [], "local": ["local"], "gemini": ["gemini"], "full": ["local", "gemini"]}


# ------------------------------------------------------------------- prepare
def stage_prepare(a):
    sp, stats = load_public(seed=a.seed, max_per_skel=a.max_per_skel)
    for k, v in sp.items():
        if a.quick:
            v = v[:300] if k == "train" else v[:120]
        name = "train_public" if k == "train" else k
        write_jsonl(S / f"{name}.jsonl", v)
    nm = load_nm_test(max_docs=10 if a.quick else a.nm_max_docs)
    write_jsonl(S / "test_nm_kvkk.jsonl", nm)
    man = {"stats": {k: (dict(v) if hasattr(v, "items") else v) for k, v in stats.items()},
           "sizes": {k: len(v) for k, v in sp.items()}, "nm_docs": len(nm),
           "train_label_dist": label_stats(sp["train"])[0], "seed": a.seed}
    json.dump(man, open(S / "manifest.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("[prepare]", json.dumps(man["sizes"]), "nm_kvkk_test_hard:", man["nm_docs"],
          "skeleton overlap:", stats["skeleton_overlap_with_train"])
    print("[prepare] train label dist:", man["train_label_dist"])


# --------------------------------------------------------------------- synth
def find_gemini_raw():
    own = C.SYNTH / "gemini_raw.jsonl"
    if own.exists() and own.stat().st_size > 0:
        return own
    bundled = C.ROOT / "gemini_raw.jsonl"
    if bundled.exists() and bundled.stat().st_size > 0:
        shutil.copy(bundled, own)
        print("[synth] seeded from bundled gemini_raw.jsonl:", bundled)
        return own
    for p in glob.glob("/kaggle/input/**/gemini_raw*.jsonl", recursive=True):
        shutil.copy(p, own)
        print("[synth] using uploaded Gemini data:", p)
        return own
    return None


def route(docs, cuts=(750, 850), key="group"):
    out = {"train": [], "val": [], "test": []}
    for d in docs:
        b = bucket(d[key], salt="synth")
        out["train" if b < cuts[0] else "val" if b < cuts[1] else "test"].append(d)
    return out


def stage_synth(a):
    from pii import synth_gemini, synth_local
    held = {text_hash(d["text"]) for f in S.glob("val.jsonl") for d in read_jsonl(f)}
    for pat in ("test_*.jsonl", "*_test.jsonl"):
        for f in S.glob(pat):
            held |= {text_hash(d["text"]) for d in read_jsonl(f)}
    loc = synth_local.generate(n_pos=100 if a.quick else a.local_pos, n_neg=30 if a.quick else a.local_neg, seed=a.seed)
    loc = [d for d in loc if text_hash(d["text"]) not in held]
    seen, uniq = set(), []
    for d in loc:
        h = text_hash(d["text"])
        if h not in seen:
            seen.add(h)
            uniq.append(d)
    parts = route(uniq)
    for k, v in parts.items():
        write_jsonl(C.SYNTH / f"local_{k}.jsonl", v)
    print("[synth] local:", {k: len(v) for k, v in parts.items()})

    # Seed from bundled or Kaggle input gemini_raw.jsonl first so generate() resumes instead of starting at 0
    find_gemini_raw()
    raw = C.SYNTH / "gemini_raw.jsonl"
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if api_key and not a.no_gemini:
        try:
            st = synth_gemini.generate(
                str(raw),
                total_calls=3 if a.quick else a.gemini_calls,
                per_call=a.gemini_per_call,
                workers=a.gemini_workers,
                seed=a.seed,
                api_key=api_key,
            )
            print("[synth] gemini stats:", st)
        except Exception as e:
            print(f"[synth] gemini warning (continuing with existing data): {e}")

    r = find_gemini_raw()
    if r:
        gd = [d for d in synth_gemini.load(str(r)) if text_hash(d["text"]) not in held]
        for d in gd:
            d["group"] = d["id"]
        parts = route(gd, cuts=(750, 850), key="id")
        for k, v in parts.items():
            write_jsonl(C.SYNTH / f"gemini_{k}.jsonl", v)
        print("[synth] gemini:", {k: len(v) for k, v in parts.items()})
        audit = random.Random(0).sample(gd, min(60, len(gd)))
        write_jsonl(C.REPORTS / "gemini_human_audit_sample.jsonl", audit)
    else:
        print("[synth] no Gemini data available")


# --------------------------------------------------------------------- train
def synth_train(parts):
    docs = []
    for p in parts:
        f = C.SYNTH / f"{p}_train.jsonl"
        if f.exists():
            docs += read_jsonl(f)
    return docs


def stage_train(a):
    import torch
    names = [v for v in a.variants.split(",") if v in VARIANTS]
    jobs = []
    pub = read_jsonl(S / "train_public.jsonl")
    for v in names:
        extra = synth_train(VARIANTS[v])
        if VARIANTS[v] and not extra:
            print(f"[train] skip variant '{v}': no synthetic data found")
            continue
        rows = pub + extra
        random.Random(a.seed).shuffle(rows)
        f = S / f"train_{v}.jsonl"
        write_jsonl(f, rows)
        print(f"[train] variant {v}: {len(pub)} public + {len(extra)} synthetic")
        out = C.MODELS / v
        if (out / "final").exists() and not a.retrain:
            print(f"[train] {v} already trained, skipping (use --retrain)")
            continue
        cmd = [sys.executable, "-m", "pii.train", "--train", str(f), "--val", str(S / "val.jsonl"),
               "--out", str(out), "--epochs", "1" if a.quick else str(a.epochs), "--bs", str(a.bs),
               "--lr", str(a.lr), "--seed", str(a.seed), "--max_len", str(a.max_len)]
        if a.quick:
            cmd += ["--max_train", "200"]
        jobs.append((v, cmd))
    ng = torch.cuda.device_count()
    print(f"[train] {len(jobs)} jobs, {ng} GPU(s)")
    i = 0
    while i < len(jobs):
        batch = jobs[i:i + max(1, ng)]
        procs = []
        for gi, (v, cmd) in enumerate(batch):
            env = {**os.environ, "TOKENIZERS_PARALLELISM": "false"}
            if ng:
                env["CUDA_VISIBLE_DEVICES"] = str(gi)
            log = open(C.LOGS / f"train_{v}.log", "w", encoding="utf-8")
            procs.append((v, subprocess.Popen(cmd, cwd=str(C.ROOT), env=env, stdout=log, stderr=subprocess.STDOUT), log))
            print(f"[train] started {v} on GPU {gi if ng else 'cpu'}")
        while any(p.poll() is None for _, p, _ in procs):
            time.sleep(60)
            for v, p, _ in procs:
                lines = open(C.LOGS / f"train_{v}.log", errors="ignore").read().replace("\r", "\n").strip().split("\n")
                tail = lines[-1] if lines else ""
                print(f"  [{v}] {tail[-160:]}", flush=True)
        for v, p, log in procs:
            log.close()
            if p.returncode != 0:
                print(open(C.LOGS / f"train_{v}.log", errors="ignore").read()[-3000:])
                raise SystemExit(f"training {v} failed")
        i += len(batch)


# ---------------------------------------------------------------------- eval
def _overlaps_ignore(sp, ignore):
    return any(min(sp[1], e) > max(sp[0], s) for s, e in ignore)


def stage_eval(a):
    from pii.infer import NerModel, calibrate_tau, decode, hybrid, regex_only, to_tuples
    from pii.metrics import evaluate
    from pii.plots import plot_all

    val = read_jsonl(S / "val.jsonl")
    edge_val = [
        d for p in ("local", "gemini")
        for d in (read_jsonl(C.SYNTH / f"{p}_val.jsonl") if (C.SYNTH / f"{p}_val.jsonl").exists() else [])
    ]
    tests = {}
    for p in ("*_test", "test_*"):
        for f in sorted(S.glob(f"{p}.jsonl")):
            tn = f.stem.replace("test_", "", 1)
            tests[tn] = read_jsonl(f)
    for p in ("local", "gemini"):
        f = C.SYNTH / f"{p}_test.jsonl"
        if f.exists() and os.path.getsize(f):
            tests[f"edge_{p}(synthetic)"] = read_jsonl(f)
    cmp_labels = set(C.YTU_MAP.values())
    models = {}
    if not a.skip_baseline:
        models["ytu_baseline"] = (C.BASELINE_MODEL, C.YTU_MAP, 0.5)
    for v in a.variants.split(","):
        d = C.MODELS / v / "final"
        if d.exists():
            models[v] = (str(d), None, None)

    gold = lambda docs: [[tuple(s) for s in d["spans"]] for d in docs]
    results, errors = {}, {}

    def score(name, docs, preds):
        ign = [d.get("ignore") for d in docs]
        if any(ign):
            preds = [[p for p in ps if not _overlaps_ignore(p, ig or [])] for ps, ig in zip(preds, ign)]
        g = gold(docs)
        return {"full": evaluate(g, preds), "cmp": evaluate(g, preds, restrict=cmp_labels)}, preds

    for tn, docs in tests.items():
        results.setdefault("regex_only", {})[tn], _ = score(tn, docs, [to_tuples(regex_only(d["text"])) for d in docs])
    sel_scores, taus = {}, {}
    for mk, (path, lmap, tau) in models.items():
        print(f"[eval] {mk}", flush=True)
        nm = NerModel(path, label_map=lmap)
        bs = 8 if a.quick else 64
        vp = None
        if tau is None:
            vp = nm.token_probs([d["text"] for d in val], batch_size=bs)
            tau, hist = calibrate_tau(vp, [d["text"] for d in val], gold(val), evaluate=evaluate)
            json.dump({"tau": tau, "grid_ov_f2": hist}, open(C.MODELS / mk / "final" / "calibration.json", "w", encoding="utf-8"))
        taus[mk] = tau
        if mk != "ytu_baseline":
            if vp is None:
                vp = nm.token_probs([d["text"] for d in val], batch_size=bs)
            vv = evaluate(gold(val), [to_tuples(decode(d["text"], o, p, tau)) for d, (o, p) in zip(val, vp)], bootstrap=0)
            sc = [vv["ov"]["f2"]]
            if edge_val:
                ep = nm.token_probs([d["text"] for d in edge_val], batch_size=bs)
                ev = evaluate(gold(edge_val), [to_tuples(decode(d["text"], o, p, tau)) for d, (o, p) in zip(edge_val, ep)], bootstrap=0)
                sc.append(ev["ov"]["f2"])
            sel_scores[mk] = sum(sc) / len(sc)
        for tn, docs in tests.items():
            tp = nm.token_probs([d["text"] for d in docs], batch_size=bs)
            neural = [decode(d["text"], o, p, tau) for d, (o, p) in zip(docs, tp)]
            results.setdefault(mk, {})[tn], pn = score(tn, docs, [to_tuples(x) for x in neural])
            if mk != "ytu_baseline":
                hyb = [hybrid(d["text"], x) for d, x in zip(docs, neural)]
                results.setdefault(mk + "+hybrid", {})[tn], ph = score(tn, docs, [to_tuples(x) for x in hyb])
                errors[(mk, tn)] = (docs, ph)
        del nm
        import torch, gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    res_payload = {"results": results, "tau": taus, "selection": sel_scores}
    json.dump(res_payload, open(C.REPORTS / "results.json", "w", encoding="utf-8"), indent=1)
    # error dumps
    for (mk, tn), (docs, preds) in errors.items():
        rows = []
        for d, p in zip(docs, preds):
            g, pp = {tuple(s) for s in d["spans"]}, {tuple(x) for x in p}
            if g != pp:
                rows.append({"text": d["text"][:600], "missed": sorted(g - pp), "spurious": sorted(pp - g)})
        write_jsonl(C.REPORTS / f"errors_{mk}_{tn.split('(')[0]}.jsonl", rows[:80])
    plot_files = plot_all(res_payload, C.PLOTS)
    write_markdown(results, taus, sel_scores, plot_files)
    print(open(C.REPORTS / "results.md", encoding="utf-8").read())


def write_markdown(results, taus, sel, plot_files=None):
    nm_docs = results.get("regex_only", {}).get("nm_kvkk", {}).get("full", {}).get("n_docs")
    L = ["# Results (document bootstrap 95% CI on overlap-F2)\n",
         "The YTU baseline uses a fixed 0.5 threshold and neural spans only; project variants use validation-calibrated thresholds and are also shown with a rule layer. These are not equal-postprocessing comparisons.\n",
         f"NewMind nm_kvkk uses spans/test_hard.jsonl ({nm_docs} documents), not the full test split. Scores cover only mapped labels and ignore unmapped gold regions.\n",
         f"tau (project variants calibrated on public val): {taus}\n", f"variant-selection score (val + synthetic val): {sel}\n"]
    if plot_files:
        L.append("\n## Evaluation & Training Charts (`work/reports/plots/`)\n")
        for pf in plot_files:
            fn = os.path.basename(pf)
            L.append(f"- `plots/{fn}`")
        L.append("")
    tests = sorted({t for r in results.values() for t in r})
    for view, title in (("cmp", "Shared mapped labels (different post-processing)"), ("full", "Mapped project taxonomy")):
        L.append(f"\n## {title}\n")
        for tn in tests:
            L.append(f"\n### {tn}\n\n| model | ov-F2 [95% CI] | ov-recall | ov-precision | exact-F1 | uncovered-gold-char | missed-span | false-redaction |\n|---|---|---|---|---|---|---|---|")
            for mk, r in results.items():
                m = r.get(tn, {}).get(view)
                if not m:
                    continue
                ci = m.get("ov_f2_ci", [0, 0])
                fr = m["false_redaction_rate"]
                L.append(f"| {mk} | {m['ov']['f2']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] | {m['ov']['recall']:.3f} | {m['ov']['precision']:.3f} | "
                         f"{m['ex']['f1']:.3f} | {m['char_leak_rate']:.4f} | {m['missed_span_rate']:.4f} | "
                         f"{'-' if fr is None else f'{fr:.3f}'} |")
    (C.REPORTS / "results.md").write_text("\n".join(L), encoding="utf-8")


# -------------------------------------------------------------------- export
DEMO = [
    "Merhaba, ben Elif Kaya. TC kimlik numaram 10000000146, IBAN'ım TR330006100519786457841326, cep 0532 111 22 33.",
    "Kartım 4532 0151 1283 0366 ile ödeme yaptım, doğrulama kodu 482913 geldi ama olmadı, mail: elif.kaya@gmail.com'a da gelmedi.",
    "Kargo Ahmet Yılmaz'a teslim edilecek, adres: Atatürk Mah. Gül Sok. No:12 D:4, Kadıköy/İstanbul.",
    "Kampanya 29 Ekim'e kadar geçerli, ürün kodu 48291736, fiyat 1.250,50 TL, detaylar https://www.turkiye.gov.tr adresinde.",
    "selam abi tc 100000001 46 yanlis yazilmis duzeltin ben mehmet demir 05551234567",
    "Plakası 34 ABC 123 olan araç Zeynep Öztürk'ün, doğum tarihi 14.03.1988.",
]


def stage_export(a):
    from pii.infer import Pipeline
    res = json.load(open(C.REPORTS / "results.json", encoding="utf-8"))
    sel = res.get("selection") or {}
    best = max(sel, key=sel.get) if sel else a.variants.split(",")[0]
    src = C.MODELS / best / "final"
    dst = C.EXPORT / "model"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    if C.PLOTS.exists():
        shutil.copytree(C.PLOTS, dst / "plots", dirs_exist_ok=True)
    print(f"[export] best variant by validation: {best} -> {dst}")
    pipe = Pipeline(str(dst), device="cpu")
    lat, lines = [], ["# Demo (hybrid pipeline, CPU)\n"]
    for t in DEMO:
        r = pipe.redact(t)
        lines += [f"**IN**  {t}\n\n**OUT** {r['redacted_text']}\n", "```\n" + "\n".join(
            f"{e['label']:<15}{e['surface']!r:<40}{e['source']:<7}{e['score']:.2f}" for e in r["entities"]) + "\n```\n"]
    bench = []
    for f in ("btx24_test.jsonl", "belgin_test.jsonl", "yusuf_test.jsonl", "test_nm_kvkk.jsonl"):
        if (S / f).exists():
            bench += [d["text"] for d in read_jsonl(S / f)[:30]]
    for t in bench:
        t0 = time.perf_counter()
        pipe.detect(t)
        lat.append((time.perf_counter() - t0) * 1000)
    if lat:
        import numpy as np
        lines.append(f"\nCPU latency (batch=1, {len(lat)} docs): p50={np.percentile(lat, 50):.1f} ms, "
                     f"p95={np.percentile(lat, 95):.1f} ms\n")
    (C.REPORTS / "demo.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    card = f"""---
language: tr
license: apache-2.0
base_model: {C.BASE_MODEL}
pipeline_tag: token-classification
tags: [pii, ner, turkish, privacy, kvkk]
---
# Turkish PII detector ({best})
Fine-tuned from `{C.BASE_MODEL}` on public synthetic Turkish PII corpora (BTX24, yusuf-said, belgin, fevzi; see licenses
in the project README) plus rule-based and Gemini-generated *synthetic* edge cases (train split only).
Labels: {', '.join(C.LABELS)}. Use with the hybrid pipeline (`pii.infer.Pipeline`): calibrated threshold
(`calibration.json`) + checksum-validated regex fallback. Redaction is not legal anonymization (KVKK);
training data is synthetic - validate on your own data before production.
"""
    (dst / "README.md").write_text(card, encoding="utf-8")
    shutil.copy(C.REPORTS / "results.md", dst / "results.md")
    zp = shutil.make_archive(str(C.WORK / "turkish_pii_model"), "zip", dst)
    print("[export] zip:", zp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all", "prepare", "synth", "train", "eval", "export"])
    ap.add_argument("--variants", default="clean,full")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=float, default=4)
    ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--max_per_skel", type=int, default=4)
    ap.add_argument("--nm_max_docs", type=int, default=0)
    ap.add_argument("--local_pos", type=int, default=5000)
    ap.add_argument("--local_neg", type=int, default=1500)
    ap.add_argument("--gemini_calls", type=int, default=120)
    ap.add_argument("--gemini_per_call", type=int, default=10)
    ap.add_argument("--gemini_workers", type=int, default=8)
    ap.add_argument("--no_gemini", action="store_true")
    ap.add_argument("--skip_baseline", action="store_true")
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    steps = ["prepare", "synth", "train", "eval", "export"] if a.stage == "all" else [a.stage]
    for s in steps:
        t0 = time.time()
        print(f"\n===== {s} =====", flush=True)
        globals()[f"stage_{s}"](a)
        print(f"===== {s} done in {time.time() - t0:.0f}s =====", flush=True)


if __name__ == "__main__":
    main()
