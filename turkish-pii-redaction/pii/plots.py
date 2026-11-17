"""Generate publication-quality evaluation & training plots under work/reports/plots/."""
import glob
import json
from pathlib import Path

import numpy as np

from .config import MODELS, REPORTS


PALETTE = {
    "regex_only": "#7f8c8d",
    "ytu_baseline": "#e67e22",
    "clean": "#5dade2",
    "clean+hybrid": "#2874a6",
    "local": "#58d68d",
    "local+hybrid": "#239b56",
    "gemini": "#af7ac5",
    "gemini+hybrid": "#7d3c98",
    "full": "#ec7063",
    "full+hybrid": "#c0392b",
}


def _color(model_name, idx=0):
    if model_name in PALETTE:
        return PALETTE[model_name]
    fallback = ["#2e86de", "#10ac84", "#ee5253", "#f368e0", "#ff9f43", "#5f27cd"]
    return fallback[idx % len(fallback)]


def _short_test(name):
    return name.replace("(synthetic)", "").replace("_test", "")


def plot_all(results_dict=None, out_dir=None):
    """Render all evaluation and training charts to out_dir (default: work/reports/plots)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir or (REPORTS / "plots"))
    out_dir.mkdir(parents=True, exist_ok=True)

    if results_dict is None:
        rpath = REPORTS / "results.json"
        if not rpath.exists():
            return []
        results_dict = json.loads(rpath.read_text(encoding="utf-8"))

    results = results_dict.get("results", {})
    if not results:
        return []

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "figure.dpi": 180,
    })

    saved = []
    models = list(results.keys())
    tests = sorted({t for r in results.values() for t in r})
    short_tests = [_short_test(t) for t in tests]

    # --------------------------------------------------------- 1. Overlap-F2 bar chart (with 95% CI)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.8), sharey=True)
    for ax, (view, title) in zip(
        axes,
        [("cmp", "Comparable Taxonomy (11 YTU-supported classes)"),
         ("full", "Full Unified Taxonomy (All 13 PII classes)")],
    ):
        x = np.arange(len(tests))
        w = 0.82 / max(1, len(models))
        for i, mk in enumerate(models):
            vals, err_lo, err_hi = [], [], []
            for tn in tests:
                m = results.get(mk, {}).get(tn, {}).get(view)
                if m:
                    f2 = m["ov"]["f2"]
                    ci = m.get("ov_f2_ci", [f2, f2])
                    vals.append(f2)
                    err_lo.append(max(0.0, f2 - ci[0]))
                    err_hi.append(max(0.0, ci[1] - f2))
                else:
                    vals.append(0.0)
                    err_lo.append(0.0)
                    err_hi.append(0.0)
            offset = (i - (len(models) - 1) / 2) * w
            bars = ax.bar(
                x + offset, vals, width=w * 0.92, label=mk, color=_color(mk, i),
                yerr=[err_lo, err_hi], capsize=2.5, error_kw={"elinewidth": 0.9, "alpha": 0.7},
            )
            for b, v in zip(bars, vals):
                if v > 0.05:
                    ax.text(
                        b.get_x() + b.get_width() / 2, min(1.01, v + 0.015),
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7, rotation=90,
                    )
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(short_tests, rotation=20, ha="right")
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("Overlap-F2 (Recall-weighted)")
    axes[0].legend(loc="lower right", frameon=True, fontsize=8.5, ncol=2)
    fig.suptitle("Turkish PII Benchmark — Overlap-F2 with 95% Bootstrap CI", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    p1 = out_dir / "01_benchmark_ov_f2.png"
    fig.savefig(p1, bbox_inches="tight")
    plt.close(fig)
    saved.append(str(p1))

    # --------------------------------------------------------- 2. Exact-F1 comparison
    fig, ax = plt.subplots(figsize=(12, 5.2))
    x = np.arange(len(tests))
    w = 0.82 / max(1, len(models))
    for i, mk in enumerate(models):
        vals = [results.get(mk, {}).get(tn, {}).get("full", {}).get("ex", {}).get("f1", 0.0) for tn in tests]
        offset = (i - (len(models) - 1) / 2) * w
        bars = ax.bar(x + offset, vals, width=w * 0.92, label=mk, color=_color(mk, i))
        for b, v in zip(bars, vals):
            if v > 0.05:
                ax.text(
                    b.get_x() + b.get_width() / 2, min(1.01, v + 0.012),
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, rotation=90,
                )
    ax.set_title("Exact-Boundary Span F1 Across Evaluation Sets (Full Taxonomy)")
    ax.set_xticks(x)
    ax.set_xticklabels(short_tests, rotation=15, ha="right")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Exact-Match F1")
    ax.legend(loc="lower right", frameon=True, ncol=3, fontsize=9)
    fig.tight_layout()
    p2 = out_dir / "02_exact_f1_comparison.png"
    fig.savefig(p2, bbox_inches="tight")
    plt.close(fig)
    saved.append(str(p2))

    # --------------------------------------------------------- 3. Privacy safety: Char Leak Rate & False Redaction
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.2))
    x = np.arange(len(tests))
    w = 0.82 / max(1, len(models))
    for i, mk in enumerate(models):
        leaks = [results.get(mk, {}).get(tn, {}).get("full", {}).get("char_leak_rate", 0.0) * 100 for tn in tests]
        offset = (i - (len(models) - 1) / 2) * w
        ax1.bar(x + offset, leaks, width=w * 0.92, label=mk, color=_color(mk, i))
    ax1.set_title("PII Character Leakage Rate (%) — Lower is Better")
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_tests, rotation=20, ha="right")
    ax1.set_ylabel("Unredacted Gold PII Characters (%)")
    ax1.legend(loc="upper right", frameon=True, fontsize=8.5)

    neg_tests = [
        tn for tn in tests
        if any(results.get(mk, {}).get(tn, {}).get("full", {}).get("false_redaction_rate") is not None for mk in models)
    ]
    if neg_tests:
        xn = np.arange(len(neg_tests))
        for i, mk in enumerate(models):
            frs = [
                (results.get(mk, {}).get(tn, {}).get("full", {}).get("false_redaction_rate") or 0.0) * 100
                for tn in neg_tests
            ]
            offset = (i - (len(models) - 1) / 2) * w
            ax2.bar(xn + offset, frs, width=w * 0.92, label=mk, color=_color(mk, i))
        ax2.set_title("False Redaction Rate on Clean/Negative Docs (%) — Lower is Better")
        ax2.set_xticks(xn)
        ax2.set_xticklabels([_short_test(t) for t in neg_tests], rotation=20, ha="right")
        ax2.set_ylabel("Negative Docs with >=1 Spurious Span (%)")
    else:
        ax2.text(0.5, 0.5, "No negative-only docs in split", ha="center", va="center")
    fig.suptitle("Privacy vs. Utility Trade-off: Leakage & Over-Redaction", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    p3 = out_dir / "03_privacy_risk_leak_vs_false_redaction.png"
    fig.savefig(p3, bbox_inches="tight")
    plt.close(fig)
    saved.append(str(p3))

    # --------------------------------------------------------- 4. Per-class Overlap-F2 heatmap (macro across tests)
    all_classes = sorted({
        cls
        for mk in models
        for tn in tests
        for cls, st in results.get(mk, {}).get(tn, {}).get("full", {}).get("per_class", {}).items()
        if st.get("support", 0) > 0
    })
    if all_classes:
        mat = np.zeros((len(all_classes), len(models)), dtype=np.float64)
        for ci, cls in enumerate(all_classes):
            for mi, mk in enumerate(models):
                num = den = 0.0
                for tn in tests:
                    st = results.get(mk, {}).get(tn, {}).get("full", {}).get("per_class", {}).get(cls)
                    if st and st.get("support", 0) > 0:
                        sup = st["support"]
                        num += st["ov_f2"] * sup
                        den += sup
                mat[ci, mi] = num / den if den > 0 else 0.0

        fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.5), max(5.5, len(all_classes) * 0.48)))
        im = ax.imshow(mat, cmap="YlGnBu", aspect="auto", vmin=0.0, vmax=1.0)
        ax.set_xticks(np.arange(len(models)))
        ax.set_xticklabels(models, rotation=20, ha="right", fontweight="bold")
        ax.set_yticks(np.arange(len(all_classes)))
        ax.set_yticklabels(all_classes, fontweight="bold")
        ax.grid(False)
        for ci in range(len(all_classes)):
            for mi in range(len(models)):
                v = mat[ci, mi]
                ax.text(
                    mi, ci, f"{v:.3f}", ha="center", va="center",
                    color="white" if v > 0.65 else "#1a1a1a", fontsize=8.5, fontweight="bold",
                )
        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
        cbar.set_label("Support-Weighted Overlap-F2")
        ax.set_title("Per-Entity-Class Overlap-F2 Across All Test Corpora")
        fig.tight_layout()
        p4 = out_dir / "04_per_class_heatmap.png"
        fig.savefig(p4, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(p4))

    # --------------------------------------------------------- 5. Training progression & threshold calibration
    fig, (ax_tr, ax_cal) = plt.subplots(1, 2, figsize=(14, 5.0))
    has_train = False
    for i, vdir in enumerate(sorted(MODELS.glob("*"))):
        if not vdir.is_dir():
            continue
        vname = vdir.name
        hist = []
        ts_file = vdir / "train_summary.json"
        if ts_file.exists():
            try:
                hist = json.loads(ts_file.read_text(encoding="utf-8")).get("log_history", [])
            except Exception:
                pass
        if not hist:
            ckpts = sorted(glob.glob(str(vdir / "checkpoint-*" / "trainer_state.json")))
            if ckpts:
                try:
                    hist = json.loads(Path(ckpts[-1]).read_text(encoding="utf-8")).get("log_history", [])
                except Exception:
                    pass
        tr_steps = [(x["step"], x["loss"]) for x in hist if "loss" in x and "step" in x]
        ev_steps = [(x["step"], x["eval_f2"]) for x in hist if "eval_f2" in x and "step" in x]
        if tr_steps:
            has_train = True
            c = _color(vname, i)
            ax_tr.plot([s for s, _ in tr_steps], [l for _, l in tr_steps], label=f"{vname} train loss", color=c, alpha=0.45, linestyle="--")
        if ev_steps:
            has_train = True
            c = _color(vname + "+hybrid", i)
            ax_tr.plot([s for s, _ in ev_steps], [f for _, f in ev_steps], label=f"{vname} val F2", color=c, marker="o", linewidth=2)

    if has_train:
        ax_tr.set_title("Training Loss & Validation Span-F2 Progression")
        ax_tr.set_xlabel("Optimization Step")
        ax_tr.set_ylabel("Loss / Validation F2")
        ax_tr.legend(loc="best", frameon=True, fontsize=8.5)
    else:
        ax_tr.text(0.5, 0.5, "No training history found", ha="center", va="center")
        ax_tr.set_title("Training Progression")

    has_cal = False
    for i, vdir in enumerate(sorted(MODELS.glob("*"))):
        cal_file = vdir / "final" / "calibration.json"
        if cal_file.exists():
            try:
                cd = json.loads(cal_file.read_text(encoding="utf-8"))
                grid = cd.get("grid_ov_f2", {})
                tau_best = cd.get("tau")
                if grid:
                    has_cal = True
                    xs = sorted(float(k) for k in grid.keys())
                    ys = [grid[str(k)] if str(k) in grid else grid.get(k, 0.0) for k in xs]
                    c = _color(vdir.name + "+hybrid", i)
                    ax_cal.plot(xs, ys, marker="s", linewidth=2, color=c, label=f"{vdir.name} (optimal τ={tau_best})")
                    if tau_best is not None and str(tau_best) in grid:
                        ax_cal.scatter([float(tau_best)], [grid[str(tau_best)]], s=90, color=c, zorder=5)
            except Exception:
                pass
    if has_cal:
        ax_cal.set_title("Probability Threshold (τ) Calibration on Validation Split")
        ax_cal.set_xlabel("Non-O Probability Threshold τ")
        ax_cal.set_ylabel("Validation Overlap-F2")
        ax_cal.legend(loc="best", frameon=True, fontsize=9)
    else:
        ax_cal.text(0.5, 0.5, "No calibration grid found", ha="center", va="center")
        ax_cal.set_title("Threshold Calibration")

    fig.tight_layout()
    p5 = out_dir / "05_training_and_calibration.png"
    fig.savefig(p5, bbox_inches="tight")
    plt.close(fig)
    saved.append(str(p5))

    print(f"[plots] saved {len(saved)} charts to {out_dir}")
    return saved
