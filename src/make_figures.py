"""Generate all report figures from result files."""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "figure.dpi": 200, "savefig.bbox": "tight", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    "font.sans-serif": ["DejaVu Sans"],
    "axes.unicode_minus": False,
})
C_PAPER = "#4C72B0"
C_OURS = "#DD8452"
C_THIRD = "#55A868"


def load(path):
    with open(path) as f:
        return json.load(f)


def main_glue(root="results/glue"):
    tasks = ["mnli", "sst2", "mrpc", "cola", "qnli", "qqp", "rte", "stsb"]
    paper = dict(mnli=87.5, sst2=95.1, mrpc=89.7, cola=63.4, qnli=93.3,
                 qqp=90.8, rte=86.6, stsb=91.5)
    ours, ours_ft = {}, {}
    for t in tasks:
        p = os.path.join(root, f"{t}_roberta-base_lora_r8_qv_s0", "result.json")
        if os.path.exists(p):
            ours[t] = load(p)["best_score"] * 100
        p = os.path.join(root, f"ft_{t}", "result.json")
        if os.path.exists(p):
            ours_ft[t] = load(p)["best_score"] * 100
    xs = np.arange(len(tasks))
    w = 0.27
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.bar(xs - w, [paper[t] for t in tasks], w, label="Paper RoBERTa-base (LoRA, 5 seeds)", color=C_PAPER)
    ax.bar(xs, [ours.get(t, np.nan) for t in tasks], w, label="Ours (LoRA, 1 seed)", color=C_OURS)
    ax.bar(xs + w, [ours_ft.get(t, np.nan) for t in tasks], w, label="Ours (full FT)", color=C_THIRD)
    ax.set_xticks(xs)
    ax.set_xticklabels([t.upper() for t in tasks])
    ax.set_ylabel("Dev score")
    ax.set_ylim(55, 100)
    ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.14))
    ax.set_title("GLUE: LoRA vs. full fine-tuning", pad=22)
    for i, t in enumerate(tasks):
        if t in ours:
            ax.text(i, ours[t] + 0.6, f"{ours[t]:.1f}", ha="center", fontsize=6.5, color=C_OURS)
    fig.savefig("report/figures/fig_glue.pdf")
    plt.close(fig)


def rank_ablation(root="results/glue"):
    ranks = [1, 2, 4, 8, 64]
    vals = []
    for r in ranks:
        p = os.path.join(root, f"abl_rank{r}_sst2", "result.json")
        vals.append(load(p)["best_score"] * 100 if os.path.exists(p) else np.nan)
    fig, ax = plt.subplots(figsize=(4.0, 2.6))
    ax.plot(ranks, vals, "o-", color=C_OURS, lw=1.8, ms=5)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ranks)
    ax.set_xticklabels([str(r) for r in ranks])
    ax.set_xlabel("LoRA rank r (Wq, Wv)")
    ax.set_ylabel("SST-2 dev accuracy")
    for r, v in zip(ranks, vals):
        if not np.isnan(v):
            ax.annotate(f"{v:.1f}", (r, v), textcoords="offset points", xytext=(0, 6),
                        ha="center", fontsize=7)
    fig.savefig("report/figures/fig_rank.pdf")
    plt.close(fig)


def target_ablation(root="results/glue"):
    cfgs = [("q", 8), ("k", 8), ("v", 8), ("o", 8), ("qv", 4), ("qkvo", 2)]
    labels, vals = [], []
    for tgt, r in cfgs:
        p = os.path.join(root, f"abl_tgt_{tgt}_r{r}_sst2", "result.json")
        if os.path.exists(p):
            labels.append(f"{tgt} (r={r})")
            vals.append(load(p)["best_score"] * 100)
    fig, ax = plt.subplots(figsize=(4.4, 2.6))
    colors = [C_PAPER] * 4 + [C_OURS, C_THIRD]
    ax.bar(range(len(vals)), vals, color=colors[:len(vals)])
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("SST-2 dev accuracy")
    ax.set_ylim(min(vals) - 1.5 if vals else 90, max(vals) + 1.0 if vals else 96)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.1, f"{v:.1f}", ha="center", fontsize=7)
    ax.set_title("Adapted weight types at matched budget", pad=6)
    fig.savefig("report/figures/fig_targets.pdf")
    plt.close(fig)


def training_curves(root="results/glue"):
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6))
    for t, ax in [("sst2", axes[0]), ("mnli", axes[1])]:
        p = os.path.join(root, f"{t}_roberta-base_lora_r8_qv_s0", "result.json")
        if not os.path.exists(p):
            continue
        r = load(p)
        epochs = [h["epoch"] for h in r["history"]]
        scores = [h["score"] * 100 for h in r["history"]]
        ax.plot(epochs, scores, "o-", color=C_OURS, label="LoRA", ms=3)
        p = os.path.join(root, f"ft_{t}", "result.json")
        if os.path.exists(p):
            rf = load(p)
            ax.plot([h["epoch"] for h in rf["history"]], [h["score"] * 100 for h in rf["history"]],
                    "s--", color=C_THIRD, label="Full FT", ms=3)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Dev score")
        ax.set_title(t.upper())
        ax.legend(frameon=False, fontsize=7)
    fig.savefig("report/figures/fig_curves.pdf")
    plt.close(fig)


def placeholder(path, text):
    fig, ax = plt.subplots(figsize=(4.0, 2.4))
    ax.axis("off")
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=9)
    fig.savefig(path)
    plt.close(fig)


def subspace(root="results/analysis"):
    p = os.path.join(root, "subspace.npz")
    if not os.path.exists(p):
        placeholder("report/figures/fig_subspace.pdf", "subspace analysis running")
        return
    d = np.load(p)
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6))
    for ax, key, title in [(axes[0], "phi_q", "Wq"), (axes[1], "phi_v", "Wv")]:
        phi = d[key]
        im = ax.imshow(phi, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=1)
        ax.set_xlabel("j (top-j dirs of r=64 update)")
        ax.set_ylabel("i (top-i dirs of r=8 update)")
        ax.set_title(title)
        ax.set_xticks([0, 5, 11, 23, 35, 47, 63])
        ax.set_xticklabels([1, 6, 12, 24, 36, 48, 64])
        ax.set_yticks(range(8))
        ax.set_yticklabels(range(1, 9))
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.savefig("report/figures/fig_subspace.pdf")
    plt.close(fig)


def efficiency(root="results/analysis"):
    p = os.path.join(root, "efficiency.json")
    if not os.path.exists(p):
        placeholder("report/figures/fig_efficiency.pdf", "efficiency measurement running")
        return
    d = load(p)["inference_ms"]
    keys = list(d)
    labels = [k.replace("_", ", ").replace("b", "B=").replace("l", " L=") for k in keys]
    base = [d[k]["base"] for k in keys]
    unm = [d[k]["unmerged"] for k in keys]
    mer = [d[k]["merged"] for k in keys]
    xs = np.arange(len(keys)); w = 0.26
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    ax.bar(xs - w, base, w, label="base", color=C_PAPER)
    ax.bar(xs, unm, w, label="LoRA (unmerged)", color=C_OURS)
    ax.bar(xs + w, mer, w, label="LoRA (merged)", color=C_THIRD)
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Forward latency (ms)")
    ax.legend(frameon=False, fontsize=7)
    fig.savefig("report/figures/fig_efficiency.pdf")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs("report/figures", exist_ok=True)
    main_glue()
    rank_ablation()
    target_ablation()
    training_curves()
    subspace()
    efficiency()
    print("figures written")
