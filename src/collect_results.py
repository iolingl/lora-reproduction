"""Collect result.json files into a compact table."""
import glob
import json
import os
import sys

TASKS = ["mnli", "sst2", "mrpc", "cola", "qnli", "qqp", "rte", "stsb"]
PAPER_LORA = {"mnli": 87.5, "sst2": 95.1, "mrpc": 89.7, "cola": 63.4,
              "qnli": 93.3, "qqp": 90.8, "rte": 86.6, "stsb": 91.5, "avg": 87.2}
PAPER_FT = {"mnli": 87.6, "sst2": 94.8, "mrpc": 90.2, "cola": 63.6,
            "qnli": 92.8, "qqp": 91.9, "rte": 78.7, "stsb": 91.2, "avg": 86.4}


def load(pattern):
    out = []
    for f in sorted(glob.glob(pattern)):
        try:
            out.append(json.load(open(f)))
        except Exception as e:
            print("skip", f, e)
    return out


def main(root="results"):
    glue = load(os.path.join(root, "glue", "*", "result.json"))
    rows = {}
    for r in glue:
        rows.setdefault(r["task"], []).append(r)

    print("== GLUE ==")
    header = f"{'task':6s} {'method':6s} {'cfg':16s} {'ep':>3s} {'trainable':>10s} {'score':>7s} {'paper':>6s} {'time':>6s}"
    print(header)
    summary = {}
    for t in TASKS:
        for r in sorted(rows.get(t, []), key=lambda x: (x["method"], x.get("rank", 0))):
            cfg = f"r{r['rank']}/{r['targets']}" if r["method"] == "lora" else "-"
            print(f"{t:6s} {r['method']:6s} {cfg:16s} {r['epochs']:3d} "
                  f"{r['n_trainable']:10,d} {r['best_score']*100:7.2f} "
                  f"{PAPER_LORA.get(t, float('nan')):6.1f} {r['elapsed_s']/60:5.1f}m")
        if t in rows:
            best = max(rows[t], key=lambda x: x["best_score"])
            summary[t] = dict(score=best["best_score"] * 100, method=best["method"],
                              cfg=f"r{best['rank']}/{best['targets']}")
    if "mnli" in summary:
        lora = [summary[t]["score"] for t in TASKS if t in summary]
        print("mean over available tasks:", sum(lora) / len(lora))
    with open(os.path.join(root, "glue_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    e2e = load(os.path.join(root, "e2e", "*", "result.json"))
    if e2e:
        print("\n== E2E ==")
        for r in e2e:
            print(r["run"], r["metrics"], f"{r['elapsed_s']/60:.1f}m", r["n_trainable"])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
