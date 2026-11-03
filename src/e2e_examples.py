"""Build a small qualitative table of E2E generations (LoRA vs full FT)."""
import csv
import json
import os

ROOT = "results/e2e"
CSV = "data/e2e/testset_w_refs.csv"
OUT = "report/tables/e2e_examples.tex"


def load(path):
    """Return a list of (refs, prediction), one entry per unique MR."""
    with open(path) as f:
        rows = [json.loads(line) for line in f]
    if rows and "refs" in rows[0]:
        return [(r["refs"], r["pred"]) for r in rows]
    with open(CSV, newline="", encoding="utf-8") as f:
        csv_rows = [(r["mr"].strip(), r["ref"].strip()) for r in csv.DictReader(f)]
    order, by_mr = [], {}
    for (mr, ref), row in zip(csv_rows, rows):
        if mr not in by_mr:
            order.append(mr)
            by_mr[mr] = {"refs": [], "pred": row["pred"]}
        by_mr[mr]["refs"].append(ref)
    return [(by_mr[m]["refs"], by_mr[m]["pred"]) for m in order]


def main():
    lp = os.path.join(ROOT, "e2e_lora_r4", "predictions.txt")
    fp = os.path.join(ROOT, "e2e_ft", "predictions.txt")
    if not (os.path.exists(lp) and os.path.exists(fp)):
        with open(OUT, "w") as f:
            f.write("% predictions not available yet\n")
        return
    lora, ft = load(lp), load(fp)
    # pick a few fluent, medium-length examples where the two outputs differ
    picks, last = [], -1000
    for i, ((refs, p), (_, q)) in enumerate(zip(lora, ft)):
        r = refs[0]
        if not p or not q or p == q or not (8 <= len(r.split()) <= 18):
            continue
        if i - last < 200:  # spread examples across the test set for variety
            continue
        last = i
        picks.append((i, r, p, q))
        if len(picks) == 3:
            break

    def esc(s):
        return s.replace("\\", "\\textbackslash{}").replace("&", "\\&").replace("%", "\\%") \
                .replace("_", "\\_").replace("#", "\\#").replace("{", "\\{").replace("}", "\\}")

    lines = [r"\begin{tabularx}{\textwidth}{c X X X}", r"\toprule",
             r"ID & 参考文本（其一） & LoRA 生成 & 全量微调生成 \\", r"\midrule"]
    for i, r, p, q in picks:
        lines.append(f"{i} & {esc(r)} & {esc(p)} & {esc(q)} \\\\")
    lines += [r"\bottomrule", r"\end{tabularx}"]
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", OUT, "with", len(picks), "examples")


if __name__ == "__main__":
    main()
