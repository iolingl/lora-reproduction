"""Re-score E2E predictions with the correct multi-reference protocol.

The E2E test CSV is exploded: one row per (meaning representation, reference).
Early runs wrote one prediction per row and scored each row against a single
reference, which understates BLEU.  This script groups rows by MR, keeps one
prediction per MR (beam search is deterministic, so they agree), and scores
against all references using ``nlg_metrics.compute_all_multi``.

Usage:
    python src/eval_e2e.py --run results/e2e/e2e_lora_r4
    python src/eval_e2e.py --run results/e2e/e2e_ft
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nlg_metrics import compute_all_multi


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return [(r["mr"].strip(), r["ref"].strip()) for r in csv.DictReader(f)]


def read_predictions(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run directory containing predictions.txt")
    ap.add_argument("--csv", default="data/e2e/testset_w_refs.csv")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    preds_path = os.path.join(args.run, "predictions.txt")
    rows = read_predictions(preds_path)
    out_path = args.out or os.path.join(args.run, "metrics_multi.json")

    if rows and "refs" in rows[0]:
        # predictions.txt already stores grouped multi-reference examples
        preds = [r["pred"] for r in rows]
        refs = [r["refs"] for r in rows]
        n_rows = len(rows)
        n_mr = len(rows)
        inconsistent = 0
    else:
        csv_rows = read_csv(args.csv)
        if len(csv_rows) != len(rows):
            raise SystemExit(f"{len(csv_rows)} CSV rows vs {len(rows)} predictions")
        order, by_mr = [], {}
        for (mr, ref), p in zip(csv_rows, rows):
            assert ref == p["ref"], "CSV and predictions are not row-aligned"
            if mr not in by_mr:
                order.append(mr)
                by_mr[mr] = {"refs": [], "preds": []}
            by_mr[mr]["refs"].append(ref)
            by_mr[mr]["preds"].append(p["pred"])
        inconsistent = sum(1 for v in by_mr.values() if len(set(v["preds"])) > 1)
        preds = [by_mr[m]["preds"][0] for m in order]
        refs = [by_mr[m]["refs"] for m in order]
        n_rows, n_mr = len(rows), len(order)

    metrics = compute_all_multi(preds, refs)
    out = dict(run=args.run, n_rows=n_rows, n_examples=n_mr,
               refs_per_example=sum(len(rs) for rs in refs) / len(refs),
               inconsistent_predictions=inconsistent, metrics=metrics)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
