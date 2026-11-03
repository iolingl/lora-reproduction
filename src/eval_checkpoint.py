"""Evaluate a released official LoRA checkpoint with our implementation."""
import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lora_lib import inject_lora_roberta, mark_only_lora_trainable, merge_all
from train_glue import TASK_INFO, evaluate, tokenize_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="roberta-base")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--task", default="mnli")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--out", default="results/analysis/official_mnli.json")
    args = ap.parse_args()

    import datasets
    info = TASK_INFO[args.task]
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=info["n_labels"]).cuda()
    inject_lora_roberta(model, args.rank, int(args.alpha), "qv")
    mark_only_lora_trainable(model)
    sd = torch.load(args.ckpt, map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("missing:", [k for k in missing if "lora" not in k][:10], "unexpected:", unexpected[:5])
    model.eval()

    raw = datasets.load_dataset("glue", args.task)
    if args.task == "mnli":
        sets = {"matched": raw["validation_matched"], "mismatched": raw["validation_mismatched"]}
    else:
        sets = {"validation": raw["validation"]}
    coll = DataCollatorWithPadding(tok, pad_to_multiple_of=8)
    scores = {}
    for name, ds in sets.items():
        ds = tokenize_split(ds, tok, info["keys"], args.max_len, num_proc=1)
        loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=coll, num_workers=2)
        score, _, _ = evaluate(model, loader, args.task, torch.device("cuda"))
        scores[name] = score * 100
        print(name, score * 100, flush=True)
    if args.task == "mnli":
        print("MNLI avg", np.mean(list(scores.values())))
    import json
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(dict(ckpt=args.ckpt, alpha=args.alpha, scores=scores,
                   avg=float(np.mean(list(scores.values())))), open(args.out, "w"), indent=2)
    merge_all(model)
    print("merged weights; sanity eval:", end=" ", flush=True)
    for name, ds in sets.items():
        ds = tokenize_split(ds, tok, info["keys"], args.max_len, num_proc=1)
        loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=coll, num_workers=2)
        print(name, evaluate(model, loader, args.task, torch.device("cuda"))[0] * 100, end="; ")
    print()


if __name__ == "__main__":
    main()
