"""GLUE experiments for the LoRA reproduction (RoBERTa base/large)."""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ensure_dir, peak_memory_mb, save_json, set_seed
from lora_lib import (count_lora, count_trainable, inject_lora_roberta,
                      lora_parameters, mark_only_lora_trainable)

TASK_INFO = {
    "mnli": dict(keys=("premise", "hypothesis"), n_labels=3),
    "sst2": dict(keys=("sentence",), n_labels=2),
    "mrpc": dict(keys=("sentence1", "sentence2"), n_labels=2),
    "cola": dict(keys=("sentence",), n_labels=2),
    "qnli": dict(keys=("question", "sentence"), n_labels=2),
    "qqp": dict(keys=("question1", "question2"), n_labels=2),
    "rte": dict(keys=("sentence1", "sentence2"), n_labels=2),
    "stsb": dict(keys=("sentence1", "sentence2"), n_labels=1),
}


def tokenize_split(ds, tokenizer, keys, max_len, num_proc=2):
    def fn(batch):
        if len(keys) == 2:
            return tokenizer(batch[keys[0]], batch[keys[1]], truncation=True, max_length=max_len)
        return tokenizer(batch[keys[0]], truncation=True, max_length=max_len)

    drop = [c for c in ds.column_names if c != "label"]
    ds = ds.map(fn, batched=True, num_proc=num_proc, remove_columns=drop)
    if "label" in ds.column_names:
        ds = ds.rename_column("label", "labels")
    return ds


@torch.no_grad()
def evaluate(model, loader, task, device, fp16=True):
    model.eval()
    preds, labels = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=fp16 and device.type == "cuda"):
            logits = model(**batch).logits
        if task == "stsb":
            preds.append(logits.squeeze(-1).float().cpu().numpy())
        else:
            preds.append(logits.argmax(-1).cpu().numpy())
        labels.append(batch["labels"].cpu().numpy())
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    if task == "cola":
        from sklearn.metrics import matthews_corrcoef
        return float(matthews_corrcoef(labels, preds)), preds, labels
    if task == "stsb":
        from scipy.stats import pearsonr
        return float(pearsonr(labels, preds)[0]), preds, labels
    return float((preds == labels).mean()), preds, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASK_INFO))
    ap.add_argument("--model", default="roberta-base")
    ap.add_argument("--method", default="lora", choices=["lora", "ft", "bias"])
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=8)
    ap.add_argument("--targets", default="qv")
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit_train", type=int, default=0)
    ap.add_argument("--eval_batch_size", type=int, default=64)
    ap.add_argument("--output_dir", default="results/glue")
    ap.add_argument("--tag", default="")
    ap.add_argument("--init_lora", default="",
                    help="optional LoRA state dict (e.g. an MNLI checkpoint) to initialize from")
    ap.add_argument("--no_fp16", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda")
    info = TASK_INFO[args.task]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=info["n_labels"]).to(device)

    n_adapted = 0
    if args.method == "lora":
        n_adapted = inject_lora_roberta(model, args.rank, int(args.alpha), args.targets, args.dropout)
        mark_only_lora_trainable(model, extra_keywords=("classifier",))
        if args.init_lora:
            sd = torch.load(args.init_lora, map_location="cpu")
            sd = {k: v for k, v in sd.items() if "lora_A" in k or "lora_B" in k}
            missing, unexpected = model.load_state_dict(sd, strict=False)
            print(f"init LoRA from {args.init_lora}: {len(sd)} tensors, "
                  f"missing={len([m for m in missing if 'lora' in m])}, unexpected={len(unexpected)}",
                  flush=True)
    elif args.method == "bias":
        for n, p in model.named_parameters():
            p.requires_grad = ("bias" in n) or n.startswith("classifier")
    # ft: everything stays trainable

    n_trainable = count_trainable(model)
    n_lora = count_lora(model)
    pct = 100.0 * n_trainable / sum(p.numel() for p in model.parameters())

    raw = __import__("datasets").load_dataset("glue", args.task)
    train_ds = raw["train"]
    if args.limit_train and args.limit_train < len(train_ds):
        train_ds = train_ds.select(range(args.limit_train))
    train_ds = tokenize_split(train_ds, tokenizer, info["keys"], args.max_len)
    collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    if args.task == "mnli":
        eval_sets = {
            "matched": tokenize_split(raw["validation_matched"], tokenizer, info["keys"], args.max_len),
            "mismatched": tokenize_split(raw["validation_mismatched"], tokenizer, info["keys"], args.max_len),
        }
    else:
        eval_sets = {"validation": tokenize_split(raw["validation"], tokenizer, info["keys"], args.max_len)}
    eval_loaders = {
        k: DataLoader(v, batch_size=args.eval_batch_size, collate_fn=collator, shuffle=False,
                      num_workers=2, pin_memory=True)
        for k, v in eval_sets.items()
    }

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator,
        num_workers=4, pin_memory=True, drop_last=False)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup = int(args.warmup_ratio * total_steps)

    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=not args.no_fp16)
    use_fp16 = not args.no_fp16

    run_name = args.tag or (
        f"{args.task}_{os.path.basename(args.model)}_{args.method}"
        + (f"_r{args.rank}_{args.targets}" if args.method == "lora" else "")
        + f"_s{args.seed}")
    out_dir = os.path.join(args.output_dir, run_name)
    ensure_dir(out_dir)

    print(f"[{run_name}] trainable={n_trainable:,} lora={n_lora:,} ({pct:.3f}%) "
          f"steps/epoch={steps_per_epoch} total={total_steps}", flush=True)

    history = []
    best = -1e9
    best_state = None
    t0 = time.time()
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, nseen = 0.0, 0
        ep_t0 = time.time()
        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_fp16):
                loss = model(**batch).loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1
            if not math.isfinite(loss.item()):
                print(f"ERROR non-finite loss at step {global_step}", flush=True)
                return
            running += loss.item() * batch["input_ids"].size(0)
            nseen += batch["input_ids"].size(0)
            if step % 100 == 0:
                lr = scheduler.get_last_lr()[0]
                print(f"  ep{epoch} step{step}/{steps_per_epoch} loss={running/nseen:.4f} lr={lr:.2e} "
                      f"{(time.time()-ep_t0)/step:.2f}s/it", flush=True)

        metrics = {}
        for name, loader in eval_loaders.items():
            m, _, _ = evaluate(model, loader, args.task, device, use_fp16)
            metrics[name] = m
        if args.task == "mnli":
            score = (metrics["matched"] + metrics["mismatched"]) / 2
        else:
            score = metrics["validation"]
        rec = dict(epoch=epoch, train_loss=running / max(1, nseen),
                   lr=scheduler.get_last_lr()[0], eval=metrics, score=score,
                   epoch_time_s=time.time() - ep_t0)
        history.append(rec)
        print(f"[{run_name}] epoch {epoch}: {json.dumps(metrics)} score={score:.4f} "
              f"({rec['epoch_time_s']:.0f}s)", flush=True)
        if score > best:
            best = score
            if args.method == "lora":
                best_state = {n: p.detach().cpu().clone() for n, p in model.named_parameters()
                              if "lora_" in n or n.startswith("classifier")}
            else:
                best_state = None

    elapsed = time.time() - t0
    result = dict(
        run=run_name, task=args.task, model=args.model, method=args.method,
        rank=args.rank, alpha=args.alpha, targets=args.targets, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
        max_len=args.max_len, seed=args.seed, limit_train=args.limit_train,
        n_trainable=n_trainable, n_lora=n_lora, n_total=sum(p.numel() for p in model.parameters()),
        n_adapted=n_adapted, best_score=best, history=history,
        elapsed_s=elapsed, peak_mem_mb=peak_memory_mb(),
        gpu=torch.cuda.get_device_name(0),
    )
    save_json(result, os.path.join(out_dir, "result.json"))
    if best_state is not None:
        torch.save(best_state, os.path.join(out_dir, "best_lora.pt"))
    with open(os.path.join(out_dir, "train.log"), "w") as f:
        f.write(f"best={best:.4f}\n")
    print(f"[{run_name}] DONE best={best:.4f} time={elapsed/60:.1f}min "
          f"peak={result['peak_mem_mb']:.0f}MB", flush=True)


if __name__ == "__main__":
    main()
