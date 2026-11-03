"""E2E NLG Challenge reproduction with GPT-2 medium (Table 3 of the paper)."""
import argparse
import csv
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ensure_dir, peak_memory_mb, save_json, set_seed
from lora_lib import (count_lora, count_trainable, inject_lora_gpt2,
                      lora_parameters, mark_only_lora_trainable, merge_all)

IGNORE = -100


def read_e2e(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append((r["mr"].strip(), r["ref"].strip()))
    return rows


def read_e2e_grouped(path):
    """Group the exploded test CSV (one row per reference) by meaning representation."""
    order, refs = [], {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            mr, ref = r["mr"].strip(), r["ref"].strip()
            if mr not in refs:
                order.append(mr)
                refs[mr] = []
            refs[mr].append(ref)
    return [(mr, refs[mr]) for mr in order]


class E2EDataset(Dataset):
    def __init__(self, rows, tokenizer, max_len, train=True):
        self.items = []
        self.train = train
        for mr, ref in rows:
            prompt = tokenizer(mr + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
            target = tokenizer(" " + ref + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
            ids = (prompt + target)[:max_len]
            labels = ([IGNORE] * len(prompt) + target)[:max_len]
            self.items.append((ids, labels, mr))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ids, labels, mr = self.items[i]
        return {"input_ids": ids, "labels": labels, "mr": mr}


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        L = max(len(b["input_ids"]) for b in batch)
        ids, labels, attn = [], [], []
        for b in batch:
            pad = L - len(b["input_ids"])
            ids.append(b["input_ids"] + [self.pad_id] * pad)
            labels.append(b["labels"] + [IGNORE] * pad)
            attn.append([1] * len(b["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


def lm_loss(model, batch, label_smooth=0.0):
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    logits = out.logits[:, :-1].contiguous()
    targets = batch["input_ids"][:, 1:].contiguous()
    mask = batch["labels"][:, 1:].contiguous() != IGNORE
    logits = logits[mask]
    targets = targets[mask]
    return F.cross_entropy(logits, targets, label_smoothing=label_smooth)


def make_loader(rows, tokenizer, max_len, batch_size, shuffle, pad_id):
    ds = E2EDataset(rows, tokenizer, max_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=Collator(pad_id),
                      num_workers=2, pin_memory=True, drop_last=False)


@torch.no_grad()
def generate_predictions(model, tokenizer, rows, max_prompt_len, max_new_tokens=64,
                         beam=10, length_penalty=0.9, batch_size=8):
    model.eval()
    tokenizer.padding_side = "left"
    preds = []
    t0 = time.time()
    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        prompts = [mr + tokenizer.eos_token for mr, _ in chunk]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                        max_length=max_prompt_len).to(model.device)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            out = model.generate(
                **enc, do_sample=False, num_beams=beam, length_penalty=length_penalty,
                no_repeat_ngram_size=4, max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id,
            )
        gen = out[:, enc["input_ids"].shape[1]:]
        preds.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
        if (i // batch_size) % 50 == 0:
            print(f"  generate {i + len(chunk)}/{len(rows)} ({time.time()-t0:.0f}s)", flush=True)
    return [p.strip() for p in preds]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="lora", choices=["lora", "ft"])
    ap.add_argument("--model", default="gpt2-medium")
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=32)
    ap.add_argument("--targets", default="qv")
    ap.add_argument("--lora_dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--label_smooth", type=float, default=0.1)
    ap.add_argument("--max_len", type=int, default=160)
    ap.add_argument("--warmup_steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit_train", type=int, default=0)
    ap.add_argument("--gen_limit", type=int, default=0)
    ap.add_argument("--beam", type=int, default=10)
    ap.add_argument("--length_penalty", type=float, default=0.9)
    ap.add_argument("--output_dir", default="results/e2e")
    ap.add_argument("--tag", default="")
    ap.add_argument("--no_fp16", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    n_adapted = 0
    if args.method == "lora":
        n_adapted = inject_lora_gpt2(model, args.rank, int(args.alpha), args.targets,
                                     args.lora_dropout)
        mark_only_lora_trainable(model, extra_keywords=[])
    n_trainable = count_trainable(model)

    train_rows = read_e2e("data/e2e/trainset.csv")
    if args.limit_train:
        train_rows = train_rows[:args.limit_train]
    dev_rows = read_e2e("data/e2e/devset.csv")
    test_rows = read_e2e_grouped("data/e2e/testset_w_refs.csv")

    pad_id = tokenizer.eos_token_id
    train_loader = make_loader(train_rows, tokenizer, args.max_len, args.batch_size, True, pad_id)
    dev_loader = make_loader(dev_rows[:512], tokenizer, args.max_len, args.batch_size, False, pad_id)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay,
                                 betas=(0.9, 0.999))
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup = min(args.warmup_steps, max(1, total_steps // 10)) if args.warmup_steps else 0

    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    use_fp16 = not args.no_fp16
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    run_name = args.tag or f"e2e_{args.method}" + (
        f"_r{args.rank}_a{int(args.alpha)}" if args.method == "lora" else "") + f"_s{args.seed}"
    out_dir = os.path.join(args.output_dir, run_name)
    ensure_dir(out_dir)
    print(f"[{run_name}] trainable={n_trainable:,} lora={count_lora(model):,} "
          f"steps/epoch={steps_per_epoch} total={total_steps}", flush=True)

    history = []
    t0 = time.time()
    best_dev = 1e9
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, n = 0.0, 0
        ep_t0 = time.time()
        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_fp16):
                loss = lm_loss(model, batch, args.label_smooth)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            if not math.isfinite(loss.item()):
                print(f"ERROR non-finite loss at epoch {epoch} step {step}", flush=True)
                return
            running += loss.item() * batch["input_ids"].size(0)
            n += batch["input_ids"].size(0)
            if step % 200 == 0:
                print(f"  ep{epoch} step{step}/{steps_per_epoch} loss={running/n:.4f} "
                      f"lr={scheduler.get_last_lr()[0]:.2e} {(time.time()-ep_t0)/step:.2f}s/it",
                      flush=True)
        model.eval()
        dev_loss, dn = 0.0, 0
        with torch.no_grad():
            for batch in dev_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_fp16):
                    l = lm_loss(model, batch, 0.0)
                dev_loss += l.item() * batch["input_ids"].size(0)
                dn += batch["input_ids"].size(0)
        rec = dict(epoch=epoch, train_loss=running / max(1, n), dev_loss=dev_loss / max(1, dn),
                   epoch_time_s=time.time() - ep_t0)
        history.append(rec)
        print(f"[{run_name}] epoch {epoch}: train={rec['train_loss']:.4f} dev={rec['dev_loss']:.4f} "
              f"({rec['epoch_time_s']:.0f}s)", flush=True)
        best_dev = min(best_dev, rec["dev_loss"])

    # generation + metrics on the E2E test set
    gen_rows = test_rows[:args.gen_limit] if args.gen_limit else test_rows
    if args.method == "lora":
        merge_all(model)  # zero-latency inference, exactly as in the paper
    preds = generate_predictions(model, tokenizer, gen_rows, max_prompt_len=args.max_len,
                                 beam=args.beam, length_penalty=args.length_penalty)
    refs = [rs for _, rs in gen_rows]
    from nlg_metrics import compute_all_multi
    metrics = compute_all_multi(preds, refs)
    print(f"[{run_name}] TEST metrics {json.dumps(metrics)}", flush=True)

    with open(os.path.join(out_dir, "predictions.txt"), "w") as f:
        for p, rs in zip(preds, refs):
            f.write(json.dumps({"pred": p, "refs": rs}, ensure_ascii=False) + "\n")
    result = dict(
        run=run_name, method=args.method, model=args.model, rank=args.rank, alpha=args.alpha,
        targets=args.targets, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        max_len=args.max_len, seed=args.seed, n_trainable=n_trainable,
        n_lora=count_lora(model), n_adapted=n_adapted, n_eval=len(gen_rows),
        metrics=metrics, history=history, elapsed_s=time.time() - t0,
        peak_mem_mb=peak_memory_mb(), gpu=torch.cuda.get_device_name(0),
    )
    save_json(result, os.path.join(out_dir, "result.json"))
    print(f"[{run_name}] DONE metrics={metrics} time={(time.time()-t0)/60:.1f}min "
          f"peak={result['peak_mem_mb']:.0f}MB", flush=True)


if __name__ == "__main__":
    main()
