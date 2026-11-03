"""Efficiency measurements: inference latency (merge vs. unmerged) and training cost."""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lora_lib import (inject_lora_roberta, mark_only_lora_trainable, merge_all,
                      unmerge_all)


@torch.no_grad()
def latency(model, ids, mask, trials=20, warmup=5):
    for _ in range(warmup):
        model(input_ids=ids, attention_mask=mask)
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(trials):
        model(input_ids=ids, attention_mask=mask)
    torch.cuda.synchronize()
    return (time.time() - t) / trials * 1000  # ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/analysis/efficiency.json")
    args = ap.parse_args()
    from transformers import AutoModelForSequenceClassification
    base_model = AutoModelForSequenceClassification.from_pretrained(
        "roberta-base", num_labels=2).cuda().eval()
    model = AutoModelForSequenceClassification.from_pretrained(
        "roberta-base", num_labels=2).cuda().eval()
    inject_lora_roberta(model, 8, 8, "qv")
    mark_only_lora_trainable(model)
    model.eval()
    res = {"inference_ms": {}}
    import statistics
    for batch, seq in [(1, 32), (1, 128), (32, 128)]:
        ids = torch.randint(0, 1000, (batch, seq), device="cuda")
        mask = torch.ones_like(ids)
        samples = {"base": [], "unmerged": [], "merged": []}
        # interleave the three variants in rotating order to cancel clock/thermal drift
        orders = [["base", "unmerged", "merged"], ["merged", "base", "unmerged"],
                  ["unmerged", "merged", "base"], ["base", "merged", "unmerged"],
                  ["merged", "unmerged", "base"]]
        for order in orders:
            for name in order:
                if name == "base":
                    model.eval()
                    samples[name].append(latency(base_model, ids, mask))
                elif name == "unmerged":
                    unmerge_all(model)
                    samples[name].append(latency(model, ids, mask))
                else:
                    merge_all(model)
                    samples[name].append(latency(model, ids, mask))
        base_ms = statistics.median(samples["base"])
        unmerged_ms = statistics.median(samples["unmerged"])
        merged_ms = statistics.median(samples["merged"])
        res["inference_ms"][f"b{batch}_l{seq}"] = dict(base=base_ms, unmerged=unmerged_ms,
                                                       merged=merged_ms)
        print(f"b{batch} l{seq}: base={base_ms:.3f} unmerged={unmerged_ms:.3f} "
              f"merged={merged_ms:.3f} ms", flush=True)

    # one training step: LoRA vs full fine-tuning, same batch
    del base_model, model
    torch.cuda.empty_cache()
    from transformers import AutoModelForSequenceClassification as AM
    out = {}
    for mode in ["lora", "ft"]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        m = AM.from_pretrained("roberta-base", num_labels=2).cuda()
        if mode == "lora":
            inject_lora_roberta(m, 8, 8, "qv")
            mark_only_lora_trainable(m)
        params = [p for p in m.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=1e-4)
        ids = torch.randint(0, 1000, (16, 256), device="cuda")
        mask = torch.ones_like(ids)
        y = torch.zeros(16, dtype=torch.long, device="cuda")
        scaler = torch.amp.GradScaler("cuda")
        for _ in range(3):
            with torch.amp.autocast("cuda", dtype=torch.float16):
                loss = m(input_ids=ids, attention_mask=mask, labels=y).loss
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); opt.zero_grad()
        torch.cuda.synchronize(); t = time.time()
        for _ in range(10):
            with torch.amp.autocast("cuda", dtype=torch.float16):
                loss = m(input_ids=ids, attention_mask=mask, labels=y).loss
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); opt.zero_grad()
        torch.cuda.synchronize()
        out[mode] = dict(step_ms=(time.time() - t) / 10 * 1000,
                         peak_mem_mb=torch.cuda.max_memory_allocated() / 1024 ** 2,
                         trainable=sum(p.numel() for p in m.parameters() if p.requires_grad))
        print(mode, out[mode], flush=True)
        del m, opt
        torch.cuda.empty_cache()
    res["train_step"] = out
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print("saved", args.out)


if __name__ == "__main__":
    main()
