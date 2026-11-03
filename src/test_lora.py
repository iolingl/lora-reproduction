"""Unit checks: init identity, merge equivalence, parameter accounting."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lora_lib import (LoRALinear, count_lora, inject_lora_gpt2, inject_lora_roberta,
                      mark_only_lora_trainable, merge_all, unmerge_all)


def test_linear():
    torch.manual_seed(0)
    base = torch.nn.Linear(16, 8).cuda()
    x = torch.randn(4, 16, device="cuda")
    ref = base(x)
    layer = LoRALinear(base, r=4, alpha=8).cuda()
    assert torch.allclose(layer(x), ref, atol=0), "LoRA must be identity at init"
    with torch.no_grad():
        layer.lora_B.normal_()
    y_unmerged = layer(x)
    layer.merge()
    y_merged = layer(x)
    assert torch.allclose(y_unmerged, y_merged, atol=1e-5), (y_unmerged - y_merged).abs().max()
    layer.unmerge()
    assert torch.allclose(layer(x), y_unmerged, atol=1e-5)
    print("linear OK: init identity, merge/unmerge exact, lora params",
          sum(p.numel() for p in layer.parameters() if p.requires_grad))


def test_roberta():
    from transformers import AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2).cuda()
    n = inject_lora_roberta(model, 8, 8, "qv")
    mark_only_lora_trainable(model)
    assert n == 24 and count_lora(model) == 294912, (n, count_lora(model))
    model.eval()
    ids = torch.randint(0, 1000, (2, 32), device="cuda")
    with torch.no_grad():
        a = model(input_ids=ids).logits
        merge_all(model)
        b = model(input_ids=ids).logits
    print("roberta OK: adapted", n, "lora params", count_lora(model),
          "max diff", (a - b).abs().max().item())
    assert torch.allclose(a, b, atol=1e-5)


def test_gpt2():
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("gpt2-medium").cuda()
    n = inject_lora_gpt2(model, 4, 32, "qv")
    mark_only_lora_trainable(model, extra_keywords=[])
    model.eval()
    ids = torch.randint(0, 1000, (2, 24), device="cuda")
    with torch.no_grad():
        a = model(input_ids=ids).logits
        merge_all(model)
        b = model(input_ids=ids).logits
        unmerge_all(model)
        c = model(input_ids=ids).logits
    print("gpt2 OK: adapted", n, "lora params", count_lora(model),
          "merged diff", (a - b).abs().max().item(),
          "unmerged diff", (a - c).abs().max().item())
    assert torch.allclose(a, b, atol=1e-4)


def test_official_loralib():
    """Numerical equivalence with microsoft/LoRA's loralib (if present)."""
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.join(here, "..", "official", "LoRA")
    if not os.path.isdir(repo):
        print("official loralib not found, skipped")
        return
    sys.path.insert(0, repo)
    import loralib as official

    torch.manual_seed(0)
    r, alpha = 8, 8
    ours = LoRALinear(torch.nn.Linear(32, 16), r, alpha)
    off = official.Linear(32, 16, r=r, lora_alpha=alpha)
    with torch.no_grad():
        off.weight.copy_(ours.base.weight)
        off.bias.copy_(ours.base.bias)
        ours.lora_A.normal_()
        ours.lora_B.normal_()
        off.lora_A.copy_(ours.lora_A)
        off.lora_B.copy_(ours.lora_B)
    x = torch.randn(4, 32)
    d_fwd = (ours(x) - off(x)).abs().max().item()
    off.eval()  # loralib merges on eval()
    ours.merge()
    d_merged = (ours(x) - off(x)).abs().max().item()
    d_w = (ours.base.weight - off.weight).abs().max().item()
    print(f"official loralib OK: forward={d_fwd}, merged={d_merged}, weight={d_w}")
    assert d_fwd == 0 and d_merged == 0 and d_w == 0


if __name__ == "__main__":
    test_linear()
    test_roberta()
    test_gpt2()
    test_official_loralib()
