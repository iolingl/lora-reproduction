"""Minimal, from-scratch re-implementation of LoRA (Hu et al., 2021).

The design mirrors microsoft/LoRA's `loralib`:
  * A is initialized with kaiming-uniform, B with zeros (official code);
    the paper describes A ~ N(0, sigma^2), B = 0.  Either way Delta W = 0 at init.
  * forward: y = W x + (alpha / r) * B A x
  * merge(): W <- W + (alpha / r) B A, so inference has zero extra latency.

We support the two layer layouts needed by the reproduced experiments:
  * nn.Linear            (RoBERTa query/key/value/output projections)
  * transformers Conv1D  (GPT-2 packed QKV projection c_attn)
"""
import math
from typing import Iterable, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """LoRA wrapper around an existing (frozen) nn.Linear."""

    def __init__(self, base: nn.Linear, r: int, alpha: int, dropout: float = 0.0):
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.merged = False
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        dev, dt = base.weight.device, base.weight.dtype
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, device=dev, dtype=dt))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=dev, dtype=dt))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        for p in self.base.parameters():
            p.requires_grad = False

    def delta_weight(self) -> torch.Tensor:
        return (self.lora_B @ self.lora_A) * self.scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if not self.merged:
            out = out + (self.lora_dropout(x) @ self.lora_A.t() @ self.lora_B.t()) * self.scaling
        return out

    @torch.no_grad()
    def merge(self) -> None:
        if self.merged:
            return
        self.base.weight.data += self.delta_weight()
        self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        if not self.merged:
            return
        self.base.weight.data -= self.delta_weight()
        self.merged = False


class LoRAConv1D(nn.Module):
    """LoRA for GPT-2's Conv1D, optionally only on a subset of packed outputs.

    GPT-2's c_attn computes [Q, K, V] in one Conv1D.  With `enable_lora`
    (e.g. [True, False, True]) we adapt Q and V separately, exactly like
    loralib.MergedLinear, without adapting K.
    """

    def __init__(self, base, r: int, alpha: int, enable_lora: Sequence[bool],
                 dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.enable_lora = list(enable_lora)
        self.n_groups = len(self.enable_lora)
        self.merged = False
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        w = base.weight  # Conv1D stores (in_features, out_features)
        in_features, out_features = w.shape
        assert out_features % self.n_groups == 0
        self.out_per_group = out_features // self.n_groups
        self.lora_A = nn.ParameterList()
        self.lora_B = nn.ParameterList()
        self.group_to_param = {}
        for gi, enabled in enumerate(self.enable_lora):
            if enabled:
                A = torch.empty(r, in_features, device=w.device, dtype=w.dtype)
                B = torch.zeros(self.out_per_group, r, device=w.device, dtype=w.dtype)
                nn.init.kaiming_uniform_(A, a=math.sqrt(5))
                self.group_to_param[gi] = len(self.lora_A)
                self.lora_A.append(nn.Parameter(A))
                self.lora_B.append(nn.Parameter(B))
        for p in self.base.parameters():
            p.requires_grad = False

    def delta_weight(self) -> torch.Tensor:
        """Full (in, out) delta, with zeros in disabled groups."""
        parts = []
        for gi, enabled in enumerate(self.enable_lora):
            if enabled:
                j = self.group_to_param[gi]
                parts.append((self.lora_B[j] @ self.lora_A[j]).t() * self.scaling)
            else:
                parts.append(torch.zeros(self.base.weight.shape[0], self.out_per_group,
                                         dtype=self.base.weight.dtype,
                                         device=self.base.weight.device))
        return torch.cat(parts, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.merged:
            return out
        xd = self.lora_dropout(x)
        for gi, enabled in enumerate(self.enable_lora):
            if enabled:
                j = self.group_to_param[gi]
                delta = (xd @ self.lora_A[j].t() @ self.lora_B[j].t()) * self.scaling
                s, e = gi * self.out_per_group, (gi + 1) * self.out_per_group
                out = torch.cat([out[..., :s], out[..., s:e] + delta, out[..., e:]], dim=-1)
        return out

    @torch.no_grad()
    def merge(self) -> None:
        if self.merged:
            return
        self.base.weight.data += self.delta_weight()
        self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        if not self.merged:
            return
        self.base.weight.data -= self.delta_weight()
        self.merged = False


# ---------------------------------------------------------------------------
# Injection helpers
# ---------------------------------------------------------------------------
ROBERTA_TARGETS = {
    "q": "query", "k": "key", "v": "value", "o": "output.dense",
}


def _parse_targets(targets) -> List[str]:
    if isinstance(targets, str):
        s = targets.strip()
        if s and not any(c in s for c in ", "):
            return list(s)  # compact form such as "qv"
        return [t for t in s.replace(",", " ").split() if t]
    return list(targets)


def inject_lora_roberta(model, r: int, alpha: int, targets="qv", dropout: float = 0.0):
    """Replace attention projections of a RoBERTa encoder with LoRA layers."""
    tnames = _parse_targets(targets)
    for tword in tnames:
        if tword not in ROBERTA_TARGETS:
            raise ValueError(f"unknown RoBERTa LoRA target: {tword}")
    count = 0
    for layer in model.roberta.encoder.layer:
        attn = layer.attention
        for tword in tnames:
            if tword == "o":
                parent, name = attn.output, "dense"
            else:
                parent, name = attn.self, ROBERTA_TARGETS[tword]
            base = getattr(parent, name)
            assert isinstance(base, nn.Linear), type(base)
            setattr(parent, name, LoRALinear(base, r, alpha, dropout))
            count += 1
    return count


def inject_lora_gpt2(model, r: int, alpha: int, targets="qv", dropout: float = 0.0):
    """Apply LoRA to GPT-2 attention.  `q`/`k`/`v` refer to slices of c_attn.

    `o` adapts c_proj.  Returns the number of adapted projection matrices.
    """
    tnames = _parse_targets(targets)
    valid = {"q", "k", "v", "o"}
    if not set(tnames) <= valid:
        raise ValueError(f"unknown GPT-2 LoRA targets: {tnames}")
    enable = [("q" in tnames), ("k" in tnames), ("v" in tnames)]
    count = 0
    for block in model.transformer.h:
        attn = block.attn
        if any(enable):
            attn.c_attn = _GPT2LoRAWrapper(attn.c_attn, r, alpha, enable, dropout)
            count += sum(enable)
        if "o" in tnames:
            attn.c_proj = LoRAConv1D(attn.c_proj, r, alpha, [True], dropout)
            count += 1
    return count


class _GPT2LoRAWrapper(nn.Module):
    """Thin wrapper so we can keep Conv1D import-free above."""

    def __init__(self, base, r, alpha, enable, dropout):
        super().__init__()
        self.inner = LoRAConv1D(base, r, alpha, enable, dropout)

    def forward(self, x):
        return self.inner(x)

    def merge(self):
        self.inner.merge()

    def unmerge(self):
        self.inner.unmerge()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def mark_only_lora_trainable(model, extra_keywords: Optional[Iterable[str]] = ("classifier",)):
    """Freeze everything except LoRA parameters and (by default) the task head."""
    for name, p in model.named_parameters():
        p.requires_grad = False
        if "lora_A" in name or "lora_B" in name:
            p.requires_grad = True
        elif extra_keywords and any(k in name for k in extra_keywords):
            p.requires_grad = True


def mark_all_trainable(model):
    for p in model.parameters():
        p.requires_grad = True


def lora_parameters(model):
    return [p for n, p in model.named_parameters() if "lora_A" in n or "lora_B" in n]


def count_trainable(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_lora(model) -> int:
    return sum(p.numel() for p in lora_parameters(model))


def all_lora_modules(model):
    for m in model.modules():
        if isinstance(m, (LoRALinear, LoRAConv1D)):
            yield m
        elif isinstance(m, _GPT2LoRAWrapper):
            yield m.inner


@torch.no_grad()
def merge_all(model):
    for m in all_lora_modules(model):
        m.merge()


@torch.no_grad()
def unmerge_all(model):
    for m in all_lora_modules(model):
        m.unmerge()
