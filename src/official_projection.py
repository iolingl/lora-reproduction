"""Table-7 style projection analysis for the released official MNLI LoRA checkpoint.

For each layer we take the top-r left/right singular directions of the LoRA update
Delta W = (alpha/r) B A and measure ||U^T W V||_F on the frozen pretrained W,
comparing with (a) the top-r subspace of W itself and (b) a random subspace.
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CKPT = "official/LoRA/examples/NLU/roberta_base_lora_mnli.bin"
ALPHA, RANK = 16.0, 8


def main():
    from transformers import AutoModel
    sd = torch.load(CKPT, map_location="cpu")
    base = AutoModel.from_pretrained("roberta-base", local_files_only=True).state_dict()
    out = {}
    for layer in range(12):
        out[layer] = {}
        for w in ["query", "value"]:
            p = f"roberta.encoder.layer.{layer}.attention.self.{w}"
            A = sd[p + ".lora_A"].float()
            B = sd[p + ".lora_B"].float()
            dW = (B @ A) * (ALPHA / RANK)
            U, S, Vh = torch.linalg.svd(dW, full_matrices=False)
            U, V = U[:, :RANK], Vh.T[:, :RANK]
            key = f"encoder.layer.{layer}.attention.self.{w}.weight"
            W = base[key].float()
            proj_dw = torch.linalg.matrix_norm(U.T @ W @ V).item()
            Uw, Sw, Vhw = torch.linalg.svd(W, full_matrices=False)
            proj_w = torch.linalg.matrix_norm(Uw[:, :RANK].T @ W @ Vhw.T[:, :RANK]).item()
            g = torch.Generator().manual_seed(layer * 10 + (w == "value"))
            Ur = torch.linalg.qr(torch.randn(W.shape[0], RANK, generator=g))[0]
            Vr = torch.linalg.qr(torch.randn(W.shape[1], RANK, generator=g))[0]
            proj_rand = torch.linalg.matrix_norm(Ur.T @ W @ Vr).item()
            out[layer][w] = dict(fro_w=W.norm().item(), fro_dw=dW.norm().item(),
                                 proj_dw=proj_dw, proj_w=proj_w, proj_rand=proj_rand,
                                 amp=dW.norm().item() / proj_dw,
                                 ratio=proj_dw / proj_rand)
            print(f"L{layer:02d} {w:5s} ||dW||={out[layer][w]['fro_dw']:.3f} "
                  f"proj_dW={proj_dw:.3f} rand={proj_rand:.3f} ratio={proj_dw/proj_rand:.2f} "
                  f"amp={dW.norm().item()/proj_dw:.2f}", flush=True)
    os.makedirs("results/analysis", exist_ok=True)
    json.dump(out, open("results/analysis/official_projection.json", "w"), indent=2)
    print("saved results/analysis/official_projection.json")


if __name__ == "__main__":
    main()
