"""Low-rank update analysis (paper Section 7.2 / 7.3, Figure 3 and Table 7).

Given two LoRA checkpoints trained with different ranks on the same task, compute
the normalized subspace similarity

    phi(i, j) = || U_i^T V_j ||_F^2 / min(i, j)

between the top-i singular directions of the rank-r1 update and the top-j
directions of the rank-r2 update.  Also project the frozen weight W onto the
update subspace and compare Frobenius norms.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_lora(path):
    sd = torch.load(path, map_location="cpu")
    return sd


def delta(sd, layer, weight, alpha, rank):
    p = f"roberta.encoder.layer.{layer}.attention.self.{weight}"
    A = sd[p + ".lora_A"].float()
    B = sd[p + ".lora_B"].float()
    return (B @ A) * (alpha / rank)


def singular(A, k=None):
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    if k is not None:
        U, S, Vh = U[:, :k], S[:k], Vh[:k]
    return U, S, Vh


def phi_matrix(Ua, Ub):
    """Ua: (d, ra), Ub: (d, rb) -> phi[i-1, j-1] for i=1..ra, j=1..rb."""
    ra, rb = Ua.shape[1], Ub.shape[1]
    out = np.zeros((ra, rb))
    for i in range(1, ra + 1):
        for j in range(1, rb + 1):
            out[i - 1, j - 1] = (torch.linalg.matrix_norm(
                Ua[:, :i].T @ Ub[:, :j], ord="fro") ** 2 / min(i, j)).item()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_a", required=True)
    ap.add_argument("--ckpt_b", required=True)
    ap.add_argument("--rank_a", type=int, required=True)
    ap.add_argument("--rank_b", type=int, required=True)
    ap.add_argument("--alpha", type=float, default=None,
                    help="shared alpha (default: use alpha=rank for each checkpoint, i.e. scaling 1)")
    ap.add_argument("--alpha_a", type=float, default=None)
    ap.add_argument("--alpha_b", type=float, default=None)
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--model", default="roberta-base")
    ap.add_argument("--out", default="results/analysis")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    sd_a, sd_b = load_lora(args.ckpt_a), load_lora(args.ckpt_b)
    from transformers import AutoModel
    base = AutoModel.from_pretrained(args.model)
    state = base.state_dict()

    out = {}
    alpha_a = args.alpha_a if args.alpha_a is not None else (
        args.alpha if args.alpha is not None else args.rank_a)
    alpha_b = args.alpha_b if args.alpha_b is not None else (
        args.alpha if args.alpha is not None else args.rank_b)
    for w in ["query", "value"]:
        dA = delta(sd_a, args.layer, w, alpha_a, args.rank_a)
        dB = delta(sd_b, args.layer, w, alpha_b, args.rank_b)
        Ua, Sa, Vha = singular(dA)
        Ub, Sb, Vhb = singular(dB)
        # right singular vectors span the input-side update subspace;
        # keep only the top rank_a / rank_b directions (the rest are numerical zeros)
        Va = Vha.T[:, :args.rank_a]
        Vb = Vhb.T[:, :args.rank_b]
        phi = phi_matrix(Va, Vb)
        out[w] = dict(phi=phi.tolist(),
                      singular_a=Sa.tolist(), singular_b=Sb.tolist(),
                      fro_a=float(torch.linalg.matrix_norm(dA)),
                      fro_b=float(torch.linalg.matrix_norm(dB)))

        # Table 7 style projection of the frozen W onto the r-dim update subspace
        wk = f"roberta.encoder.layer.{args.layer}.attention.self.{w}.weight"
        if wk not in state:  # AutoModel (no task head) has no "roberta." prefix
            wk = wk[len("roberta."):]
        W = state[wk].float()
        r = args.rank_a
        U, V = Ua[:, :r], Vha.T[:, :r]
        proj_dw = torch.linalg.matrix_norm(U.T @ W @ V, ord="fro")
        Uw, _, Vhw = singular(W)
        proj_w = torch.linalg.matrix_norm(Uw[:, :r].T @ W @ Vhw.T[:, :r], ord="fro")
        g = torch.Generator().manual_seed(0)
        Ur = torch.linalg.qr(torch.randn(W.shape[0], r, generator=g))[0]
        Vr = torch.linalg.qr(torch.randn(W.shape[1], r, generator=g))[0]
        proj_rand = torch.linalg.matrix_norm(Ur.T @ W @ Vr, ord="fro")
        # same projection on the higher-rank side (used to compare r vs 2r etc.)
        rb = args.rank_b
        Ub_, Vb_ = Ub[:, :rb], Vhb.T[:, :rb]
        proj_dw_b = torch.linalg.matrix_norm(Ub_.T @ W @ Vb_, ord="fro")
        gb = torch.Generator().manual_seed(1)
        Urb = torch.linalg.qr(torch.randn(W.shape[0], rb, generator=gb))[0]
        Vrb = torch.linalg.qr(torch.randn(W.shape[1], rb, generator=gb))[0]
        proj_rand_b = torch.linalg.matrix_norm(Urb.T @ W @ Vrb, ord="fro")
        out[w].update(W_fro=float(torch.linalg.matrix_norm(W)),
                      proj_dw=float(proj_dw), proj_w=float(proj_w),
                      proj_rand=float(proj_rand),
                      proj_dw_b=float(proj_dw_b), proj_rand_b=float(proj_rand_b))
        print(f"{w}: phi(8,64) diag={np.diag(phi)[:8].round(3)}")
        print(f"{w}: ||W||F={out[w]['W_fro']:.2f} ||dW||F={out[w]['fro_a']:.2f} "
              f"proj_dW={out[w]['proj_dw']:.3f} proj_W={out[w]['proj_w']:.3f} "
              f"proj_rand={out[w]['proj_rand']:.3f}")

    out["meta"] = dict(rank_a=args.rank_a, rank_b=args.rank_b, layer=args.layer,
                       ckpt_a=args.ckpt_a, ckpt_b=args.ckpt_b)
    with open(os.path.join(args.out, "subspace.json"), "w") as f:
        json.dump(out, f, indent=2)
    np.savez(os.path.join(args.out, "subspace.npz"),
             phi_q=np.array(out["query"]["phi"]), phi_v=np.array(out["value"]["phi"]))
    print("saved", args.out)


if __name__ == "__main__":
    main()
