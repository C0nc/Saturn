"""Run MISO in its own env (sat-miso, py3.7) and dump the joint embedding.

MISO (Coleman et al., Nat Methods 2025) integrates M spatial modalities into a
single joint embedding for clustering. It does NOT emit a per-modality gradient,
so the caller selects one dimension per modality by variance explained.

Usage:  python miso_helper.py in.npz out.npz
  in.npz  : counts_0..counts_{M-1} (N,G) int, coords (N,2)
  out.npz : emb (N, D) joint embedding
"""
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch
from miso import Miso

def main(inp, outp):
    z = np.load(inp)
    M = sum(1 for k in z.files if k.startswith("counts_"))
    feats = []
    for m in range(M):
        X = z[f"counts_{m}"].astype(np.float64)
        lib = X.sum(1, keepdims=True).clip(min=1)
        feats.append(np.log1p(X / lib * 1e4).astype(np.float32))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = Miso(feats, device=dev)
    model.train()
    emb = model.emb
    emb = emb.detach().cpu().numpy() if torch.is_tensor(emb) else np.asarray(emb)
    np.savez_compressed(outp, emb=emb.astype(np.float32))
    print(f"MISO ok: emb {emb.shape}", flush=True)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
