"""Compare the rebuilt LPC10 S1 AnnData with the published files, one modality at a time (content, not bytes)."""
import os, numpy as np, scipy.sparse as sp, anndata as ad

ROOT, PUB = os.environ["SATURN_ROOT"], os.environ.get("SATURN_PUBLISHED")
if not PUB:
    print("SATURN_PUBLISHED not set: no comparison"); raise SystemExit
for m in ("rna", "atac"):
    rel = f"claude_code_tutorial_lpc/results/timepoints/arcseq_LPC10S1_{m}_adata.h5ad"
    a, b = ad.read_h5ad(f"{ROOT}/{rel}"), ad.read_h5ad(f"{PUB}/{rel}")
    same_obs, same_var = list(a.obs_names) == list(b.obs_names), list(a.var_names) == list(b.var_names)
    d = sp.csr_matrix(a.X, dtype=np.float64) - sp.csr_matrix(b.X, dtype=np.float64) if a.shape == b.shape else None
    print(f"{m}: shape {a.shape} vs {b.shape}; obs order same {same_obs}; var same {same_var}; "
          f"X differing entries {d.count_nonzero() if d is not None else 'n/a'}; "
          f"spatial max diff {np.abs(a.obsm['spatial'] - b.obsm['spatial']).max() if same_obs else 'n/a'}")
