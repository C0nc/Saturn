"""NSF helper — reads counts + coords CSVs, fits NSF, writes factor matrix CSV.
Runs in the `nsf` conda env (needs TensorFlow + spatial_factorization).
Usage:  python nsf_helper.py <counts.csv> <coords.csv> <out.csv> <n_factors>
"""
from __future__ import annotations
import sys, os, warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
from spatial_factorization import SpatialFactorization
from spatial_factorization.training import ModelTrainer

counts_path, coords_path, out_path, K = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
K = int(K)

X = np.loadtxt(counts_path, delimiter=",", dtype=np.float32)   # spots × genes
Z_coords = np.loadtxt(coords_path, delimiter=",", dtype=np.float32)  # spots × 2
N, J = X.shape
try:
    # inducing points: subsample spots as knots
    n_ind = min(120, N)
    idx = np.random.RandomState(0).choice(N, n_ind, replace=False)
    Z_ind = Z_coords[idx]

    # NSF convention: X = spatial coords, Y = counts, sz = size factors (per-spot lib)
    sz = X.sum(1, keepdims=True) / np.mean(X.sum(1))
    D_train = {"X": Z_coords.astype(np.float32),
                "Y": X.astype(np.float32),
                "sz": sz.astype(np.float32)}
    ds = tf.data.Dataset.from_tensor_slices(D_train).batch(N)   # full batch

    model = SpatialFactorization(J=J, L=K, Z=Z_ind, lik="poi",
                                   nonneg=True, isotropic=True,
                                   length_scale=0.1)
    model.init_loadings(X.astype(np.float32), X=Z_coords.astype(np.float32),
                          sz=sz.astype(np.float32).reshape(-1, 1))

    trainer = ModelTrainer(model, lr=0.01)
    trainer.train_model(ds, Ntr=N, num_epochs=200, verbose=False,
                          status_freq=1000, ckpt_freq=1000, tol=1e-3)

    # per-spot factor scores = latent GP function values at each spot
    Fhat = np.asarray(model.sample_latent_GP_funcs(Z_coords.astype(np.float32), S=1, chol=True))
    # sample_latent_GP_funcs returns (S, L, N) if S>1 else (L, N); squeeze
    Fhat = np.squeeze(Fhat)
    if Fhat.ndim == 2 and Fhat.shape[0] == K:
        Fhat = Fhat.T                                             # (N, K)
    np.savetxt(out_path, Fhat, fmt="%.6f", delimiter=",")
    print(f"nsf done: {Fhat.shape}", flush=True)
except Exception as e:
    print(f"nsf ERROR: {e}", flush=True)
    import traceback; traceback.print_exc()
    Fhat = np.zeros((N, K))
    np.savetxt(out_path, Fhat, fmt="%.6f", delimiter=",")
    sys.exit(1)
