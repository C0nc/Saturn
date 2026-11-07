"""Additional baselines for the noise benchmark:
  - recover_multispati_pca : MULTISPATI-PCA via multispaeti (SpatialPCA Python port)
  - recover_gaston_single  : official `gaston` package fit INDEPENDENTLY per modality
                             (i.e., two-independent-GASTON baseline for M=2 data)
  - recover_stamp          : STAMP (Nat Methods 2024) via sctm, per modality
  - recover_spatial_nmf    : NMF on kNN-smoothed counts (spatial-aware NMF baseline)
"""
from __future__ import annotations
from typing import List
import numpy as np
from sklearn.preprocessing import StandardScaler


def _log_norm(counts):
    lib = counts.sum(1, keepdims=True); lib[lib == 0] = 1.0
    return np.log1p(counts / lib * np.median(lib))


def _knn_connectivity(coords: np.ndarray, k: int = 6):
    from sklearn.neighbors import kneighbors_graph
    A = kneighbors_graph(coords, n_neighbors=k, include_self=False, mode="connectivity")
    A = (A + A.T) / 2
    return A


def recover_multispati_pca(counts_list, coords, n_factors=10):
    """MULTISPATI-PCA per modality on spatial connectivity graph."""
    from multispaeti import MultispatiPCA
    conn = _knn_connectivity(coords, k=6)
    out = []
    for X in counts_list:
        Xn = StandardScaler().fit_transform(_log_norm(X))
        k = min(n_factors, Xn.shape[1] - 1, Xn.shape[0] - 1)
        try:
            m = MultispatiPCA(n_components=k, connectivity=conn, random_state=0)
            Z = m.fit_transform(Xn)          # (N, k)
        except Exception:
            return None
        out.append(np.asarray(Z))
    return out


def recover_gaston_single_per_mod(counts_list, coords, epochs=500,
                                    S_hidden=[50], A_hidden=[50, 50],
                                    embed_size=4, sigma=0.1, seed=0):
    """Fit the official `gaston` NN INDEPENDENTLY per modality (adam, lr=1e-2).
    Returns per-modality (N, 1) isodepth arrays (best-and-only factor).
    """
    from gaston.neural_net import train as gaston_train
    import torch, io, contextlib
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    out = []
    S = torch.tensor(coords, dtype=torch.float32)
    for X in counts_list:
        Xn = _log_norm(X).astype(np.float32)
        A = torch.tensor(Xn, dtype=torch.float32)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                model, _ = gaston_train(
                    S, A, S_hidden_list=list(S_hidden), A_hidden_list=list(A_hidden),
                    epochs=epochs, optim="adam", lr=1e-2, seed=seed,
                    embed_size=embed_size, sigma=sigma, device=dev,
                    loss_reduction="mean",
                )
                model.eval()
                with torch.no_grad():
                    rho = model.spatial_embedding(S.to(dev)).cpu().numpy()  # (N,1)
        except Exception:
            rho = np.zeros((coords.shape[0], 1))
        out.append(rho)
    return out


def recover_spatial_nmf(counts_list, coords, n_factors=8, k_neighbors=6):
    """NMF on kNN-smoothed counts — spatial-aware NMF baseline.
    Smooth each spot by the mean of itself + k-nearest-neighbors before NMF.
    """
    from sklearn.decomposition import NMF
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(coords)
    _, idx = nn.kneighbors(coords)         # (N, k+1) includes self
    out = []
    for X in counts_list:
        Xn = _log_norm(X)
        # kNN mean-smoothing
        Xs = Xn[idx].mean(axis=1)          # (N, G)
        Xs = Xs - Xs.min()
        k = min(n_factors, Xs.shape[1])
        try:
            m = NMF(n_components=k, init="nndsvda", max_iter=400, tol=1e-3)
            Z = m.fit_transform(Xs)
        except Exception:
            Z = np.zeros((Xs.shape[0], k))
        out.append(Z)
    return out


def recover_stamp(counts_list, coords, n_topics=8, epochs=150, k_neighbors=6):
    """STAMP per modality: build AnnData with spatial_connectivities, train,
    return per-cell topic proportions as factor matrix.
    """
    import io, contextlib
    import anndata as ad
    from sklearn.neighbors import kneighbors_graph
    A = kneighbors_graph(coords, n_neighbors=k_neighbors, include_self=False)
    A = ((A + A.T) > 0).astype(np.float32)
    out = []
    for X in counts_list:
        try:
            a = ad.AnnData(X.astype(np.float32))
            a.obsm["spatial"] = coords
            a.layers["counts"] = X.astype(np.int32)
            a.obsp["spatial_connectivities"] = A
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                from sctm.stamp import STAMP
                model = STAMP(a, n_topics=n_topics, gene_likelihood="nb",
                                verbose=False, mode="sign", n_layers=1)
                model.setup(a, layer="counts", categorical_covariate_keys=None,
                              continous_covariate_keys=None)
                dev = "cuda:0" if _cuda_available() else "cpu"
                model.train(max_epochs=epochs, learning_rate=0.01, device=dev,
                             batch_size=256, early_stop=True, patience=15,
                             min_epochs=40)
                Z = np.asarray(model.get_cell_by_topic(a))
        except Exception:
            Z = np.zeros((coords.shape[0], n_topics))
        out.append(Z)
    return out


def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False
