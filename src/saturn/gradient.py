"""Isodepth gradients and feature gradients.

rho_gradient   d rho_m / d(x, y) by autograd through each isodepth network (GASTON's method). The
               model is fitted on `zscore_coords(coords)` = (coords - mean) / s with ONE scalar s, so by
               the chain rule d rho / d coords = (d rho / d coords_z) / s. Evaluate the network at exactly
               those coords_z: scaling x and y separately both evaluates it at the wrong positions and
               bends gradient directions on non-square sections.
frames         unit vector ALONG the isodepth gradient (u) and ACROSS it (v = u rotated +90 degrees)
spatial_gradients
               per-spot gradient of every feature by a local least-squares plane over the k nearest spots
along_across   project feature gradients on (u, v): how fast a feature changes along rho (what Saturn
               models) vs across it (what a 1-D axis discards)
"""
from __future__ import annotations
import numpy as np
import torch
from scipy.spatial import cKDTree

KGRAD = 12


def coord_scale(coords) -> float:
    """the single scale zscore_coords divides by"""
    C = np.asarray(coords, np.float64)
    return float((C - C.mean(0)).std())


def rho_gradient(model, coords, knn: int = 0, per_axis_scaling: bool = False):
    """(N, M, 2) gradients of every isodepth in raw coordinate units. knn > 0 averages each spot's
    vector over its knn nearest spots (the ReLU network's autograd gradient is piecewise constant).
    per_axis_scaling=True reproduces the manuscript's Fig 2i run, which evaluated the network at
    (coords - mean) / std per axis and divided by the per-axis std (wrong for non-square sections)."""
    from .preprocess import zscore_coords
    C = np.asarray(coords, np.float64); model = model.cpu().eval()
    if per_axis_scaling:
        s = C.std(0); Z = ((C - C.mean(0)) / s).astype(np.float32)
    else:
        s = coord_scale(C); Z = zscore_coords(C)
    t = torch.tensor(Z, dtype=torch.float32, requires_grad=True)
    d = model.isodepth(t)
    G = np.stack([torch.autograd.grad(d[:, m].sum(), t, retain_graph=True)[0].numpy() / s
                  for m in range(d.shape[1])], 1)
    if knn:
        nn = cKDTree(C).query(C, k=knn)[1]; G = G[nn].mean(1)
    return G


def frames(G):
    """G (N, 2) -> u along, v across, |G|"""
    n = np.maximum(np.linalg.norm(G, axis=1, keepdims=True), 1e-12)
    u = G / n
    return u, np.c_[-u[:, 1], u[:, 0]], n.ravel()


def spatial_gradients(V, xy, k: int = KGRAD):
    """local least-squares plane per spot -> (N, F, 2), vectorised over features"""
    V = np.asarray(V); xy = np.asarray(xy, float); n, F = V.shape
    idx = cKDTree(xy).query(xy, k=k)[1]
    G = np.zeros((n, F, 2), np.float32)
    for i in range(n):
        nb = idx[i]
        pinv = np.linalg.pinv(np.c_[xy[nb] - xy[i], np.ones(k)])[:2]
        G[i] = (pinv @ V[nb]).T
    return G


def along_across(Gf, G_rho):
    """Gf (N, F, 2) feature gradients, G_rho (N, 2) one isodepth's gradient -> (along, across), each (N, F)"""
    u, v, _ = frames(G_rho)
    return (Gf * u[:, None, :]).sum(2), (Gf * v[:, None, :]).sum(2)
