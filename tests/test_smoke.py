import numpy as np
import pandas as pd
import pytest
import saturn
from saturn.sim import SimConfig, simulate
from saturn import piecewise as pw, crossterm as ct


@pytest.fixture(scope="module")
def small():
    d = simulate(SimConfig(n_side=14, n_genes=120, n_peaks=200, seed=1))
    return d, saturn.zscore_coords(d.coords)


def test_targets_scaled(small):
    d, _ = small
    for Y in (d.Y_R, d.Y_A):
        assert np.allclose(Y.mean(0), 0, atol=1e-4)
        assert abs(Y.var(0).mean() - 1) < 1e-3          # global scaling: mean column variance 1


def test_fit_and_decompose(small):
    d, xy = small
    res = saturn.fit([d.Y_R, d.Y_A], xy, ["RNA", "ATAC"], seeds=[0], epochs=300, n_restarts=1,
                     restart_epochs=100, refit_all_epochs=50)
    assert res.axes.shape == (len(xy), 2)
    assert res.coupling.shape == (2, 2) and np.allclose(res.coupling.sum(1), 1, atol=1e-6)
    D, full, own, cross = saturn.decompose(res.model, xy)
    assert np.allclose(full[0], own[0] + cross[0], atol=1e-5)


def test_piecewise_gauss_gate():
    rng = np.random.default_rng(0); d = np.linspace(0, 1, 400); lab = (d > 0.5).astype(int)
    Y = np.c_[np.where(lab == 0, 3 * d, 1.5) + rng.normal(0, .1, 400), rng.normal(0, 1, 400)]
    slope, inter = pw.fit_gauss(Y, lab, d)
    assert abs(slope[0, 0] - 3) < 0.2 and slope[0, 1] == 0 or abs(slope[0, 1]) < 0.3
    assert pw.n_sloped(slope)[1] <= 1


def test_binomial():
    rng = np.random.default_rng(0); d = np.linspace(-1, 1, 300); lab = np.zeros(300, int)
    n = np.full(300, 50); s = rng.binomial(n, 1 / (1 + np.exp(-2 * d)))
    slope, _ = pw.fit_binomial(s[:, None], n, lab, d)
    assert abs(slope[0, 0] - 2) < 0.4


def test_crossterm_split():
    rng = np.random.default_rng(0); a, b = rng.uniform(size=500), rng.uniform(size=500)
    Y = np.c_[np.sin(3 * a), np.sin(3 * a) + np.cos(4 * b)] + rng.normal(0, .05, (500, 2))
    r0, r1, p, prof = ct.fit_split(Y, a, b)
    assert p[1] < 1e-10 and (r1 - r0)[1] > (r1 - r0)[0]
    assert ct.bh(np.array([0.01, 0.02, 0.5])).max() <= 1


def test_postfit_chain(small, tmp_path):
    import matplotlib; matplotlib.use("Agg")
    from saturn import domains, annotate, plotting, transition
    d, xy = small
    res = saturn.fit([d.Y_R, d.Y_A], xy, ["RNA", "ATAC"], seeds=[0], epochs=300, n_restarts=0, refit_all_epochs=0)
    dR = res.axes[:, 0]
    lab = domains.gaston_domains(dR, d.Y_R, n_domains=3)
    T = annotate.annotate_counts(d.counts_R, [f"g{i}" for i in range(d.counts_R.shape[1])], lab, dR, "RNA",
                                 n_top_gradient=50)
    assert len(T) == lab.max() + 1 and "markers" in T
    scr, slope, inter, idx = annotate.screen(d.Y_A, [f"c{i}" for i in range(d.Y_A.shape[1])], lab, dR, kind="gauss")
    fig = plotting.function_figure(d.coords, dR, lab, d.Y_A, slope, inter, np.array([f"c{i}" for i in idx]), [0, 1])
    fig.savefig(tmp_path / "f.png")
    steep = res.axes[:, 1]                       # a spatially smooth score: its top quintile is one region
    tr = transition.transect(d.coords, res.axes[:, 1], steep)
    fits, prof = transition.transition_table(pd.DataFrame({"a": dR, "b": res.axes[:, 1]}), tr, n_boot=10)
    assert {"s0", "steep_at", "steep_lo", "transient"} <= set(fits.columns)


def test_gradient_chain_rule(small):
    """rho_gradient must equal the finite-difference gradient in raw coordinates"""
    from saturn import gradient as SG
    d, xy = small
    res = saturn.fit([d.Y_R, d.Y_A], xy, ["RNA", "ATAC"], seeds=[0], epochs=200, n_restarts=0, refit_all_epochs=0)
    G = SG.rho_gradient(res.model, d.coords)
    C = d.coords.astype(np.float64); h = 1e-3 * C.std()
    f = lambda c: saturn.decompose(res.model, ((c - C.mean(0)) / (C - C.mean(0)).std()).astype(np.float32))[0]
    for j in range(2):
        e = np.zeros(2); e[j] = h
        fd = (f(C + e) - f(C - e)) / (2 * h)
        ok = np.isfinite(fd).all(1)
        assert np.median(np.abs(fd[ok] - G[ok, :, j])) < 0.05 * np.median(np.abs(G[..., j])) + 1e-6
    u, v, n = SG.frames(G[:, 0])
    assert np.allclose((u * v).sum(1), 0, atol=1e-6)


def test_choose_k_and_cis_tf():
    from saturn import domains, cis, tfactivity as TA
    rng = np.random.default_rng(0); d = np.sort(rng.uniform(0, 3, 300))
    Y = np.c_[np.floor(d), -np.floor(d) * 2] + rng.normal(0, .05, (300, 2))
    k, u = domains.choose_k(d, Y, k_max=5, n_buckets=30)
    assert k == 3 and u[0] > u[-1]
    lab = np.floor(d).astype(int); grid = np.linspace(0.1, 2.9, 60)
    sl, it = np.ones((2, 3)), np.zeros((2, 3)); sl[1] *= -1
    C = cis.curves(sl, it, d, lab, grid)
    assert np.allclose(C[:, 0], -C[:, 1])
    M = rng.normal(size=(50, 400)); hits = np.zeros((400, 2), bool); hits[:40, 0] = True; hits[40:80, 1] = True
    M[:25, :40] += 2
    Z = TA.deviations_fast(M, hits, nperm=50)
    assert Z[:25, 0].mean() > 3 and abs(Z[:, 1].mean()) < 1
