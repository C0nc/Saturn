"""Full v5 benchmark: Saturn vs single-modality GASTON vs multi-modal integration
methods, on BOTH axes, three questions each.

METHODS
    Saturn                  new Saturn (defaults: held-out restart acceptance)
    GASTON (default)        official gaston, one fit per modality, its own z-scoring
    GASTON (global-scaled)  official gaston on Saturn's globally scaled targets
    MEFISTO                 mofapy2 spatial factor model, joint factors
    MISO                    joint embedding (sat-miso env)
    SCIGMA                  per-modality latents (sat-scigma env)
    SpatialGlue             per-modality latents (sat-scigma env)

AXIS PER MODALITY for multi-dimensional methods: the single dimension that
explains the most variance of that modality's counts (`select_factor_fair`), taken
from the modality's OWN latent where the method exposes one (SCIGMA, SpatialGlue),
else from the joint embedding (MEFISTO, MISO). No ground truth is used.

THREE QUESTIONS, BOTH AXES
    1. recovery      |Spearman| of each learned axis with its true axis
    2. leakage       incremental R2 of the OTHER true axis in each learned axis,
                     beyond its own true axis (0 = clean)
    3. contribution  share of each modality's target variance attributed to the
                     OTHER modality's axis, by one estimator for EVERY method: an
                     additive piecewise-linear (hinge) regression of the modality's
                     targets on (own axis, other axis). Scored against
                       oracle = the same estimator run on the TRUE axes
                     so the error isolates axis quality, not estimator bias.
                     Saturn additionally reports its built-in coupling().
Per-config axes of every method are saved, so metrics can be recomputed.
"""
from __future__ import annotations
import os, sys, io, json, time, argparse, itertools, tempfile, subprocess, contextlib, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "sim")); sys.path.insert(0, "/mnt/HDD2B/Users/yancui/saturn_repo/src")
from simulate_v5 import SimConfigV5, simulate_v5                        # noqa
from saturn import train_saturn, zscore_coords                          # noqa
from metrics_v2 import spearman, select_factor_fair                     # noqa
from gaston.neural_net import train as gaston_train, load_rescale_input_data  # noqa
import baselines as BASE                                                # noqa

EPOCHS, HID = 10000, [20, 20]
# Tiny models (900 spots, width-20 layers) are kernel-launch bound on a GPU: measured
# 1000 Saturn epochs = 3.0 s on 2 CPU threads vs 4.8 s on a GPU shared by 4 jobs. With
# 128 cores, many CPU shards beat a queued GPU. Helpers (MISO/SCIGMA/SpatialGlue) still
# use the GPU from their own envs -- they are short.
DEV = os.environ.get("SAT_DEVICE", "cpu")
torch.set_num_threads(int(os.environ.get("SAT_THREADS", "2")))
CONDA = "/home/yancui/miniconda3/bin/conda"
HELPERS = {"MISO": ("sat-miso", "miso_helper.py"), "SCIGMA": ("sat-scigma", "scigma_helper.py"),
           "SpatialGlue": ("sat-scigma", "spatialglue_helper.py")}


# ------------------------------------------------------------------ metrics
def leak_inc(v, own, other):
    def r2(X):
        X = np.c_[np.ones(len(v)), X]
        res = v - X @ np.linalg.lstsq(X, v, rcond=None)[0]
        return 1 - res.var() / (v.var() + 1e-12)
    P = lambda z: np.c_[z, z ** 2, z ** 3]
    return float(max(0.0, r2(np.c_[P(own), P(other)]) - r2(P(own))))


def _hinge(z, n_knots=8):
    z = (z - z.mean()) / (z.std() + 1e-12)
    kn = np.quantile(z, np.linspace(0.1, 0.9, n_knots))
    return np.c_[z, np.maximum(0.0, z[:, None] - kn[None, :])]


def contribution(Y, own, other):
    """Share of Y's fitted variance carried by the OTHER axis in an additive
    piecewise-linear model Y ~ f(own) + g(other). Same estimator for all methods."""
    Bo, Bx = _hinge(own), _hinge(other)
    X = np.c_[np.ones(len(own)), Bo, Bx]
    C = np.linalg.lstsq(X, Y, rcond=None)[0]
    po = Bo @ C[1:1 + Bo.shape[1]]
    px = Bx @ C[1 + Bo.shape[1]:]
    vo, vx = po.var(0).sum(), px.var(0).sum()
    return float(vx / (vo + vx + 1e-12))


def true_share(own, cross):
    vo, vc = own.var(0), cross.var(0)
    return float(np.mean(vc / (vo + vc + 1e-12)))


# ------------------------------------------------------------------ methods
def gaston_axis(S, A, seed):
    with contextlib.redirect_stdout(io.StringIO()):
        mod, _ = gaston_train(torch.as_tensor(S, dtype=torch.float32), torch.as_tensor(A, dtype=torch.float32),
                              S_hidden_list=HID, A_hidden_list=HID, epochs=EPOCHS,
                              optim="adam", lr=1e-3, seed=seed, device=DEV)
    return mod.spatial_embedding(torch.as_tensor(S, dtype=torch.float32).to(DEV)).detach().cpu().numpy().ravel()


def run_helper(method, d, timeout=1800):
    env, script = HELPERS[method]
    tmp = tempfile.mkdtemp()
    fin, fout = os.path.join(tmp, "in.npz"), os.path.join(tmp, "out.npz")
    np.savez_compressed(fin, coords=d.coords, counts_0=d.counts_R, counts_1=d.counts_A)
    r = subprocess.run([CONDA, "run", "-n", env, "--no-capture-output", "python",
                        os.path.join(ROOT, "sim", script), fin, fout],
                       capture_output=True, text=True, timeout=timeout)
    if not os.path.exists(fout):
        raise RuntimeError(f"{method}: {r.stderr[-300:]}")
    return dict(np.load(fout))


def pick(emb, counts):
    v, k, ve = select_factor_fair(emb, counts)
    return np.asarray(v).ravel(), int(k)


def all_methods(d, s):
    """Yield (method, axis_R, axis_A, extra) for every method."""
    xyn = zscore_coords(d.coords)
    m, h = train_saturn([d.Y_R, d.Y_A], xyn, seed=s, device=DEV)
    xy = torch.as_tensor(xyn, device=DEV)
    D = m.isodepth(xy).detach().cpu().numpy(); C = m.coupling(xy)
    yield "Saturn", D[:, 0], D[:, 1], dict(builtin_RA=C[0, 1], builtin_AR=C[1, 0])

    ax = []
    for Y in (d.Y_R, d.Y_A):
        S, A = load_rescale_input_data(d.coords, Y)
        ax.append(gaston_axis(S.numpy(), A.numpy(), s))
    yield "GASTON (default)", ax[0], ax[1], {}
    yield "GASTON (global-scaled)", gaston_axis(xyn, d.Y_R, s), gaston_axis(xyn, d.Y_A, s), {}

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        F = BASE.recover_mefisto([d.counts_R, d.counts_A], d.coords, n_factors=10, n_iter=30)
    if F is not None:
        (vR, kR), (vA, kA) = pick(F[0], d.counts_R), pick(F[1], d.counts_A)
        yield "MEFISTO", vR, vA, dict(dim_R=kR, dim_A=kA)

    o = run_helper("MISO", d)
    (vR, kR), (vA, kA) = pick(o["emb"], d.counts_R), pick(o["emb"], d.counts_A)
    yield "MISO", vR, vA, dict(dim_R=kR, dim_A=kA)

    for meth in ("SCIGMA", "SpatialGlue"):
        o = run_helper(meth, d)
        (vR, kR), (vA, kA) = pick(o["emb_0"], d.counts_R), pick(o["emb_1"], d.counts_A)
        yield meth, vR, vA, dict(dim_R=kR, dim_A=kA)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=4)
    ap.add_argument("--seeds_easy", type=int, default=2)
    ap.add_argument("--seeds_hard", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "bench_v5_full"))
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "axes"), exist_ok=True)

    grid = [dict(grid="easy", axis_angle_deg=ang, cross_RA=cra, cross_AR=car, seed=s)
            for ang, (cra, car), s in itertools.product(
                [30, 60, 90], [(0.0, 0.0), (0.5, 0.2), (0.8, 0.3), (0.95, 0.5)], range(a.seeds_easy))]
    grid += [dict(grid="hard", axis_angle_deg=ang, cross_RA=cra, cross_AR=0.2,
                  coupled_frac=cf, rna_libsize=1000.0, seed=s)
             for ang, cra, cf, s in itertools.product([45, 90], [0.8, 0.95], [0.6, 0.9], range(a.seeds_hard))]
    grid = [g for i, g in enumerate(grid) if i % a.nshard == a.shard]

    rows, t0 = [], time.time()
    out_csv = os.path.join(a.out, f"bench_full_s{a.shard}.csv")
    for i, cfg in enumerate(grid, 1):
        kw = {k: v for k, v in cfg.items() if k != "grid"}
        d = simulate_v5(SimConfigV5(**kw))
        T = d.d
        base = dict(grid=cfg["grid"], angle=cfg["axis_angle_deg"], cross_RA=cfg["cross_RA"],
                    cross_AR=cfg["cross_AR"], coupled_frac=cfg.get("coupled_frac", 0.4), seed=cfg["seed"],
                    true_corr=spearman(T[:, 0], T[:, 1]),
                    true_RA=true_share(d.parts["s"], d.parts["c"]),
                    true_AR=true_share(d.parts["t"], d.parts["e"]),
                    oracle_RA=contribution(d.Y_R, T[:, 0], T[:, 1]),
                    oracle_AR=contribution(d.Y_A, T[:, 1], T[:, 0]))
        tag = f"{cfg['grid']}_a{cfg['axis_angle_deg']}_c{cfg['cross_RA']}_{cfg.get('coupled_frac',0.4)}_s{cfg['seed']}"
        saved = {"true": T}
        try:
            for meth, dR, dA, extra in all_methods(d, cfg["seed"]):
                saved[meth] = np.c_[dR, dA]
                rows.append(dict(base, method=meth,
                                 rec_R=spearman(dR, T[:, 0]), rec_A=spearman(dA, T[:, 1]),
                                 leak_R=leak_inc(dR, T[:, 0], T[:, 1]), leak_A=leak_inc(dA, T[:, 1], T[:, 0]),
                                 contrib_RA=contribution(d.Y_R, dR, dA), contrib_AR=contribution(d.Y_A, dA, dR),
                                 axis_dup=spearman(dR, dA), **extra))
                pd.DataFrame(rows).to_csv(out_csv, index=False)
        except Exception as e:
            print(f"  ERROR {tag}: {type(e).__name__}: {str(e)[:200]}", flush=True)
        np.savez_compressed(os.path.join(a.out, "axes", f"{tag}.npz"), **saved)
        el = (time.time() - t0) / 60
        print(f"[{i}/{len(grid)}] {tag}  {el:.1f}m, ~{el / i * (len(grid) - i):.0f}m left", flush=True)


if __name__ == "__main__":
    main()
