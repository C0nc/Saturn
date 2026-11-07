#!/usr/bin/env python
"""Saturn arm of the Fig. 1 hard-grid benchmark: runs the benchmark's own main(), keeping only its first method.

    python fig1_saturn_bench.py                 all 24 hard runs, one process per run (--jobs, default 24)
    python fig1_saturn_bench.py --shard 3       one run

bench_v5_full.py is not edited. Its all_methods() generator yields Saturn first, so taking only that first item leaves
simulation, settings, metrics, file names and result tables exactly as published, and no competing method is run. With
--seeds_easy 0 --nshard 24, shard i holds hard run i, which is the run that published shard i held after its easy run.
The grid has no dependence on the easy run: train_saturn reseeds everything and simulate_v5 uses a seeded generator.
The official gaston package is imported at the top of the benchmark but only used by the competing arms: it is stubbed.
"""
import argparse, os, subprocess, sys, types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HOLE = Path(os.environ["SATURN_ROOT"]).expanduser().resolve() / "Hole/gaston_mix_mm"
N_RUNS = 24


def one(shard, out):
    stub = types.ModuleType("gaston"); nn = types.ModuleType("gaston.neural_net")
    nn.train = nn.load_rescale_input_data = None; stub.neural_net = nn
    sys.modules.update({"gaston": stub, "gaston.neural_net": nn})
    sys.path.insert(0, str(HOLE / "scripts"))
    import bench_v5_full as B
    every_method = B.all_methods
    B.all_methods = lambda d, s: iter([next(every_method(d, s))])
    sys.argv = ["bench_v5_full.py", "--shard", str(shard), "--nshard", str(N_RUNS), "--seeds_easy", "0", "--seeds_hard", "3"]
    if out: sys.argv += ["--out", out]
    B.main()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--shard", type=int); ap.add_argument("--jobs", type=int, default=24); ap.add_argument("--out")
    a = ap.parse_args()
    if a.shard is not None:
        return one(a.shard, a.out)
    def run(i):
        cmd = [sys.executable, __file__, "--shard", str(i)] + (["--out", a.out] if a.out else [])
        return i, subprocess.run(cmd, capture_output=True, text=True)
    with ThreadPoolExecutor(min(a.jobs, 30)) as ex:
        bad = [(i, r.stderr[-400:]) for i, r in ex.map(run, range(N_RUNS)) if r.returncode]
    if bad: raise SystemExit(f"failed runs: {bad}")


if __name__ == "__main__":
    main()
