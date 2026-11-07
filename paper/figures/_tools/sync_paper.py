"""Copy the manuscript code (aligned with the Saturn package) into saturn_repo/paper/.

    python published_figures/_tools/sync_paper.py

paper/scripts/   every analysis script a figure depends on (figure builders' imports, the documented source
                 scripts, the fit / target scripts), followed recursively through imports
paper/figures/   figN/*.py builders, fig_supp/, _tools/
paper/style/     _style/*.py;   paper/palettes/  sticky palettes;   paper/sim/  simulators and metrics
paper/config/    domain_k_elbow/k_elbow.json (per-axis domain K)
Copies are verbatim (absolute lab paths kept): they document exactly how each number was made.
"""
import glob, os, re, shutil

H = "/mnt/HDD2B/Users/yancui"
SCRIPTS = f"{H}/Hole/gaston_mix_mm/scripts"
PF = f"{H}/Saturn/published_figures"
DEST = f"{H}/saturn_repo/paper"
EXTRA = ["run_real2", "prep_real2", "lpc10_fov_prep", "melanoma_tissue_prep", "mel_hic_abins_prep", "sproxseq_prep",
         "sproxseq_prep_depthreg", "rtt_isoform_ratio_pca", "rtt_isoform_refit_nocollapse", "gaston_default_r2",
         "bench_v5_full", "domain_k_elbow", "meth_dr_bench", "saturn_cis_gradient", "mel_hic_prep", "rtt_isoform_multinomial",
         "mel_hic_granger_domains"]


def main():
    local = {os.path.basename(p)[:-3] for p in glob.glob(f"{SCRIPTS}/*.py")}
    seeds = set(EXTRA)
    for f in glob.glob(f"{H}/saturn_repo/notebooks/src/*.py"):              # every script a notebook runs
        seeds |= {m for m in re.findall(r"([A-Za-z0-9_]+)\.py", open(f).read()) if m in local}
    figs = sorted(glob.glob(f"{PF}/fig[1-7]/*.py") + glob.glob(f"{PF}/fig_supp/*.py"))
    for f in figs + glob.glob(f"{PF}/fig*/repo/run_*.sh"):
        t = open(f).read()
        seeds |= {m for m in re.findall(r"([A-Za-z0-9_]+)\.py", t) if m in local}
        seeds |= {m for m in re.findall(r"(?:from|import)\s+([A-Za-z0-9_]+)", t) if m in local}
    seen, st = set(), list(seeds & local)
    while st:
        m = st.pop()
        if m in seen:
            continue
        seen.add(m)
        st += [x for x in re.findall(r"(?:^|\n)\s*(?:from|import)\s+([A-Za-z0-9_]+)", open(f"{SCRIPTS}/{m}.py").read())
               if x in local and x not in seen]
    for d in ("scripts", "figures", "style", "sim", "palettes", "config"):
        shutil.rmtree(f"{DEST}/{d}", ignore_errors=True); os.makedirs(f"{DEST}/{d}")
    for m in sorted(seen):
        shutil.copy(f"{SCRIPTS}/{m}.py", f"{DEST}/scripts/{m}.py")
    for f in figs:
        sub = os.path.basename(os.path.dirname(f)); os.makedirs(f"{DEST}/figures/{sub}", exist_ok=True)
        shutil.copy(f, f"{DEST}/figures/{sub}/{os.path.basename(f)}")
    for f in glob.glob(f"{PF}/fig*/repo/run_*.sh"):
        shutil.copy(f, f"{DEST}/figures/{f.split('/')[-3]}/{os.path.basename(f)}")
    os.makedirs(f"{DEST}/figures/_tools", exist_ok=True)
    for f in glob.glob(f"{PF}/_tools/*.py"):
        shutil.copy(f, f"{DEST}/figures/_tools/")
    for f in glob.glob(f"{PF}/_style/*.py"):
        shutil.copy(f, f"{DEST}/style/")
    for f in glob.glob(f"{PF}/_palettes/*"):
        if os.path.isfile(f): shutil.copy(f, f"{DEST}/palettes/")
    for f in glob.glob(f"{H}/Hole/gaston_mix_mm/sim/*.py"):
        shutil.copy(f, f"{DEST}/sim/")
    for sub, src in (("fig2/steps", f"{PF}/fig2/repo/steps"), ("fig3/upstream", f"{PF}/fig3/repo/upstream")):   # recovered upstream steps
        shutil.copytree(src, f"{DEST}/figures/{sub}", dirs_exist_ok=True)
    os.makedirs(f"{DEST}/config/domain_k_elbow", exist_ok=True)         # per-axis domain K, read by domain_k.py at import
    shutil.copy(f"{H}/Hole/gaston_mix_mm/results/real_saturn/domain_k_elbow/k_elbow.json", f"{DEST}/config/domain_k_elbow/")
    shutil.copy(f"{H}/Hole/gaston_mix_mm/sim/simulate_v5.py", f"{DEST}/scripts/simulate_v5.py")
    print(f"{len(seen)} scripts, {len(figs)} figure builders -> {DEST}")


if __name__ == "__main__":
    main()
