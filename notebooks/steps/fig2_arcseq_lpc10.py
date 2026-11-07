"""LPC10 S1 RNA and ATAC AnnData from the GEO files (GSM9247593 RNA, GSM9249009 ATAC).

Recovered from mouse_atlas/process.py (2025-12-13), which wrote mouse_atlas/process_data/arcseq_<sample>_{rna,atac}_adata.h5ad.
The files under claude_code_tutorial_lpc/results/timepoints/ are byte-identical copies of those. Logic and parameters are
unchanged; only the sample loop was reduced to LPC10S1, and paths were made relative to SATURN_ROOT.

One value cannot be recomputed: process.py ordered the spots with list(set(a) & set(b)), and Python randomises string hashes
per process, so that order is not reproducible. Spot order enters the Saturn targets and the fit, so the published order is
pinned from fig2_lpc10s1_obs_order.txt (a list of the 8,180 barcodes, nothing else). With PIN=0 the set order is kept.
"""
import os, re
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, scanpy as sc, snapatac2 as snap

H = os.environ["SATURN_ROOT"]
AGING = f"{H}/mouse_atlas/mouse_aging"
RNA = f"{AGING}/GSM9247593_07_LPC10S1_RNA_matrix.csv.gz"
POS = f"{AGING}/GSM9247593_07_LPC10S1_tissue_positions_list.csv.gz"
FRAG = f"{AGING}/GSM9249009_07_LPC10S1_atac_fragments.tsv.gz"
BED = f"{H}/mouse_atlas/encode_mouse.bed"
OUT = f"{H}/claude_code_tutorial_lpc/results/timepoints"
ORDER = Path(__file__).with_name("fig2_lpc10s1_obs_order.txt")
SAMPLE = "arcseq_LPC10S1"


def transfer_atac_adata(fragment_file):
    data = snap.pp.import_fragments(fragment_file, chrom_sizes=snap.genome.mm10, sorted_by_barcode=False)
    peak_adata = snap.pp.make_peak_matrix(data, peak_file=BED, counting_strategy="paired-insertion", inplace=False).to_memory()
    sc.pp.filter_genes(peak_adata, min_cells=int(0.01 * peak_adata.shape[0]))

    def decompose_peak_name(peak):
        m = re.match(r'^(chr[^:]+):(\d+)-(\d+)$', peak)
        if m:
            chrom, start, end = m.groups()
            return chrom, int(start), int(end)
        return None, None, None

    decomposed = peak_adata.var_names.to_series().apply(decompose_peak_name)
    peak_adata.var['chr'] = decomposed.apply(lambda x: x[0])
    peak_adata.var['start'] = decomposed.apply(lambda x: x[1])
    peak_adata.var['end'] = decomposed.apply(lambda x: x[2])
    if '-' in peak_adata.obs_names[0]:
        peak_adata.obs_names = peak_adata.obs_names.map(lambda x: x.split('-')[0])
    return peak_adata


def transfer_rna_adata_from_csv(counts_file, spatial_coords_file):
    counts = pd.read_csv(counts_file, index_col=0)
    adata = ad.AnnData(X=counts.T.values)
    adata.var_names = counts.index
    adata.obs_names = counts.columns
    if '-' in adata.obs_names[0]:
        adata.obs_names = adata.obs_names.map(lambda x: x.split('-')[0])
    spatial_coords = pd.read_csv(spatial_coords_file, header=None,
                                 names=["barcode", "in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"])
    spatial_coords.set_index("barcode", inplace=True)
    adata.obs = adata.obs.join(spatial_coords, how="left")
    adata.obs['in_tissue'] = adata.obs['in_tissue'].astype(int)
    adata = adata[adata.obs['in_tissue'] == 1].copy()
    sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.filter_cells(adata, min_genes=10)
    return adata


def get_overlap_adata(atac_adata, rna_adata, sample_ids):
    shared = list(set(atac_adata.obs_names) & set(rna_adata.obs_names))
    if os.environ.get("PIN", "1") == "1":
        pinned = ORDER.read_text().split()
        assert set(pinned) == set(shared), f"pinned order has {len(pinned)} barcodes, data {len(shared)}"
        shared = pinned
    atac_adata = atac_adata[shared].copy()
    rna_adata = rna_adata[shared].copy()
    rna_adata.obs['sample_id'] = sample_ids
    atac_adata.obs['sample_id'] = sample_ids
    rna_adata.obsm['spatial'] = np.stack([rna_adata.obs['pxl_row_in_fullres'].values, rna_adata.obs['pxl_col_in_fullres'].values], axis=1)
    atac_adata.obsm['spatial'] = rna_adata.obsm['spatial']
    atac_adata.obs = rna_adata.obs
    os.makedirs(OUT, exist_ok=True)
    atac_adata.write_h5ad(f"{OUT}/{sample_ids}_atac_adata.h5ad")
    rna_adata.write_h5ad(f"{OUT}/{sample_ids}_rna_adata.h5ad")


if __name__ == "__main__":
    get_overlap_adata(transfer_atac_adata(FRAG), transfer_rna_adata_from_csv(RNA, POS), SAMPLE)
    print("written", OUT)
