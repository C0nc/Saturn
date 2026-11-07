"""ATAC 5 kb tile matrix from the raw fragments, for Fig 3 (results/melanoma/atac_snap.h5ad).

Recovered from paper/figures/fig3/upstream/melanoma_multimodal.py: the import block of build_atac(), code unchanged. Left out:
its GASTON-Mix-MM fit, an old baseline that Fig 3 does not read. The published object also carries obs['cc_K7'], added later
by other code (file date Aug 9); no Fig 3 step reads it.
"""
import os
import snapatac2 as snap

ROOT = os.environ["SATURN_ROOT"]
M = f"{ROOT}/Yufan_Melanoma"
OUT = f"{ROOT}/Hole/gaston_mix_mm/results/melanoma"; os.makedirs(OUT, exist_ok=True)
out = os.path.join(OUT, "atac_snap.h5ad")
data = snap.pp.import_fragments(f"{M}/ATAC/HmelanomaA.fragments.sort.bed.gz",
                                chrom_sizes=snap.genome.hg38, file=out,
                                sorted_by_barcode=False, min_num_fragments=100)
snap.pp.add_tile_matrix(data, bin_size=5000)
data.close()
print("->", out)
