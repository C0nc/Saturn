#!/usr/bin/env python
"""Higashi / Fast-Higashi for SpHIC117 melanoma. NOT rerun for the manuscript notebook (GPU, days, > 600 GB of temp files).

Recovered from SpHIC117/prep_dataset.py (stage 500kb_prep), imputate.py (500kb_train) and imputate_100.py (100kb_train).
Settings are in SpHIC117/config.JSON and config_100kb.JSON, copied here with the lab prefix replaced. Reference files stay
under /mnt/HDD2B/Reference. Only source of change from those scripts: paths, and removal of symlinked outputs first so
that nothing writes back into the input folder.
Not recovered: the 100 kb process_data call, and the command lines of Higashi's scCompartment.py and scTAD.py that wrote
tmp/scCompartment.hdf5 and tmp/scTAD.hdf5 (only nbr_5 imputations exist in tmp/, which suggests -n, but flags are unknown).
Usage: SATURN_ROOT=<workspace> <higashi env>/python fig6_higashi.py {500kb_prep,500kb_train,100kb_train}
"""
import json, os, sys
LAB = "/mnt/HDD2B/Users/yancui"; ROOT = os.environ["SATURN_ROOT"]; stage = sys.argv[1]
is500 = stage.startswith("500kb")
d = f"{ROOT}/SpHIC117/{'HIC' if is500 else 'HIC_100kb'}/higashi/"
for rel in ("filelist.txt", "read_count_all.npy", "fh_embedding.npy", "tmp/scCompartment.hdf5", "tmp/scTAD.hdf5"):
    if os.path.islink(d + rel): os.unlink(d + rel)
config = f"{d}config_workspace.JSON"; os.makedirs(d + "tmp", exist_ok=True)
txt = open(f"{ROOT}/SpHIC117/{'config.JSON' if is500 else 'config_100kb.JSON'}").read().replace(LAB, ROOT)
json.loads(txt); open(config, "w").write(txt)
from higashi.Higashi_wrapper import Higashi
higashi_model = Higashi(config)
if stage == "500kb_prep":
    import pickle
    import numpy as np
    from fasthigashi.FastHigashi_Wrapper import FastHigashi
    with open(d + "filelist.txt", "w") as f:
        for b in np.load(f"{ROOT}/SpHIC117/filter_name.npy", allow_pickle=True): f.write(f"{d}data/{b}.contact.tsv\n")
    pickle.dump({}, open(d + "label_info.pickle", "wb"))
    higashi_model.process_data()
    fh = FastHigashi(config_path=config, path2input_cache=d, path2result_dir=d, off_diag=100, filter=True, do_conv=True,
                     do_rwr=True, do_col=False, no_col=False)
    fh.prep_dataset(batch_norm=False)
    fh.run_model(dim1=.6, rank=256, n_iter_parafac=1, extra="")
    np.save(d + "fh_embedding.npy", fh.fetch_cell_embedding(final_dim=256, restore_order=True)["embed_l2_norm"])
else:
    higashi_model.prep_model()
    higashi_model.train_for_embeddings()
    higashi_model.train_for_imputation_nbr_0()
    higashi_model.train_for_imputation_with_nbr()
    higashi_model.impute_with_nbr()
