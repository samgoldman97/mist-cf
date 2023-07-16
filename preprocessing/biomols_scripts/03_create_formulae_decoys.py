""" 03_create_formulae_decoys.py

Create many formulae decoys for the list of smiles


python preprocessing/biomols_scripts/03_create_formulae_decoys.py  --num-decoys
256

"""

import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm


import mist_cf.common as common
import mist_cf.decomp as decomp


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--formulae-list", default="data/biomols/biomols_filter_formulae.txt"
    )
    parser.add_argument("--num-decoys", type=int, default=256)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--out", default="data/biomols/biomols_with_decoys.txt")
    parser.add_argument("--decomp-filter", type=str, default="RDBE")
    return parser.parse_args()


def main():
    """main."""
    args = get_args()
    formulae_file = Path(args.formulae_list)
    formulae = [j.strip() for j in open(formulae_file, "r").readlines()]
    num_decoys = args.num_decoys
    debug = args.debug
    out = args.out
    decomp_filter = args.decomp_filter
    BATCH_SIZE = 2048

    if debug:
        BATCH_SIZE = 512
        formulae = formulae[:1024]

    formulae_masses = [common.formula_mass(i) for i in formulae]
    formulae_masses = decomp.get_rounded_masses(formulae_masses)
    mass_to_formulae = defaultdict(lambda: [])

    for mass, formula in zip(formulae_masses, formulae):
        mass_to_formulae[mass].append(formula)

    dict_entries = []
    batches = list(common.batches(formulae_masses, BATCH_SIZE))

    for temp_masses in batches:
        out_dict = decomp.run_sirius(
            temp_masses, filter_=decomp_filter, mass_sort=False
        )

        for k, neg_list in tqdm(out_dict.items()):
            pos_list = mass_to_formulae.get(k, [])
            neg_set = set(neg_list).difference(pos_list)
            neg_list = list(neg_set)
            if num_decoys is not None:
                num_negs = len(neg_list)
                num_sample = min(num_decoys, num_negs)
                if num_sample > 0:
                    neg_list = np.random.choice(neg_list, num_sample, replace=False)
            neg_str = ",".join(neg_list)
            pos_str = ",".join(pos_list)
            dict_entries.append(dict(mass=k, pos=pos_str, neg=neg_str))

    df = pd.DataFrame(dict_entries)
    df = df.sort_values(by="mass").reset_index(drop=True)
    df.to_csv(out, index=None, sep="\t")

if __name__ == "__main__":
    main()
