""" 04_create_formulae_split.py

Create formulae split

TODO: Ammend to have more exlusion formuale

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
        "--formula-decoy-file", default="data/biomols/biomols_with_decoys.txt"
    )
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument(
        "--exclude-labels",
        default="data/nist_canopus/labels.tsv",
    )
    parser.add_argument("--out", default="data/biomols/biomols_with_decoys_split.tsv")
    return parser.parse_args()


def main():
    """main."""
    args = get_args()
    decoy_file = args.formula_decoy_file
    exclude_labels = args.exclude_labels
    debug = args.debug
    out = args.out

    # Get spec df
    spec_df = pd.read_csv(exclude_labels, sep="\t")
    formulae = spec_df["formula"].values
    formulae = list(set(formulae))

    # Get all masses
    formulae_masses = [common.formula_mass(i) for i in formulae]
    formulae_masses = decomp.get_rounded_masses(formulae_masses)

    # Build splits and completely exclude everything in labels
    df = pd.read_csv(
        decoy_file,
        sep="\t",
    ).fillna("")
    df_masses = set(df["mass"].values)
    new_masses = set(formulae_masses)
    overlap = df_masses.intersection(new_masses)
    total_len = len(df_masses)

    # Create test split
    train_frac, val_frac, test_frac = 0.8, 0.1, 0.1

    test_num = int(total_len * test_frac)
    train_num = total_len - test_num
    val_num = int(total_len * val_frac)
    train_num = total_len - val_num - test_num

    # Start by adding every mass into test
    test_masses = overlap
    remaining_test = test_num - len(test_masses)
    remaining_masses = df_masses.difference(test_masses)
    test_masses.update(
        np.random.choice(list(remaining_masses), remaining_test, replace=False)
    )

    remaining_masses = remaining_masses.difference(test_masses)
    val_masses = set(np.random.choice(list(remaining_masses), val_num, replace=False))
    train_masses = remaining_masses.difference(val_masses)

    # Sanity check
    tr_v = train_masses.intersection(val_masses)
    tr_te = train_masses.intersection(test_masses)
    v_te = val_masses.intersection(test_masses)

    # Now build splits
    mass_to_entries = defaultdict(lambda: {})
    fold_name = "Canopus_exclude"
    for mass in tqdm(df["mass"].values):
        if mass in train_masses:
            fold = "train"
        elif mass in test_masses:
            fold = "test"
        elif mass in val_masses:
            fold = "val"
        else:
            fold = "exclude"

        mass_to_entries[mass][fold_name] = fold
        mass_to_entries[mass]["mass"] = mass

    export_df = pd.DataFrame(list(mass_to_entries.values()))

    # Name first
    export_df = export_df.sort_index(axis=1, ascending=False)
    export_df.to_csv(out, sep="\t", index=False)


if __name__ == "__main__":
    main()
