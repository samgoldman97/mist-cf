"""filter_smiles.py

Filter list of smiles

"""

import argparse
import numpy as np

import mist_cf.common as common


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles-list", default="data/biomols/biomols.txt")
    parser.add_argument("--smiles-out", default="data/biomols/biomols_filter.txt")
    return parser.parse_args()


def main():
    """main."""
    args = get_args()
    in_smiles = [j.strip() for j in open(args.smiles_list, "r").readlines()]
    smis_ar = np.array(in_smiles)
    out_name = args.smiles_out

    # Debug
    # in_smiles = in_smiles[:100000]
    min_formal_charges = common.chunked_parallel(in_smiles, common.min_formal_from_smi)
    max_formal_charges = common.chunked_parallel(in_smiles, common.max_formal_from_smi)

    all_forms = common.chunked_parallel(in_smiles, common.form_from_smi)
    num_atoms = common.chunked_parallel(in_smiles, common.atoms_from_smi)
    masses = common.chunked_parallel(in_smiles, common.mass_from_smi)

    is_not_empty = common.chunked_parallel(all_forms, lambda x: len(x) > 0)
    single_fragment = common.chunked_parallel(in_smiles, lambda x: "." not in x)
    only_valid_els = common.chunked_parallel(all_forms, common.has_valid_els)
    ge_2_atoms = np.array(num_atoms) > 2
    le_1500_mass = np.array(masses) <= 1500
    form_min_ge = np.array(min_formal_charges) >= -2
    form_max_le = np.array(max_formal_charges) <= 2

    mask = np.ones(len(in_smiles)).astype(bool)
    mask = np.logical_and(mask, np.array(is_not_empty))
    mask = np.logical_and(mask, np.array(single_fragment))
    mask = np.logical_and(mask, np.array(only_valid_els))
    mask = np.logical_and(mask, np.array(ge_2_atoms))
    mask = np.logical_and(mask, le_1500_mass)
    mask = np.logical_and(mask, form_min_ge)
    mask = np.logical_and(mask, form_max_le)

    filtered_smis = np.array(in_smiles)[~mask].tolist()
    out_smiles = np.array(in_smiles)[mask].tolist()
    print(filtered_smis)

    print(f"Len of old smiles: {len(in_smiles)}")
    print(f"Len of out smiles: {len(out_smiles)}")
    print(f"Len of filtered smiles: {len(in_smiles) - len(out_smiles)}")

    with open(out_name, "w") as fp:
        fp.write("\n".join(out_smiles))


if __name__ == "__main__":
    main()
