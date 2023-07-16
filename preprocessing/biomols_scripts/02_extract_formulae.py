"""filter_smiles.py

Filter list of smiles

"""

import argparse
import numpy as np

import mist_cf.common as common


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles-list", default="data/biomols/biomols_filter.txt")
    parser.add_argument(
        "--form-out", default="data/biomols/biomols_filter_formulae.txt"
    )
    return parser.parse_args()


def main():
    """main."""
    args = get_args()
    in_smiles = [j.strip() for j in open(args.smiles_list, "r").readlines()]
    out_name = args.form_out
    all_forms = common.chunked_parallel(in_smiles, common.form_from_smi)
    all_forms = list(set(all_forms))

    print("Num orig smiles", len(in_smiles))
    print("Num unique formuale ", len(all_forms))

    with open(out_name, "w") as fp:
        fp.write("\n".join(all_forms))


if __name__ == "__main__":
    main()
