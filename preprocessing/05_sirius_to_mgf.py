import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from mist_cf import common


def single_spec_process(spec_name, max_formulae=100, inten_thresh=0.003):
    """single_spec_process.

    Args:
        spec_name:
        data_dir:
        max_formulae:
        inten_thresh:
    """
    meta, spec = common.parse_spectra(spec_name)
    spec_name = Path(spec_name).stem
    meta["FEATURE_ID"] = spec_name
    if "INSTRUMENT" in meta:
        ins_string = meta.get("INSTRUMENT", "Unknown (LC/MS)")
    else:
        ins_string = meta.get("instrumentation", "Unknown (LC/MS)")
    meta["INSTRUMENT"] = ins_string.strip()
    keep_keys = [
        "INSTRUMENT",
        "PEPMASS",
        "parentmass",
        "PRECURSOR_MZ",
        "precursor_mz",
        "FEATURE_ID",
    ]
    meta = {i: meta[i] for i in meta if i in keep_keys}
    return (meta, spec)


def main(args):

    split_file = args.split_file
    debug = args.debug
    ms_file_dir = Path(args.ms_file_dir)
    save_name = args.save_name

    # Read split file and get test names
    split_df = pd.read_csv(split_file, sep="\t")
    test_names = split_df[split_df["Fold_0"] == "test"]["spec"].values
    if debug:
        test_names = test_names[:100]

    print(f"Number of test files: {len(test_names)}")

    # Read all the test files and save them in a single file
    full_spec_names = [ms_file_dir / f"{i}.ms" for i in test_names]
    specs = common.chunked_parallel(
        full_spec_names,
        single_spec_process,
        max_cpu=20,
        chunks=100,
    )

    out_str = common.build_mgf_str(specs)
    with open(save_name, "w") as f:
        f.write(out_str)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split-file",
        default="data/nist_canopus/splits/split_1.tsv",
        help="Path to split file",
    )
    parser.add_argument(
        "--save-name",
        default="data/nist_canopus/split_1_test.mgf",
        help="Save path specified by the user",
    )
    parser.add_argument(
        "--ms-file-dir",
        default="data/nist_canopus/spec_files/",
        help="Path to data directory",
    )
    parser.add_argument(
        "--debug", action="store_true", default=False, help="Debug flag"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
