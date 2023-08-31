""" create_subformula_assignment.py
Given a set of spectra and candidates from a labels file, assign subformulae and save to JSON files.

Notes:
1. All the candidate formula need to be associated with an ionization.
2. There are only 2 legitimate way for subformula assignment:
    a. We only focus on one one type of ionization, say [M+H]+. 
    The user need to set --ionization [M+H]+
    The split file must exclude all other ionization
    
    b. Otherwise, we consider all types of ionization
    The decoy label file provided in --decoy-label must contain decoys by all types of ionization
    The split file need to be the split of the full canopus dataset
3. Each MS2 spec file has one corresponding JSON file.


"""
from pathlib import Path
import argparse
from functools import partial
import numpy as np
import pandas as pd
import time
import json
from collections import defaultdict

from tqdm import tqdm
from mist_cf import common


def get_args():
    """get args"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", default="data/canopus_train", help="data directory"
    )
    parser.add_argument("--out-name", default=None, help="Out suffix")
    parser.add_argument(
        "--max-formulae",
        default=100,
        type=int,
        help="Maximum number of subpeak formulae assignment for each MS2 file.",
    )
    parser.add_argument(
        "--decoy-label",
        type=str,
        default="data/canopus_train/label_decoy_RDBE_256.tsv",
        help="Path of decoy label file.",
    )
    parser.add_argument(
        "--ionization",
        type=str,
        default=None,
        help="Ion that the user want to focused on (if applicable).",
    )
    parser.add_argument(
        "--debug", action="store_true", default=False, help="Debug flag."
    )
    parser.add_argument(
        "--assign-test-only",
        action="store_true",
        default=False,
        help="A flag indicating subformula assignment for test data only.",
    )
    parser.add_argument(
        "--split-file",
        type=str,
        default="data/canopus_train/splits/canopus_[M+H]+_0.3_100.tsv",
        help="Path of split file.",
    )
    parser.add_argument(
        "--mass-diff-type",
        default="ppm",
        type=str,
        help="Type of mass difference - absolute differece (abs) or relative difference (ppm).",
    )
    parser.add_argument(
        "--mass-diff-thresh",
        action="store",
        default=0.01,
        type=float,
        help="Threshold of mass difference.",
    )
    parser.add_argument(
        "--inten-thresh",
        action="store",
        default=0.0,
        type=float,
        help="Threshold of MS2 subpeak intensity (normalized to 1).",
    )
    parser.add_argument(
        "--num-workers",
        action="store",
        default=20,
        type=int,
        help="Maximum number of cpus.",
    )
    return parser.parse_args()


def single_spec_process(spec_name, data_dir, max_formulae, inten_thresh):
    """single_spec_process.

    Args:
        spec_name:
        data_dir:
        max_formulae:
        inten_thresh:
    """
    spec = common.process_spec_file_unbinned(spec_name, data_dir=data_dir)
    return (
        spec[0],
        common.max_thresh_spec(
            spec[1], max_peaks=max_formulae, inten_thresh=inten_thresh
        ),
    )


def main():
    args = get_args()

    data_dir = Path(args.data_dir)
    label_path = Path(args.decoy_label)
    assign_test_only = args.assign_test_only

    subform_dir = data_dir / "subformulae"
    subform_dir.mkdir(exist_ok=True)
    labels_df = pd.read_csv(label_path, sep="\t")
    debug = args.debug
    max_formulae = args.max_formulae
    inten_thresh = args.inten_thresh
    ionization = args.ionization
    mass_diff_type = args.mass_diff_type
    mass_diff_thresh = args.mass_diff_thresh
    split_file = args.split_file
    num_workers = args.num_workers
    out_name = args.out_name

    if debug:
        inds = np.random.choice(len(labels_df), 50)
        labels_df = labels_df.loc[inds]

    label_name = label_path.stem
    if out_name is None:
        out_name = f"formulae_spec_{label_name}"

    output_dir = subform_dir / out_name
    output_dir.mkdir(exist_ok=True)

    if assign_test_only:
        split_file_path = Path(split_file)
        split_df = pd.read_csv(split_file_path, sep="\t")
        test_spec_name_lst = set(
            split_df[split_df["Fold_0"] == "test"]["spec"].to_list()
        )
        labels_df = labels_df[labels_df["spec"].isin(test_spec_name_lst)]

    if ionization is not None:
        labels_df = labels_df[labels_df["ionization"] == ionization]

    spec_lst = labels_df["spec"].to_list()
    labels_df = labels_df.fillna("").reset_index()

    proc_spec_full = partial(
        single_spec_process,
        data_dir=data_dir,
        max_formulae=max_formulae,
        inten_thresh=inten_thresh,
    )

    # Process and subset to top k peaks and above certain thresh
    if debug:
        input_specs = [proc_spec_full(i) for i in tqdm(spec_lst)]
    else:
        input_specs = common.chunked_parallel(
            spec_lst, proc_spec_full, chunks=500, max_cpu=num_workers
        )

    input_specs = {k: v for k, v in input_specs}

    # Build up all output dicts to map
    spec_to_assigns = defaultdict(lambda: [])
    for _, spec_entry in labels_df.iterrows():
        spec_name = spec_entry["spec"]
        spec = input_specs[spec_name]
        true_form = spec_entry["formula"]
        true_ion = spec_entry["ionization"]
        instrument = spec_entry["instrument"]
        mass_diff_thresh = common.get_instr_tol(instrument)
        export_dicts = spec_to_assigns[spec_name]

        cand_forms = spec_entry["decoy_formulae"]
        cand_ions = spec_entry["decoy_ions"]
        if cand_forms == "":
            cand_forms = []
            cand_ions = []
        else:
            cand_forms = cand_forms.split(",")
            cand_ions = cand_ions.split(",")

        cand_forms.append(true_form)
        cand_ions.append(true_ion)
        for cand_ion, cand_form in zip(cand_ions, cand_forms):
            export_dicts.append(
                {
                    "spec": spec,
                    "mass_diff_type": mass_diff_type,
                    "spec_name": spec_name,
                    "mass_diff_thresh": mass_diff_thresh,
                    "form": cand_form,
                    "ion_type": cand_ion,
                }
            )
        print(f"There are {len(export_dicts)} spec-cand pairs this spec files")

        if debug:
            spec_to_assigns[spec_name] = export_dicts[:2]

        # port these
        # spec_to_assigns[spec_name] = export_dicts[:2]

    # Define how to parallelize these assignments
    parallel_list = [
        {"spec_name": k, "export_dicts": v, "output_dir": output_dir}
        for k, v in spec_to_assigns.items()
        # if not (output_dir / f"{k}.json").exists()
    ]

    export_wrapper = lambda x: common.assign_single_spec(**x)
    print(f"Processing {len(parallel_list)} different spectra")

    if num_workers == 0 or debug:
        [export_wrapper(i) for i in tqdm(parallel_list)]
    else:
        common.chunked_parallel(
            parallel_list, export_wrapper, chunks=500, max_cpu=num_workers
        )


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Program finished in: {end_time - start_time} seconds")