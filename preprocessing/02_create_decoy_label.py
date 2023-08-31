"""create_decoy_label.py
Take the original labels file and generate decoy file using SIRIUS decomp.
"""

import logging
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import time

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl


import mist_cf.common as common
import mist_cf.decomp as decomp
from mist_cf.fast_form_score import fast_form_data, fast_form_model
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label-file",
        default="data/canopus_train/labels.tsv",
        help="Path to label file",
    )
    parser.add_argument(
        "--max-decoy",
        type=int,
        default=1e10,
        help="Maximum number of decoys to be added for a given mass.",
    )
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument(
        "--decomp-filter",
        type=str,
        default="RDBE",
        help="Filter used for decoy generation.",
    )
    parser.add_argument("--data-dir", type=str, default="data/canopus_train")
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for subsampling max-decoy from all the possible decoys.",
    )
    parser.add_argument("--sample-strat", action="store", default="uniform")
    parser.add_argument("--softmax-temperature", type=float, default=1.0)
    parser.add_argument("--resample-precursor-mz", action="store_true", default=False)
    parser.add_argument("--decoy-suffix", action="store", default=None)
    # parser.add_argument(
    #     "--gpu", default=False, action="store_true", help="Use GPU flag"
    # )
    parser.add_argument(
        "--num-workers",
        default=0,
        action="store",
        type=int,
        help="Set number of available CPUs",
    )
    parser.add_argument(
        "--fast-model", type=str, default=None, help="Path to fast model checkpoint"
    )

    return parser.parse_args()


def sample_decoys(
    spec,
    decoy_ion_lst,
    decoy_ions,
    parentmass,
    max_decoy,
    sample_strat,
    temperature,
    fast_model,
    device,
):
    """sample_decoys.

    Args:
        spec:
        decoy_ion_lst:
        decoy_ions:
        parentmass:
        max_decoy:
        sample_strat: 'uniform', 'normalized_inverse', 'softmax_inverse', 'sorted'
        temperature:
    return:
        sampled decoy list
    """
    decoy_ion_lst = np.array(decoy_ion_lst)
    adduct_masses = np.array([common.ion_to_mass[i] for i in decoy_ions])
    form_masses = np.array([common.formula_mass(i) for i in decoy_ion_lst])
    decoy_masses = form_masses + adduct_masses
    decoy_ppm = (
        common.clipped_ppm(
            np.abs(parentmass - decoy_masses), np.ones_like(decoy_masses) * parentmass
        )
        + 1e-20
    )
    sample_num = min(max_decoy, len(decoy_ion_lst))

    sample_ind_choices = np.arange(len(decoy_ion_lst))
    if sample_strat == "uniform":
        # sampled_decoys = np.random.choice(
        #    decoy_ion_lst, sample_num, replace=False
        # )
        sampled_decoys = np.random.choice(sample_ind_choices, sample_num, replace=False)

    elif sample_strat == "normalized_inverse":
        weights = np.reciprocal(decoy_ppm)
        sampled_decoys = np.random.choice(
            sample_ind_choices,
            sample_num,
            replace=False,
            p=weights / (np.sum(weights)),
        )
    elif sample_strat == "softmax_inverse":
        weights = np.reciprocal(decoy_ppm)
        if np.sum(weights) == 0:
            sampled_decoys = []
        else:
            weights = weights / (np.sum(weights))
            weights = np.exp(weights / temperature)
            sampled_decoys = np.random.choice(
                sample_ind_choices,
                sample_num,
                replace=False,
                p=weights / (np.sum(weights)),
            )
    elif sample_strat == "sorted":
        sorted_idx = np.argsort(decoy_ppm)
        sampled_decoys = sample_ind_choices[sorted_idx][:sample_num]

    elif sample_strat == "fast_filter":
        sampled_decoys = fast_model.fast_filter_sampling(
            spec, decoy_ion_lst, decoy_ions, max_decoy, device, batch_size=1024
        )
    else:
        raise ValueError(f"Weighting method {sample_strat} is not defined :(")

    np.random.shuffle(sampled_decoys)
    return sampled_decoys


def calculate_resampling_std(true_mass, error=15):
    """calculate_resampling_std.

    Args:
        true_mass:
        error:
    """
    # according to buddy, use sigma= (1/5)*max_error
    return true_mass * error / (5 * 1e6)


def resample_precursor_fn(true_masses, errors):
    """resample_precursor_fn.

    Args:
        true_masses: list of true masses
        errors: list of errors in ppm
    """
    within_error_thresh = np.zeros_like(true_masses).astype(bool)
    resampling_std = calculate_resampling_std(true_masses, error=errors)
    resample_masses = true_masses * 1
    while not np.all(within_error_thresh):
        new_masses = np.random.normal(
            loc=true_masses,
            scale=resampling_std,
        )
        new_masses = np.round(new_masses, 4)

        resample_masses[~within_error_thresh] = new_masses[~within_error_thresh]
        abs_mass_diff = np.abs(resample_masses - true_masses)
        rel_mass_diff = common.clipped_ppm(abs_mass_diff, true_masses)
        within_error_thresh = rel_mass_diff <= errors
    return new_masses


def main():
    """main."""
    args = get_args()
    labels_file = Path(args.label_file)
    debug = args.debug
    decomp_filter = args.decomp_filter
    max_decoy = args.max_decoy
    resample_precursor_mz = args.resample_precursor_mz
    data_dir = args.data_dir
    sample_strat = args.sample_strat
    softmax_temperature = args.softmax_temperature
    seed = args.seed
    decoy_suffix = args.decoy_suffix
    fast_model = args.fast_model
    if fast_model is not None:
        # Load fast model
        fast_model = fast_form_model.FastFFN.load_from_checkpoint(fast_model)

    # gpu = args.gpu
    # device = torch.device("cuda") if gpu else torch.device("cpu")
    device = torch.device("cpu")
    num_workers = args.num_workers

    df = pd.read_csv(labels_file, sep="\t")
    df = df.drop(columns=["name"])
    ion_lst = common.ION_LST
    if debug:
        # df = df[df['spec'] == "CCMSLIB00000577934"]
        df = df[:100]
        # num_workers = 0
        # ion_lst = ["[M]+", "[M+H]+"]

    specs = df["spec"].to_list()
    true_formulae = df["formula"].to_list()
    true_ionizations = df["ionization"].to_list()
    true_masses = [
        (common.formula_mass(true_form) + common.ion_to_mass[true_ion])
        for true_form, true_ion in zip(true_formulae, true_ionizations)
    ]

    df["true_mass"] = true_masses
    instruments = df["instrument"].values
    true_masses = np.array(true_masses)

    #  resample ms1 mass and get decoy by different instrument
    if resample_precursor_mz:
        errors = [common.get_instr_tol(i) for i in instruments]
        precursor_mz = resample_precursor_fn(true_masses, errors)
    else:
        # if do not resample ms1 mass, then get the measured massby inspecting the files
        precursor_mz = []
        for spec in specs:
            spec_file = Path(data_dir) / "spec_files" / f"{spec}.ms"
            meta, tuples = common.parse_spectra(spec_file)
            parentmass = float(meta.get("precursor_mz", meta["parentmass"]))
            precursor_mz.append(parentmass)
    precursor_mz = np.array(precursor_mz)

    # Computerel ppm and mass diffs
    abs_diffs = np.abs(true_masses - precursor_mz)
    rel_diffs = common.clipped_ppm(abs_diffs, true_masses)

    # We should not have very high absolute differences
    if np.any(abs_diffs > 3):
        raise ValueError()

    # true_mass is theoretical MS1 precursor mass
    spec2form = dict(zip(specs, true_formulae))

    spec2parentmass = dict(zip(specs, precursor_mz))

    # Iterate over all ions
    spec_to_form_list = dict()
    spec_to_ion_list = dict()
    spec_to_found = defaultdict(lambda: False)
    all_out_dicts = defaultdict(lambda: set())
    for ion in ion_lst:

        # equation: parentmass = decoy formula + decoy ionization
        decoy_masses = [
            (parentmass - common.ion_to_mass[ion]) for parentmass in precursor_mz
        ]
        decoy_masses = decomp.get_rounded_masses(decoy_masses)
        spec2mass = dict(zip(specs, decoy_masses))

        # Let's replace decomp.run_sirius
        # Switch to ppm=10 for speed
        out_dict = decomp.run_sirius(decoy_masses, filter_=decomp_filter, ppm=10)
        out_dict = {k: {(ion, vv) for vv in v} for k, v in out_dict.items()}

        # Update the existing all_out_dicts with the new out_dict
        for spec, mass in spec2mass.items():
            # Add out_dict to all_out dicts
            all_out_dicts[spec].update(out_dict.get(mass, {}))

    entries = list(all_out_dicts.items())

    def filter_spec_entries(entry):
        """filter_spec_entries."""
        spec, dict_entry = entry
        ions, cands = [], []
        if len(dict_entry) > 0:
            ions, cands = zip(*dict_entry)
        ions, cands = np.array(ions), np.array(cands)

        # Remove the true candidate to create decoy list
        inds = np.array(cands) != spec2form[spec]

        cands = cands[inds]
        ions = ions[inds]
        was_found = np.sum(~inds) > 0

        # Remove the true candidate to create decoy list
        if len(cands) < max_decoy:
            pass
        else:
            cand_inds = sample_decoys(
                spec,
                cands,
                ions,
                spec2parentmass[spec],
                max_decoy,
                sample_strat,
                softmax_temperature,
                fast_model,
                device,
            )
            ions = ions[cand_inds]
            cands = cands[cand_inds]

        return {
            "spec": spec,
            "was_found": was_found,
            "out_ions": ions,
            "out_cands": cands,
        }

    if num_workers == 0:
        output_dicts = list(map(filter_spec_entries, entries))
    else:
        output_dicts = common.chunked_parallel(
            entries, filter_spec_entries, max_cpu=num_workers
        )

    for out_dict in output_dicts:
        spec = out_dict["spec"]
        cands = out_dict["out_cands"]
        ions = out_dict["out_ions"]
        was_found = out_dict["was_found"]

        spec_to_form_list[spec] = [str(i) for i in cands]
        spec_to_ion_list[spec] = [str(i) for i in ions]
        spec_to_found[spec] = was_found

    df[f"decoy_formulae"] = [
        ",".join(spec_to_form_list[i]) if len(spec_to_form_list.get(i, [])) > 0 else ""
        for i in specs
    ]
    df[f"decoy_ions"] = [
        ",".join(spec_to_ion_list[i]) if len(spec_to_ion_list.get(i, [])) > 0 else ""
        for i in specs
    ]

    decoy_forms_is_null = [str(i).strip() == "[]" for i in df["decoy_formulae"]]
    print(f"Num switched: {np.sum(decoy_forms_is_null)}")
    df[decoy_forms_is_null]["decoy_formulae"] = ""
    df[decoy_forms_is_null]["decoy_ions"] = ""

    df[f"decomp_recover"] = [spec_to_found[i] for i in specs]

    df["parentmass"] = precursor_mz
    df["abs_mass_diff"] = abs_diffs
    df["rel_mass_diff"] = rel_diffs

    save_dir = labels_file.parent / "decoy_labels"
    save_dir.mkdir(exist_ok=True)

    # decoy_label_RDBE
    if decoy_suffix is None:
        decoy_suffix = f"{decomp_filter}"

    save_path = save_dir / f"decoy_label_{decoy_suffix}.tsv"
    print(f"Save to {save_path}")
    df.to_csv(save_path, sep="\t", index=None)


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Program finished in: {end_time - start_time} seconds")
