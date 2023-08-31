""" Evaluate the formatted model predictions.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from mist_cf import common


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-dataset", default="none", action="store")
    parser.add_argument(
        "--true-labels",
        type=str,
        default="data/canopus_train/decoy_labels/decoy_label_COMMON.tsv",
    )
    parser.add_argument(
        "--res-dir",
        type=str,
        default="results/mist_cf_score/2022_11_04_mist_cf_train",
        help="Path to the parent directory of model formatted output",
    )
    parser.add_argument(
        "--split-file",
        type=str,
        default="data/canopus_train/splits/canopus_[M+H]+_0.3_100.tsv",
    )
    return parser.parse_args()


escape_adduct = lambda x: x.replace(" ", "")

# Mass buckets
range_left = [0] + list(range(200, 1000, 100))
range_right = [200] + list(range(300, 1100, 100))
mr_str = [f"[{str(le)}, {str(re)}]" for le, re in zip(range_left, range_right)]

# Rel diff
rel_left = [0, 1e-9, 1e-6, 1e-3, 0.1, 1, 5, 10, 20]
rel_right = [1e-9, 1e-6, 1e-3, 0.1, 1, 5, 20, 999999]
diff_str = [f"[{str(le)}, {str(re)}]" for le, re in zip(rel_left, rel_right)]
top_k_vals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100, 200, 300, 400, 500, 600, 1000]
unk_val = 1001


def process_scores(match_ind, sorted_scores):
    """process_scores."""
    if len(match_ind) == 0:
        match_ind = unk_val
    elif len(match_ind) > 1:
        raise ValueError()
    else:
        # Break ties conservatively by picking last ind
        match_ind = match_ind[0]
        match_score = sorted_scores[match_ind]
        match_inds = np.where(sorted_scores == match_score)[0].flatten()

        # Add 1 to start at top 1 acc
        match_ind = np.max(match_inds) + 1
    return match_ind


def get_first_inds(arr):
    """
    Returns the indices of the first occurrence of each unique item in the array.
    """
    seen = set()
    unique_indices = [
        i for i, val in enumerate(arr) if val not in seen and not seen.add(val)
    ]
    return unique_indices


def main(args):
    """main."""
    subset_dataset = args.subset_dataset
    res_dir = Path(args.res_dir)
    subset_dataset = args.subset_dataset
    split_file = args.split_file
    pred_file = res_dir / "formatted_output.tsv"
    base_dir = res_dir / "evaluation_results"
    base_dir.mkdir(exist_ok=True)
    true_labels = args.true_labels

    # Load predictions
    print("Loading predictions...")
    pred_df = pd.read_csv(pred_file, sep="\t")
    true_labels = pd.read_csv(true_labels, sep="\t")
    print("Finished loading predictions.")

    # Normalize pred df forms and true forms first just to be safe?
    pred_forms = pred_df["cand_form"].values
    pred_df["cand_form"] = common.chunked_parallel(pred_forms, common.standardize_form)
    # Standardize adducts
    pred_adducts = pred_df["cand_ion"].values
    pred_df["cand_ion"] = common.chunked_parallel(
        pred_adducts, common.standardize_adduct
    )

    print("Standardizing formula...")
    true_forms = true_labels["formula"].values
    true_labels["formula"] = common.chunked_parallel(
        true_forms, common.standardize_form
    )
    true_adducts = true_labels["ionization"].values
    true_labels["ionization"] = common.chunked_parallel(
        true_adducts, common.standardize_adduct
    )
    print("Finished standardizing formula.")

    spec_to_ion = dict(true_labels[["spec", "ionization"]].values)
    spec_to_form = dict(true_labels[["spec", "formula"]].values)
    # spec_to_mass_diff = dict(true_labels[["spec", "rel_mass_diff"]].values)

    # Subset pred df
    if subset_dataset == "none":
        pass
    elif subset_dataset == "test_only":
        split_df = pd.read_csv(split_file, sep="\t")
        test_names = split_df[split_df["Fold_0"] == "test"]["spec"].to_list()
        test_names = set(test_names)
        mask = pred_df["spec"].isin(test_names)
        pred_df = pred_df[mask]
    else:
        raise NotImplementedError()

    outputs = []
    for spec, spec_group in pred_df.groupby("spec"):
        true_ion = spec_to_ion[spec]
        true_form = spec_to_form[spec]
        parent = spec_group["parentmasses"].values[0]
        mass_bin = np.digitize(parent, bins=range_right)
        if mass_bin > len(mr_str) - 1:
            mass_bin_str = f">{max(range_right)}"
        else:
            mass_bin_str = mr_str[mass_bin]

        # Sort by scores
        sort_order = np.argsort(spec_group["scores"].values)[::-1]
        sorted_group = spec_group.iloc[sort_order].reset_index()

        # Get loc of match
        cand_forms = sorted_group["cand_form"].values
        cand_scores = sorted_group["scores"].values
        uniq_inds = get_first_inds(cand_forms)
        cand_scores, cand_forms = cand_scores[uniq_inds], cand_forms[uniq_inds]
        is_match = cand_forms == true_form
        match_ind = np.argwhere(is_match).flatten()
        match_ind = process_scores(match_ind, cand_scores)

        # Get match ind for adduct
        cand_ions = sorted_group["cand_ion"].values
        cand_scores = sorted_group["scores"].values
        uniq_inds = get_first_inds(cand_ions)
        cand_scores, cand_ions = cand_scores[uniq_inds], cand_ions[uniq_inds]
        is_match = cand_ions == true_ion
        match_ind_adduct = np.argwhere(is_match).flatten()
        match_ind_adduct = process_scores(match_ind_adduct, cand_scores)

        # Get match ind for full form
        true_full_form = common.add_ion(true_form, true_ion)

        cand_forms = sorted_group["cand_form"].values
        cand_ions = sorted_group["cand_ion"].values
        cand_scores = sorted_group["scores"].values
        cand_full_forms = [
            common.add_ion(form, ion) for form, ion in zip(cand_forms, cand_ions)
        ]
        cand_full_forms = np.array(cand_full_forms)

        uniq_inds = get_first_inds(cand_full_forms)
        cand_scores, cand_full_forms = (
            cand_scores[uniq_inds],
            cand_full_forms[uniq_inds],
        )
        is_match = cand_full_forms == true_full_form
        match_ind_full_form = np.argwhere(is_match).flatten()
        match_ind_full_form = process_scores(match_ind_full_form, cand_scores)

        output = {
            "spec": spec,
            "ind_found": match_ind,
            "ind_found_adduct": match_ind_adduct,
            "ind_found_full_form": match_ind_full_form,
            "mass": parent,
            "mass_bin": mass_bin_str,
            "true_form": true_form,
            "true_ion": true_ion,
        }

        top_k_dict = {f"Top {top_k} acc.": match_ind <= top_k for top_k in top_k_vals}
        output.update(top_k_dict)

        outputs.append(output)

    # Aggregate and sort in various ways
    df = pd.DataFrame(outputs)
    full_out = base_dir / f"full_out.tsv"
    ion_out = base_dir / f"ion_acc_summary.tsv"
    mass_out = base_dir / f"mass_acc_summary.tsv"
    diff_out = base_dir / f"mass_diff_summary.tsv"
    df.to_csv(full_out, sep="\t", index=None)

    for k, outfile in zip(["true_ion", "mass_bin"], [ion_out, mass_out]):
        df_grouped = pd.concat([df.groupby(k).mean(), df.groupby(k).size()], axis=1)
        df_grouped = df_grouped.rename({0: "num_examples"}, axis=1)
        all_mean = df.mean()
        all_mean["num_examples"] = len(df)
        all_mean.name = "avg"
        all_mean = pd.DataFrame([all_mean])
        df_grouped = pd.concat([df_grouped, all_mean], axis=0)

        for top_k in top_k_vals:
            df_grouped[f"Top {top_k} acc."] = np.round(
                df_grouped[f"Top {top_k} acc."].values, 4
            )
        df_grouped.to_csv(outfile, sep="\t")


if __name__ == "__main__":
    args = get_args()
    main(args)
