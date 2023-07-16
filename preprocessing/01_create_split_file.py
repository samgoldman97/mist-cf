""" make_splits.py
Make train-test-val splits by formula.
"""


import argparse
from pathlib import Path

import pandas as pd
import numpy as np


def get_args():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--data-dir", default="data/canopus_train", help="Data directory."
    )
    args.add_argument(
        "--label-file",
        default="data/canopus_train/labels.tsv",
        help="Path to label file.",
    )
    args.add_argument("--out-name", default=None, help="Out prefix")
    args.add_argument(
        "--ionization",
        type=str,
        default=None,
        help="Ion the user want to focused on (if applicable)",
    )
    args.add_argument(
        "--val-frac",
        type=float,
        default=0.1,
        help="Percentage of validation data out of all training data.",
    )
    args.add_argument(
        "--test-frac",
        type=float,
        default=0.3,
        help="Percentage of test data out of all data.",
    )
    args.add_argument("--seed", type=int, default=1, help="Random seed be reproducible")
    return args.parse_args()


def main():
    args = get_args()
    seed = args.seed
    ionization = args.ionization
    test_frac = args.test_frac
    val_frac = args.val_frac
    data_dir = args.data_dir
    label_file = args.label_file
    out_name = args.out_name

    hyperopt_train_num = 10000

    if out_name is None:
        out_name = f"split_{seed}.tsv"
    out_name = Path(out_name)
    out_with_nist = f"{out_name.stem}_with_nist.tsv"
    out_hyperopt = f"{out_name.stem}_hyperopt_{hyperopt_train_num}.tsv"


    label_path = Path(label_file)
    df = pd.read_csv(label_path, sep="\t")

    nist_df_subs = df['dataset'] == "nist2020"

    df_nist = df[nist_df_subs].reset_index(drop=True)
    df = df[~nist_df_subs].reset_index(drop=True)

    if ionization is not None:
        df = df[df["ionization"] == ionization]

    formula_set = set(df["formula"].values)

    train_frac, test_frac = (1 - test_frac), test_frac
    num_train = int(train_frac * len(formula_set))

    # Divide by formulaes
    full_formula_list = list(formula_set)
    np.random.seed(seed)
    np.random.shuffle(full_formula_list)

    train = set(full_formula_list[:num_train])
    test = set(full_formula_list[num_train:])

    output_dir = Path(data_dir) / "splits"
    output_dir.mkdir(exist_ok=True)

    fold_num = 0
    fold_name = f"Fold_{fold_num}"
    val_num = int(len(train) * val_frac)
    np.random.seed(seed)
    val = set(np.random.choice(list(train), val_num, replace=False))

    # Remove val formulae inds from train formulae
    train = train.difference(val)

    print(f"Num train total formulae: {len(train)}")
    print(f"Num val total formulae: {len(val)}")
    print(f"Num test total formulae: {len(test)}")

    split_data = {"spec": [], fold_name: []}
    for _, row in df.iterrows():
        spec_form = row["formula"]
        spec = row["spec"]

        if spec_form in train:
            fold = "train"
        elif spec_form in test:
            fold = "test"
        elif spec_form in val:
            fold = "val"
        else:
            fold = "exclude"
        split_data["spec"].append(spec)
        split_data[fold_name].append(fold)

    assert len(split_data["spec"]) == df.shape[0]
    assert len(split_data[fold_name]) == df.shape[0]
    export_df = pd.DataFrame(split_data)
    export_df = export_df.sort_values("spec", ascending=True).reset_index(drop=True)
    export_df.to_csv(output_dir / out_name, sep="\t", index=False)

    # Create a new variant that has nist data
    for _, row in df_nist.iterrows():
        spec_form = row["formula"]
        spec = row["spec"]
        if (spec_form not in test) and (spec_form not in val):
            fold = "train"
        else:
            fold = "exclude"
        split_data["spec"].append(spec)
        split_data[fold_name].append(fold)

    export_df = pd.DataFrame(split_data)
    export_df = export_df.sort_values("spec", ascending=True).reset_index(drop=True)
    export_df.to_csv(output_dir / out_with_nist, sep="\t", index=False)

    # Subset train to 10k for a hyperopt split
    all_train = (export_df['Fold_0'] == 'train').values
    train_inds = np.where(all_train)[0]
    train_num = len(train_inds)
    to_exclude = max(0, train_num - hyperopt_train_num)
    exclude_inds = np.random.choice(train_inds, to_exclude, replace=False)

    new_train = np.sum((export_df['Fold_0'] == 'train').values)
    export_df.loc[exclude_inds, "Fold_0"] = "exclude"

    # Verify new export
    new_train = np.sum((export_df['Fold_0'] == 'train').values)
    print("New_train", new_train)

    export_df = export_df.sort_values("spec", ascending=True)
    export_df.to_csv(output_dir / out_hyperopt, sep="\t", index=False)


if __name__ == "__main__":
    main()
