"""predict.py

Make predictions with trained model

"""
import logging
import yaml
import argparse
from pathlib import Path
import time

import pandas as pd

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

import mist_cf.common as common
from mist_cf.ffn_score import ffn_data, ffn_model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug", default=False, action="store_true", help="Debug flag"
    )
    parser.add_argument(
        "--gpu", default=False, action="store_true", help="Use GPU flag"
    )
    parser.add_argument(
        "--num-workers",
        default=0,
        action="store",
        type=int,
        help="Set number of available CPUs",
    )
    parser.add_argument(
        "--batch-size",
        default=64,
        action="store",
        type=int,
        help="Set number of batch size",
    )
    parser.add_argument(
        "--save-name", default=None, help="Set name of the prediction file"
    )
    parser.add_argument("--save-dir", default=None, help="Path to save directory")

    parser.add_argument(
        "--pred-label",
        help="Path to prediction label",
    )
    parser.add_argument(
        "--dataset-name", default="canopus_train", help="Name of the dataset"
    )
    parser.add_argument(
        "--checkpoint-pth",
        help="Path to check point file",
    )

    return parser.parse_args()


def predict():
    args = get_args()
    kwargs = args.__dict__
    dataset_name = kwargs["dataset_name"]
    debug = kwargs["debug"]

    save_dir = Path(kwargs["save_dir"])
    save_name = kwargs.get("save_name")
    save_name = "formatted_output.tsv" if save_name is None else save_name
    save_name = save_dir / save_name

    common.setup_logger(
        kwargs["save_dir"], log_name=f"ffn_pred.log", debug=kwargs["debug"]
    )
    pl.utilities.seed.seed_everything(kwargs.get("seed"))

    # Dump args
    yaml_args = yaml.dump(kwargs, indent=2, default_flow_style=False)
    logging.info(f"Args:\n{yaml_args}")
    with open(Path(kwargs["save_dir"]) / "args.yaml", "w") as fp:
        fp.write(yaml_args)

    # Get dataset
    data_dir = common.get_data_dir(dataset_name)

    # Get corresponding prediction label
    pred_label_path = Path(kwargs["pred_label"])
    df = pd.read_csv(pred_label_path, sep="\t")

    if debug:
        df = df[:200]
        kwargs["num_workers"] = 0

    # Get train, val, test inds
    num_workers = kwargs.get("num_workers", 0)

    # Create model and load
    best_checkpoint = kwargs["checkpoint_pth"]

    # Load from checkpoint
    model = ffn_model.ForwardFFN.load_from_checkpoint(best_checkpoint)
    logging.info(f"Loaded model with from {best_checkpoint}")

    # Create dataset
    # Define num bins
    pred_dataset = ffn_data.PredDataset(
        df,
        data_dir=data_dir,
        num_bins=model.num_spec_bins,
        num_workers=num_workers,
    )
    # Define dataloaders
    collate_fn = pred_dataset.get_collate_fn()
    pred_loader = DataLoader(
        pred_dataset,
        num_workers=kwargs["num_workers"],
        collate_fn=collate_fn,
        shuffle=False,
        batch_size=kwargs["batch_size"],
    )

    model.eval()
    gpu = kwargs["gpu"]
    device = torch.device("cuda") if gpu else torch.device("cpu")
    model = model.to(device)

    out_names, out_ions, out_forms, out_scores = [], [], [], []
    with torch.no_grad():
        for batch in pred_loader:
            parent_mass_diffs = batch["rel_mass_diffs"]
            spec_ars = batch["spec_ars"]
            forms = batch["formula_tensors"]
            ions = batch["ion_tensors"]
            instrument_tensors = batch["instrument_tensors"]

            parent_mass_diffs = parent_mass_diffs.to(device)
            spec_ars = spec_ars.to(device)
            forms = forms.to(device)
            ions = ions.to(device)
            instrument_tensors = instrument_tensors.to(device)

            model_outs = model.forward(
                spec_ars.float(),
                forms.float(),
                parent_mass_diffs.float(),
                ions.float(),
                instrument_tensors.float(),
            )
            # ex_inds = batch['example_inds'].long()
            # num_inputs = batch['num_inputs']

            actual_forms = batch["str_forms"]
            actual_ions = batch["cand_ions"]
            scores = model_outs.squeeze().cpu().numpy()
            names = batch["names"]

            out_names.extend(names)
            out_scores.extend(scores)
            out_ions.extend(actual_ions)
            out_forms.extend(actual_forms)

        fn2true_parentmass = dict(df[["spec", "parentmass"]].values)
        output = {
            "names": out_names,
            "forms": out_forms,
            "ions": out_ions,
            "scores": out_scores,
            "parentmasses": [fn2true_parentmass[fn] for fn in out_names],
        }

        out_df = pd.DataFrame(output)
        out_df = out_df.rename(
            columns={"names": "spec", "forms": "cand_form", "ions": "cand_ion"}
        )
        out_df.to_csv(save_name, sep="\t", index=None)


if __name__ == "__main__":
    start_time = time.time()
    predict()
    end_time = time.time()
    print(f"Program finished in: {end_time - start_time} seconds")
