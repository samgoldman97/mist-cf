"""predict.py

Make predictions with trained model

"""
import logging
import yaml
import argparse
from pathlib import Path
import time

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

import mist_cf.common as common
from mist_cf.mist_cf_score import mist_cf_data, mist_cf_model


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
        default=8,
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
        default="data/canopus_train/pred_labels/pred_label_decoy_RDBE_256_22_by_adduct_canopus_test_0.3_val_0.1_22.tsv",
        help="Path to prediction label",
    )
    parser.add_argument(
        "--dataset-name", default="canopus_train", help="Name of the dataset"
    )
    parser.add_argument(
        "--checkpoint-pth",
        default="results/2022_09_22_ffn_train/version_0/epoch=20-val_loss=0.46.ckpt",
        help="Path to check point file",
    )
    parser.add_argument(
        "--subform-dir",
        type=str,
        default="",
        help="Path to subformulae assignment directory",
    )
    return parser.parse_args()


def predict():
    args = get_args()
    kwargs = args.__dict__
    debug = kwargs["debug"]

    if args.save_dir is None:
        raise ValueError("Please specify the save dir.")

    save_dir = Path(kwargs['save_dir'])
    save_name = kwargs.get("save_name")
    save_name = "formatted_output.tsv" if save_name is None else save_name
    save_name = save_dir / save_name

    common.setup_logger(
        kwargs["save_dir"], log_name=f"mist_cf_pred.log", debug=kwargs["debug"]
    )
    pl.utilities.seed.seed_everything(kwargs.get("seed"))

    # Dump args
    yaml_args = yaml.dump(kwargs, indent=2, default_flow_style=False)
    logging.info(f"Args:\n{yaml_args}")
    with open(Path(kwargs["save_dir"]) / "args.yaml", "w") as fp:
        fp.write(yaml_args)

    # Get dataset
    dataset_name = kwargs['dataset_name']
    data_dir = common.get_data_dir(dataset_name)

    # Get corresponding prediction label
    pred_label_path = Path(kwargs['pred_label'])
    df = pd.read_csv(pred_label_path, sep="\t")

    if debug:
        df = df[:10000]

    # Get train, val, test inds
    num_workers = kwargs.get("num_workers", 0)

    # Create model and load
    best_checkpoint = kwargs["checkpoint_pth"]

    # Load from checkpoint
    model = mist_cf_model.MistNet.load_from_checkpoint(best_checkpoint)
    logging.info(f"Loaded model with from {best_checkpoint}")

    # Create dataset
    # Define num bins
    pred_dataset = mist_cf_data.PredDataset(
        df,
        data_dir=data_dir,
        subform_dir=kwargs['subform_dir'],
        num_workers=num_workers,
        max_subpeak=model.max_subpeak,
        ablate_cls_error=not model.cls_mass_diff
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

    out_names, out_forms, out_scores, out_ions, out_parentmasses = [], [], [], [], []
    with torch.no_grad():
        for batch in pred_loader:
            peak_types, form_vec, ion_vec, instrument_vec, intens, rel_mass_diffs, num_peaks = (
                batch["types"],
                batch["form_vec"],
                batch["ion_vec"],
                batch["instrument_vec"],
                batch["intens"],
                batch["rel_mass_diffs"],
                batch["num_peaks"],
            )
            peak_types = peak_types.to(device)
            form_vec = form_vec.to(device)
            ion_vec = ion_vec.to(device)
            instrument_vec = instrument_vec.to(device)
            intens = intens.to(device)
            rel_mass_diffs = rel_mass_diffs.to(device)
            num_peaks = num_peaks.to(device)

            model_outs = model.forward(
                num_peaks, peak_types, form_vec, ion_vec, instrument_vec, intens, rel_mass_diffs
            )
            # ex_inds = batch['example_inds'].long()
            # num_inputs = batch['num_inputs']

            actual_forms = batch["str_forms"]
            actual_ions = batch["str_ions"]
            parentmasses = batch["parentmasses"]
            scores = model_outs.squeeze().cpu().numpy()
            # names = np.array(batch['names'])[batch['example_inds'].cpu().numpy()]
            names = np.array(batch["names"])

            out_names.extend(names)
            out_scores.extend(scores)
            out_forms.extend(actual_forms)
            out_ions.extend(actual_ions)
            out_parentmasses.extend(parentmasses)

        output = {
            "names": out_names,
            "forms": out_forms,
            "scores": out_scores,
            "ions": out_ions,
            "parentmasses": out_parentmasses,
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
