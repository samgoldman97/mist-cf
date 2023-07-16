"""predict.py

Make predictions with trained model

"""
import logging
import yaml
import argparse
import pickle
from pathlib import Path

from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

import mist_cf.common as common
import mist_cf.nn_utils as nn_utils
from mist_cf.fast_form_score import fast_form_data, fast_form_model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--gpu", default=False, action="store_true")
    parser.add_argument("--num-workers", default=0, action="store", type=int)
    parser.add_argument("--batch-size", default=64, action="store", type=int)
    parser.add_argument(
        "--save-name", default=None, help="Set name of the prediction file"
    )
    parser.add_argument("--save-dir", default=None, help="Path to save directory")
    parser.add_argument(
        "--pred-label",
        default="data/canopus_train/pred_labels/pred_label_decoy_RDBE_256_22_by_adduct_canopus_test_0.3_val_0.1_22.tsv",
        help="Path to prediction label",
    )
    parser.add_argument("--dataset-name", default="canopus_train")
    parser.add_argument(
        "--checkpoint-pth",
        help="name of checkpoint file",
        default="results/2022_10_05_fast_form_score/version_0/epoch=2-val_loss=0.03.ckpt",
    )
    return parser.parse_args()


def predict():
    args = get_args()
    kwargs = args.__dict__
    dataset_name = kwargs['dataset_name']
    debug = kwargs["debug"]

    save_dir = Path(kwargs['save_dir'])
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
    pred_label_path = Path(kwargs['pred_label'])
    df = pd.read_csv(pred_label_path, sep="\t")

    if debug:
        df = df[:200]
        kwargs['num_workers'] = 0

    # Get train, val, test inds
    num_workers = kwargs.get("num_workers", 0)

    # Create model and load
    best_checkpoint = kwargs["checkpoint_pth"]

    # Load from checkpoint
    model = fast_form_model.FastFFN.load_from_checkpoint(best_checkpoint)
    logging.info(f"Loaded model with from {best_checkpoint}")

    # Create dataset
    # Define num bins
    pred_dataset = fast_form_data.PredDataset(
        df,
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
    # test_batch = next(iter(pred_loader))

    model.eval()
    gpu = kwargs["gpu"]
    device = torch.device("cuda") if gpu else torch.device("cpu")
    model = model.to(device)

    out_names, out_forms, out_ions, out_scores = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(pred_loader):
            x = batch["x"].to(device)
            model_outs = model.forward(x.float())

            scores = model_outs.squeeze().cpu().numpy()
            actual_forms = batch["str_forms"]
            names = np.array(batch["names"])
            ions = np.array(batch["ions"])

            out_names.extend(names)
            out_scores.extend(scores)
            out_forms.extend(actual_forms)
            out_ions.extend(ions)

    fn2true_parentmass = dict(df[["spec", "parentmass"]].values)
    output = {"names": out_names, "forms": out_forms, "scores": out_scores,
              "ions": out_ions,
              "parentmasses": [fn2true_parentmass[fn] for fn in out_names],
              }

    out_df = pd.DataFrame(output)
    out_df = out_df.rename(
        columns={"names": "spec", "forms": "cand_form", "ions": "cand_ion"}
    )
    out_df.to_csv(save_name, index=None, sep="\t")


if __name__ == "__main__":
    import time
    start_time = time.time()
    predict()
    end_time = time.time()
    print(f"Program finished in: {end_time - start_time} seconds")
