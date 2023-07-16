"""ffn_hyperopt.py

Hyperopt parameters

"""
import os
import copy
import logging
import argparse
from pathlib import Path
import pandas as pd
from typing import List, Dict

from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from ray import tune

# from ray.tune.integration.pytorch_lightning import TuneReportCallback

from mist_cf import common
from mist_cf.ffn_score import ffn_data, ffn_model, train
from mist_cf.nn_utils import base_hyperopt


def score_function(config, base_args, orig_dir=""):
    """score_function.

    Args:
        config: All configs passed by hyperoptimizer
        base_args: Base arguments
        orig_dir: ""
    """
    # tunedir = tune.get_trial_dir()
    # Switch s.t. we can use relative data structures
    os.chdir(orig_dir)

    kwargs = copy.deepcopy(base_args)
    kwargs["spec_info"] = not kwargs["no_spec_info"]
    kwargs["cls_mass_diff"] = not kwargs["no_cls_mass_diff"]
    kwargs["instrument_info"] = not kwargs["no_instrument_info"]
    kwargs["ion_info"] = not kwargs["no_ion_info"]
    kwargs.update(config)
    pl.utilities.seed.seed_everything(kwargs.get("seed"))

    # Get dataset
    # Load smiles dataset and split into 3 subsets
    dataset_name = kwargs["dataset_name"]
    data_dir = common.get_data_dir(dataset_name)
    labels = Path(kwargs["decoy_label"])
    split_file = Path(kwargs["split_file"])

    # Get train, val, test inds
    df = pd.read_csv(labels, sep="\t")
    spec_names = df["spec"].values
    train_inds, val_inds, test_inds = common.get_splits(spec_names, split_file)
    train_df = df.iloc[train_inds]
    val_df = df.iloc[val_inds]

    if kwargs.get("debug", False):
        train_df = train_df[:20]
        val_df = val_df[:20]

    num_workers = kwargs.get("num_workers", 0)
    logging.info("Making datasets")

    train_dataset = ffn_data.BinnedDataset(
        train_df,
        data_dir=data_dir,
        num_bins=kwargs["num_bins"],
        max_decoy=kwargs["max_decoy"],
        num_workers=num_workers,
        val_test=False,
        use_ray=True,
    )
    val_dataset = ffn_data.BinnedDataset(
        val_df,
        data_dir=data_dir,
        num_bins=kwargs["num_bins"],
        max_decoy=kwargs["max_decoy"],
        num_workers=num_workers,
        val_test=True,
        use_ray=True,
    )

    # Define dataloaders
    collate_fn = train_dataset.get_collate_fn()
    train_loader = DataLoader(
        train_dataset,
        num_workers=kwargs["num_workers"],
        collate_fn=collate_fn,
        shuffle=True,
        batch_size=kwargs["batch_size"],
    )
    val_loader = DataLoader(
        val_dataset,
        num_workers=kwargs["num_workers"],
        collate_fn=collate_fn,
        shuffle=False,
        batch_size=kwargs["batch_size"],
    )

    # Define model
    # test_batch = next(iter(train_loader))
    logging.info("Building model")
    model = ffn_model.ForwardFFN(
        hidden_size=kwargs["hidden_size"],
        layers=kwargs["layers"],
        dropout=kwargs["dropout"],
        ion_info=kwargs["ion_info"],
        spec_info=kwargs["spec_info"],
        cls_mass_diff=kwargs["cls_mass_diff"],
        instrument_info=kwargs["instrument_info"],
        learning_rate=kwargs["learning_rate"],
        num_spec_bins=kwargs["num_bins"],
        lr_decay_frac=kwargs['lr_decay_frac'],
        weight_decay=kwargs['weight_decay'],
        form_encoder=kwargs['form_encoder']
    )

    # Create trainer
    tb_logger = pl_loggers.TensorBoardLogger(tune.get_trial_dir(), "", ".")

    # Replace with custom callback that utilizes maximum loss during train
    tune_callback = common.TuneReportCallback(["val_loss"])

    # val_check_interval = None#2000 #2000
    # check_val_every_n_epoch = 1

    monitor = "val_loss"
    # tb_path = tb_logger.log_dir
    earlystop_callback = EarlyStopping(monitor=monitor, patience=5)
    callbacks = [earlystop_callback, tune_callback]
    logging.info("Starting train")
    trainer = pl.Trainer(
        logger=[tb_logger],
        accelerator="gpu" if kwargs["gpu"] else "cpu",
        gpus=1 if kwargs["gpu"] else 0,
        callbacks=callbacks,
        gradient_clip_val=5,
        max_epochs=kwargs["max_epochs"],
        gradient_clip_algorithm="value",
    )
    trainer.fit(model, train_loader, val_loader)


def get_args():
    parser = argparse.ArgumentParser()
    train.add_args(parser)
    base_hyperopt.add_hyperopt_args(parser)
    return parser.parse_args()


def get_param_space(trial):
    """get_param_space.

    Use optuan to define this ydanmically

    """
    trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
    trial.suggest_float("dropout", 0, 0.5, step=0.1)
    trial.suggest_categorical("hidden_size", [32, 64, 128, 256, 512])
    trial.suggest_int("layers", 1, 4)
    trial.suggest_categorical("batch_size", [16, 32, 64])
    trial.suggest_categorical("num_bins", list(range(1000, 11000, 1000)))

    trial.suggest_float("lr_decay_frac", 0.7, 1.0, log=True)
    trial.suggest_categorical("weight_decay", [1e-6, 1e-7, 0.0],)
    trial.suggest_categorical("form_encoder", ["abs-sines"],)



def get_initial_points() -> List[Dict]:
    """get_intiial_points.

    Create dictionaries defining initial configurations to test

    """
    init_base = {
        "learning_rate": 0.0007713366639953808,
        "dropout": 0.3,
        "hidden_size": 512,
        "layers": 1,
        "batch_size": 64,
        "num_bins": 5000,
        "weight_decay": 1e-6,
        "lr_decay_frac": 0.8512732067157704,
        "form_encoder": "abs-sines"
    }
    return [init_base]


def run_hyperopt():
    args = get_args()
    kwargs = args.__dict__
    base_hyperopt.run_hyperopt(kwargs=kwargs, score_function=score_function,
                               param_space_function=get_param_space,
                               initial_points=get_initial_points())

if __name__=="__main__":
    import time
    start_time = time.time()
    run_hyperopt()
    end_time = time.time()
    print(f"Program finished in: {end_time - start_time} seconds")
