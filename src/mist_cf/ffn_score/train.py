"""train.py

Train mist_cf transformer to predict binned specs

"""
from datetime import datetime
import logging
import yaml
import argparse
from pathlib import Path
import pandas as pd
import time

from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

import mist_cf.common as common
from mist_cf.ffn_score import ffn_data, ffn_model


def add_args(parser):
    parser.add_argument(
        "--debug", default=False, action="store_true", help="Debug flag"
    )
    parser.add_argument(
        "--gpu", default=False, action="store_true", help="Use GPU flag"
    )
    parser.add_argument(
        "--no-ion-info",
        default=False,
        action="store_true",
        help="Use one hot encoding of ionization information flag",
    )
    parser.add_argument(
        "--no-spec-info",
        default=False,
        action="store_true",
        help="Do not use MS2 spectrum information flag",
    )
    parser.add_argument(
        "--no-instrument-info",
        default=False,
        action="store_true",
        help="Do not use instrument encoding",
    )
    parser.add_argument(
        "--no-cls-mass-diff",
        default=False,
        action="store_true",
        help="Do not rel mass diff feature flag",
    )
    parser.add_argument(
        "--seed", default=42, action="store", type=int, help="Set random seed"
    )
    parser.add_argument(
        "--num-workers",
        default=20,
        action="store",
        type=int,
        help="Set number of CPUs available",
    )
    parser.add_argument(
        "--batch-size", default=128, action="store", type=int, help="Set batch size"
    )
    parser.add_argument(
        "--max-decoy",
        default=256,
        action="store",
        type=int,
        help="Set maximum number of decoy to subsample during FFN training due to GPU memory constraint",
    )
    parser.add_argument(
        "--max-epochs",
        default=100,
        action="store",
        type=int,
        help="Set maximum number of epochs",
    )
    parser.add_argument(
        "--learning-rate",
        default=7e-4,
        action="store",
        type=float,
        help="Set maximum number of epochs",
    )
    parser.add_argument("--lr-decay-frac", default=1.0, action="store", type=float)
    parser.add_argument("--weight-decay", default=0.0, action="store", type=float)
    date = datetime.now().strftime("%Y_%m_%d_%H")
    parser.add_argument(
        "--save-dir",
        default=f"results/ffn_model/{date}_ffn_train/",
        help="Path of save directory",
    )

    parser.add_argument(
        "--dataset-name", default="canopus_train", help="Name of the dataset"
    )
    parser.add_argument(
        "--split-file", default="canopus_[M+H]+_0.3_100.tsv", help="Path to split file"
    )
    parser.add_argument(
        "--decoy-label",
        default="label_decoy_RDBE_256_by_adduct.tsv",
        help="Path to decoy label",
    )
    parser.add_argument(
        "--num-bins",
        default=1000,
        action="store",
        type=int,
        help="Number of bins for processing MS2 spectrum",
    )
    parser.add_argument(
        "--layers", default=3, action="store", type=int, help="Number of layers of FFN"
    )
    parser.add_argument(
        "--dropout",
        default=0.1,
        action="store",
        type=float,
        help="Dropout parameter of FFN",
    )
    parser.add_argument(
        "--hidden-size",
        default=256,
        action="store",
        type=int,
        help="Hidden size parameter of FFN",
    )
    parser.add_argument("--form-encoder", type=str, default="abs-sines")


def get_args():
    parser = argparse.ArgumentParser()
    add_args(parser)
    return parser.parse_args()


def train_model():
    args = get_args()
    kwargs = args.__dict__
    kwargs["spec_info"] = not kwargs["no_spec_info"]
    kwargs["cls_mass_diff"] = not kwargs["no_cls_mass_diff"]
    kwargs["instrument_info"] = not kwargs["no_instrument_info"]
    kwargs["ion_info"] = not kwargs["no_ion_info"]
    save_dir = Path(kwargs["save_dir"])
    save_dir.mkdir(exist_ok=True)

    common.setup_logger(save_dir, log_name="ffn_train.log", debug=kwargs["debug"])
    pl.utilities.seed.seed_everything(kwargs.get("seed"))

    # Dump args
    yaml_args = yaml.dump(kwargs, indent=2, default_flow_style=False)

    with open(Path(save_dir) / "args.yaml", "w") as fp:
        fp.write(yaml_args)

    logging.info(f"Args:\n{yaml_args}")
    # Get dataset
    # Load smiles dataset and split into 3 subsets
    dataset_name = kwargs["dataset_name"]
    data_dir = common.get_data_dir(dataset_name)
    labels = Path(args.decoy_label)
    split_file = Path(args.split_file)

    # Get train, val, test inds
    df = pd.read_csv(labels, sep="\t")
    spec_names = df["spec"].values
    train_inds, val_inds, test_inds = common.get_splits(spec_names, split_file)

    train_df = df.iloc[train_inds]
    val_df = df.iloc[val_inds]
    test_df = df.iloc[test_inds]

    if args.debug:
        train_df = train_df[:100]
        val_df = val_df[:100]
        test_df = test_df[:100]
        kwargs['num_workers'] = 0

    num_bins = kwargs.get("num_bins")
    num_workers = kwargs.get("num_workers", 0)

    train_dataset = ffn_data.BinnedDataset(
        train_df,
        data_dir=data_dir,
        num_bins=num_bins,
        max_decoy=args.max_decoy,
        val_test=False,
        num_workers=num_workers,
    )
    val_dataset = ffn_data.BinnedDataset(
        val_df,
        data_dir=data_dir,
        num_bins=num_bins,
        max_decoy=args.max_decoy,
        val_test=True,
        num_workers=num_workers,
    )
    test_dataset = ffn_data.BinnedDataset(
        test_df,
        data_dir=data_dir,
        num_bins=num_bins,
        max_decoy=args.max_decoy,
        val_test=True,
        num_workers=num_workers,
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
    test_loader = DataLoader(
        test_dataset,
        num_workers=kwargs["num_workers"],
        collate_fn=collate_fn,
        shuffle=False,
        batch_size=kwargs["batch_size"],
    )

    # Define model
    test_batch = next(iter(train_loader))

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
    # model_outs = model(test_batch['spec_ars'], test_batch['formula_tensors'],)
    # Create trainer
    tb_logger = pl_loggers.TensorBoardLogger(save_dir, name="")
    console_logger = common.ConsoleLogger()

    tb_path = tb_logger.log_dir
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=tb_path,
        filename="best", #"{epoch}-{val_loss:.2f}",
        save_weights_only=True,
    )
    earlystop_callback = EarlyStopping(monitor="val_loss", patience=5)
    callbacks = [earlystop_callback, checkpoint_callback]

    trainer = pl.Trainer(
        logger=[tb_logger, console_logger],
        accelerator="gpu" if kwargs["gpu"] else "cpu",
        gpus=1 if kwargs["gpu"] else 0,
        callbacks=callbacks,
        gradient_clip_val=5,
        max_epochs=kwargs["max_epochs"],
        gradient_clip_algorithm="value",
    )

    trainer.fit(model, train_loader, val_loader)
    checkpoint_callback = trainer.checkpoint_callback
    best_checkpoint = checkpoint_callback.best_model_path
    best_checkpoint_score = checkpoint_callback.best_model_score.item()

    # Load from checkpoint

    model = ffn_model.ForwardFFN.load_from_checkpoint(best_checkpoint)
    logging.info(
        f"Loaded model with from {best_checkpoint} with val loss of {best_checkpoint_score}"
    )

    model.eval()
    test_out = trainer.test(dataloaders=test_loader)

    out_yaml = {"args": kwargs, "test_metrics": test_out[0]}
    out_str = yaml.dump(out_yaml, indent=2, default_flow_style=False)

    with open(Path(save_dir) / "test_results.yaml", "w") as fp:
        fp.write(out_str)

    logging.info(f"num of train instances: {len(train_inds)}")
    logging.info(f"num of val instances: {len(val_inds)}")
    logging.info(f"num of test instances: {len(test_inds)}")


if __name__ == "__main__":
    start_time = time.time()
    train_model()
    end_time = time.time()
    logging.info(f"Program finished in: {end_time - start_time} seconds")
