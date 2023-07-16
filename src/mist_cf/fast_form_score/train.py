"""train.py

Train fast formula filter as a function _only_ of the fingerprint

"""

from datetime import datetime
import logging
import yaml
import argparse
from pathlib import Path
import pandas as pd

from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

import mist_cf.common as common
from mist_cf.fast_form_score import fast_form_data, fast_form_model

def add_args(parser):
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--gpu", default=False, action="store_true")
    parser.add_argument("--seed", default=42, action="store", type=int)
    parser.add_argument("--num-workers", default=32, action="store", type=int)
    parser.add_argument("--batch-size", default=128, action="store", type=int)

    parser.add_argument("--learning-rate", default=7e-4, action="store", type=float)
    parser.add_argument("--lr-decay-frac", default=1.0, action="store", type=float)
    parser.add_argument("--weight-decay", default=0.0, action="store", type=float)

    parser.add_argument("--max-decoy", default=32, action="store", type=int)
    parser.add_argument("--max-epochs", default=100, action="store", type=int)
    date = datetime.now().strftime("%Y_%m_%d")
    parser.add_argument("--save-dir", default=f"results/{date}_fast_form_score/")

    parser.add_argument(
        "--dataset-file", default="data/biomols/biomols_with_decoys.txt"
    )
    parser.add_argument(
        "--split-file", default="data/biomols/biomols_with_decoys_split.tsv"
    )

    parser.add_argument("--layers", default=3, action="store", type=int)
    parser.add_argument("--dropout", default=0.1, action="store", type=float)
    parser.add_argument("--hidden-size", default=512, action="store", type=int)
    parser.add_argument("--form-encoder", type=str, default="abs-sines")
    return parser

def get_args():
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    return parser.parse_args()


def train_model():
    args = get_args()
    kwargs = args.__dict__
    debug = kwargs["debug"]

    save_dir = kwargs.get("save_dir")
    common.setup_logger(save_dir, log_name="ffn_train.log", debug=kwargs["debug"])
    pl.utilities.seed.seed_everything(kwargs.get("seed"))

    # Dump args
    yaml_args = yaml.dump(kwargs, indent=2, default_flow_style=False)
    with open(Path(save_dir) / "args.yaml", "w") as fp:
        fp.write(yaml_args)

    logging.info(f"Args:\n{yaml_args}")

    # Get dataset
    # Load smiles dataset and split into 3 subsets
    dataset_file = kwargs["dataset_file"]
    split_file = kwargs["split_file"]

    # Get train, val, test inds
    df = pd.read_csv(dataset_file, sep="\t").fillna("")
    if debug:
        df = df[:1000]

    masses = df["mass"].values
    train_inds, val_inds, test_inds = common.get_splits(masses, split_file,
                                                        key="mass")

    train_df = df.iloc[train_inds]
    val_df = df.iloc[val_inds]
    test_df = df.iloc[test_inds]

    num_workers = kwargs.get("num_workers", 0)
    train_dataset = fast_form_data.FormDataset(
        train_df, num_workers=num_workers, decoys_per_pos=kwargs["max_decoy"]
    )
    val_dataset = fast_form_data.FormDataset(
        val_df, num_workers=num_workers, decoys_per_pos=kwargs["max_decoy"],
        val_test=True
    )
    test_dataset = fast_form_data.FormDataset(
        test_df, num_workers=num_workers, decoys_per_pos=kwargs["max_decoy"],
        val_test=True

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
    model = fast_form_model.FastFFN(
        hidden_size=kwargs["hidden_size"],
        layers=kwargs["layers"],
        dropout=kwargs["dropout"],
        learning_rate=kwargs["learning_rate"],
        lr_decay_frac=kwargs['lr_decay_frac'],
        weight_decay=kwargs['weight_decay'],
        form_encoder=kwargs['form_encoder']
    )
    # model_outs = model(test_batch['x'])

    # Create trainer
    tb_logger = pl_loggers.TensorBoardLogger(save_dir, name="")
    console_logger = common.ConsoleLogger()

    tb_path = tb_logger.log_dir
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=tb_path,
        filename="best",# "{epoch}-{val_loss:.2f}",
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
    model = fast_form_model.FastFFN.load_from_checkpoint(best_checkpoint)
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
    import time
    start_time = time.time()
    train_model()
    end_time = time.time()
    logging.info(f"Program finished in: {end_time - start_time} seconds")
