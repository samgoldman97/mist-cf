import logging
import torch
import pytorch_lightning as pl
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

import mist_cf.nn_utils as nn_utils

import pandas as pd
from mist_cf.fast_form_score import fast_form_data
from torch.utils.data import DataLoader


class FastFFN(pl.LightningModule):
    def __init__(
        self,
        hidden_size: int,
        layers: int = 2,
        dropout: float = 0.0,
        formula_size: int = 17,
        learning_rate: float = 7e-4,
        lr_decay_frac: float = 1.0,
        weight_decay: float = 0.0,
        form_encoder: str = "abs-sines",
        **kwargs,
    ):
        """__init__.
        Args:
            hidden_size (int): Hidden size
            layers (int): Num layers
            dropout (float): Amount of dropout
            num_spec_bins (int): Number of spectra bins
            formula_size (int): Size of chemical formula
            lr_decay_frac (float): Amount of learning rate decay
            learning_rate (float): Learning rate
            min_lr (float): Min lr
        """
        super().__init__()
        self.save_hyperparameters()
        self.hidden_size = hidden_size

        self.layers = layers
        self.dropout = dropout

        self.learning_rate = learning_rate
        self.lr_decay_frac = lr_decay_frac
        self.weight_decay = weight_decay
        self.form_embedder = nn_utils.get_embedder(form_encoder)
        self.input_dim = self.form_embedder.full_dim

        self.mlp = nn_utils.MLPBlocks(
            input_size=self.input_dim,
            hidden_size=self.hidden_size,
            dropout=self.dropout,
            num_layers=self.layers,
        )
        self.output_layer = nn.Linear(self.hidden_size, 1)
        self.output_layer = nn.Linear(self.hidden_size, 1)
        self.output_activation = nn.Sigmoid()

    def forward(self, formulae):
        """predict spec."""
        inputs = self.form_embedder(formulae)
        output = self.mlp(inputs)
        output = self.output_layer(output)
        output = self.output_activation(output)
        return output.squeeze()

    def _common_step(self, batch, name="train"):
        x, y = batch["x"], batch["y"].float()
        model_outs = self.forward(x.float())
        bce_loss = F.binary_cross_entropy(model_outs, y)
        output_loss = {"loss": bce_loss}
        self.log(f"{name}_loss", bce_loss)
        return output_loss

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, name="train")

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, name="val")

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, name="test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        scheduler = nn_utils.build_lr_scheduler(
            optimizer, lr_decay_rate=self.lr_decay_frac
        )
        ret = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "frequency": 1,
                "interval": "step",
            },
        }
        return ret

    def fast_filter_sampling(
        self,
        spec,
        decoy_ion_lst,
        decoy_ions,
        max_decoy,
        device: torch.device,
        batch_size: int = 128,
        num_workers: int = 0,
    ) -> pd.DataFrame:
        """fast_filter.
        Args:
        Returns:
            Indices into decoy_ion_lst
        """
        if len(decoy_ion_lst) == 0:
            return []
        num = len(decoy_ion_lst)
        data = {
            "spec": [spec] * num,
            "cand_form": decoy_ion_lst,
            "cand_ion": decoy_ions,
        }
        label_df = pd.DataFrame.from_dict(data)

        # Create dataset
        pred_dataset = fast_form_data.PredDataset(
            label_df,
            num_workers=num_workers,
        )
        # Define dataloaders
        collate_fn = pred_dataset.get_collate_fn()
        pred_loader = DataLoader(
            pred_dataset,
            num_workers=num_workers,
            collate_fn=collate_fn,
            shuffle=False,
            batch_size=batch_size,
        )
        # test_batch = next(iter(pred_loader))

        self.eval()
        self = self.to(device)

        out_names, out_forms, out_ions, out_scores = [], [], [], []

        with torch.no_grad():
            for batch in pred_loader:
                x = batch["x"].to(device)
                model_outs = self.forward(x.float())

                scores = model_outs.squeeze().cpu().numpy()
                actual_forms = batch["str_forms"]

                out_scores.extend(scores.reshape(-1))
                out_forms.extend(actual_forms.reshape(-1))

        out_scores = np.array(out_scores)
        out_forms = np.array(out_forms)

        # Higher should be better
        sorted_idx = np.argsort(out_scores)[::-1]
        sorted_idx = sorted_idx[:max_decoy]
        return sorted_idx


def fast_filter_df(
    label_df: pd.DataFrame,
    fast_num: int,
    fast_model: str,
    device: torch.device,
    batch_size: int = 128,
    num_workers: int = 16,
) -> pd.DataFrame:
    """fast_filter.

    Args:
        label_df (pd.DataFrame): label_df
        fast_num (int): fast_num
        device (torch.device): device
        batch_size (int): batch_size
        num_workers (int): num_workers

    Returns:
        pd.DataFrame:
    """
    model = FastFFN.load_from_checkpoint(fast_model)
    logging.info(f"Loaded fast model with from {fast_model}")

    # Create dataset
    # Define num bins
    pred_dataset = fast_form_data.PredDataset(
        label_df,
        num_workers=num_workers,
    )
    # Define dataloaders
    collate_fn = pred_dataset.get_collate_fn()
    pred_loader = DataLoader(
        pred_dataset,
        num_workers=num_workers,
        collate_fn=collate_fn,
        shuffle=False,
        batch_size=batch_size,
    )
    # test_batch = next(iter(pred_loader))

    model.eval()
    model = model.to(device)

    out_names, out_forms, out_ions, out_scores = [], [], [], []
    with torch.no_grad():
        for batch in pred_loader:
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

    spec_to_parent = dict(label_df[["spec", "parentmass"]].values)
    output = {
        "names": out_names,
        "forms": out_forms,
        "scores": out_scores,
        "ions": out_ions,
        "parentmass": [spec_to_parent[fn] for fn in out_names],
    }

    out_df = pd.DataFrame(output)
    out_df = out_df.rename(
        columns={"names": "spec", "forms": "cand_form", "ions": "cand_ion"}
    )

    # Subset to top k
    topk_fn = lambda x: x.nlargest(fast_num, ["scores"])
    new_df = out_df.groupby(["spec"]).apply(topk_fn).reset_index(drop=True)
    return new_df
