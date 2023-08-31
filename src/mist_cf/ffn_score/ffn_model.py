import torch
import pytorch_lightning as pl
import torch.nn as nn
from torch.nn import functional as F

import mist_cf.nn_utils as nn_utils
import mist_cf.common as common


class ForwardFFN(pl.LightningModule):
    def __init__(
        self,
        ion_info: bool,
        instrument_info: bool,
        spec_info: bool,
        cls_mass_diff: bool,
        hidden_size: int,
        layers: int = 2,
        dropout: float = 0.0,
        num_spec_bins: int = 1000,
        num_valid_ion: int = 7,
        lr_decay_frac: float = 0.9,
        learning_rate: float = 7e-4,
        weight_decay: float = 0.0,
        form_encoder: str = "abs-sines",
        **kwargs,
    ):
        """_summary_

        Args:
            ion_info (bool): _description_
            instrument_info (bool): _description_
            spec_info (bool): _description_
            cls_mass_diff (bool): _description_
            hidden_size (int): _description_
            layers (int, optional): _description_. Defaults to 2.
            dropout (float, optional): _description_. Defaults to 0.0.
            num_spec_bins (int, optional): _description_. Defaults to 1000.
            num_valid_ion (int, optional): _description_. Defaults to 7.
            lr_decay_frac (float, optional): _description_. Defaults to 0.9.
            learning_rate (float, optional): _description_. Defaults to 7e-4.
            weight_decay (float, optional): _description_. Defaults to 0.0.
            form_encoder (str, optional): _description_. Defaults to "abs-sines".
        """
        super().__init__()
        self.save_hyperparameters()
        self.hidden_size = hidden_size

        self.layers = layers
        self.dropout = dropout
        self.num_spec_bins = num_spec_bins
        self.ion_info = ion_info
        self.spec_info = spec_info
        self.instrument_info = instrument_info

        self.cls_mass_diff = cls_mass_diff
        self.num_valid_ion = num_valid_ion
        self.num_valid_instrument = common.max_instr_idx

        self.learning_rate = learning_rate
        self.lr_decay_frac = lr_decay_frac
        self.weight_decay = weight_decay

        # Define network
        self.activation = nn.ReLU()

        # Define form embedder
        self.form_embedder = nn_utils.get_embedder(form_encoder)
        self.input_dim = self.form_embedder.full_dim

        # Use identity
        # Concatenate together
        if self.cls_mass_diff:
            self.input_dim += 1
        if self.spec_info:
            self.input_dim += self.num_spec_bins
        if self.ion_info:
            self.input_dim += self.num_valid_ion
        if self.instrument_info:
            self.input_dim += self.num_valid_instrument

        self.mlp = nn_utils.MLPBlocks(
            input_size=self.input_dim,
            hidden_size=self.hidden_size,
            dropout=self.dropout,
            num_layers=self.layers,
        )
        self.output_layer = nn.Linear(self.hidden_size, 1)

    def forward(
        self,
        binned_specs,
        encoded_formulae,
        parent_mass_diffs,
        ion_inputs,
        instrument_inputs,
    ):
        """predict spec."""

        # Encoded formula
        input_vecs = self.form_embedder(encoded_formulae)
        cat_vec = [input_vecs]
        if self.cls_mass_diff:
            cat_vec.append(parent_mass_diffs.float())

        if self.spec_info:
            cat_vec.append(binned_specs.float())

        if self.ion_info:
            cat_vec.append(ion_inputs.float())

        if self.instrument_info:
            cat_vec.append(instrument_inputs.float())

        input_vecs = torch.cat(cat_vec, -1)

        output = self.mlp(input_vecs)
        output = self.output_layer(output)
        return output

    def _common_step(self, batch, name="train"):
        parent_mass_diffs = batch["rel_mass_diffs"]
        spec_ars, forms = batch["spec_ars"], batch["formula_tensors"]
        ions = batch["ion_tensors"]
        instrument_inputs = batch["instrument_tensors"]

        model_outs = self.forward(
            spec_ars.float(),
            forms.float(),
            parent_mass_diffs.float(),
            ions.float(),
            instrument_inputs.float(),
        )
        ex_inds = batch["example_inds"].long()
        num_inputs = batch["num_inputs"]
        targ_inds = batch["matched"]

        packed_out = nn_utils.pad_packed_tensor(
            model_outs.squeeze(), num_inputs, -float("inf")
        )
        true_targs = nn_utils.pad_packed_tensor(
            targ_inds.squeeze().float(), num_inputs, -float("inf")
        )

        logsoftmax_probs = torch.log_softmax(packed_out, -1)
        targ_inds = true_targs.argmax(-1)
        nll_loss = F.nll_loss(logsoftmax_probs, targ_inds, reduction="none")
        output_loss = {"loss": nll_loss.mean()}
        self.log(f"{name}_loss", nll_loss.mean(), batch_size=len(targ_inds))
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