import numpy as np
import torch
import pytorch_lightning as pl
import torch.nn as nn
from torch.nn import functional as F

import mist_cf.common as common
import mist_cf.nn_utils as nn_utils


class XformerModel(pl.LightningModule):
    def __init__(
        self,
        ion_info: bool,
        cls_mass_diff: bool,
        instrument_info: bool,
        hidden_size: int,
        layers: int = 2,
        dropout: float = 0.0,
        num_valid_ion: int = 7,
        lr_decay_frac: float = 0.9,
        learning_rate: float = 7e-4,
        weight_decay: float = 0.0,
        form_encoder: str = "abs-sines",
        **kwargs
    ):
        """_summary_

        Args:
            ion_info (bool): _description_
            cls_mass_diff (bool): _description_
            instrument_info (bool): _description_
            hidden_size (int): _description_
            layers (int, optional): _description_. Defaults to 2.
            dropout (float, optional): _description_. Defaults to 0.0.
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
        self.ion_info = ion_info
        self.instrument_info = instrument_info
        self.cls_mass_diff = cls_mass_diff
        self.num_valid_ion = num_valid_ion
        self.num_valid_instrument = common.max_instr_idx

        self.learning_rate = learning_rate
        self.lr_decay_frac = lr_decay_frac
        self.weight_decay = weight_decay

        # Define network
        self.activation = nn.ReLU()

        # Define form and freq embedder
        self.form_embedder = nn_utils.get_embedder(form_encoder)
        self.freq_embedder = FourierEmbedder(d=hidden_size)
        
        # Add 1 for intensity
        self.input_dim = self.form_embedder.full_dim + self.hidden_size + 1

        # Use identity
        # Concatenate together
        if self.cls_mass_diff:
            self.input_dim += 1

        if self.ion_info:
            self.input_dim += self.num_valid_ion

        if self.instrument_info:
            self.input_dim += self.num_valid_instrument

        # Convert the input concatenations into linear vectors at each position
        self.input_compress = nn.Linear(self.input_dim, self.hidden_size)

        peak_attn_layer = nn_utils.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=8,
            dim_feedforward=self.hidden_size * 4,
            dropout=self.dropout,
            additive_attn=False,
            pairwise_featurization=False,
        )
        self.peak_attn_layers = nn_utils.get_clones(peak_attn_layer, self.layers)
        self.output_layer = nn.Linear(self.hidden_size, 1)


    def forward(self, spec_ars, num_peaks, encoded_formulae, parent_mass_diffs,
                ion_inputs, instrument_inputs):
        """predict spec."""

        # Encoded formula
        device = spec_ars.device
        mz_vec = spec_ars[:, :, 0]
        inten_vec = spec_ars[:, :, 1]
        embedded_mz = self.freq_embedder(mz_vec)

        # Concatenate the other necessary feaatures
        input_vecs = self.form_embedder(encoded_formulae)
        cat_vec = [input_vecs]

        if self.cls_mass_diff:
            cat_vec.append(parent_mass_diffs.float())

        if self.ion_info:
            cat_vec.append(ion_inputs.float())

        if self.instrument_info:
            cat_vec.append(instrument_inputs.float())
        cat_vec = torch.cat(cat_vec, -1)
        
        # Expand cat vec to be the same size as the embedded mz in middle dim
        # Then concat
        cat_vec = cat_vec[:, None, :].expand(-1, embedded_mz.shape[1], -1)
        embedded_mz = torch.cat([embedded_mz, cat_vec, inten_vec[:, :, None]], -1)
        peak_tensor = self.input_compress(embedded_mz)

        peak_dim = peak_tensor.shape[1]
        peaks_aranged = torch.arange(peak_dim).to(device)

        # batch x num peaks
        attn_mask = ~(peaks_aranged[None, :] < num_peaks[:, None])

        # Transpose to peaks x batch x features
        peak_tensor = peak_tensor.transpose(0, 1)
        for peak_attn_layer in self.peak_attn_layers:
            peak_tensor, pairwise_features = peak_attn_layer(
                peak_tensor,
                src_key_padding_mask=attn_mask,
            )

        peak_tensor = peak_tensor.transpose(0, 1)

        # Get only the class token
        h0 = peak_tensor[:, 0, :]

        output= self.output_layer(h0)
        return output

    def _common_step(self, batch, name="train"):
        parent_mass_diffs = batch["rel_mass_diffs"]
        spec_ars, forms = batch["spec_ars"], batch["formula_tensors"]
        ions = batch['ion_tensors']
        num_peaks = batch["num_peaks"]
        instrument_inputs = batch["instrument_tensors"]
        model_outs = self.forward(
            spec_ars.float(), 
            num_peaks.long(),
            forms.float(), parent_mass_diffs.float(),
            ions.float(),  instrument_inputs.float()
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
            self.parameters(), lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        scheduler = nn_utils.build_lr_scheduler(optimizer,
                                                lr_decay_rate=self.lr_decay_frac)
        ret = {
            "optimizer": optimizer,
            "lr_scheduler": {
                   "scheduler": scheduler,
                   "frequency": 1,
                   "interval": "step"
            }
        }
        return ret

class FourierEmbedder(torch.nn.Module):
    """ Embed a set of mz float values using frequencies"""

    def __init__(self, d=512, logmin=-2.5, logmax=3.3, **kwargs):
        super().__init__()
        self.d = d
        self.logmin = logmin
        self.logmax = logmax

        lambda_min = np.power(10, -logmin)
        lambda_max = np.power(10, logmax)
        index = torch.arange(np.ceil(d / 2))
        exp = torch.pow(lambda_max/lambda_min, (2 * index) / (d - 2))
        freqs = 2 * np.pi * (lambda_min * exp) ** (-1)

        self.freqs = nn.Parameter(freqs, requires_grad=False)

        # Turn off requires grad for freqs
        self.freqs.requires_grad = False

    def forward(self, mz: torch.FloatTensor):
        """ forward

        Args:
            mz: FloatTensor of shape (batch_size, mz values)

        Returns:
            FloatTensor of shape (batch_size, peak len, mz )
        """
        freq_input = torch.einsum("bi,j->bij", mz, self.freqs)
        embedded = torch.cat(
            [torch.sin(freq_input), torch.cos(freq_input)], -1)
        embedded = embedded[:, :, :self.d]
        return embedded