import torch
import pytorch_lightning as pl
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

import mist_cf.nn_utils as nn_utils
import mist_cf.mist_cf_score.mist_cf_data as mist_cf_data
import mist_cf.common as common


class MistNet(pl.LightningModule):
    def __init__(
        self,
        hidden_size: int,
        layers: int = 2,
        dropout: float = 0.0,
        weight_decay: float = 0.0,
        lr_decay_frac: float = 0.9,
        learning_rate: float = 7e-4,
        ion_info: bool = False,
        instrument_info: bool = False,
        cls_mass_diff: bool = False,
        form_encoder: str = "abs-sines",
        max_subpeak: int = 10,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.hidden_size = hidden_size
        self.max_subpeak = max_subpeak

        self.layers = layers
        self.dropout = dropout
        self.num_layers = layers

        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.lr_decay_frac = lr_decay_frac
        self.ion_info = ion_info
        self.instrument_info = instrument_info
        self.cls_mass_diff = cls_mass_diff

        # Define network
        self.activation = nn.ReLU()

        # Define model
        self.xformer = FormulaTransformer(
            form_encoder=form_encoder,
            hidden_size=self.hidden_size,
            peak_attn_layers=self.num_layers,
            spectra_dropout=dropout,
            pairwise_featurization=True,
            set_pooling="cls",
            ion_info=ion_info,
            instrument_info=instrument_info,
        )
        self.output_layer = nn.Linear(self.hidden_size, 1)

    def forward(self, num_peaks, peak_types, form_vec, ion_vec, instrument_vec,
                intens, rel_mass_diffs):
        """predict spec."""
        output = self.xformer(
            num_peaks, peak_types, form_vec, ion_vec, instrument_vec,
            intens, rel_mass_diffs, return_aux=False
        )
        output = self.output_layer(output)
        return output


    def _common_step(self, batch, name="train"):
        """ _common_step """

        peak_types, form_vec, ion_vec, instrument_vec, intens, rel_mass_diffs, num_peaks = (
            batch["types"],
            batch["form_vec"],
            batch['ion_vec'],
            batch['instrument_vec'],
            batch["intens"],
            batch["rel_mass_diffs"],
            batch["num_peaks"],

        )

        model_outs = self.forward(
            num_peaks, peak_types, form_vec, ion_vec, instrument_vec, 
            intens, rel_mass_diffs
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


class FormulaTransformer(nn.Module):
    """FormulaTransformer"""

    def __init__(
        self,
        form_encoder: str,
        hidden_size: int,
        peak_attn_layers: int,
        set_pooling: str = "cls",
        spectra_dropout: float = 0.1,
        additive_attn: bool = False,
        pairwise_featurization: bool = False,
        num_heads: int = 8,
        ion_info: bool = False,
        instrument_info: bool = False,
        num_valid_ion: int = 7,
        num_valid_instrument: int = common.max_instr_idx,
        **kwargs
    ):
        """__init__.

        Args:
            form_encoder (str): form_encoder
            hidden_size (int): hidden_size
            peak_attn_layers (int): peak_attn_layers
            set_pooling (str): set_pooling
            spectra_dropout (float): spectra_dropout
            additive_attn (bool): additive_attn
            pairwise_featurization (bool): pairwise_featurization
            num_heads (int): num_heads
            ion_info (bool): ion_info
            num_valid_ion (int): Num valid ions
            kwargs:
        """
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.attn_heads = num_heads
        self.dim_feedforward = self.hidden_size * 4
        self.spectra_dropout = spectra_dropout
        self.set_pooling = set_pooling
        self.dropout = nn.Dropout(spectra_dropout)
        self.num_valid_ion = num_valid_ion
        self.num_valid_instrument = num_valid_instrument

        self.form_encoder = nn_utils.get_embedder(form_encoder)

        self.num_types = mist_cf_data.num_types
        self.cls_type = mist_cf_data.cls_type
        self.additive_attn = additive_attn
        self.pairwise_featurization = pairwise_featurization

        # Define dense encoders and root formula encoder
        self.ion_info = ion_info
        self.instrument_info = instrument_info
        self.formula_dim_base = self.form_encoder.full_dim

        # Also use diffs 
        self.formula_dim = self.formula_dim_base * 2 

        # Add in concatenate features of relative mass diff, intensity, peak types, num peaks
        self.formula_dim += 4
        
        # Optional feats of instrument info and ion info
        if self.instrument_info:
            self.formula_dim+= self.num_valid_instrument
        if self.ion_info:
            self.formula_dim += self.num_valid_ion

        # Use the same encoder for everything (it's just easier)
        self.formula_encoder = nn_utils.MLPBlocks(input_size=self.formula_dim,
                                                  hidden_size=self.hidden_size,
                                                  dropout=spectra_dropout,
                                                  num_layers=1) 

        self.pairwise_featurizer = None
        if self.pairwise_featurization:
            self.pairwise_featurizer = nn_utils.MLPBlocks(input_size=self.formula_dim_base,
                                                          hidden_size=self.hidden_size,
                                                          dropout=spectra_dropout,
                                                          num_layers=1)


        # Multihead attention block with residuals
        peak_attn_layer = nn_utils.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=self.attn_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.spectra_dropout,
            additive_attn=self.additive_attn,
            pairwise_featurization=pairwise_featurization,
        )
        self.peak_attn_layers = nn_utils.get_clones(peak_attn_layer, peak_attn_layers)

    def forward(
        self, num_peaks, peak_types, form_vec, ion_vec, instrument_vec,
        intens, rel_mass_diffs, return_aux=False
    ):
        """forward."""
        # Step 1: Create embeddings
        device = num_peaks.device
        batch_size, peak_dim, _form_dim  = form_vec.shape
        

        cls_type = (peak_types == self.cls_type)
        cls_tokens = form_vec[cls_type]
        diff_vec = cls_tokens[:, None, :] - form_vec

        diff_embedded = self.form_encoder(diff_vec)
        form_embedded = self.form_encoder(form_vec)
        cat_input = [form_embedded, diff_embedded, cls_type[:, :, None].float()]

        # Step 1 is to encode the formula, difference, ion, and peak type information
        if self.ion_info:
            cat_input.append(ion_vec)

        if self.instrument_info:
            cat_input.append(instrument_vec)

        inten_tensor = intens[:, :, None]
        rel_mass_diff_tensor = rel_mass_diffs[:, :, None]
        num_peak_feat = num_peaks[:, None, None].expand(
            batch_size, peak_dim, 1
        ) / 10

        cat_input.extend([inten_tensor, num_peak_feat, rel_mass_diff_tensor])
        input_vec = torch.cat(cat_input, -1)

        # Get formula encoders for each class type
        peak_tensor = self.formula_encoder(input_vec)

        # Step 3: Run transformer
        # B x Np x d -> Np x B x d
        peak_tensor = peak_tensor.transpose(0, 1)
        # Mask before summing
        peak_dim = peak_tensor.shape[0]
        peaks_aranged = torch.arange(peak_dim).to(device)

        # batch x num peaks
        attn_mask = ~(peaks_aranged[None, :] < num_peaks[:, None])
        pairwise_features = None
        if self.pairwise_featurization:
            form_diffs = form_vec[:, None, :, :] - form_vec[:, :, None, :]

            # Make sure to _only_ consider subset fragments, rather than only
            # partial additions/subtractions from across the tree branches
            # Comment out to consider _any_ loss embedding with new parameters
            same_sign = torch.all(form_diffs >= 0, -1) | torch.all(form_diffs <= 0, -1)
            form_diffs[~same_sign].fill_(0)
            form_diffs = torch.abs(form_diffs)

            encoded_diffs = self.form_encoder(form_diffs)
            pairwise_features = self.pairwise_featurizer(encoded_diffs)

        # Np x B x d
        aux_output = {}
        for peak_attn_layer in self.peak_attn_layers:
            peak_tensor, pairwise_features = peak_attn_layer(
                peak_tensor,
                pairwise_features=pairwise_features,
                src_key_padding_mask=attn_mask,
            )

        # Step 4: Pool output
        output, peak_tensor = self._pool_out(
            peak_tensor,
            inten_tensor,
            rel_mass_diff_tensor,
            peak_types,
            attn_mask,
            batch_size,
        )
        aux_output["peak_tensor"] = peak_tensor.transpose(0, 1)

        # Now convert into output dim
        if return_aux:
            output = (output, aux_output)

        return output

    def _pool_out(
        self,
        peak_tensor,
        inten_tensor,
        rel_mass_diff_tensor,
        peak_types,
        attn_mask,
        batch_dim,
    ):
        """_pool_out.

        pool the output of the network

        Return:
            (output (B x H), peak_tensor : L x B x H)

        """
        EPS = 1e-22

        #  Np x B x d
        zero_mask = attn_mask[:, :, None].repeat(1, 1, self.hidden_size).transpose(0, 1)
        # Mask over NaN
        peak_tensor[zero_mask] = 0
        if self.set_pooling == "intensity":
            inten_tensor = inten_tensor.reshape(batch_dim, -1)
            intensities_sum = inten_tensor.sum(1).reshape(-1, 1) + EPS
            inten_tensor = inten_tensor / intensities_sum
            pool_factor = inten_tensor * ~attn_mask
        elif self.set_pooling == "mean":
            inten_tensor = inten_tensor.reshape(batch_dim, -1)
            pool_factor = torch.clone(inten_tensor).fill_(1)
            pool_factor = pool_factor * ~attn_mask
            # Replace all zeros with 1
            pool_factor[pool_factor == 0] = 1
            pool_factor = pool_factor / pool_factor.sum(1).reshape(-1, 1)
        elif self.set_pooling == "rel_mass_diff":
            rel_mass_diff_tensor = rel_mass_diff_tensor.reshape(batch_dim, -1)
            pool_factor = torch.clone(rel_mass_diff_tensor).fill_(1)
            pool_factor = pool_factor * ~attn_mask
            # Replace all zeros with 1
            pool_factor[pool_factor == 0] = 1
            pool_factor = pool_factor / pool_factor.sum(1).reshape(-1, 1)
        elif self.set_pooling == "cls":
            pool_factor = (peak_types == self.cls_type).float()
        else:
            raise NotImplementedError()

        # Weighted average over peak intensities
        output = torch.einsum("nbd,bn->bd", peak_tensor, pool_factor)
        return output, peak_tensor
