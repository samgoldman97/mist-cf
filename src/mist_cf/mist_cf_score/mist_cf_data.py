"""Mist CF data
"""

import logging
from pathlib import Path

import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import json
from collections import defaultdict

import mist_cf.common as common

cat_types = {"frags": 0, "cls": 1}
num_types = len(cat_types)
cls_type = cat_types.get("cls")


def double_unroll(input_list, key, cast_fn=lambda x: x):
    return [cast_fn(j) for i in input_list for j in i[key]]


def pad_stack(tensors, padding_amts):
    """pad_stack."""
    num_dims = len(tensors[0].shape)
    zeros = (num_dims - 1) * 2 + 1
    tensors = [
        torch.nn.functional.pad(i, (*([0] * zeros), pad_len))
        for i, pad_len in zip(tensors, padding_amts)
    ]
    tensors = torch.stack(tensors, dim=0)
    return tensors


to_float = lambda x: torch.FloatTensor(x)
to_bool = lambda x: torch.BoolTensor(x)
to_long = lambda x: torch.LongTensor(x)


class JsonExtractor:
    """JsonExtractor.

    Abstract parsing to inject as dependen;cy

    """

    def __init__(self, max_subpeak):
        self.max_subpeak = max_subpeak
        self.ion_mat = np.eye(len(common.ION_LST))
        self.instrument_mat = np.eye(len(common.instrument_to_idx) + 1)

    def get_ion_embed(self, ion):
        return self.ion_mat[common.get_ion_idx(ion)]

    def get_instrument_embed(self, instrument):
        return self.instrument_mat[common.get_instr_idx(instrument)]

    def extract_json(
        self, json_obj, form, ion, parentmass, instrument, ablate_cls_error=False
    ):

        cand_tbl = json_obj["cand_tbl"]
        assert json_obj["cand_ion"] == ion

        # embed true_formula and true decoy ROOT
        embed_form = common.formula_to_dense(form)
        embed_ion = self.get_ion_embed(ion)

        # calculate ppm for parentmass
        if ablate_cls_error:
            cls_ppm = 0
        else:
            cls_mass_diff = common.get_cls_mass_diff(
                parentmass, form=form, ion=ion, corr_electrons=True
            )
            cls_ppm = common.clipped_ppm_single_norm(cls_mass_diff, parentmass)
        if cand_tbl is None:
            form_vecs = []
            ion_vecs = []
            peak_types = []
            frag_intens = []
            rel_mass_diffs = []
        else:
            # Embed peak subtypes
            form_vecs = list(
                map(common.formula_to_dense, cand_tbl["formula"][: self.max_subpeak])
            )
            ion_vecs = list(
                map(self.get_ion_embed, cand_tbl["ions"][: self.max_subpeak])
            )
            peak_types = [cat_types.get("frags")] * len(form_vecs)
            frag_intens = cand_tbl["ms2_inten"][: self.max_subpeak]

            rel_mass_diffs = np.array(cand_tbl["mass_diff"][: self.max_subpeak])
            rel_mass_diffs = common.norm_mass_diff_ppm(rel_mass_diffs).tolist()

        # Add root to this
        form_vecs = [embed_form] + form_vecs
        ion_vecs = [embed_ion] + ion_vecs
        peak_types = [cls_type] + peak_types
        frag_intens = [1] + frag_intens
        rel_mass_diffs = [cls_ppm] + rel_mass_diffs
        # Add intrument encoding
        embed_instrument = self.get_instrument_embed(instrument)
        instrument_vecs = [embed_instrument] * len(form_vecs)

        return {
            "form_vecs": form_vecs,
            "ion_vecs": ion_vecs,
            "instrument_vecs": instrument_vecs,
            "peak_types": peak_types,
            "frag_intens": frag_intens,
            "rel_mass_diffs": rel_mass_diffs,
        }


class FormDataset(Dataset):
    """FormDataset."""

    def __init__(
        self,
        df,
        data_dir,
        subform_dir,
        max_subpeak,
        max_decoy,
        num_workers=0,
        val_test: bool = False,
        ablate_cls_error: bool = False,
        **kwargs,
    ):
        """__init__.

        Args:
            df:
            data_dir:
            subform_dir:
            num_workers:
            val_test (bool):
            kwargs:
        """
        self.df = df.fillna("").reset_index()

        # Filter out all entries where "formula == []"
        self.df = self.df[self.df["decoy_formulae"] != "[]"].reset_index(drop=True)

        self.num_workers = num_workers
        self.data_dir = data_dir
        self.subform_dir = Path(subform_dir)
        self.max_decoy = max_decoy
        self.val_test = val_test
        self.ablate_cls_error = ablate_cls_error
        self.json_extractor = JsonExtractor(max_subpeak=max_subpeak)

        self.spec_names = self.df["spec"].values
        self.instruments = self.df["instrument"].values
        self.parentmasses = self.df["parentmass"].values
        self.json_paths = [subform_dir / f"{i}.json" for i in self.spec_names]
        self.true_formulae = self.df["formula"].values
        self.true_ions = self.df["ionization"].values

        ion_decoys = self.df["decoy_ions"].values
        formulae_decoys = self.df["decoy_formulae"].values

        self.decoy_ions = [i.split(",") if len(i) > 0 else [] for i in ion_decoys]
        self.decoy_formulae = [
            i.split(",") if len(i) > 0 else [] for i in formulae_decoys
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        """__getitem__.

        Args:
            idx (int): idx
        """

        name = self.spec_names[idx]
        instrument = self.instruments[idx]
        # Vectorize the full formula (root, cls token)
        decoy_formulae = np.array(self.decoy_formulae[idx])
        decoy_ions = np.array(self.decoy_ions[idx])

        true_formula = self.true_formulae[idx]
        true_ion = self.true_ions[idx]

        parentmass = self.parentmasses[idx]
        json_file_path = self.json_paths[idx]

        # downsample decoys
        full_decoy_idx_list = list(range(len(decoy_formulae)))
        num_decoys = len(full_decoy_idx_list)
        num_sample = min(self.max_decoy, num_decoys)
        if self.val_test:
            sample_inds = np.arange(0, num_sample)
        else:
            sample_inds = np.random.choice(num_decoys, num_sample, replace=False)

        decoy_formulae = decoy_formulae[sample_inds].tolist()
        decoy_ions = decoy_ions[sample_inds].tolist()

        # get subformulae assignment tbls and decoy ions
        with open(json_file_path, "r") as openfile:
            json_object = json.load(openfile)

        true_json = self.json_extractor.extract_json(
            json_obj=json_object[true_formula],
            form=true_formula,
            ion=true_ion,
            parentmass=parentmass,
            instrument=instrument,
            ablate_cls_error=self.ablate_cls_error,
        )

        (
            peak_types_list,
            form_vecs_list,
            ion_vecs_list,
            instrument_vecs_list,
            intens_list,
            diffs_list,
        ) = (
            [true_json["peak_types"]],
            [true_json["form_vecs"]],
            [true_json["ion_vecs"]],
            [true_json["instrument_vecs"]],
            [true_json["frag_intens"]],
            [true_json["rel_mass_diffs"]],
        )

        # Extract decoys
        for decoy_formula, decoy_ion in zip(decoy_formulae, decoy_ions):
            if decoy_formula not in json_object:
                logging.info(f"Could not find {decoy_formula} for {name}")
                continue
            decoy_tbl = json_object[decoy_formula]
            decoy_json = self.json_extractor.extract_json(
                json_obj=decoy_tbl,
                form=decoy_formula,
                ion=decoy_ion,
                parentmass=parentmass,
                instrument=instrument,
                ablate_cls_error=self.ablate_cls_error,
            )

            peak_types_list.append(decoy_json["peak_types"])
            form_vecs_list.append(decoy_json["form_vecs"])
            ion_vecs_list.append(decoy_json["ion_vecs"])
            instrument_vecs_list.append(decoy_json["instrument_vecs"])
            intens_list.append(decoy_json["frag_intens"])
            diffs_list.append(decoy_json["rel_mass_diffs"])

        # Create meta
        cand_formulae = [true_formula] + decoy_formulae
        cand_ions = [true_ion] + decoy_ions
        cand_instruments = [instrument] * len(cand_ions)
        matched = [True] + [False] * len(decoy_formulae)
        outdict = {
            "name": name,
            "matched": matched,
            "formulae": cand_formulae,
            "ions": cand_ions,
            "instruments": cand_instruments,
            "num_inputs": len(matched),
            # Define input components
            "peak_type_list": peak_types_list,
            "form_vec_list": form_vecs_list,
            "ion_vec_list": ion_vecs_list,
            "instrument_vec_list": instrument_vecs_list,
            "frag_intens_list": intens_list,
            "rel_mass_diff_list": diffs_list,
        }
        return outdict

    @classmethod
    def get_collate_fn(cls):
        return FormDataset.collate_fn

    @staticmethod
    def collate_fn(input_list):
        """collate_fn"""

        # Stack all decoys
        names = [j["name"] for j in input_list]
        str_forms = [j for i in input_list for j in i["formulae"]]
        str_ions = [j for i in input_list for j in i["ions"]]
        str_instruments = [j for i in input_list for j in i["instruments"]]
        num_inputs = [i["num_inputs"] for i in input_list]

        matched = double_unroll(input_list, "matched")
        example_inds = [ind for ind, num in enumerate(num_inputs) for _ in range(num)]

        matched = to_bool(matched)
        num_inputs = to_long(num_inputs)
        example_inds = to_long(example_inds)

        # Extract and pad all entries and examples
        inten_tensors = double_unroll(input_list, "frag_intens_list", to_float)
        rel_diff_tensors = double_unroll(input_list, "rel_mass_diff_list", to_float)
        type_tensors = double_unroll(input_list, "peak_type_list", to_float)
        peak_form_tensors = double_unroll(input_list, "form_vec_list", to_float)
        peak_ion_tensors = double_unroll(input_list, "ion_vec_list", to_float)
        peak_instrument_tensors = double_unroll(
            input_list, "instrument_vec_list", to_float
        )

        peak_form_lens = np.array([i.shape[0] for i in peak_form_tensors])
        max_len = np.max(peak_form_lens)
        padding = max_len - peak_form_lens

        type_tensors = pad_stack(type_tensors, padding).long()
        inten_tensors = pad_stack(inten_tensors, padding).float()
        rel_diff_tensors = pad_stack(rel_diff_tensors, padding).float()
        peak_form_tensors = pad_stack(peak_form_tensors, padding).float()
        peak_ion_tensors = pad_stack(peak_ion_tensors, padding).float()
        peak_instrument_tensors = pad_stack(peak_instrument_tensors, padding).float()
        num_peaks = torch.from_numpy(peak_form_lens).long()

        return_dict = {
            "names": names,
            "example_inds": example_inds,
            "formulas": str_forms,
            "ions": str_ions,
            "instruments": str_instruments,
            "num_inputs": num_inputs,
            "matched": matched,
            # Actual inputs
            "types": type_tensors,
            "form_vec": peak_form_tensors,
            "ion_vec": peak_ion_tensors,
            "instrument_vec": peak_instrument_tensors,
            "intens": inten_tensors,
            "rel_mass_diffs": rel_diff_tensors,
            # "abs_mass_diffs": abs_mass_diff_tensors,
            "num_peaks": num_peaks,
        }
        return return_dict


class PredDataset(Dataset):
    """PredDataset."""

    def __init__(
        self,
        df,
        subform_dir,
        max_subpeak,
        num_workers=0,
        ablate_cls_error: bool = False,
        **kwargs,
    ):
        """__init__ _summary_

        Args:
            df (_type_): _description_
            subform_dir (_type_): _description_
            max_subpeak (_type_): _description_
            num_workers (int, optional): _description_. Defaults to 0.
            ablate_cls_error (bool, optional): _description_. Defaults to False.
        """    
        self.df = df
        self.num_workers = num_workers
        self.subform_dir = Path(subform_dir)
        self.max_subpeak = max_subpeak
        self.ablate_cls_error = (ablate_cls_error,)
        self.json_extractor = JsonExtractor(max_subpeak=max_subpeak)

        self.names = self.df["spec"].values
        self.cand_forms = self.df["cand_form"].values
        self.cand_ions = self.df["cand_ion"].values
        self.cand_instruments = self.df["instrument"].values
        self.parentmasses = self.df["parentmass"].values

        name_to_idxs = defaultdict(lambda: [])
        for ind, j in enumerate(self.names):
            name_to_idxs[j].append(ind)

        # Define a new dict that maps a new index
        self.new_ind_to_old_inds = {}
        self.new_ind_to_name = {}
        cur_new_ind = 0
        for name, old_inds in name_to_idxs.items():
            # Define 256 as the number of items to include max at each idx
            old_ind_batches = common.batches(old_inds, 256)
            for old_ind_batch in old_ind_batches:
                self.new_ind_to_old_inds[cur_new_ind] = old_ind_batch
                self.new_ind_to_name[cur_new_ind] = name
                cur_new_ind += 1
        self.len = len(self.new_ind_to_old_inds)

    def __len__(self):
        return self.len

    def __getitem__(self, idx: int):
        spec_name = self.new_ind_to_name[idx]

        json_file_path = f"{self.subform_dir}/{spec_name}.json"
        with open(json_file_path, "r") as openfile:
            json_object = json.load(openfile)

        out_dict = defaultdict(lambda: [])
        for list_idx in self.new_ind_to_old_inds[idx]:
            formula = self.cand_forms[list_idx]
            parentmass = self.parentmasses[list_idx]
            ion = self.cand_ions[list_idx]
            instrument = self.cand_instruments[list_idx]
            decoy_tbl = json_object[formula]
            decoy_json = self.json_extractor.extract_json(
                json_obj=decoy_tbl,
                form=formula,
                ion=ion,
                parentmass=parentmass,
                instrument=instrument,
                ablate_cls_error=self.ablate_cls_error,
            )
            out_dict["name"].append(spec_name)
            out_dict["formula"].append(formula)
            out_dict["ion"].append(ion)
            out_dict["instrument"].append(instrument)
            out_dict["parentmass"].append(parentmass)
            out_dict["peak_type"].append(decoy_json["peak_types"])
            out_dict["form_vecs"].append(decoy_json["form_vecs"])
            out_dict["ion_vecs"].append(decoy_json["ion_vecs"])
            out_dict["instrument_vecs"].append(decoy_json["instrument_vecs"])
            out_dict["frag_intens"].append(decoy_json["frag_intens"])
            out_dict["rel_mass_diffs"].append(decoy_json["rel_mass_diffs"])
        return out_dict

    @classmethod
    def get_collate_fn(cls):
        return PredDataset.collate_fn

    @staticmethod
    def collate_fn(input_list):
        """collate_fn"""

        # Stack all candidates
        names = double_unroll(input_list, "name")
        str_forms = double_unroll(input_list, "formula")
        str_ions = double_unroll(input_list, "ion")
        str_instruments = double_unroll(input_list, "instrument")
        parentmasses = double_unroll(input_list, "parentmass")
        inten_tensors = double_unroll(input_list, "frag_intens", to_float)
        rel_diff_tensors = double_unroll(input_list, "rel_mass_diffs", to_float)
        type_tensors = double_unroll(input_list, "peak_type", to_long)
        peak_form_tensors = double_unroll(input_list, "form_vecs", to_float)
        peak_ion_tensors = double_unroll(input_list, "ion_vecs", to_float)
        peak_instrument_tensors = double_unroll(input_list, "instrument_vecs", to_float)

        peak_form_lens = np.array([i.shape[0] for i in peak_form_tensors])
        max_len = np.max(peak_form_lens)
        padding = max_len - peak_form_lens

        type_tensors = pad_stack(type_tensors, padding).long()
        inten_tensors = pad_stack(inten_tensors, padding).float()
        rel_diff_tensors = pad_stack(rel_diff_tensors, padding).float()
        peak_form_tensors = pad_stack(peak_form_tensors, padding).float()
        peak_ion_tensors = pad_stack(peak_ion_tensors, padding).float()
        peak_instrument_tensors = pad_stack(peak_instrument_tensors, padding).float()
        num_peaks = torch.from_numpy(peak_form_lens).long()

        return_dict = {
            "names": names,
            "str_forms": str_forms,
            "str_ions": str_ions,
            "str_instruments": str_instruments,
            "parentmasses": parentmasses,
            # Actual inputs
            "types": type_tensors,
            "form_vec": peak_form_tensors,
            "ion_vec": peak_ion_tensors,
            "instrument_vec": peak_instrument_tensors,
            "intens": inten_tensors,
            "rel_mass_diffs": rel_diff_tensors,
            "num_peaks": num_peaks,
        }
        return return_dict