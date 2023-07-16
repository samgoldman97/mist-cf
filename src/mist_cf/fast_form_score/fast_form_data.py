import numpy as np

import torch
from torch.utils.data.dataset import Dataset
from functools import partial
from tqdm import tqdm

import mist_cf.common as common


def extract_ar(form_str):
    """extract_ar."""
    forms = [i for i in form_str.split(",") if len(i) > 0]
    # forms = [np.vstack(common.formula_to_dense_mass_norm(i)) for i in forms
    #         if len(i) > 0]
    return forms


class FormDataset(Dataset):
    """FormDataset."""

    def __init__(self, df, num_workers=0, decoys_per_pos=16, 
                 use_ray: bool = False, val_test: bool = False,
                 **kwargs):
        """__init__.

        Args:
            df:
            num_workers:
            kwargs:
        """
        self.df = df
        self.num_workers = num_workers
        self.formula_size = common.ELEMENT_DIM_MASS
        self.decoys_per_pos = decoys_per_pos
        self.val_test = val_test

        # Need to get all formulae in normal and decoys
        self.pos_str = self.df["pos"].values
        self.neg_str = self.df["neg"].values
        self.masses = self.df["mass"].values

        # Get binned spec file
        if self.num_workers == 0 or True:
            self.neg_ar = [extract_ar(i) for i in tqdm(self.neg_str)]
            self.pos_ar = [extract_ar(i) for i in tqdm(self.pos_str)]
        else:
            self.pos_ar = common.chunked_parallel(
                self.pos_str,
                extract_ar,
                chunks=100,
                max_cpu=self.num_workers,
                timeout=4000,
                max_retries=3,
                use_ray=use_ray,
            )
            self.neg_ar = common.chunked_parallel(
                self.neg_str,
                extract_ar,
                chunks=100,
                max_cpu=self.num_workers,
                timeout=4000,
                max_retries=3,
                use_ray=use_ray
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        mass = self.masses[idx]
        pos = self.pos_ar[idx]
        neg = self.neg_ar[idx]

        pos = np.random.choice(pos)
        num_negs = len(neg)
        sample_num = min(num_negs, self.decoys_per_pos)
        if self.val_test:
            neg = neg[:sample_num]
        else:
            neg = np.random.choice(neg, sample_num,
                                   replace=False)

        pos_ar = [common.formula_to_dense(pos)]
        neg_ar = [common.formula_to_dense(i) for i in neg]

        ars = pos_ar + neg_ar
        labels = [1] + [0] * len(neg)

        outdict = {"x": ars, "y": labels}
        return outdict

    @classmethod
    def get_collate_fn(cls):
        return FormDataset.collate_fn

    @staticmethod
    def collate_fn(input_list):
        """collate_fn"""

        # Stack all decoys
        x_vals = torch.FloatTensor([i for j in input_list for i in j["x"]])
        y_vals = torch.LongTensor([i for j in input_list for i in j["y"]])

        return_dict = {"x": x_vals, "y": y_vals}
        return return_dict


class PredDataset(Dataset):
    """PredDataset."""

    def __init__(self, df, num_workers=0, **kwargs):
        """__init__.

        Args:
            df:
            num_workers:
            decoys_per_pos:
            kwargs:
        """
        self.df = df
        self.num_workers = num_workers
        self.formula_size = common.ELEMENT_DIM_MASS

        self.cand_forms = []

        # Get formulae
        self.cand_forms = self.df["cand_form"].values
        self.cand_ions = self.df["cand_ion"].values
        
        if self.num_workers == 0:
            self.embedded_forms = [
                common.formula_to_dense(i)
                for i in tqdm(self.cand_forms)
            ]
        else:
            self.embedded_forms = common.chunked_parallel(
                self.cand_forms,
                common.formula_to_dense,
                chunks=100,
                max_cpu=self.num_workers,
                timeout=4000,
                max_retries=3,
            )
        self.names = self.df["spec"].values

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx: int):
        spec_name = self.names[idx]
        ar = self.embedded_forms[idx]
        form = self.cand_forms[idx]
        ion = self.cand_ions[idx]


        # Create meta
        outdict = {"spec": spec_name, "form": form, "x": ar,
                   "ion": ion, }
        return outdict

    @classmethod
    def get_collate_fn(cls):
        return PredDataset.collate_fn

    @staticmethod
    def collate_fn(input_list):
        """collate_fn"""
        # Stack all decoys
        str_forms = np.array([i['form'] for i in input_list])
        x = torch.FloatTensor(np.array([i['x'] for i in input_list]))
        names = np.array([i['spec'] for i in input_list])
        ions = np.array([i['ion'] for i in input_list])
        return_dict = {"names": names,
                       "str_forms": str_forms,
                       "x": x,
                       "ions": ions}
        return return_dict
