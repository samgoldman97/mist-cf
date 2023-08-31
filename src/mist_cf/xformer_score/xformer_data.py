import numpy as np

import torch
from torch.utils.data.dataset import Dataset
from functools import partial
from tqdm import tqdm
from collections import defaultdict

import mist_cf.common as common


class InstrEmbedder:
    def __init__(self):
        self.instrument_mat = np.eye(len(common.instrument_to_idx) + 1)

    def embed_instr(self, instrument):
        return self.instrument_mat[common.get_instr_idx(instrument)]


def max_peaks(x, max_p=100):
    # x is a N x 2 array
    # Sort and subset to  max_p
    x = x[x[:, 1].argsort()[::-1]]
    return x[:max_p, :]


class XformerDataset(Dataset):
    """_summary_

    Args:
        Dataset (_type_): _description_
    """

    def __init__(
        self,
        df,
        data_dir,
        max_decoy=256,
        num_workers=0,
        val_test: bool = False,
        use_ray: bool = False,
        **kwargs,
    ):
        self.df = df.fillna("").reset_index()
        self.num_workers = num_workers
        self.max_decoy = max_decoy
        self.data_dir = data_dir
        self.val_test = val_test
        self.instr_embedder = InstrEmbedder()
        self.ion_mat = np.eye(len(common.ION_LST))

        # embedded decoy forms
        # add the decoy by ionization
        # add one-hot encoding of ionization information
        self.decoy_forms = []
        self.decoy_form_vecs = []
        self.decoy_ion_vecs = []
        self.decoy_ppms = []
        self.decoy_instr_vecs = []
        self.true_form_vecs = []
        self.true_instr_vecs = []
        self.true_ppms = []
        self.true_ion_vecs = []
        self.true_forms = self.df["formula"].values
        self.true_instrs = self.df["instrument"].values

        ion_decoys = self.df["decoy_ions"].values
        formulae_decoys = self.df["decoy_formulae"].values
        specs = self.df["spec"].values
        parentmasses = self.df["parentmass"].values
        true_formulae = self.df["formula"].values
        true_ions = self.df["ionization"].values
        true_instrs = self.df["instrument"].values

        for spec, true_form, true_ion, ions, formulae, parentmass, instrument in tqdm(
            zip(
                specs,
                true_formulae,
                true_ions,
                ion_decoys,
                formulae_decoys,
                parentmasses,
                true_instrs,
            )
        ):

            true_form_vec = common.formula_to_dense(true_form)
            self.true_form_vecs.append(true_form_vec)

            true_instr_vec = self.instr_embedder.embed_instr(instrument)
            self.true_instr_vecs.append(true_instr_vec)

            ion_encoding = self.ion_mat[common.get_ion_idx(true_ion)]
            self.true_ion_vecs.append(ion_encoding)

            cls_mass_diff = common.get_cls_mass_diff(
                parentmass, form=true_form, ion=true_ion, corr_electrons=True
            )
            cls_ppm = common.clipped_ppm_single_norm(cls_mass_diff, parentmass)
            self.true_ppms.append(cls_ppm)

            ion_sublist, formulae_sublist = ions.split(","), formulae.split(",")
            (
                decoy_instr_vecs,
                decoy_forms,
                decoy_form_vecs,
                decoy_ion_vecs,
                decoy_ppms,
            ) = ([], [], [], [], [])
            for ion, formula in zip(ion_sublist, formulae_sublist):
                if ion == "" or ion == "[]":
                    continue

                decoy_forms.append(formula)

                # Create ion vec
                ion_encoding = self.ion_mat[common.get_ion_idx(ion)]
                decoy_ion_vecs.append(ion_encoding)

                # Create form vec
                form_vec = common.formula_to_dense(formula)
                decoy_form_vecs.append(form_vec)

                # Add instr vecs
                decoy_instr_vecs.append(true_instr_vec)

                # calculate ppm for parentmass
                cls_mass_diff = common.get_cls_mass_diff(
                    parentmass, form=formula, ion=ion, corr_electrons=True
                )
                cls_ppm = common.clipped_ppm_single_norm(cls_mass_diff, parentmass)
                decoy_ppms.append(cls_ppm)

            self.decoy_instr_vecs.append(decoy_instr_vecs)
            self.decoy_forms.append(decoy_forms)
            self.decoy_form_vecs.append(decoy_form_vecs)
            self.decoy_ion_vecs.append(decoy_ion_vecs)
            self.decoy_ppms.append(decoy_ppms)

        self.decoy_forms = np.array(self.decoy_forms, dtype=object)
        self.decoy_form_vecs = np.array(self.decoy_form_vecs, dtype=object)
        self.decoy_instr_vecs = np.array(self.decoy_instr_vecs, dtype=object)
        self.decoy_ion_vecs = np.array(self.decoy_ion_vecs, dtype=object)
        self.decoy_ppms = np.array(self.decoy_ppms, dtype=object)

        # Read in all specs
        self.spec_names = self.df["spec"].values
        process_spec_file = partial(
            common.process_spec_file_unbinned,
            data_dir=self.data_dir,
        )

        # Get binned spec file
        if self.num_workers == 0 or True:
            spec_outputs = [
                process_spec_file(spec_name) for spec_name in tqdm(self.spec_names)
            ]
        else:
            spec_outputs = common.chunked_parallel(
                self.spec_names,
                process_spec_file_unbinned,
                chunks=100,
                max_cpu=self.num_workers,
                timeout=4000,
                max_retries=3,
                use_ray=use_ray,
            )
        self.spec_ars = spec_outputs
        _spec_names, self.spec_ars = zip(*self.spec_ars)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        name = self.spec_names[idx]
        ar = self.spec_ars[idx]
        ar = max_peaks(ar, max_p=100)

        decoy_formulae = np.array(self.decoy_forms[idx])
        decoy_cls_ppm = np.array(self.decoy_ppms[idx])
        decoy_ion_vecs = np.array(self.decoy_ion_vecs[idx])
        decoy_instr_vecs = np.array(self.decoy_instr_vecs[idx])

        true_formula = self.true_forms[idx]
        true_cls_ppm = self.true_ppms[idx]
        true_ion_vec = self.true_ion_vecs[idx]
        true_instr_vec = self.true_instr_vecs[idx]

        embedded_true = self.true_form_vecs[idx]
        embedded_decoy = np.array(self.decoy_form_vecs[idx])

        full_decoy_idx_list = list(range(len(decoy_formulae)))
        num_decoys = len(full_decoy_idx_list)
        num_sample = min(self.max_decoy, num_decoys)
        if self.val_test:
            sample_inds = np.arange(0, num_sample)
        else:
            sample_inds = np.random.choice(num_decoys, num_sample, replace=False)

        decoy_form_lst = decoy_formulae[sample_inds].tolist()
        embedded_decoy_lst = embedded_decoy[sample_inds].tolist()
        decoy_cls_ppm_lst = decoy_cls_ppm[sample_inds].tolist()
        decoy_ion_vecs = decoy_ion_vecs[sample_inds].tolist()
        decoy_instr_vecs = decoy_instr_vecs[sample_inds].tolist()

        input_formulae = [embedded_true] + embedded_decoy_lst
        input_cls_ppms = [true_cls_ppm] + decoy_cls_ppm_lst
        input_ion_vec = [true_ion_vec] + decoy_ion_vecs
        input_instr_vec = [true_instr_vec] + decoy_instr_vecs

        input_ars = [ar] + [ar] * len(decoy_form_lst)

        # concatenate cls tokens
        parent_masses = np.vstack(input_formulae).dot(common.VALID_MONO_MASSES)
        cls_tokens = np.vstack([parent_masses, 2 * np.ones_like(parent_masses)]).T
        input_ars = [
            np.vstack([cls_token, input_ar])
            for input_ar, cls_token in zip(input_ars, cls_tokens)
        ]

        cand_formulae = [true_formula] + decoy_form_lst
        matched = [True] + [False] * len(decoy_form_lst)

        outdict = {
            "name": name,
            "matched": matched,
            "formulae": cand_formulae,
            "num_inputs": len(matched),
            # Define input comopnents
            "input_binned": input_ars,
            "input_forms": input_formulae,
            "input_ions": input_ion_vec,
            "input_cls_ppms": input_cls_ppms,
            "input_instr_vec": input_instr_vec,
        }

        return outdict

    @classmethod
    def get_collate_fn(cls):
        return XformerDataset.collate_fn

    @staticmethod
    def collate_fn(input_list):
        """collate_fn"""

        # Stack all decoys
        names = [j["name"] for j in input_list]

        binned_inputs = [j for i in input_list for j in i["input_binned"]]
        formulae_inputs = [j for i in input_list for j in i["input_forms"]]
        ion_inputs = [j for i in input_list for j in i["input_ions"]]
        instr_inputs = [j for i in input_list for j in i["input_instr_vec"]]

        cls_ppm_inputs = [float(j) for i in input_list for j in i["input_cls_ppms"]]
        str_forms = [j for i in input_list for j in i["formulae"]]
        num_inputs = [i["num_inputs"] for i in input_list]

        matched = [j for i in input_list for j in i["matched"]]
        example_inds = [ind for ind, num in enumerate(num_inputs) for _ in range(num)]

        # Now pad everything else to the max channel dim
        spectra_tensors = [torch.tensor(spec) for spec in binned_inputs]

        # Pad these
        num_peaks = [len(spec) for spec in spectra_tensors]
        num_peaks = torch.LongTensor(num_peaks)
        spectra_tensors = torch.nn.utils.rnn.pad_sequence(
            spectra_tensors, batch_first=True, padding_value=0
        )

        formula_tensors = torch.stack([torch.tensor(spec) for spec in formulae_inputs])
        ion_tensors = torch.stack([torch.tensor(spec) for spec in ion_inputs])
        instr_tensors = torch.stack([torch.tensor(instr) for instr in instr_inputs])

        rel_diff_tensors = torch.FloatTensor(cls_ppm_inputs)[:, None]
        matched = torch.BoolTensor(matched)
        num_inputs = torch.LongTensor(num_inputs)
        example_inds = torch.LongTensor(example_inds)

        return_dict = {
            "names": names,
            "example_inds": example_inds,
            "spec_ars": spectra_tensors,
            "num_peaks": num_peaks,
            "rel_mass_diffs": rel_diff_tensors,
            "formula": str_forms,
            "formula_tensors": formula_tensors,
            "ion_tensors": ion_tensors,
            "instrument_tensors": instr_tensors,
            "num_inputs": num_inputs,
            "matched": matched,
        }
        return return_dict


class PredDataset(Dataset):
    """PredDataset."""

    def __init__(
        self,
        df,
        data_dir,
        num_workers=0,
        **kwargs,
    ):
        """__init__.

        Args:
            df:
            data_dir:
            num_workers:
            spec_types (str): spec_types
            kwargs:
        """
        self.df = df
        self.num_workers = num_workers
        self.data_dir = data_dir
        self.ion_mat = np.eye(len(common.ION_LST))
        self.instr_embedder = InstrEmbedder()

        # for each (spec, cand_form, cand_ion) tuple, there is one corresponding embedded form
        self.spec_names = self.df["spec"].values
        self.cand_forms = self.df["cand_form"].values
        self.cand_ions = self.df["cand_ion"].values
        self.parentmasses = self.df["parentmass"].values
        self.instruments = self.df["instrument"].values
        self.form_vecs = []
        self.instr_vecs = []
        self.cls_ppms = []
        self.ion_vecs = []

        for cand_form, cand_ion, parentmass, instrument in zip(
            self.cand_forms, self.cand_ions, self.parentmasses, self.instruments
        ):

            # calculate ppm for parentmass
            cls_mass_diff = common.get_cls_mass_diff(
                parentmass, form=cand_form, ion=cand_ion, corr_electrons=True
            )
            cls_ppm = common.clipped_ppm_single_norm(cls_mass_diff, parentmass)
            self.cls_ppms.append(cls_ppm)

            form_vec = common.formula_to_dense(cand_form)
            self.form_vecs.append(form_vec)

            ion_vec = self.ion_mat[common.get_ion_idx(cand_ion)]
            self.ion_vecs.append(ion_vec)

            instr_vec = self.instr_embedder.embed_instr(instrument)
            self.instr_vecs.append(instr_vec)

        # Map names to indices
        name_to_idxs = defaultdict(lambda: [])
        for ind, j in enumerate(self.spec_names):
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

        self.process_spec_file = partial(
            common.process_spec_file_unbinned,
            data_dir=self.data_dir,
        )

    def __len__(self):
        return self.len

    def __getitem__(self, idx: int):
        name = self.new_ind_to_name[idx]
        _spec_name, spec_outputs = self.process_spec_file(name)
        spec_outputs = max_peaks(spec_outputs, max_p=100)

        out_list = []
        for list_idx in self.new_ind_to_old_inds[idx]:
            cand_form = self.cand_forms[list_idx]

            embedded_form = self.form_vecs[list_idx]
            cand_form_mass = embedded_form.dot(common.VALID_MONO_MASSES)
            spec_outputs_temp = np.vstack([[cand_form_mass, 2], spec_outputs])
            input_ion = self.ion_vecs[list_idx]
            instr_vec = self.instr_vecs[list_idx]

            parentmass = self.parentmasses[list_idx]
            cand_ion = self.cand_ions[list_idx]
            cand_cls_ppm = self.cls_ppms[list_idx]

            # Create meta
            outdict = {
                "name": name,
                "formula": cand_form,
                "parentmass": parentmass,
                "input_ar": spec_outputs_temp,
                "input_form": embedded_form,
                "cand_ion": cand_ion,
                "input_cls_ppm": cand_cls_ppm,
                "input_ion": input_ion,
                "instr_vec": instr_vec,
            }
            out_list.append(outdict)
        return out_list

    @classmethod
    def get_collate_fn(cls):
        return PredDataset.collate_fn

    @staticmethod
    def collate_fn(input_list):
        """collate_fn"""

        # Stack all decoys
        names = [i["name"] for j in input_list for i in j]
        binned_inputs = [i["input_ar"] for j in input_list for i in j]
        formulae_inputs = [i["input_form"] for j in input_list for i in j]
        cls_ppm_inputs = [i["input_cls_ppm"] for j in input_list for i in j]
        ion_inputs = [j["input_ion"] for i in input_list for j in i]
        instr_inputs = [j["instr_vec"] for i in input_list for j in i]

        cand_ions = [i["cand_ion"] for j in input_list for i in j]
        str_forms = np.array([i["formula"] for j in input_list for i in j])

        # Now pad everything else to the max channel dim
        spectra_tensors = [torch.tensor(spec) for spec in binned_inputs]
        formula_tensors = torch.stack([torch.tensor(spec) for spec in formulae_inputs])
        cls_ppm_tensors = torch.stack([torch.tensor([spec]) for spec in cls_ppm_inputs])
        ion_tensors = torch.stack([torch.tensor(spec) for spec in ion_inputs])
        instr_tensors = torch.stack([torch.tensor(spec) for spec in instr_inputs])

        num_peaks = [len(spec) for spec in spectra_tensors]
        num_peaks = torch.LongTensor(num_peaks)
        spectra_tensors = torch.nn.utils.rnn.pad_sequence(
            spectra_tensors, batch_first=True, padding_value=0
        )

        return_dict = {
            "names": names,
            "str_forms": str_forms,
            "cand_ions": cand_ions,
            "spec_ars": spectra_tensors,
            "num_peaks": num_peaks,
            "formula_tensors": formula_tensors,
            "ion_tensors": ion_tensors,
            "rel_mass_diffs": cls_ppm_tensors,
            "instrument_tensors": instr_tensors,
        }
        return return_dict
