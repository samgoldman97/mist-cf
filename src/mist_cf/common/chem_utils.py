"""chem_utils.py"""

import re
import numpy as np
import pandas as pd
import json
from functools import reduce

import torch
from rdkit import Chem
from rdkit.Chem import Atom
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.MolStandardize import rdMolStandardize

P_TBL = Chem.GetPeriodicTable()

ROUND_FACTOR = 4

ELECTRON_MASS = 0.00054858
CHEM_FORMULA_SIZE = "([A-Z][a-z]*)([0-9]*)"

VALID_ELEMENTS = [
    "C",
    "N",
    "P",
    "O",
    "S",
    "Si",
    "I",
    "H",
    "Cl",
    "F",
    "Br",
    "B",
    "Se",
    "Fe",
    "Co",
    "As",
    "K",
    "Na",
]


# Set the exact molecular weight?
# Use this to define an element priority queue
VALID_ATOM_NUM = [Atom(i).GetAtomicNum() for i in VALID_ELEMENTS]


CHEM_ELEMENT_NUM = len(VALID_ELEMENTS)

ATOM_NUM_TO_ONEHOT = torch.zeros((max(VALID_ATOM_NUM) + 1, CHEM_ELEMENT_NUM))

# Convert to onehot
ATOM_NUM_TO_ONEHOT[VALID_ATOM_NUM, torch.arange(CHEM_ELEMENT_NUM)] = 1

# Use Monoisotopic
# VALID_MASSES = np.array([Atom(i).GetMass() for i in VALID_ELEMENTS])
VALID_MONO_MASSES = np.array(
    [P_TBL.GetMostCommonIsotopeMass(i) for i in VALID_ELEMENTS]
)
CHEM_MASSES = VALID_MONO_MASSES[:, None]

ELEMENT_VECTORS = np.eye(len(VALID_ELEMENTS))
ELEMENT_VECTORS_MASS = np.hstack([ELEMENT_VECTORS, CHEM_MASSES])
ELEMENT_TO_MASS = dict(zip(VALID_ELEMENTS, CHEM_MASSES.squeeze()))

ELEMENT_DIM_MASS = len(ELEMENT_VECTORS_MASS[0])
ELEMENT_DIM = len(ELEMENT_VECTORS[0])

# Reasonable normalization vector for elements
# Estimated by max counts (+ 1 when zero)
NORM_VEC = np.array(
    [81, 19, 6, 34, 6, 6, 6, 158, 10, 17, 3, 1, 2, 1, 1, 2, 1, 2]
)

NORM_VEC_MASS = np.array(
    NORM_VEC.tolist() + [1471]
)

# Assume 64 is the highest repeat of any 1 atom
MAX_ELEMENT_NUM = 64

element_to_ind = dict(zip(VALID_ELEMENTS, np.arange(len(VALID_ELEMENTS))))
element_to_position = dict(zip(VALID_ELEMENTS, ELEMENT_VECTORS))
element_to_position_mass = dict(zip(VALID_ELEMENTS, ELEMENT_VECTORS_MASS))

ION_LST = [
    "[M+H]+",
    "[M+Na]+",
    "[M+K]+",
    "[M-H2O+H]+",
    "[M+H3N+H]+",
    "[M]+",
    "[M-H4O2+H]+",
]

ion_remap = dict(zip(ION_LST, ION_LST))
ion_remap.update({
    "[M+NH4]+": "[M+H3N+H]+",
    'M+H': '[M+H]+',
    'M+Na': "[M+Na]+",
    'M+H-H2O': "[M-H2O+H]+",
    'M-H2O+H': "[M-H2O+H]+",
    'M+NH4': "[M+H3N+H]+",
    'M-2H2O+H': "[M-H4O2+H]+",
    '[M-2H2O+H]+': "[M-H4O2+H]+",
})

ion_to_idx = dict(zip(ION_LST, np.arange(len(ION_LST))))

ion_to_mass = {
    "[M+H]+": ELEMENT_TO_MASS["H"] - ELECTRON_MASS,
    "[M+Na]+": ELEMENT_TO_MASS["Na"] - ELECTRON_MASS,
    "[M+K]+": ELEMENT_TO_MASS["K"] - ELECTRON_MASS,
    "[M-H2O+H]+": -ELEMENT_TO_MASS["O"] - ELEMENT_TO_MASS["H"] - ELECTRON_MASS,
    "[M+H3N+H]+": ELEMENT_TO_MASS["N"] + ELEMENT_TO_MASS["H"] * 4 - ELECTRON_MASS,
    "[M]+": 0 - ELECTRON_MASS,
    "[M-H4O2+H]+": -ELEMENT_TO_MASS["O"] * 2 - ELEMENT_TO_MASS["H"] * 3 - ELECTRON_MASS,
}


ion_to_add_vec = {
    "[M+H]+": element_to_position["H"],
    "[M+Na]+": element_to_position["Na"],
    "[M+K]+": element_to_position["K"],
    "[M-H2O+H]+": - element_to_position["O"] - element_to_position["H"],
    "[M+H3N+H]+": element_to_position["N"] + element_to_position["H"] * 4,
    "[M]+": np.zeros_like(element_to_position["H"]),
    "[M-H4O2+H]+": -element_to_position["O"] * 2 - element_to_position["H"] * 3,
}


instrument_to_type = {
    "Thermo Finnigan Velos Orbitrap": "orbitrap",
    "Thermo Finnigan Elite Orbitrap": "orbitrap",
    "Orbitrap Fusion Lumos": "orbitrap",
    "Q-ToF (LCMS)": "qtof",
    "Unknown (LCMS)": "unknown",
    "Ion Trap (LCMS)": "iontrap",
    "ion trap": "iontrap",
    "FTICR (LCMS)": "fticr",
    "Bruker Q-ToF (LCMS)": "qtof",
    "Orbitrap (LCMS)": "orbitrap"
}

instruments = sorted(list(set(instrument_to_type.values())))
max_instr_idx = len(instruments) + 1
instrument_to_idx = dict(zip(instruments, np.arange(len(instruments))))
instrument_to_tol = {
    "qtof": 10,
    "orbitrap": 5,
    "iontrap": 15,
    "fticr": 5,
    "unknown": 15
}

# Define rdbe mult
rdbe_mult = np.zeros_like(ELEMENT_VECTORS[0])
els = ["C", "N", "P", "H", "Cl", "Br", "I", "F"]
weights = [2, 1, 1, -1, -1, -1, -1, -1]
for k, v in zip(els, weights):
    rdbe_mult[element_to_ind[k]] = v 


def get_ion_idx(ionization: str) -> int:
    """ "map ionization to its index in one hot encoding"""
    return ion_to_idx[ionization]

def get_instr_idx(instrument: str) -> int:
    """ "map instrument to its index in one hot encoding"""
    inst = instrument_to_type.get(instrument, "uknown")
    return instrument_to_idx[inst]

def get_instr_tol(instrument: str) -> int:
    """ "map instrument to its mass tolerance"""
    inst = instrument_to_type.get(instrument, "uknown")
    return instrument_to_tol[inst]

def cross_sum(x, y):
    """cross_sum."""
    return (np.expand_dims(x, 0) + np.expand_dims(y, 1)).reshape(-1, y.shape[-1])


def get_all_subsets_dense(dense_formula: str, element_vectors) -> (np.ndarray, np.ndarray):
    """get_all_subsets.

    Args:
        chem_formula (str): Chem formula
    Return:
        Tuple of vecs and their masses
    """

    non_zero = np.argwhere(dense_formula > 0).flatten()

    vectorized_formula = []
    for nonzero_ind in non_zero:
        temp = element_vectors[nonzero_ind] * np.arange(0, dense_formula[nonzero_ind] + 1).reshape(-1, 1)
        vectorized_formula.append(temp)

    zero_vec = np.zeros((1, element_vectors.shape[-1]))
    cross_prod = reduce(cross_sum, vectorized_formula, zero_vec)

    cross_prod_inds = rdbe_filter(cross_prod)
    cross_prod = cross_prod[cross_prod_inds]
    all_masses = cross_prod.dot(VALID_MONO_MASSES)
    return cross_prod, all_masses


def get_all_subsets(chem_formula: str):
    dense_formula = formula_to_dense(chem_formula)
    return get_all_subsets_dense(dense_formula,
                                 element_vectors=ELEMENT_VECTORS)


#@jit(nopython=True)
def rdbe_filter(cross_prod):
    """rdbe_filter.
    Args:
        cross_prod:
    """
    rdbe_total = 1 + 0.5 * cross_prod.dot(rdbe_mult)
    filter_inds = np.argwhere(rdbe_total >= 0).flatten()
    return filter_inds


def assign_subforms(form, spec, ion_type, mass_diff_thresh=15):
    """assign_subforms.

    Args:
        form:
        spec:
        ion_type:
        mass_diff_thresh:
    """
    cross_prod, masses = get_all_subsets(form)
    spec_masses, spec_intens = spec[:, 0], spec[:, 1]

    ion_masses = ion_to_mass[ion_type]
    masses_with_ion = masses + ion_masses
    ion_types = np.array([ion_type] * len(masses_with_ion))

    mass_diffs = np.abs(spec_masses[:, None] - masses_with_ion[None, :])

    formula_inds = mass_diffs.argmin(-1)
    min_mass_diff = mass_diffs[np.arange(len(mass_diffs)), formula_inds]
    rel_mass_diff = clipped_ppm(min_mass_diff, spec_masses)

    # Filter by mass diff threshold (ppm)
    valid_mask = rel_mass_diff < mass_diff_thresh
    spec_masses = spec_masses[valid_mask]
    spec_intens = spec_intens[valid_mask]
    min_mass_diff = min_mass_diff[valid_mask]
    rel_mass_diff = rel_mass_diff[valid_mask]
    formula_inds = formula_inds[valid_mask]

    formulas = np.array([vec_to_formula(j)
                         for j in cross_prod[formula_inds]])
    formula_masses = masses_with_ion[formula_inds]
    ion_types = ion_types[formula_inds]

    # Build mask for uniqueness on formula and ionization
    # note that ionization are all the same for one subformula assignment
    # hence we only need to consider the uniqueness of the formula
    formula_idx_dict = {}
    uniq_mask = []
    for idx, formula in enumerate(formulas):
        uniq_mask.append(formula not in formula_idx_dict)
        gather_ind = formula_idx_dict.get(formula, None)
        if gather_ind is None:
            continue
        spec_intens[gather_ind] += spec_intens[idx]
        formula_idx_dict[formula] = idx

    spec_masses = spec_masses[uniq_mask]
    spec_intens = spec_intens[uniq_mask]
    min_mass_diff = min_mass_diff[uniq_mask]
    rel_mass_diff = rel_mass_diff[uniq_mask]
    formula_masses = formula_masses[uniq_mask]
    formulas = formulas[uniq_mask]
    ion_types = ion_types[uniq_mask]

    # To calculate explained intensity, preserve the original normalized
    # intensity
    if spec_intens.size == 0:
        output_tbl = None
    else:
        output_tbl = {
            "mz": list(spec_masses),
            "ms2_inten": list(spec_intens),
            "mono_mass": list(formula_masses),
            "abs_mass_diff": list(min_mass_diff),
            "mass_diff": list(rel_mass_diff),
            "formula": list(formulas),
            "ions": list(ion_types),
        }
    output_dict = {
        "cand_form": form,
        "cand_ion": ion_type,
        "output_tbl": output_tbl,
    }
    return output_dict


def get_output_dict(
    spec_name: str,
    spec: np.ndarray,
    form: str,
    mass_diff_type: str,
    mass_diff_thresh: float,
    ion_type: str,
) -> dict:
    """get_output_dict.
    This function attemps to take an array of mass intensity values and assign
    formula subsets to subpeaks
    Args:
        spec (np.ndarray): spec
        form (str): form
        abs_mass_diff (float): abs_mass_diff
        inten_thresh (float): Intensity threshold
    Returns:
        python dictionary
    """
    assert(mass_diff_type == "ppm")
    # This is the case for some erroneous MS2 files for which proc_spec_file return None
    # All the MS2 subpeaks in these erroneous MS2 files has mz larger than parentmass
    output_dict = {"cand_form": form, "cand_ion": ion_type, "output_tbl": None}
    if spec is not None and ion_type in ION_LST:
        output_dict = assign_subforms(form, spec, ion_type,
                                      mass_diff_thresh=mass_diff_thresh)
    return output_dict


def assign_single_spec(spec_name, export_dicts, output_dir):
    """assign_single_spec.

    Batch assign calculations for a single spectra in sequence.

    Args:
        spec_name:
        export_dicts:
        output_dir:
    """
    res_dict = {}
    for export_dict in export_dicts:
        output = get_output_dict(**export_dict)


        res_dict[output["cand_form"]] = {
            "cand_ion": output["cand_ion"],
            "cand_tbl": output["output_tbl"],
        }

    if output_dir is not None:
        with open(output_dir / f"{spec_name}.json", "w") as f:
            json.dump(res_dict, f, indent=4)
            f.close()
    return res_dict


def clipped_ppm(mass_diff: np.ndarray, parentmass: np.ndarray) -> np.ndarray:
    """clipped_ppm.

    Args:
        mass_diff (np.ndarray): mass_diff
        parentmass (np.ndarray): parentmass

    Returns:
        np.ndarray:
    """
    parentmass_copy = parentmass * 1
    parentmass_copy[parentmass < 200] = 200
    ppm = mass_diff / parentmass_copy * 1e6
    return ppm


def clipped_ppm_single(cls_mass_diff: float, parentmass: float,):
    """clipped_ppm_single.

    Args:
        cls_mass_diff (float): cls_mass_diff
        parentmass (float): parentmass
    """
    div_factor = 200 if parentmass < 200 else parentmass
    cls_ppm = cls_mass_diff / div_factor * 1e6
    return cls_ppm


def clipped_ppm_single_norm(cls_mass_diff: float, parentmass: float):
    """clipped_ppm_single_norm.

    Args:
        cls_mass_diff (float): cls_mass_diff
        parentmass (float): parentmass
        normalize_factor (float): normalize_factor
    """
    return norm_mass_diff_ppm(clipped_ppm_single(cls_mass_diff, parentmass))

def norm_mass_diff_ppm(mass_diff):
    return mass_diff / 10


def get_cls_mass_diff(parentmass: float, form: str, ion: str,
                      corr_electrons=True):
    """get_cls_mass_diff.

    Args:
        parentmass (float): parentmass
        form (str): form
        ion (str): ion
        corr_electrons:
    """

    true_val = formula_mass(form) + ion_to_mass[ion]
    if corr_electrons:
        true_val = electron_correct(true_val)
    return abs(parentmass - true_val)


def formula_to_dense(chem_formula: str) -> np.ndarray:
    """formula_to_dense.

    Args:
        chem_formula (str): Input chemical formal
    Return:
        np.ndarray of vector

    """
    total_onehot = []
    for (chem_symbol, num) in re.findall(CHEM_FORMULA_SIZE, chem_formula):
        # Convert num to int
        num = 1 if num == "" else int(num)
        one_hot = element_to_position[chem_symbol].reshape(1, -1)
        one_hot_repeats = np.repeat(one_hot, repeats=num, axis=0)
        total_onehot.append(one_hot_repeats)

    # Check if null
    if len(total_onehot) == 0:
        dense_vec = np.zeros(len(element_to_position))
    else:
        dense_vec = np.vstack(total_onehot).sum(0)

    return dense_vec


def formula_to_dense_mass(chem_formula: str) -> np.ndarray:
    """formula_to_dense_mass.

    Return formula including full compound mass

    Args:
        chem_formula (str): Input chemical formal
    Return:
        np.ndarray of vector

    """
    total_onehot = []
    for (chem_symbol, num) in re.findall(CHEM_FORMULA_SIZE, chem_formula):
        # Convert num to int
        num = 1 if num == "" else int(num)
        one_hot = element_to_position_mass[chem_symbol].reshape(1, -1)
        one_hot_repeats = np.repeat(one_hot, repeats=num, axis=0)
        total_onehot.append(one_hot_repeats)

    # Check if null
    if len(total_onehot) == 0:
        dense_vec = np.zeros(len(element_to_position_mass["H"]))
    else:
        dense_vec = np.vstack(total_onehot).sum(0)

    return dense_vec


def formula_to_dense_mass_norm(chem_formula: str) -> np.ndarray:
    """formula_to_dense_mass_norm.

    Return formula including full compound mass and normalized

    Args:
        chem_formula (str): Input chemical formal
    Return:
        np.ndarray of vector

    """
    dense_vec = formula_to_dense_mass(chem_formula)
    dense_vec = dense_vec / NORM_VEC_MASS

    return dense_vec


def formula_mass(chem_formula: str) -> float:
    """get formula mass"""
    mass = 0
    for (chem_symbol, num) in re.findall(CHEM_FORMULA_SIZE, chem_formula):
        # Convert num to int
        num = 1 if num == "" else int(num)
        mass += ELEMENT_TO_MASS[chem_symbol] * num
    return mass


def electron_correct(mass: float) -> float:
    """subtract the rest mass of an electron"""
    return mass - ELECTRON_MASS


def formula_difference(formula_1, formula_2):
    """formula_1 - formula_2"""
    form_1 = {
        chem_symbol: (int(num) if num != "" else 1)
        for chem_symbol, num in re.findall(CHEM_FORMULA_SIZE, formula_1)
    }
    form_2 = {
        chem_symbol: (int(num) if num != "" else 1)
        for chem_symbol, num in re.findall(CHEM_FORMULA_SIZE, formula_2)
    }

    for k, v in form_2.items():
        form_1[k] = form_1[k] - form_2[k]
    out_formula = "".join([f"{k}{v}" for k, v in form_1.items() if v > 0])
    return out_formula


def get_mol_from_structure_string(structure_string, structure_type):
    if structure_type == "InChI":
        mol = Chem.MolFromInchi(structure_string)
    else:
        mol = Chem.MolFromSmiles(structure_string)
    return mol


def vec_to_formula(form_vec):
    """vec_to_formula."""
    build_str = ""
    for i in np.argwhere(form_vec > 0).flatten():
        el = VALID_ELEMENTS[i]
        ct = int(form_vec[i])
        new_item = f"{el}{ct}" if ct > 1 else f"{el}"
        build_str = build_str + new_item
    return build_str


def standardize_form(i):
    """ standardize_form. """
    return vec_to_formula(formula_to_dense(i))


def standardize_adduct(adduct):
    """standardize_adduct."""
    adduct = adduct.replace(" ", "")
    adduct = ion_remap.get(adduct, adduct)  
    if adduct not in ION_LST:
        raise ValueError(f"Adduct {adduct} not in ION_LST") 
    return adduct


def calc_structure_string_type(structure_string):
    """calc_structure_string_type.

    Args:
        structure_string:
    """
    structure_type = None
    if pd.isna(structure_string):
        structure_type = "empty"
    elif structure_string.startswith("InChI="):
        structure_type = "InChI"
    elif Chem.MolFromSmiles(structure_string) is not None:
        structure_type = "Smiles"
    return structure_type


def uncharged_formula(mol, mol_type="mol") -> str:
    """Compute uncharged formula"""
    if mol_type == "mol":
        chem_formula = CalcMolFormula(mol)
    elif mol_type == "smiles":
        mol = Chem.MolFromSmiles(mol)
        if mol is None:
            return None
        chem_formula = CalcMolFormula(mol)
    else:
        raise ValueError()

    return re.findall(r"^([^\+,^\-]*)", chem_formula)[0]


def form_from_smi(smi: str) -> str:
    """form_from_smi.

    Args:
        smi (str): smi

    Return:
        str
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return ""
    else:
        return CalcMolFormula(mol)


def inchikey_from_smiles(smi: str) -> str:
    """inchikey_from_smiles.

    Args:
        smi (str): smi

    Returns:
        str:
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return ""
    else:
        return Chem.MolToInchiKey(mol)


def contains_metals(formula: str) -> bool:
    """returns true if formula contains metals"""
    METAL_RE = "(Fe|Co|Zn|Rh|Pt|Li)"
    return len(re.findall(METAL_RE, formula)) > 0


class SmilesStandardizer(object):
    """Standardize smiles"""

    def __init__(self, *args, **kwargs):
        self.fragment_standardizer = rdMolStandardize.LargestFragmentChooser()
        self.charge_standardizer = rdMolStandardize.Uncharger()

    def standardize_smiles(self, smi):
        """Standardize smiles string"""
        mol = Chem.MolFromSmiles(smi)
        out_smi = self.standardize_mol(mol)
        return out_smi

    def standardize_mol(self, mol) -> str:
        """Standardize smiles string"""
        mol = self.fragment_standardizer.choose(mol)
        mol = self.charge_standardizer.uncharge(mol)

        # Round trip to and from inchi to tautomer correct
        # Also standardize tautomer in the middle
        output_smi = Chem.MolToSmiles(mol, isomericSmiles=False)
        return output_smi


def mass_from_smi(smi: str) -> float:
    """mass_from_smi.

    Args:
        smi (str): smi

    Return:
        str
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return 0
    else:
        return ExactMolWt(mol)


def min_formal_from_smi(smi: str):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return 0
    else:
        formal = np.array([j.GetFormalCharge() for j in mol.GetAtoms()])
        return formal.min()


def max_formal_from_smi(smi: str):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return 0
    else:
        formal = np.array([j.GetFormalCharge() for j in mol.GetAtoms()])
        return formal.max()


def atoms_from_smi(smi: str) -> int:
    """atoms_from_smi.

    Args:
        smi (str): smi

    Return:
        int
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return 0
    else:
        return mol.GetNumAtoms()


def has_valid_els(chem_formula: str) -> bool:
    """has_valid_els"""
    for (chem_symbol, num) in re.findall(CHEM_FORMULA_SIZE, chem_formula):
        if chem_symbol not in VALID_ELEMENTS:
            return False
    return True

def add_ion(form: str, ion: str):
    """ add_ion.
    Args:
        form (str): form
        ion (str): ion
    """
    ion_vec = ion_to_add_vec[ion]
    form_vec = formula_to_dense(form)
    return vec_to_formula(form_vec + ion_vec)