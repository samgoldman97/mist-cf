""" sirius_decomp.py

Implement wrapper calls around SIRIUS to extract formula decompositions

Must first download sirius

"""

import math
import tempfile
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import os
import re

import mist_cf.common as common

SIRIUS_LOC = Path(os.getenv("SIRIUS_PATH"))
ROUND_FACTOR = 4
EL_STR_DEFAULT = 'C[0-]N[0-]O[0-]H[0-]S[0-5]P[0-3]I[0-1]Cl[0-1]F[0-1]Br[0-1]'


def parse_element_str(el_str, max_default=999):
    """ Accept el str and return a dict from el to dict(min=min,max=max)"""

    regex = r"([A-Z][a-z]?)(\[\d+-?\d*\])?"
    matches = re.finditer(regex, el_str, re.MULTILINE)
    out = {}
    for matchNum, match in enumerate(matches, start=1):
        el = match.group(1)
        if match.group(2) is None:
            out[el] = dict(min=0, max=0)
        else:
            min_, max_ = match.group(2).strip("[]").split("-")
            if max_ == "":
                max_ = max_default
            out[el] = dict(min=int(min_), max=int(max_))
    return out


def run_sirius(
    masses: list, adduct=None, verbose=False, mass_sort=True, filter_="NONE",
    ppm=15, max_batch=10000, el_str=EL_STR_DEFAULT, cores=16,
    loglevel="WARNING",
):
    """run_sirius."""

    if loglevel == "NONE": 
        log_cmds = {"stdout": subprocess.DEVNULL,
                    "stderr": subprocess.DEVNULL }
        loglevel = "WARNING"
    else:
        log_cmds =  {}


    if cores == 0: 
        cores = 1

    masses = [np.round(i, ROUND_FACTOR) for i in masses]
    masses = np.array(masses).astype(str)

    out_dfs = []
    num_sections = math.ceil(masses.shape[0] / max_batch)
    mass_splits = np.array_split(masses, num_sections)
    for mass_split in mass_splits:
        temp_file = tempfile.NamedTemporaryFile()
        file_name = temp_file.name
        mass_list = ",".join(mass_split)
        cmd = f"{SIRIUS_LOC} --cores {cores} --log {loglevel} decomp \
        --mass {mass_list} --output {file_name} \
        --elements {el_str} \
        --ppm {ppm} "
        if adduct is not None:
            cmd = f"{cmd} --ion {adduct}"
        if filter_ is not None:
            cmd = f"{cmd} --filter {filter_}"

        if verbose:
            print(f"Running sirius command:\n {cmd}")

        subprocess.run(cmd, shell=True, **log_cmds)
        df = pd.read_csv(file_name, sep="\t")
        out_dfs.append(df)
    df = pd.concat(out_dfs).reset_index(drop=True)
    mass_to_forms = dict(df[["m/z", "decompositions"]].values)

    mass_to_form_lists = {}
    for i, j in mass_to_forms.items():
        if not isinstance(j, str):
            cands = []
        else:
            cands = j.strip().split(",")
        if mass_sort:
            cands_masses = np.array([common.formula_mass(cand) for cand in cands])
            new_inds = np.argsort(np.abs(cands_masses - i))
            cands = np.array(cands)[new_inds].tolist()
        mass_to_form_lists[i] = cands

    return mass_to_form_lists


def get_rounded_masses(masses: list):
    """get_rounded_masses.

    Args:
        masses (List[float]): List of float masses

    Return:
        List[float]: Float rounded

    """
    return [np.round(i, ROUND_FACTOR) for i in masses]


def test():
    """test."""
    forms = ["C23H32N2O5", "C18H19N3O3S"]
    true_masses = [common.formula_mass(i) for i in forms]
    mass_to_form_lists = run_sirius(
        true_masses,
        verbose=True,
    )  # adduct="[M+H]+")

    # Filter by accuracy
    # Mass of interest
    moi = np.round(true_masses[0], ROUND_FACTOR)
    cands = np.array(mass_to_form_lists[moi])
    cands_masses = np.array([common.formula_mass(i) for i in cands])
    new_inds = np.argsort(np.abs(cands_masses - moi))
    cands_sorted = cands[new_inds]
    cands_masses = cands_masses[new_inds]

    # Check if the true form was recovered correctly


if __name__ == "__main__":
    test()
