"""
Evaluate prospective accuracy for bile acid.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import mist_cf.common as common

ground_truth_csv = "data/bile_acid/bile_acid_refined.csv"
ground_truth_df = pd.read_csv(ground_truth_csv)
ground_truth_df.columns = ground_truth_df.iloc[0]
ground_truth_df = ground_truth_df.drop(ground_truth_df.index[0])
ground_truth_df = ground_truth_df.dropna()

# drop ground_truth df with unseen adducts
# All adducts: nan, 'M-2H2O+H', 'M+Na', 'M+H-H2O', '[M-2H2O+H]+', 'M+NH4', 'M+H', 'Unknown', 'M-H2O+H', 'M-3H2O+H', '2M+H', '[M+H]+'
valid_adduct_lst = ['M+H', '[M+H]+']
ground_truth_df = ground_truth_df[ground_truth_df['Adduct'].isin(valid_adduct_lst)]

# Using apply function single column
def format_H_plus(x):
    if x == '[M+H]+':
        return '[M+H]+'
    elif x == 'M+H':
        return '[M+H]+'
    else: 
        raise ValueError(f"{x} is not a valid H+ adduct")
ground_truth_df['Adduct'] = ground_truth_df['Adduct'].apply(format_H_plus)

def standarize_formulas(x):
    return common.standardize_form(x)
ground_truth_df['molecular_formula'] = ground_truth_df['molecular_formula'].apply(standarize_formulas)

# list(ground_truth_df['Adduct'].values).count('[M+H]+') - 722
# type(ground_truth_df['#Scan#'].values[0]) - <class 'str'>
gt_scan_to_ion = pd.Series(ground_truth_df.Adduct.values,index=ground_truth_df['#Scan#']).to_dict()
gt_scan_to_formula = pd.Series(ground_truth_df.molecular_formula.values,index=ground_truth_df['#Scan#']).to_dict()
gt_scan_to_SpecID = pd.Series(ground_truth_df.SpectrumID.values,index=ground_truth_df['#Scan#']).to_dict()
gt_scan_to_Precursor_MZ = pd.Series(ground_truth_df.Precursor_MZ.values,index=ground_truth_df['#Scan#']).to_dict()

sirius_csv_dir = Path("results/sirius_bile_acid_default_args/summary")
sirius_pred_dir_name = "Refined_24d96e55_ScanNumber"

# top_k = 10
# assert top_k in list(range(11))
cnt = 0
correct = 0
for scan_str, ion_str in gt_scan_to_ion.items():
    if ion_str != '[M+H]+':
        pass
    else: 
        cnt += 1
        prefix = str(int(scan_str)-1)
        subdir_name = f"{prefix}_{sirius_pred_dir_name}{str(scan_str)}"
        subdir = sirius_csv_dir / subdir_name 
        csv_path = subdir / 'formula_candidates.tsv'
        pred_df = pd.read_csv(csv_path, sep='\t')

        pred_cand_lst = pred_df.molecularFormula.values
        pred_cand_lst = [common.standardize_form(i) for i in pred_cand_lst]
        pred_cand_lst = pred_cand_lst[:1]
        gt_form = gt_scan_to_formula[scan_str]
        # breakpoint()
        if gt_form in pred_cand_lst:
            correct += 1

print(f'Total H+ labeled specs: {cnt}')
print(f'Result of prediction: {sirius_csv_dir.parents[0]}')
print(f'Ground truth found in top 1: {correct}')
print(f'Top 1 annotation acc: {np.round(100*(correct/cnt),2)}%')