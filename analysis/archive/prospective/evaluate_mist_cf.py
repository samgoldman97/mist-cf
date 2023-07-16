"""
Evaluate prospective accuracy for bile acid.
# TODO: normalize formula
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
        pass
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
# breakpoint()

# p type(pred_df['spec'][0]) - <class 'numpy.int64'>
result_dir = Path("results/")
for dir in result_dir.glob('bile_acid_H_only_resample_instrument*'):
    fast_filter_pred_path =  dir / 'pred_labels_filter.tsv'
    mist_cf_pred_path =  dir / 'formatted_output.tsv'
    fast_filter_pred_df = pd.read_csv(fast_filter_pred_path, sep='\t')
    mist_cf_pred_df = pd.read_csv(mist_cf_pred_path, sep='\t')
    fast_filter_pred_df = fast_filter_pred_df.sort_values(by='scores',ascending=False)
    mist_cf_pred_df = mist_cf_pred_df.sort_values(by='scores',ascending=False)

    cnt = 0
    fast_filter_correct = 0
    fast_filter_fnd = 0
    mist_cf_correct = 0
    mist_cf_correct_k10 = 0
    for scan_str, ion_str in gt_scan_to_ion.items():
        if ion_str != '[M+H]+':
            pass
        else: 
            cnt += 1
            gt_form = gt_scan_to_formula[scan_str]

            fast_filter_pred = fast_filter_pred_df[fast_filter_pred_df['spec']==int(scan_str)].cand_form.values
            fast_filter_pred_cand_lst = [common.standardize_form(i) for i in fast_filter_pred]
            if gt_form in fast_filter_pred_cand_lst[:1]:
                fast_filter_correct += 1

            mist_cf_pred = mist_cf_pred_df[mist_cf_pred_df['spec']==int(scan_str)].cand_form.values
            mist_cf_pred_cand_lst = [common.standardize_form(i) for i in mist_cf_pred]
            if gt_form in mist_cf_pred_cand_lst[:1]:
                mist_cf_correct += 1
            if gt_form in mist_cf_pred_cand_lst:
                fast_filter_fnd += 1
            if gt_form in mist_cf_pred_cand_lst[:10]:
                mist_cf_correct_k10 += 1
            # breakpoint()

    with open(dir/'evaluation.txt', 'w') as f:
        f.write(f'Total H+ labeled specs: {cnt}\n')
        f.write(f'mist_cf Top 1 acc: {np.round(100*(mist_cf_correct/cnt),2)}%\n')
        f.write(f'mist_cf Top 10 acc: {np.round(100*(mist_cf_correct_k10/cnt),2)}%\n')
        f.write(f'Fast Filter Top 1 acc: {np.round(100*(fast_filter_correct/cnt),2)}%\n')
        f.write(f'Fast Filter Top 256 acc: {np.round(100*(fast_filter_fnd/cnt),2)}%\n')
        f.close()


