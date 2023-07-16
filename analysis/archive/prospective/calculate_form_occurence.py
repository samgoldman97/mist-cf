"""
Calculate the occurance of training formula in prospective data.
# TODO: normalize formula
"""

import numpy as np
import pandas as pd
from pathlib import Path
import mist_cf.common as common

prospective_csv = Path("data/bile_acid/bile_acid_refined.csv")
prospective_df = pd.read_csv(prospective_csv)
prospective_df.columns = prospective_df.iloc[0]
prospective_df = prospective_df.drop(prospective_df.index[0])
prospective_df = prospective_df.dropna()
# drop ground_truth df with unseen adducts
# All adducts: nan, 'M-2H2O+H', 'M+Na', 'M+H-H2O', '[M-2H2O+H]+', 'M+NH4', 'M+H', 'Unknown', 'M-H2O+H', 'M-3H2O+H', '2M+H', '[M+H]+'
valid_adduct_lst = ['M+H', '[M+H]+']
prospective_df = prospective_df[prospective_df['Adduct'].isin(valid_adduct_lst)]
# Using apply function single column
def format_H_plus(x):
    if x == '[M+H]+':
        pass
    elif x == 'M+H':
        return '[M+H]+'
    else: 
        raise ValueError(f"{x} is not a valid H+ adduct")
prospective_df['Adduct'] = prospective_df['Adduct'].apply(format_H_plus)
# 'molecular_formula'
# breakpoint()

training_tsv = Path("/home/jiayixin/mist_cf/data/nist_canopus/labels.tsv")
training_df = pd.read_csv(training_tsv, sep='\t')
# 'formula'
all_training_forms = set([common.standardize_form(i) for i in training_df.formula.values])

scan_to_ion = pd.Series(prospective_df.Adduct.values,index=prospective_df['#Scan#']).to_dict()
scan_to_formula = pd.Series(prospective_df.molecular_formula.values,index=prospective_df['#Scan#']).to_dict()

unique_forms = set()
unique_occur_forms = set()
total_cnt = 0
unique_cnt = 0
total_occur = 0
unique_occur = 0

for scan_str, ion_str in scan_to_ion.items():
    if ion_str != '[M+H]+':
        pass
    else: 
        total_cnt += 1
        prospective_form = scan_to_formula[scan_str]
        prospective_form = common.standardize_form(prospective_form)
        if prospective_form not in unique_forms:
            unique_cnt += 1
            unique_forms.add(prospective_form)
        if prospective_form in all_training_forms:
            total_occur += 1
            if prospective_form not in unique_occur_forms:
                unique_occur += 1
                unique_occur_forms.add(prospective_form)

print(f'Total num of H+ bile acid specs: {total_cnt}')
print(f'Num of unique H+ bile acid specs: {unique_cnt}')
print(f'Total num of ocurrence in NIST training: {total_occur}')
print(f'Num of unique ocurrence in NIST training: {unique_occur}')
print(f'Frac of occurence: {np.round(100*(total_occur/total_cnt),2)}%')
print(f'Frac of unique occurence: {np.round(100*(unique_occur/unique_cnt),2)}%')
