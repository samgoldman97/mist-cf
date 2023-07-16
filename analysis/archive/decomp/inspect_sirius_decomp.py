import mist_cf.decomp as decomp
import mist_cf.common as common
import pandas as pd
import numpy as np

# df1 = pd.read_csv('data/canopus_train/labels.tsv',sep='\t')
# df2 = pd.read_csv('results/decomp_recover/recover_failed_specs_decoy_label_RDBE_mz_resampled.tsv',sep='\t')

# specs1 = df1['spec'].values
# forms1 = df1['formula'].values
# spec_to_form1 = {specs1[idx]: forms1[idx] for idx in range(len(specs1))}

# specs2 = df2['spec'].values
# forms2 = df2['formula'].values
# spec_to_form2 = {specs2[idx]: forms2[idx] for idx in range(len(specs2))}

# not_recovered_forms = []
# for spec in specs2:
#     if not ('Cl' in spec_to_form1[spec]) and not 'F' in spec_to_form1[spec]:
#         not_recovered_forms.append(spec_to_form1[spec])
#     else:
#         pass

# print(len(not_recovered_forms))
# print(not_recovered_forms)
true_formulas = [
    "C19H25BN4O4",
    "C16H15F2N3Si",
    "C16H15F2N3Si",
    "C28H23BCl2F4N2O4",
    "C19H25BN4O4",
    "C15H11BF3NO3",
]
true_masses = [
    np.round(common.formula_mass(true_formula), 4) for true_formula in true_formulas
]
form2mass = {f: m for f, m in zip(true_formulas, true_masses)}
decomp_filter = "RDBE"

out_dict = decomp.run_sirius(true_masses, filter_=decomp_filter)


for true_formula in true_formulas:
    mass = form2mass[true_formula]
    cands = out_dict.get(mass, [])
    print(len(cands))
    if true_formula in cands:
        print("Recovered!")
    else:
        print("Not recovered!")
