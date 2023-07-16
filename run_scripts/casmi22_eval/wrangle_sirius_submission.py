""" Process sirius results """
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from mist_cf import common

res_dir  = Path("results/sirius_predict_casmi22/sirius_submission")
res_dir.mkdir(exist_ok=True, parents=True)

raw_file = "data/casmi22/duehrkop_CASMI2022.csv"
labels_file = "data/casmi22/CASMI_labels.tsv"

raw_file = pd.read_csv(raw_file, sep="\t")
labels_file = pd.read_csv(labels_file, sep="\t")

true_spec = set([str(i) for i in labels_file['spec'].values])
outputs = []
for _, row in raw_file.iterrows():
    parentmass = row['Precursor m/z (Da)']
    spec = str(row['Compound Number'])
    cand_form = row['Molecular Formula']
    score = 1
    cand_ion = row['Adduct'].replace(" ", "")
    if cand_ion  not in common.ion_remap:
        print(f"Ignoring because of ion {cand_ion}")
        continue
    if spec not in true_spec:
        print(spec, row['File'])
        continue
    new_entry = dict(spec=spec, cand_form=cand_form, scores=score, cand_ion=cand_ion, parentmasses=parentmass)
    outputs.append(new_entry)

out_df = pd.DataFrame(outputs).reset_index(drop=True)
out_df.to_csv(res_dir / "formatted_output.tsv", 
                sep="\t", index=False)