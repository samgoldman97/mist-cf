import numpy as np
import pandas as pd
from pathlib import Path

# Clean this up & run with both versions of split_name 

# Step 1: Subset split

# @JIAYI CHANGE THIS
split_name = "split_1.tsv"
split_name = "split_1_with_nist.tsv"

base_path = Path("data/nist_canopus/")
split_file = base_path / f"split/{split_name}"
out_split = split_file.parent / f"{split_file.stem}_only_h.tsv"

df = pd.read_csv(split_file, sep="\t")

# Subset to h+
df = df[df['ionization'] == "[M+H]+"]
df.to_csv(out_split, sep="\t", index=None)

# Step 2: Subset decoy labels
old_decoys = base_path / "decoy_label_COMMON.tsv"
new_decoys = old_decoys.parent / f"{old_decoys.stem}_only_h.tsv"

# Subset to h+
decoy_df = pd.read_csv(new_decoys, sep="\t")
new_df = []
for _, row in decoy_df.iterrows():
    new_obj = dict(row)
    forms = np.array(new_obj['decoy_formula'])
    ions = np.array(new_obj['decoy_ions'])

    valid = ions == "[M+H]+"
    new_obj['decoy_formula'] = forms[valid].tolist()
    new_obj['decoy_ions'] = ions[valid].tolist()
    new_df.append(new_obj)

new_df = pd.DataFrame(new_df)
new_df.to_csv(new_decoys, sep="\t", index=None)
