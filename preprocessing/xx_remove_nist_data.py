import pandas as pd
from pathlib import Path
from shutil import copyfile, copytree


src_dir_base = Path("data/nist_canopus/")
dst_dir_base = Path("data/canopus_train/")
dst_dir_base.mkdir(parents=True, exist_ok=True)

# Step 1: Copy decoy labels
src_decoys = src_dir_base / "decoy_labels"
trg_decoys = dst_dir_base / "decoy_labels"

trg_decoys.mkdir(parents=True, exist_ok=True)

if True:
    for src_file in src_decoys.glob("*.tsv"):
        if "RDBE" in src_file.name:
            continue
        print(src_file)
        trg_file = trg_decoys / src_file.name
        df = pd.read_csv(src_file, sep="\t")
        # Sub 1
        # Filter df where "dataset == "canopus_train"
        #df1 = df[df["dataset"] == "canopus_train"]
        # Sub 2
        # Filter spec where "nist" not in spec
        df2 = df[~df["spec"].str.contains("nist")]
        df2.to_csv(trg_file, sep="\t", index=False)
    
src_pred_labels = src_dir_base / "pred_labels"
trg_pred_labels = dst_dir_base / "pred_labels"

trg_pred_labels.mkdir(parents=True, exist_ok=True)

# Copy recursively all from src ot trg
if True:
    for src_file in src_pred_labels.glob("*.tsv"):
        # Ignore RDBE
        if "RDBE" in src_file.name:
            continue
        print(src_file)
        trg_file = trg_pred_labels / src_file.name
        copyfile(src_file, trg_file)

src_spec_files = src_dir_base / "spec_files"
trg_spec_files = dst_dir_base / "spec_files"

trg_spec_files.mkdir(parents=True, exist_ok=True)

# Loop over all spec and copy only if "nist" not in spec
if True:
    for src_file in src_spec_files.glob("*.ms"):
        trg_file = trg_spec_files / src_file.name
        src_name = src_file.name
        if "nist" not in src_name:
            print(src_file)
            copyfile(src_file, trg_file)

# Copy split folder in its entirety
if True:
    src_split = src_dir_base / "splits"
    trg_split = dst_dir_base / "splits"
    copytree(src_split, trg_split, dirs_exist_ok=True)


# Copy all .mgf files in root
if True:
    for src_file in src_dir_base.glob("*.mgf"):
        trg_file = dst_dir_base / src_file.name
        print(src_file)
        copyfile(src_file, trg_file)

src_subformulae = src_dir_base / "subformulae/formulae_spec_decoy_label_COMMON"
trg_subformulae = dst_dir_base / "subformulae/formulae_spec_decoy_label_COMMON"

trg_subformulae.mkdir(parents=True, exist_ok=True)

# Loop over all spec and copy only if "nist" not in spec
if True:
    for src_file in src_subformulae.glob("*.json"):
        trg_file = trg_subformulae/ src_file.name
        src_name = src_file.name
        if "nist" not in src_name:
            print(src_file)
            copyfile(src_file, trg_file)

# Copy labels.tsv file but only keep spec that don't have nist
src_file = src_dir_base / "labels.tsv"
trg_file = dst_dir_base / "labels.tsv"
src_df = pd.read_csv(src_file, sep="\t")
trg_df = src_df[~src_df["spec"].str.contains("nist")]
trg_df.to_csv(trg_file, sep="\t", index=False)