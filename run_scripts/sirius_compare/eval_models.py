""" Make predictions for all models that were trained """
import yaml
import pandas as pd
from pathlib import Path
import subprocess
import re
import yaml

run_cmds = True
dataset = "nist_canopus"
data_dir = Path(f"data/{dataset}")
true_label = data_dir / "labels.tsv"
split = data_dir / f"splits/split_1.tsv"

results = [
    dict(save_dir="results/mist_cf_predict_sirius/mist_cf_1",
         split=f"data/{dataset}/splits/split_1.tsv"),
    dict(save_dir="results/mist_cf_predict_sirius/mist_cf_2",
         split=f"data/{dataset}/splits/split_2.tsv"),
    dict(save_dir="results/mist_cf_predict_sirius/mist_cf_3",
         split=f"data/{dataset}/splits/split_3.tsv"),
    dict(save_dir="results/sirius_gnps_pred/sirius_1/",
         split=f"data/{dataset}/splits/split_1.tsv"),
    dict(save_dir="results/sirius_gnps_pred/sirius_2/",
         split=f"data/{dataset}/splits/split_2.tsv"),
    dict(save_dir="results/sirius_gnps_pred/sirius_3/",
         split=f"data/{dataset}/splits/split_3.tsv"),
]

for res_dict in results:
    save_dir = res_dict['save_dir']
    split = res_dict['split']

    # Run evaluation
    eval_cmd = f"""
    python analysis/evaluate_pred.py \\
    --true-label {true_label} \\
    --res-dir {save_dir}  \\
    --split-file {split} \\
    --subset-dataset test_only \\
    """
    print(eval_cmd)
    cmd = eval_cmd if run_cmds else ""
    subprocess.run(eval_cmd, shell=True)
