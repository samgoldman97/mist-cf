""" Make predictions for all models that were trained """
import yaml
import pandas as pd
from pathlib import Path
import subprocess
import re
import yaml

run_cmds = True
dataset = "casmi22"
data_dir = Path(f"data/{dataset}")
true_label = data_dir / "CASMI_labels.tsv"

results = [
    dict(save_dir="results/mist_cf_predict_casmi22_50_peaks"),
    dict(save_dir="results/mist_cf_predict_casmi22"),
    dict(save_dir="results/sirius_predict_casmi22/sirius_1/"),
    dict(save_dir="results/sirius_predict_casmi22/sirius_1_structure/"),
    dict(save_dir="results/sirius_predict_casmi22/sirius_submission/"),
]

for res_dict in results:
    save_dir = res_dict["save_dir"]

    # Run evaluation
    eval_cmd = f"""
    python analysis/evaluate_pred.py \\
    --true-label {true_label} \\
    --res-dir {save_dir}  \\
    """
    print(eval_cmd)
    cmd = eval_cmd if run_cmds else ""
    subprocess.run(eval_cmd, shell=True)
