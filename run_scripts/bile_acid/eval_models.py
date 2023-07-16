""" Make predictions for all models that were trained """
import pandas as pd
from pathlib import Path
import subprocess
import re

run_cmds = True
dataset = "bile_acid"
data_dir = Path(f"data/{dataset}")
true_label = data_dir / "bile_acid_refined_processed_labels.tsv"

results = [
    dict(save_dir="results/mist_cf_predict_bile_acid"),
    dict(save_dir="results/sirius_predict_bile_acid/sirius_1/"),
]

for res_dict in results:
    save_dir = res_dict['save_dir']

    # Run evaluation
    eval_cmd = f"""
    python analysis/evaluate_pred.py \\
    --true-label {true_label} \\
    --res-dir {save_dir}  \\
    """
    print(eval_cmd)
    cmd = eval_cmd if run_cmds else ""
    subprocess.run(eval_cmd, shell=True)
