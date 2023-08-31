""" Make predictions for all models that were trained """
import yaml
import pandas as pd
from pathlib import Path
import subprocess
import re
import yaml


def extract_split_number(file_name):

    pattern = r"split_(\d+)"  # Regular expression pattern to match "split_" followed by one or more digits
    match = re.search(pattern, file_name)  # Search for the pattern in the file name
    if match:
        split_number = match.group(
            1
        )  # Extract the captured group containing the split number
        return int(split_number)  # Convert the split number to an integer
    return None  # Return None if no match is found


res_dirs = {
    "mist_cf": ["results/mist_cf_nist_max_subpeak/"],
}
name_to_pyfile = {
    "ms1": "src/mist_cf/ffn_score/predict.py",
    "ffn": "src/mist_cf/ffn_score/predict.py",
    "mist_cf": "src/mist_cf/mist_cf_score/predict.py",
    "xformer": "src/mist_cf/xformer_score/predict.py",
}

summary_names = [
    "ion_acc_summary.tsv",
    "mass_acc_summary.tsv",
]
dataset = "nist_canopus"
devices = ",".join(["1"])

run_cmds = True
data_dir = Path(f"data/{dataset}")
pred_filter = "COMMON"

for model_name, res_dir_list in res_dirs.items():
    py_file = name_to_pyfile[model_name]
    for res_dir in res_dir_list:
        res_dir = Path(res_dir)
        for model in res_dir.rglob("version_0/*.ckpt"):
            save_dir_root = model.parent.parent

            args_file = save_dir_root / "args.yaml"
            args = yaml.safe_load(open(args_file, "r"))

            split = args["split_file"]
            split_num = str(save_dir_root)[-1]
            split_num = model.parent.parent / "args.yaml"
            split_num = extract_split_number(Path(split).stem)

            pred_label = (
                data_dir
                / f"pred_labels/pred_split_{split_num}_decoy_label_{pred_filter}.tsv"
            )

            # Make preds one out
            save_dir = save_dir_root / f"preds_{pred_filter}"
            save_dir.mkdir(exist_ok=True)

            # split = Path(args['split_file'])
            decoy_label = args["decoy_label"]

            # Get model specific args
            if model_name == "mist_cf":
                subform_dir = args.get("subform_dir")
                aux_args = f"""--subform-dir {subform_dir}"""
            else:
                aux_args = r""

            cmd = f"""python {py_file} \\
            --batch-size 32 \\
            --num-workers 16 \\
            --pred-label  {pred_label} \\
            --dataset-name {dataset} \\
            --checkpoint {model} \\
            --save-dir {save_dir} \\
            --gpu \\
            {aux_args}
            """

            device_str = f"CUDA_VISIBLE_DEVICES={devices}"
            cmd = f"{device_str} {cmd}"
            print(cmd + "\n")
            cmd = cmd if run_cmds else ""
            subprocess.run(cmd, shell=True)

            # Run evaluation
            eval_cmd = f"""
            python analysis/evaluate_pred.py \\
            --true-label {decoy_label} \\
            --res-dir {save_dir}  \\
            --split-file {split} \\
            --subset-dataset test_only \\
            """
            print(eval_cmd)
            cmd = cmd if run_cmds else ""
            subprocess.run(eval_cmd, shell=True)
