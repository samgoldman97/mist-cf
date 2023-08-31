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
    "ms1": ["results/ms1", "results/ms1_nist"],
    "mist_cf": ["results/mist_cf", "results/mist_cf_nist"],
    "ffn": ["results/ffn", "results/ffn_nist"],
    "xformer": ["results/xformer", "results/xformer_nist"],
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
# "mass_diff_summary.tsv"]
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
                # TODO: Consider replacing with RDBE
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

        res_dir_cv = res_dir / f"cv_results_{pred_filter}"
        res_dir_cv.mkdir(exist_ok=True)
        for summary_name in summary_names:
            all_files = res_dir.rglob(summary_name)
            dfs = [
                pd.read_csv(i, sep="\t", index_col=0).reset_index() for i in all_files
            ]
            new_df = pd.concat(dfs, axis=0)

            cv_mean = new_df.groupby("index").mean().round(decimals=4)
            cv_std = new_df.groupby("index").std().round(decimals=4)

            out_pth_mean = res_dir_cv / f"CV_mean_{summary_name}"
            out_pth_std = res_dir_cv / f"CV_std_{summary_name}"
            cv_mean.to_csv(out_pth_mean, sep="\t")
            cv_std.to_csv(out_pth_std, sep="\t")
