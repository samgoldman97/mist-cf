import yaml
import pandas as pd
from pathlib import Path
import subprocess



def conduct_cv(res_folder, summary_names, pred_filter):
    res_folder_cv = res_folder / f"cv_results_{pred_filter}"
    res_folder_cv.mkdir(exist_ok=True)
    for summary_name in summary_names:
        all_files = res_folder.rglob(summary_name)
        dfs = [pd.read_csv(i, sep="\t", index_col=0).reset_index()
               for i in all_files]
        if len(dfs) == 0:
            import pdb
            pdb.set_trace()
        new_df = pd.concat(dfs, axis=0)

        cv_mean = new_df.groupby("index").mean().round(decimals=4)
        cv_std = new_df.groupby("index").std().round(decimals=4)

        out_pth_mean = res_folder_cv / f"CV_mean_{summary_name}"
        out_pth_std = res_folder_cv / f"CV_std_{summary_name}"
        cv_mean.to_csv(out_pth_mean, sep='\t')
        cv_std.to_csv(out_pth_std, sep='\t')

summary_names = ["ion_acc_summary.tsv", "mass_acc_summary.tsv",]
                 #"mass_diff_summary.tsv"]
dataset = "nist_canopus"
res_folder = Path(f"results/fast_filter/")
python_file = "src/mist_cf/fast_form_score/predict.py"
devices = ",".join(['1'])
# pred_filter = "COMMON"
pred_filter = "RDBE"
run_cmds = True
data_dir = Path(f"data/{dataset}")
decoy_label = data_dir / f"decoy_labels/decoy_label_{pred_filter}.tsv"

# Run over all splits
# Note: this model was explicitly trained to exlcude these splits, so we can
# run on all of them
split_stems = ["split_1", "split_2", "split_3"]
for model in res_folder.rglob("version_0/*.ckpt"):

    # Make preds one out
    save_dir_root = model.parent.parent
    save_dir = save_dir_root / f"preds_{pred_filter}"
    save_dir.mkdir(exist_ok=True)

    args_file = save_dir_root / "args.yaml"
    args = yaml.safe_load(open(args_file, "r"))

    for split_stem in split_stems:
        split = data_dir / f"splits/{split_stem}.tsv"
        save_dir_split = save_dir / split_stem
        save_dir_split.mkdir(exist_ok=True)
        pred_label = data_dir / f"pred_labels/pred_{split_stem}_decoy_label_{pred_filter}.tsv"
        cmd = f"""python {python_file} \\
        --batch-size 1024 \\
        --num-workers 16 \\
        --pred-label  {pred_label} \\
        --dataset-name {dataset} \\
        --checkpoint {model} \\
        --save-dir {save_dir_split} \\
        --gpu \\
        """
        device_str = f"CUDA_VISIBLE_DEVICES={devices}"
        cmd = f"{device_str} {cmd}"
        print(cmd + "\n")
        cmd = cmd if run_cmds else ""
        #subprocess.run(cmd, shell=True)

        # Run evaluation
        eval_cmd = f"""
        python analysis/evaluate_pred.py \\
        --true-label {decoy_label} \\
        --res-dir {save_dir_split}  \\
        --split-file {split} \\
        --subset-dataset test_only \\
        """
        print(eval_cmd)
        cmd = cmd if run_cmds else ""
        #subprocess.run(eval_cmd, shell=True)
    conduct_cv(save_dir, summary_names, pred_filter)
