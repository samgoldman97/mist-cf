import json
import pandas as pd
import time
import subprocess
from pathlib import Path
import os

input_file = f"data/demo_specs.mgf"
base_output = Path("results/timing_experiments/")
out_res_mist = base_output / "timing_res_mist.tsv"
out_res_sirius = base_output / "timing_res_sirius.tsv"
base_output.mkdir(exist_ok=True, parents=True)
sirius_cores = 1
mist_cores = 1
replicates = [1, 2, 3]


def run_sirius(base_folder, replicate_num=1):
    sirius_pth = os.environ["SIRIUS_PATH"]
    timeout = 60
    form_str = "C[0-]N[0-]O[0-]H[0-]S[0-5]P[0-3]I[0-1]Cl[0-1]F[0-1]Br[0-1]"
    adduct_str = "[M+H]+,[M+K]+,[M+Na]+,[M+H-H2O]+,[M+H-H4O2]+,[M+NH4]+,[M]+"

    outfolder_raw = base_folder / f"sirius_{replicate_num}_raw"
    outfolder = base_folder / f"sirius_{replicate_num}/"

    # Mkdirs
    outfolder_raw.mkdir(parents=True, exist_ok=True)
    outfolder.mkdir(parents=True, exist_ok=True)

    sirius_str = f"""{sirius_pth}  \\
        --cores {sirius_cores} \\
        --output  {outfolder_raw} \\
        --input {input_file} \\
        formula  \\
        -i {adduct_str} \\
        -e {form_str } \\
        --tree-timeout {timeout} \\
        --compound-timeout {timeout} \\
        write-summaries \\
        --output {outfolder}
    """
    subprocess.run(sirius_str, shell=True)


def run_mist(base_folder, replicate_num=1):
    outfolder = base_folder / f"mist_{replicate_num}/"
    outfolder.mkdir(parents=True, exist_ok=True)
    mist_model = "results/mist_cf_nist/split_1_with_nist/version_0/best.ckpt"
    fast_model = "results/fast_filter/split/version_0/best.ckpt"
    mist_str = f"""python src/mist_cf/mist_cf_score/predict_mgf.py \\
        --id-key FEATURE_ID \\
        --num-workers {mist_cores} \\
        --batch-size 32 \\
        --save-dir {outfolder} \\
        --mgf-file {input_file} \\
        --checkpoint-pth {mist_model} \\
        --fast-model  {fast_model} \\
        --fast-num 256 \\
        --decomp-ppm 10 \\
        --decomp-filter RDBE"""
    subprocess.run(mist_str, shell=True)


def time_fn(fn, kwargs):
    start = time.time()
    fn(**kwargs)
    end = time.time()
    return end - start


out_dicts_mist = []
out_dicts_sirius = []
for replicate in replicates:
    out_time_sirius = time_fn(
        run_sirius, dict(base_folder=base_output, replicate_num=replicate)
    )
    out_dicts_sirius.append(
        {
            "Method": "SIRIUS",
            "Time": out_time_sirius,
            "Specs": 10,
            "Replicate": replicate,
        }
    )
    print(json.dumps(out_dicts_sirius[-1], indent=2))

if len(out_dicts_sirius) > 0:
    df = pd.DataFrame(out_dicts_sirius)
    df.to_csv(out_res_sirius, index=False, sep="\t")

if len(out_dicts_mist) > 0:
    df = pd.DataFrame(out_dicts_mist)
    df.to_csv(out_res_mist, index=False, sep="\t")
    print(df)
