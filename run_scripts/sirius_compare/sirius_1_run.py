import subprocess
from pathlib import Path
import os 

sirius_pth = os.environ["SIRIUS_PATH"]
outfolder_base = Path("results/sirius_gnps_pred/")
input_file_base = Path("data/nist_canopus/")
cores = 32
timeout = 300
form_str = "C[0-]N[0-]O[0-]H[0-]S[0-5]P[0-3]I[0-1]Cl[0-1]F[0-1]Br[0-1]"
adduct_str = "[M+H]+,[M+K]+,[M+Na]+,[M+H-H2O]+,[M+H-H4O2]+,[M+NH4]+,[M]+"

for split in [1,2,3]:
    outfolder_raw = outfolder_base / f"sirius_{split}_raw"
    outfolder = outfolder_base / f"sirius_{split}/"

    # Mkdirs
    outfolder_raw.mkdir(parents=True, exist_ok=True)
    outfolder.mkdir(parents=True, exist_ok=True)
    input_file=input_file_base / f"split_{split}_test.mgf"

    sirius_str = f"""{sirius_pth}  \\
        --cores {cores} \\
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
    print(sirius_str)