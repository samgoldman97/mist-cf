from pathlib import Path
import pandas as pd
from tqdm import tqdm
from mist_cf import common

res_dirs = ["results/sirius_gnps_pred/sirius_1",
            "results/sirius_gnps_pred/sirius_2",
            "results/sirius_gnps_pred/sirius_3"]

for res_dir in res_dirs:
    best_df = Path(res_dir) / "formula_identifications.tsv"
    best_df = pd.read_csv(best_df, sep="\t")
    id_to_mass = dict(best_df[['id', 'ionMass']].values)

    out_df = []

    for cand in tqdm(Path(res_dir).rglob("formula_candidates.tsv")):
        df = pd.read_csv(cand, sep="\t")

        spec_str = cand.parent.name
        spec = spec_str.split("_", 4)[-1]

        # replace space and strip
        df['cand_ion'] = df['adduct'].str.replace(" ", "")

        # Filter df to rows where
        bool_mask = [cand_ion in common.ion_remap 
                     for cand_ion in df['cand_ion']]
        df = df[bool_mask]


        df['scores'] = df['SiriusScore']
        df['cand_form'] = df['molecularFormula']
        df['parentmasses'] = id_to_mass.get(spec_str, -1)
        df['spec'] = spec
        out_df.append(df[["spec", "cand_form", "scores", "cand_ion", "parentmasses"]])
    # Concat df
    out_df = pd.concat(out_df).reset_index(drop=True)
    out_df.to_csv(Path(res_dir) / "formatted_output.tsv", 
                  sep="\t", index=False)
